"""
FusionAceStepLayerwise
======================
Wan video DiT(30 层)  <->  ACE-Step base audio DiT(24 层) 的「真·逐层双向融合」。

与旧版(fusion_acestep.py)的根本区别
-----------------------------------
旧版是「塔级 cross-attn」:ACE 整塔跑完 24 层 -> 把整段 audio hidden 广播给 video 每一层;
video 的信息要等到 *下一个 denoise step* 才喂回 ACE。 => 不对称 + 跨 step 延迟 = 弱耦合。

本版与原始 Ovi fusion.py 一致的思想:**在每一个 fusion rendezvous,两塔各自取对方
当前 hidden 的快照,同 step 双向 cross-attn,然后各跑自己的原生层。** 每层只跑一次,
深度不齐(30 vs 24)用「调度表」吸收。这样音画在每一层都互相看见,过拟合时对齐才会涌现。

设计要点
--------
1. 不改两塔任何原生 forward:video 调 WanAttentionBlock.forward,audio 调 AceStepDiTLayer
   (经 AceDiTRunner)。融合是**额外的残差子层**,o_proj 近零初始化 => 训练起点 ≈ 两塔
   各自的预训练行为,稳定且可控,正是「完全一致的过拟合」需要的起点。
2. ACE 自带的 text/lyric/timbre cross-attn 原样保留(满足「text/lyric 复用」)。
3. 时间步对齐口径(训推一致的关键,见 README):两塔共用同一个 flow-matching 调度与同一个
   timestep;ACE 侧 timestep_r = timestep(meanflow 的 (t-r) 项=0,退化为标准 rectified flow)。
   训练与推理都走同一套单一 timestep,保证 train==infer。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ovi.modules.model import WanModel
from ovi.utils.acestep_loader import init_acestep_handler, AceDiTRunner


# --------------------------------------------------------------------------- #
#  融合用的多头 cross-attn(Q 来自本塔,KV 来自对方塔)
#  —— 用 SDPA,CPU/CUDA 都能跑,不依赖 flash;o_proj 零初始化。
# --------------------------------------------------------------------------- #
class FusionCrossAttn(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, num_heads: int):
        super().__init__()
        assert q_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.norm_q = nn.LayerNorm(q_dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(kv_dim, eps=1e-6)
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, q_dim, bias=False)     # kv_dim -> q_dim,顺带换维
        self.v_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, q_dim, bias=False)
        # gate:逐通道可学习缩放,初值 0 => 融合在训练起点完全关闭
        self.gate = nn.Parameter(torch.zeros(q_dim))
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, x, kv):
        """x:[B, Lx, q_dim](本塔 hidden);kv:[B, Lk, kv_dim](对方塔 hidden 快照)。返回残差增量。"""
        B, Lx, _ = x.shape
        Lk = kv.shape[1]
        q = self.q_proj(self.norm_q(x)).view(B, Lx, self.num_heads, self.head_dim).transpose(1, 2)
        kvn = self.norm_kv(kv)
        k = self.k_proj(kvn).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kvn).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        # 跨模态、时间轴不同,不加 RoPE;对齐由注意力权重学。
        o = F.scaled_dot_product_attention(q, k, v)            # [B, H, Lx, hd]
        o = o.transpose(1, 2).reshape(B, Lx, -1)
        return self.gate * self.o_proj(o)


def _build_schedule(n_video: int, n_audio: int):
    """
    把 n_video 个 video 层分配到 n_audio 个 rendezvous。
    返回长度 n_audio 的列表,元素是该 rendezvous 要跑的 video 层数(累计=n_video)。
    例:30 video / 24 audio -> 6 个 rendezvous 跑 2 层、18 个跑 1 层。
    """
    sched, prev = [], 0
    for j in range(1, n_audio + 1):
        cur = round(j * n_video / n_audio)
        sched.append(cur - prev)
        prev = cur
    assert sum(sched) == n_video, (sched, n_video)
    return sched


class FusionAceStepLayerwise(nn.Module):
    def __init__(self,
                 video_config: dict,
                 acestep_project_root: str,
                 ace_config_path: str = "acestep-v15-base",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()

        # ---- 视频塔(Wan,原版) ----
        self.video_model = WanModel(**video_config)
        self.video_dim = video_config["dim"]
        self.video_heads = video_config["num_heads"]
        self.num_video_layers = video_config["num_layers"]

        # ---- 音频塔(ACE-Step base,官方 handler) ----
        self._ace_handler = init_acestep_handler(
            acestep_project_root, config_path=ace_config_path, device=device, dtype=dtype)
        self.ace = AceDiTRunner(self._ace_handler.model.decoder)
        self.audio_dim = self._ace_handler.config.hidden_size            # base: 2048
        self.audio_heads = self._ace_handler.config.num_attention_heads  # base: 16
        self.num_audio_layers = self.ace.num_layers                      # base: 24

        # ---- 融合子层(唯一可训练的「新」参数) ----
        # video 每层一个:Q=video, KV=audio
        self.v_fuse = nn.ModuleList([
            FusionCrossAttn(self.video_dim, self.audio_dim, self.video_heads)
            for _ in range(self.num_video_layers)])
        # audio 每层一个:Q=audio, KV=video
        self.a_fuse = nn.ModuleList([
            FusionCrossAttn(self.audio_dim, self.video_dim, self.audio_heads)
            for _ in range(self.num_audio_layers)])

        # 深度不齐调度表
        self.schedule = _build_schedule(self.num_video_layers, self.num_audio_layers)

        self._freeze_base()
        self.gradient_checkpointing = False

    # ----- 便捷属性 -----
    @property
    def acestep_handler(self):
        return self._ace_handler

    @property
    def acestep_vae(self):
        return self._ace_handler.vae

    # ----- 冻结策略:两塔 base 全冻,只训融合子层 -----
    def _freeze_base(self):
        for p in self.video_model.parameters():
            p.requires_grad = False
        for p in self.ace.dit.parameters():
            p.requires_grad = False
        for p in self._ace_handler.model.encoder.parameters():
            p.requires_grad = False
        for p in self._ace_handler.vae.parameters():
            p.requires_grad = False
        for p in self.v_fuse.parameters():
            p.requires_grad = True
        for p in self.a_fuse.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable
        self.video_model.set_gradient_checkpointing(enable)
        self.ace.dit.gradient_checkpointing = enable

    def set_rope_params(self):
        self.video_model.set_rope_params()

    # ----- video 原生 block(不带融合) -----
    def _video_native_block(self, i, x, e, kw):
        blk = self.video_model.blocks[i]
        return blk(x, e, kw["seq_lens"], kw["grid_sizes"], kw["freqs"],
                   kw["context"], kw["context_lens"])

    # ----------------------------------------------------------------- #
    #  forward
    # ----------------------------------------------------------------- #
    def forward(self,
                vid,                         # list[ [C,F,H,W] ]  video latent(已加噪)
                audio_latent,                # [B, T, 64]         ACE audio latent(已加噪)
                t,                           # [B]                共用 timestep
                vid_context,                 # T5 text emb (list)
                ace_encoder_hidden_states,   # [B, L_enc, 2048]   ACE encoder 输出(text+lyric+timbre)
                ace_encoder_attention_mask,  # [B, L_enc]
                ace_context_latents,         # [B, T, 128]
                vid_seq_len,
                first_frame_is_clean=False,
                slg_layer=False):
        """返回 (video_pred [C,F,H,W], audio_pred [B,T,64])。"""

        # ---- video 进 block 循环前的准备 ----
        vid_x, vid_e, vid_kw = self.video_model.prepare_transformer_block_kwargs(
            x=vid, t=t, context=vid_context, seq_len=vid_seq_len,
            clip_fea=None, y=None, first_frame_is_clean=first_frame_is_clean)

        # ---- audio 进 layer 循环前的准备(timestep_r = t => 退化为标准 flow) ----
        a_hidden, a_state = self.ace.prepare(
            hidden_states=audio_latent, timestep=t, timestep_r=t,
            encoder_hidden_states=ace_encoder_hidden_states,
            encoder_attention_mask=ace_encoder_attention_mask,
            context_latents=ace_context_latents, attention_mask=None)

        # ---- 交错逐层:每个 rendezvous 同 step 双向融合,再各跑原生层 ----
        vi = 0
        for j in range(self.num_audio_layers):
            # 1) 取双方快照(对称,顺序无关)
            v_snap = vid_x
            a_snap = a_hidden
            # 2) 双向融合残差
            #    audio <- video
            a_hidden = a_hidden + self.a_fuse[j](a_snap, v_snap)
            # 3) 跑本 rendezvous 的 video 层(进入前先注入 audio 快照)
            for _ in range(self.schedule[j]):
                if slg_layer and vi == slg_layer:
                    vi += 1
                    continue
                vid_x = vid_x + self.v_fuse[vi](vid_x, a_snap)
                vid_x = self._video_native_block(vi, vid_x, vid_e, vid_kw)
                vi += 1
            # 4) 跑 audio 第 j 层(原生:含 text/lyric cross-attn)
            a_hidden = self.ace.run_layer(j, a_hidden, a_state)

        # 收尾(理论上 schedule 已覆盖全部 video 层;保险处理)
        while vi < self.num_video_layers:
            vid_x = vid_x + self.v_fuse[vi](vid_x, a_hidden)
            vid_x = self._video_native_block(vi, vid_x, vid_e, vid_kw)
            vi += 1

        video_pred = self.video_model.post_transformer_block_out(
            vid_x, vid_kw["grid_sizes"], vid_e)
        audio_pred = self.ace.finalize(a_hidden, a_state)
        return video_pred, audio_pred
