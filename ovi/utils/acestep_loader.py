"""
ACE-Step (v15 *base*) 加载与「逐层驱动」辅助工具。

设计目标
--------
1. 复用官方 AceStepHandler 把整套音频栈(DiT decoder / VAE / text+lyric+timbre encoder /
   text_tokenizer)加载进来,**不重写 ACE 的 conditioning 管线**(满足「text/lyric 复用」)。
2. 暴露把 ACE DiT *拆成逐层* 运行所需要的全部「前处理 / 后处理」函数,
   这样 fusion 模型才能在 ACE 的每一层之间插入 video<->audio 的 cross-attn,
   而不是像旧版那样「整塔跑完 24 层」。

注意
----
* `config_path` 用 base 权重目录(用户要求用 base,最全的一档)。
* 这里所有函数都假定 batch_size == 1(与现有 train_t2av / 推理引擎一致)。
  扩 batch 时 mask / pad 逻辑需要再核对。

★★★ 关键修复(音频真值处理) ★★★
--------------------------------
ACE 的 VAE 是 diffusers `AutoencoderOobleck`,**只接受 立体声(2 通道)、48kHz、[B,2,S]、
幅度∈[-1,1]** 的波形(见 ACE handler 的 `_normalize_audio_to_stereo_48k`)。
但上游 T2MD 数据管线(diffsynth/trainers/t2mv_dataset.py: load_audio)会把音频
**下混成单声道并重采样到 16kHz**,返回 [1, L]。旧版 `encode_audio_to_latent` 把这个
mono/16k 张量**直接**喂给 ACE VAE:
    - 采样率不符(16k 当 48k 用)→ 时间轴/音高被 3× 扭曲;
    - 通道数不符(mono 而非 stereo)。
于是 VAE 编码出的 latent 解码回来是「被扭曲的音频」。这个错误是**确定且自洽**的:
    原始音乐 → (mono/16k) VAE.encode → x0 → VAE.decode → 扭曲音频
    模型对 x0 过拟合 → 拟合输出解码 ≈ 同一段扭曲音频
=> 「真值解码 ≈ 拟合解码」(过拟合 sanity 通过),但两者都 ≠ 真实音乐。
这正是「模型在学习一个有问题的真值」的现象。

修复:在喂给 ACE VAE 之前,**统一整理成 stereo / 48kHz / [B,2,S] / clamp** 的输入
(`prepare_audio_for_ace_vae`)。同时过拟合 sanity 用 `posterior.mode()`(确定性均值)
而非 `.sample()`,避免每次编码往真值里注入新的 VAE 采样噪声,保证真值可复现。
"""
import sys
import torch
import torch.nn as nn


# ACE AutoencoderOobleck 的目标输入规格
ACE_VAE_SAMPLE_RATE = 48000
ACE_VAE_CHANNELS = 2


# --------------------------------------------------------------------------- #
#  handler 加载
# --------------------------------------------------------------------------- #
def init_acestep_handler(acestep_project_root: str,
                         config_path: str = "acestep-v15-base",
                         device: str = "cuda",
                         dtype: torch.dtype = torch.bfloat16,
                         offload_to_cpu: bool = False):
    """加载完整 ACE-Step base 栈。返回 AceStepHandler。"""
    if acestep_project_root not in sys.path:
        sys.path.insert(0, acestep_project_root)

    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    status_msg, ok = handler.initialize_service(
        project_root=acestep_project_root,
        config_path=config_path,
        device=device,
        offload_to_cpu=offload_to_cpu,
        # 关键:让 ACE DiT 走 sdpa/eager,产生 4D mask,且不与 Ovi 的 flash-attn 互相踩。
        use_flash_attention=False,
    )
    if not ok:
        raise RuntimeError(f"ACE-Step handler init failed: {status_msg}")
    handler.dtype = dtype
    return handler


# --------------------------------------------------------------------------- #
#  ★ 音频输入规范化(核心修复)
# --------------------------------------------------------------------------- #
def prepare_audio_for_ace_vae(audio: torch.Tensor,
                              src_sample_rate: int,
                              device,
                              dtype: torch.dtype) -> torch.Tensor:
    """把任意来源的波形整理成 ACE AutoencoderOobleck 期望的输入。

    输出: [B, 2, S] 立体声、48kHz、幅度 clamp 到 [-1, 1]。

    支持的输入形状:
        [S]           单条 mono
        [C, S]        单条(C 通道)
        [B, S]        (dataset 里 `audio.unsqueeze(0)` 得到的 [1, L] 属于这种)
        [B, C, S]     标准三维

    注意 [B, S] 与 [C, S] 都是二维,靠启发式区分:第 0 维 <= 2 且第 1 维 > 2 时按 [C, S]
    处理,否则按 [B, S](batch)处理。当前训练/推理均为 batch=1,两种解释结果一致。
    """
    x = audio
    if x.dim() == 1:                                   # [S] -> [1, 1, S]
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 2:
        if x.shape[0] <= ACE_VAE_CHANNELS and x.shape[1] > ACE_VAE_CHANNELS:
            x = x.unsqueeze(0)                         # [C, S] -> [1, C, S]
        else:
            x = x.unsqueeze(1)                         # [B, S] -> [B, 1, S]
    elif x.dim() == 3:                                 # [B, C, S]
        pass
    else:
        raise ValueError(f"prepare_audio_for_ace_vae: 不支持的音频形状 {tuple(audio.shape)}")

    x = x.to(device=device, dtype=torch.float32)

    # 重采样到 48kHz(★ 修复采样率不符导致的时间轴/音高扭曲)
    if src_sample_rate is not None and int(src_sample_rate) != ACE_VAE_SAMPLE_RATE:
        import torchaudio
        x = torchaudio.functional.resample(x, int(src_sample_rate), ACE_VAE_SAMPLE_RATE)

    # mono -> stereo / 裁到 2 通道(★ 修复通道数不符)
    C = x.shape[1]
    if C == 1:
        x = x.repeat(1, ACE_VAE_CHANNELS, 1)
    elif C > ACE_VAE_CHANNELS:
        x = x[:, :ACE_VAE_CHANNELS, :]

    x = torch.clamp(x, -1.0, 1.0)
    return x.to(dtype)


# --------------------------------------------------------------------------- #
#  VAE 编解码
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_audio_to_latent(handler,
                           audio: torch.Tensor,
                           src_sample_rate: int = ACE_VAE_SAMPLE_RATE,
                           deterministic: bool = True) -> torch.Tensor:
    """wav -> ACE latent。

    Args:
        audio: 任意形状的波形([S] / [C,S] / [B,S] / [B,C,S])。
        src_sample_rate: 传入 audio 的真实采样率。**必须如实填写**,否则会被当成 48k
            处理而扭曲。上游 T2MD 数据集默认 16000;若已把数据集改成 48k 立体声则填 48000。
        deterministic: True 用后验均值 mode()(过拟合 sanity 的正确做法,真值可复现);
            False 用 sample()(与 ACE 生成语义一致,真实大规模训练可用)。

    Returns:
        [B, T, 64] 的 ACE audio latent。
    """
    vae = handler.vae
    device = next(vae.parameters()).device
    # ★ 关键修复:规范化到 stereo / 48k / [B,2,S] / clamp
    audio = prepare_audio_for_ace_vae(audio, src_sample_rate, device, vae.dtype)   # [B, 2, S] @48k
    posterior = vae.encode(audio).latent_dist
    latent = posterior.mode() if deterministic else posterior.sample()             # [B, 64, T]
    return latent.transpose(1, 2).to(handler.dtype)                                # [B, T, 64]


@torch.no_grad()
def decode_latent_to_audio(handler, latent: torch.Tensor) -> torch.Tensor:
    """ACE latent -> wav。latent: [B, T, 64]  ->  [B, 2, S] @48k(stereo)"""
    vae = handler.vae
    device = next(vae.parameters()).device
    x = latent.transpose(1, 2).to(device=device, dtype=vae.dtype)   # [B, 64, T]
    return vae.decode(x).sample


def build_context_latents_default(handler, latent_length: int, device, dtype):
    """无 source-audio(纯生成)时的 context_latents:静音 latent + chunk mask -> [1, T, 128]"""
    silence = handler.silence_latent[:, :latent_length, :].to(device=device, dtype=dtype)
    if silence.shape[1] < latent_length:
        pad = latent_length - silence.shape[1]
        silence = torch.cat([silence, silence[:, :pad, :].expand(1, -1, -1)], dim=1)
    chunk_masks = torch.ones(1, latent_length, 64, device=device, dtype=dtype)
    return torch.cat([silence, chunk_masks], dim=-1)               # [1, T, 128]


@torch.no_grad()
def build_ace_encoder_hidden(handler,
                             text_caption: str,
                             lyrics: str = "[Instrumental]",
                             device: str = "cuda"):
    """
    用 ACE 自带的 text/lyric/timbre encoder 生成 cross-attn 用的 encoder_hidden_states。
    => 满足「text/lyric 复用」:这里直接调官方 encoder,不另起炉灶。

    返回: (encoder_hidden [B, L_enc, 2048], encoder_mask [B, L_enc])
    """
    tok = handler.text_tokenizer
    text_enc = handler.text_encoder

    inputs = tok(text_caption, return_tensors="pt", padding=True, truncation=True).to(device)
    out = text_enc(**inputs, output_hidden_states=False)
    text_hidden = out.last_hidden_state.to(handler.dtype)          # [1, Lt, text_hidden_dim]
    text_mask = inputs["attention_mask"].to(handler.dtype)

    # --- lyric ---
    # 纯器乐时 ACE 用 [Instrumental];若有歌词,应走 handler 的 lyric tokenizer。
    # 这里给出占位实现,真实歌词训练时替换 lyric_hidden/lyric_mask 即可。
    lyric_hidden = torch.zeros(1, 1, handler.config.text_hidden_dim,
                               device=device, dtype=handler.dtype)
    lyric_mask = torch.zeros(1, 1, device=device, dtype=handler.dtype)

    # --- refer audio(timbre):无参考音色时给 zeros ---
    refer_hidden = torch.zeros(1, 1, handler.config.audio_acoustic_hidden_dim,
                               device=device, dtype=handler.dtype)
    refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)

    encoder_hidden, encoder_mask = handler.model.encoder(
        text_hidden_states=text_hidden,
        text_attention_mask=text_mask,
        lyric_hidden_states=lyric_hidden,
        lyric_attention_mask=lyric_mask,
        refer_audio_acoustic_hidden_states_packed=refer_hidden,
        refer_audio_order_mask=refer_order_mask,
    )
    return encoder_hidden, encoder_mask


# --------------------------------------------------------------------------- #
#  ACE DiT「逐层驱动」前/后处理
#  —— 把 AceStepDiTModel.forward 的 pre-loop / post-loop 部分复刻出来,
#     中间的 `for layer in self.layers` 交给 fusion 模型逐层调度。
# --------------------------------------------------------------------------- #
class AceDiTRunner:
    """
    包住 ACE 的 AceStepDiTModel,提供:
      prepare(...) -> 跑到「进入 layer 循环之前」的全部状态
      run_layer(j, hidden, state) -> 跑第 j 层(原生 AceStepDiTLayer.forward)
      finalize(hidden, state)     -> norm_out + proj_out + 裁剪,得到 flow 预测
    这样 fusion 就能在 run_layer 之间插 cross-attn。
    """
    def __init__(self, ace_dit):
        self.dit = ace_dit
        self.config = ace_dit.config
        self.num_layers = ace_dit.config.num_hidden_layers
        # 引入官方 mask 构造器,保证与原 forward 完全一致的注意力语义
        mod = sys.modules[type(ace_dit).__module__]
        self._create_4d_mask = getattr(mod, "create_4d_mask")

    @torch.no_grad()
    def _build_masks(self, hidden, encoder_hidden, attention_mask):
        dit, cfg = self.dit, self.config
        seq_len = hidden.shape[1]
        enc_len = encoder_hidden.shape[1]
        dtype, device = hidden.dtype, hidden.device
        is_flash = (cfg._attn_implementation == "flash_attention_2")
        if is_flash:
            full = attention_mask
            sliding = attention_mask if cfg.use_sliding_window else None
            enc = None
        else:
            full = self._create_4d_mask(seq_len=seq_len, dtype=dtype, device=device,
                                        attention_mask=attention_mask, sliding_window=None,
                                        is_sliding_window=False, is_causal=False)
            max_len = max(seq_len, enc_len)
            enc = self._create_4d_mask(seq_len=max_len, dtype=dtype, device=device,
                                       attention_mask=attention_mask, sliding_window=None,
                                       is_sliding_window=False, is_causal=False)
            enc = enc[:, :, :seq_len, :enc_len]
            sliding = None
            if cfg.use_sliding_window:
                sliding = self._create_4d_mask(seq_len=seq_len, dtype=dtype, device=device,
                                               attention_mask=attention_mask,
                                               sliding_window=cfg.sliding_window,
                                               is_sliding_window=True, is_causal=False)
        return {"full_attention": full, "sliding_attention": sliding,
                "encoder_attention_mask": enc}

    def prepare(self, hidden_states, timestep, timestep_r,
                encoder_hidden_states, encoder_attention_mask,
                context_latents, attention_mask=None):
        """复刻 AceStepDiTModel.forward 的 pre-loop。返回 (hidden, state-dict)。"""
        dit = self.dit
        # --- 时间步 (与原版一致;timestep_r 由调用方决定,见 fusion 的对齐口径) ---
        temb_t, tproj_t = dit.time_embed(timestep)
        temb_r, tproj_r = dit.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = tproj_t + tproj_r

        # --- 拼 context_latents + patchify ---
        hidden = torch.cat([context_latents, hidden_states], dim=-1)
        original_seq_len = hidden.shape[1]
        if hidden.shape[1] % dit.patch_size != 0:
            pad = dit.patch_size - (hidden.shape[1] % dit.patch_size)
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad), value=0)
        hidden = dit.proj_in(hidden)
        enc = dit.condition_embedder(encoder_hidden_states)

        # --- position / rotary ---
        cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = dit.rotary_emb(hidden, position_ids)

        mask_mapping = self._build_masks(hidden, enc, attention_mask)
        state = dict(timestep_proj=timestep_proj, temb=temb,
                     position_embeddings=position_embeddings, position_ids=position_ids,
                     cache_position=cache_position, mask_mapping=mask_mapping,
                     enc=enc, enc_mask=mask_mapping["encoder_attention_mask"],
                     original_seq_len=original_seq_len)
        return hidden, state

    def run_layer(self, j, hidden, state):
        """原生 AceStepDiTLayer.forward(self-attn + (text/lyric)cross-attn + mlp)。"""
        layer = self.dit.layers[j]
        out = layer(
            hidden,
            state["position_embeddings"],
            state["timestep_proj"],
            state["mask_mapping"][layer.attention_type],
            state["position_ids"],
            None,                # past_key_value
            False,               # output_attentions
            False,               # use_cache
            state["cache_position"],
            state["enc"],
            state["enc_mask"],
        )
        return out[0]

    def finalize(self, hidden, state):
        """norm_out + 去 patch + 裁剪 -> [B, T, 64] 的 flow 预测。"""
        dit = self.dit
        shift, scale = (dit.scale_shift_table + state["temb"].unsqueeze(1)).chunk(2, dim=1)
        hidden = (dit.norm_out(hidden) * (1 + scale) + shift).type_as(hidden)
        hidden = dit.proj_out(hidden)
        return hidden[:, :state["original_seq_len"], :]


class CrossAttnAdapter(nn.Module):
    """维度适配(保留以兼容旧脚本)。fusion 内部已用 k/v proj 直接换维,通常不需要它。"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x):
        return self.proj(self.norm(x))# --------------------------------------------------------------------------- #
#  ★ 音频输入规范化(核心修复)
# --------------------------------------------------------------------------- #
def prepare_audio_for_ace_vae(audio: torch.Tensor,
                              src_sample_rate: int,
                              device,
                              dtype: torch.dtype) -> torch.Tensor:
    """把任意来源的波形整理成 ACE AutoencoderOobleck 期望的输入。

    输出: [B, 2, S] 立体声、48kHz、幅度 clamp 到 [-1, 1]。

    支持的输入形状:
        [S]           单条 mono
        [C, S]        单条(C 通道)
        [B, S]        (dataset 里 `audio.unsqueeze(0)` 得到的 [1, L] 属于这种)
        [B, C, S]     标准三维

    注意 [B, S] 与 [C, S] 都是二维,靠启发式区分:第 0 维 <= 2 且第 1 维 > 2 时按 [C, S]
    处理,否则按 [B, S](batch)处理。当前训练/推理均为 batch=1,两种解释结果一致。
    """
    x = audio
    if x.dim() == 1:                                   # [S] -> [1, 1, S]
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 2:
        if x.shape[0] <= ACE_VAE_CHANNELS and x.shape[1] > ACE_VAE_CHANNELS:
            x = x.unsqueeze(0)                         # [C, S] -> [1, C, S]
        else:
            x = x.unsqueeze(1)                         # [B, S] -> [B, 1, S]
    elif x.dim() == 3:                                 # [B, C, S]
        pass
    else:
        raise ValueError(f"prepare_audio_for_ace_vae: 不支持的音频形状 {tuple(audio.shape)}")

    x = x.to(device=device, dtype=torch.float32)

    # 重采样到 48kHz(★ 修复采样率不符导致的时间轴/音高扭曲)
    if src_sample_rate is not None and int(src_sample_rate) != ACE_VAE_SAMPLE_RATE:
        import torchaudio
        x = torchaudio.functional.resample(x, int(src_sample_rate), ACE_VAE_SAMPLE_RATE)

    # mono -> stereo / 裁到 2 通道(★ 修复通道数不符)
    C = x.shape[1]
    if C == 1:
        x = x.repeat(1, ACE_VAE_CHANNELS, 1)
    elif C > ACE_VAE_CHANNELS:
        x = x[:, :ACE_VAE_CHANNELS, :]

    x = torch.clamp(x, -1.0, 1.0)
    return x.to(dtype)


# --------------------------------------------------------------------------- #
#  VAE 编解码
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_audio_to_latent(handler,
                           audio: torch.Tensor,
                           src_sample_rate: int = ACE_VAE_SAMPLE_RATE,
                           deterministic: bool = True) -> torch.Tensor:
    """wav -> ACE latent。

    Args:
        audio: 任意形状的波形([S] / [C,S] / [B,S] / [B,C,S])。
        src_sample_rate: 传入 audio 的真实采样率。**必须如实填写**,否则会被当成 48k
            处理而扭曲。上游 T2MD 数据集默认 16000;若已把数据集改成 48k 立体声则填 48000。
        deterministic: True 用后验均值 mode()(过拟合 sanity 的正确做法,真值可复现);
            False 用 sample()(与 ACE 生成语义一致,真实大规模训练可用)。

    Returns:
        [B, T, 64] 的 ACE audio latent。
    """
    vae = handler.vae
    device = next(vae.parameters()).device
    # ★ 关键修复:规范化到 stereo / 48k / [B,2,S] / clamp
    audio = prepare_audio_for_ace_vae(audio, src_sample_rate, device, vae.dtype)   # [B, 2, S] @48k
    posterior = vae.encode(audio).latent_dist
    latent = posterior.mode() if deterministic else posterior.sample()             # [B, 64, T]
    return latent.transpose(1, 2).to(handler.dtype)                                # [B, T, 64]


@torch.no_grad()
def decode_latent_to_audio(handler, latent: torch.Tensor) -> torch.Tensor:
    """ACE latent -> wav。latent: [B, T, 64]  ->  [B, 2, S] @48k(stereo)"""
    vae = handler.vae
    device = next(vae.parameters()).device
    x = latent.transpose(1, 2).to(device=device, dtype=vae.dtype)   # [B, 64, T]
    return vae.decode(x).sample


def build_context_latents_default(handler, latent_length: int, device, dtype):
    """无 source-audio(纯生成)时的 context_latents:静音 latent + chunk mask -> [1, T, 128]"""
    silence = handler.silence_latent[:, :latent_length, :].to(device=device, dtype=dtype)
    if silence.shape[1] < latent_length:
        pad = latent_length - silence.shape[1]
        silence = torch.cat([silence, silence[:, :pad, :].expand(1, -1, -1)], dim=1)
    chunk_masks = torch.ones(1, latent_length, 64, device=device, dtype=dtype)
    return torch.cat([silence, chunk_masks], dim=-1)               # [1, T, 128]


@torch.no_grad()
def build_ace_encoder_hidden(handler,
                             text_caption: str,
                             lyrics: str = "[Instrumental]",
                             device: str = "cuda"):
    """
    用 ACE 自带的 text/lyric/timbre encoder 生成 cross-attn 用的 encoder_hidden_states。
    => 满足「text/lyric 复用」:这里直接调官方 encoder,不另起炉灶。

    返回: (encoder_hidden [B, L_enc, 2048], encoder_mask [B, L_enc])
    """
    tok = handler.text_tokenizer
    text_enc = handler.text_encoder

    inputs = tok(text_caption, return_tensors="pt", padding=True, truncation=True).to(device)
    out = text_enc(**inputs, output_hidden_states=False)
    text_hidden = out.last_hidden_state.to(handler.dtype)          # [1, Lt, text_hidden_dim]
    text_mask = inputs["attention_mask"].to(handler.dtype)

    # --- lyric ---
    # 纯器乐时 ACE 用 [Instrumental];若有歌词,应走 handler 的 lyric tokenizer。
    # 这里给出占位实现,真实歌词训练时替换 lyric_hidden/lyric_mask 即可。
    lyric_hidden = torch.zeros(1, 1, handler.config.text_hidden_dim,
                               device=device, dtype=handler.dtype)
    lyric_mask = torch.zeros(1, 1, device=device, dtype=handler.dtype)

    # --- refer audio(timbre):无参考音色时给 zeros ---
    refer_hidden = torch.zeros(1, 1, handler.config.audio_acoustic_hidden_dim,
                               device=device, dtype=handler.dtype)
    refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)

    encoder_hidden, encoder_mask = handler.model.encoder(
        text_hidden_states=text_hidden,
        text_attention_mask=text_mask,
        lyric_hidden_states=lyric_hidden,
        lyric_attention_mask=lyric_mask,
        refer_audio_acoustic_hidden_states_packed=refer_hidden,
        refer_audio_order_mask=refer_order_mask,
    )
    return encoder_hidden, encoder_mask


# --------------------------------------------------------------------------- #
#  ACE DiT「逐层驱动」前/后处理
#  —— 把 AceStepDiTModel.forward 的 pre-loop / post-loop 部分复刻出来,
#     中间的 `for layer in self.layers` 交给 fusion 模型逐层调度。
# --------------------------------------------------------------------------- #
class AceDiTRunner:
    """
    包住 ACE 的 AceStepDiTModel,提供:
      prepare(...) -> 跑到「进入 layer 循环之前」的全部状态
      run_layer(j, hidden, state) -> 跑第 j 层(原生 AceStepDiTLayer.forward)
      finalize(hidden, state)     -> norm_out + proj_out + 裁剪,得到 flow 预测
    这样 fusion 就能在 run_layer 之间插 cross-attn。
    """
    def __init__(self, ace_dit):
        self.dit = ace_dit
        self.config = ace_dit.config
        self.num_layers = ace_dit.config.num_hidden_layers
        # 引入官方 mask 构造器,保证与原 forward 完全一致的注意力语义
        mod = sys.modules[type(ace_dit).__module__]
        self._create_4d_mask = getattr(mod, "create_4d_mask")

    @torch.no_grad()
    def _build_masks(self, hidden, encoder_hidden, attention_mask):
        dit, cfg = self.dit, self.config
        seq_len = hidden.shape[1]
        enc_len = encoder_hidden.shape[1]
        dtype, device = hidden.dtype, hidden.device
        is_flash = (cfg._attn_implementation == "flash_attention_2")
        if is_flash:
            full = attention_mask
            sliding = attention_mask if cfg.use_sliding_window else None
            enc = None
        else:
            full = self._create_4d_mask(seq_len=seq_len, dtype=dtype, device=device,
                                        attention_mask=attention_mask, sliding_window=None,
                                        is_sliding_window=False, is_causal=False)
            max_len = max(seq_len, enc_len)
            enc = self._create_4d_mask(seq_len=max_len, dtype=dtype, device=device,
                                       attention_mask=attention_mask, sliding_window=None,
                                       is_sliding_window=False, is_causal=False)
            enc = enc[:, :, :seq_len, :enc_len]
            sliding = None
            if cfg.use_sliding_window:
                sliding = self._create_4d_mask(seq_len=seq_len, dtype=dtype, device=device,
                                               attention_mask=attention_mask,
                                               sliding_window=cfg.sliding_window,
                                               is_sliding_window=True, is_causal=False)
        return {"full_attention": full, "sliding_attention": sliding,
                "encoder_attention_mask": enc}

    def prepare(self, hidden_states, timestep, timestep_r,
                encoder_hidden_states, encoder_attention_mask,
                context_latents, attention_mask=None):
        """复刻 AceStepDiTModel.forward 的 pre-loop。返回 (hidden, state-dict)。"""
        dit = self.dit
        # --- 时间步 (与原版一致;timestep_r 由调用方决定,见 fusion 的对齐口径) ---
        temb_t, tproj_t = dit.time_embed(timestep)
        temb_r, tproj_r = dit.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = tproj_t + tproj_r

        # --- 拼 context_latents + patchify ---
        hidden = torch.cat([context_latents, hidden_states], dim=-1)
        original_seq_len = hidden.shape[1]
        if hidden.shape[1] % dit.patch_size != 0:
            pad = dit.patch_size - (hidden.shape[1] % dit.patch_size)
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad), value=0)
        hidden = dit.proj_in(hidden)
        enc = dit.condition_embedder(encoder_hidden_states)

        # --- position / rotary ---
        cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = dit.rotary_emb(hidden, position_ids)

        mask_mapping = self._build_masks(hidden, enc, attention_mask)
        state = dict(timestep_proj=timestep_proj, temb=temb,
                     position_embeddings=position_embeddings, position_ids=position_ids,
                     cache_position=cache_position, mask_mapping=mask_mapping,
                     enc=enc, enc_mask=mask_mapping["encoder_attention_mask"],
                     original_seq_len=original_seq_len)
        return hidden, state

    def run_layer(self, j, hidden, state):
        """原生 AceStepDiTLayer.forward(self-attn + (text/lyric)cross-attn + mlp)。"""
        layer = self.dit.layers[j]
        out = layer(
            hidden,
            state["position_embeddings"],
            state["timestep_proj"],
            state["mask_mapping"][layer.attention_type],
            state["position_ids"],
            None,                # past_key_value
            False,               # output_attentions
            False,               # use_cache
            state["cache_position"],
            state["enc"],
            state["enc_mask"],
        )
        return out[0]

    def finalize(self, hidden, state):
        """norm_out + 去 patch + 裁剪 -> [B, T, 64] 的 flow 预测。"""
        dit = self.dit
        shift, scale = (dit.scale_shift_table + state["temb"].unsqueeze(1)).chunk(2, dim=1)
        hidden = (dit.norm_out(hidden) * (1 + scale) + shift).type_as(hidden)
        hidden = dit.proj_out(hidden)
        return hidden[:, :state["original_seq_len"], :]


class CrossAttnAdapter(nn.Module):
    """维度适配(保留以兼容旧脚本)。fusion 内部已用 k/v proj 直接换维,通常不需要它。"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x):
        return self.proj(self.norm(x))
