"""
TMD++ SFT 训练模块(Stage 4.2)
=================================
在 examples/Ovi/train_t2av.py 的基础上做最小必要改动:

1. image reference 进训练(原 train_t2av 缺这一步,是训推不一致的根因):
   取视频 latent 的首帧作为 clean ref,覆盖加噪后的首帧,并 first_frame_is_clean=True;
   首帧不计入 loss。 => 与推理引擎 i2v 路径(每步 video_noise[:, :1]=ref)完全一致。

2. 音频塔换成 ACE-Step base,走真·逐层融合(ovi.modules.fusion_acestep_layerwise)。

3. 两塔共用同一个 flow-matching 调度与同一个 timestep(对齐口径),保证 train==infer。

用法:把本文件放到 examples/Ovi/,在 train_t2av.py 里把
   model = OviTrainingModule(...)  改为  model = TMDppTrainingModule(...)
其余(WanTrainingModule / launch_training_task / dataloader)沿用原 train_t2av.py。
"""
import os
import torch
import torch.nn as nn
from accelerate import Accelerator

from diffsynth.trainers.utils import DiffusionTrainingModule

from ovi.modules.fusion_acestep_layerwise import FusionAceStepLayerwise
from ovi.utils.acestep_loader import (
    encode_audio_to_latent, build_ace_encoder_hidden, build_context_latents_default,
)
# 复用原 train_t2av 的组件加载(VAE / T5 / scheduler 等)
from ovi.utils.model_loading_utils import (
    init_wan_vae_2_2, init_text_model, load_fusion_checkpoint,
)

accelerator = Accelerator()


class TMDppTrainingModule(DiffusionTrainingModule):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, config=None,
                 acestep_project_root=None, ace_config_path="acestep-v15-base"):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.target_dtype = torch_dtype
        self.config = config

        # ---- video DiT 配置(沿用 ovi 的 video.json) ----
        import json
        with open(config["video_config"]) as f:
            video_cfg = json.load(f)

        # ---- 融合模型:Wan video DiT + ACE-Step base audio DiT(逐层) ----
        self.model = FusionAceStepLayerwise(
            video_config=video_cfg,
            acestep_project_root=acestep_project_root,
            ace_config_path=ace_config_path,
            device=device, dtype=torch_dtype,
        )
        # 只把 Wan video DiT 主干权重灌进来(ACE 由 handler 自带权重);融合层从零初始化。
        ckpt = config.get("video_fusion_ckpt", None)
        if ckpt:
            load_fusion_checkpoint(self.model.video_model, checkpoint_path=ckpt, from_meta=False)

        # ---- 冻结的 video VAE / T5(audio VAE & ACE encoder 已在 fusion 内冻结) ----
        self.vae_model_video = init_wan_vae_2_2(config["ckpt_dir"], rank=device)
        self.vae_model_video.model.requires_grad_(False).eval()
        self.text_model = init_text_model(config["ckpt_dir"], rank=device)
        for p in self.text_model.parameters():
            p.requires_grad = False

        # ---- 调度器(两塔共用) ----
        from ovi.utils.fm_solvers_unipc import FlowMatchScheduler  # 与 train_t2av 同源,按实际路径调整
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        self.handler = self.model.acestep_handler

    # ------------------------------------------------------------------ #
    def forward(self, **inputs):
        cfg = self.config
        # ---------- 文本 / 视频 latent ----------
        text_embeddings = self.text_model(inputs["prompt"], self.text_model.device)
        text_embeddings = [emb.to(self.target_dtype) for emb in text_embeddings]

        latent_video = self.vae_model_video.wrapped_encode(inputs["video"]).to(self.torch_dtype)
        # latent_video: [B, C, F, H, W]  (B=1)

        # ---------- 音频 latent(ACE VAE) ----------
        audio_latent = encode_audio_to_latent(self.handler, inputs["audio"]).to(self.torch_dtype)  # [1, T, 64]
        Ta = audio_latent.shape[1]
        ace_enc, ace_enc_mask = build_ace_encoder_hidden(
            self.handler,
            text_caption=inputs.get("audio_caption", inputs["prompt"][0]),
            lyrics=inputs.get("lyrics", "[Instrumental]"),
            device=self.device)
        ace_ctx = build_context_latents_default(self.handler, Ta, self.device, self.torch_dtype)

        # ---------- 共用 timestep + flow matching ----------
        max_b = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_b = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        tid = torch.randint(min_b, max_b, (1,))
        timestep = self.scheduler.timesteps[tid].to(dtype=self.torch_dtype, device=self.device)
        # ACE 时间步 == 同一噪声级的 sigma ∈ [0,1](见 fusion.forward 说明)
        sigma = self.scheduler.get_sigma_from_timestep(timestep).to(
            dtype=self.torch_dtype, device=self.device)

        noise_v = torch.randn_like(latent_video)
        noise_a = torch.randn_like(audio_latent)

        vid_t = self.scheduler.add_noise(latent_video, noise_v, timestep)
        vid_target = self.scheduler.training_target(latent_video, noise_v, timestep)
        audio_t = self.scheduler.add_noise(audio_latent, noise_a, timestep)
        audio_target = self.scheduler.training_target(audio_latent, noise_a, timestep)

        # ---------- image reference:首帧设为 clean(= GT 首帧 latent),i2v 路径 ----------
        # latent_video[:, :, 0] 即首帧 latent;覆盖加噪结果的首帧。
        vid_t[:, :, :1] = latent_video[:, :, :1]

        # ---------- 计算 max_seq_len_video ----------
        ph, pw = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
        max_seq_len_video = vid_t.shape[2] * vid_t.shape[3] * vid_t.shape[4] // (ph * pw)

        # fusion 的 video 塔接口吃 list[ [C,F,H,W] ]
        vid_pred, audio_pred = self.model(
            vid=[vid_t.squeeze(0)],
            audio_latent=audio_t,
            t=timestep,
            vid_context=text_embeddings,
            ace_encoder_hidden_states=ace_enc,
            ace_encoder_attention_mask=ace_enc_mask,
            ace_context_latents=ace_ctx,
            vid_seq_len=max_seq_len_video,
            ace_timestep=sigma,
            first_frame_is_clean=True,
        )

        weight = self.scheduler.training_weight(timestep)

        # ---------- loss:首帧(clean condition)不计 video loss ----------
        vp = vid_pred[0] if isinstance(vid_pred, (list, tuple)) else vid_pred  # [C, F, H, W]
        vt = vid_target.squeeze(0)                                            # [C, F, H, W]
        loss_video = torch.nn.functional.mse_loss(
            vp[:, 1:].float(), vt[:, 1:].float()) * weight
        loss_audio = torch.nn.functional.mse_loss(
            audio_pred.float(), audio_target.float()) * weight

        loss = 0.85 * loss_video + 0.15 * loss_audio
        if accelerator.is_main_process:
            print(f"[TMD++] loss_video {loss_video.item():.4f} "
                  f"loss_audio {loss_audio.item():.4f} t={int(timestep.item())}")
        return loss

    # 只暴露可训练参数给 optimizer
    def trainable_parameters(self):
        return self.model.trainable_parameters()
