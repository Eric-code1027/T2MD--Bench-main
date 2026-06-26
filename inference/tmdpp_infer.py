"""
TMD++ 推理(与 train_tmdpp.py 严格镜像 => 训推一致)
====================================================
关键一致性保证:
* 同一个 FlowMatchScheduler、同一套 timestep。
* 同一个 fusion.forward(逐层双向融合)。
* i2v:每个 denoise step 把 video latent 首帧覆盖为 clean ref(与训练 vid_t[:, :,:1]=ref 对应)。
* audio 时间步 r=t(退化标准 flow),与训练一致。
* video 用 Wan VAE 解码,audio 用 ACE VAE 解码。

CFG 可选;过拟合验证(确认 train==infer / 音画对齐涌现)时建议先关 CFG(guidance=1.0),
排除 CFG 带来的训推差异,验证通过后再开。
"""
import torch
from tqdm import tqdm

from ovi.utils.acestep_loader import (
    decode_latent_to_audio, build_ace_encoder_hidden, build_context_latents_default,
)
from ovi.utils.processing_utils import preprocess_image_tensor


@torch.no_grad()
def tmdpp_generate(
    model,                       # FusionAceStepLayerwise(已 load 训练好的融合层)
    vae_model_video,             # Wan VAE
    text_model,                  # T5
    scheduler,                   # 与训练同一个 FlowMatchScheduler
    *,
    prompt: str,
    audio_caption: str,
    image_ref_path: str,
    video_latent_shape,          # (C, F, H, W)  latent 维度
    audio_latent_len: int,       # T (ACE latent 帧数)
    num_steps: int = 50,
    guidance: float = 1.0,       # 1.0 = 关 CFG
    neg_prompt: str = "",
    lyrics: str = "[Instrumental]",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    handler = model.acestep_handler
    C, F, H, W = video_latent_shape

    # ---- 文本 ----
    txt_pos = [e.to(dtype) for e in text_model([prompt], text_model.device)]
    txt_neg = [e.to(dtype) for e in text_model([neg_prompt], text_model.device)] if guidance != 1.0 else None

    # ---- ACE encoder hidden / context latents ----
    ace_enc, ace_enc_mask = build_ace_encoder_hidden(handler, audio_caption, lyrics, device)
    ace_ctx = build_context_latents_default(handler, audio_latent_len, device, dtype)

    # ---- image ref -> clean 首帧 latent ----
    first_frame = preprocess_image_tensor(image_ref_path, device, dtype)         # [C_img, H, W]
    ref_latent = vae_model_video.wrapped_encode(first_frame[:, :, None]).to(dtype).squeeze(0)  # [C, 1, h, w]

    # ---- 初始噪声 ----
    g = torch.Generator(device=device).manual_seed(seed)
    video_noise = torch.randn((C, F, H, W), device=device, dtype=dtype, generator=g)
    audio_noise = torch.randn((1, audio_latent_len, 64), device=device, dtype=dtype, generator=g)

    ph, pw = model.video_model.patch_size[1], model.video_model.patch_size[2]
    vid_seq_len = F * H * W // (ph * pw)

    scheduler.set_timesteps(num_steps, training=False)
    vid = video_noise
    aud = audio_noise

    for t in tqdm(scheduler.timesteps):
        timestep = torch.full((1,), float(t), device=device, dtype=dtype)
        # ACE 时间步 == sigma ∈ [0,1](与训练一致;否则音频塔时间嵌入错乱)
        sigma = scheduler.get_sigma_from_timestep(timestep).to(device=device, dtype=dtype)
        # i2v:每步覆盖首帧(与训练一致)
        vid[:, :1] = ref_latent

        def _fwd(txt):
            return model(
                vid=[vid], audio_latent=aud, t=timestep, vid_context=txt,
                ace_encoder_hidden_states=ace_enc,
                ace_encoder_attention_mask=ace_enc_mask,
                ace_context_latents=ace_ctx,
                vid_seq_len=vid_seq_len, ace_timestep=sigma, first_frame_is_clean=True)

        v_pos, a_pos = _fwd(txt_pos)
        if guidance != 1.0:
            v_neg, a_neg = _fwd(txt_neg)
            v_pred = v_neg + guidance * (v_pos - v_neg)
            a_pred = a_neg + guidance * (a_pos - a_neg)
        else:
            v_pred, a_pred = v_pos, a_pos

        # flow-matching 一步更新(用与训练同一个 scheduler 的 step)
        vp = v_pred[0] if isinstance(v_pred, (list, tuple)) else v_pred
        vid = scheduler.step(vp, t, vid)
        aud = scheduler.step(a_pred, t, aud)

    vid[:, :1] = ref_latent  # 输出前再钉一次首帧
    video = vae_model_video.wrapped_decode(vid.unsqueeze(0))
    audio = decode_latent_to_audio(handler, aud)
    return video, audio
