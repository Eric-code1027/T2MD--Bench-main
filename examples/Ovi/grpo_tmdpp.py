"""
TMD++ GRPO 偏好后训练(对齐 proposal 第六/七节)
================================================
组内相对偏好优化(group-relative),不是完整 diffusion-trajectory GRPO。
复用 SFT 的融合塔与候选 loss(train_tmdpp),保证训推口径一致。

公式:
  A_i      = (score_i - mean(scores)) / std(scores)      # detach,常量
  L_i      = 0.85 * L_video_i + 0.15 * L_audio_i          # 单候选 flow-matching 去噪 loss
  log p_i  = log_softmax(-L_i / temperature)
  L_GRPO   = - mean(A_i * log p_i)
  L_best   = 最高分候选的 L_i(SFT 锚)
  L_total  = L_GRPO + lambda_sft * L_best

关键:同一个 prompt group 的 4 个候选共用同一个 timestep(噪声各自采),
让 L_i 的差异反映模型对候选的拟合好坏,而非 timestep 方差。
"""
import os
import json
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

from examples.Ovi.train_tmdpp import TMDppTrainingModule
from ovi.utils.acestep_loader import (
    encode_audio_to_latent, build_ace_encoder_hidden, build_context_latents_default,
)
from ovi.utils.processing_utils import load_video_tensor, load_audio_tensor, preprocess_image_tensor

accelerator = Accelerator()


# --------------------------------------------------------------------------- #
#  数据:每次返回一个 prompt group(4 候选)
# --------------------------------------------------------------------------- #
class GroupedPreferenceDataset(Dataset):
    def __init__(self, metadata_path, base_path=".", repeat=1):
        with open(metadata_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        groups = OrderedDict()
        for r in rows:
            groups.setdefault(r["prompt_id"], []).append(r)
        # 每组按 score 升序,保证索引稳定
        self.groups = [sorted(v, key=lambda x: x["score"]) for v in groups.values()]
        for g in self.groups:
            assert len(g) == 4, f"每个 prompt 必须 4 候选,得到 {len(g)}"
        self.base = base_path
        self.repeat = repeat

    def __len__(self):
        return len(self.groups) * self.repeat

    def __getitem__(self, idx):
        g = self.groups[idx % len(self.groups)]
        cands = []
        for r in g:
            cands.append(dict(
                prompt=r["prompt"],
                audio_caption=r.get("audio_caption", r["prompt"]),
                lyrics=r.get("lyrics", "[Instrumental]"),
                video_path=os.path.join(self.base, r["video_path"]),
                audio_path=os.path.join(self.base, r["audio_path"]),
                image_ref_path=os.path.join(self.base, r["image_ref_path"]),
                score=float(r["score"]),
            ))
        return cands   # list[4]


def _collate(batch):
    # batch_size 固定为 1 个 group
    assert len(batch) == 1
    return batch[0]


# --------------------------------------------------------------------------- #
#  GRPO Trainer
# --------------------------------------------------------------------------- #
class TMDppGRPOTrainer(TMDppTrainingModule):
    def __init__(self, *args, group_temperature=1.0, sft_weight=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = group_temperature
        self.sft_weight = sft_weight

    # ---- 单候选 flow-matching 去噪 loss(= SFT 那套,但 timestep 由外部传入以组内共用)----
    def _candidate_loss(self, sample, timestep):
        # 视频 / 音频 latent
        latent_video = self.vae_model_video.wrapped_encode(
            load_video_tensor(sample["video_path"], self.device, self.torch_dtype)
        ).to(self.torch_dtype)                                   # [1,C,F,H,W]
        audio = load_audio_tensor(sample["audio_path"], self.device, self.torch_dtype)
        audio_latent = encode_audio_to_latent(self.handler, audio).to(self.torch_dtype)  # [1,T,64]
        Ta = audio_latent.shape[1]

        # 文本 / ACE 条件
        text_emb = [e.to(self.target_dtype) for e in
                    self.text_model([sample["prompt"]], self.text_model.device)]
        ace_enc, ace_enc_mask = build_ace_encoder_hidden(
            self.handler, sample["audio_caption"], sample["lyrics"], self.device)
        ace_ctx = build_context_latents_default(self.handler, Ta, self.device, self.torch_dtype)

        # flow matching(共用 timestep,噪声各自采)
        noise_v = torch.randn_like(latent_video)
        noise_a = torch.randn_like(audio_latent)
        vid_t = self.scheduler.add_noise(latent_video, noise_v, timestep)
        vid_tgt = self.scheduler.training_target(latent_video, noise_v, timestep)
        aud_t = self.scheduler.add_noise(audio_latent, noise_a, timestep)
        aud_tgt = self.scheduler.training_target(audio_latent, noise_a, timestep)
        # ACE 时间步 == sigma ∈ [0,1]
        sigma = self.scheduler.get_sigma_from_timestep(timestep).to(
            dtype=self.torch_dtype, device=self.device)

        # image-ref 首帧 clean(i2v),与 SFT/推理一致
        vid_t[:, :, :1] = latent_video[:, :, :1]

        ph, pw = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
        seq_len = vid_t.shape[2] * vid_t.shape[3] * vid_t.shape[4] // (ph * pw)

        vid_pred, aud_pred = self.model(
            vid=[vid_t.squeeze(0)], audio_latent=aud_t, t=timestep,
            vid_context=text_emb, ace_encoder_hidden_states=ace_enc,
            ace_encoder_attention_mask=ace_enc_mask, ace_context_latents=ace_ctx,
            vid_seq_len=seq_len, ace_timestep=sigma, first_frame_is_clean=True)

        w = self.scheduler.training_weight(timestep)
        vp = vid_pred[0] if isinstance(vid_pred, (list, tuple)) else vid_pred
        vt = vid_tgt.squeeze(0)
        Lv = F.mse_loss(vp[:, 1:].float(), vt[:, 1:].float()) * w   # 首帧不计
        La = F.mse_loss(aud_pred.float(), aud_tgt.float()) * w
        return 0.85 * Lv + 0.15 * La

    # ---- GRPO forward:一个 prompt group ----
    def forward(self, group):
        scores = torch.tensor([c["score"] for c in group],
                              device=self.device, dtype=torch.float32)
        # 组内优势(detach)
        adv = (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-6)
        adv = adv.detach()

        # 组内共用 timestep
        max_t = self.scheduler.num_train_timesteps
        tid = torch.randint(0, max_t, (1,))
        timestep = self.scheduler.timesteps[tid].to(dtype=self.torch_dtype, device=self.device)

        losses = [self._candidate_loss(c, timestep) for c in group]   # 4 个 L_i
        L = torch.stack(losses)                                       # [4]

        # log p_i = log_softmax(-L_i / temp)
        logp = F.log_softmax(-L / self.temperature, dim=0)
        L_grpo = -(adv * logp).mean()

        # best-candidate SFT 锚(score 升序排过 => 最后一个是最高分)
        L_best = losses[-1]
        L_total = L_grpo + self.sft_weight * L_best

        if accelerator.is_main_process:
            with torch.no_grad():
                print(f"[GRPO] L_total {L_total.item():.4f} | L_grpo {L_grpo.item():.4f} "
                      f"| L_best {L_best.item():.4f} | L_i {[round(x.item(),3) for x in losses]} "
                      f"| logp {[round(x.item(),3) for x in logp]}")
        return L_total


# --------------------------------------------------------------------------- #
#  保存:只存可训练参数(融合子层)
# --------------------------------------------------------------------------- #
def save_checkpoint(trainer, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    sd = {n: p.detach().cpu() for n, p in trainer.model.named_parameters() if p.requires_grad}
    path = os.path.join(out_dir, f"grpo_fusion_{tag}.pt")
    torch.save(sd, path)
    if accelerator.is_main_process:
        print(f"[GRPO] saved {len(sd)} tensors -> {path}")


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    import yaml
    cfg_path = os.environ.get("CONFIG", "ovi/configs/training/finetune_tmdpp.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    metadata = os.environ["METADATA_PATH"]
    base = os.environ.get("DATASET_BASE_PATH", ".")
    ace_root = os.environ.get("ACESTEP_ROOT", config.get("acestep_project_root"))
    out_dir = os.environ.get("OUTPUT_DIR", "models/train/TMDpp_grpo")
    epochs = int(os.environ.get("NUM_EPOCHS", config.get("num_epochs", 50)))
    repeat = int(os.environ.get("DATASET_REPEAT", config.get("dataset_repeat", 20)))
    lr = float(os.environ.get("LR", config.get("learning_rate", 1e-5)))
    temp = float(os.environ.get("GROUP_TEMPERATURE", 1.0))
    sft_w = float(os.environ.get("SFT_WEIGHT", 0.2))
    sft_ckpt = os.environ.get("SFT_CKPT", config.get("tmdpp_sft_ckpt", None))

    trainer = TMDppGRPOTrainer(
        device="cuda", config=config,
        acestep_project_root=ace_root,
        ace_config_path=config.get("ace_config_path", "acestep-v15-base"),
        group_temperature=temp, sft_weight=sft_w,
    )
    # 正式 GRPO 必须从 TMD++ SFT/full checkpoint 初始化,而非从零
    if sft_ckpt and os.path.exists(sft_ckpt):
        sd = torch.load(sft_ckpt, map_location="cpu")
        miss = trainer.model.load_state_dict(sd, strict=False)
        if accelerator.is_main_process:
            print(f"[GRPO] init from SFT ckpt {sft_ckpt} | missing={len(miss.missing_keys)}")

    ds = GroupedPreferenceDataset(metadata, base, repeat=repeat)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_collate, num_workers=2)

    opt = torch.optim.AdamW(trainer.model.trainable_parameters(), lr=lr)
    trainer, opt, dl = accelerator.prepare(trainer, opt, dl)
    trainer.train()

    step = 0
    for ep in range(epochs):
        for group in dl:
            opt.zero_grad()
            loss = trainer(group)
            accelerator.backward(loss)
            opt.step()
            step += 1
        if (ep + 1) % 10 == 0 and accelerator.is_main_process:
            save_checkpoint(accelerator.unwrap_model(trainer), out_dir, f"ep{ep+1}")
    if accelerator.is_main_process:
        save_checkpoint(accelerator.unwrap_model(trainer), out_dir, "final")


if __name__ == "__main__":
    main()
