# TMD++ : ACE-Step base + image-ref 逐层融合 + GRPO 偏好优化

在 `Yangxiaoda1/T2MD-Bench`(Ovi 双塔)基础上,把音频塔换成 **ACE-Step base** 并做 **真·逐层双向融合**,
补上训练侧的 **image reference (i2v)**,再叠加 **GRPO** 组内偏好后训练。
覆盖 proposal 的 Stage 4.2(SFT 基座)与第六/七节(GRPO)。

---

## 1. 环境配置

```bash
# 基座(Ovi / T2MD-Bench),Linux + CUDA
conda create -n tmdpp python=3.12.2 -y
conda activate tmdpp

# 1) 克隆干净基座
git clone https://github.com/Yangxiaoda1/T2MD-Bench.git
cd T2MD-Bench
pip install -r requirements.txt              # 含 torch / transformers / diffusers / accelerate / deepspeed

# 2) 放入 ACE-Step 源码(来自 repo2: Eric-code1027/Commit-of-T2MD-Bench-Initial- 里 vendor 的 ACE-Step-main)
#    复制整个 ACE-Step-main/ 到仓库根目录,并装它的依赖
cp -r /path/to/repo2/ACE-Step-main ./ACE-Step-main
pip install -r ACE-Step-main/requirements.txt
```

> **版本注意**:ACE-Step 要求 `transformers>=4.51,<4.58`。若 Ovi 的 requirements 钉了别的版本导致冲突,
> 以 ACE 的区间为准(融合时两塔同进程,transformers 必须统一)。`flash_attn` 对 Ovi 视频塔可选;
> ACE 侧本项目强制走 sdpa(`use_flash_attention=False`)以产生 4D mask,无需 flash。

**硬件**:过拟合 sanity ≥1×A100 80G;TMD++ full 建议 8×A100 80G(见 proposal 十一)。

---

## 2. 放置本项目交付的文件(全部新增,基于 T2MD-Bench)

```
ovi/modules/fusion_acestep_layerwise.py        # 核心:Wan(30层)<->ACE-base(24层) 真·逐层融合
ovi/utils/acestep_loader.py                    # ACE 加载 + VAE编解码 + encoder_hidden + AceDiTRunner
examples/Ovi/train_tmdpp.py                    # SFT 训练模块(image-ref + 共用调度 + 双loss)
examples/Ovi/grpo_tmdpp.py                     # GRPO 组内偏好训练
examples/Ovi/grpo_preference_5x4.example.jsonl # 5x4 偏好 metadata 示例
examples/Ovi/run_tmdpp_overfit.sh              # SFT 过拟合启动
examples/Ovi/run_grpo_tmdpp.sh                 # GRPO 启动
inference/tmdpp_infer.py                        # 与训练镜像的推理(训推一致)
ovi/configs/training/finetune_tmdpp.yaml       # 训练配置
```

唯一需改原文件:`examples/Ovi/train_t2av.py` 的 `main()` 加一行——当 `--use_tmdpp_module` 为真时
用 `TMDppTrainingModule` 替代 `OviTrainingModule`(启动脚本已传该参数)。

权重:`ckpts/` 放 Wan VAE / T5 / Ovi video DiT;ACE-base 权重由 `ace_config_path` 指向的目录加载。

---

## 3. 数据准备

**SFT 过拟合(10 条)**:`data/tmdpp_sft10/metadata.jsonl`,字段同 proposal 4.1
(`image_ref_path/video_path/audio_path/video_caption/audio_caption/joint_prompt`);
`image_ref` 取视频首帧即可。

**GRPO 5x4**:目录结构(proposal 8.9)
```
data/grpo_5x4/
  metadata.jsonl              # 20 行,见 grpo_preference_5x4.example.jsonl
  prompt_001/ ref.png score_1.mp4 score_1.wav ... score_4.mp4 score_4.wav
  ...
  prompt_005/
```
mp4 需提前抽出对应 wav;`prompt` 用 `<audio_cap>...</audio_cap><sep><video_cap>...</video_cap>`。

---

## 4. 运行流程

```bash
# Step 1  SFT 基座过拟合(先 480、关 CFG)
bash examples/Ovi/run_tmdpp_overfit.sh
#   验收:loss 降、不 NaN;同 prompt+ref 推理逐渐贴近 GT;训推一致

# Step 2  推理(过拟合后做三段对比)
python inference/tmdpp_infer.py   # 或在脚本里调用 tmdpp_generate(...)

# Step 3  GRPO 组内偏好后训练(从 SFT checkpoint 初始化)
SFT_CKPT=models/train/TMDpp_sft_overfit/grpo_fusion_final.pt \
bash examples/Ovi/run_grpo_tmdpp.sh
#   验收(8.11):L_best 降;logp 中 score=4 最大、score=1 最小;推理偏向高分候选
```

关键超参(`finetune_tmdpp.yaml` / 环境变量):`lr=1e-5`、`dataset_repeat=20`、`epochs=50`、
`group_temperature=1.0`、`sft_weight=0.2`、video/audio loss = 0.85/0.15。

---


**未验证(需有权重的 CUDA 机器)**:端到端前反传、显存、与真实 ACE/Wan 权重的数值行为 —— 见第 6 节。
