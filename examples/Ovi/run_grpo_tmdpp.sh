#!/usr/bin/env bash
# TMD++ GRPO 5x4 过拟合验证(proposal 第七节示例 + 8.10 配置)
# 正式 GRPO 必须从 TMD++ SFT/full checkpoint 初始化(SFT_CKPT)。
set -e

export METADATA_PATH=${METADATA_PATH:-data/grpo_5x4/metadata.jsonl}
export DATASET_BASE_PATH=${DATASET_BASE_PATH:-.}
export ACESTEP_ROOT=${ACESTEP_ROOT:-./ACE-Step-main}
export OUTPUT_DIR=${OUTPUT_DIR:-models/train/TMDpp_grpo_overfit}
export SFT_CKPT=${SFT_CKPT:-models/train/TMDpp_sft_overfit/grpo_fusion_final.pt}  # 从 SFT 融合层初始化

export NUM_EPOCHS=${NUM_EPOCHS:-50}
export DATASET_REPEAT=${DATASET_REPEAT:-20}
export LR=${LR:-1e-5}
export GROUP_TEMPERATURE=${GROUP_TEMPERATURE:-1.0}
export SFT_WEIGHT=${SFT_WEIGHT:-0.2}
export CONFIG=${CONFIG:-ovi/configs/training/finetune_tmdpp.yaml}

accelerate launch --num_processes 1 --mixed_precision bf16 \
  examples/Ovi/grpo_tmdpp.py
