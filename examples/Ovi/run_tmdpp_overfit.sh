#!/usr/bin/env bash
# TMD++ SFT 小数据过拟合启动脚本(Stage 4.2 sanity)
# 先用 10 条干净音乐舞蹈样本 + 480x480 验证:loss 能降、训推一致、音画对齐涌现。
set -e

export METADATA_PATH=${METADATA_PATH:-data/tmdpp_sft10/metadata.jsonl}
export DATASET_BASE_PATH=${DATASET_BASE_PATH:-.}
export ACESTEP_ROOT=${ACESTEP_ROOT:-./ACE-Step-main}
export OUTPUT_DIR=${OUTPUT_DIR:-models/train/TMDpp_sft_overfit}

# 单卡先跑通(A100 80G);多卡用 accelerate config 后改 --num_processes
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  examples/Ovi/train_t2av.py \
  --config-file ovi/configs/training/finetune_tmdpp.yaml \
  --metadata_path "$METADATA_PATH" \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --acestep_project_root "$ACESTEP_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --use_tmdpp_module 1     # 在 train_t2av.py 的 main 里据此选择 TMDppTrainingModule
