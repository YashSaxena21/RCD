#!/bin/bash
set -euo pipefail

COMMON=(
  --encoder e5
  --teacher imrnns
  --teacher-fraction 0.50
  --dataset_format comlq
  --dataset_dir datasets/comlq
  --teacher_checkpoint_path runs/teachers/comlq/imrnns-e5-f050-seed42.pt
  --seeds 42
  --ks 1 2 4 8 16 32 64
)

rcd-train margin-mse "${COMMON[@]}" --output_dir runs/margin-mse-comlq-f050
rcd-train embeddistill "${COMMON[@]}" --output_dir runs/embeddistill-comlq-f050
rcd-train supervised \
  --encoder e5 \
  --dataset_format standard \
  --dataset_dir datasets/comlq \
  --dataset_name comlq \
  --seed 42 \
  --device cuda \
  --fp16 \
  --output_dir runs/supervised-e5-comlq
