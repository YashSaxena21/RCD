#!/bin/bash
set -euo pipefail

rcd-train rcd \
  --encoder e5 \
  --teacher imrnns \
  --teacher-fraction 0.50 \
  --dataset_format comlq \
  --dataset_dir datasets/comlq \
  --teacher_checkpoint_path runs/teachers/comlq/imrnns-e5-f050-seed42.pt \
  --variants RANKING_ONLY CORRECTION_ONLY FULL_RCD \
  --seeds 42 \
  --ks 1 2 4 8 16 32 64 \
  --output_dir runs/e5-imrnns-comlq-f050
