#!/bin/bash
set -euo pipefail

rcd-train rcd \
  --encoder qwen \
  --teacher search-adaptor \
  --teacher-fraction 0.50 \
  --dataset_format comlq \
  --dataset_dir datasets/comlq \
  --search_adaptor_datasets comlq \
  --variants FULL_RCD \
  --seeds 42 \
  --ks 1 2 4 8 16 32 64 \
  --output_dir runs/qwen-search-adaptor-comlq-f050
