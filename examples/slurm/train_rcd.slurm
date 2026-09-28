#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=rcd
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40s
#SBATCH --mem=44G
#SBATCH --time=23:00:00
#SBATCH --output=rcd-%j.out
#SBATCH --error=rcd-%j.err

set -euo pipefail
source /path/to/venv/bin/activate
cd /path/to/rcd-retrieval

rcd-train rcd \
  --encoder e5 \
  --teacher imrnns \
  --teacher-fraction 0.50 \
  --dataset_format comlq \
  --dataset_dir /path/to/datasets/comlq \
  --teacher_checkpoint_path /path/to/runs/teachers/imrnns-e5-comlq-f050-seed42.pt \
  --variants FULL_RCD \
  --seeds 42 \
  --ks 1 2 4 8 16 32 64 \
  --output_dir /path/to/runs/e5-imrnns-comlq-f050
