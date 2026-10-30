#!/bin/bash
#SBATCH --job-name=eval_dbonly_ctc
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_eval_dbonly_ctc.out
#SBATCH --error=slurm_logs/%j_eval_dbonly_ctc.err

set -e

export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data

CKPT="lightning_logs/lightning_logs/version_98/checkpoints/dense-best-epoch=197-val_loss=0.00000.ckpt"

python evaluate_bd.py \
  --ckpt-path "${CKPT}" \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_dbonly" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --tile-size 512 \
  --output bd_results_ctc_v98_dbonly.csv

echo "=== Done ==="
