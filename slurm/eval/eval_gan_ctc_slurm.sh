#!/bin/bash
#SBATCH --job-name=eval_gan_ctc
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_eval_gan_ctc.out
#SBATCH --error=slurm_logs/%j_eval_gan_ctc.err

set -e

export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data

CKPT="lightning_logs/lightning_logs/version_95/checkpoints/last.ckpt"

python evaluate_bd_gan.py \
  --ckpt-path "${CKPT}" \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --tile-size 512 \
  --output bd_results_ctc_gan_v95.csv

echo "=== Done ==="
