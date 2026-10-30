#!/bin/bash
#SBATCH --job-name=eval_lpips
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e
export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs


python eval_lpips.py \
  --decoded-root /net/tscratch/people/plgkarolbogumil/data/decoded_CTC_YUV420_10bit_nofilter \
  --neighbor-root /net/tscratch/people/plgkarolbogumil/data/dograna_data/decoded_CTC_neighbors_nofilter \
  --original-root /net/tscratch/people/plgkarolbogumil/data/splitted_CTC \
  --mse-ckpt "lightning_logs/lightning_logs/version_2664635/checkpoints/temporal-best-epoch=199-val_loss=0.00000.ckpt" \
  --gan-ckpt "lightning_logs/lightning_logs/version_2784784/checkpoints/last.ckpt" \
  --output lpips_comparison.csv
