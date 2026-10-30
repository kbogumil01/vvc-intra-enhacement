#!/bin/bash
#SBATCH --job-name=crossref_bd_ctc
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data

python evaluate_bd_crossref.py \
  --anchor-decoded-root "${DATA}/decoded_CTC_YUV420_10bit" \
  --nofilter-decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --original-root "${DATA}/splitted_CTC" \
  --filter-ckpt-path "/net/people/plgrid/plgkarolbogumil/mgr/dense/lightning_logs/lightning_logs/version_92_wytrenowany_ppff_baseline/checkpoints/dense-best-epoch=193-val_loss=0.00000.ckpt" \
  --nofilter-ckpt-path "/net/people/plgrid/plgkarolbogumil/mgr/dense/lightning_logs/lightning_logs/version_93/checkpoints/dense-best-epoch=199-val_loss=0.00000.ckpt" \
  --model ppff_base \
  --no-split \
  --output bd_results_crossref_ctc.csv
