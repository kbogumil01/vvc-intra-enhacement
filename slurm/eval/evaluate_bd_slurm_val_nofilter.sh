#!/bin/bash
#SBATCH --job-name=eval_bd_ppff_nf_val
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data

python evaluate_bd.py \
  --model ppff_base \
  --ckpt-path "/net/people/plgrid/plgkarolbogumil/mgr/dense/lightning_logs/lightning_logs/version_93/checkpoints/dense-best-epoch=199-val_loss=0.00000.ckpt" \
  --decoded-root "${DATA}/decoded_BVI_frames_YUV420_10bit_nofilter" \
  --original-root "${DATA}/splitted_DVI_BVC" \
  --output bd_results_val_v93_nofilter.csv
