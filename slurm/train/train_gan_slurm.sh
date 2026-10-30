#!/bin/bash
#SBATCH --job-name=ppff_gan_nofilter
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data
MEMMAP=${DATA}/memmap_cache_nofilter

GEN_CKPT="/net/people/plgrid/plgkarolbogumil/mgr/dense/lightning_logs/lightning_logs/version_93_wytrenowany_pff_baseline_nofilter/checkpoints/dense-best-epoch=199-val_loss=0.00000.ckpt"


python train_gan.py \
  --generator-ckpt "${GEN_CKPT}" \
  --generator-model ppff_base \
  --decoded-root "${DATA}/decoded_BVI_frames_YUV420_10bit_nofilter" \
  --original-root "${DATA}/splitted_DVI_BVC" \
  --extra-decoded-root "${DATA}/decoded_DIV2K_YUV420_10bit_nofilter" \
  --extra-original-root "${DATA}/splitted_DIV2K" \
  --memmap-dir "${MEMMAP}" \
  --patch-size 256 \
  --patches-per-frame 2 \
  --batch-size 8 \
  --accumulate-grad-batches 4 \
  --lambda-pixel 1.0 \
  --lambda-perceptual 0.01 \
  --lambda-adv 0.01 \
  --lambda-r1 10.0 \
  --lr-g 1e-5 \
  --lr-d 2e-4 \
  --warmup-epochs 5 \
  --ndf 64 \
  --d-layers 3 \
  --r1-every 16 \
  --max-epochs 100 \
  --early-stop-patience 30 \
  --precision 32
