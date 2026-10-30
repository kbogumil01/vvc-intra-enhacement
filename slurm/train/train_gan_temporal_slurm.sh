#!/bin/bash
#SBATCH --job-name=ppff_gan_cond
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=800G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data/temporal_memmap_nofilter
GEN_CKPT="lightning_logs/lightning_logs/version_2664635/checkpoints/temporal-best-epoch=199-val_loss=0.00000.ckpt"


python train_gan.py \
  --generator-ckpt "${GEN_CKPT}" \
  --generator-model ppff_base \
  --temporal-memmap-dir "${DATA}" \
  --in-channels 12 \
  --patch-size 240 \
  --patches-per-frame 2 \
  --batch-size 8 \
  --accumulate-grad-batches 4 \
  --lambda-pixel 1.0 \
  --lambda-perceptual 0.0 \
  --lambda-adv 0.05 \
  --lambda-r1 0.0 \
  --lr-g 5e-6 \
  --lr-d 5e-4 \
  --warmup-epochs 0 \
  --d-pretrain-epochs 5 \
  --d-type ppff \
  --d-num-blocks 16 \
  --d-channels 128 \
  --d-conditional \
  --d-no-spectral-norm \
  --r1-every 16 \
  --max-epochs 100 \
  --early-stop-patience 40 \
  --precision 32
