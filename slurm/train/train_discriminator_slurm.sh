#!/bin/bash
#SBATCH --job-name=ppff_d_train
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


python train_discriminator.py \
  --temporal-memmap-dir "${DATA}" \
  --generator-ckpt "${GEN_CKPT}" \
  --generator-model ppff_base \
  --in-channels 12 \
  --d-num-blocks 16 \
  --d-channels 128 \
  --lr 2e-4 \
  --hard-negative-ratio 0.5 \
  --patch-size 240 \
  --patches-per-frame 2 \
  --batch-size 12 \
  --max-epochs 30 \
  --early-stop-patience 10 \
  --precision 32

echo ""
echo "Phase 1 complete. D checkpoint saved."
echo "Submit train_gan_with_pretrained_d_slurm.sh next with the D checkpoint path."
