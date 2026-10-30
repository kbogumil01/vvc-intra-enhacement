#!/bin/bash
#SBATCH --job-name=gen_compare
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e
export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data

MSE_CKPT="lightning_logs/lightning_logs/version_2664635/checkpoints/temporal-best-epoch=199-val_loss=0.00000.ckpt"
GAN_CKPT="lightning_logs/lightning_logs/version_2784784/checkpoints/last.ckpt"

python generate_comparison.py \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --neighbor-root "${DATA}/dograna_data/decoded_CTC_neighbors_nofilter" \
  --original-root "${DATA}/splitted_CTC" \
  --mse-ckpt "${MSE_CKPT}" \
  --gan-ckpt "${GAN_CKPT}" \
  --sequence "RitualDance_1920x1080_10bit_420" \
  --qp 37 \
  --crop 150 1280 180 240 \
  --output comparison_gan_ritual.png

python generate_comparison.py \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --neighbor-root "${DATA}/dograna_data/decoded_CTC_neighbors_nofilter" \
  --original-root "${DATA}/splitted_CTC" \
  --mse-ckpt "${MSE_CKPT}" \
  --gan-ckpt "${GAN_CKPT}" \
  --sequence "PartyScene_832x480_10bit_420" \
  --qp 37 \
  --output comparison_gan_party.png

python generate_comparison.py \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --neighbor-root "${DATA}/dograna_data/decoded_CTC_neighbors_nofilter" \
  --original-root "${DATA}/splitted_CTC" \
  --mse-ckpt "${MSE_CKPT}" \
  --gan-ckpt "${GAN_CKPT}" \
  --sequence "Cactus_1920x1080_10bit_420" \
  --qp 37 \
  --output comparison_gan_cactus.png

echo "=== Done ==="
ls -la comparison_gan_*.png
