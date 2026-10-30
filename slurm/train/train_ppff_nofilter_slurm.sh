#!/bin/bash
#SBATCH --job-name=ppff_base_nofilter
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

python train.py \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_BVI_frames_YUV420_10bit_nofilter" \
  --original-root "${DATA}/splitted_DVI_BVC" \
  --extra-decoded-root "${DATA}/decoded_DIV2K_YUV420_10bit_nofilter" \
  --extra-original-root "${DATA}/splitted_DIV2K" \
  --memmap-dir "${MEMMAP}" \
  --patch-size 256 \
  --patches-per-frame 2 \
  --batch-size 12 \
  --accumulate-grad-batches 3 \
  --lr 1e-4 \
  --max-epochs 200 \
  --early-stop-patience 20 \
  --precision 16-mixed \
  --num-workers 14
