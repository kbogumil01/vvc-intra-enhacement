#!/bin/bash
#SBATCH --job-name=preprocess_memmap
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_preprocess.out
#SBATCH --error=slurm_logs/%j_preprocess.err

set -e

export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data
OUTDIR=${DATA}/memmap_cache

python preprocess_memmap.py \
  --decoded-root \
    "${DATA}/decoded_BVI_frames_YUV420_10bit" \
    "${DATA}/decoded_DIV2K_YUV420_10bit" \
  --original-root \
    "${DATA}/splitted_DVI_BVC" \
    "${DATA}/splitted_DIV2K" \
  --output-dir "${OUTDIR}"

echo "=== Preprocessing complete ==="
ls -lh "${OUTDIR}/"
