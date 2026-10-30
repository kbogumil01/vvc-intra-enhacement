#!/bin/bash
#SBATCH --job-name=preproc_part_v2
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_preproc_part_v2.out
#SBATCH --error=slurm_logs/%j_preproc_part_v2.err

set -e

export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data

echo "=== Preprocessing partition memmap v2 (4ch): filter ==="
python preprocess_partition_memmap.py \
  --decoded-root \
    "${DATA}/decoded_BVI_frames_YUV420_10bit" \
    "${DATA}/decoded_DIV2K_YUV420_10bit" \
  --original-root \
    "${DATA}/splitted_DVI_BVC" \
    "${DATA}/splitted_DIV2K" \
  --output-dir "${DATA}/partition_memmap_cache_v2"

echo ""

echo "=== Preprocessing partition memmap v2 (4ch): nofilter ==="
python preprocess_partition_memmap.py \
  --decoded-root \
    "${DATA}/decoded_BVI_frames_YUV420_10bit_nofilter" \
    "${DATA}/decoded_DIV2K_YUV420_10bit_nofilter" \
  --original-root \
    "${DATA}/splitted_DVI_BVC" \
    "${DATA}/splitted_DIV2K" \
  --output-dir "${DATA}/partition_memmap_cache_v2_nofilter"

echo ""

echo "=== Preprocessing partition memmap v2 (4ch): dbonly ==="
python preprocess_partition_memmap.py \
  --decoded-root \
    "${DATA}/decoded_BVI_frames_YUV420_10bit_dbonly" \
    "${DATA}/decoded_DIV2K_YUV420_10bit_dbonly" \
  --original-root \
    "${DATA}/splitted_DVI_BVC" \
    "${DATA}/splitted_DIV2K" \
  --output-dir "${DATA}/partition_memmap_cache_v2_dbonly"

echo ""
echo "=== All preprocessing done ==="
