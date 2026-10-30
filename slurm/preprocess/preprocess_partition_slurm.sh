#!/bin/bash
#SBATCH --job-name=preproc_partition
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_preproc_partition.out
#SBATCH --error=slurm_logs/%j_preproc_partition.err

set -e

export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data

echo "=== Preprocessing partition memmap: filter ==="
python preprocess_partition_memmap.py \
  --decoded-root \
    "${DATA}/decoded_BVI_frames_YUV420_10bit" \
    "${DATA}/decoded_DIV2K_YUV420_10bit" \
  --original-root \
    "${DATA}/splitted_DVI_BVC" \
    "${DATA}/splitted_DIV2K" \
  --output-dir "${DATA}/partition_memmap_cache"

echo ""

echo "=== Preprocessing partition memmap: nofilter ==="
python preprocess_partition_memmap.py \
  --decoded-root \
    "${DATA}/decoded_BVI_frames_YUV420_10bit_nofilter" \
    "${DATA}/decoded_DIV2K_YUV420_10bit_nofilter" \
  --original-root \
    "${DATA}/splitted_DVI_BVC" \
    "${DATA}/splitted_DIV2K" \
  --output-dir "${DATA}/partition_memmap_cache_nofilter"

echo ""

echo "=== Preprocessing partition memmap: dbonly ==="
python preprocess_partition_memmap.py \
  --decoded-root \
    "${DATA}/decoded_BVI_frames_YUV420_10bit_dbonly" \
    "${DATA}/decoded_DIV2K_YUV420_10bit_dbonly" \
  --original-root \
    "${DATA}/splitted_DVI_BVC" \
    "${DATA}/splitted_DIV2K" \
  --output-dir "${DATA}/partition_memmap_cache_dbonly"

echo ""
echo "=== All partition memmaps done ==="
for d in partition_memmap_cache partition_memmap_cache_nofilter partition_memmap_cache_dbonly; do
    echo "--- ${d} ---"
    ls -lh "${DATA}/${d}/"
done
