#!/bin/bash
#SBATCH --job-name=fix_partition
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%j_fix_partition.out
#SBATCH --error=slurm_logs/%j_fix_partition.err

set -e
export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data

echo "=== Step 1: Regenerate partition maps with correct dimensions ==="
echo "Start: $(date)"

for variant in "" "_nofilter" "_dbonly"; do
    for dataset in "decoded_BVI_frames_YUV420_10bit${variant}" \
                   "decoded_DIV2K_YUV420_10bit${variant}" \
                   "decoded_CTC_YUV420_10bit${variant}"; do
        ROOT="${DATA}/${dataset}"
        echo ""
        echo "--- Processing: $dataset ---"
        python csv_to_partition_maps.py --root "$ROOT" --workers 16 --force
    done
done

echo ""
echo "=== Step 1 verification ==="
for variant in "" "_nofilter" "_dbonly"; do
    for dataset in "decoded_BVI_frames_YUV420_10bit${variant}" \
                   "decoded_DIV2K_YUV420_10bit${variant}" \
                   "decoded_CTC_YUV420_10bit${variant}"; do
        CSV_COUNT=$(find "${DATA}/${dataset}" -name "bitstream_partition.csv" 2>/dev/null | wc -l)
        DEPTH_COUNT=$(find "${DATA}/${dataset}" -name "depth_map.raw" 2>/dev/null | wc -l)
        BOUND_COUNT=$(find "${DATA}/${dataset}" -name "boundary_map.raw" 2>/dev/null | wc -l)
        if [[ "$CSV_COUNT" -eq "$DEPTH_COUNT" && "$CSV_COUNT" -eq "$BOUND_COUNT" ]]; then
            echo "OK: $dataset  $CSV_COUNT"
        else
            echo "MISMATCH: $dataset  csv=$CSV_COUNT depth=$DEPTH_COUNT boundary=$BOUND_COUNT"
        fi
    done
done

echo ""
echo "Step 1 done: $(date)"


for variant in "" "_nofilter" "_dbonly"; do
    DIR="${DATA}/partition_memmap_cache${variant}"
    if [[ -d "$DIR" ]]; then
        echo "Removing old memmap: $DIR"
        rm -rf "$DIR"
    fi
done

echo ""
echo "=== Step 2: Build partition memmaps ==="

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
echo "=== All partition memmaps done: $(date) ==="

echo ""
echo "=== Step 3: Verification and visualization ==="
python verify_partition_data.py --viz-dir viz_partition_verify

echo ""
echo "=== Everything done: $(date) ==="
