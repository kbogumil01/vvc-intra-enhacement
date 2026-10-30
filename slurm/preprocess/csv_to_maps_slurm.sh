#!/bin/bash
#SBATCH --job-name=csv_to_maps
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%j_csv_to_maps.out
#SBATCH --error=slurm_logs/%j_csv_to_maps.err

cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA="/net/tscratch/people/plgkarolbogumil/data"

echo "=== CSV to partition maps: $(date) ==="

for variant in "" "_nofilter" "_dbonly"; do
    for dataset in "decoded_BVI_frames_YUV420_10bit${variant}" \
                   "decoded_DIV2K_YUV420_10bit${variant}" \
                   "decoded_CTC_YUV420_10bit${variant}"; do
        ROOT="${DATA}/${dataset}"
        echo ""
        echo "--- Processing: $dataset ---"
        python csv_to_partition_maps.py --root "$ROOT" --workers 16
    done
done

echo ""
echo "=== Verification ==="
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
echo "=== Done: $(date) ==="
