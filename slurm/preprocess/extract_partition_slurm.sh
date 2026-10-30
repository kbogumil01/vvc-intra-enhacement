#!/bin/bash
#SBATCH --job-name=extract_partition
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j_extract_partition.out
#SBATCH --error=slurm_logs/%j_extract_partition.err

export PYTHONUNBUFFERED=1

cd /net/people/plgrid/plgkarolbogumil/mgr/dense
mkdir -p slurm_logs

VTM_DEC="/net/people/plgrid/plgkarolbogumil/mgr/dense/VVCSoftware_VTM/bin/umake/gcc-11.5/x86_64/release/DecoderApp"
DATA="/net/tscratch/people/plgkarolbogumil/data"
PROGRESS_FILE="/tmp/extract_partition_progress_$$"

decode_one() {
    local bs="$1"
    local outcsv="${bs%.bin}_partition.csv"

    if [[ -f "$outcsv" ]]; then
        echo "1" >> "$PROGRESS_FILE"
        return 0
    fi

    local tmpcsv="${outcsv}.tmp"
    "$VTM_DEC" \
        -b "$bs" \
        -o /dev/null \
        --TraceFile="$tmpcsv" \
        --TraceRule="D_BLOCK_STATISTICS_ALL:poc==0" \
        > /dev/null 2>&1

    if [[ $? -eq 0 && -f "$tmpcsv" ]]; then
        mv "$tmpcsv" "$outcsv"
        echo "1" >> "$PROGRESS_FILE"
    else
        echo "FAIL: $bs" >&2
        rm -f "$tmpcsv"
        return 1
    fi
}
export -f decode_one
export VTM_DEC PROGRESS_FILE

echo "=== Collecting bitstreams: $(date) ==="
TMPLIST=$(mktemp /tmp/bitstream_list.XXXXXX)

for variant in "" "_nofilter" "_dbonly"; do
    for dataset in "decoded_BVI_frames_YUV420_10bit${variant}" \
                   "decoded_DIV2K_YUV420_10bit${variant}" \
                   "decoded_CTC_YUV420_10bit${variant}"; do
        find "${DATA}/${dataset}" -name "bitstream.bin" 2>/dev/null >> "$TMPLIST"
    done
done

TOTAL=$(wc -l < "$TMPLIST")
echo "Total bitstreams: $TOTAL"

> "$PROGRESS_FILE"

echo "=== Processing with 16 parallel workers: $(date) ==="

cat "$TMPLIST" | xargs -P 16 -I {} bash -c 'decode_one "$@"' _ {} &
BGPID=$!

while kill -0 $BGPID 2>/dev/null; do
    sleep 60
    CURRENT=$(wc -l < "$PROGRESS_FILE" 2>/dev/null || echo 0)
    echo "[$(date +%H:%M:%S)] Progress: $CURRENT / $TOTAL"
done
wait $BGPID
EXIT_CODE=$?

echo ""
echo "=== Extraction complete: $(date) ==="
FINAL=$(wc -l < "$PROGRESS_FILE" 2>/dev/null || echo 0)
echo "Processed: $FINAL / $TOTAL"
echo "xargs exit code: $EXIT_CODE"

echo ""
echo "=== Verification per dataset ==="
for variant in "" "_nofilter" "_dbonly"; do
    for dataset in "decoded_BVI_frames_YUV420_10bit${variant}" \
                   "decoded_DIV2K_YUV420_10bit${variant}" \
                   "decoded_CTC_YUV420_10bit${variant}"; do
        BS_COUNT=$(find "${DATA}/${dataset}" -name "bitstream.bin" 2>/dev/null | wc -l)
        CSV_COUNT=$(find "${DATA}/${dataset}" -name "bitstream_partition.csv" 2>/dev/null | wc -l)
        if [[ "$BS_COUNT" -ne "$CSV_COUNT" ]]; then
            echo "MISMATCH: $dataset  bitstreams=$BS_COUNT  csvs=$CSV_COUNT"
        else
            echo "OK: $dataset  $CSV_COUNT / $BS_COUNT"
        fi
    done
done

rm -f "$TMPLIST" "$PROGRESS_FILE"
