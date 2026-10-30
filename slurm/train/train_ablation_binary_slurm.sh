#!/bin/bash
#SBATCH --job-name=ablation_binary
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=800G
#SBATCH --time=40:00:00
#SBATCH --output=slurm_logs/%j_%a_%x.out
#SBATCH --error=slurm_logs/%j_%a_%x.err
#SBATCH --array=1-2

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data
CKPT_DIR="lightning_logs/lightning_logs"

VARIANTS=("filter" "nofilter" "dbonly")
SUFFIXES=("" "_nofilter" "_dbonly")
MEMMAP_DIRS=("memmap_cache" "memmap_cache_nofilter" "memmap_cache_dbonly")
PART_DIRS=("partition_memmap_cache_v2" "partition_memmap_cache_v2_nofilter" "partition_memmap_cache_v2_dbonly")
DECODED_CTC_DIRS=("decoded_CTC_YUV420_10bit" "decoded_CTC_YUV420_10bit_nofilter" "decoded_CTC_YUV420_10bit_dbonly")

IDX=${SLURM_ARRAY_TASK_ID}
VARIANT=${VARIANTS[$IDX]}
SUFFIX=${SUFFIXES[$IDX]}
MEMMAP=${DATA}/${MEMMAP_DIRS[$IDX]}
PART_MEMMAP=${DATA}/${PART_DIRS[$IDX]}
DECODED_CTC=${DATA}/${DECODED_CTC_DIRS[$IDX]}

echo "=============================================="
echo "=== PHASE 1: TRAINING ==="
echo "=== Ablation: ${VARIANT} + binary maps (ch1+ch3, 6ch) ==="
echo "=============================================="

python train.py \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_BVI_frames_YUV420_10bit${SUFFIX}" \
  --original-root "${DATA}/splitted_DVI_BVC" \
  --extra-decoded-root "${DATA}/decoded_DIV2K_YUV420_10bit${SUFFIX}" \
  --extra-original-root "${DATA}/splitted_DIV2K" \
  --memmap-dir "${MEMMAP}" \
  --partition-memmap-dir "${PART_MEMMAP}" \
  --partition-channels 1 3 \
  --patch-size 256 \
  --patches-per-frame 2 \
  --batch-size 12 \
  --accumulate-grad-batches 3 \
  --lr 1e-4 \
  --max-epochs 200 \
  --early-stop-patience 20 \
  --precision 16-mixed \
  --num-workers 14

echo ""
echo "=============================================="
echo "=== PHASE 2: EVALUATION on CTC ==="
echo "=============================================="

VERSION_NUM=$SLURM_JOB_ID
LATEST_VERSION="${CKPT_DIR}/version_${VERSION_NUM}"

if [[ ! -d "$LATEST_VERSION" ]]; then
    echo "WARNING: version_${VERSION_NUM} not found, searching latest..."
    LATEST_VERSION=$(ls -td ${CKPT_DIR}/version_* | while read d; do
        if ls "${d}/checkpoints/dense-best-"*".ckpt" &>/dev/null; then echo "$d"; break; fi
    done)
    VERSION_NUM=$(basename $LATEST_VERSION | sed 's/version_//')
fi

CKPT=$(ls ${LATEST_VERSION}/checkpoints/dense-best-*.ckpt 2>/dev/null | head -1)

if [[ -z "$CKPT" ]]; then
    echo "ERROR: No best checkpoint found in ${LATEST_VERSION}/checkpoints/"
    echo "Available files:"
    ls -la ${LATEST_VERSION}/checkpoints/ 2>/dev/null
    exit 1
fi

echo "Using checkpoint: $CKPT"
echo "Version: $VERSION_NUM"
echo "Variant: $VARIANT"

OUTPUT="bd_results_ctc_ablation_${VARIANT}_ch1_ch3_binary.csv"

python evaluate_bd.py \
  --model ppff_base \
  --ckpt-path "$CKPT" \
  --decoded-root "$DECODED_CTC" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --partition-maps \
  --partition-channels 1 3 \
  --output "$OUTPUT"

echo ""
echo "=============================================="
echo "=== PHASE 3: REPORT ==="
echo "=============================================="
echo "Output: $OUTPUT"
echo "Version: $VERSION_NUM"

BASELINE_Y=(-2.161 -6.496 -5.327)
BASELINE_U=(-11.576 -16.628 -13.866)
BASELINE_V=(-12.976 -18.384 -15.007)
FULL8_Y=(-2.403 -6.584 -5.524)
FULL8_U=(-12.203 -16.233 -14.477)
FULL8_V=(-13.470 -18.249 -15.896)

AVG_LINE=$(tail -1 "$OUTPUT")
BDR_Y=$(echo "$AVG_LINE" | cut -d, -f2)
BDR_U=$(echo "$AVG_LINE" | cut -d, -f3)
BDR_V=$(echo "$AVG_LINE" | cut -d, -f4)
BDR_SSIM_Y=$(echo "$AVG_LINE" | cut -d, -f8)

echo ""
echo "=== ABSOLUTE BD-RATE [%] (lower = better) ==="
printf "%-35s %10s %10s %10s %12s\n" "Model" "BDR-Y" "BDR-U" "BDR-V" "BDR-SSIM-Y"
echo "--------------------------------------------------------------------------"
printf "%-35s %10.3f %10.3f %10.3f %12s\n" "Baseline 4ch" "${BASELINE_Y[$IDX]}" "${BASELINE_U[$IDX]}" "${BASELINE_V[$IDX]}" "n/a"
printf "%-35s %10.3f %10.3f %10.3f %12.3f\n" "Binary maps ch1+ch3 (6ch)" "$BDR_Y" "$BDR_U" "$BDR_V" "$BDR_SSIM_Y"
printf "%-35s %10.3f %10.3f %10.3f %12s\n" "8ch (all 4 maps)" "${FULL8_Y[$IDX]}" "${FULL8_U[$IDX]}" "${FULL8_V[$IDX]}" "n/a"

echo ""
echo "=== RELATIVE vs BASELINE 4ch [p.p.] (negative = improvement) ==="
DELTA_Y=$(echo "$BDR_Y - (${BASELINE_Y[$IDX]})" | bc -l)
DELTA_U=$(echo "$BDR_U - (${BASELINE_U[$IDX]})" | bc -l)
DELTA_V=$(echo "$BDR_V - (${BASELINE_V[$IDX]})" | bc -l)
DELTA8_Y=$(echo "${FULL8_Y[$IDX]} - (${BASELINE_Y[$IDX]})" | bc -l)
DELTA8_U=$(echo "${FULL8_U[$IDX]} - (${BASELINE_U[$IDX]})" | bc -l)
DELTA8_V=$(echo "${FULL8_V[$IDX]} - (${BASELINE_V[$IDX]})" | bc -l)

printf "%-35s %10s %10s %10s\n" "Model" "ΔY [p.p.]" "ΔU [p.p.]" "ΔV [p.p.]"
echo "--------------------------------------------------------------------"
printf "%-35s %10.3f %10.3f %10.3f\n" "Binary maps ch1+ch3 (6ch)" "$DELTA_Y" "$DELTA_U" "$DELTA_V"
printf "%-35s %10.3f %10.3f %10.3f\n" "8ch (all 4 maps)" "$DELTA8_Y" "$DELTA8_U" "$DELTA8_V"

echo ""
echo "=== RELATIVE vs 8CH [p.p.] (positive = worse than 8ch) ==="
REL8_Y=$(echo "$BDR_Y - (${FULL8_Y[$IDX]})" | bc -l)
REL8_U=$(echo "$BDR_U - (${FULL8_U[$IDX]})" | bc -l)
REL8_V=$(echo "$BDR_V - (${FULL8_V[$IDX]})" | bc -l)
printf "%-35s %10.3f %10.3f %10.3f\n" "Binary ch1+ch3 vs 8ch" "$REL8_Y" "$REL8_U" "$REL8_V"

echo ""
echo "=================================================================="
echo "=== DONE: ${VARIANT} + binary maps (ch1+ch3) v${VERSION_NUM} ==="
echo "=================================================================="
