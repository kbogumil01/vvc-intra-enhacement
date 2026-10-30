#!/bin/bash
#SBATCH --job-name=ablation_resume
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%A_%a_ablation_resume.out
#SBATCH --error=slurm_logs/%A_%a_ablation_resume.err
#SBATCH --array=0-11

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data
CKPT_DIR="lightning_logs/lightning_logs"

VARIANTS=(dbonly dbonly dbonly dbonly filter filter filter filter nofilter nofilter nofilter nofilter)
CHANNELS=(0 1 2 3 0 1 2 3 0 1 2 3)
VERSIONS=(2615352 2615398 2615810 2615303 2615976 2616160 2616169 2615304 2616782 2616793 2616814 2615305)
NAMES=("luma_depth" "luma_boundary" "chroma_depth" "chroma_boundary")

IDX=${SLURM_ARRAY_TASK_ID}
VARIANT=${VARIANTS[$IDX]}
CHANNEL=${CHANNELS[$IDX]}
VERSION=${VERSIONS[$IDX]}
CH_NAME=${NAMES[$CHANNEL]}

BEST_CKPT=$(ls ${CKPT_DIR}/version_${VERSION}/checkpoints/dense-best-*.ckpt 2>/dev/null | tail -1)
if [[ -n "$BEST_CKPT" ]]; then
    EPOCH=$(echo "$BEST_CKPT" | grep -oP 'epoch=\K\d+')
    if [[ $EPOCH -ge 175 ]]; then
        echo "SKIP: v${VERSION} (${VARIANT} + ${CH_NAME}) already at epoch ${EPOCH} >= 175"
        exit 0
    fi
fi

RESUME="${CKPT_DIR}/version_${VERSION}/checkpoints/last.ckpt"
if [[ ! -f "$RESUME" ]]; then
    echo "ERROR: No last.ckpt for version ${VERSION}"
    exit 1
fi

case $VARIANT in
    filter)
        DECODED="${DATA}/decoded_BVI_frames_YUV420_10bit"
        EXTRA_DECODED="${DATA}/decoded_DIV2K_YUV420_10bit"
        MEMMAP="${DATA}/memmap_cache"
        PART_MEMMAP="${DATA}/partition_memmap_cache_v2"
        ;;
    nofilter)
        DECODED="${DATA}/decoded_BVI_frames_YUV420_10bit_nofilter"
        EXTRA_DECODED="${DATA}/decoded_DIV2K_YUV420_10bit_nofilter"
        MEMMAP="${DATA}/memmap_cache_nofilter"
        PART_MEMMAP="${DATA}/partition_memmap_cache_v2_nofilter"
        ;;
    dbonly)
        DECODED="${DATA}/decoded_BVI_frames_YUV420_10bit_dbonly"
        EXTRA_DECODED="${DATA}/decoded_DIV2K_YUV420_10bit_dbonly"
        MEMMAP="${DATA}/memmap_cache_dbonly"
        PART_MEMMAP="${DATA}/partition_memmap_cache_v2_dbonly"
        ;;
esac

echo "=== RESUME: ${VARIANT} + ${CH_NAME} (ch${CHANNEL}), v${VERSION}, from ${RESUME} ==="

python train.py \
  --model ppff_base \
  --decoded-root "${DECODED}" \
  --original-root "${DATA}/splitted_DVI_BVC" \
  --extra-decoded-root "${EXTRA_DECODED}" \
  --extra-original-root "${DATA}/splitted_DIV2K" \
  --memmap-dir "${MEMMAP}" \
  --partition-memmap-dir "${PART_MEMMAP}" \
  --partition-channels ${CHANNEL} \
  --patch-size 256 \
  --patches-per-frame 2 \
  --batch-size 12 \
  --accumulate-grad-batches 3 \
  --lr 1e-4 \
  --max-epochs 200 \
  --early-stop-patience 20 \
  --precision 16-mixed \
  --num-workers 14 \
  --resume-path "${RESUME}"
