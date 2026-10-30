#!/bin/bash
#SBATCH --job-name=resume_ablation
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j_%a_resume_ablation.out
#SBATCH --error=slurm_logs/%j_%a_resume_ablation.err
#SBATCH --array=0-9

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data
CKPT_DIR="lightning_logs/lightning_logs"
CHANNEL_NAMES=("luma_depth" "luma_boundary" "chroma_depth" "chroma_boundary")

VARIANTS=(  nofilter nofilter nofilter nofilter filter filter filter filter dbonly dbonly)
CHANNELS=(  0        1        2        3        0      1      2      3      2      3)
VERSIONS=(  2616782  2616793  2616814  2615305  2615976 2616160 2616169 2615304 2615810 2615303)

IDX=${SLURM_ARRAY_TASK_ID}
VARIANT=${VARIANTS[$IDX]}
CHANNEL=${CHANNELS[$IDX]}
VERSION=${VERSIONS[$IDX]}
CH_NAME=${CHANNEL_NAMES[$CHANNEL]}

RESUME_PATH="${CKPT_DIR}/version_${VERSION}/checkpoints/last.ckpt"
if [[ ! -f "$RESUME_PATH" ]]; then
    echo "ERROR: No last.ckpt at $RESUME_PATH"
    exit 1
fi

echo "=== Resume: ${VARIANT} + ${CH_NAME} (ch${CHANNEL}), from version_${VERSION} ==="
echo "Resume checkpoint: $RESUME_PATH"

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

python train.py \
  --model ppff_base \
  --decoded-root "$DECODED" \
  --original-root "${DATA}/splitted_DVI_BVC" \
  --extra-decoded-root "$EXTRA_DECODED" \
  --extra-original-root "${DATA}/splitted_DIV2K" \
  --memmap-dir "$MEMMAP" \
  --partition-memmap-dir "$PART_MEMMAP" \
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
  --resume-path "$RESUME_PATH"
