#!/bin/bash
#SBATCH --job-name=eval_ablation
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data
CKPT_DIR="lightning_logs/lightning_logs"
CHANNEL_NAMES=("luma_depth" "luma_boundary" "chroma_depth" "chroma_boundary")

DBONLY_VERSIONS=(2615352 2615398 2619480 2619096)
FILTER_VERSIONS=(2619344 2619345 2619355 2619478)
NOFILTER_VERSIONS=(2619253 2619329 2619334 2619337)

evaluate_model() {
    local VERSION=$1
    local VARIANT=$2
    local CHANNEL=$3
    local CH_NAME=${CHANNEL_NAMES[$CHANNEL]}

    if [[ -z "$VERSION" ]]; then
        echo "SKIP: ${VARIANT} + ${CH_NAME} (no version specified)"
        return
    fi

    local CKPT_PATTERN="${CKPT_DIR}/version_${VERSION}/checkpoints/dense-best-*.ckpt"
    local CKPT=$(ls $CKPT_PATTERN 2>/dev/null | head -1)
    if [[ -z "$CKPT" ]]; then
        echo "ERROR: No checkpoint found for version ${VERSION}"
        return
    fi

    local DECODED_ROOT
    case $VARIANT in
        filter)   DECODED_ROOT="${DATA}/decoded_CTC_YUV420_10bit" ;;
        nofilter) DECODED_ROOT="${DATA}/decoded_CTC_YUV420_10bit_nofilter" ;;
        dbonly)   DECODED_ROOT="${DATA}/decoded_CTC_YUV420_10bit_dbonly" ;;
    esac

    local OUTPUT="bd_results_ctc_ablation_${VARIANT}_ch${CHANNEL}_${CH_NAME}.csv"
    echo "=== Evaluating v${VERSION} (${VARIANT} + ${CH_NAME}, channel ${CHANNEL}) ==="
    python evaluate_bd.py \
      --model ppff_base \
      --ckpt-path "$CKPT" \
      --decoded-root "$DECODED_ROOT" \
      --original-root "${DATA}/splitted_CTC" \
      --no-split \
      --partition-maps \
      --partition-channels ${CHANNEL} \
      --output "$OUTPUT"
    echo ""
}

for ch in 0 1 2 3; do
    evaluate_model "${DBONLY_VERSIONS[$ch]}" "dbonly" "$ch"
done

for ch in 0 1 2 3; do
    evaluate_model "${FILTER_VERSIONS[$ch]}" "filter" "$ch"
done

for ch in 0 1 2 3; do
    evaluate_model "${NOFILTER_VERSIONS[$ch]}" "nofilter" "$ch"
done

echo "=== All ablation evaluations done ==="
