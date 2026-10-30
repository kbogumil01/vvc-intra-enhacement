#!/bin/bash
#SBATCH --job-name=regen_partition_maps
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j_regen_partition_maps.out
#SBATCH --error=slurm_logs/%j_regen_partition_maps.err

set -e
export PYTHONUNBUFFERED=1

cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data

echo "=== Regenerating partition maps (v2: inner borders + chroma + Depth/7 luma + Depth/10 chroma) ==="
echo ""

for variant in "" "_nofilter" "_dbonly"; do
    for root_prefix in decoded_BVI_frames_YUV420_10bit decoded_DIV2K_YUV420_10bit decoded_CTC_YUV420_10bit; do
        ROOT="${DATA}/${root_prefix}${variant}"
        if [ ! -d "$ROOT" ]; then
            echo "SKIP (not found): $ROOT"
            continue
        fi
        echo "=== Processing: ${root_prefix}${variant} ==="
        python csv_to_partition_maps.py --root "$ROOT" --workers 16 --force
        echo ""
    done
done

echo "=== All done ==="
