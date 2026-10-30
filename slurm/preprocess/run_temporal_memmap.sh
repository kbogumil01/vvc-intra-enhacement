#!/bin/bash
#SBATCH --job-name=temporal_memmap
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=08:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --output=temporal_memmap_%j.log
#SBATCH --error=temporal_memmap_%j.err

cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

DATA=/net/tscratch/people/plgkarolbogumil/data
OUTPUT=$DATA/temporal_memmap_nofilter

python preprocess_temporal_memmap.py \
    --bvi-decoded $DATA/decoded_BVI_frames_YUV420_10bit_nofilter \
    --bvi-original $DATA/splitted_DVI_BVC \
    --bvi-neighbors-decoded $DATA/dograna_data/decoded_BVI_neighbors_nofilter \
    --xiph-decoded $DATA/dograna_data/decoded_XIPH_nofilter \
    --xiph-original $DATA/dograna_data/splitted_XIPH \
    --output-dir $OUTPUT \
    --no-4k

echo "=== Memmap completed ==="
ls -lh $OUTPUT/
