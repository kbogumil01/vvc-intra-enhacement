#!/bin/bash
#SBATCH --job-name=build_vtm_trace
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%j_build_vtm_trace.out
#SBATCH --error=slurm_logs/%j_build_vtm_trace.err

set -e

cd /net/people/plgrid/plgkarolbogumil/mgr/dense/VVCSoftware_VTM

module load CMake/3.23.1

echo "=== Cleaning old build ==="
rm -rf build_trace
mkdir build_trace

echo "=== Verifying BLOCK_STATS_AS_CSV ==="
grep "BLOCK_STATS_AS_CSV" source/Lib/CommonLib/TypeDef.h

echo ""
echo "=== CMake configure ==="
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_TRACING=ON \
      -DENABLE_SEARCH_OPENSSL=OFF \
      -S . -B build_trace 2>&1

echo ""
echo "=== Verify ENABLE_TRACING in cache ==="
grep "ENABLE_TRACING" build_trace/CMakeCache.txt

echo ""
echo "=== Building (16 threads) ==="
cmake --build build_trace -j16 2>&1

echo ""
echo "=== Verify binaries ==="
ls -lh build_trace/source/App/DecoderApp/DecoderApp 2>/dev/null || echo "DecoderApp NOT found!"
ls -lh build_trace/source/App/EncoderApp/EncoderApp 2>/dev/null || echo "EncoderApp NOT found!"

echo ""
echo "=== Quick test: decoder help ==="
VTM_BIN=$(find build_trace -name "DecoderApp" -type f 2>/dev/null || find bin -name "DecoderApp" -path "*gcc-*" -newer build_trace/CMakeCache.txt -type f 2>/dev/null)
echo "Found: $VTM_BIN"
$VTM_BIN --help 2>&1 | head -5

echo ""
echo "=== Build complete ==="
