#!/bin/sh
# Run in WSL with the same ARM toolchain/ncnn installation as the working runner.
set -eu
PACKAGE=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
PROJECT=${1:-/home/tiff/cpu_baseline_tools}
NCNN_DIR="$PROJECT/build/ncnn-arm-install/lib/cmake/ncnn"
test -f "$NCNN_DIR/ncnnConfig.cmake" || { echo "Missing ARM ncnn installation: $NCNN_DIR" >&2; exit 1; }
command -v cmake
command -v arm-linux-gnueabihf-g++
cmake -S "$PACKAGE/software/hps/model_runner" -B "$PROJECT/build/hps-local-camera-arm" \
  -DCMAKE_TOOLCHAIN_FILE="$PACKAGE/configs/cortex_a9.toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release -Dncnn_DIR="$NCNN_DIR" \
  -DCMAKE_EXE_LINKER_FLAGS=-static
cmake --build "$PROJECT/build/hps-local-camera-arm" --target camera_runner --parallel 2
mkdir -p "$PACKAGE/bin"
cp "$PROJECT/build/hps-local-camera-arm/camera_runner" "$PACKAGE/bin/camera_runner"
file "$PACKAGE/bin/camera_runner"
echo "Built $PACKAGE/bin/camera_runner; transfer the package to HPS next."
