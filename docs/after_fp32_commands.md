# After the FP32 HPS pilot: profiling and INT8 commands

Start here after successful `fp32_t1_pilot`, `fp32_t2_pilot`, both 10-image
evaluations and reference/HPS logit comparisons. These are HPS CPU measurements.
The custom FPGA array has not executed these inferences.

Keep the current uninstrumented `build/model_runner` on the board. New profile
builds and output directories below are separate. Commands labelled WSL execute
on the Windows computer; PuTTY commands execute on the ARM board.

## 1. Build the HPS layer profiler — WSL

In the Ubuntu terminal:

```sh
cd /home/tiff/cpu_baseline_tools
source .venv-model/bin/activate
project_root="$PWD"

cmake -S external/ncnn -B build/ncnn-arm-profile \
  -DCMAKE_TOOLCHAIN_FILE="$project_root/configs/cortex_a9.toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$project_root/build/ncnn-arm-profile-install" \
  -DNCNN_SHARED_LIB=OFF -DNCNN_VULKAN=OFF -DNCNN_VFPV4=OFF \
  -DNCNN_RUNTIME_CPU=OFF -DNCNN_OPENMP=ON -DNCNN_BENCHMARK=ON \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF

cmake --build build/ncnn-arm-profile --parallel 2
cmake --install build/ncnn-arm-profile

cmake -S software/hps/model_runner -B build/model-runner-arm-profile \
  -DCMAKE_TOOLCHAIN_FILE="$project_root/configs/cortex_a9.toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS="-static" \
  -DOpenMP_gomp_LIBRARY="$(arm-linux-gnueabihf-gcc -print-file-name=libgomp.a)" \
  -Dncnn_DIR="$project_root/build/ncnn-arm-profile-install/lib/cmake/ncnn"

cmake --build build/model-runner-arm-profile --parallel 2
file build/model-runner-arm-profile/model_runner
cp build/model-runner-arm-profile/model_runner /mnt/c/Users/tiffa/Downloads/model_runner_profile
```

Each command must succeed before continuing. In particular, keep the explicit
static `libgomp.a` setting: it fixes the earlier attempted-static-link-of-libgomp.so
error. `NCNN_BENCHMARK` instruments layer execution; `NCNN_BUILD_BENCHMARK`
controls a different standalone executable and stays OFF.

## 2. Transfer only the new executable

In PuTTY, finish USB use and run:

```sh
umount /mnt/usb
```

Move the flash drive to Windows. Copy `model_runner_profile` from Downloads
onto its root, safely eject it, and return it to the HPS USB HOST port.

In PuTTY (verify `/dev/sda1` still denotes that USB partition):

```sh
mount -o ro /dev/sda1 /mnt/usb
cd ~/vit_model_fp32
cp /mnt/usb/model_runner_profile build/model_runner_profile
chmod +x build/model_runner_profile
./build/model_runner_profile
```

The last command should display usage. Keep the original runner in place.
Read-only mounting is sufficient for transferring files from the drive.

## 3. Measure matched profiling/control runs — PuTTY

Run the uninstrumented control first (about 40 seconds at the measured speed):

```sh
./build/model_runner bench model/tinyvit_fp32.ncnn.param model/tinyvit_fp32.ncnn.bin inputs/inputs.tsv results/fp32_t2_profile_control 2 3 5 in0 out0 0 fp32 > results/fp32_t2_profile_control.stdout 2> results/fp32_t2_profile_control.stderr
```

Then the same five calls using the profile build:

```sh
./build/model_runner_profile bench model/tinyvit_fp32.ncnn.param model/tinyvit_fp32.ncnn.bin inputs/inputs.tsv results/fp32_t2_profile 2 3 5 in0 out0 0 fp32 > results/fp32_t2_profile.stdout 2> results/fp32_t2_profile.layers.log
```

Layer output is redirected to a board file to avoid sending every layer over
the slow serial terminal during inference. Examine:

```sh
head -n 10 results/fp32_t2_profile.layers.log
cat results/fp32_t2_profile/run.json
```

The log should contain operator types, layer names and times ending in `ms`.
This run includes nine passes: one first inference, three warmups, five measured
calls. The parser uses the first executed layer to delimit each inference and
removes the first four complete passes. Repeated layer invocations (e.g. Split)
are summed within each image; call counts must be stable across measured passes.
Its checks detect missing inference starts and inconsistent measured call counts.

Repeat with thread argument `1` and new `fp32_t1_profile_control` /
`fp32_t1_profile` paths if a one-thread operator comparison is required.
Use the uninstrumented runner for headline latency and throughput.

## 4. Return results to WSL and summarize

Archive the updated board results in PuTTY:

```sh
tar -czf /tmp/hps_profile_results.tar.gz results
```

Use the established PuTTY logging transfer: select a new Windows log file,
print `HPS_OCT_BEGIN`, execute `od -v -b /tmp/hps_profile_results.tar.gz`, then
print `HPS_OCT_END`. Decode using the previously verified Python procedure,
changing its log path and output folder to a new location, for example
`results/raw/hps_profile`. Never decode over the previous pilot directory.

After extraction, run in WSL:

```sh
python scripts/summarize_ncnn_profile.py --log results/raw/hps_profile/results/fp32_t2_profile.layers.log --run results/raw/hps_profile/results/fp32_t2_profile/run.json --control results/raw/hps_profile/results/fp32_t2_profile_control/run.json --out results/reports/fp32_t2_operator_profile
```

Outputs:

- `layers.csv`: each deployed layer, calls/image, shapes when logged, average
  execution time and shares.
- `operator_groups.csv`: aggregate convolution, Gemm, normalization, softmax,
  activation, layout and other costs by runtime type.
- `summary.json`: logged-layer sum, direct profiled wall time, unaccounted time
  and observed overhead versus the matching control.

These are deployed-layer execution timings. A fused layer's time includes its
internal work. Do not call them primitive operator self-times. Printed timing
resolution is 0.01 ms. Shares use the same instrumented-run denominator; do not
divide them by an unrelated uninstrumented model time.

Map layers back to source blocks using `tinyvit_fp32.ncnn.param`, the generated
`tinyvit_fp32_pnnx.py` / `tinyvit_fp32.pnnx.param`, and captured source shapes.
The export unrolls windows and may duplicate weights. Verify mapping rather
than assuming numbered `gemm_N` names directly encode model stages. Stage
aggregation remains a separate reviewed mapping task.

## 5. Complete architecture inspection and exact-size fields — WSL

If `results/raw/architecture_smoke10` already exists, review it; do not rerun
merely to create another copy. Otherwise:

```sh
python scripts/model_baseline.py inspect --source external/Cream/TinyViT --checkpoint models/checkpoints/tinyvit21m.pth --manifest data/manifests/smoke10.json --split dev --limit 10 --threads 1 --capture-linear 'layers\.[123]\.blocks\.0\.(attn\.(qkv|proj)|mlp\.fc[12])' --out results/raw/architecture_capture10
```

The saved operator shapes, first captured Linear inputs/weights, and boundary
tensors support Tables A/F. Existing inspection without `--capture-linear`
does not contain those real Linear tensors; use the new capture directory if
they are needed. Shapes are source-layout shapes, distinct from ncnn packed
tensor layouts.

Get exact deployment storage without another board run:

```sh
python - <<'PY'
from pathlib import Path
files = [Path('models/exports/smoke10/tinyvit_fp32.ncnn.param'),
         Path('models/exports/smoke10/tinyvit_fp32.ncnn.bin')]
for p in files:
    print(p.name, p.stat().st_size, 'bytes')
print('Model total MiB:', sum(p.stat().st_size for p in files)/2**20)
PY
```

## 6. Prepare disjoint calibration and compare Q1–Q4 — WSL

For a small workflow check using ten EXTRA images:

```sh
python scripts/download_calibration_pilot.py
```

This preserves the original ten development images and adds ten different
repository-labelled calibration examples into `data/manifests/quant_pilot.json`.
It verifies file readability and rejects identical content across splits.
Source: https://github.com/nihui/imagenet-sample-images . These extra examples
are fish/bird-heavy and nonrepresentative: use them only to check the workflow.
Replace/expand calibration before claiming a final accuracy tradeoff.

Define this helper once in the current WSL bash terminal:

```sh
quant_step() {
  action=$1
  shift
  python scripts/model_baseline.py "$action" --source external/Cream/TinyViT --checkpoint models/checkpoints/tinyvit21m.pth --manifest data/manifests/quant_pilot.json "$@"
}
```

Then:

```sh
quant_step calibrate --split calib --limit 10 --threads 1 --out results/raw/calibration_pilot10
quant_step evaluate --split dev --limit 10 --threads 1 --out results/raw/F0_quant_pilot10
for variant in Q1 Q2 Q3 Q4; do
  quant_step evaluate --split dev --limit 10 --threads 1 --variant "$variant" --calibration results/raw/calibration_pilot10/calibration.json --out "results/raw/${variant}_pilot10" || break
  python scripts/compare_model_runs.py --reference results/raw/F0_quant_pilot10 --candidate "results/raw/${variant}_pilot10" --out "results/raw/${variant}_pilot10_comparison.json" || break
done
```

Inspect every command's status; stop on a failed variant. Q3/Q4 may reject
unrepresentable q16 multipliers, naming the layer. Investigate on development/
calibration data and record a revised scaling policy as a NEW configuration.
Do not silently clamp the multiplier or tune using final evaluation data.

Q1–Q4 are numerical Linear-only simulations; bias and remaining operators stay
FP32. Report their accuracy, logit error and clipping diagnostics. Their timing
is not native INT8 performance. Local tensor errors use the same propagated
input; comparing full-path stage errors requires matching boundary captures.

## 7. Build a native INT8 candidate — WSL

The native ncnn configuration is R1, not an automatic implementation of Q1–Q4.
Build the conversion tools for the WSL computer (NO ARM toolchain option):

```sh
sudo apt install -y build-essential
cmake -S external/ncnn -B build/ncnn-tools -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF
cmake --build build/ncnn-tools --target ncnn2table ncnn2int8 --parallel 2
quant_step export --split calib --limit 10 --threads 1 --out models/exports/calibration_pilot10
build/ncnn-tools/tools/quantize/ncnn2table models/exports/smoke10/tinyvit_fp32.ncnn.param models/exports/smoke10/tinyvit_fp32.ncnn.bin models/exports/calibration_pilot10/npy_list.txt models/exports/native_pilot10.table 'shape=[224,224,3]' thread=1 method=kl type=1 > models/exports/native_pilot10_calibration.log 2>&1
build/ncnn-tools/tools/quantize/ncnn2int8 models/exports/smoke10/tinyvit_fp32.ncnn.param models/exports/smoke10/tinyvit_fp32.ncnn.bin models/exports/tinyvit_int8_pilot10.param models/exports/tinyvit_int8_pilot10.bin models/exports/native_pilot10.table > models/exports/native_pilot10_conversion.log 2>&1
```

Inspect logs and quantized graph/weight scales. Record converted operators,
static/dynamic scale policies and FP32 paths; INT8 enablement alone does not
prove coverage. The input NPY files already contain the reference preprocessing.

Transfer the two INT8 model files via USB, preserving those names under the
board's `~/vit_model_fp32/model/`. Reuse the verified inputs and existing
uninstrumented runner. Record the new file hashes and exact size.

## 8. Measure full-model native INT8 on HPS — PuTTY

First a one-image check:

```sh
cd ~/vit_model_fp32
./build/model_runner eval model/tinyvit_int8_pilot10.param model/tinyvit_int8_pilot10.bin inputs/one_image.tsv results/R1_one_image 1 0 1 in0 out0 0 runtime-int8
cat results/R1_one_image/run.json
```

If loading/inference succeeds, evaluate all ten images:

```sh
./build/model_runner eval model/tinyvit_int8_pilot10.param model/tinyvit_int8_pilot10.bin inputs/inputs.tsv results/R1_check10_t1 1 0 1 in0 out0 0 runtime-int8
./build/model_runner eval model/tinyvit_int8_pilot10.param model/tinyvit_int8_pilot10.bin inputs/inputs.tsv results/R1_check10_t2 2 0 1 in0 out0 0 runtime-int8
```

Recover results to a new WSL directory and use `compare_model_runs.py --native`
against `models/exports/smoke10`. INT8 logits are expected to differ from FP32;
use quality/error metrics, not the FP32 allclose tolerance as its acceptance gate.
Preserve failures and investigate large quality changes before long benchmarks.

Then measure the COMPLETE native quantized CPU graph:

```sh
./build/model_runner bench model/tinyvit_int8_pilot10.param model/tinyvit_int8_pilot10.bin inputs/inputs.tsv results/R1_t1_pilot 1 3 20 in0 out0 0 runtime-int8
./build/model_runner bench model/tinyvit_int8_pilot10.param model/tinyvit_int8_pilot10.bin inputs/inputs.tsv results/R1_t2_pilot 2 3 20 in0 out0 0 runtime-int8
```

These times include required runtime quantize/dequantize, layout operations and
remaining FP32 operators along the graph. Timing starts with preprocessed FP32
input available and ends with usable logits. This is full-model INT8/mixed-precision
CPU inference latency, not camera/application latency.

Use the profile runner with the INT8 files and `runtime-int8` to profile R1,
again with a matched control and redirected layer log. Keep graph/precision
labels explicit; do not describe it as RTL-compatible without conformance tests.

## 9. Finish real kernels, final distributions and resources

After actual Linear capture and matching calibration, generate real GEMM cases:

```sh
python scripts/export_model_kernel_cases.py --capture results/raw/architecture_capture10 --calibration results/raw/calibration_pilot10/calibration.json --out data/layers_real_pilot --max-columns 49
```

Transfer cases and benchmark on HPS using the existing C program/suite runner.
These remain GEMM-only, bias-excluded cases; 49 limits token columns for a pilot.
Expand to complete representative shapes and an optimized SAME-contract CPU
kernel. Measure packing separately where the final implementation needs it.

For final timing, freeze graph/calibration after a broader quality evaluation.
Use at least 10 warmups/100 samples when affordable, and independent repeated
processes (e.g. three sessions). For example, AFTER the final INT8 model has been
selected and named `model/tinyvit_int8.param/.bin`, run on HPS:

```sh
PRECISION=runtime-int8 RUNS=100 WARMUPS=10 sh scripts/run_hps_model_metrics.sh model/tinyvit_int8.param model/tinyvit_int8.bin inputs/inputs.tsv results/R1_final_t2 2
```

The wrapper creates an accuracy run and three timing sessions. Run a matching
final FP32 configuration if required by the report; existing 20-sample results
stay labelled pilots. Use measured pilot duration to budget long runs.

A long final timing session can also test stability: record CPU frequency,
available temperature, and repeated live-process VmRSS/VmHWM snapshots during
it. Compare early/late latency and report failures. The existing runner's loaded/
end RSS are snapshots, not a full memory-growth trace. Record clock/thermal
fields as unavailable if the BSP does not expose them. CPU utilization/page
faults can be collected with a separate GNU time/perf diagnostic when supported.

## 10. What still needs application/hardware integration

For actual HPS application latency, insert monotonic timers into the deployed
image application around decoded-image-ready -> exact preprocessing -> inference
-> predicted-label-ready. Time preprocessing and postprocessing separately too.
Directly time the total; do not add stage medians/p95s. The tensor runner does
not contain HPS image decoding, camera capture, overlays or the actual application
preprocessor, so those timings cannot be generated merely by changing its flags.

For the FPGA comparison, implement and test the driver/tiler and all required
precision boundaries. Match the CPU graph/checkpoint/arithmetic, and include
packing, transfers, cache handling, control and output handling in direct hybrid
wall time. Standard native ncnn INT8 may need custom kernels to match Q3/Q4.

Complete Tables A–F with measured/derived fields, scoped accuracy, unsupported
operators and remaining limitations. Preserve raw samples, layer logs, manifests,
model/binary hashes, calibration/image hashes and exact deployment sizes. Report
power/energy only if measured with a suitable instrument.
