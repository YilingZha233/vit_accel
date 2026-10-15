# Code to transfer to Windows and run on the Cyclone V HPS

The package contains source code, not a Windows or ARM executable. Windows/WSL
prepares the model and builds software; **HPS Linux on the board executes the
CPU benchmark**. No custom FPGA bitstream is required for CPU-only inference.

The existing matrix benchmark is already measured. Do not repeat it merely to
obtain full-model metrics: it cannot classify images. The commands below add
the full-model measurements. All output directories must be new.

## 1. Files to use

| File | Run where | Purpose |
|---|---|---|
| `scripts/model_baseline.py` | Windows WSL | Architecture checks, FP32 accuracy/timing, profiling, calibration, Q1–Q4 simulation, export |
| `scripts/compare_model_runs.py` | Windows WSL | Accuracy loss, agreement, logit error, paired intervals |
| `scripts/export_model_kernel_cases.py` | Windows WSL | Real-weight/activation cases for the existing C GEMM benchmark |
| `software/hps/model_runner/main.cpp` | Compile for HPS | Native ncnn CPU inference, raw timings, accuracy counts and peak RSS |
| `scripts/run_hps_model_metrics.sh` | HPS Linux | Environment capture, evaluation and three timing sessions |
| `scripts/summarize_model_results.py` | Windows WSL | Combine completed runs into a CSV report |

Prerequisites you provide: a TinyViT-21M-224 **1000-class trained checkpoint**,
labelled images with the checkpoint's class indices, and access to HPS Linux.
These are not included in the code archive. Example names below are paths you
create, not pre-existing trained models or evaluation results.

## 2. Prepare the reference on Windows using WSL Ubuntu

Unzip into your WSL home directory and open a terminal in that folder. Use
Python 3.9–3.12; the tested dependencies are pinned. A Windows venv is not a
WSL venv, so create a new one:

```sh
python3 -m venv .venv-model
. .venv-model/bin/activate
python -m pip install -r configs/model_requirements.txt
python -m pip install pnnx==20260526
python -m unittest discover -s tests -v
mkdir -p external models/checkpoints models/exports data/manifests results/raw
git clone https://github.com/microsoft/Cream.git external/Cream
git -C external/Cream checkout 4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9
```

If `external/Cream` already exists, inspect its revision rather than cloning over
it. Copy your chosen checkpoint to `models/checkpoints/tinyvit21m.pth`. Use the
1000-class checkpoint, not the 21,841-class pretraining checkpoint. Record its
origin and hash:

```sh
sha256sum models/checkpoints/tinyvit21m.pth
python -m pip freeze > results/raw/model_python_environment.txt
```

Create `data/manifests/images.csv` using **real paths and labels**, for example:

```csv
id,path,label,split
calib_0001,/home/USER/images/calibration/image1.jpg,281,calib
dev_0001,/home/USER/images/development/image2.jpg,207,dev
test_0001,/home/USER/images/test/image3.jpg,340,test
```

Those labels illustrate the format only; do not copy them as labels for your
images. `label` is a zero-based checkpoint class index. Keep calib/dev/test
disjoint. Use ~512 calibration images, ~1000 development images, and an untouched
final test set; 10 development images suffice for the first execution check.
For a class-folder dataset, `scripts/make_model_dataset.py --help` provides an
indexer requiring an explicit 1000-class folder-to-checkpoint-index JSON map.

```sh
python scripts/model_baseline.py freeze-data \
  --csv data/manifests/images.csv --out data/manifests/dataset.json
```

This hashes files and rejects duplicate IDs/content. It does not verify that
your supplied labels or class map are correct.

Define a helper in the current WSL bash terminal to avoid repeating paths:

```sh
model_step() {
  action=$1
  shift
  python scripts/model_baseline.py "$action" \
    --source external/Cream/TinyViT \
    --checkpoint models/checkpoints/tinyvit21m.pth \
    --manifest data/manifests/dataset.json "$@"
}
```

## 3. Check architecture, accuracy and desktop profile

```sh
model_step inspect --split dev --limit 10 --out results/raw/architecture \
  --capture-linear 'layers\.[123]\.blocks\.0\.mlp\.fc[12]'
model_step evaluate --split dev --limit 1000 --out results/raw/F0_dev
model_step benchmark --split dev --limit 10 --threads 1 \
  --warmups 10 --runs 100 --out results/raw/F0_desktop_t1
model_step profile --split dev --limit 10 --out results/raw/F0_desktop_profile
```

Outputs: checked stage-boundary tensors, operator shapes, captured Linear data,
`accuracy.json`, `predictions.csv`, `logits.npy`, `latency.csv`, `timing.json`,
`operators.csv`, `stage_profile.csv` and profiler trace, depending on the action.
The profiler's named stage ranges are non-overlapping block/transition groups;
their inclusive times must not be added to child operator times.

These are **desktop** results. The board measurements are step 7.
For desktop application latency (decoded image -> preprocessing -> top-5):

```sh
model_step benchmark --split dev --limit 10 --scope application \
  --warmups 10 --runs 100 --out results/raw/F0_desktop_application
```

This excludes file decoding/camera capture. The HPS runner below accepts already
preprocessed tensors and measures model latency; HPS application latency requires
integrating your actual image preprocessing and overlay pipeline on the board.

## 4. Compare quantization accuracy

```sh
model_step calibrate --split calib --limit 512 --out results/raw/calibration
for variant in Q1 Q2 Q3 Q4; do
  model_step evaluate --split dev --limit 1000 --variant "$variant" \
    --calibration results/raw/calibration/calibration.json \
    --out "results/raw/${variant}_dev"
  python scripts/compare_model_runs.py --reference results/raw/F0_dev \
    --candidate "results/raw/${variant}_dev" \
    --out "results/raw/${variant}_comparison.json"
done
```

Start with `--limit 10` for ALL paired evaluation runs before the 1000-image
run. Use different output names for each run; comparison requires identical IDs.

| Variant | Implemented numerical experiment |
|---|---|
| Q1 | Selected Linear W8A8, per-tensor scales, full sum, ties-to-even output rounding |
| Q2 | Q1 with per-output-channel weight scales |
| Q3 | Full sum with Q0.16 multiplier and floor output rounding |
| Q4 | Same Q0.16 scales, per-16-term clipping and saturating merges |

**These four are numerical simulations, not native INT8 benchmarks.** The default
quantizes the 40 QKV/projection/MLP Linear modules across Stages 2–4. Conv,
attention matrix products, nonlinearities, normalization and classifier remain
FP32. Bias is FP32 and added after dequantization consistently across all four.
This is a declared pilot policy, not a claim that the full graph is INT8 or that
the hardware implements fused INT32 bias. Per-module diagnostics record local
error and clipping; they compare against FP32 on the same propagated input.

Q3/Q4 reject unrepresentable scales rather than silently changing arithmetic.
If rejected, inspect the named layer, revise calibration/output-scale policy and
rerun it as a new documented experiment. Do not modify test data to tune scales.

After selecting/fixing settings, evaluate F0 and chosen variants using `--split
test` and omit `--limit` for the complete frozen test split. Compare those paired
runs again. Small/degenerate bootstrap intervals are not proof of generalization
or a <=1 pp bound; report sample size and uncertainty limitations.

## 5. Export model and input tensors for native inference

```sh
model_step export --split dev --limit 10 --out models/exports/dev10
pnnx models/exports/dev10/tinyvit_fp32.pt 'inputshape=[1,3,224,224]' fp16=0 \
  > models/exports/dev10/conversion.log 2>&1
```

Inspect `conversion.log`: converter exit status alone does not prove success.
The export code rewrites window partitioning to independent batch-1 windows and
checks TorchScript logits against the original model on exported images. This
avoids unsupported batch-axis/high-rank operations seen in direct conversion.
The converted graph may duplicate shared weights and has different scheduling;
report its actual file size and runtime cost. Check the `.ncnn.param` input/output
names (normally `in0`/`out0`) before invoking the runner.

The folder contains model `.ncnn.param/.ncnn.bin`, `inputs.tsv`, input `.bin`
files and original FP32 `.logits.bin` goldens. Do not use a fake/random checkpoint
for actual accuracy results. The package validation used synthetic fixtures only.

## 6. Compile the native runner for the board

In WSL, obtain ncnn once:

```sh
git clone https://github.com/Tencent/ncnn.git external/ncnn
git -C external/ncnn checkout c15904bb8918187ec100f1ebe708942af83b84be
```

Use your **board-compatible ARMv7 hard-float compiler/sysroot**, preferably the
same working toolchain as the matrix benchmark. The board's old Linux/libc means
a generic current Ubuntu compiler may produce an executable it cannot load.
Set `CC`/`CXX` to the actual BSP compiler paths if their names differ from
`arm-linux-gnueabihf-gcc/g++`. Do not substitute an AArch64 compiler.

```sh
project_root="$PWD"
cmake -S external/ncnn -B build/ncnn-arm \
  -DCMAKE_TOOLCHAIN_FILE="$project_root/configs/cortex_a9.toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$project_root/build/ncnn-arm-install" \
  -DNCNN_VULKAN=OFF -DNCNN_VFPV4=OFF -DNCNN_RUNTIME_CPU=OFF \
  -DNCNN_OPENMP=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF
cmake --build build/ncnn-arm --parallel 4
cmake --install build/ncnn-arm
cmake -S software/hps/model_runner -B build/model-runner-arm \
  -DCMAKE_TOOLCHAIN_FILE="$project_root/configs/cortex_a9.toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$project_root/build/ncnn-arm-install/lib/cmake/ncnn"
cmake --build build/model-runner-arm --parallel 4
file build/model-runner-arm/model_runner
```

Review the build logs for OpenMP support. If unavailable, build with
`-DNCNN_OPENMP=OFF` and report one-thread execution only. Check dynamic libraries
on the board with `ldd`; resolve toolchain/libc/libstdc++/libgomp compatibility
before benchmarking. Cross-compilation and HPS execution were not tested here.

If the board has a compatible compiler/CMake, a native build is another option:
use the same source commands there with the toolchain option omitted and suitable
Cortex-A9 compiler flags. Desktop-built ARM64 binaries cannot be copied to HPS.

## 7. Transfer and run on HPS Linux

Create a folder on USB or transfer by SCP. Preserve this layout on the board:

```text
vit_model/
  build/model_runner                 <- ARM executable from step 6
  scripts/run_hps_model_metrics.sh
  scripts/collect_hps_environment.sh
  model/tinyvit_fp32.ncnn.param
  model/tinyvit_fp32.ncnn.bin
  inputs/inputs.tsv
  inputs/00000.input.bin ...          <- all input files named in TSV
```

In the **board's Linux terminal**, inside `vit_model`:

```sh
chmod +x build/model_runner
mkdir -p results
RUNS=3 WARMUPS=1 sh scripts/run_hps_model_metrics.sh \
  model/tinyvit_fp32.ncnn.param model/tinyvit_fp32.ncnn.bin \
  inputs/inputs.tsv results/fp32_pilot 1
```

Copy `results/fp32_pilot` back to WSL as `results/raw/hps_fp32_pilot`, then verify:

```sh
python scripts/compare_model_runs.py --native --reference models/exports/dev10 \
  --candidate results/raw/hps_fp32_pilot/evaluation \
  --out results/raw/hps_fp32_pilot_check.json
```

Inspect `all_logits_within_tolerance`, errors and prediction agreement. Default
atol=1e-4/rtol=1e-3 are starting diagnostics, not a universal acceptance standard.
Resolve unexpected mismatches before reporting speed. This comparison writes a
report; a zero command exit alone does not assert numerical equivalence.

Then run final distributions on the **board**:

```sh
sh scripts/run_hps_model_metrics.sh \
  model/tinyvit_fp32.ncnn.param model/tinyvit_fp32.ncnn.bin \
  inputs/inputs.tsv results/fp32_t1 1
sh scripts/run_hps_model_metrics.sh \
  model/tinyvit_fp32.ncnn.param model/tinyvit_fp32.ncnn.bin \
  inputs/inputs.tsv results/fp32_t2 2
```

Each wrapper invocation saves one evaluation and three independent timing
processes, each with 10 warmups/100 calls. Inputs are preloaded (maximum 32 images
for timing). `run.json` contains mean/median/p95/min/max/stddev/CV, throughput,
load/first-inference times, peak RSS and available steady RSS. `latency.csv`
contains raw samples. Two threads are meaningful only with working parallel kernels.

For one sustained test, run the executable directly on the board:

```sh
./build/model_runner bench model/tinyvit_fp32.ncnn.param model/tinyvit_fp32.ncnn.bin \
  inputs/inputs.tsv results/fp32_sustained 1 10 100 in0 out0 300 fp32
```

This continues for at least 100 calls AND 300 seconds. Compare early/late samples
and track `/proc/PID/status`/temperature separately if needed. Peak RSS covers
the entire process; the inputs also occupy memory. Throughput from summed model
durations and direct service-loop throughput are separately labelled.

For more accuracy images, export successive shards with `--offset 0 --limit 100`,
then `--offset 100 --limit 100`, etc. Run runner `eval` mode, not the <=32-image
benchmark wrapper, for each shard. Sum correct counts and image counts across
unique shards; do not average unequal-sized shard percentages. Reuse the same
converted model. Archive every shard's manifest and predictions; only claim
full HPS accuracy if all intended held-out images were executed.

## 8. Native INT8 runtime comparison (separate from Q1–Q4)

Build ncnn's conversion tools on **WSL**, without the ARM toolchain:

```sh
cmake -S external/ncnn -B build/ncnn-tools -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF
cmake --build build/ncnn-tools --target ncnn2table ncnn2int8 --parallel 4
model_step export --split calib --limit 512 --out models/exports/calib512
build/ncnn-tools/tools/quantize/ncnn2table \
  models/exports/dev10/tinyvit_fp32.ncnn.param models/exports/dev10/tinyvit_fp32.ncnn.bin \
  models/exports/calib512/npy_list.txt models/exports/native.table \
  'shape=[224,224,3]' thread=1 method=kl type=1
build/ncnn-tools/tools/quantize/ncnn2int8 \
  models/exports/dev10/tinyvit_fp32.ncnn.param models/exports/dev10/tinyvit_fp32.ncnn.bin \
  models/exports/tinyvit_int8.param models/exports/tinyvit_int8.bin models/exports/native.table
```

NumPy calibration inputs already contain the correct resize/crop/normalization;
do not normalize again. Inspect conversion logs, graph precision and operator
support. Call this configuration **R1: ncnn-native quantization**, not Q1–Q4:
coverage, activation scaling, bias and accumulation behavior can differ.

Transfer the two INT8 model files, then on HPS:

```sh
PRECISION=runtime-int8 sh scripts/run_hps_model_metrics.sh \
  model/tinyvit_int8.param model/tinyvit_int8.bin \
  inputs/inputs.tsv results/native_int8_t1 1
```

Repeat held-out accuracy/logit comparison against the same FP32 goldens. Enabling
INT8 in a runtime does not prove every operator uses INT8; record actual coverage
and fallbacks. Q3/Q4 need custom matching runtime operators before their native
speed can be measured; standard ncnn conversion does not implement your RTL
tile contract. Use the supplied numerical simulator and existing C arithmetic
as the reference when implementing those custom operators.

## 9. Remaining profiles and final report

- **HPS operator times:** build a separate ncnn and runner with
  `-DNCNN_BENCHMARK=ON`, run a short test and capture stderr. This prints native
  layer timings. Map deployed layers to the exported graph/stages; keep the
  profiling build separate from uninstrumented timing. Do not use desktop
  stage percentages as HPS percentages.
- **Real GEMM inputs:** use `export_model_kernel_cases.py --capture
  results/raw/architecture --calibration results/raw/calibration/calibration.json
  --out data/layers_real --max-columns 49`, transfer this folder, and use the
  existing C benchmark. The default is a partial-token pilot; use 0 for all
  captured tokens. Bias is excluded and the original contracts are preserved.
- **CPU utilization/page faults:** if GNU time exists on HPS, prefix a fresh
  direct runner command with `/usr/bin/time -v`. Preserve its stderr separately.
  If `perf` exists and PMU access works, use a separate `perf stat` diagnostic
  pass. Record unavailable counters rather than estimating them.
- **HPS application latency:** in your camera/image application, start a monotonic
  timer after image decoding, perform the exact reference preprocessing, run
  inference and choose the label, then stop. Save preprocessing/model/postprocess
  intervals and direct total. The provided tensor runner does not implement
  camera capture, decoding, overlays or image preprocessing on the HPS.
- **Power:** only if measured energy is part of your claims, record board-input
  power with an instrument; J/image = integrated joules / completed images.
  No software timing script can manufacture a board power measurement.

Copy results back and create an index, for example:

```sh
python scripts/summarize_model_results.py \
  --runs results/raw/F0_dev results/raw/Q1_dev results/raw/Q2_dev \
  results/raw/hps_fp32_pilot/timing_1 \
  --out results/reports/model_results_index.csv
```

Use Tables A–F in `docs/full_cpu_baseline_steps.md` for the final report. Keep
kernel latency, model latency, application latency, desktop results and HPS
results distinct. Compare FPGA execution later against the identical CPU
quantized graph, including transfer/packing overhead in the hybrid total.

## Validation status of this package

Python unit/reference tests passed; desktop smoke checks exercised the complete
architecture, Q1–Q4, workload export and TorchScript conversion using synthetic
images and a random checkpoint. The C++ runner compiled and its FP32 desktop
ncnn logits matched the original synthetic model within the stated tolerance.
These are software checks, not trained-model accuracy or Cyclone V measurements.
You must still verify the actual checkpoint, conversion and ARM/BSP deployment.
