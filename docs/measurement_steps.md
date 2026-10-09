# Step-by-step measurements: TinyViT-21M INT8

## 0. Machines and current status

Confirmed: TinyViT-21M, INT8 operands; proposed fixed evaluation input is 224x224,
batch 1. Agree the checkpoint and complete quantization contract before model
accuracy measurements. INT8 inputs with INT32 accumulation do not by themselves
specify softmax, normalization, bias or residual arithmetic.

| Task | Mac alone? | DE1-SoC required? | Windows role |
|---|---|---|---|
| Golden arithmetic, synthetic vectors, reports | Yes | No | Optional |
| Desktop FP32 reference / quantization development | Yes, compatible runtime needed | No | Optional |
| Actual HPS kernel/model latency and peak RSS | No | Yes | Can serve as terminal |
| DMA performance / FPGA correctness and timing | No | Yes | Quartus build/programming host |
| Quartus fitter and timing reports | No native Quartus on Mac | Board optional for compile | Supported Quartus host |

If Linux already boots on the board, the Mac can connect by UART/SSH and perform
all CPU measurements. No custom FPGA accelerator is needed for CPU-only tests.
Keep the boot image's expected base FPGA configuration if its boot flow uses one.

Rechecked teammate repository main on 2026-10-09: still commit 966e3459.
It contains RTL, register generation, simulation models/tests and a dimension
document; no full TinyViT HPS application, deployable model, CPU profiler or DMA
benchmark source is checked into that branch. Work outside this branch is unknown.

## 1. Prepare and verify vectors on the Mac

From the project root, with an installed Python 3 (prefer a maintained release):

```sh
cd /Users/yilingzha/Documents/ECE496/ViT_fpga
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
python -m pip freeze > configs/baseline_python_environment.txt
python -m unittest discover -s tests -v
python scripts/generate_vectors.py
python scripts/generate_layer_vectors.py
make
```

The first generator creates eight small hardware-compatible cases. The second
creates CPU-only full-sized GEMMs for stage-1 QKV (576,192,49), FC1
(768,192,784), and FC2 (192,768,784). Both provide full-sum and tile-clipped
golden outputs. The new vectors use synthetic signed INT8 values and q16=128;
they are not checkpoint-derived activations, calibrated scales or an accuracy test.
Full-layer vector generation requires NumPy only on the Mac; the board does not
need NumPy to run the C benchmark.

## 2. Run a desktop smoke test

```sh
python scripts/run_cpu_profile.py --vectors data/layers --target desktop_cpu \
  --out results/raw/mac_layers_pilot --runs 3 --warmups 1
```

The output directory must be new; use a different name on repeats. Review:

- `summary.csv`: two rows per case, one per arithmetic contract.
- `*.samples.csv`: raw timings, including group duration and calls/group.
- `*.log`: actual target, compiler, flags and timing scope.
- `run.json`: environment, command lines, vector hashes and suite wall times.

These files are Mac results, never HPS results. Counts of 3 are smoke tests,
not enough samples for a meaningful p95 distribution.

## 3. Connect to the HPS and collect its environment

Boot the board's known-good Linux image. Find its IP over the UART console with
`ip addr` or `ifconfig`. Connect from Mac Terminal, substituting real credentials:

```sh
ssh USER@BOARD_IP
uname -a
cat /proc/cpuinfo
cat /etc/os-release
free -m
gcc --version
python3 --version
```

Use the onboard ARM Cortex-A9 Linux shell. The suite runner expects Python >=3.6
and identifies HPS runs as ARMv7 Linux; it cannot prove the physical board model.
Record the board revision separately. If Python is unavailable, run C directly
as in step 5's fallback. If GCC is absent, install through the BSP's supported
package route or cross-compile with its ARMv7 Linux ABI/sysroot. Do not use the
Mac executable, a bare-metal executable, or an AArch64 compiler.

## 4. Transfer source and data; compile on the board

On the Mac:

```sh
tar -czf /tmp/vit_cpu_baseline.tar.gz Makefile software/hps \
  scripts/run_cpu_profile.py data/generated data/layers
scp /tmp/vit_cpu_baseline.tar.gz USER@BOARD_IP:~/
```

On HPS Linux, use a fresh folder:

```sh
mkdir -p ~/vit_cpu_baseline_run1
cd ~/vit_cpu_baseline_run1
tar -xzf ~/vit_cpu_baseline.tar.gz
make CC=gcc
```

If an older Linux libc needs the realtime library:

```sh
make CC=gcc LDLIBS='-lm -lrt'
```

The baseline is simple single-thread C; keep that label. Final claims should
also compare against a reasonable cache-friendly/NEON or runtime implementation.
Keep arithmetic identical when comparing implementations.

## 5. Run the actual HPS kernel measurements

First perform a short correctness/timing pilot:

```sh
python3 scripts/run_cpu_profile.py --vectors data/generated --target hps_cpu \
  --out results/raw/hps_small_pilot --runs 5 --warmups 1
python3 scripts/run_cpu_profile.py --vectors data/layers --target hps_cpu \
  --out results/raw/hps_layers_pilot --runs 3 --warmups 1
```

Inspect `summary.csv`: all `verified` fields must be true. This checks the C
implementation before/after timed repetitions against Python golden files; it
does not establish FPGA correctness. A failing comparison exits nonzero.

Use the pilot's times to budget the next run. For a preliminary report:

```sh
python3 scripts/run_cpu_profile.py --vectors data/layers --target hps_cpu \
  --out results/raw/hps_layers_report --runs 20 --warmups 3
```

For final distributions increase to 100 samples/10 warmups when affordable.
Each case runs BOTH contracts; include both when estimating time. Tiny kernels
need grouping, for example `--inner 1000 --runs 20`. Increase/decrease inner
until group duration is comfortably measurable (roughly 10 ms or more is a
useful starting point). Inspect group_ms, rather than assuming 1000 is enough
for every tiny case. Grouped p95 describes averages, not individual tail latency.

No-Python fallback, on HPS:

```sh
mkdir -p results/raw/manual
BENCH_SAMPLES_PATH=results/raw/manual/fc1.samples.csv \
  ./build/bench_matmul data/layers/stage1_mlp_fc1 20 1 3 \
  > results/raw/manual/fc1.csv 2> results/raw/manual/fc1.log
```

Repeat for FC2 and QKV. Arguments are case directory, sample count, calls/sample,
and warmup calls. Save environment output manually with this fallback.

Return results to Mac:

```sh
scp -r USER@BOARD_IP:~/vit_cpu_baseline_run1/results/raw/hps_layers_report results/raw/
```

## 6. First professor-facing report: available without full-model deployment

Use the CPU report template. Include the three shapes, contracts, verified
outputs, median/mean/p95, CV, GMAC/s, CPU/OS/compiler and sample policy. Record
synthetic data and GEMM-only timing. Present numerical differences between
contracts as algorithmic error, not a hardware failure.

MACs=K*N*M; GMAC/s=MACs/(mean_ms*1e6). The two C methods implement the same MAC
count with different requantization overhead. Warmup and timed runs reuse data;
this is a steady-state/cache-reuse measurement, not a DRAM bandwidth benchmark.

Do not yet claim model FPS, model accuracy, CPU-to-FPGA speedup, or layer runtime
shares. Those require the following deployment and hardware phases.

## 7. Recreate the model reference on Mac if team code is unavailable

Use a separate model environment so dependency changes do not affect the kernel
tools. Obtain the official Microsoft Cream/TinyViT code and TinyViT-21M 224
checkpoint. The published 84.8% result corresponds to the distilled 22k-to-1k
checkpoint; record the exact source revision and downloaded checkpoint hash.
The official `TinyViT/inference.py` is a FP32 CPU inference example and uses its
matching evaluation transform. Follow the upstream dependency instructions in
that checkout; historical pinned packages may need compatibility work with a
current Python/PyTorch version. The example has not been executed here.

Steps:

1. Run one image through the unmodified reference successfully.
2. Set eval mode, disable gradients, use CPU explicitly and batch 1.
3. Save preprocessed input tensors, FP32 logits, top-1/top-5, preprocessing
   configuration and hashes for 10 fixed development images.
4. Capture Linear input/output shapes and real weights/activations. Reconcile
   synthetic benchmark dimensions with the actual runtime graph.
5. Collect a disjoint calibration set (start with a few hundred representative
   images), define weight/activation scales, zero points, rounding, bias,
   residual, and clipping rules. Validate quantization on held-out images.
6. Decide where FP32 remains (typically nonlinear stages initially) and report
   it explicitly. Calling a graph INT8 does not imply every operation is INT8.
7. Compare full-sum and current tile-clipped arithmetic. If quality fails, change
   the precision/accumulation contract before deploying the complete graph.

Casting the whole model to int8 is not a valid quantization procedure. Do not
time a float simulation of quantization and label it native INT8 execution.

## 8. Deploy a real full-model runtime on HPS (implementation still needed)

This repository does not contain a `tinyvit_hps` executable. Deployment is a
separate engineering task, not another switch to the matrix benchmark.

An ARMv7 C++ inference runtime is a candidate; ncnn documents ARM Cortex-A Linux
cross-compilation, INT8 paths and custom layers. This does not establish turnkey
TinyViT compatibility with your BSP or arithmetic contract. Run this gate first:

1. Record board kernel, libc, ABI and compiler; choose a matching ARMv7 toolchain.
2. Build a minimal runtime example for Cortex-A9 (ARMv7/NEON; avoid instructions
   unsupported by Cortex-A9 such as ARMv8 dot-product or VFPv4). Disable GPU use.
3. Convert/export the fixed model with all operators accounted for. Resolve any
   unsupported operators explicitly; log precision and fallback paths.
4. Run FP32 one-image inference on HPS, then compare logits to Mac reference.
5. Enable real supported INT8 kernels with calibrated scales; compare quantized
   outputs against the same CPU graph. Runtime defaults may not match FPGA q16
   or per-tile clipping, requiring custom operations.
6. Confirm actual native INT8 execution through runtime/operator reporting.
7. Time the deployed CPU graph. Keep the matching CPU graph for the hybrid
   comparison so quantization speedup is not confused with FPGA speedup.

If this gate fails, publish the verified kernel report and call full-model
deployment pending. Do not extrapolate complete TinyViT latency from a few GEMMs.

## 9. Full-model timing and operator profile (after step 8)

1. Load model/weights once and record load time separately.
2. Set runtime threads=1; later repeat at 2 threads if supported.
3. Preprocess images and measure preprocessing separately.
4. Warm the model for 3 runs, then record 20 complete individual-call durations
   initially. Timing starts with input tensor ready and ends with logits ready.
5. Save each duration and output after stopping the clock. Summarize mean,
   median, p95, stddev, CV and images/total elapsed inference time.
6. Run a SEPARATE instrumented pass using the runtime's operator profiler or
   timers in its execution loop. Group self-times into conv, QKV/projection,
   attention products, MLP, nonlinearities and copies. Capture call counts.
7. Compute fractions using non-overlapping timings; report profiler overhead and
   do not sum parent block and child op durations. Repeat final latency without
   profiler instrumentation.
8. On Linux with GNU time installed, run `/usr/bin/time -v` around the actual
   deployed inference command. Maximum resident set size is process peak RSS,
   not model checkpoint size. BusyBox time may lack this option; process
   VmHWM is another option if the application records it before exit.

## 10. Quality, FPGA comparison and targets

On held-out labelled inputs, compute top-1 accuracy, reference prediction
agreement, logit/tensor absolute errors and saturation rates. Keep calibration
disjoint from evaluation. Start a small pilot, then expand for the final <=1 pp
accuracy-loss target; report statistical uncertainty.

When hardware works, send the small banked vectors, wait for completion, save
the complete 2048-byte output image, and run:

```sh
python scripts/compare_output.py data/generated/identity16 board-output.bin --reference rtl
```

Require zero element mismatches under the frozen contract. Repeat for multiblock,
partial, negative/extreme and consecutive transactions. The helper itself does
not send data to the board. Full-sized layer files require a host tiler.

Measure direct offload wall time including packing, transfers, cache maintenance,
launch/wait and unpacking. Record hardware compute and transfer intervals
separately without double-counting. Sweep actual tile sizes for DMA bandwidth.

Proposed targets remain >=2x per offloaded operation including overhead and
>=1.5x full-model over an equivalent HPS-only quantized graph. Use the measured
eligible fraction f with S=1/((1-f)+f/r+h) to test plausibility. Do not promise
an absolute FPS until measured model latency exists.

Sources: [official TinyViT inference](https://github.com/microsoft/Cream/blob/main/TinyViT/inference.py),
[model code](https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py),
[ncnn build guidance](https://github.com/Tencent/ncnn/wiki/how-to-build).
