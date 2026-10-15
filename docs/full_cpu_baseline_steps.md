# Completing the CPU baseline and quantization report

For executable code and Windows/WSL-to-HPS commands, start with
[windows_hps_commands.md](windows_hps_commands.md). This document defines the
measurement scope and report requirements.

Repository review: 2026-10-15. This is an execution plan, not new model measurements.
Scope: TinyViT-21M, 224x224, batch 1, DE1-SoC Cortex-A9 CPU; desktop for reference
and quantization development. HPS means the board's hard processor system.

Architecture alignment: the user-supplied architecture diagram is the project
target for this guide. Its widths, resolutions and block counts match the
published TinyViT-21M-224 configuration. This is the full four-stage classifier,
not the earlier illustrative custom model with only 1–3 transformer blocks.

## Architecture to validate before measurements

Use the diagram's **one-based Stage 1–4 names** in all new reports:

| Component | Resolution | Channels | Blocks | Expected output before next downsampling |
|---|---|---:|---|---|
| Input | 224x224 | 3 | — | `[1,3,224,224]` |
| Convolutional stem | 56x56 | 96 | Two convolutions | `[1,96,56,56]` |
| Stage 1 | 56x56 | 96 | 2 MBConv | `[1,96,56,56]` |
| Stage 2 | 28x28 | 192 | 2 TinyViT | `[1,784,192]` |
| Stage 3 | 14x14 | 384 | 6 TinyViT | `[1,196,384]` |
| Stage 4 | 7x7 | 576 | 2 TinyViT | `[1,49,576]` |
| Global average pooling | — | 576 | — | `[1,576]` |
| Head LayerNorm | — | 576 | — | `[1,576]` |
| Classifier | — | 1000 classes | Linear head | `[1,1000]` logits |

The diagram combines head LayerNorm and classifier; they are separated here
for measurement. Four-dimensional tensors above use batch/channel/height/width;
three-dimensional tensors use batch/token/channel. This is image classification;
the shown graph has no bounding-box or segmentation head.

Include all three stage transitions: 56x56/96 -> 28x28/192,
28x28/192 -> 14x14/384, and 14x14/384 -> 7x7/576. They have computation and
memory costs even though the diagram shows outputs **before** downsampling.
Save boundary tensors before and after each transition. Check hook locations:
a stage module may include downsampling, so its returned tensor may already
belong to the next stage. Assert shape, layout, dtype and finite values.

The diagram does not specify attention windows/heads, MLP expansion, exact
convolution parameters or checkpoint. For workload planning below, use the
official 21M-224 defaults: Stage 2/3/4 windows 7/14/7, heads 6/12/18,
head dimension 32 and MLP expansion 4. The reference uses a convolutional stem,
MBConv, convolutional patch merging and local depthwise convolutions in TinyViT
blocks. Verify these details against the team's pinned implementation before
generating real vectors; matching external shapes alone does not prove equivalence.
[Reference implementation](https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py).

The CPU reference must execute the entire graph above. Record a separate
execution-placement map for the FPGA version: CPU, FPGA or pending for each
operator. CPU+FPGA execution is a hybrid implementation; claiming all-model FPGA
execution requires every remaining operation to be implemented there as well.
The current plan measures a full CPU reference regardless of the final placement.

## 1. Current status

The preliminary **single-thread matrix-kernel baseline is complete**. The full
CPU inference baseline is not complete in this folder.

| Evidence | Status | What it establishes |
|---|---|---|
| `baseline/reference.py`, `tests/test_reference.py` | Present | Integer arithmetic and memory-layout reference checks |
| `software/hps/bench_matmul.c` | Present | Simple single-thread GEMM, two arithmetic contracts, bytewise verification and timing |
| `results/raw/hps_board_run1/` | Present; numeric audit rerun successfully | 11 fixed cases x 2 contracts; 440 report samples and 66 pilot samples |
| `results/reports/hps_board_run1.md` | Present | Preliminary measured board kernel report |
| `scripts/generate_layer_vectors.py` | Present | Three synthetic Stage 2 GEMMs, historically named `stage1_*`; not trained weights/activations |
| Trained checkpoint and labelled evaluation dataset | Team must supply | Needed for classification accuracy; synthetic tooling fixtures are not evaluation data |
| Desktop model evaluator | Implemented | See `scripts/model_baseline.py` and Windows command sheet |
| Full-model HPS application and deployment model | Runner source supplied; board deployment pending | Requires ARM build, trained model conversion and board validation |
| Calibrated quantized model variants | Tools supplied; real-data runs pending | Linear-only numerical Q1–Q4 and separate ncnn-native workflow |
| Optimized same-contract CPU kernel baseline | Not found | Needed for credible final accelerator performance claims |

Recorded kernel medians, milliseconds:

| Synthetic case | K,N,M: W[K,N] x X[N,M] | Full accumulation | RTL compatibility |
|---|---|---:|---:|
| Stage 2 QKV, one window | 576,192,49 | 22.768 | 53.479 |
| Stage 2 MLP FC1 | 768,192,784 | 619.096 | 1130.260 |
| Stage 2 MLP FC2 | 192,768,784 | 971.128 | 1137.588 |

Naming reconciliation: `stage1_qkv_window`, `stage1_mlp_fc1` and
`stage1_mlp_fc2` refer to the first attention stage (reference `layers[1]`),
which is **Stage 2 in the supplied diagram**. Preserve historical filenames,
hashes and raw results; use the corrected stage label in new reports. They do
not measure the diagram's Stage 1 MBConv blocks or complete Stage 2 execution.

These are synthetic GEMM measurements, not whole-layer/model latencies. Bias,
normalization, nonlinearities, complete attention and preprocessing are absent.
Both columns use INT8 operands; neither is FP32. They differ in when they
requantize and saturate, rather than being two calibrated model quantizers.

The report records GCC 15.2.0, Cortex-A9/NEON flags, one compute thread, 20 report
samples and 3 warmups. It does not establish CPU frequency, governor, affinity,
binary hash or trustworthy acquisition date. Do not fill historical gaps with
today's environment. NEON compiler flags do not prove vectorized inner loops.

## 2. Freeze the experiment and preserve the evidence

**Where:** project host, with environment collection on HPS.

1. Back up existing raw data: `results/raw/` is ignored by Git. Use new run IDs
   and directories for future measurements.
2. Recover any teammate-owned checkpoint, export script, HPS runtime and dataset
   mapping before recreating them. Absence here does not establish absence elsewhere.
3. Pin model source revision, checkpoint SHA-256, class ordering and preprocessing:
   resize/crop, interpolation, RGB order, normalization, tensor layout and dtype.
   Confirm every boundary in the architecture table, the 1000-class head, all
   2/2/6/2 blocks, attention settings and three downsampling operations. Keep
   224x224 fixed for this baseline because the supplied architecture specifies it.
4. Copy `configs/run_manifest.template.json` per experiment. Add export/runtime
   versions, binary/model hashes, split hashes, quantization configuration,
   calibration seed, CPU affinity/governor and durable raw-data location.
5. Record board revision, OS, ABI/libc, compiler flags, available memory, swap
   policy and background load. Record frequency/temperature when exposed;
   unavailable fields must remain explicitly unavailable.
6. Fix batch=1 and start with one thread. Also measure two threads if supported;
   retain both and use the best practical CPU setting for headline comparisons.
7. Adopt an accuracy target before tuning. The existing plan proposes at most
   **1 percentage point top-1 loss** versus the pinned FP32 model. It is not
   an achieved result.

Existing audit command, from the repository root:

```sh
python3 scripts/audit_hps_results.py results/raw/hps_board_run1 /tmp/vit_hps_audit.json
```

This audit checks the first run's specific 3/20 sample counts and compiler log.
It is not a generic validator for future runs, and does not rerun the board.

**Deliverable:** reproducibility manifest, protected kernel evidence and targets.

## 3. Establish FP32 accuracy and dataset splits

**Where:** Mac/desktop CPU. Start this while preparing HPS deployment.

1. Use a separate model environment; pin working dependencies. Start from the
   [official TinyViT model/inference example](https://github.com/microsoft/Cream/tree/main/TinyViT),
   using evaluation mode, gradients disabled and CPU explicitly selected.
2. Run one labelled image and check class-index mapping. Then save preprocessed
   input tensors, logits and top-5 predictions for 10 fixed development images.
   Also capture stem, Stage 1–4 pre-downsampling outputs, transition outputs,
   pooled vector and head output. Compare these boundaries after conversion,
   quantization and HPS deployment to locate the first divergence.
3. Create disjoint calibration, development/tuning and final-evaluation manifests.
   An initial budget is 256–512 representative calibration images and 1000
   development evaluation images. Fix seeds/class balance and check duplicate
   identities or hashes across splits.
4. Prefer training images for calibration/tuning and reserve official validation
   data for the final test. If validation images are used for tuning, reserve a
   separate final subset and disclose this; tuned-on images are not untouched tests.
5. Measure FP32 top-1/top-5 on exactly the IDs used for subsequent quantized
   comparisons. Preserve per-image labels, predictions and failures.
6. Use the full 50,000-image ImageNet-1k validation set for final evaluation when
   available and untouched, or state the held-out subset size and uncertainty.
   A custom image set measures that set, not standard ImageNet accuracy.

The official 224 model table lists 84.8% top-1 / 97.3% top-5 for the distilled
22k-to-1k checkpoint, versus 83.1% / 96.5% for the 1k-only checkpoint. These are
published reference values, not your measured baseline.
[Source: official TinyViT model table](https://github.com/microsoft/Cream/tree/main/TinyViT).

**Deliverable:** runnable evaluator, fixed preprocessing/class mapping, split
manifests, FP32 predictions and measured labelled accuracy.

## 4. Pass the full-model HPS feasibility gate early

**Where:** HPS; desktop may build/export and transfer artifacts.

1. Reuse a working team runtime if available. Otherwise choose a runtime buildable
   for this ARMv7 Linux ABI and exported graph. ncnn documents ARM Cortex-A builds,
   making it a candidate, not proof of TinyViT/operator/old-BSP compatibility.
   [Source: ncnn build documentation](https://github.com/Tencent/ncnn/wiki/how-to-build).
2. Run a minimal runtime example first. Match Cortex-A9 instructions and ABI;
   do not assume desktop Python wheels or AArch64 executables work on this board.
3. Export the FP32 model and inspect operator coverage, fusions, layouts and
   fallback implementations. Preserve conversion logs and model hash.
4. Compare exported-runtime outputs against the original model on the 10 saved
   input tensors, then run the same tensors on HPS. Record logit MAE/max error
   and label agreement. Establish FP32 absolute/relative tolerances appropriate
   to the runtime; label agreement alone is insufficient.
5. Confirm complete inference fits RAM and runs repeatedly without crashes,
   invalid outputs or memory growth. Record rough load/first-inference time and RSS.
6. Establish native INT8 kernel support per selected operator. Save runtime
   kernel/operator information and FP32 fallback coverage. Quantized storage or
   a quantize/dequantize graph alone does not establish native INT8 execution.

**Deliverable:** working full FP32 HPS inference and explicit INT8 support map.
If blocked, record the actual build/operator/memory failure. Do not substitute
extrapolated GEMM latency. Runner source is now provided in
`software/hps/model_runner/`; compilation and actual board validation remain pending.

## 5. Capture real workloads and strengthen the kernel baseline

**Where:** capture on desktop; timing on HPS.

1. Inventory actual deployed operator shapes, layouts and call counts, including
   windows/heads. Cover patch embedding, convolutions/patch merging, QKV and
   projection, both attention matrix products, MLP, normalization, GELU, softmax,
   residuals, pooling and classifier. These operations appear in the
   [official TinyViT implementation](https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py).
2. Capture representative weights/activations. After calibration, generate goldens
   using actual scales. Retain synthetic cases as directed correctness tests.
3. Compare the simple C kernel with a practical runtime kernel or optimized
   cache-friendly/NEON implementation. Require equal results under the same
   integer contract. Keep CPU and later FPGA comparison arithmetic matched.
4. Measure compute-only latency and a second total including required packing,
   layout changes and requantization. State which work is amortized/reused.
5. Target 10 warmups and 100 samples, inspecting stability; repeat in three
   independent sessions when practical. Record session variation. Twenty
   samples provide only a preliminary p95; p99 needs more samples and is optional.

Important constraints in the current code:

- `bench_matmul.c` requires N divisible by 16 for both contracts. For actual
  N=49 attention workloads, extend the CPU implementation or disclose padding
  to 64 and count useful versus executed MACs.
- Its single scalar q16 and absence of bias cannot express every real layer.
- Large layer matrices exceed accelerator buffers and require host tiling.
- Full-sized synthetic GEMMs are not complete attention or MLP blocks.

Commands supported **now**, on HPS with the correct binary/source/data already
transferred (each output directory must be new):

```sh
python3 scripts/run_cpu_profile.py --vectors data/layers --target hps_cpu \
  --out results/raw/hps_layers_final_run1 --runs 100 --warmups 10 --inner 1
python3 scripts/run_cpu_profile.py --vectors data/generated --target hps_cpu \
  --out results/raw/hps_small_final_run1 --runs 100 --warmups 10 --inner 1000
```

These rerun the simple synthetic kernels only. They do not run model accuracy
or optimized kernels. Inspect group duration for tiny kernels and increase
grouping if needed; grouped p95 describes group averages, not individual-call
tail latency. The USB wrapper currently offers only pilot/report modes.

**Deliverable:** real-shape inventory, optimized CPU comparison, matched-contract
correctness, with synthetic and checkpoint-derived workloads distinguished.

### Workloads required by the supplied architecture

The following are derived planning shapes, not new measurements. Let C be stage
width, L the number of image tokens, T the tokens per attention window and h the
number of heads. Use the reference defaults declared above until the pinned
model confirms them. Logical repetitions are not necessarily runtime call counts:
a runtime can batch windows or heads into a single operator call.

| Stage | Blocks | C | L | Window | T | Windows/block | h | Head-window pairs/image |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Stage 2 | 2 | 192 | 784 | 7x7 | 49 | 16 | 6 | 192 |
| Stage 3 | 6 | 384 | 196 | 14x14 | 196 | 1 | 12 | 72 |
| Stage 4 | 2 | 576 | 49 | 7x7 | 49 | 1 | 18 | 36 |

Using the repository convention W[K,N] x X[N,M], derive QKV as (3C,C,T),
projection as (C,C,T), MLP FC1 as (4C,C,L) and FC2 as (C,4C,L):

| Stage | QKV per window: K,N,M | Projection per window | FC1 over all stage tokens | FC2 over all stage tokens |
|---|---|---|---|---|
| Stage 2 | 576,192,49 | 192,192,49 | 768,192,784 | 192,768,784 |
| Stage 3 | 1152,384,196 | 384,384,196 | 1536,384,196 | 384,1536,196 |
| Stage 4 | 1728,576,49 | 576,576,49 | 2304,576,49 | 576,2304,49 |

QKV and projection have 32/6/2 logical window operations per image in Stage
2/3/4 respectively. Each MLP linear has 2/6/2 logical block invocations.
Do not multiply one block's latency and call the result a measured model time.

Each head/window additionally computes QK^T with (K,N,M)=(T,32,T) and P*V
with (T,T,32), where P is the softmax attention matrix. Thus benchmark
(49,32,49), (49,49,32), (196,32,196), and (196,196,32), with the pair counts
above for each product. Here both operands are activations; W is only a matrix
position in the benchmark notation, not a learned weight. Define their scales
independently from weight-based Linear quantization, including attention scaling,
relative-position bias and softmax. Use one consistent orientation/layout.

Benchmark the classifier as (1000,576,1), plus its bias, and measure head
LayerNorm and averaging across 49 tokens separately. The skinny M=1 operation
may have poor array utilization; offload eligibility does not establish benefit.

For the stem, both MBConv blocks, all three transitions and local convolutions,
extract actual convolution input/output channels, kernel, stride, padding and
groups from the graph. Report their native CPU convolution times. If mapping
to GEMM, include input lowering/packing and workspace; depthwise convolutions
must retain their grouped semantics rather than becoming dense convolutions.

The current generator provides only the three existing Stage 2 cases. Extend
it or add a real-tensor exporter for this inventory before running the expanded
suite. Existing commands do not automatically benchmark these additional shapes.

### Mapping these workloads to the 16x16 systolic array

The 16x16 array tiles internal matrices; it does not define image patch size or
image resolution. Keep the diagram's 224x224 input. The convolutional stem is
not a generic 16x16 patch-extraction layer.

For contraction tiling, N_pad = 16*ceil(N/16). Attention P*V therefore needs
49 -> 64 or 196 -> 208 if using the current N-multiple-of-16 contract. Pad the
GEMM operands with real-zero values **after softmax**, not attention logits
with zero before softmax, which would change the probability distribution.
For asymmetric formats real zero has the zero-point code, not necessarily byte 0.

Output dimensions K/M can use valid-lane handling or padding according to the
driver. With full rectangular padding, K_pad/M_pad use the same ceiling rule;
49 -> 64, 196 -> 208, 1000 -> 1008 and classifier M=1 -> 16. Do not assume
the current interface requires every output dimension to be padded: verify the
actual scheduler and valid-output handling. Discard padded outputs correctly.

Record useful MACs=K*N*M, executed MACs for the actual padded schedule, their
ratio, transaction count, bytes transferred and peak temporary storage. A
rectangular-padding estimate is K*N*M/(K_pad*N_pad*M_pad); this is arithmetic
utilization, not measured PE activity. Host tiling must also respect the weight,
input and output bank capacities in `configs/accelerator_contract.json`.

## 6. Compare quantization schemes in a controlled order

**Where:** desktop numerical experiments first; native HPS performance second.

PTQ means post-training quantization. W8A8 means 8-bit weights and activations
at selected operations. Start with supported Linear/Conv operations, retaining
normalization, softmax, GELU and sensitive/unsupported operations in FP32.
Explicitly declare treatment of QK and attention-probability-times-V products.
Call this mixed INT8/FP32 unless all stages really execute integer arithmetic.

Maintain a precision map for stem, Stage 1 MBConv, each transition, every
Stage 2/3/4 block and the classification head. Include local convolutions,
residual additions and head pooling/LayerNorm. Compare quantized weight-MAC
coverage and runtime fallback coverage per stage; attention products have two
activation operands and need their own policy. A first mixed-precision pilot
may retain stem/head in FP32, but must be labelled as such. For sensitive layers,
restore one operator group or stage at a time on development data. Report
boundary errors for all stages, especially propagation across the six Stage 3
blocks, without assuming that Stage 3 dominates latency before measuring it.

Static quantization derives activation parameters from calibration data;
dynamic quantization calculates them during inference and adds runtime work.
Calibration methods and tensor comparisons help diagnose quality loss.
[Source: ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
This is methodological guidance, not proof that ONNX Runtime deploys on this BSP.

### Core comparison matrix

Hold checkpoint, selected quantized operators, preprocessing and evaluation IDs
fixed unless explicitly changed. Record the actual deployed graph and fusion policy.

| ID | Configuration | Question answered | Current accelerator compatibility |
|---|---|---|---|
| F0 | Original FP32 reference and exported CPU model | Reference quality and deployment cost | Reference |
| Q1 | Static W8A8, symmetric zero-point 0, per-tensor weight/activation scales, full INT32 sum, normal runtime requantization | Basic quantization loss and CPU benefit | Multiplier, rounding and bias may differ from RTL |
| Q2 | Q1 with per-output-channel weight scales | Does finer weight scaling recover quality? | Single scale register cannot express arbitrary channel scales in one multi-row call |
| Q3 | Q1 with Q0.16 multiplier and floor rounding, still full-sum | Error from hardware multiplier/rounding limits | Needs full cross-tile accumulation support to offload as written |
| Q4 | Q3 with per-16-term requantization/clipping and saturating merges | Additional error from current RTL arithmetic | Closest arithmetic match; bias/layout/tiling still need definition |

Evaluate F0 and Q1–Q4 numerically. Time only variants actually implemented on
HPS. Mark unsupported timing cells unavailable; floating-point emulation is
valid for numerical studies, but its latency is not native INT8 performance.

For Q3 versus Q4, hold integer inputs, scales, bias policy and all other stages
fixed. Paired layer tests isolate tile arithmetic; full-model differences also
include propagation through later layers. Q1 versus Q3 changes two factors;
if necessary, add floor-only and Q0.16-only ablations to separate their effects.

Optional follow-ups after the core matrix:

- Compare MinMax with a fixed percentile or MSE-based clipping calibration rule,
  holding everything else fixed. Choose settings on development data only.
- Dynamic W8A8 if supported; include online scale computation in latency.
- Mixed precision: restore sensitive layers to FP32 and measure the tradeoff.
- Asymmetric activations: a separate experiment requiring zero-point correction,
  absent from the current raw signed dot product.
- QAT or INT4 only if needed after PTQ assessment; neither is required to finish
  this INT8 baseline, and INT4 has no current accelerator contract here.

### Required quantization specification

Save tensor dtype/range (including whether -128 is used), scale, zero point,
granularity/axis, calibration method/count/IDs, rounding/tie behavior, clipping,
bias placement/precision, accumulator/output precision and residual rescaling.
Calibrate the deployed graph, including agreed folding/fusion, and version its config.

For symmetric operands, a conventional integer linear operation is:

```text
q_x = clamp(round(x / s_x)); q_w = clamp(round(w / s_w))
a = sum(q_w * q_x)                         # full INT32 sum
q_b = round(b / (s_w * s_x))                # when using INT32 bias
q_y = clamp(round((a + q_b) * m)); m = s_w * s_x / s_y
```

Per-channel weights imply per-channel bias scales and usually output multipliers.
Check accumulator bounds including bias and any zero-point corrections. The
starter has no bias. Adding bias after an already clipped INT8 result is not
equivalent to adding it before requantization. If bias stays FP32 outside the
offloaded operation, model and report that mixed-precision boundary explicitly.

The current contract represents m as q16/65536 with 0 <= q16 <= 65535, floors
scaled sums, and clips tiles and merges. For Q3/Q4 record the m-to-q16 conversion,
represented multiplier, absolute/relative error, values rounded to zero and
values above 65535/65536. Do not silently clamp unrepresentable multipliers and
call graphs equivalent. Scale changes/recalibration create a new variant;
alternatively revise the hardware contract. Per-channel scaling may require
more calls or a new interface, so include that cost in future comparisons.

**Deliverable:** saved configurations, per-layer error/clipping diagnostics,
development accuracy and a selected deployable candidate.

## 7. Run accuracy and numerical-correctness tests

1. Run each frozen variant on identical labelled IDs. Save true label, top-5
   indices/scores, variant ID, success/failure and logits. Save intermediate
   tensors on a smaller fixed diagnostic subset to limit storage.
2. Compute top-1/top-5, accuracy loss in **percentage points**, and FP32 top-1
   prediction agreement. Agreement is not labelled accuracy.
3. Report 95% accuracy intervals (e.g. Wilson) and a paired bootstrap interval
   for accuracy loss, resampling the same image IDs for both variants. Record
   seed/resample count; stratify appropriately for stratified sampling.
   Repeating identical images does not enlarge the accuracy sample size.
   Full-validation accuracy is exact for that finite set; intervals describe
   uncertainty about generalization beyond it.
4. Loss <=1 pp is a point-estimate pass. A stronger statistical claim requires
   the one-sided 95% upper bound on loss <=1 pp. A 1000-image pilot alone is
   not automatic evidence for a tight one-point bound.
5. On dequantized tensors/logits calculate MAE, RMSE, maximum absolute error
   and cosine similarity (undefined for zero norms). SQNR is optional. Widen
   integer values before subtraction to avoid error-metric overflow.
6. Count true pre-clamp clipping events separately for input/output quantizers,
   partial tiles and merges, with denominators. Stored endpoint occupancy at
   -128/127 is only a clipping proxy, not proof of a clipping event.
7. Require zero mismatches at exact integer boundaries against each variant's
   own golden contract. Differences between contracts are algorithmic error.
8. Verify deployed outputs against desktop on fixed tensors. For final deployment
   accuracy, evaluate on HPS or prove execution equivalence. If only a subset
   is checked on HPS, state that scope and keep desktop accuracy labelled.
9. Report attempted/completed/failed images. Do not silently exclude failures:
   give overall success-and-correct rate and label any successful-only accuracy.

**Deliverable:** paired final accuracy table, uncertainty, predictions and
runtime/integer correctness evidence. Freeze tuning before the final run.

## 8. Measure full-model performance and memory on HPS

Keep accuracy, uninstrumented latency, operator profiling and resource diagnostics
as separate runs. Save outputs after timing; avoid tensor dumps/logging in the
headline timed path.

1. Set batch=1, threads=1, and record affinity/frequency policy. Repeat with two
   threads if supported under matching conditions. Record actual kernel fallbacks.
2. Load once. Measure model load and first inference separately. Distinguish a
   cold process with warm filesystem cache from cold storage/cache conditions.
3. Use the same fixed representative image/tensor list for every variant. Have
   inputs ready for model-only timing; do not preload all validation data into RAM.
4. Start with 3 warmups/20 calls to budget runtime. Target at least 10 warmups/
   100 individual calls and three independent sessions when practical. Inspect
   whether warmup actually reaches steady state.
5. Time input-tensor-ready to logits-ready with a monotonic clock. Save every
   call; report mean/median/p95/min/max/stddev/CV, N and percentile convention.
6. Measure throughput as completed images / measured service wall time, specifying
   the boundary. Do not call 1000/median_ms measured sustained throughput.
7. Measure preprocessing/postprocessing separately and directly time the full
   decoded-image-ready to class-ready application. Do not sum medians or p95s.
   File decode/I/O and camera capture belong in separately labelled pipeline tests.
8. Run shortlisted settings continuously (e.g. 5 minutes, longer if stability
   requires it). Report completions, failures, memory growth, first/last-window
   latency and available frequency/temperature evidence.
9. Measure deployment file bytes, peak process RSS, steady RSS after load and
   activation/workspace allocations if exposed. Distinguish process-lifetime
   peak from inference-phase memory. Use GNU `/usr/bin/time -v` if available,
   or live-process `VmHWM`/`VmRSS`; BusyBox time may differ. MiB=bytes/2^20.
10. Record CPU time/utilization, major page faults and swap behavior in a separate
    diagnostic run. Define whether 100% utilization means one core or the board.

Use the new ncnn runner for model-only HPS measurements after ARM deployment.
Integrate board image preprocessing separately for application latency;
`run_cpu_profile.py` remains a matrix-only tool.

**Deliverable:** per-inference samples, model/app latency, throughput, load time,
memory and stability for FP32 and supported quantized CPU graphs.

## 9. Profile operators and choose offload candidates

1. Use runtime profiling or non-overlapping execution timers in a separate pass
   on the same inputs/configuration. Record fused operators as actually deployed.
2. Record operator/type, shapes/layouts, dtype, calls/image, self-time, aggregate
   ms/image, MACs, tensor bytes, native INT8/fallback and offload eligibility.
3. Group conv/patch merging, QKV/projection, attention products, MLP, nonlinear/
   normalization/residual work, and quantization/layout/copies.
   Also provide a stage view: stem, Stage 1 (2 MBConv), transition 1->2,
   Stage 2 (2 blocks), transition 2->3, Stage 3 (6 blocks), transition 3->4,
   Stage 4 (2 blocks), pooling, head LayerNorm and classifier. Assign transitions
   once; do not count them both inside stage totals and as additional work.
4. Calculate shares within that instrumented run using a consistent denominator;
   include unaccounted time and profiler overhead. Do not double-count parent
   and child durations or divide instrumented times by unrelated wall times.
5. Use measured eligible time fraction f, eligible-compute speedup r, and added
   overhead h normalized to original CPU latency:

```text
predicted speedup = 1 / ((1-f) + f/r + h)
```

This is an estimate until hardware is measured. Include packing, tiling,
transfers, cache maintenance and control in h. MAC share is not time share.
With zero overhead and infinitely fast eligible work, speedup is bounded by 1/(1-f).

**Deliverable:** operator time-share table and justified offload shortlist.

## 10. Metrics required in the report

“Required” means needed for the complete CPU baseline/quantization comparison.
Optional diagnostics are useful if supported, not prerequisites.

| Category | Required | Optional / conditional |
|---|---|---|
| Reproducibility | Board/OS/ABI, clock/governor availability, affinity, threads, compiler/runtime/source versions, hashes, preprocessing, splits/calibration config, timing/sample policy, raw archive | Continuous clocks/temperature |
| Correctness | Integer mismatches/test counts, FP32 export/HPS errors, completed/failed runs | Extra random/boundary transactions for hardware validation |
| Classification | Top-1/top-5, N, loss vs FP32 in pp, accuracy/difference intervals, prediction agreement, failures | Per-class accuracy, confusion analysis; ECE if confidence calibration matters |
| Quantization | Scale/zero point/granularity/range, bias/nonlinear policy, MAE/RMSE/max/cosine, clipping rates, q16 representability, native INT8/fallback coverage | SQNR, sensitivity and calibration-size sweeps |
| Kernel speed | Real shapes/layout/calls/contract, implementation, mean/median/p95/min/stddev/CV, N/warmups/grouping, useful/executed MACs, GMAC/s, packing scope | Cycles, instructions, cache misses, bandwidth diagnostics |
| Model speed | Same latency statistics plus max; images/s and scope, batch/threads, first inference/load, preprocessing/postprocessing, direct app latency, session variation | p99 with enough samples, extra batch/resolution settings |
| Operator profile | Shapes/dtypes/calls, self-time/share, MACs, copies/quantization, fallback/eligibility, overhead/unaccounted time | Memory traffic and roofline analysis |
| Memory/storage | File size, peak/steady RSS, sustained memory trend, swap/fault behavior | Activation/workspace peak, allocator detail |
| Stability | Sustained duration, completions/failures, latency drift, available clock/thermal evidence | Longer endurance runs |
| Power/energy | Required only for energy claims: instrument/boundary, idle/active W, J/image | External board DC-input measurement |

The old kernel may not expose usable `perf` counters; report unavailable counters
honestly. CPU utilization is not power. Whole-board DC power is not CPU-only
power. If measuring energy, integrate power over a timed multi-image interval
and divide by completed images; label total versus idle-subtracted energy and
define the idle state. Do not replace measurements with TDP estimates.

### Final tables to fill

**A — Kernel baseline**

```text
case | synthetic/real | K,N,M/layout | contract | implementation | threads |
calls/sample | N | mean/median/p95 ms | stddev/CV | useful/executed MACs |
GMAC/s | pack+compute ms | mismatches | run ID
```

**B — Accuracy and quantization**

```text
variant | target/runtime | quantized ops | W/A granularity | calibration method/N |
evaluation N | top-1/top-5 % | top-1 loss pp + paired CI | FP32 agreement % |
logit RMSE | clipping summary | failures | run ID
```

**C — HPS performance/resources**

```text
variant | native INT8/fallback coverage | threads | timed N | mean/median/p95 ms |
images/s | application median/p95 ms | load/first inference ms |
peak/steady RSS MiB | deployment MiB | sustained failures | run ID
```

**D — Operator profile**

```text
op/group | shapes/dtype | calls/image | self ms/image | time share % |
MACs | tensor bytes | padding/tiling | eligible? | fallback? | run ID
```

**E — Fair comparisons**

```text
comparison | numerator config | denominator config | matched timing scope |
mean-latency ratio | accuracy loss pp | memory/size ratio | caveats
```

**F — Architecture coverage and stage results**

```text
component/stage/block | expected/observed output shape | invocation count |
CPU self ms/image | CPU time share % | boundary error/clipping |
precision/native-kernel status | activation bytes | intended CPU/FPGA placement |
useful/executed MACs | measured/pending | run ID
```

Include every component and transition listed above. Activation bytes at a
boundary are not peak live memory; liveness, residual buffers and runtime
workspace determine the latter. Record block IDs so six distinct Stage 3 blocks
are not mistaken for six timing repetitions of one block.

Useful definitions:

```text
top-1 accuracy (%) = 100 * correct_top1 / evaluated_images
accuracy loss (pp) = FP32 accuracy (%) - variant accuracy (%)
MAE = mean(abs(reference - candidate))
RMSE = sqrt(mean((reference - candidate)^2))
CV (%) = 100 * stddev_ms / mean_ms
GMAC/s = useful_MACs / (mean_ms * 1e6)
quantization speedup = HPS_FP32_mean_ms / HPS_quantized_mean_ms
FPGA offload speedup = HPS_same_quantized_graph_mean_ms / hybrid_mean_ms
size reduction factor = FP32_deployment_bytes / quantized_deployment_bytes
```

Specify denominator/failure treatment for accuracy. Use the same statistic and
scope in every latency ratio. GMAC/s counts a multiply-accumulate as one MAC;
declare a two-operations/MAC convention if also reporting GOP/s.

Recommended plots: latency by variant/thread count, accuracy versus latency,
accuracy versus model size, operator time shares, per-layer error/clipping.
Label dataset, target and scope; include uncertainty/session variation as available.

## 11. Software and artifacts still needed

The implementation now provides these tools. Full board deployment and real
dataset evaluation remain to be performed by the team:

| Proposed artifact | Responsibility |
|---|---|
| `scripts/model_baseline.py evaluate` | Shared preprocessing, FP32 and Linear-only simulated quantization, predictions and accuracy |
| `scripts/model_baseline.py inspect` | Real shapes, architecture boundaries and selected Linear weights/activations |
| `scripts/model_baseline.py calibrate` | Saved Linear scales/configs for Q1–Q4 numerical comparisons; other operations remain FP32 |
| `software/hps/model_runner/` | ncnn CPU runner source: logits, threads, timing and memory; ARM build/BSP validation required |
| `scripts/summarize_model_results.py` | Accuracy/paired uncertainty, timing statistics and report tables |

Paired accuracy/error calculations are in `scripts/compare_model_runs.py`;
`summarize_model_results.py` collects completed-run summary rows. Standard ncnn
INT8 is a separate runtime configuration, not an implementation of Q3/Q4.

Preserve each run as:

```text
results/raw/<run_id>/
  manifest.json       # environment, hashes, splits, quantization, timing policy
  environment.txt
  predictions.csv     # image ID, label, predictions, success/failure
  latency.csv         # session, image ID, iteration, warmup flag, elapsed ms
  operators.csv       # separate profiling pass
  tensor_errors.csv   # layer, error metrics, clipping counts, scales
  memory.csv
  stdout.log
  stderr.log
```

Hash/version dependency files, model exports and scales. Commit small summaries
and manifests; back up bulky raw data and record its durable location.

## 12. Execution order and completion gates

1. Preserve the kernel report, recover team artifacts, freeze the model and
   dataset setup, and run desktop FP32 on 10 images. Validate all architecture
   boundary shapes and reconcile old `stage1_*` filenames with diagram Stage 2.
2. Demonstrate full HPS FP32 inference and rough memory/latency. Desktop
   calibration/shape capture can proceed independently of board deployment.
3. Evaluate Q1–Q4 numerically; resolve bias/scaling/tile issues. Deploy supported
   candidates and verify their outputs.
4. Finish practical CPU kernel comparison; measure one/two-thread full-model
   timing, memory and operators across the stem, all four stages, transitions
   and head. Expand the three-case workload suite as specified in section 5.
   Use pilots to budget longer runs.
5. Freeze settings, run untouched accuracy evaluation and final latency/stability
   sessions, fill Tables A–F, plot comparisons and archive inputs/results.

Budget accuracy runtime as images * measured seconds/image per variant, plus
I/O. A complete HPS sweep may be long. If only desktop full accuracy plus HPS
subset conformance is feasible, publish that limitation rather than claiming
full board validation.

The CPU baseline is complete when FP32 and the chosen deployable quantized CPU
graph have accuracy, latency distributions, throughput, memory, operator
profiles, reproducibility artifacts and an appropriate optimized kernel reference.
Unsupported variants can retain numerical results with HPS timing unavailable.

FPGA conformance, DMA/packing/control breakdown, achieved clock, Quartus resources,
hardware power and CPU/hybrid speedup belong to the later accelerator report.
They do not prevent completion of the CPU report. Existing >=1.5x end-to-end
and >=2x offload targets remain proposals until measured against the equivalent
practical CPU graph.
