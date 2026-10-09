# CPU baseline report: scope, audit and fill-in template

Prepared 2026-10-09 after reviewing the supplied presentation.
This is a measurement specification and report template, not completed HPS results.

## Resolve existing evidence first

The presentation selects TinyViT-21M (page 2), reports GEMM=498 ms and
DMA=1.4–6.4 ms with roughly 54 MACs/byte (page 5), and marks full HPS inference
and profiling completed (page 7). Those implementation artifacts and raw
measurements are absent from the local baseline starter. They may exist with
another teammate. Obtain them before recreating the runtime.

The team has now confirmed TinyViT-21M with INT8; earlier 5M suggestions are
superseded. The standalone integer kernels are model-independent, but layer
dimensions and model latency are not. Freeze resolution, checkpoint,
preprocessing, runtime and the complete quantization rules before the report.

See [measurement_steps.md](measurement_steps.md) for the executable workflow.
The subsequent implementation adds full-layer vector generation, raw samples,
grouped timing, mean/dispersion, and a CPU suite runner; those items in the
original audit below are now addressed. Full HPS model inference remains absent.

For 498 ms, recover: K/N/M, batch and call count, dtype, CPU target/clock,
implementation/library, compiler flags, thread count, warmups, repetitions,
raw timings, and whether copies/packing are included. For DMA recover byte
count, direction, bridge/path, burst size, cache operations, allocation method,
and setup/completion boundaries. Compare matched workloads only.

## Three measurement levels

1. Kernel: one Y[K,M]=W[K,N]X[N,M], with the specified quantization semantics.
   Run on HPS ARM Linux, data in CPU-accessible DDR, no FPGA offload.
2. Block: one complete attention or MLP block, including its norms, biases,
   nonlinearities, layout transforms and additions as defined by the real model.
3. Model: a complete saved-image inference. Report tensor-ready-to-logits model
   time, plus preprocessing separately. Camera and filesystem I/O are optional
   extra application measurements. Model loading is separate from steady-state.

Keep the faithful mathematical reference and the practical optimized performance
baseline distinct. Use equivalent graphs/arithmetic for CPU/hybrid comparisons.
FP32 CPU -> quantized CPU measures quantization/runtime gains; quantized CPU ->
hybrid measures offload gains. Current RTL clips each contraction tile; evaluate
this separately from ordinary full-INT32-sum requantization.

## First real-layer workloads

Notation matches the accelerator: K=output rows, N=contraction, M=columns.
Examples below derive from the official 224x224 model source, for the first
attention stage after the convolutional stage. Confirm against the frozen model.

| Operation | TinyViT-21M K,N,M | TinyViT-5M K,N,M | Scope |
|---|---|---|---|
| QKV projection | 576,192,49 | 384,128,49 | One 7x7 window, GEMM only |
| Attention output projection | 192,192,49 | 128,128,49 | One window, GEMM only |
| MLP FC1 | 768,192,784 | 512,128,784 | Whole 28x28 stage map, GEMM only |
| MLP FC2 | 192,768,784 | 128,512,784 | Whole stage map, GEMM only |
| QK^T | 49,32,49 | 49,32,49 | One head in one window |
| Attention probabilities times V | 49,49,32 | 49,49,32 | One head; arithmetic must be defined |

For Linear, model activation [tokens,in_features] is transposed to X[in_features,
tokens], and W is [out_features,in_features]. QKV above occurs per window; MLP
uses all tokens. Count windows, heads and blocks when projecting total cost.
Do not assume a profiler reports one call for each logical window: implementations
may batch windows into one operation.

For CPU, preserve the real N=49 in attention-times-V. Hardware currently needs
zero padding to 64; include padding and wasted MACs in its offload accounting.
Attention probabilities and projections do not automatically share one INT8
quantization scheme. The supplied starter does not implement this conversion.

These full layer shapes exceed current on-chip buffers. CPU can compute them
directly; FPGA needs multiple calls/tiles. Do not multiply a warm-cache 16x16
microbenchmark time by tile count and call that measured full-layer latency.

## Local folder audit

Present: integer arithmetic references, eight small cases, bank images, CPU C
benchmark, elementwise output comparison, manifests, protocol, desktop checks.

Missing for the complete report:

- Confirmed model variant/checkpoint and runnable HPS full-model application.
- Exact real-layer shapes and captured representative activations/weights.
- Full-layer vector generation independent of FPGA capacity validation; current
  generator always packs into hardware banks and rejects oversized layers.
- FP32 and practical optimized quantized HPS baselines (current C is simple).
- Grouped timing for tiny kernels, raw per-sample times, mean and dispersion.
  Current C stores timings internally but emits only median/p95/min summaries.
- Stage/operator profiler, call counts and total runtime shares.
- Full-model/app timing, preprocessing time, peak RSS and accuracy evaluation.
- Actual HPS/FPGA/DMA logs; existing local checks are desktop software results.

## Short first pass

1. Collect existing team artifacts and freeze one experiment configuration.
2. Compile/run the eight small cases on HPS to establish the correctness path.
3. Benchmark one QKV and the two MLP GEMMs at real model dimensions with
   reasonable CPU code. Use captured tensors or clearly labelled synthetic data.
4. Reuse the working model to time a complete inference, then separately profile
   stage/operator contributions. Begin with 3 warmups and 20 measurements for
   a pilot; use 10 warmups/100 measurements when affordable for final statistics.
5. Export logits for a small fixed image set and compare with the model reference.
6. Fill the tables below and label incomplete items pending.

Kernel timing execution should be minutes for a small selected suite, once the
board/toolchain works; this is an expectation to check with a pilot, not a
guarantee. With runnable team code, organizing the first report is plausibly a
half-day to one-day task. Without a working HPS model runtime, porting/building
that runtime can dominate and cannot be promised to finish in that time.

Estimate run time from one pilot: total seconds ~= (warmups+repetitions)*seconds
per call, summed over configurations. A 0.498 s kernel takes about 54.8 seconds
for 10+100 calls, excluding setup. A 5-second model takes about 115 seconds for
3+20 calls. These are examples, not measured current-board performance.

For microsecond kernels, time L consecutive calls in a sample and divide by L.
Choose L so the sample lasts roughly >=10 ms. Label percentiles of these samples
as percentiles of group averages, not true individual-call tail latency.
Retain individual-run timing for full inference. Keep validation and file I/O
outside kernel timing, and consume outputs to prevent compiler elimination.

## Required metric definitions

| Metric | Definition / purpose |
|---|---|
| Mean, median, p95 latency | Mean for throughput/cost, median typical, p95 variability |
| Stddev or CV | Noise/stability; CV=stddev/mean |
| MAC count | K*N*M for one GEMM; count repetitions/windows/heads separately |
| Effective GMAC/s | K*N*M / (time_ms * 1e6); say one MAC is counted as one MAC |
| Runtime share | Non-overlapping aggregate operation time / measured model time |
| Peak RSS | Maximum process resident memory, not checkpoint file size |
| Exact mismatch count | Number of output elements unequal to integer golden |
| Max/mean absolute error | Compare tensors in the same quantized or dequantized units |
| Saturation rate | Fraction clipped; instrument tile and final clipping separately |
| Top-1 agreement | Same predicted class as reference on the same inputs |
| Labelled top-1 accuracy | Correct against ground truth; distinct from agreement |
| CPU configuration | CPU clock/governor, threads, affinity if used, OS, runtime, flags |

Profile conv, QKV/projection, both attention products, MLP FC1/FC2, softmax,
normalization, GELU, residuals, layout/copies and preprocessing. Do not sum
inclusive parent-module times with their child-operation times. Measure total
latency again with profiling instrumentation disabled.

A lower-bound operand traffic estimate is sW*K*N+sX*N*M+sY*K*M bytes, where s
is bytes/element. Actual offload traffic includes tile reloads, padding and
packing. Arithmetic intensity=MACs/actual bytes at a clearly named boundary.
MAC/s and FLOP/s are not interchangeable without a stated counting convention.

## Report tables (fill with measurements)

### A. Experiment identity

Model/checkpoint/hash: PENDING. Board/clock/kernel: PENDING. Runtime/flags/threads:
PENDING. Input/preprocessing and precision: PENDING. Warmup/sample policy: PENDING.

### B. Kernel profile

| Operation | K,N,M | Dtype/contract | Calls/image | Median ms | p95 ms | GMAC/s | Mismatches |
|---|---|---|---|---|---|---|---|
| QKV | pending | pending | pending | pending | pending | pending | pending |
| FC1 | pending | pending | pending | pending | pending | pending | pending |
| FC2 | pending | pending | pending | pending | pending | pending | pending |

### C. Full-model breakdown

| Category | Aggregate time/image | Runtime share | Offload candidate? |
|---|---|---|---|
| Convolutions | pending | pending | pending |
| QKV + attention output projection | pending | pending | yes, subject to overhead |
| QK^T + attention-times-V | pending | pending | requires quantization definition |
| MLP FC1 + FC2 | pending | pending | yes, subject to overhead |
| Softmax/norm/GELU/residual/layout | pending | pending | initially CPU |
| Uninstrumented full model | pending | 100% | — |
| Preprocessing (outside model total) | pending | separate | CPU |

Also report peak RSS, images/s from total completed images/elapsed time,
output comparisons, and the exact baseline implementation quality.

### D. Numerical comparison

| Comparison | Required evidence |
|---|---|
| Python integer -> C | Exact match; current starter covers small cases |
| C golden -> FPGA | Exact match under frozen arithmetic; not yet measured |
| Full sum -> tile-clipped arithmetic | Tensor error and saturation rates |
| FP32 -> quantized model | Logit error, prediction agreement, labelled accuracy |

## Answering the professor with targets

Propose >=1.5x end-to-end speedup over the same quantized graph on HPS and >=2x
for selected operations INCLUDING offload overhead. They are engineering targets
pending measurements, not literature guarantees. Accuracy goal: <=1 percentage
point top-1 loss versus the chosen FP32 checkpoint on the same held-out labelled
set; a tiny demo set cannot establish that bound. Use a larger validation set
and report uncertainty for the final accuracy claim.

The official table lists 21M at 4.3G MACs, 5M at 1.3G MACs, both at 224x224.
A 16x16 array at an assumed 100 MHz has 25.6 GMAC/s ideal peak. For 21M,
4.3/25.6 ~= 0.168 s is an idealized all-MAC compute estimate, before software,
transfers and utilization losses. This is neither achieved latency nor a strict
bound on every hybrid architecture, since CPU work can overlap. Do not promise
30 FPS from array size alone.

Use Amdahl's estimate S=1/((1-f)+f/r+h): measured eligible CPU fraction f,
accelerated compute factor r, normalized added overhead h. At f=.7,r=4,h=.1,
S~=1.74. At f=.3, even infinite compute acceleration and no overhead gives 1.43.
Thus profile f before promising 1.5x. If the measured model baseline is T ms,
the 1.5x target is <=T/1.5 ms, with separately stated p95 behavior.

The slide's 498 ms is a GEMM measurement, not established whole-model latency.
1.4–6.4 ms is 0.28–1.29% of 498 ms but 2.8–12.9% of a hypothetical 49.8 ms
accelerated compute time. The relative overhead grows as compute speeds up.
Packing/control/output transfers may add more; quantify the complete path.

## Suggested 4-page report

1. Scope, reproducibility and baseline definition; reconcile slide claims.
2. Kernel table plus non-overlapping full-model latency breakdown chart.
3. Correctness methodology, numerical-error results and pending hardware tests.
4. Target derivation, Amdahl estimate, next integration tasks and limitations.

Sources: supplied presentation, local source audit, and Microsoft's official
[TinyViT model table](https://github.com/microsoft/Cream/tree/main/TinyViT) and
[model implementation](https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py).
