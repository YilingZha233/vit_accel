# Current TinyViT CPU baseline and profiling measurements

Status snapshot: 15 October 2026. This is the current measurement report, not a declaration that the complete baseline is finished.

## Evidence and measurement scope

The matrix-kernel results below come from the locally preserved and audited first board run. Full-model results come from JSON, CSV and terminal output supplied by the team in this conversation. Their raw files currently reside on the Windows/WSL computer; they have not all been independently inspected in this checkout. Rounded values are shown here; preserve raw outputs for exact values.

All board model measurements execute on the Cyclone V **HPS ARM CPU**, using ncnn. No custom FPGA fabric accelerator, DMA, camera capture or overlay is included. Batch size is one. Model latency includes extractor creation, input binding, inference and output extraction using preprocessed inputs. It excludes image decoding, resizing, normalization and file loading outside the timed interval. Report images/s as inference throughput, not camera-to-display FPS. Load and initial inference are separate metrics.

## Configuration

| Item | Current configuration |
|---|---|
| Board / processor | DE1-SoC / Cyclone V HPS, dual-core Cortex-A9, `armv7l` |
| Board software | Linux 3.12; BusyBox 1.20.2; original kernel report records Poky 8.0 / Yocto 1.3 |
| Build host | Windows WSL Ubuntu; Python 3.12 virtual environment |
| ARM build | GCC/G++ 15.2.0; statically linked ARM executable; Cortex-A9, NEON, hard-float target |
| Runtime | ncnn; separate uninstrumented and `NCNN_BENCHMARK=ON` profiling builds |
| ncnn source pin | `c15904bb8918187ec100f1ebe708942af83b84be` in configured workflow; preserve build manifests to verify deployed binary provenance |
| Model | TinyViT-21M, 224×224 RGB, 1,000 classifier outputs |
| Source | Microsoft Cream/TinyViT; checkout `4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9` |
| Checkpoint | `tiny_vit_21m_22kto1k_distill.pth`, saved as `models/checkpoints/tinyvit21m.pth` |
| Checkpoint SHA-256 | `eb20032633663fc3e1e49c1cb21332c11e6804c9f920b77da5f832dad248b897` |
| Preprocessing | RGB, shorter side 256, bicubic resize, center crop 224; ImageNet mean/std |
| Evaluation data | 10 repository-labelled development images; a smoke test, not representative benchmark accuracy |
| Calibration data | 10 additional disjoint images; fish/bird-heavy workflow pilot, not final representative calibration |
| Export | Batch-one window-unrolled graph; export may duplicate shared weights and change scheduling |

The 224-pixel resolution follows the chosen checkpoint/model configuration. A 16×16 systolic array tiles internal matrix dimensions; it does not require every image or feature-map dimension to be divisible by 16.

## A. Single-thread matrix-kernel baseline

Synthetic signed INT8 inputs, INT32 accumulation, simple C implementation, one CPU thread. Matrix convention is W[K,N] × X[N,M]. Three warmups and 20 samples per contract; larger cases have one call/sample. Timing includes arithmetic and requantization, excludes input loading, allocation, bias and the rest of the model.

| Case | K,N,M | Arithmetic contract | Mean ms | Median ms | p95 ms | CV % | Useful GMAC/s |
|---|---|---|---:|---:|---:|---:|---:|
| QKV, one window | 576,192,49 | Full accumulation | 22.774 | 22.768 | 22.814 | 0.1026 | 0.237949 |
| QKV, one window | 576,192,49 | RTL-compatible | 53.464 | 53.479 | 53.643 | 0.2913 | 0.101359 |
| MLP FC1 | 768,192,784 | Full accumulation | 619.339 | 619.096 | 621.842 | 0.1412 | 0.186660 |
| MLP FC1 | 768,192,784 | RTL-compatible | 1130.116 | 1130.260 | 1132.105 | 0.1155 | 0.102295 |
| MLP FC2 | 192,768,784 | Full accumulation | 971.233 | 971.128 | 971.404 | 0.0459 | 0.119030 |
| MLP FC2 | 192,768,784 | RTL-compatible | 1137.519 | 1137.588 | 1140.215 | 0.1809 | 0.101629 |

All 22 case/contract combinations passed elementwise golden verification. The audit covers 440 report samples and 66 pilot samples across 11 cases and two contracts. Full accumulation requantizes after the complete dot product; RTL-compatible arithmetic requantizes/clips each 16-element partial sum and merges with saturation. They are different numerical contracts, not interchangeable implementations of identical arithmetic.

These shapes describe selected operations in the 192-channel transformer stage (the proposed architecture's Stage 2). Do not sum these times into model latency. Real captured tensors, executed/padded MAC counts, packing-inclusive performance and an optimized equivalent-contract CPU comparison remain pending. Detailed evidence: [first board report](hps_board_run1.md), [numeric audit](hps_board_run1_audit.json), raw run `results/raw/hps_board_run1/`.

## B. Full-model accuracy and quantization

### Numerical schemes on WSL

Q1–Q4 simulate quantization of 40 selected QKV/projection/MLP Linear modules. Convolutions, attention matrix products, normalization, nonlinearities and classifier remain FP32; bias is added in FP32 after dequantization. These are numerical experiments, not native integer latency tests.

| Variant | Arithmetic | Evaluation N | Top-1 % | Top-5 % | FP32 prediction agreement % | Top-1 loss pp | Logit MAE | Logit RMSE | Max absolute logit error | Mean cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | FP32 reference | 10 | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 1 |
| Q1 | Symmetric INT8; per-tensor W/A scales; full sum, ties-to-even | 10 | 100 | 100 | 100 | 0 | 0.158008 | 0.237563 | 2.273077 | 0.959372 |
| Q2 | Per-output-channel W scales; per-tensor A; full sum, ties-to-even | 10 | 100 | 100 | 100 | 0 | 0.157725 | 0.243796 | 2.527555 | 0.960554 |
| Q3 | Per-channel W; Q16 multiplier, full-sum floor requantization | 10 | 100 | 100 | 100 | 0 | 0.377983 | 0.523509 | 4.848175 | 0.801348 |
| Q4 | Q16 floor requantization/clipping per 16-term tile; saturated merge | 10 | 0 | 0 | 0 | 100 | 1.148421 | 1.481535 | 13.243314 | 0.007853 |

Q1–Q3 and native INT8 top-1 Wilson 95% interval: 72.2467–100%. Q4: 0–27.7533%. Empirical paired-bootstrap loss intervals are [0,0] pp for Q1–Q3 and [100,100] pp for Q4. These degenerate intervals from ten images do not bound unseen rare errors or establish population accuracy. FP32 allclose fails for all quantized variants; exact FP32 equivalence is not their acceptance criterion.

| Diagnostic | Q3 | Q4 |
|---|---:|---:|
| Input clipping | 54 / 56,636,160 (0.000095%) | 9,486 / 56,636,160 (0.016749%) |
| Output clipping | 46 / 72,817,920 (0.000063%) | 0 / 72,817,920 |
| Tile clipping | Not applicable | 0 / 1,977,937,920 |
| Merge clipping | Not applicable | 67,466 / 1,977,937,920 (0.003411%) |
| Largest reported local Linear RMSE | 1.963045, `layers.2.blocks.2.mlp.fc2` | 64.742903, `layers.3.blocks.0.mlp.fc2` |

Q4 is rejected on this pilot. No tile clipping occurred, but merge clipping and large local errors occurred. Repeated floor-rounding can accumulate bias; its contribution versus saturation has not been isolated experimentally. Local errors compare a quantized Linear with FP32 on the same propagated input, not with the original full-model FP32 boundary. A 16×16 array can retain wide partial sums and requantize once; tile size does not mandate INT8 partial-sum storage.

Run IDs: `F0_quant_pilot10`, `Q1_pilot10` through `Q4_pilot10`; calibration `calibration_pilot10`; manifest `data/manifests/quant_pilot.json`.

### Native HPS output comparisons

| Candidate | N | Top-1/top-5 % | Agreement % | Logit MAE | Logit RMSE | Maximum error | Mean cosine |
|---|---:|---|---:|---:|---:|---:|---:|
| FP32, 1 thread | 10 | 100 / 100 | 100 | 0.000001087 | 0.000001543 | 0.000012994 | ≈1 |
| FP32, 2 threads | 10 | 100 / 100 | 100 | 0.000001087 | 0.000001543 | 0.000012994 | ≈1 |
| Native INT8 pilot, 1 thread | 10 | 100 / 100 | 100 | 0.177630 | 0.246531 | 2.204901 | 0.947904 |
| Native INT8 pilot, 2 threads | 10 | 100 / 100 | 100 | 0.177630 | 0.246531 | 2.204901 | 0.947904 |

Reference: saved FP32 logits in `models/exports/smoke10`. FP32 passed atol=1e-4, rtol=1e-3; native INT8 did not, as expected for changed numerical outputs. Each native evaluation completed ten images without failure. Matching aggregate thread metrics do not independently prove byte-identical logits.

Native INT8 uses ncnn KL calibration and its own conversion arithmetic. It is a separate configuration from Q1–Q4 and is not an implementation of Q4. Conversion succeeded for 12 Convolution, 15 ConvolutionDepthWise, 100 Gemm and 1 InnerProduct layers (128 total). These are conversion events, not verified native kernel execution coverage. W/A granularity, actual runtime precision and fallbacks require source/graph/runtime verification.

## C. HPS model performance and resources

### Uninstrumented timing pilots

All rows: batch 1, three warmups, 20 timed inferences, zero failures. There is also a separately timed initial inference. Percentiles use nearest rank; standard deviation is population standard deviation. CV below is expressed as percent, not the raw JSON ratio.

| Variant / run ID | Threads | Mean ms | Median ms | p95 ms | Min ms | Max ms | Stddev ms | CV % | Inference images/s | Service-loop images/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 / `fp32_t1_pilot` | 1 | 7640.891425 | 7640.523615 | 7643.868170 | 7638.380150 | 7644.290460 | 1.538499 | 0.020135 | 0.130875 | 0.130874 |
| FP32 / `fp32_t2_pilot` | 2 | 4480.743421 | 4480.801765 | 4482.645180 | 4477.700120 | 4483.223620 | 1.406738 | 0.031395 | 0.223177 | 0.223174 |
| INT8 / `int8_t1_pilot` | 1 | 5072.312036 | 5071.277160 | 5078.608460 | 5069.766940 | 5080.409160 | 2.710072 | 0.053429 | 0.197149 | 0.197146 |
| INT8 / `int8_t2_pilot` | 2 | 3198.891453 | 3199.010095 | 3200.854050 | 3195.767940 | 3201.132110 | 1.438726 | 0.044976 | 0.312608 | 0.312602 |

| Variant / threads | Load ms | Initial inference ms | Peak process RSS MiB | RSS after load KiB | RSS at end KiB | Service-loop duration s |
|---|---:|---:|---:|---:|---:|---:|
| FP32 / 1 | 731.18020 | 7663.66983 | 154.605469 | 115588 | 157780 | 152.819072 |
| FP32 / 2 | 668.36103 | 4488.87579 | 154.621094 | 115604 | 157796 | 89.616181 |
| INT8 / 1 | 389.97903 | 5101.13714 | 78.085938 | 40408 | 79424 | 101.447489 |
| INT8 / 2 | 374.50490 | 3213.24362 | 78.105469 | 40428 | 79444 | 63.979088 |

Peak RSS covers the process lifetime; loaded/end RSS are snapshots, not a steady-state trace. Load times depend on filesystem/cache conditions and are not a controlled cold-load comparison. Benchmark accuracy counters are zero because accuracy is not evaluated in bench mode, not because predictions failed.

### Evaluation and smoke-run timing (not headline benchmarks)

| Run ID | Threads | Evaluated images | Mean ms | Peak RSS MiB | Top-1/top-5 correct |
|---|---:|---:|---:|---:|---|
| `fp32_one_image` | 1 | 1 | 7655.556640 | 148.875000 | 1 / 1 |
| `fp32_check10` | 1 | 10 | 7643.937343 | 149.453125 | 10 / 10 |
| `fp32_check10_t2` | 2 | 10 | 4482.191767 | 149.468750 | 10 / 10 |
| `int8_one_image` | 1 | 1 | 5106.950750 | 72.894531 | 1 / 1 |
| `int8_check10_t1` | 1 | 10 | 5073.209625 | 72.933594 | 10 / 10 |
| `int8_check10_t2` | 2 | 10 | 3201.396995 | 72.953125 | 10 / 10 |

### Deployment storage

Combined `.ncnn.param` + `.ncnn.bin`: FP32 114,121,431 bytes (108.834678 MiB); native INT8 37,439,950 bytes (35.705519 MiB). INT8 is approximately 67.2% smaller, or 3.048× smaller. Executable, input images, logs and checkpoint files are excluded. Exported weight duplication means this is deployment graph storage, not the original checkpoint size.

### CPU utilization

`int8_t2_cpu_usage`, measured using BusyBox `/usr/bin/time -v`, same 3 warmups / 20 timed-call settings:

| Metric | Observed |
|---|---:|
| User CPU seconds | 143.90 |
| System CPU seconds | 0.53 |
| Elapsed seconds | 79.63 |
| Average process CPU utilization | 181.38% (BusyBox displayed 181%) |
| Equivalent occupied cores | 1.814 |
| Fraction of two-core capacity | 90.69% |
| Major / minor page faults | 0 / 28,721 |
| Voluntary / involuntary context switches | 2,487 / 1,113 |
| Swaps / exit status | 0 / 0 |

This includes loading, initial inference, warmups and timed execution. It is not inference-only or per-core sampled utilization. FP32 utilization has been requested but no result supplied yet. BusyBox reported maximum RSS as 319,920 kbytes, inconsistent with the runner's ~78 MiB. A unit-reporting issue is suspected but not verified; exclude this BusyBox memory field from comparisons pending same-run validation.

## D. Operator profiling

- **Gemm:** activation × trained weights, such as QKV projections and MLP Linear layers.
- **MatMul:** activation × activation, such as **Q × Kᵀ** and **softmax(scores) × V** in attention.

However, those are **roles in this model**, not universal definitions of the operator names. Confirm individual layers in the exported graph before assigning FPGA operations.

Two-thread instrumented ncnn runs, five measured images after one initial and three warmup passes. Each image has 899 distinct logged layers and 901 calls in the FP32 summary; two Split layers run twice. Deployed-layer execution intervals include fused internal work and are not primitive self-times. Printed resolution is 0.01 ms.

| Operator group | FP32 ms/image | FP32 logged share % | INT8-model ms/image | INT8 logged share % |
|---|---:|---:|---:|---:|
| Gemm | 1941.954 | 43.268 | 867.694 | 27.024 |
| GELU | 868.838 | 19.358 | 986.400 | 30.721 |
| Convolution | 670.606 | 14.942 | 279.342 | 8.700 |
| MatMul | 222.316 | 4.953 | 226.130 | 7.043 |
| BinaryOp | 171.178 | 3.814 | 171.972 | 5.356 |
| Softmax | 134.868 | 3.005 | 135.010 | 4.205 |
| Reshape | 122.042 | 2.719 | 158.180 | 4.927 |
| Permute | 105.996 | 2.362 | 105.912 | 3.299 |
| ConvolutionDepthWise | 71.618 | 1.596 | 101.008 | 3.146 |
| Crop | 63.290 | 1.410 | 64.020 | 1.994 |
| MemoryData | 38.090 | 0.849 | 38.706 | 1.205 |
| LayerNorm | 34.074 | 0.759 | 34.470 | 1.074 |
| Slice | 21.218 | 0.473 | 21.220 | 0.661 |
| Concat | 9.668 | 0.215 | 9.734 | 0.303 |
| Split | 9.636 | 0.215 | 9.770 | 0.304 |
| InnerProduct | 2.622 | 0.058 | 1.060 | 0.033 |
| Reduction | 0.160 | 0.004 | 0.160 | 0.005 |

| Profiling validation | FP32 | INT8 |
|---|---:|---:|
| Matched uninstrumented control mean ms | 4478.419010 | 3201.271130 |
| Instrumented mean ms | 4604.569582 | 3327.443924 |
| Observed overhead % | 2.8169 | 3.9413 |
| Logged layer sum ms/image | 4488.174 | 3210.788 (sum of supplied groups) |
| Instrumented wall minus logged sum ms | 116.395582 | 116.655924 (derived) |

Shares use each instrumented run's own logged sum. Wall-minus-layer time is unassigned runtime/logging/timing-resolution work, not a measured stage. The FP32 profile-control load time was 19,428.06 ms, an outlier that reinforces the need for controlled cache conditions when comparing load times.

Gemm improves ~2.24× and Convolution ~2.40×; MatMul is nearly unchanged. GELU and depthwise times increase. Conversion, packing and precision changes inside layer intervals may contribute; causes require investigation. GELU is now the largest operator group. The first ten logged calls do not establish native precision coverage; `*4` / `*8` denote packing, not sufficient dtype evidence.

Largest supplied FP32 individual-layer times: `conv_141` 110.110 ms, `conv_145` 99.696 ms, `conv_143` 99.326 ms, `conv_142` 85.942 ms, `conv_144` 85.848 ms. Full per-layer shapes/calls/timings are in the WSL `layers.csv` files. Source-stage mapping is pending; runtime numbered layer names are not stage identifiers.

## E. Fair comparisons

| Comparison | Numerator / denominator | Mean-latency ratio | Interpretation / caveat |
|---|---|---:|---|
| FP32 thread scaling | FP32 t1 / FP32 t2 | 1.705× | Same graph, batch and timing scope; separate pilot runs |
| INT8 thread scaling | INT8 t1 / INT8 t2 | 1.586× | Same converted graph; separate pilot runs |
| Quantization, one thread | FP32 t1 / INT8 t1 | 1.506× | 33.6% lower latency; numerical arithmetic differs |
| Quantization, two threads | FP32 t2 / INT8 t2 | 1.401× | 28.6% lower latency; 49.5% lower process peak RSS |
| Deployment size | FP32 bytes / INT8 bytes | 3.048× | Param + weights only |

These are pilot comparisons, not repeated-session confidence intervals or FPGA speedups. All schemes' measured accuracy loss is on only ten development images. No optimized CPU baseline with the exact Q4/RTL contract has been timed as a complete model.

WSL FP32 pilot (`F0_wsl_pilot`): one thread, 20 calls, mean 158.815379 ms, median 158.802293 ms, p95 173.925618 ms, stddev 10.721736 ms, CV 6.7511%, 6.296619 images/s. This is different host/runtime execution and must not be presented as an FPGA acceleration comparison.

## F. Architecture coverage

| Component | Expected output, batch 1 | Blocks | Current measured coverage |
|---|---|---:|---|
| Input | [1,3,224,224] | — | Complete-model execution uses this input |
| Two-convolution stem | [1,96,56,56] | 2 convolutions | Runtime convolution profiling; stage mapping pending |
| Stage 1 | [1,96,56,56] | 2 MBConv | Included in full-model timing; boundary validation pending review |
| Stage 2 | [1,784,192] | 2 TinyViT | Included; selected numerical Linear simulations |
| Stage 3 | [1,196,384] | 6 TinyViT | Included; selected numerical Linear simulations |
| Stage 4 | [1,49,576] | 2 TinyViT | Included; selected numerical Linear simulations |
| Global mean pooling | [1,576] | — | Included in graph; stage mapping pending |
| LayerNorm/classifier | [1,1000] | — | Native classifier conversion logged; output comparisons measured |

Expected shapes come from the proposed architecture; this table does not assert independently verified observed boundary tensors. Per-stage invocation counts, timing, activation bytes, boundary error/clipping, executed/padded MACs, native precision and intended CPU/FPGA placement remain to be assembled. Complete-model inclusion is not sufficient evidence for stage-specific correctness or timing.

## Saved evidence locations on WSL

Paths below are relative to `/home/tiff/cpu_baseline_tools` on the Windows WSL machine, not this Mac checkout.

| Evidence | Location |
|---|---|
| FP32 original logits / export | `models/exports/smoke10/` |
| Numerical calibration | `results/raw/calibration_pilot10/calibration.json` |
| Numerical scheme outputs | `results/raw/F0_quant_pilot10/`, `results/raw/Q1_pilot10/` through `Q4_pilot10/` |
| Scheme comparisons | `results/raw/Q1_pilot10_comparison.json` through `Q4_pilot10_comparison.json` |
| Native calibration table/log | `models/exports/native_pilot10.table`, `native_pilot10_calibration.log` |
| Conversion log | `models/exports/native_pilot10_conversion.log` |
| Converted graph | `models/exports/tinyvit_int8_pilot10.ncnn.param` and `.bin` |
| Recovered FP32 pilot | `results/raw/hps_fp32_pilot/results/` |
| Recovered FP32 profile | `results/raw/hps_profile/results/` |
| Recovered INT8 results | `results/raw/hps_int8_pilot/results/` |
| Native INT8 comparisons | `results/raw/hps_int8_pilot/comparison_int8_t1.json`, `comparison_int8_t2.json` |
| Operator reports | `results/reports/fp32_t2_operator_profile/`, `int8_t2_operator_profile/` |
| CPU usage output, on board | `/home/root/vit_model_fp32/results/int8_t2_cpu_usage.time.txt` |

Each operator report contains `layers.csv`, `operator_groups.csv` and `summary.json`. The INT8 UART transfer had one truncated 16-byte line; bytes were reread from the same archive, the decoder required gzip integrity before extraction, and the user subsequently generated the operator summary. Preserve original logs and repaired archive. CPU-usage measurement occurred after that transfer and still needs exporting.

## Remaining measurements and decisions

1. **CPU resources:** collect FP32 CPU utilization under the matching command; review same-run runner RSS versus BusyBox output; add one-thread utilization if needed. Per-core sampling and inference-only CPU utilization remain optional deeper measurements.
2. **Native precision coverage:** verify each eligible operator's converted parameters and actual selected ARM kernel/fallback; quantify coverage by time and MACs where possible. Do not label the complete model as integer-only.
3. **Representative quality:** expand calibration and evaluation, retain disjoint content and frozen labels/hashes, report failures and uncertainty. Ten-image smoke accuracy is not final model accuracy.
4. **Stage profiling:** map ncnn layers to original blocks, validate observed boundaries, aggregate timing/activation bytes, measure boundary error/clipping and useful/executed MACs. Resolve duplicated export operations explicitly.
5. **Arithmetic choice:** keep Q4 failure in the report; isolate floor-rounding versus merge saturation. Evaluate wide accumulators with one final requantization before freezing the accelerator contract.
6. **Repeatability and sustained testing:** repeat independent timing sessions, choose a justified larger run duration/sample count, log CPU frequency/governor, affinity, temperature if available, resource drift and failures. Twenty timed calls and five profile images are pilots.
7. **Application latency:** separately measure decode/resize/normalize + model + postprocess and eventual camera/overlay latency. Current results provide full-model inference latency, not full application latency.
8. **Kernel completion:** benchmark real captured model tensors; record packing/unpacking, padded/executed MACs, layout, mismatches and an optimized same-contract CPU implementation.
9. **FPGA integration:** actual hardware correctness, DMA/cache-maintenance/launch costs, integrated offload latency, resource utilization, clock, power/energy and CPU-versus-FPGA comparisons are unmeasured.
10. **Reproducibility:** archive current raw results, binaries, hashes, exact commands, source/build flags, model/input hashes and environment. Board calendar timestamps are unreliable; use monotonic durations and externally recorded session dates.

Current supported conclusion: the native ncnn INT8 pilot preserves all ten development predictions, reduces two-thread model latency from 4.481 s to 3.199 s, reduces deployed model storage by 67.2%, and reduces benchmark peak process RSS by about 49.5%. GELU is the largest remaining logged operator group. These are HPS CPU pilot measurements, not final accuracy validation or FPGA acceleration results.

## Appended analysis and engineering summary

The following summarizes the supplied analysis of this report. It interprets existing measurements and proposes next steps; it does not introduce new measurements. Chart-interface text and conversational offers from the supplied text are omitted.

### 1. CPU baseline and practical reference

TinyViT-21M successfully runs on the Cyclone V HPS CPU using ncnn. The best measured pilot configuration is native INT8 with two threads: **3.199 seconds/image and 0.313 inference images/s**, compared with FP32 at **4.481 seconds/image and 0.223 images/s**. This is below real-time video throughput.

Two threads provide approximately 1.71× FP32 and 1.59× INT8 speedup over one thread. Native INT8 reduces two-thread latency by 28.6%, peak process RSS by approximately 49.5% (155 to 78 MiB), and model storage by 67.2% (109 to 36 MiB).

Use two-thread native INT8 as the practical runtime CPU performance reference, while retaining FP32 as the numerical reference. An FPGA comparison also needs matched arithmetic or explicitly documented numerical differences and acceptance criteria.

### 2. Bottlenecks and implications for acceleration

| INT8 runtime group | Logged ms/image | Logged share |
|---|---:|---:|
| GELU | 986.4 | 30.7% |
| Gemm | 867.7 | 27.0% |
| Convolution | 279.3 | 8.7% |
| MatMul | 226.1 | 7.0% |
| Softmax | 135.0 | 4.2% |

GELU is the largest remaining logged operator group after conversion. Likely matrix-accelerator targets include Gemm for learned QKV/MLP projections and MatMul for attention products. Confirm each layer's graph mapping before assigning hardware work. Gemm + MatMul account for **34.07% of logged INT8 operator time**.

For illustration, Amdahl's model is:

```text
S = 1 / ((1 - f) + f / s)

f = fraction of original execution time accelerated
s = speedup of that portion
```

Using the logged fraction f ≈ 0.3407 as a proxy gives approximately **1.44× overall speedup** for 10× faster matrix operations, or an illustrative latency of **2.22 seconds** from the 3.199-second pilot baseline. Infinite matrix speed gives approximately **1.52×** in this simplified model.

These are hypothetical estimates, not measured FPGA results or strict wall-time bounds. The fraction comes from an instrumented logged-layer denominator, whereas headline latency comes from an uninstrumented run. The estimates assume unchanged remaining work and ignore DMA, synchronization, conversion and integration overhead. Fusion, GELU optimization, reduced data movement or changed execution paths could alter the accelerated fraction and outcome.

### 3. Quantization and accumulator design

| Numerical scheme | Pilot top-1 | Logit RMSE | Interpretation |
|---|---:|---:|---|
| Q1, per-tensor | 100% | 0.238 | Passed ten-image smoke check |
| Q2, per-channel weights | 100% | 0.244 | Passed ten-image smoke check |
| Q3, Q16 full accumulation | 100% | 0.524 | Passed smoke check with greater logit distortion |
| Q4, Q16 per-tile requantization and saturated merge | 0% | 1.482 | Reject this configuration on the pilot |

The current simulated RTL-compatible Q4 arithmetic substantially degrades predictions. This is evidence about the numerical scheme, not a measured failure of an executed FPGA implementation. Repeated floor-rounding and saturated merging must be isolated to explain their separate contributions.

Preserve the 16×16 array as a hardware target, but evaluate a revised datapath:

```text
INT8 multiplication
→ wide accumulation across all contraction tiles
→ one final requantization
→ output activation
```

A 16×16 systolic array does not require each partial result to be reduced to INT8. INT32 is a candidate accumulator width; verify range bounds for the actual contraction lengths, bias and scaling before freezing RTL. Keep Q4's original results as evidence and assign a new configuration ID to revised arithmetic.

The ten development images do not establish representative ImageNet accuracy for Q1–Q3 or native INT8. Expand calibration and evaluation before accepting final quality claims.

### 4. Separate kernel and integrated comparisons

The synthetic simple-C full-accumulation baseline measured approximately 22.77 ms / 0.238 GMAC/s for one QKV window, 619.34 ms / 0.187 GMAC/s for FC1, and 971.23 ms / 0.119 GMAC/s for FC2. These single-thread results are useful arithmetic references, but are not equivalent to optimized ncnn execution.

Maintain three comparisons:

1. FPGA matrix arithmetic versus an equivalent-contract CPU matrix kernel.
2. Complete FPGA offload, including packing, DMA, cache maintenance, launch/wait and unpacking, versus optimized CPU execution of equivalent work.
3. Integrated ARM+FPGA TinyViT versus ARM-only TinyViT with matched model, inputs, numerical acceptance criteria and timing scope.

Only the third supports a complete-model acceleration claim. Current measurements do not demonstrate FPGA acceleration.

### 5. Engineering priorities

| Priority | Task | Purpose |
|---|---|---|
| P0 | Revise and validate Q4 accumulation/requantization | Establish acceptable hardware numerical behavior |
| P0 | Map runtime layers to TinyViT stages and source operations | Identify exactly what offload replaces |
| P1 | Capture real matrix operands and profile data movement | Establish realistic shapes, ranges and throughput requirements |
| P1 | Measure DMA, packing and launch overhead | Determine whether complete offload is worthwhile |
| P1 | Expand disjoint calibration and evaluation datasets | Assess quality beyond the smoke test |
| P2 | Investigate GELU optimization, conversion overhead and fusion | Address the largest remaining logged group |
| P2 | Repeat independent timing sessions | Establish performance repeatability |

Evaluate the array across transformer Stages 2, 3 and 4 rather than only the 192-channel Stage 2 examples. Shapes, padding, reuse, utilization and memory traffic differ across stages.

### 6. Suggested capstone wording and next milestone

The initial TinyViT-21M CPU baseline demonstrates deployment on the DE1-SoC Cyclone V ARM HPS using ncnn. Native INT8 conversion reduces two-thread inference latency from 4.481 seconds to 3.199 seconds, peak process RSS by approximately 49.5%, and deployed model storage by 67.2%. Preliminary profiling identifies GELU and Gemm as the dominant remaining runtime groups. Numerical experiments show that the tested per-tile requantization and saturated-merge scheme can severely degrade predictions, motivating evaluation of wide accumulation with final requantization. These results establish a CPU pilot reference; representative accuracy, verified precision coverage, FPGA execution, transfer overhead and integrated acceleration remain to be evaluated.

The next hardware milestone is to demonstrate **one real TinyViT matrix operation executing correctly and faster on FPGA after including data movement and control costs**, then extend the design based on measured stage-level benefit. Accumulator arithmetic is as important as raw MAC throughput, and matrix-only acceleration should be assessed against the measured remaining CPU workload.
