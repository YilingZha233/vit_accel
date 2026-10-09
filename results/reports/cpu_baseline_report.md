# TinyViT-21M on DE1-SoC: CPU Baseline and Acceleration Evaluation Plan

**Status:** CPU matrix-kernel baseline measured; full-model inference and FPGA evaluation pending.  
**Evidence:** HPS board run 1, including raw timing samples, compiler logs and environment records.  
**Purpose:** Establish a reproducible reference and define concrete correctness and performance goals for the hybrid CPU–FPGA prototype.

## Executive summary

This project aims to accelerate TinyViT-21M image classification on the DE1-SoC by moving selected matrix operations from its ARM processor to custom FPGA hardware. A useful evaluation must establish both that the accelerator computes the intended answers and that its benefit exceeds the cost of transferring and managing data.

We have completed the first measurement stage: a single-thread C benchmark executed on the board's ARM Cortex-A9 processor. Eleven synthetic matrix cases were tested under two explicit arithmetic rules. All 22 case/rule combinations reported exact agreement with their corresponding golden outputs. An independent audit reconciled 440 report timing samples and 66 pilot samples with their summaries.

For the CPU routine that reproduces the current accelerator's arithmetic, median times were **53.479 ms for one-window QKV projection, 1,130.260 ms for MLP FC1, and 1,137.588 ms for MLP FC2**. These are selected matrix-operation timings, not complete image-inference times. [Measured results](../raw/hps_board_run1/report/summary.csv)

The proposed acceptance goals are:

- **Numerical correctness:** zero mismatched integer output elements between FPGA and CPU under the same frozen arithmetic contract.
- **Operation performance:** at least **2× median speedup** for each selected complete matrix offload, including data movement and control overhead.
- **System performance:** at least **1.5× median end-to-end inference speedup** over an equivalent HPS-only quantized pipeline, with no p95 latency regression.
- **Model quality:** no more than **1 percentage point of top-1 accuracy loss** relative to the selected FP32 reference on the same held-out labelled evaluation set.

These are proposed engineering acceptance criteria. FPGA correctness, speedup and model accuracy have not yet been demonstrated.

## 1. System context and the role of the baseline

The DE1-SoC combines a **hard processor system (HPS)**, which runs ordinary software on ARM CPUs, with an **FPGA**, whose programmable logic can implement a specialized circuit. In the intended hybrid system, the CPU will manage the model and execute operations retained in software. The FPGA will execute selected matrix calculations.

**Direct memory access (DMA)** allows hardware to move blocks of data between memory and accelerator buffers without the CPU copying each element individually. Transfer setup, synchronization and cache handling still consume time. A faster arithmetic circuit is useful only if the complete offload is faster than the CPU operation it replaces.

The baseline is therefore a measured reference, not a claim about the CPU's theoretical maximum performance. This run establishes the behavior of one simple, reproducible C implementation. Final comparisons must also include a reasonably optimized CPU implementation or supported inference runtime with equivalent arithmetic.

## 2. Experimental configuration and scope

| Item | Configuration or evidence |
|---|---|
| Board | DE1-SoC revision G, identified from the board photograph |
| Execution processor | ARM Cortex-A9 / ARMv7; two cores visible, one benchmark thread |
| Operating system | Poky 8.0 / Yocto 1.3; Linux 3.12.0-00307-g507abb4-dirty |
| Compiler | GCC 15.2.0 |
| Recorded flags | `-O3 -std=c11 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard` |
| Operand format | Signed INT8; INT32 dot-product accumulation |
| Output processing | Integer scaling, floor rounding and saturation to INT8 |
| Input data | Fixed synthetic matrices; not captured model weights or image activations |
| Larger-case sampling | Three warmup calls, then 20 separately timed calls per arithmetic rule |
| Small-case sampling | Three warmup calls, then 20 groups of 1,000 calls; times normalized per call |
| Timing mechanism | Monotonic elapsed wall-clock time on the HPS |

Sources: [environment record](../raw/hps_board_run1/report/environment.txt) and [representative build log](../raw/hps_board_run1/report/stage1_mlp_fc1.log).

Each routine is checked against golden output before and after timing. File loading, memory allocation, verification, USB transfer and result-file writing are outside the timed interval. Matrix accesses through the CPU cache/memory hierarchy, arithmetic, output scaling and clipping are inside it. The timed loops also include small loop/result-consumption overhead. The RTL-compatible routine clears its output buffer within the timed call.

The repeated calls reuse the same data. This is a steady-state kernel measurement, not a cold-start, DRAM-bandwidth or multi-image accuracy experiment. NEON target flags do not prove vectorized instructions were emitted. CPU clock/governor, executable hash, affinity and authoritative acquisition date were not captured in this export; they must be recorded in subsequent comparison runs. System memory information is not a measurement of peak process memory.

## 3. Why these matrix workloads were selected

Microsoft's TinyViT model table identifies the 224 × 224 TinyViT-21M variant as approximately 21 million parameters and 4.3 billion MACs. It also reports V100 GPU throughput, which is not a prediction for a Cortex-A9 or DE1-SoC implementation. The table provides workload context rather than a transferable speed target. [Official TinyViT model table](https://github.com/microsoft/Cream/blob/main/TinyViT/README.md)

Our matrix convention is `Y[K,M] = W[K,N] × X[N,M]`: K is the number of output features, N is the number of products summed per output, and M is the number of token columns. A **MAC** is one multiply-accumulate; one matrix call performs K × N × M useful MACs.

| Workload | K, N, M | Shape rationale | MACs per call |
|---|---|---|---:|
| QKV projection, one window | 576, 192, 49 | Three 192-feature query/key/value projections for 7 × 7 tokens | 5,419,008 |
| MLP FC1 | 768, 192, 784 | Expand 192 features by a factor of four across a 28 × 28 token map | 115,605,504 |
| MLP FC2 | 192, 768, 784 | Reduce the expanded 768 features back to 192 for the same token map | 115,605,504 |

The shape choices follow the 224-resolution model's first attention-stage dimensions, windowing and MLP expansion. They are manually generated representative workloads and must be reconciled with the exact checkpoint and deployed graph. QKV measures projection only, not attention-score products or softmax. FC1 and FC2 exclude bias, activation functions and other block operations. [Official model implementation](https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py)

The eight smaller cases test identity behavior, zero inputs, signed random values, multiple contraction tiles, nonmultiple output dimensions, rounding differences and saturation/cancellation. They support arithmetic validation rather than model-FPS estimates.

## 4. Two arithmetic references and why both matter

INT8 values range from −128 to 127. After accumulating products, the benchmark converts a sum back to INT8 using:

```text
requantize(sum) = clamp(floor(sum × q16 / 65536), -128, 127)
```

For the larger cases, `q16=128`, representing a multiplier of 1/512. This is a synthetic test scale, not a calibrated model scale. A wider intermediate is used during the scaling multiplication.

**Full accumulation** sums every product in the dot product before applying this conversion once. **RTL compatibility** sums groups of 16 products, converts each group, and merges the partial outputs with saturation. “RTL” refers to the register-transfer-level circuit description; both routines in this report execute on the CPU.

This distinction matters because early rounding and clipping discard information. For example, two partial sums of 1 at a scale of 0.5 give `floor((1+1)×0.5)=1` under full accumulation, but `floor(1×0.5)+floor(1×0.5)=0` when rounded separately. Early conversion is a current design choice, not an FPGA requirement.

The full-accumulation reference helps evaluate the numerical consequence of that choice. The RTL-compatible reference enables a like-for-like check of the current accelerator and a matched CPU timing comparison. Passing each reference independently does not mean the two references agree, nor does it establish model accuracy.

## 5. Measured CPU results

| Operation | CPU arithmetic | Median ms | p95 ms | Mean ms | CV (%) | GMAC/s |
|---|---|---:|---:|---:|---:|---:|
| QKV, one window | Full accumulation | 22.768 | 22.814 | 22.774 | 0.1026 | 0.237949 |
| QKV, one window | RTL compatibility | 53.479 | 53.643 | 53.464 | 0.2913 | 0.101359 |
| MLP FC1 | Full accumulation | 619.096 | 621.842 | 619.339 | 0.1412 | 0.186660 |
| MLP FC1 | RTL compatibility | 1,130.260 | 1,132.105 | 1,130.116 | 0.1155 | 0.102295 |
| MLP FC2 | Full accumulation | 971.128 | 971.404 | 971.233 | 0.0459 | 0.119030 |
| MLP FC2 | RTL compatibility | 1,137.588 | 1,140.215 | 1,137.519 | 0.1809 | 0.101629 |

Source: [report summary](../raw/hps_board_run1/report/summary.csv). All summary statistics were reconciled with raw samples; [audit record](hps_board_run1_audit.json).

**Median** describes typical measured latency. **p95** is the nearest-rank 95th percentile—the 19th sorted value among 20 samples—and remains preliminary with this sample count. **CV** is standard deviation divided by mean, expressed here as a percentage. **GMAC/s** is useful MACs divided by mean execution time; it includes the effect of extra scaling/clipping work in the denominator.

Three findings guide the next design stage:

1. **The larger operations have measurable CPU costs.** RTL-compatible FC1 and FC2 each require approximately 1.13 seconds in this implementation, providing concrete offload comparison points.
2. **Arithmetic count alone does not predict runtime.** FC1 and FC2 have equal MAC counts but different full-accumulation timings. Shape-dependent memory access is a plausible explanation, not a demonstrated bandwidth bottleneck.
3. **Timing is consistent within this run.** Larger-case CV is below 0.3%. This supports repeatability for these inputs and conditions, not a guarantee across workloads or operating conditions.

These values must not be summed to claim TinyViT latency. QKV covers one window, MLP covers a different token scope, and much of the model is absent. The exported 22 verified rows cover 11 fixed cases under two rules, not 22 independent image classifications. The audit confirms file/statistic consistency; it does not independently execute the FPGA or prove the model's quantization is correct.

## 6. Answer to the professor: how will FPGA answers be checked against the CPU?

Correctness will be evaluated at three levels so that a circuit bug is not confused with a quantization-quality problem.

### 6.1 Exact accelerator-versus-CPU arithmetic

Before integration, freeze a versioned contract covering signedness, accumulator width and overflow policy, scale/zero-point interpretation, rounding, tile order, saturation, bias handling, padding and output layout. Use a CPU routine implementing those exact rules. If the FPGA changes to full accumulation, update the matched reference rather than comparing incompatible outputs.

For each test:

1. Supply identical input bytes, scales and dimensions to the CPU and FPGA.
2. Run the CPU reference and retain its complete output array.
3. Transfer inputs, start the FPGA, wait for confirmed completion with a timeout, and read the entire valid output.
4. Decode the banked/tiled layout into the same logical matrix ordering.
5. Count unequal elements and record the first failing coordinate and both values. Widen signed values before calculating errors.

**Acceptance:** zero unequal integer elements and zero maximum absolute error for every valid output in the agreed suite. Checksums alone are insufficient. A timeout, incomplete output or stale result is a failure, not a numerical pass.

The proposed suite comprises the existing eight directed cases, at least 1,000 seeded randomized supported configurations, and the three larger workloads executed through the complete host tiler. Include tile boundaries, supported scale extremes, positive/negative extremes, padding where needed, reset and consecutive requests with changing inputs. Randomized coverage and hardware execution are planned; the current board report does not establish their completion.

### 6.2 Numerical effect of the accelerator contract

Compare RTL-compatible arithmetic with full accumulation on the same captured model tensors. Report mismatch fraction, mean/max absolute error and saturation rates at intermediate and final conversion points. Errors must be compared in clearly stated integer or common dequantized units.

The current small tests already expose differences: `rounding_gap` produces 1 versus 0, and `saturation_cancel` produces 0 versus −1. These motivate model-quality validation. They do not quantify classification loss. If early clipping harms quality, retain wider partial sums or revise quantization before claiming model-equivalent acceleration.

### 6.3 Complete-model quality

Freeze the checkpoint hash, preprocessing and calibration procedure. Calibration data must be separate from held-out labelled evaluation data. Evaluate the FP32 reference, quantized CPU model and hybrid implementation on the same images. Report top-1 accuracy, CPU/hybrid prediction agreement and output-score errors.

The proposed quality target is **at most 1 percentage point top-1 accuracy loss** against the selected FP32 reference. For example, a measured 84.0% reference would imply a target of at least 83.0%, not an assumed use of a published accuracy. Begin with a small functional set, then use a substantially larger labelled set for the final claim; report sample count and uncertainty in the paired accuracy difference. A point estimate near the boundary with wide uncertainty is inconclusive. If the same deterministic graph differs only by exact integer offload, require identical offloaded tensors and 100% CPU/hybrid top-1 agreement; any floating-point tolerance elsewhere must be documented before evaluation.

## 7. Answer to the professor: what is the inference speed target?

### 7.1 Choose a CPU-relative target

We propose a **baseline-relative objective**, because no application requiring a particular frame rate has been selected and no full-model HPS latency has been measured. Published GPU throughput cannot establish a DE1-SoC FPS requirement. The scientific question is whether selective offloading improves this board's inference at acceptable numerical quality.

This approach follows the general hardware-acceleration principle that end-to-end gain depends on the accelerated fraction and communication costs, as explained in AMD's acceleration tutorial. Its platform-specific PCIe examples do not describe this project's HPS–FPGA interconnect; only the system-level reasoning is used here. [Identifying Acceleration](https://docs.amd.com/r/2024.1-English/Vitis-Tutorials-Hardware-Acceleration/Identifying-Acceleration)

### 7.2 Operation-level target: at least 2×

Define speedup as the CPU median divided by the total offload median for the same matrix and arithmetic. Initial budgets derived from the current RTL-compatible baseline are:

| Selected complete operation | Current CPU median ms | Proposed maximum offload median ms |
|---|---:|---:|
| QKV, one window | 53.479 | 26.739 |
| MLP FC1 | 1,130.260 | 565.130 |
| MLP FC2 | 1,137.588 | 568.794 |

Start offload timing with operands ready in HPS memory; stop when the output is usable by the CPU in its required layout. Include packing, all tiles, weight/input/output transfers, cache maintenance, launch, wait and unpacking. Document weight-residency assumptions and charge any preload separately. Measure total wall time directly if transfers and computation overlap.

The 2× goal is a meaningful engineering margin rather than a literature guarantee. These numerical budgets apply to this simple C implementation. Report both this reproducible baseline and an optimized equivalent CPU comparator; final performance claims must not rely only on choosing the slower implementation.

### 7.3 Full-pipeline target: at least 1.5×, with no p95 regression

The primary steady-state application boundary will be **a decoded image available in HPS memory to its predicted class available on the CPU**, with the model already loaded. It includes preprocessing, the complete model and class selection. Camera capture, image-file decoding and model loading will be reported separately if used. Also report tensor-ready-to-output-score model latency to isolate inference computation.

For the same checkpoint, input resolution, batch size 1, quantization and software operations:

```text
median(hybrid application time) ≤ median(HPS-only application time) / 1.5
p95(hybrid application time) ≤ p95(HPS-only application time)
```

This is a proposed 33.3% reduction in median application latency. Once the HPS-only pipeline is runnable, convert the ratio into an absolute millisecond budget before measuring the hybrid result. If, illustratively, that baseline were 6,000 ms, the budget would be 4,000 ms; 6,000 ms is not a measured result. Report sequential throughput as completed images divided by total elapsed time, rather than treating inverse median latency as measured FPS.

### 7.4 Check feasibility with Amdahl's law

Let `f` be the fraction of baseline time eligible for acceleration, `r` its compute speedup, and `h` added non-overlapped overhead divided by baseline time. An approximate model is:

```text
Overall speedup = 1 / ((1 - f) + f/r + h)
```

This is a project feasibility calculation, not a prediction from the current three matrices. With `f=0.70`, `r=4` and `h=0.10`, it gives approximately 1.74×. If only 30% is eligible, even unlimited acceleration and zero overhead cannot reach 1.5×; the limit is approximately 1.43×. At `r=2` and zero overhead, at least two-thirds of the baseline must be eligible to reach 1.5×. Measure `f` in the real application, then assess the target; do not silently lower it after observing results.

## 8. Completion plan and required evidence

| Evaluation item | Current status | Next evidence required |
|---|---|---|
| HPS matrix timing and software-reference checks | Completed for this simple C run | Preserve raw data and build metadata |
| End-to-end inference latency | Pending | Complete equivalent quantized CPU and hybrid graphs; median/p95 and sequential throughput |
| FPGA kernel latency and DMA overhead | Pending | Compute cycle counts, achieved clock, transfer sizes/directions, effective bandwidth and complete offload wall time |
| FPGA logic, DSP and memory use | Pending | Quartus post-fit utilization and timing reports; distinguish total design from accelerator-only use where possible |
| FPGA output consistency | Pending | Full output comparisons under the frozen contract, directed/random/tiled tests and failure logs |
| Model accuracy and memory use | Pending | Held-out labelled evaluation, accuracy difference and peak process resident memory |

The implementation sequence is: confirm the arithmetic contract; obtain or recreate the HPS model runtime; capture real tensor shapes and profile non-overlapping operation times; establish optimized CPU timings; integrate the FPGA data path; verify outputs; then measure total speedup. Team slide claims without matching raw artifacts should be reconciled rather than merged into this run's evidence.

For final latency experiments, start with a short pilot, then aim for at least 10 warmup calls and 100 individually timed inferences per configuration when runtime permits. Keep the input sequence, thread settings, preprocessing and residency policy matched. Record clock/governor, affinity, compiler/runtime versions and hashes. Collect diagnostic operator timings separately and measure final latency with profiling disabled. Repeat independent runs when needed to assess drift; do not pool samples from different configurations.

## 9. Measurement protocol for the pending inference and FPGA metrics

**Readiness update:** the team confirmed that there is currently no runnable full-model HPS application or integrated FPGA/DMA board test. The existing matrix benchmark cannot collect these additional metrics. Their implementation prerequisites and board procedures are documented in [the step-by-step inference, FPGA and DMA measurement guide](../../docs/inference_fpga_measurement_steps.md).

### 9.1 Execution order

1. Freeze the checkpoint, 224 × 224/batch-1 configuration, preprocessing and integer arithmetic contract. Capture all floating-point fallbacks.
2. Build an ARMv7-compatible model runtime and validate one complete CPU inference. The old HPS OS requires a compatible executable; the earlier small C-program success does not establish runtime compatibility.
3. Instrument preprocessing, full synchronous model execution and class selection with the HPS monotonic timer. Record decoded-image-to-class application latency and tensor-ready-to-scores model latency separately.
4. Independently integrate the Quartus design, register interface, mapped DMA buffers and driver. Verify transfer-only byte patterns and then one supported matrix tile. Local RTL offsets are not verified HPS physical addresses.
5. Add hardware start/end cycle snapshots and establish the actual operating clock. Measure repeated verified tile computations; confirm that kernel boundaries include final output commit.
6. Measure valid-size DMA transfers in both directions, then whole-matrix offload through all required tiles. Record transfer sizes, cache/driver overhead and pipeline overlap.
7. Integrate the validated operator replacement into the same quantized graph. Verify output equivalence, then run the same CPU/hybrid timing harness and fixed image sequence.
8. Audit exported samples and replace the pending cells below with actual measurements. Preserve unsuccessful requests and timeout counts.

### 9.2 End-to-end inference measurements

Load the model and decoded images before steady-state sampling. Preallocate reusable application buffers where deployment permits it. For each request, record four timestamps: before preprocessing, after preprocessing, after synchronous inference completion, and after class selection. Write CSV output after the timed sequence. Model loading, decoding and camera capture are outside the chosen application boundary and must be reported separately if included in a demonstration.

Begin with a short correctness/timing pilot, then 20 samples and three warmups for a preliminary report; target 100 samples and ten warmups for final distributions if affordable. One sample is one completed inference. Measure an outer sequence interval as well as individual-call intervals. Sequential throughput is completed images divided by that sequence's elapsed seconds, not inverse median latency.

| Mode | Samples / warmups | Model median / p95 ms | Application median / p95 ms | Sequence wall time ms | Completed images/s | Failures |
|---|---|---|---|---|---|---|
| Equivalent quantized HPS-only | Pending | Pending | Pending | Pending | Pending | Pending |
| Equivalent quantized hybrid | Pending | Pending | Pending | Pending | Pending | Pending |

| Comparison | Definition | Target | Measured result |
|---|---|---|---|
| Application median speedup | CPU median / hybrid median, matched pipeline | ≥1.5× | Pending |
| Application tail latency | Hybrid p95 compared with CPU p95 | No regression | Pending |
| Sequential throughput | Total completed images / sequence seconds | Report both modes | Pending |
| Numerical validity | Same-contract tensors and model-quality checks | Section 6 criteria | Pending |

Collect operator self-times in a separate diagnostic pass. A trace with profiling enabled must not silently replace the uninstrumented latency result. Specify thread count, residency policy, input sequence, CPU operating conditions, executable/model hashes and driver/runtime versions for both modes.

### 9.3 FPGA kernel latency and clock measurements

Latch a free-running counter in the accelerator clock domain at accepted start and final output commit. Validate boundary/off-by-one behavior and use stable, atomic software reads across clock domains. Retain stalls and controller overhead in whole-kernel counts; PE-active cycles, if available, are a separate metric. A CPU start-to-done timer includes polling and bus overhead and must not be labelled isolated hardware latency.

Convert cycles using `kernel_ms = cycles × 1000 / operating_clock_hz`. Report the operating clock separately from TimeQuest Fmax, which estimates a timing limit. Save the actual bitstream's PLL/clock configuration and post-fit timing evidence; do not assume a 100 MHz clock or use Fmax as the running frequency. [Intel Standard Edition Timing Analyzer guide](https://www.intel.com/programmable/technical-pdfs/683068.pdf)

| Workload and scope | Number of tiles | Operating clock MHz | Post-fit Fmax / timing status | Measured kernel cycles | Kernel latency ms | CPU-observed launch/wait ms | Verified |
|---|---|---|---|---|---|---|---|
| Initial supported tile | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Complete selected QKV window | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Complete selected FC1 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Complete selected FC2 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

State whether whole-operation cycles are a sum of sequential tile service counts or a spanning start/end interval. These differ under overlap or idle waits. The current full-sized matrices exceed accelerator buffers and cannot be represented by one small-tile measurement.

### 9.4 DMA transfer measurements

Use real driver-managed DMA buffers, verified device-visible addresses and correct CPU/device cache ownership. Do not pass userspace pointers as physical addresses. Apply the target BSP's supported APIs; modern kernel documentation provides principles rather than a drop-in driver for Linux 3.12. [Linux DMA API HOWTO](https://www.kernel.org/doc/html/v6.12/core-api/dma-api-howto.html)

First disable computation and test both transfer directions using changing known data and bytewise validation. Sweep lengths only within endpoint capacity, alignment and descriptor limits. Measure the CPU-visible transaction including required synchronization, submission and completion. Record packing separately. Hardware DMA-engine latency requires its own accepted-request-to-complete counter; a post-submission software wait is not automatically engine latency.

Label direction by actual endpoints. An FPGA DMA master reading HPS DDR still transfers input data from HPS DDR to FPGA buffers. For each configuration report sample count, bytes, descriptor count, direction, burst/alignment configuration, cache policy, latency distribution and verification status.

| Data direction | Bytes / request | Descriptors | CPU-visible transfer-service median / p95 ms | Hardware engine cycles / time | Effective MB/s | Byte verification |
|---|---|---|---|---|---|---|
| HPS DDR → FPGA buffer | Pending | Pending | Pending | Pending or unavailable | Pending | Pending |
| FPGA buffer → HPS DDR | Pending | Pending | Pending | Pending or unavailable | Pending | Pending |

Transfer-service timing here excludes packing but includes required cache synchronization and driver completion handling. If the implemented boundary differs, rename it explicitly. Effective decimal MB/s = total transferred bytes / (total interval_ms × 1000). Record direction separately; a round trip moves data twice. Do not average per-transfer bandwidth ratios when an aggregate byte/time ratio is intended.

### 9.5 Complete offload overhead and reporting

Time from operands ready in HPS memory until the entire result is CPU-visible in the required layout. Record all tile transfers, repeated weights, padding, launch/wait and unpacking. Compare this total against the matched CPU operation, and include an optimized CPU comparator in final claims.

| Complete operation | Current matched CPU median ms | Total offload median / p95 ms | Actual input / weight / output bytes | Compute and DMA overlap policy | Median speedup |
|---|---:|---|---|---|---|
| Selected QKV window | 53.479 | Pending | Pending | Pending | Pending |
| Selected FC1 | 1,130.260 | Pending | Pending | Pending | Pending |
| Selected FC2 | 1,137.588 | Pending | Pending | Pending | Pending |

Use a non-overlapped diagnostic pass to inspect component costs, then benchmark the intended pipeline directly. Concurrent DMA and computation intervals cannot simply be added. The difference between CPU wall time and a cycle-derived compute time is not automatically DMA overhead: it may include packing, polling, cache operations, scheduler delays or inter-tile gaps. Only directly instrumented intervals should receive those labels.

These tables define the requested metrics without inventing values. Filling them requires the model runtime, board driver, complete FPGA integration and hybrid operator adapter described in the measurement guide.

## Conclusion

The first HPS run establishes a reproducible operation-level starting point and exposes the importance of matching numerical rules. It supports a concrete next evaluation: zero FPGA-versus-CPU integer mismatches, at least 2× complete-operation speedup, and a proposed 1.5× application-level speedup subject to measured feasibility and acceptable model quality. The current evidence supports CPU timing and software-reference correctness; it does not yet demonstrate FPGA acceleration or full TinyViT inference performance.

## Evidence and reproducibility

- [Detailed explanatory board-run report](hps_board_run1.md): terminology, individual test purposes and measurement limitations.
- [Report CSV](../raw/hps_board_run1/report/summary.csv), [pilot CSV](../raw/hps_board_run1/pilot/summary.csv), and adjacent per-case sample/log files: 70 original exported files preserved in the project.
- [Numeric audit and file hashes](hps_board_run1_audit.json): 440 report samples, 66 pilot samples, no numeric reconciliation issues.
- [CPU implementation](../../software/hps/bench_matmul.c) and [arithmetic-contract configuration](../../configs/accelerator_contract.json): project-side implementation references. The configuration remains a proposal until confirmed against the deployed FPGA revision.
- [Official TinyViT model table](https://github.com/microsoft/Cream/blob/main/TinyViT/README.md), [model implementation](https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py), and [AMD acceleration tutorial](https://docs.amd.com/r/2024.1-English/Vitis-Tutorials-Hardware-Acceleration/Identifying-Acceleration): external background, distinct from measured board evidence. Pin the exact model revision before deployment.

The original board executable and input manifests were not included in the results export, so their identity cannot be retroactively proven from these files alone. No new board execution was performed during preparation of this report.
