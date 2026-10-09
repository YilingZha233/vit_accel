# DE1-SoC CPU matrix-kernel baseline: first board run

## What this report is about

Our capstone aims to make an image-classification model run faster on a small development board. Before adding a custom accelerator, we need to know how long the board's ordinary processor takes to perform the computations we plan to accelerate. That starting measurement is called a **CPU baseline**.

In this experiment, we ran a C program on the DE1-SoC's ARM processor. It multiplied matrices containing small signed integers, checked the answers against saved expected answers, and measured the time taken. Matrix multiplication is an important building block of TinyViT, but the program did not load a complete trained model or classify an image.

**The question answered here is: “How quickly and consistently does this particular CPU implementation perform selected matrix calculations, and does it follow our specified arithmetic rules?”** The report does not yet answer “How many camera images per second can the finished system classify?”

The Windows computer was used to build and transfer the executable and display the board's terminal. The Mac was used for reference data and analysis. The execution times in this report were measured on the board, not on either laptop. No custom FPGA computation or DMA transfer was part of these measurements.

## Hardware and software terms in plain language

| Term | Meaning | Role in this experiment |
|---|---|---|
| DE1-SoC | The physical development board, containing the main chip, memory and connectors. | The platform under test. |
| SoC: system on a chip | A chip that combines several computing functions. | The Cyclone V SoC combines a processor subsystem and programmable FPGA logic. |
| CPU: central processing unit | A general-purpose processor that executes software instructions. | Executes all the matrix arithmetic measured here. |
| HPS: hard processor system | The fixed processor subsystem built into the chip, including ARM CPUs and associated controllers. “Hard” means implemented in silicon rather than constructed from FPGA logic. | Runs Linux and the benchmark. “HPS CPU baseline” means a baseline measured on these CPUs. |
| ARM Cortex-A9 / ARMv7 | Cortex-A9 is the CPU design; ARMv7 identifies its instruction-set architecture family. | The executable was compiled for this processor, rather than for the laptop's processor. |
| FPGA: field-programmable gate array | Programmable digital logic that can implement a custom parallel circuit. | Intended to accelerate matrix operations later. No accelerator performance is measured here. |
| RTL: register-transfer level | A way to describe a digital circuit's registers and operations, commonly using Verilog. | “RTL compatibility” names a CPU arithmetic routine designed to reproduce the current circuit's numerical behavior. It does not mean the FPGA ran. |
| Accelerator / offload | A specialized processing unit, and the act of sending work to it. | A future CPU-to-FPGA comparison will use this baseline. |
| PE / systolic array | A processing element performs arithmetic; a systolic array connects many PEs so data can move through them with reuse. | Part of the proposed accelerator architecture, not exercised by this run. |
| DDR3 / RAM | Working memory holding software, inputs and outputs. | The CPU accesses matrix arrays in memory during execution. |
| CPU cache | Small, fast memory near a CPU that retains recently used data. | Repeated calls can reuse cached data; cache effects are included in elapsed times. |
| DMA: direct memory access | A hardware engine moves blocks of data without the CPU copying every element itself. The CPU still sets up and synchronizes the transfer. | Not used or timed here. Later, DMA overhead must be included in the FPGA offload comparison. |
| UART / PuTTY | A serial communication interface and a Windows terminal application. | Lets the user start programs and retrieve output. It does not execute the benchmark on Windows. |
| Cross-compilation | Building an executable on one computer for a different processor/OS target. | WSL on Windows built an ARM Linux executable that ran on the board. |
| Compiler / GCC | Software that translates C source into machine instructions. | GCC 15.2.0 generated the measured program. |
| Thread | One sequence of software execution. | The benchmark has one compute thread, even though two CPU cores are available. It is not a two-core performance result. |

## How the matrix tests relate to TinyViT

A machine-learning model transforms input data through a sequence of operations. During **training**, its numerical parameters, called **weights**, are learned from examples. During **inference**, those weights are used to produce a prediction for a new input. TinyViT-21M refers to the team's chosen model with approximately 21 million parameters, not 21 million images or operations.

An **activation** is an intermediate numerical representation produced while the model runs. A **tensor** is a multidimensional array; a matrix is a two-dimensional tensor. A **token** represents a position or patch-derived feature in a vision model. A **layer** or **block** groups model computations. A **kernel** here is a small computational routine, not the Linux operating-system kernel.

The selected tests represent these model ingredients:

- **QKV projection:** a linear transformation constructs query, key and value features used in attention. Attention uses these features to combine information across tokens. Our QKV test measures the projection matrix multiplication only. It does not compute attention scores, softmax or the final attention-weighted output.
- **MLP:** a multi-layer perceptron transforms each token's features through linear layers and an activation function. **FC1** is the first fully connected/linear transform, expanding features. **FC2** reduces them again. These tests measure each matrix multiplication separately; they do not execute the complete MLP block.
- **Softmax, normalization, bias and nonlinear activation:** other model operations that turn matrix results into useful model features. They are excluded from this benchmark.

The arrays are **synthetic**, meaning generated test numbers rather than weights and activations extracted from the trained checkpoint. A **checkpoint** is a saved set of model parameters. These tests establish arithmetic behavior and representative matrix timing; they do not test image-classification quality. The selected dimensions should still be checked against the actual deployed model graph, which is the complete sequence of model operations.

## Exactly what runs and what the clock includes

For each case and each arithmetic contract, the benchmark follows this sequence:

1. Read the saved input matrices and allocate memory. This is outside timing.
2. Compute an output and compare every output byte with the saved golden output. This is outside timing.
3. Execute warmup calls to reduce first-call effects. These calls are not included in the reported samples.
4. Start a monotonic timer, execute the matrix routine, and stop the timer. Repeat to obtain the samples.
5. Save each sample after stopping the timer. After the timed repetitions, verify the output again and calculate summary statistics.

A **monotonic timer** measures elapsed intervals without depending on the calendar date. It measures elapsed wall-clock time, so operating-system interruptions can affect a sample. It is not a count of CPU instructions or cycles. Small loop and result-consumption overhead is included along with arithmetic; timing and sample logging are implemented consistently across cases. RTL-compatible timing also includes clearing its output buffer.

For each large matrix there are 20 samples, each measuring one call. For a small case, each of the 20 samples measures a group of 1,000 calls and divides that duration by 1,000. For example, the identity full-accumulation mean of 0.034806218 ms is approximately 34.81 microseconds **per call**; its average 1,000-call group takes approximately 34.81 ms.

The same input values are reused. Twenty repetitions do not mean twenty different images. The routine includes reading array elements through the CPU memory/cache hierarchy, multiplication, accumulation, output rescaling and clipping. It excludes loading files, copying files by USB, serial transfer, DMA, FPGA execution and the full inference pipeline.

## How to read the matrix dimensions and integer arithmetic

The benchmark's notation is:

```text
W has K rows and N columns
X has N rows and M columns
W × X produces Y with K rows and M columns
```

Each output element is a **dot product**: multiply corresponding values and add the products. A **MAC**, or multiply-accumulate, is one multiplication together with its addition into a running sum. Computing the output requires K × N × M MACs. Some publications count multiplication and addition as two operations; this report counts them together as one MAC. GMAC/s is therefore not the same numerical quantity as a two-operations-per-MAC GOP/s figure.

For FC1, W is 768 × 192 and X is 192 × 784. The output contains 768 × 784 = 602,112 values. Each requires 192 products, giving 115,605,504 MACs. The synthetic QKV test uses 49 token columns, representing one 7 × 7 window; the MLP tests use 784 columns. The 576 QKV output features correspond to three groups of 192, and FC1 expands 192 features to 768. These are workload interpretations of the selected shapes, not evidence that a full model was executed.

**INT8** means an 8-bit signed integer, with values from -128 to 127. The operands are stored as INT8, but their products are added into an **INT32** accumulator, a wider signed integer that provides more room for the sum. This description does not imply that the CPU executed specialized 8-bit vector instructions. The implementation also uses a 64-bit intermediate during output rescaling to avoid overflowing the scaling multiplication.

To return an accumulated value to INT8, the program uses:

```text
scaled = floor(accumulator × q16 / 65536)
output = clamp(scaled, -128, 127)
```

Here **q16** is an integer representation of the multiplier q16 / 2^16. For example, q16=32768 represents 0.5; q16=128 represents 1/512. It is not a 16-bit model precision label. **Floor** rounds toward negative infinity, so floor(-0.5)=-1. **Clipping**, **clamping** and **saturation** mean keeping values within the allowed range: 200 becomes 127 and -200 becomes -128. **Requantization** is the scaling and conversion back to the output integer representation.

In a deployed quantized model, suitable scales are determined from weights and representative calibration data. The scale used here is a synthetic test setting, not evidence of successful model quantization.

An **arithmetic contract** specifies exactly when sums, scaling, rounding and clipping occur. The two contracts are different algorithms:

- `full_accumulation`: form the full dot product in INT32, then rescale and saturate once.
- `rtl_compatibility`: split the dot product into **tiles** of 16 terms, rescale and saturate each partial result, then add it to the output with saturation after each addition.

Why timing of rounding matters: two partial sums of 1, scaled by 0.5, give floor((1+1)×0.5)=1 if combined first, but floor(1×0.5)+floor(1×0.5)=0 if rounded separately. This is the purpose of `rounding_gap`. Neither result is a mysterious processor error: they follow different contracts.

## Every column in summary.csv

| Column | Meaning and interpretation |
|---|---|
| `case` | Test-data folder identifying the fixed matrices used. It is not an image filename. |
| `contract` | The arithmetic routine being tested: full accumulation or RTL compatibility. Both run on the CPU. |
| `k`, `n`, `m` | K output rows, N terms per dot product, and M output columns. These letters follow this project's convention. |
| `q16` | Integer output scale; divide by 65536 to obtain its multiplier. |
| `repetitions` | Number of timing samples per contract: 20 in the report. |
| `median_ms` | Middle duration after sorting samples; for 20 samples, average of positions 10 and 11. A useful typical latency. |
| `p95_ms` | Nearest-rank 95th percentile: position ceil(0.95 × count), which is 19 for 20 samples. A preliminary view of slower executions, not a maximum or a guarantee for future runs. |
| `min_ms` | Fastest observed per-call duration. It is not a guaranteed achievable latency. |
| `checksum` | Sum of all signed INT8 output values, useful as a quick identifier. Different arrays can have the same sum, so checksum equality alone does not prove correctness. This is not a cryptographic hash. |
| `verified` | Whether every output byte matched the relevant golden reference before and after timing. It does not mean the FPGA was tested or that model predictions are accurate. |
| `calls_per_sample` | Calls timed together: 1 for larger cases, 1000 for small ones. Summary times are divided by this count. |
| `warmups` | Untimed preparation calls per contract: 3 in the report. There is also an initial correctness call. |
| `mean_ms` | Arithmetic average of the per-call sample durations. |
| `stddev_ms` | Population standard deviation of those durations, calculated with division by the sample count. It describes measured spread, not uncertainty in the mean. |
| `cv` | Coefficient of variation = standard deviation / mean. The CSV contains a ratio: 0.001412 means 0.1412%, not 0.001412%. |
| `macs` | K × N × M useful multiply-accumulates per matrix call. This does not count extra scaling/clipping operations. |
| `gmac_per_s` | Billions of useful MACs per second, computed from MAC count divided by mean duration. Higher is faster. It is effective kernel throughput, not peak hardware capacity. |

**Worked example: FC1, RTL compatibility.** One call produces 602,112 output elements from 115,605,504 MACs. Its median is 1,130.259505 ms, or approximately 1.130 seconds. Its mean is 1,130.115947 ms, so effective throughput is 115,605,504 / (1130.115947 × 1,000,000) = approximately 0.102295 GMAC/s. This is roughly 102.3 million useful MACs per second for this implementation. It is not 0.885 complete TinyViT images per second: only one selected matrix operation was timed.

## What each small test is for

| Case | Main purpose |
|---|---|
| `identity16` | Uses a doubled identity matrix and a 0.5 scale to reproduce the input, exercising indexing and rescaling. |
| `zeros` | Checks that zero inputs produce zero outputs. |
| `random_single` | Checks general signed data with a 16-term contraction. |
| `random_multi` | Checks a larger shape with multiple 16-term contraction tiles. |
| `partial_rows` | Exercises output dimensions 25 × 18 that are not multiples of 16; this run tests the CPU logic, not a hardware tiler's boundary behavior. |
| `layer_tile` | Exercises a 16 × 128 by 128 × 16 workload with a longer contraction. |
| `rounding_gap` | Deliberately exposes the numerical difference from rounding partial sums early. |
| `saturation_cancel` | Uses large positive and negative contributions to expose the difference from clipping before cancellation. |

## Files, evidence and the meaning of a passing result

`pilot` is a short trial with three samples per contract and one warmup. `report` contains the longer run with 20 samples and three warmups. Each folder contains a combined `summary.csv`, individual case summaries, individual timing files and logs, and an `environment.txt` snapshot.

In a `.samples.csv` file, `sample` is a zero-based sample number; `group_ms` is the total time of the timed group, and `per_call_ms` is that time divided by `calls_per_sample`. In a `.log` file, compiler version, flags, architecture and timing scope describe how the executable reports it was built and run. `environment.txt` describes the OS and available memory; it does not provide process peak memory usage or continuously recorded CPU frequency.

A **golden reference** is a saved expected output from a separate software implementation. “22 verified rows” means 11 fixed cases each passed under two contracts. It does not mean 22 FPGA tests, 22 classifications or 22 different model layers. The independent audit recomputes statistics from the exported timings and checks consistency between files. It does not run the board again. The SHA-256 inventory identifies exact exported file contents; unlike the arithmetic checksum, it is used for file-integrity checks.

## Outcome

The exported results establish a preliminary single-thread CPU matrix-kernel baseline on the DE1-SoC. All 22 case/contract combinations report elementwise verification against their corresponding golden files. An independent numeric audit of the exported CSVs found no discrepancies in sample counts, median, mean, nearest-rank p95, minimum, population standard deviation, coefficient of variation, MAC count or GMAC/s, within printed precision.

The audit covers 440 report samples and 66 pilot samples across 11 cases and two arithmetic contracts. These are timing samples, not 506 distinct input datasets. Eight cases exercise small arithmetic cases; three exercise larger synthetic matrices. The audit does not rerun hardware or independently establish that golden data implement the intended model.

## Environment and method

- Platform: Altera SOCFPGA, ARMv7, Cortex-A9; two cores visible, one benchmark thread. Board revision G was identified from the earlier board photograph, not the Linux Revision field.
- OS: Poky 8.0 / Yocto 1.3, Linux 3.12.0-00307-g507abb4-dirty.
- Compiler recorded in all case logs: GCC 15.2.0.
- Recorded flags: `-O3 -std=c11 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard`.
- NEON target flags establish compiler configuration, not proof that the inner loop uses vector instructions. No disassembly or optimized-library comparison has been supplied.
- Memory: MemTotal 1,031,824 kB; this is system memory, not benchmark peak RSS.
- Report: three warmup calls and 20 samples per contract. Larger matrices use one call per sample; small cases use 1,000 calls per sample, normalized to per-call time.
- Timing scope: C GEMM arithmetic including each contract's requantization; steady-state reuse of allocated inputs. Excludes file loading, USB transfers, model loading, bias and nonlinear model operations.
- The benchmark uses synthetic signed INT8 operands and INT32 accumulation. Large-case q16=128 is an exercise scale, not a calibrated TinyViT scale.
- CPU frequency/governor, CPU affinity, binary hash and actual acquisition date are not established by this export. Board timestamps should not be treated as authoritative dates. Preserve these details on future runs.

## Larger matrix results

Matrix convention: W[K,N] multiplied by X[N,M]. Times are milliseconds. Throughput uses the mean: GMAC/s = K*N*M / (mean_ms * 1e6).

| Case | K,N,M | Contract | Median ms | p95 ms | Mean ms | CV % | GMAC/s |
|---|---|---|---:|---:|---:|---:|---:|
| QKV, one window | 576,192,49 | Full accumulation | 22.768 | 22.814 | 22.774 | 0.1026 | 0.237949 |
| QKV, one window | 576,192,49 | RTL compatibility | 53.479 | 53.643 | 53.464 | 0.2913 | 0.101359 |
| MLP FC1 | 768,192,784 | Full accumulation | 619.096 | 621.842 | 619.339 | 0.1412 | 0.186660 |
| MLP FC1 | 768,192,784 | RTL compatibility | 1130.260 | 1132.105 | 1130.116 | 0.1155 | 0.102295 |
| MLP FC2 | 192,768,784 | Full accumulation | 971.128 | 971.404 | 971.233 | 0.0459 | 0.119030 |
| MLP FC2 | 192,768,784 | RTL compatibility | 1137.588 | 1140.215 | 1137.519 | 0.1809 | 0.101629 |

Larger-case variability was low within this run. Twenty samples give only a preliminary tail estimate; the reported p95 is the 19th sorted sample. Small-case percentiles describe averages of 1,000 calls, not individual-call tails. The rounding and saturation microcases still have sub-millisecond group durations and should primarily support correctness claims.

FC1 and FC2 each perform 115,605,504 MACs, but full-accumulation FC2 is slower. Shape-dependent data access/cache effects are a plausible explanation, not a measured diagnosis. These timings must not be summed and called model latency: QKV covers one window, MLP uses 784 columns, and the remaining graph is absent.

## Correctness and the arithmetic decision

The full contract requantizes and clips after the complete dot product. The RTL contract requantizes and clips each 16-element partial dot product, then combines it with saturation. Golden checking is byte-by-byte before and after timed execution, not a checksum-only test.

The contracts differ: rounding_gap checksums are 1 versus 0; saturation_cancel gives 0 versus -1; FC1 gives -125614 versus -1911217. Both contracts can pass their own references while disagreeing with one another. Checksum differences demonstrate disagreement, but do not quantify tensor error or classification loss.

For FPGA acceptance, require zero element mismatches against the frozen RTL-compatible CPU contract for identical inputs. Separately evaluate that contract against the intended quantized model and an FP32 reference, including held-out classification accuracy. Current results demonstrate neither FPGA correctness nor model accuracy.

## Proposed performance targets

**Latency** is the elapsed time to finish a specified task; lower is better. **Throughput** is the amount of completed work per unit time; higher is better. **FPS** means frames/images per second and requires a complete, clearly defined image-processing workload. **Speedup** is baseline latency divided by accelerated latency for the same work and arithmetic. A 2x speedup means half the time, not a reduction of two milliseconds.

For the future hybrid pipeline, the CPU arranges data (**packing**), requests movement to FPGA buffers, starts the accelerator, waits for completion, and restores output layout (**unpacking**). **Cache maintenance** ensures CPU-cached data and device-visible memory are consistent when required by the platform. These costs can consume the benefit of faster arithmetic, which is why the target includes the entire offload interval. With overlap, measure actual elapsed offload time rather than adding intervals that occur simultaneously.

Use at least 2x speedup over the equivalent current CPU contract for a complete offloaded operation. Derived median offload budgets are 26.739 ms for this QKV window, 565.130 ms for FC1, and 568.794 ms for FC2. Include packing, all tiles, DMA, cache maintenance, launch/wait and unpacking in the offload duration. These are provisional engineering targets against simple C, not optimized CPU limits or observed FPGA results.

Keep at least 1.5x full-model speedup as an unvalidated project goal. Measure the complete equivalent INT8 CPU graph before setting an absolute FPS target or applying Amdahl's law. The fraction of inference eligible for acceleration is not available from this suite.

**Amdahl's law** explains why accelerating one part does not accelerate the whole program by the same factor. If a fraction f of CPU time is accelerated by r, ignoring extra overhead, overall speedup is 1 / ((1-f) + f/r). As an illustrative example, speeding up half of a program by 2x gives only 1.33x overall speedup. We have not measured f for TinyViT in this experiment, so this example is not a prediction for our board.

## What has and has not been measured

| Question | Status after this report |
|---|---|
| Does this CPU program match its two golden arithmetic references? | Yes for the supplied test cases, as reported by elementwise checks. |
| How long do the selected matrix routines take on the board? | Measured, with raw repeated timing samples. |
| Do the summary statistics agree with saved samples? | Independently checked; no discrepancies beyond printed precision. |
| Does the FPGA produce the same outputs as the CPU? | Not measured. Requires actual FPGA execution and output comparison. |
| How fast are DMA transfers or HPS-to-FPGA communication? | Not measured. No DMA benchmark ran. |
| What is complete TinyViT-21M latency or FPS? | Not measured. Requires the complete deployed model. |
| Is INT8 classification accuracy acceptable? | Not measured. Requires calibrated model execution on held-out labelled images. |
| Is this the fastest CPU implementation? | Not established. Requires optimization and a fair runtime/library comparison. |
| How much memory does inference require? | Not measured. System MemTotal is not peak process memory. |
| What are FPGA resource use, power and energy? | Not measured. Requires separate hardware/tool measurements. |

For later evaluation, **top-1 accuracy** is the proportion of labelled images whose highest-scoring predicted class is correct. **Prediction agreement** measures whether two implementations choose the same class, which is different from correctness against labels. **Logits** are model output scores before probability conversion. **Held-out** data were not used to train or calibrate the model. **Peak RSS** is the highest resident physical memory attributed to a process during execution. **Batch 1** means processing one image per inference call. These terms describe future measurements, not results obtained here.

## Remaining work

1. Preserve this run as the simple C baseline; no immediate repeat is necessary.
2. Freeze the arithmetic contract with the accelerator teammate and evaluate per-tile clipping/rounding quality before model integration.
3. Implement or obtain a reasonably optimized cache-friendly/NEON CPU or runtime baseline with equivalent arithmetic; verify outputs before timing.
4. Obtain the actual TinyViT-21M checkpoint, preprocessing, quantization specification and HPS runtime. Reconcile benchmark shapes with its executed graph.
5. Measure full-model batch-1 latency, operator self-times, peak RSS and held-out quality. Record precision fallbacks explicitly.
6. Compare integrated FPGA offload with equivalent CPU operations and subsequently with the full CPU model.

## Reproducibility and sources

Original supplied folder: `/Users/yilingzha/Downloads/vit_results_export/results/`.
Preserved project copy: `results/raw/hps_board_run1/` (pilot and report, 70 files).
Numeric audit and SHA-256 inventory: `results/reports/hps_board_run1_audit.json`.
Reproduce the CSV audit with `python3 scripts/audit_hps_results.py results/raw/hps_board_run1 /tmp/hps_audit.json`.
The source files were preserved without alteration. No original on-board executable or vector manifests were included in this export, so their hashes cannot be retrospectively established from it.

## Presentation wording

We established a single-thread CPU baseline on the DE1-SoC for synthetic INT8 matrices representative of selected TinyViT-21M operations. All 22 case/contract combinations matched their corresponding golden outputs, and the exported timing statistics reconcile with 440 raw report samples. RTL-compatible median execution times were 53.48 ms for one-window QKV, 1130.26 ms for MLP FC1 and 1137.59 ms for MLP FC2. Full-model inference, quantization quality and FPGA speedup remain to be evaluated.
