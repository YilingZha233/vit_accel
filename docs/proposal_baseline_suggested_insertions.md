# Suggested proposal additions: CPU baseline and measurable goals

Read-only review of the “ECE496 Proposal” tab in the shared project proposal on October 9, 2026. No edits were made to the Google Doc. The following text is offered for team review and manual insertion.

## Recommended placement and decisions

- Section 4: state the fixed workload and baseline-relative project goal. The current 10 FPS goal is not established by CPU matrix timings or a demonstrated full-model runtime. Retain it only as a separately justified application objective, not a supported commitment.
- Section 5: put testable correctness, latency boundaries and acceptance thresholds here. Do not place requirements only in References.
- Section 2: add a brief “Preliminary CPU Baseline” subsection, replacing repeated unsupported references to 498 ms and 1.4–6.4 ms unless the team recovers their original workload/transfer evidence.
- Appendix C: include a compact matrix-results table and measurement method. Cite the detailed baseline report as internal project evidence.
- References: add the internal report's bibliographic entry and a stable shared attachment/location. A local filesystem path is not a usable reference for the professor.

The proposal currently specifies a minimum 20% operation-latency reduction (1.25× speedup) and a 10% full-pipeline reduction (approximately 1.11×). The separately prepared baseline report proposes stronger 2× and 1.5× targets. A practical reconciliation is to retain the proposal's minimum thresholds and identify 2×/1.5× as stretch objectives. This is a recommendation, not a team-approved change. If the team instead adopts the stronger targets as requirements, revise Sections 3, 4, 5 and 8 consistently and update the baseline report's status language.

## Proposed replacement for Section 4.0 Project Goal

The goal of this project is to develop and evaluate a hybrid CPU–FPGA implementation of TinyViT-21M image classification on the DE1-SoC, using a fixed 224 × 224 input resolution, batch size of one, and a documented INT8 quantization scheme. The HPS will execute model control and non-accelerated operations, while selected matrix operations will be offloaded to the FPGA. The minimum performance requirement is a 20% reduction in latency for at least one complete offloaded matrix operation, including communication and control overhead. The system-level objective is a 10% reduction in end-to-end application latency relative to an equivalent HPS-only quantized pipeline. Stretch objectives are 2× operation-level and 1.5× application-level speedup. These targets will be evaluated using matched workloads and explicit output-correctness tests; an absolute frame-rate target will be established only after full-model profiling and application requirements are confirmed.

## Proposed Section 2.4: Preliminary CPU Baseline

A preliminary single-thread C benchmark has been executed on the DE1-SoC ARM Cortex-A9 to establish CPU reference timings for selected synthetic INT8 matrix workloads representative of TinyViT-21M. Eight directed arithmetic cases and three larger matrices were tested under two rules: complete INT32 accumulation followed by output conversion, and a software routine reproducing the current RTL's per-tile conversion and saturation. All 22 case/rule combinations reported elementwise agreement with their corresponding golden outputs. The larger-case RTL-compatible median latencies were 53.479 ms for one-window QKV projection, 1,130.260 ms for MLP FC1, and 1,137.588 ms for MLP FC2. Each used three warmup calls and 20 timed repetitions; an independent audit reconciled the report statistics with the saved samples [6].

These measurements provide operation-level comparison points, not complete TinyViT inference latency or FPGA speedup. The operands are synthetic rather than checkpoint-derived, and no FPGA or DMA execution is included. The full-model HPS runtime and integrated accelerator path remain to be implemented and measured. Final evaluation will also include an optimized CPU comparator to avoid attributing gains solely to an inefficient software baseline. Appendix C summarizes the measured matrix workloads.

## Proposed Section 5.1: Requirements and verification

The prototype must execute the fixed TinyViT workload and produce a class prediction for every image in the declared functional test set. Accelerator correctness will be tested independently of model quantization quality. Integer FPGA outputs must match a CPU reference element by element under identical input values, scales, signedness, rounding, accumulation and saturation rules. Classification agreement alone is insufficient to establish arithmetic correctness.

| ID | Requirement | Verification |
|---|---|---|
| R1 | Execute the complete fixed TinyViT-21M image-classification pipeline. | Produce a valid output for every image in the declared test set; record failures and timeouts. Valid output does not imply the predicted class is correct. |
| R2 | Reduce latency of at least one selected complete matrix operation by at least 20% relative to an equivalent HPS-only implementation. | Compare matched CPU and offload medians. Offload timing includes packing, all tiles, transfers, cache maintenance, launch/wait and output reconstruction. |
| R3a | Match FPGA integer outputs exactly to the CPU reference under the frozen arithmetic contract. | Require zero element mismatches on directed cases, supported boundary cases, seeded random tests and complete tiled workloads. Compare full outputs, not checksums alone. |
| R3b | Evaluate model-level consistency separately from integer-kernel correctness. | Use the same held-out images and preprocessing. Retain the proposal's ≥95% top-1 agreement threshold only with an explicitly named reference and declared comparison. An equivalent deterministic quantized CPU/hybrid graph is expected to agree on every prediction; discrepancies require investigation. |
| R4 | Measure inference, FPGA computation and data-transfer performance separately. | Aim for at least 100 timed requests after warmup in final evaluation; report median, mean, p95, sample count and sequential throughput. Retain raw samples and document any reduced-count preliminary run. |

### Timing and precision definitions to append to Section 5.1

Steady-state end-to-end application latency is measured from a decoded image available in HPS memory to its predicted class available on the CPU, with the model already loaded. It includes preprocessing, complete inference and class selection. Camera capture, decoding and model loading are reported separately. Model-only latency is measured from a preprocessed tensor ready to output scores ready. FPGA kernel time is derived from measured hardware cycles and the actual operating clock; total offload latency is measured separately on the HPS. DMA latency and bandwidth use documented transfer boundaries, byte counts and data directions. Overlapping intervals are not added as though they were sequential.

The CPU and hybrid comparisons must use the same checkpoint, preprocessing, resolution, batch size and arithmetic. Full accumulation and the current RTL-compatible routine can produce different integer outputs because they round and clip at different stages. Any change to that contract requires a matching CPU reference and renewed model-quality evaluation.

### Optional stronger model-quality objective

If the team accepts a labelled evaluation requirement, add: “Limit top-1 accuracy loss to at most one percentage point relative to the selected FP32 model on the same held-out labelled set; report sample size and uncertainty.” This is distinct from prediction agreement. For example, two implementations can agree on an incorrect class. Do not describe either the one-percentage-point target or the existing 95% agreement target as achieved by the matrix benchmark.

## Proposed Section 5.3: Reconcile minimum and stretch objectives

| ID | Objective | Evaluation |
|---|---|---|
| O1 | Reduce median end-to-end application latency by at least 10%; stretch target: 1.5× speedup. | Compare equivalent quantized CPU and hybrid pipelines; also report p95, targeting no p95 regression. An absolute FPS value follows from actual full-pipeline measurements. |
| O2 | Keep non-overlapped communication overhead within 20% of total offload latency. | Define included cache, submission, transfer and completion costs. Use a non-overlapped diagnostic run for the ratio; report actual overlapped pipeline wall time separately. |
| O3 | Target no more than 80% utilization of available logic, DSP and block-memory resources. | Use post-fit Quartus reports for the implemented design; fitting and timing closure remain mandatory even if a utilization target is met. |
| O4 | Achieve 2× speedup for selected complete matrix offloads as a stretch objective. | Include all required tiling and host overhead; compare against equivalent CPU arithmetic and report an optimized CPU comparator. |

The 10% and 20% reduction thresholds are retained from the current proposal. They are not equivalent to 1.5× and 2× speedup: a 20% latency reduction gives 1/0.8=1.25× speedup, whereas 2× means a 50% reduction.

## Proposed Appendix C: Preliminary CPU matrix measurements

The following results were obtained using single-thread C code on the DE1-SoC HPS, compiled with GCC 15.2.0 and `-O3` with Cortex-A9 target options. Inputs are synthetic signed INT8 values. Each larger case used three warmups and 20 individually timed executions per arithmetic rule. File I/O and result validation were outside timing; output scaling and saturation were included.

| Operation | K, N, M in W[K,N] × X[N,M] | Full-accumulation median ms | RTL-compatible median ms | RTL-compatible p95 ms |
|---|---|---:|---:|---:|
| QKV, one window | 576, 192, 49 | 22.768 | 53.479 | 53.643 |
| MLP FC1 | 768, 192, 784 | 619.096 | 1,130.260 | 1,132.105 |
| MLP FC2 | 192, 768, 784 | 971.128 | 1,137.588 | 1,140.215 |

Full accumulation converts once after the complete dot product. RTL compatibility converts 16-term partial sums and merges them with saturation. Both columns measure CPU execution. All 22 case/rule combinations passed their own golden-output checks; this does not establish equality between the two rules or correctness of an FPGA implementation. The reported p95 is preliminary with 20 samples. QKV represents one window, whereas MLP uses a larger token scope; the three times must not be summed and labelled full-model latency.

For the proposal's minimum 20% latency reduction, provisional maximum offload medians against the current RTL-compatible CPU routine are 42.783 ms for QKV, 904.208 ms for FC1 and 910.070 ms for FC2. For the 2× stretch objective they are 26.739 ms, 565.130 ms and 568.794 ms respectively. All budgets refer to the complete corresponding operation including overhead, and must be reassessed against an optimized CPU comparator. Full-model inference, DMA and FPGA measurements remain pending.

## Suggested additional reference

[6] TinyViT DE1-SoC project team, “TinyViT-21M on DE1-SoC: CPU Baseline and Acceleration Evaluation Plan,” internal project report, October 2026, with HPS board run 1 measurement archive.

Before submission, attach the report and archive or provide a stable team-accessible URL. The report's preparation date is not proof of the board's acquisition date. The local report is `results/reports/cpu_baseline_professor_report.md`; its detailed companion is `results/reports/hps_board_run1.md`.

## Consistency checks elsewhere in the proposal

1. Executive Summary, Introduction, Sections 2.3 and 3: the 498 ms CPU / 1.4–6.4 ms DMA values need their own raw source, matrix sizes, byte counts, direction and timing boundaries. They are neither reproduced nor disproved by board run 1. Recover that evidence or replace the statements with the verified matrix-only findings above.
2. Section 4: “video classification” can imply temporal classification. If the intended model independently classifies camera frames, use “image classification of sampled video frames.” A 10 FPS target requires a defined stream, latency/throughput boundary and feasible full-pipeline evidence; current measurements do not establish it.
3. Section 3 and Executive Summary: INT32 partial accumulation alone does not guarantee full-INT32 accumulation across all contraction tiles. Describe the final agreed contract accurately.
4. Section 8: mirror whichever minimum/stretch thresholds the team accepts; do not introduce another target set. Keep the conclusion brief, with methodology and detailed results in Sections 5 and Appendix C.
5. Section 7 fallback: standalone FPGA/DMA measurements remain possible only if those subsystems are actually integrated. A CPU-only fallback cannot substantiate FPGA latency, DMA overhead or hardware-output consistency.
6. Table of Contents: add Appendix C and list the already-present Appendix B; replace placeholder table/figure/page numbers before submission.
