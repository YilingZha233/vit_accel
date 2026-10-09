# Baseline and evaluation protocol

**Model-selection update (2026-10-09):** the team confirmed TinyViT-21M and INT8.
The earlier 5M suggestion is superseded. See
[the CPU report audit](cpu_profile_report.md) for the missing measurements and
the short first-pass plan. No existing small-kernel timings are full-model results.

## Concrete project objective

Demonstrate batch-1 TinyViT-21M classification at 224x224 on the DE1-SoC, with
selected operations offloaded to the FPGA. Preserve the agreed integer
arithmetic exactly at the accelerator boundary and quantify accuracy loss
relative to the original FP32 checkpoint. Aim for at least 1.5x end-to-end
speedup over the same quantized graph running entirely on the HPS. This is a
proposed target, not a measured result or a hardware guarantee.

## References and comparison matrix

| Run | Purpose | Fair interpretation |
|---|---|---|
| Desktop FP32 TinyViT | Golden model outputs and debugging | Not the HPS latency baseline |
| HPS FP32 TinyViT | Original deployment baseline | Includes runtime/operator costs |
| HPS full-sum quantized graph | Quantization baseline | Separates quantization from acceleration |
| HPS RTL-compatible quantized graph, if retained | Same-graph hardware baseline | Separates current tile arithmetic from offload |
| Hybrid, same graph/checkpoint/preprocessing | Acceleration evaluation | Include all transfer/packing/control costs |
| Simple C INT8 kernel supplied here | Initial kernel measurements | Not full TinyViT or an optimized final CPU baseline |

Keep thread counts and runtime settings explicit. Report both single-thread
and best practical two-core HPS baselines where available; do not compare an
optimized accelerator only against intentionally slow Python loops.

## Correctness acceptance gates

1. **Arithmetic specification:** use Python exact-integer accumulation and
   a C implementation with proven bounds. Inputs INT8, scale integer 0..65535,
   signed floor shift, INT8 clipping. Bias is absent from the starter kernel;
   define its scale and placement before implementing real linear layers.
2. **Hardware conformance:** every valid output byte must equal the frozen
   RTL-compatible reference. Zero mismatches across directed tests plus at
   least 1000 seeded varied hardware/simulation transactions. Report scope:
   passing software-only tests is not passing hardware.
3. **Algorithmic error:** compare RTL-compatible results against full-sum
   requantization, separately from hardware conformance. Current tile clipping
   can introduce error even when FPGA implementation is perfect.
4. **Model quality:** proposed acceptance of <=1 percentage-point top-1 accuracy
   loss versus the pinned FP32 checkpoint on the same held-out labelled set.
   Start with 1000 images for a pilot; report uncertainty and use the full
   validation set when feasible. A 1000-image pilot alone does not establish a
   tight one-point accuracy bound. Keep calibration disjoint from evaluation.
5. **Mixed floating-point stages:** define absolute/relative tolerances using
   actual precision/runtime behavior; inspect logits and intermediate tensors,
   not only predicted labels. Top-1 agreement with FP32 is not labelled accuracy.

Directed tests: zero, identity, negative odd products, +/-128/127 operand limits,
scales 0/1/32768/65535, saturation/cancellation, output partial rows, unaligned
M, multiblock contraction, capacity boundaries, zero padding, consecutive runs
without reset, reset and timeout recovery, bus stalls and stale output detection.
Unsupported shapes should be rejected by software before MMIO.

## Timing definition

Use CLOCK_MONOTONIC on HPS and a hardware cycle counter with the **achieved**
clock frequency for FPGA compute. Start with 10 warmups and 100 measurements,
increasing sample counts if the variability warrants it. Report min, median,
p95, number of runs, CPU utilization/clock policy, thread count and memory.

- Kernel compute: input arrays already in memory; stop after output materializes.
- Offload total: start before packing/quantizing (if needed for offload), stop
  after output is CPU-visible, unpacked and usable. Include cache maintenance,
  launch/wait, all host copies and transfers. Save compute-only numbers separately.
- Model latency: model-input tensor ready to final logits/class available.
- Application latency: decoded image ready through preprocessing and inference
  to predicted class. Report file I/O and camera capture separately unless the
  selected application requirement explicitly includes them.
- Model loading: report cold start separately; keep checkpoint loaded for
  steady-state inference, with the same policy in both CPU and hybrid runs.

Sequential estimate:

    T_offload = T_pack + T_upload + T_launch_wait + T_compute + T_download + T_unpack

Avoid double-counting: if launch/wait is measured from start to done it already
includes compute. Either subtract compute to estimate control overhead or
report that interval as one combined component. With overlap, component times
overlap; direct total wall time is authoritative.

For batch-1 sequential service, images/s = 1000 / mean_ms. Report median and
p95 latency separately; inverse median is not measured sustained throughput.

## Speed targets and research basis

The [official model table](https://github.com/microsoft/Cream/tree/main/TinyViT)
lists TinyViT-21M at 224x224 as 21M parameters and 4.3G MACs. Published desktop/GPU
throughput is not a DE1-SoC measurement. Our theoretical examples below are
derived estimates, not paper benchmarks.

For 256 PEs at an assumed 100 MHz, one useful MAC/PE/cycle gives 25.6 GMAC/s.
Mapping all 4.3G MACs ideally would take about 168 ms. This assumes all operations
map, perfect utilization, and zero transfers/overhead. The source forces
multipliers into logic, and no achieved clock/resource report is present.
Theoretical peak therefore cannot establish an inference target. At 30 FPS,
the 33.3 ms budget would require approximately 129 GMAC/s to execute all 4.3G
MACs in that interval. This illustrates why a 30 FPS commitment is premature;
it is not a proof about every possible CPU-FPGA architecture.

| Level | Proposed gate |
|---|---|
| Functional | Correct full-model classification and reproducible kernel results |
| Kernel performance | >=2x vs appropriate HPS kernel, including offload overhead |
| End-to-end | >=1.5x over the equivalent HPS quantized graph (33.3% less latency) |
| Stretch | >=2x end-to-end |
| Optional application | Static-scene classifier responding within 1 s at p95, if feasible |

The optional one-second requirement is a proposed user-experience choice, not
a literature-established need. Choose CPU-relative performance as the capstone's
primary target until HPS profiling is available. Example only: if HPS takes
3 s, the 1.5x target is <=2 s; that would still miss the optional 1 s target.

Let f be the fraction of original CPU latency eligible for offload, r the
compute speedup on that portion, and h all additional overhead divided by
original latency. Then:

    speedup = 1 / ((1-f) + f/r + h)

Example: f=0.7, r=4, h=0.1 gives 1.74x. If f=0.3, even infinite compute
speedup with zero overhead yields only 1.43x. Measure f before freezing 1.5x.

Benchmark transfer sizes and actual padded/packed layer shapes. A lower bound
for GEMM with one load per operand and one store per output is:

    operand/output bytes = K*N + N*M + K*M     (INT8 output)

Actual traffic can be higher because of bank padding, tile reloads and CPU
copies. Use measured bytes and transfer time, not DDR peak bandwidth. INT32
outputs or partial sums require a different byte count.

## Required result artifacts

- `configs/run_manifest.template.json`, completed per experiment.
- Raw per-run CSV, stdout/stderr, clocks, compiler flags, checkpoint/image hashes.
- Operator table with real shapes, CPU time and offload eligibility.
- Numerical error table for both arithmetic contracts and held-out accuracy.
- Latency breakdown and CPU/hybrid end-to-end comparison.
- Quartus resource and timing reports tied to the measured bitstream.

Keep raw files backed up in an agreed team location; commit concise summaries
and manifests. Ignore generated files only when they can be reproduced or have
a separate durable copy. Never fill missing FPGA/HPS fields with desktop timings.
