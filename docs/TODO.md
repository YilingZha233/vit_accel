# Baseline work checklist

These are ordered work sessions, not promised completion dates. Board/runtime
availability can change elapsed time. Start desktop work while board setup proceeds.

## Session 1: independently useful deliverable

- [x] Inspect and pin the teammate accelerator source.
- [x] Create arithmetic references, vectors, and a CPU benchmark starter.
- [ ] Run the README quick start yourself and understand each generated case.
- [ ] Review the numerical and address-contract findings with the teammate.
- [ ] Agree Y=W@X dimensions, 128/32-bit bus configuration, bank layouts,
  zero padding, bias handling, scale rounding, and when saturation happens.
- [ ] Choose who owns the HPS driver and DMA integration.
- [ ] Share the vector manifest and expected outputs through your agreed team process.

Deliverable: a shared correctness contract plus eight reproducible vector cases.

## Session 2: HPS CPU measurements

- [ ] Finish Windows Quartus/USB-Blaster setup and identify the physical board revision.
- [ ] Boot a compatible Linux image; record kernel, CPU clock, RAM and compiler.
- [ ] Compile `software/hps/bench_matmul.c` ON the board, or with its matching
  ARMv7 Linux sysroot/toolchain. Do not copy the Mac/Windows executable.
- [ ] Run all vectors on HPS; retain CSV and stderr metadata per run.
- [ ] Record this as simple single-thread C, compute-only; add an optimized
  cache-friendly/NEON or supported library baseline before final speedup claims.

Deliverable: first measured HPS CPU kernel table, with reproducible inputs.

## Session 3: full TinyViT feasibility and shape inventory

- [ ] Freeze TinyViT-21M INT8, 224x224, batch=1, checkpoint type/hash and source revision.
- [ ] Run the official FP32 reference on desktop with fixed preprocessing/images.
- [ ] Export logits, top-1/top-5 and selected intermediate tensors.
- [ ] Establish a CPU runtime that actually runs on 32-bit Cortex-A9 Linux.
  Check build/operator support and peak RAM early; do not assume PyTorch or ONNX
  Runtime desktop wheels install on this board. If necessary use a supported
  C/C++ runtime or a source build matching the BSP.
- [ ] Run one full image on HPS before committing to extensive accelerator tuning.
- [ ] Extract actual module/operator shapes from the pinned model. Include
  convolutions, QKV/projection/MLP, both attention products and software ops.
- [ ] Profile operator time separately from uninstrumented full-model latency.

Deliverable: HPS model feasibility result, shape inventory, and measured bottlenecks.

## Session 4: numerical assessment and targets

- [ ] Select a calibration set separate from held-out labelled evaluation images.
- [ ] Compare FP32, full-sum quantized, and current RTL-compatible arithmetic.
- [ ] Measure mismatch count, max/mean tensor error, saturation frequency,
  top-1 agreement and accuracy difference; record scale representability errors.
- [ ] Extend tests to at least 1000 varied accelerator transactions in simulation
  or hardware, including multiblock, capacity limits, reset and consecutive runs.
  The included 1000 Python cases are reference tests, not this hardware milestone.
- [ ] Revise proposed 1.5x total / 2x offloaded-operation targets using measured
  offload fraction, transfers, packing, driver cost, and achieved FPGA clock.

Deliverable: quantitative target justification and numerical-error report.

## Session 5: one real accelerator operation

- [ ] Obtain the working `.qsys`, `.sopcinfo`, bitstream, generated IPs, driver,
  tool versions and timing/resource reports from the hardware owner.
- [ ] Confirm address units and bus width with a documented register/RAM smoke test.
- [ ] Run identity16 first, then negative/extreme, multiblock and partial cases.
- [ ] Wait for completion correctly; STATUS.done is read-to-clear.
- [ ] Save a real output-bank dump and compare it with `compare_output.py`.
- [ ] Measure packing, upload, launch/wait, compute, download, unpacking, and
  direct total wall time. Apply DMA allocation/cache rules if DMA is used.

Deliverable: bit-exact hardware comparison and a fair kernel speedup table.

## Later: complete capstone evaluation

- [ ] Integrate one profitable TinyViT layer, then additional layers selectively.
- [ ] Compare the same quantized graph on HPS and hybrid execution.
- [ ] Run held-out accuracy evaluation and full latency distribution.
- [ ] Add DMA overlap/weight reuse only after sequential execution is correct.
- [ ] Add camera capture after saved-image inference is stable.
- [ ] Produce reproducible plots, resource reports, limitations and demo steps.

Avoid duplicating RTL, training a new model, adding HBM, or making camera capture
the critical path for your baseline assignment.
