# Measuring full inference, FPGA computation and DMA on DE1-SoC

## 1. Starting point and what must be implemented

The user confirmed that the team does **not** currently have a runnable full TinyViT HPS application or an integrated FPGA/DMA board test. The existing `bench_matmul` program only runs CPU matrix routines. Changing its repetition count cannot produce model inference, FPGA or DMA measurements.

This document is an implementation and measurement procedure. C-like examples below are **pseudocode for integration**, not commands or a supplied working TinyViT/DMA application. The report's new measurement tables must remain pending until these prerequisites work.

| Workstream | Required artifact | First passing test |
|---|---|---|
| CPU model | ARMv7 Linux inference executable, exact checkpoint/converted model, quantization parameters, preprocessing | One image produces validated output scores on HPS |
| FPGA integration | Quartus project, generated RAM/IP, Platform Designer system, programmed bitstream, timing report | One supported matrix operation completes and matches CPU |
| DMA integration | Driver, mapped buffers, address map, descriptor setup and completion handling | Known input data move in both directions with exact byte equality |
| Hybrid model | Runtime operator adapter and complete tiler connected to the working accelerator | One image follows the same graph as quantized CPU and passes output checks |

You can develop the CPU-model and FPGA-board workstreams independently. Measure CPU inference once its runtime works; measure tile compute/DMA once board integration works. Hybrid end-to-end timing requires both.

Windows/WSL cross-compiles ARM applications. Windows/Quartus builds and programs hardware. HPS Linux executes the measurements. Mac/Windows analyzes exported results. Python need not be installed on HPS. USB or the already-proven UART archive method can transfer files; Ethernet remains optional.

## 2. Freeze the experiment before development

1. Use the team's TinyViT-21M model; propose 224 × 224 inputs and batch size 1 and record the choice.
2. Pin the actual checkpoint, model-source revision, preprocessing, class labels and hashes. The exact checkpoint has not yet been selected in the board measurements.
3. Define the INT8 graph: which operators are integer, scales and zero points, bias precision, rounding, saturation, residual arithmetic and any floating-point fallbacks. Attention probabilities need their own numerical treatment.
4. Decide whether hardware will preserve full INT32 accumulation across tiles or retain the present tile-clipping behavior. A conventional runtime's default quantized GEMM must not silently be compared to a different RTL contract.
5. Establish two equivalent execution modes in the future application: CPU-only and hybrid. The same saved inputs, model, precision and non-offloaded code must be used; only selected operations change backend.
6. Record thread count, clock/governor if available, affinity policy, driver/runtime/compiler versions, input-list hash, bitstream hash and weight-residency policy. Do not infer active CPU MHz from the processor model or BogoMIPS.

Keep the existing simple C CPU baseline as historical evidence, then add an optimized equivalent CPU/runtime comparison. Native INT8 execution must be confirmed; float emulation of INT8 rounding is a numerical reference, not a native INT8 performance baseline.

## 3. Get one complete CPU inference working

1. On the development computer, run the official [TinyViT inference example](https://github.com/microsoft/Cream/blob/main/TinyViT/inference.py) with the selected checkpoint and its matching preprocessing. This establishes a reference; it does not itself implement the HPS INT8 runtime.
2. Save one preprocessed input, output scores (logits), and predicted class. Then prepare a small fixed functional image set. Store decoded images on the board for the timing boundary defined below.
3. Select an ARMv7/Cortex-A9-capable runtime or implement the required graph. Verify export/operator support first, including convolutions, attention products, softmax, normalization, GELU, reshapes and residuals. For example, ncnn documents ARM Linux builds, but that alone does not establish TinyViT or the current RTL arithmetic compatibility. [ncnn build documentation](https://github.com/Tencent/ncnn/wiki/how-to-build)
4. First cross-compile a minimal runtime example and run it on the board. The old glibc 2.15 environment remains a compatibility constraint. Success of the small statically linked C benchmark does not prove a larger C++ runtime's compatibility.
5. Deploy the model and execute one full FP32 inference; compare outputs to the development reference using documented tolerances. Then implement/calibrate the chosen INT8 path and validate it on held-out inputs. If an operator needs a fallback or a custom implementation, record it explicitly.
6. Before performance measurements, establish that CPU-only quantized execution is correct and that the actual operator shapes match the selected workloads. This deployment can take substantial implementation work; it is not just another benchmark command.

**Deliverable:** a real HPS application entry point that takes one decoded image, preprocesses it, executes the complete graph synchronously, and returns output scores and a class. Its filename and CLI will be documented when implemented; no `tinyvit_hps` executable currently exists in this repository.

## 4. Instrument CPU end-to-end inference

### Step 4.1: use precise boundaries

- **Application latency:** decoded image ready in HPS memory → predicted class available on CPU. Includes preprocessing, complete inference and class selection; model already loaded.
- **Model latency:** preprocessed tensor ready → output scores usable by CPU. For hybrid execution this includes all internal packing, transfers and waits.
- **Load/decode times:** model loading and image decoding are recorded separately, not silently added to steady-state inference.

All accelerator work must be complete before stopping a timer. Timing only an asynchronous launch measures submission, not inference.

### Step 4.2: insert timing around actual functions

Use the same HPS monotonic timer principle that already worked in `hello_arm`. Use a 64-bit duration representation. Example instrumentation to implement in C/C++:

```c
/* PSEUDOCODE: replace functions with the actual model application's API. */
load_model_once();
prepare_decoded_images();
allocate_reusable_buffers();

for (i = 0; i < warmups; ++i) {
    preprocess(image_for(i));
    execute_complete_graph_and_wait(mode);
    choose_class();
}

sequence_start = monotonic_ns();
for (i = 0; i < repetitions; ++i) {
    t0 = monotonic_ns();
    preprocess(image_for(i));
    t1 = monotonic_ns();
    execute_complete_graph_and_wait(mode);
    t2 = monotonic_ns();
    prediction = choose_class();
    t3 = monotonic_ns();

    /* Store timings and prediction in preallocated memory; no printing. */
    record[i] = {t1-t0, t2-t1, t3-t2, t3-t0, prediction};
}
sequence_end = monotonic_ns();
/* Only now write CSVs, checksums and saved outputs. */
```

Allocate/pack dynamically inside the model call if that is genuinely required during deployment; do not remove real costs solely for a favorable benchmark. Document what is preallocated or preloaded. Consume outputs so computation cannot be optimized away. Use a separate validation run or explicitly document any validation overhead included within the measured path.

### Step 4.3: run and summarize

1. Run one-image correctness first, then a 3-sample/1-warmup pilot to estimate duration.
2. Collect a preliminary 20-sample/3-warmup run; aim for 100 samples/10 warmups for the final distribution if affordable. Each sample is one complete inference, not a grouped microbenchmark.
3. Use a fixed image sequence for both modes, one inference at a time. Record when repeated images are used; timing samples do not imply unique accuracy-test images.
4. Save per-sample fields: `run_id,mode,sample,image_id,preprocess_ms,model_ms,postprocess_ms,app_ms,predicted_class,status`.
5. Separately save `sequence_wall_ms,completed_images,warmups,repetitions`. Sequential images/s = `1000 * completed_images / sequence_wall_ms`. This includes loop bookkeeping but excludes loading and later CSV export. Do not substitute `1000 / median_ms` for measured sequential throughput.
6. Report median, mean, nearest-rank p95, min, standard deviation and CV for model and application times. With 20 samples, p95 is the 19th sorted value and is preliminary. Failed/time-out runs must be reported, not silently dropped.
7. Record process peak RSS separately if supported. `/proc/meminfo` describes the system, not the inference process's peak usage.

An operator-profile pass should be separate from final timing. Group non-overlapping self-times by convolution, projections, attention products, MLP, nonlinear operations and layout/copies. Do not add parent block duration to its children's duration. The measured eligible time fraction f determines whether 1.5× overall speedup is plausible.

## 5. Build and verify the FPGA board path

### Step 5.1: obtain the real integration artifacts

Create or obtain the complete Quartus/Platform Designer project, generated RAM/FIFO IP, HPS bridge/reset configuration, address map, driver and bitstream. The repository's local accelerator offsets are **not** confirmed HPS physical addresses. Do not derive `/dev/mem` writes or DMA descriptors from guessed addresses or earlier register-offset examples.

Start with one supported tile, such as the existing `identity16` vector, not a full FC1 matrix. Confirm the deployed bus width, byte addressing, byte enables, signed operand layout, bank mapping, status acknowledgements and reset behavior. CPU MMIO writes may help initial bring-up, but their elapsed time is programmed-I/O time, not DMA latency.

### Step 5.2: implement the DMA driver/buffer contract

The Linux driver must allocate/map memory appropriately for the actual DMA device and return device-visible addresses. A userspace `malloc` pointer is not a DMA address. Follow the target BSP's DMA mapping and synchronization rules, including CPU/device ownership and cache maintenance. The modern Linux guide explains these principles, but exact APIs must match this older 3.12 BSP. [Linux DMA API HOWTO](https://www.kernel.org/doc/html/v6.12/core-api/dma-api-howto.html)

Distinguish **data direction** (HPS DDR → FPGA input buffer, or FPGA output buffer → HPS DDR) from which bridge/master issues transactions. An FPGA-side DMA master can read HPS memory to load inputs. Label results by source/destination, not ambiguous labels such as “f2h DMA.”

Verify transfer-only patterns byte-for-byte before enabling computation. Use changing patterns and sentinels outside valid regions to detect stale data, incorrect lengths or out-of-bounds writes. Verify completion/error status and impose a timeout. Synchronize device-written data for CPU access before checking it.

### Step 5.3: verify one computation

1. Reset or clear stale state according to the deployed interface.
2. Load a known tile and its scale through the verified data path.
3. Launch the accelerator and wait for real completion.
4. Retrieve the full valid output and normalize its layout.
5. Compare every element with the same-contract CPU reference. Repeat with negative, saturation, rounding and back-to-back cases.

The existing reference configuration describes done as read-to-clear. Confirm that on the deployed revision and avoid multiple readers consuming completion. A timer read must not accidentally acknowledge or erase status.

## 6. Measure FPGA kernel latency and clock

### Step 6.1: add hardware timestamps

Implement a free-running 64-bit counter in the accelerator clock domain and latch start/end values per request. Define start as the cycle in which a valid command is accepted, and end as the cycle when its final output is committed to the output buffer. Document cycle inclusion so a known one-cycle test gives the intended duration. Preserve the latched result until software explicitly acknowledges it; expose counter data across clock domains safely and read it atomically.

This measures a whole tile kernel, including controller activity, pipeline fill/drain and internal stalls. If measuring only PE-array active cycles, expose a separate counter and label it accordingly. A CPU timer around start/poll/done includes bus access, polling and scheduling; call that launch-to-completion wall time, not isolated FPGA compute time.

Use simulation/SignalTap to check counter boundaries, then collect counters from physical board executions. Do not report simulation cycles alone as measured board latency. Check reset, wraparound and repeated requests.

### Step 6.2: record the operating frequency correctly

Save the programmed clock/PLL configuration and post-fit TimeQuest timing evidence for the bitstream actually used. Record both the **operating accelerator clock** and the **timing-analysis Fmax/slack**. Fmax is an estimated timing limit, not necessarily the frequency at which the board is running. Do not assume the design achieved 100 MHz or convert time using the Fmax value if it runs at another frequency. Confirm the clock source and programmed configuration; independently measure frequency when needed. [Intel Standard Edition Timing Analyzer guide](https://www.intel.com/programmable/technical-pdfs/683068.pdf)

```text
kernel_ms = measured_cycles / operating_clock_hz × 1000
          = measured_cycles / (operating_clock_MHz × 1000)
```

Illustration only: 25,000 cycles at 50 MHz is 0.5 ms. It is not a project measurement.

### Step 6.3: collect samples

After correctness passes, collect a pilot and then repeated requests. Save `run_id,case,sample,contract,K,N,M,tile_id,cycles,clock_hz,kernel_ms,launch_wait_ms,status,verified`. Include input-dependent stalls when present. First-load, steady-state resident-weight and streaming tests are different configurations and must not be pooled.

Whole FC1/FC2 workloads exceed current buffers. Their result must include every required tile and correct cross-tile accumulation. Summing non-overlapping tile cycles measures total accelerator service time; wall time remains necessary when tiles overlap or the accelerator waits for CPU/DMA work. Never compare one tile's latency to a whole CPU matrix call.

## 7. Measure DMA overhead and effective bandwidth

### Step 7.1: establish a transfer-only experiment

Use real allocated buffers and the intended bridge/path, with computation disabled. Test DDR → FPGA and FPGA → DDR separately. Sweep only valid lengths that fit the endpoint's allocated buffer, descriptor limits and alignment. Begin with actual tile sizes; extend to larger buffers only when the integrated design provides them. Record descriptor count, alignment, burst configuration, buffer cache policy and total bytes.

### Step 7.2: record CPU-visible timing boundaries

For one direction, time the actual driver sequence:

```text
t0: begin per-transfer packing / required preparation
t1: packed data ready
t2: required cache ownership synchronization completed
t3: descriptor prepared and transfer submitted
t4: actual completion observed, including driver wait
t5: required CPU visibility synchronization completed
```

Ownership transitions differ by direction; record the real sequence, including any prerequisite synchronization before a device write. Fields that do not apply should be explicitly marked rather than invented. Driver submission may overlap transfer, so `t4-t3` is the remaining observed wait, not necessarily total engine transfer duration. The complete CPU-visible transaction interval is `t5-t0`; for transfer service excluding packing use `t5-t1` and label that boundary.

For a separate hardware engine interval, latch request-accept and transfer-complete events in a known clock domain. Completion must include required write responses/output visibility, not merely descriptor acceptance. Only that measured interval may be labelled hardware DMA-engine latency. If instrumentation is absent, leave the engine-only field unavailable and report CPU-visible latency.

### Step 7.3: validate and calculate bandwidth

Verify bytes outside timing after completion and cache synchronization. Change input patterns between repetitions. Warm up, then collect at least 20 samples initially; 100 per valid size/direction is a useful final objective.

```text
effective_MB_per_s = transferred_bytes / (elapsed_ms × 1000)
```

Use decimal MB and state the timing boundary. For repeated transfers, aggregate bandwidth is total bytes divided by total measured duration, not the arithmetic mean of per-transfer bandwidths. Example only: 4,096 bytes in 0.10 ms is 40.96 MB/s. A round trip moves bytes in both directions; report directional times as well as the total rather than treating it as one transfer.

Save `run_id,sample,direction,bytes,descriptors,pack_ms,sync_ms,submit_ms,wait_ms,transaction_ms,engine_cycles,engine_clock_hz,status,verified`. Leave unsupported hardware counters blank with an explicit explanation. The reported latency includes software overhead unless the hardware engine boundary is separately available.

## 8. Measure complete matrix offload

After DMA and compute are independently verified, instrument the complete operator replacement. Start with CPU-visible matrices ready and stop with the complete output CPU-visible in its required layout. Include all packing, transfers, cache operations, control, compute, multi-tile accumulation and unpacking.

1. Run a non-overlapped diagnostic pass to separate stages clearly.
2. Run the intended optimized pipeline, including double buffering if implemented, and measure total wall time directly.
3. Record weight/input/output bytes actually transferred, including repeated weights, padding and reloaded tiles; do not substitute a theoretical minimum traffic estimate.
4. Compare with a CPU routine using the exact same matrices and arithmetic. Verify outputs before accepting timing results.
5. Report median/p95 total offload time alongside hardware compute cycles and DMA intervals. Component intervals may overlap; their sum need not equal wall time. Do not present an unexplained residual as measured “DMA overhead.”

Use `CPU median / offload median` for operation speedup. Preliminary 2× budgets against board run 1 are 26.739 ms for the selected QKV window, 565.130 ms for FC1 and 568.794 ms for FC2. Also compare against an optimized CPU baseline before final claims.

## 9. Integrate and measure the hybrid model

1. Replace selected CPU operations with the validated full-operation offload adapter. Retain the same graph and arithmetic; no skipped layers, altered resolution or changed weights.
2. Compare offloaded intermediate tensors and complete model output on the functional image set. Then perform the planned held-out accuracy evaluation.
3. Run the identical application-timing harness from section 4 with the fixed input sequence. Ensure every asynchronous operation has completed before returning scores.
4. Alternate CPU and hybrid measurement sessions when practical to assess environmental drift. Keep thread count and non-offloaded work equivalent; do not compare a two-thread hybrid pipeline only to an intentionally restricted single-thread CPU without also reporting a fair optimized CPU comparison.
5. Report CPU and hybrid application/model median/p95, sequence duration, images/s, failures, and speedup. The proposed target is ≥1.5× median application speedup with no p95 regression, subject to the separate model-quality requirement.

## 10. Collect files and populate the report

Keep each configuration in a new result directory. Export original samples, run metadata, bitstream and executable hashes, correctness logs, and the Quartus timing/utilization reports. The existing UART archive procedure can transfer the directory without Ethernet or changing the USB filesystem.

Before filling the professor report, check units, boundaries, equal sample counts, error rates, clock conversion, contract consistency and whether transfer/computation overlap. Compare full operations, not mismatched tile scopes. Mark absent measurements as pending or unavailable; never use 0 to mean unmeasured.

No application, driver, bitstream or cycle-counter implementation is delivered by this guide. These are the required next implementation artifacts, after which the defined data can be collected on the board.
