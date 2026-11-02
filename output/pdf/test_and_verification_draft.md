# IV. Test and Verification

**Evaluation completed to date**

TinyViT-21M has been deployed on the DE1-SoC HPS using ncnn, an open-source neural-network inference framework optimized for mobile and embedded devices. It loads the trained model and executes its operations on the ARM processor. With one CPU thread, mean model inference latency was 7.640891425 s for FP32 and 5.072312036 s for native ncnn INT8. Both pilots used batch size one, 224 x 224 inputs, three warm-ups, and 20 timed runs. These model-only timings exclude image loading and preprocessing and establish the measured CPU baseline for subsequent acceleration tests.

Prediction accuracy was also compared between FP32 and the project's selected Q1 INT8 method using the same ten labelled development images. Q1 applies symmetric per-tensor quantization to weights and activations in selected linear layers. This software accuracy experiment is separate from native ncnn INT8 timing. A larger evaluation set is needed to confirm accuracy; the expanded results will be entered below.

Expanded evaluation (to be filled): Number of images: _____; FP32 top-1 accuracy: _____%; Q1 INT8 top-1 accuracy: _____%; Q1-FP32 top-1 prediction agreement: _____%.

FPGA simulation, DMA validation, and integrated CPU-FPGA inference testing remain planned.

**Planned verification setup and module tests**

Verification will use the DE1-SoC, a computer with Quartus and an RTL simulator, a USB camera, and Linux. The current CPU baseline uses ncnn; the technical design proposes MNN for FPGA integration. A new CPU-only baseline will therefore be measured in the final integration runtime. CPU-only and hybrid tests will use one HPS inference thread, batch size one, and the same checkpoint, 224 x 224 resolution, preprocessing, and quantization settings. Software references, HPS monotonic-clock timestamps, FPGA cycle counters, and SignalTap traces will support diagnosis. Clock settings, software versions, and the bitstream will be recorded.

Module tests will progress from the MAC array and buffers to DMA/control and runtime integration. An independent integer reference will specify the selected Q1 scaling, rounding, accumulation, and saturation rules. Tests will cover zero, identity, extreme INT8, and seeded random inputs; dimensions around 16-element tile boundaries and the 64-element operand limit; and larger operations tiled in software. Each representative shape will receive 100 random trials plus directed cases. Integer outputs and DMA round trips must match their references exactly. Reset, consecutive commands, status polling, and Avalon-MM stalls will be tested. Quartus will check setup and hold timing at the proposed 90 MHz clock. The activation LUT will be checked over all 256 INT8 input codes and in bypass mode. If hardware softmax is included, its numerical tolerance will be defined before testing and its classification impact assessed through R3.

**System tests, conditions, and evidence**

A fixed set of 200 distinct, labelled images, excluded from quantization calibration, will test complete inference. FP32 and Q1 top-1 accuracy will be measured against ground-truth labels; Q1-FP32 prediction agreement will be reported separately. R3 will compare hybrid predictions with the matched quantized CPU reference. Timing tests will use ten warm-ups and at least 100 measured runs per configuration, with one inference thread, batch size one, identical preloaded inputs, and stable clocks. Model-only latency will be reported separately from application latency, which includes preprocessing and postprocessing through class output. Startup and model loading will also be reported separately.

Complete offloaded-operation latency will include setup, all tile transfers, FPGA computation, polling, and conversion. Kernel and transfer intervals will be measured separately without double-counting overlapping work. Means will determine latency acceptance; medians and 95th percentiles will describe variability. Three 60-second camera trials will measure sustained throughput, counting only fully classified frames and recording timestamps and drops. Evidence will include labelled FP32/Q1 predictions, matched CPU/hybrid results, timing logs, simulation traces, and Quartus resource/timing reports. The Design Fair demonstration will show live classifications and the measured comparison. Requirements and constraints will be assessed separately from objectives; published accelerators will not substitute for verification of this implementation.

**Verification table - planned final-system assessment**

IDs and thresholds follow the proposal SyRS. R1-R4 are requirements, C1-C4 are constraints, and O1-O3 are objectives. All activities below are planned.

| ID | System requirement / constraint / objective | Method and procedure | Acceptance criterion |
|---|---|---|---|
| R1 | Complete TinyViT classification | TEST: Run the full hybrid pipeline on the fixed 200-image set. | 200/200 images produce a valid class index and finite scores without failure. |
| R2 | Accelerate at least one MSA or MLP matrix operation | TEST: Compare the complete offloaded operation with the corresponding optimized HPS-only operation using one thread and >=100 runs each. | For at least one operation, mean hybrid latency <=0.50 x mean CPU latency, including all offload overhead. |
| R3 | Maintain classification consistency | TEST: Compare matched quantized CPU-only and hybrid top-1 predictions on the same 200 images. | At least 190/200 predictions agree (>=95%). This measures agreement, not ground-truth accuracy. |
| R4 | Measure computation and communication latency | TEST + ANALYSIS: Timestamp full inference, kernel execution, and transfers under the stated timing protocol. | Report each mean in ms from >=100 measured runs per configuration, with raw logs and timing boundaries. |
| C1 | Use the DE1-SoC platform | REVIEW OF DESIGN + TEST: Inspect the build target and demonstrate the integrated pipeline on the board. | ARM Cortex-A9 HPS executes host software and Cyclone V fabric executes the accelerator. |
| C2 | Fit available FPGA resources | ANALYSIS: Inspect final Quartus fitter reports and reconcile reported resource units with the SyRS. | Successful fit within device capacity and stated ceilings: 85K logic elements, 87 DSP blocks, 4.45 Mb block memory. |
| C3 | Use on-board memory where possible | REVIEW OF DESIGN + TEST: Inspect memory mapping and measure peak allocation during full inference. | Allocations fit 1 GB HPS DDR3 and, if used, 64 MB FPGA SDRAM; any external-memory need is documented and justified. |
| C4 | Execute a fixed TinyViT workload | REVIEW OF DESIGN: Compare saved model and execution configurations for both paths. | Same checkpoint, resolution, preprocessing, quantization, runtime, and one HPS inference thread; offload differences documented. |
| O1 | Reduce transfer overhead | TEST + ANALYSIS: Divide mean input/weight/output transfer time by mean complete accelerated-operation time. | Target: transfer fraction <=20%, including every required tile transfer. |
| O2 | Improve resource efficiency | ANALYSIS: Calculate utilization from final fitter reports for each resource class. | Target: logic, DSP, and block-memory utilization each <=90% of device capacity. |
| O3 | Sustain video throughput | TEST: Run three 60-second camera trials; compare with the remeasured CPU-only pipeline. | Target: >=60 completed classifications in each 60-second trial (>=1 FPS). Report measured speedup against the matched single-threaded CPU baseline. |
