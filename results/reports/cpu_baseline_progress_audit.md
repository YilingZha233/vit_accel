# CPU baseline measurement progress: A–F audit

Audit prepared 2026-10-15. This records the existing repository kernel evidence
and the real checkpoint/HPS outputs supplied by the user in this conversation.
The later raw results are on the user's Windows/WSL computer; they were not
independently reopened on this Mac. Generated synthetic tooling fixtures in
this repository are excluded from the project's measured model results.

## Overall status

| Table | Current status | Main remaining work |
|---|---|---|
| A: Kernel | Preliminary synthetic baseline measured | Real captured tensors, wider operator coverage, optimized same-contract kernels, packing cost |
| B: Accuracy/quantization | FP32 10-image development check and cross-runtime output checks measured | Calibration, Q1–Q4/R1 experiments, larger disjoint accuracy evaluation, clipping diagnostics |
| C: HPS performance/resources | FP32 one/two-thread pilots measured | Final repeated measurements, native quantized CPU runs, application timing, stability/telemetry |
| D: Operator profile | Architecture/shape plan exists; no project HPS timings supplied | HPS operator self-times, counts, shares and profiling overhead |
| E: Fair comparisons | One/two-thread FP32 pilot comparison available | Native quantization and same-graph CPU/hybrid comparisons |
| F: Architecture/stages | Entire classifier executes and matches reference logits on 10 inputs | Confirm saved boundary checks; per-stage errors, times, bytes and placement |

## Confirmed FP32 model evidence

Model: TinyViT-21M-224, batch 1, 1000 classes. Checkpoint file supplied by the
user is `tinyvit21m.pth`, with SHA-256:

```text
eb20032633663fc3e1e49c1cb21332c11e6804c9f920b77da5f832dad248b897
```

Checkpoint loaded strictly into the selected architecture. Both ARM compilers
reported GCC 15.2.0. The runner built as a statically linked ARM EABI5 executable
and ran successfully on the board's ARMv7 Linux. The user successfully recovered
the results archive and ran both desktop/HPS numerical comparisons.

Accuracy input: ten repository-labelled ImageNet-class examples, all in the
development split. This is a deliberately selected functional set, not an
untouched representative ImageNet validation sample. No project calibration
split or larger evaluation has been supplied yet.

| Metric | HPS FP32, 1 thread | HPS FP32, 2 threads |
|---|---:|---:|
| Accuracy images completed / failed | 10 / 0 | 10 / 0 |
| Top-1 / top-5 on these examples | 100% / 100% | 100% / 100% |
| Prediction agreement with desktop reference | 100% | 100% |
| Logit RMSE against original FP32 reference | 1.542631237e-6 | 1.542631237e-6 |
| Maximum absolute logit error | 1.299381256e-5 | 1.299381256e-5 |
| All tested logits within atol=1e-4, rtol=1e-3 | Yes | Yes |
| Timed calls, benchmark mode | 20 | 20 |
| Explicit warmup calls | 3 | 3 |
| Mean model latency, ms | 7640.891425 | 4480.743421 |
| Median model latency, ms | 7640.523615 | 4480.801765 |
| p95 model latency, ms | 7643.868170 | 4482.645180 |
| Population standard deviation, ms | 1.538499 | 1.406738 |
| CV, percent | 0.020135 | 0.031395 |
| Images per summed inference second | 0.130874782 | 0.223177251 |
| Direct service-loop images/s | 0.130873717 | 0.223173983 |
| Model load time in benchmark process, ms | 731.18020 | 668.36103 |
| First inference in benchmark process, ms | 7663.66983 | 4488.87579 |
| Peak process RSS in benchmark, MiB | 154.605469 | 154.621094 |
| RSS just after model load, MiB | 112.878906 | 112.894531 |
| End-of-run RSS, MiB | 154.082031 | 154.097656 |
| Benchmark completed / failed | 20 / 0 | 20 / 0 |

Run IDs: `fp32_check10`, `fp32_check10_t2`, `fp32_t1_pilot`,
`fp32_t2_pilot`. Board paths were under `~/vit_model_fp32/results/` and recovered
WSL paths under `results/raw/hps_fp32_pilot/results/`.

The benchmark preloads ten input tensors; evaluation streams inputs. Their
peak RSS values need not be equal. End RSS is one snapshot, not a memory-growth
trace or a proof of stable long-term memory use. Loading used unspecified
filesystem-cache conditions, so label it process model-load time rather than
cold-storage loading. The first-inference call is separate from explicit warmups.

Two-thread mean-latency speedup is 1.705273x; latency reduction is 41.3584%.
This is CPU threading benefit, not quantization or FPGA acceleration benefit.

## A — Kernel baseline

Measured in `hps_board_run1`: 11 synthetic cases x two integer contracts, with
20 report samples/contract, one CPU thread and three warmups. The eight small
cases group 1000 calls/sample; the three larger cases use one call/sample.
Saved fields include shapes, contracts, simple-C implementation/build metadata,
raw timings, mean/median/p95/stddev/CV, useful MACs and effective GMAC/s. All
case/contract combinations reported bytewise verification. The saved numeric
audit found no discrepancies in 440 report samples and 66 pilot samples.

K,N,M use W[K,N] x X[N,M]; the sample-count field should be called n_samples
to avoid confusing it with contraction N. Historical `stage1_*` cases are
Stage 2 in the supplied architecture. Their three GEMMs exclude the rest of
the stage. No original-case CPU zero-padding is required; useful/executed GEMM
MACs can be derived from the loops, but hardware executed MACs are unmeasured.

Missing: checkpoint-derived weight/activation cases, Stage 3/4 and remaining
profitable operator shapes, an optimized implementation with the SAME integer
contract, and measured packing+compute where packing is required. Current code
does no separate packing; do not invent a packing latency or treat this as a
hardware tiling benchmark. The native FP32 ncnn model does not establish an
optimized INT8 kernel baseline under either old contract.

## B — Accuracy and quantization

Measured: FP32 top-1/top-5, ten-image count, failures, prediction agreement,
logit RMSE/max error, Wilson interval and paired reference/HPS comparison outputs.
The Wilson top-1 interval is approximately 72.25–100%. The empirical paired
bootstrap is [0,0] because no correctness differences exist in these ten inputs;
it does not establish a tight accuracy-loss bound beyond this selected set.

Not yet measured: any real-data calibration, Q1–Q4 accuracy, native ncnn INT8
accuracy, per-layer quantization errors or clipping rates. Quantized operators,
weight/activation granularity and calibration method/count must be specified
for every future variant. For F0 these quantization fields are not applicable.

The supplied Q1–Q4 tools simulate selected Linear operators only; bias and other
operators remain FP32. Standard ncnn-native quantization is a separate R1
configuration and is not automatically equivalent to Q1–Q4. Simulation timings
must not be entered as native INT8 performance.

## C — HPS performance/resources

Measured: FP32 threads 1/2, sample count, latency statistics, both throughput
boundaries, model load/first inference, process peak RSS, post-load/end RSS,
short-run failure counts and run IDs. The runner explicitly disables INT8,
FP16 and BF16 in these FP32 runs; INT8 coverage is not applicable to F0.

Derivable without rerunning inference: exact deployment bytes/MiB from the
converted `.param` plus `.bin`, or total deployed footprint including executable
as a separate metric. The user supplied rounded sizes of 109M for `.bin` and
70K for `.param`; obtain exact bytes before reporting precise MiB.

Missing: native quantized graph results, application preprocessing/postprocessing
and direct total latency, final repeated timing sessions, and longer stability
tests with memory/frequency/temperature observations when available. CPU
utilization and page-fault diagnostics are useful additional measurements.
The saved `hps_environment.txt` should be reviewed; its collection was requested,
but contents were not supplied here. Do not assume it establishes continuous
frequency/thermal behavior. Board calendar time was inconsistent; preserve
monotonic durations and use the known acquisition context for report dates.

## D — Operator profile

No HPS operator self-times, shares, profiling overhead or call counts have been
supplied. Full-model latency and standalone kernel timing cannot fill them.
The earlier architecture-inspection command was requested, but its saved output
was not supplied; inspect that artifact before declaring shapes/counts complete.

Required new measurement: separate instrumented HPS run, with deployed operator
names, shapes/dtypes, execution counts, non-overlapping timings and wall-time
denominator. A profiling ncnn build can provide layer timings; fused-layer
inclusive timings must not be mislabeled as primitive operator self-times.
MACs and tensor bytes are derived from verified graph shapes; padding, transfer
bytes, bank-capacity eligibility and fallbacks require explicit mapping analysis.
The converter's aggregate FLOPS estimate is not a measured layer-time profile.

## E — Fair comparisons

Available: one versus two threads in the same FP32 ncnn graph/timing scope,
plus original-reference versus HPS prediction/logit agreement. The two integer
kernel contracts can be compared on matched synthetic shapes, but their ratio
includes different algorithms and must not be called FPGA speedup.

Missing: FP32 CPU versus native quantized CPU, Q3 versus Q4 quality, model-size/
memory reduction ratios across variants, and quantized CPU versus the SAME graph
on hybrid execution. WSL PyTorch and HPS ncnn timings are different hardware and
runtime configurations; their ratio is not an FPGA acceleration claim.

The export's independent-window rewrite can duplicate shared weights and change
scheduling. Record actual exported storage/call structure; the first runtime
baseline is practical evidence, not proof of optimal CPU implementation.

## F — Architecture coverage and stage results

The whole stem -> Stage 1 (2 MBConv) -> Stage 2 (2 TinyViT) -> Stage 3 (6) ->
Stage 4 (2) -> pooling -> head LayerNorm -> 1000-class head executes end to end.
Output agreement on ten inputs has been checked. Individual expected stage
shapes exist in the architecture guide; they are not all independently observed
measurements until saved inspection data is reviewed.

Still needed: observed boundary shapes/layouts, invocation counts, per-stage
CPU timing/share, stage/intermediate errors, activation bytes and precision map.
Obtain CPU/FPGA placement from implementation decisions, and executed/useful
MACs from the actual padded/tiled schedule. These fields are not all experiments:
some are graph inspection, arithmetic derivation or mapping decisions. Hardware
execution and utilization remain pending until the accelerator runs.

## Recommended next work

1. Review/index existing files: environment, checkpoint/binary/model hashes,
   exact deployment bytes, inspection outputs and raw timing distributions.
   This closes report gaps without repeating measurements.
2. Profile native HPS operators/stages separately. Use measured bottlenecks to
   select real kernel cases and FPGA candidates.
3. Prepare disjoint calibration images and a broader development/evaluation
   sample; run numerical quantization/clipping studies. A tiny additional set
   can check the workflow; representative 256–512 calibration images and larger
   evaluation sets are quality-study starting budgets, not prerequisites for
   the first pilot. Keep the ten current images as development examples.
4. Deploy the supported native quantized candidate, verify outputs/accuracy and
   repeat HPS timing/memory measurements. Implement custom operators separately
   if the RTL-specific arithmetic is retained.
5. Benchmark captured real GEMMs with appropriate optimized same-contract CPU
   implementations; measure required packing/padding explicitly.
6. Freeze the graph/settings, expand correctness/accuracy evidence, and collect
   final repeated distributions and sustained resource observations. A sufficiently
   long timing run can double as the stability run if the required telemetry is
   recorded; avoid redundant benchmark runs.
7. Measure actual application preprocessing/inference/postprocessing and direct
   total latency. Add power only if energy efficiency is a claimed outcome.

FPGA correctness, transfer/control costs, achieved clock/resources and hybrid
speedup are subsequent accelerator measurements, not completed CPU evidence.
