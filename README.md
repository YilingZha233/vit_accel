# TinyViT CPU baseline and accelerator evaluation

Your workstream: reproducible CPU references, correctness tests, HPS measurements,
and end-to-end evaluation. Hardware implementation remains in
[j3rryhu/matmul-accel](https://github.com/j3rryhu/matmul-accel).

Start with [the task list](docs/TODO.md), [repository analysis](docs/teammate_review.md),
[measurement protocol](docs/baseline_plan.md), and [Git/Windows instructions](docs/setup.md).

**Confirmed scope:** TinyViT-21M with INT8 operands. The [step-by-step measurement
guide](docs/measurement_steps.md) separates Mac preparation, HPS measurements,
and full-model deployment. The full HPS model runtime is not present here.

## What works now

- Pure Python integer reference with no third-party packages.
- Two arithmetic contracts: full INT32 accumulation with one requantization, and
  the current accelerator's per-16-term requantization/saturation semantics.
- Deterministic vectors, manifests, and banked RAM images for the inspected RTL.
- A C single-thread starting baseline that verifies against the Python vectors.
- Comparison of an accelerator output-bank dump with the selected golden model.

No HPS timings, FPGA simulation results, FPGA timings, or full TinyViT inference
results have been measured by this project yet. Desktop timings are development
checks and must be labelled with the actual host. The C kernel is a simple
starting baseline; an optimized HPS implementation is required for final claims.

## Quick start on macOS/Linux or WSL

```sh
python3 -m unittest discover -s tests -v
python3 scripts/generate_vectors.py
make
mkdir -p results/raw
./build/bench_matmul data/generated/identity16 100 > results/raw/identity16.csv
./build/bench_matmul data/generated/random_multi 100 > results/raw/random_multi.csv
```

The generator prints **differences between the two arithmetic contracts**, not
FPGA failures. `rounding_gap` deliberately yields full=1 and RTL-compatible=0;
`saturation_cancel` yields full=0 and RTL-compatible=-1.

On Windows, the Python steps also work with `py -3` in place of `python3`.
Build the C program in WSL for a development check, or natively on the HPS for
the actual board measurement. These environments produce different executables.

To check a real board dump after a driver has been implemented:

```sh
python3 scripts/compare_output.py data/generated/identity16 board-output.bin --reference rtl
```

`board-output.bin` must contain the entire **2048-byte output bank region** in
accelerator-local byte order. The script does not communicate with hardware.
The driver must wait for completion, collect bytes correctly through the actual
bridge, and save this format. A generated expected image can exercise this
command, but that is a software check only.

## Layout

```text
baseline/                  arithmetic references and memory-image helpers
configs/                   proposed hardware contract and run metadata template
docs/                      analysis, priorities, reproducibility instructions
scripts/                   vector generation and result comparison
software/hps/              C code to compile/run on the Cortex-A9
tests/                     reference, layout, rounding and saturation tests
data/generated/            reproducible vectors (generated; ignored by Git)
data/images/               local images/calibration data (ignored)
models/checkpoints/        downloaded weights (ignored)
results/raw/               raw runs and logs (ignored; preserve externally)
results/reports/           reviewed summaries suitable for Git
external/                  optional pinned third-party checkouts (ignored)
build/                     compiled executables (ignored)
```

Create the data/model directories when needed. Keep checkpoint checksums,
dataset manifests, source revisions, and concise published results in Git.
Do not copy a second editable RTL tree into this repository. Agree on a pinned
hardware commit and update `configs/accelerator_contract.json` plus tests when
that interface changes. The contract JSON documents assumptions; it does not
automatically reconfigure the Python or C implementations.
