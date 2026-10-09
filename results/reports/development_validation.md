# Development validation — 2026-10-08

Environment: macOS / Apple Silicon, Python 3.9.6, Apple Clang 14.0.3.
Build: `-O3 -std=c11 -Wall -Wextra`.

| Check | Observed result |
|---|---|
| Python unit tests | 10 passed, including 1000 seeded single-contraction-block cases |
| Generated vector sets | 8 generated with SHA-256 manifests |
| C full-sum reference vs Python golden | All 8 vector sets passed |
| C RTL-compatible reference vs Python golden | All 8 vector sets passed |
| Comparator using generated expected bank image | Passed (software-only loopback) |
| Comparator using deliberately corrupted image | Rejected with one mismatch |
| Configuration JSON, including duplicate-key check | Passed |

The deliberate arithmetic differences were reproduced:

- `rounding_gap`: full-sum result 1; current RTL-compatible result 0.
- `saturation_cancel`: full-sum result 0; current RTL-compatible result -1.

Local timing smoke runs used 10 warmups and 100 repetitions per contract/case.
The raw logs are in ignored `results/raw/desktop_smoke/`; these are development
measurements, not publishable DE1-SoC speedup evidence.

No RTL simulation was run (Icarus Verilog was not installed). No Quartus build,
HPS execution, FPGA transaction, DMA transfer or full-model inference was run.
Both the software and hardware arithmetic contracts still need team agreement.
