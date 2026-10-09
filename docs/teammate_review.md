# Teammate repository analysis

Inspected 2026-10-08 at commit
`966e3459be039e9e51f74cff9ce833c085a0178a` (commit date 2026-09-28).
This is a source inspection, not a claim that the design passes simulation,
fits the FPGA, meets timing, or has demonstrated end-to-end inference.

## Scope and division of work

The repository implements a signed INT8, weight-stationary matrix accelerator
with a 16x16 PE array, 32-bit partial sums, banked weight/input/output storage,
load/dispatch logic, a tile controller, an Avalon-MM slave, generated control
registers, and cocotb tests. Address decoding is one component of this system.

Useful ownership split:

| Teammate | Baseline/evaluation owner |
|---|---|
| RTL, register interface, buffering, Quartus integration | Independent golden outputs and numerical contract |
| Board driver/transfer integration (agree owner) | CPU benchmarks and shared vectors |
| Cycle counts, resource/timing reports | End-to-end timing, accuracy, fair comparisons |

## Findings that affect integration immediately

1. **Arithmetic changes across tiles.** The current output loader scales and
   clips each 16-term partial sum to INT8, then saturates each addition to the
   previous INT8 result. It does not retain the entire contraction in INT32.
   A matching CPU model verifies implementation conformance, but does not prove
   equivalence to an ordinary quantized linear layer. Measure model accuracy
   separately. Prefer full INT32 partial-sum retention across tiles if the
   agreed project contract requires ordinary GEMM semantics.

2. **README transfer width is stale.** README describes 32-bit buffer words.
   `accel_top.sv` defaults `AVS_DATA_WIDTH=128`, test code uses 16-byte words,
   and the actual behavioral RAM ports are 128 bits wide. Do not write a
   32-bit MMIO driver based only on README. Confirm generated component address
   units, byte enables, width adaptation, and the exact deployed parameters.

3. **Weight layout is banked in the source.** README describes flat row-major
   storage. `weight_buffer_32_bank.v` and `write_weights()` instead map element
   W[k,n] to local offset `(k % 16)*1024 + (k // 16)*N + n` inside WBUF.
   The first bank starts at local base 0x1000. The per-bank bound is
   `ceil(K/16)*N <= 1024`, which is stronger than a total-byte check alone.

4. **HPS address mapping remains unverified.** A test comment refers to CPU
   byte offsets of `4*a`, while the RTL labels its local addresses as bytes.
   The generated Platform Designer component/system is not present here, so
   the CPU mapping cannot be settled from this snapshot. Never multiply the
   local offsets by four just because that comment says so. Obtain the `.qsys`,
   `.sopcinfo`, component metadata, and actual host code.

5. **TinyViT layers exceed the on-chip working set.** For Y[K,M]=W[K,N]X[N,M],
   source-derived limits are `ceil(K/16)*N <= 1024`, `(N/16)*M <= 256`, and
   `ceil(K/16)*M <= 128`, with N padded to a multiple of 16. A whole stage-1
   MLP is not one hardware call. A 16x128 weight tile with 128x16 activations
   fits these limits. Host tiling and packing must be included in latency.

6. **Reproducible board artifacts are absent from the inspected tree.** I found
   no complete Quartus project, Platform Designer system, generated synthesis
   RAM/FIFO IPs, HPS application, DMA implementation, or hardware measurement
   report. The testbench provides behavioral IP replacements. Comments mention
   a `matmul_min.c` hardware test that is not included. This does not establish
   that the teammate has not run hardware privately; ask for those artifacts.

7. **Documentation maturity varies.** TODO.md still describes a disconnected
   32x32 design, while the current top connects a 16x16 datapath. Use source and
   executable tests to establish the current state. The README also has an
   output-row prose description inconsistent with its formula: within one bank,
   a row's M elements are contiguous. The formula is what the helper follows.

8. **Multipliers are forced into logic.** `pe.v` uses `multstyle="logic"`.
   Do not assume DSP utilization or a 100 MHz achieved clock. Ask for fitter and
   timing reports; evaluate DSP mapping using actual device packing/resource
   reports rather than counting each INT8 multiply as one whole DSP block.

## Arithmetic example to bring to the team

Set N=32, M=K=1, q16=32768 (scale 0.5). Let W[0]=W[16]=1, all other
weights zero, and X all ones. Full accumulation gives floor(2*0.5)=1.
Current per-tile arithmetic gives floor(1*0.5)+floor(1*0.5)=0.
The included `rounding_gap` vector captures this without saturation.

For saturation, W=[127 repeated 16, -127 repeated 16], X all 127, scale 0.5:
full-sum output is 0; per-tile output is 127 + (-128) = -1.

## Evidence links pinned to the reviewed commit

- [Top and parameters](https://github.com/j3rryhu/matmul-accel/blob/966e3459be039e9e51f74cff9ce833c085a0178a/rtl/accel_top.sv)
- [PE arithmetic](https://github.com/j3rryhu/matmul-accel/blob/966e3459be039e9e51f74cff9ce833c085a0178a/rtl/pe.v)
- [Output arithmetic](https://github.com/j3rryhu/matmul-accel/blob/966e3459be039e9e51f74cff9ce833c085a0178a/rtl/output_loader.v)
- [Weight banking](https://github.com/j3rryhu/matmul-accel/blob/966e3459be039e9e51f74cff9ce833c085a0178a/rtl/weight_buffer_32_bank.v)
- [Tests and packer](https://github.com/j3rryhu/matmul-accel/blob/966e3459be039e9e51f74cff9ce833c085a0178a/tb/test_accel_top.py)
- [Behavioral IPs](https://github.com/j3rryhu/matmul-accel/blob/966e3459be039e9e51f74cff9ce833c085a0178a/tb/models/buffer_ram_models.v)
