# Setup and Git workflow

## Current machines

The folder created here is on your **Mac**:
`/Users/yilingzha/Documents/ECE496/ViT_fpga`.
Quartus and USB-Blaster setup happen on your **Windows 11 Intel computer**.
The measured embedded baseline runs on **DE1-SoC ARM Linux**.
Git synchronizes source between these machines; it does not synchronize installed
tools or make a desktop executable run on ARM.

## Run locally

On Mac/Linux:

```sh
python3 -m unittest discover -s tests -v
python3 scripts/generate_vectors.py
make
./build/bench_matmul data/generated/random_multi 100
```

On Windows PowerShell with Python 3 installed:

```powershell
py -3 -m unittest discover -s tests -v
py -3 scripts/generate_vectors.py
```

Use WSL or a Linux development machine for the POSIX C benchmark development
build. To measure HPS, copy the C source and vector directories to the board,
compile there with `gcc -O3 -std=c11 -Wall -Wextra bench_matmul.c -o bench_matmul`,
and run against the same vector directories. Very old BSPs may require `-lrt`.
Record the build command and compiler version. A cross build must use the BSP's
ARMv7 Linux ABI/sysroot, not a bare-metal toolchain or an AArch64 compiler.

## Remote repository

Requested remote: `https://github.com/YilingZha233/vit_accel.git`.
It advertised no refs when checked during setup. The local repository is
initialized on `main` and its `origin` points to this URL. Check with:

```sh
git status
git remote -v
```

Review files before publishing. When ready, make the first commit and push:

```sh
git add .
git commit -m "Add TinyViT baseline references and evaluation plan"
git push -u origin main
```

These publishing commands have not been executed by this setup. Authenticate
using your normal GitHub credential manager/browser flow. If someone adds a
remote commit first, fetch and integrate it instead of force-pushing.

After the first push, on Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force C:\fpga
Set-Location C:\fpga
git clone https://github.com/YilingZha233/vit_accel.git
Set-Location vit_accel
```

Use one work branch per independent change; pull before starting and commit/push
before switching machines. Keep the teammate RTL repository separate. Record
its commit in the interface contract rather than silently copying its sources.

## Hardware artifacts to request from the teammate

- Deployed RTL commit and parameters (especially bus width).
- Quartus version, exact FPGA device and board revision.
- Complete `.qpf`, `.qsf`, `.sdc`, `.qsys`, `.sopcinfo` and component metadata.
- Generated RAM/FIFO IP configuration plus reproducible build instructions.
- Boot image, FPGA configuration, driver and missing `matmul_min.c` if used.
- Actual HPS physical base/address units and validated register access code.
- Fitter/timing reports and existing measurements with timing boundaries.

Do not guess physical addresses from accelerator-local offsets. No MMIO or
DMA commands are included in this starter until that integration is confirmed.
