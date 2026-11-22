# Faster HPS camera demonstration: persistent inference

This update preserves the original `model_runner`, `hps_camera_viewer.py`, model files and preprocessing. It adds `camera_runner` and `hps_camera_live.py`. Use both new components together; the old runner does not speak the new protocol.

## What changed

- Model parameters/weights load once when the service starts.
- Exactly one extractor/inference per selected frame; no extra initial inference or repeated warmup. The first real request is marked cold in the CSV.
- One persistent SSH execution channel carries normalized tensors and logits. No per-frame process launch, TSV, tensor file or result-directory creation.
- JPEG capture drains into a single latest-frame slot. Preview decodes at its display rate; inference independently decodes the selected stable JPEG. Intermediate preview frames are dropped. This bounds application memory but does not prove that camera/FFmpeg/SSH buffers have zero latency.
- Preview defaults to 20 refreshes/s rather than 10. Unchanged frames and completed-result crops are not rebuilt on every refresh.
- Separate decode/preprocessing/HPS/round-trip timings, matching frame IDs, and optional CSV logging.

Capture is still on HPS, preprocessing on Windows, inference on HPS, and display on Windows. Same MJPEG 640x480 at the camera's advertised 30 fps. Model input remains normalized RGB CHW FP32 even for runtime-int8. No FPGA acceleration or model replacement.

## 1. Put the new files on Windows

Extract `hps_camera_persistent_update.zip` to:

```text
C:\Users\tiffa\Downloads\hps_camera_persistent_update
```

Keep the old viewer and environment in Downloads. The extracted folder contains the new viewer, label file, timing-summary tool, runner C++ source, toolchain file, tests and this guide. **It does not contain a board-ready ARM executable.** The runner was compiled/tested on the local Mac; do not copy that host executable to HPS.

## 2. Build camera_runner in your existing WSL model project

Use the same ARMv7 toolchain and ncnn build that produced the working model_runner. No PyTorch installation on HPS is needed. Open **Ubuntu/WSL**, not PuTTY or PowerShell.

The earlier project location was `/home/tiff/cpu_baseline_tools`; substitute your actual location. Confirm the existing ARM ncnn installation:

```sh
cd /home/tiff/cpu_baseline_tools
ls build/ncnn-arm-install/lib/cmake/ncnn/ncnnConfig.cmake
command -v arm-linux-gnueabihf-g++
```

If either is missing, stop and locate the working build/toolchain. Do not install an unrelated ncnn version or use the Mac host build. If needed, locate it with:

```sh
find "$HOME" -name ncnnConfig.cmake -print 2>/dev/null
```

Choose the ARM installation, not `ncnn-host-install`. If your working compiler has a different name, export `CC` and `CXX` to its absolute compiler paths first, as in your previous build.

Configure a separate build directory:

```sh
project_root="$PWD"
camera_package=/mnt/c/Users/tiffa/Downloads/hps_camera_persistent_update
cmake -S "$camera_package/software/hps/model_runner" \
  -B "$project_root/build/camera-live-arm" \
  -DCMAKE_TOOLCHAIN_FILE="$camera_package/configs/cortex_a9.toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$project_root/build/ncnn-arm-install/lib/cmake/ncnn" \
  -DCMAKE_EXE_LINKER_FLAGS=-static
cmake --build "$project_root/build/camera-live-arm" --target camera_runner --parallel 2
file "$project_root/build/camera-live-arm/camera_runner"
cp "$project_root/build/camera-live-arm/camera_runner" "$camera_package/camera_runner"
```

Expected: a 32-bit ARM Linux executable, ideally statically linked. The recipe requires static runtime libraries in the existing toolchain; if linking fails, retain the full error. Do not blindly remove `-static`: dynamic builds need board-compatible libc/libstdc++/OpenMP libraries. Reuse any additional ABI/sysroot settings from your known-working build. Static linking alone does not certify kernel compatibility.

The package includes the unchanged benchmark source because its CMake project declares both targets; the command builds only camera_runner. No old board binary is replaced.

## 3. Transfer and test startup

Close the old viewer before live testing, so it releases the camera. In **Windows PowerShell**:

```powershell
cd "$env:USERPROFILE\Downloads\hps_camera_persistent_update"
scp .\camera_runner ubuntu@192.168.137.39:/home/ubuntu/vit_camera/build/camera_runner
```

Use `scp -O` if the old board SSH server lacks SFTP. Use the current board address if DHCP has changed it.

In **PuTTY**:

```sh
chmod +x /home/ubuntu/vit_camera/build/camera_runner
/home/ubuntu/vit_camera/build/camera_runner
```

Expected output is a usage line naming `camera_runner`. This checks executable startup only. If it reports `Illegal instruction`, missing libraries, or cannot execute, stop and retain that error. Do not run the service manually with model arguments in PuTTY: its stdout is binary. The Windows viewer launches it automatically.

## 4. Test the same saved snapshot first

In **Windows PowerShell**:

```powershell
cd "$env:USERPROFILE\Downloads\hps_camera_persistent_update"
```

Run this as one line:

```powershell
..\.venv-camera\Scripts\python.exe .\scripts\hps_camera_live.py --host 192.168.137.39 --user ubuntu --password --runner /home/ubuntu/vit_camera/build/camera_runner --param /home/ubuntu/vit_camera/model/int8.param --model /home/ubuntu/vit_camera/model/int8.bin --precision runtime-int8 --image ..\camera.jpg --labels .\imagenet_labels.txt --timings .\snapshot_new.csv
```

The existing Windows environment already has numpy, Pillow and Paramiko; no new Python packages are required. Host-key checking still uses Windows `.ssh/known_hosts`. Enter the board password when prompted. Keep the model paths and labels identical to the working demo.

Compare the top-5 names/scores with the original viewer on the same `camera.jpg`. The model and preprocessing are unchanged; unexplained numerical differences need investigation. The initial prediction is cold, but there is no discarded extra inference. A new CSV path is required on every run; the viewer refuses to overwrite evidence.

## 5. Run live mode and collect measurements

Close the snapshot viewer. In **Windows PowerShell**, run as one line:

```powershell
..\.venv-camera\Scripts\python.exe .\scripts\hps_camera_live.py --host 192.168.137.39 --user ubuntu --password --runner /home/ubuntu/vit_camera/build/camera_runner --param /home/ubuntu/vit_camera/model/int8.param --model /home/ubuntu/vit_camera/model/int8.bin --precision runtime-int8 --device /dev/video0 --format mjpeg --size 640x480 --fps 30 --preview-hz 20 --labels .\imagenet_labels.txt --timings .\live_new.csv
```

Let it complete at least 30 predictions, then close the window normally. Summarize:

```powershell
..\.venv-camera\Scripts\python.exe .\scripts\summarize_camera_timings.py .\live_new.csv
```

The summary excludes the first cold request by default. It prints median/p95 of completed requests, not failures or an overall camera FPS measurement. For the one-image snapshot use `--include-first`.

Expected benefit: removal of one full inference per request and repeated model loading. The old video displayed approximately 7.4-7.7 s/request; the historical standalone INT8 inference baseline was about 3.2 s. A new total closer to one inference is a hypothesis, not a measured result. Do not promise 30 classifications/s. Keep the model/precision/thread count/camera mode fixed when comparing.

If preview competes with inference, repeat into a new CSV with `--preview-hz 10`. This reduces Windows decode/render work, not camera capture or SSH bandwidth. A separate experiment with `--threads 1` can test HPS contention but may slow inference. Keep `--fps 30`, since this is the advertised MJPEG rate. Changing to 320x240 is optional and changes image detail; it is not necessary for the initial comparison.

## Timing definitions

| Field | Meaning |
|---|---|
| load_ms | One-time model load on HPS; repeated as metadata in CSV |
| decode_ms | Selected JPEG decoding/RGB conversion on Windows |
| preprocess_ms | Windows bicubic resize, crop, normalization, serialization |
| pack_ms | HPS input validation and conversion into ncnn Mat |
| infer_ms | HPS extractor creation, binding, execution, output extraction |
| service_ms | HPS processing after full input receipt through output validation; excludes response writing |
| roundtrip_ms | Windows send-start to receipt/validation of the complete response |
| transport_overhead_ms | max(0, roundtrip_ms - service_ms); transfer, scheduling and protocol overhead estimate, NOT pure wire time |
| request_ms | Windows selected-frame decode-start through response validation; excludes one-time model load and subsequent GUI refresh |
| received_to_result_ms | Windows complete-JPEG receipt to response validation; not exposure-to-result |

All durations use monotonic clocks on their respective machines. No subtraction between unsynchronized absolute timestamps. Result age grows while the next inference runs and is not preprocessing time. Physical camera-to-screen delay still needs a visible timer or synchronized capture timestamps to measure.

## Stop, failures and rollback

Close the window to close SSH and terminate the service. The service also exits on input EOF. No per-frame temporary directories are created. Invalid headers, truncated payloads, nonfinite inputs, model errors, timeout and mismatched IDs stop the viewer with a stale/error message; no silent fallback to benchmark or CPU/FPGA alternative. Camera disconnect requires restart after reconnection. If an abrupt disconnect leaves a process, inspect `ps` before restarting; do not launch duplicate capture processes.

Rollback: close the new viewer and run the old `hps_camera_viewer.py` command with `/home/ubuntu/vit_camera/build/model_runner`. Both files remain available. Keep your original working microSD and model files intact.

## Validation performed locally

- CMake compiled both native targets against the repository's existing ncnn host installation.
- Three sequential requests in one persistent process produced logits exactly matching the unchanged benchmark runner on a deterministic small FP32 test network.
- Invalid magic, truncated request, NaN input, missing model, fragmented send/receive, wrong response ID, nonfinite logits, dropped connection and timeout cases were checked.
- New preprocessing exactly matched the project's torchvision reference on six image shapes; the old viewer's three existing contract tests also pass.
- No physical DE1-SoC execution, Windows GUI test, or real INT8 checkpoint performance measurement was performed here. Those are the deployment checks above.

Protocol v1 is little-endian: ready `RDY1` + float64 load_ms; request `VIT1` + uint32 frame_id + 602112 input bytes; response `RES1` + uint32 frame_id + three float64 timings (pack, inference, service) + 4000 logit bytes. One request is in flight at a time. stdout is binary; diagnostics go to stderr.
