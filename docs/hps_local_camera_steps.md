# HPS-only capture, preprocessing and TinyViT demonstration

This package is for the verified Arducam UVC camera, Linux 4.5 board image, working native ncnn INT8 model and Windows/WSL setup in this chat. All capture, JPEG decoding, RGB resize/crop/normalization, inference, softmax and label lookup run on HPS. Windows only compiles ARM software in WSL, transfers files and displays the board's web page. A browser is not needed for classification to continue.

This implementation uses Ubuntu's Python 3 + Pillow + NumPy **on HPS** for preprocessing and the persistent C++ ncnn runner for inference. It avoids inventing a different C++ bicubic algorithm. Its preprocessing equivalence and speed must still be verified on your older board libraries. No GUI, OpenCV, PyTorch, Tk or Paramiko is installed on HPS.

## 1. Close the old Windows viewer

Only one process should capture /dev/video0. Keep PuTTY connected. Do not change the microSD, MSEL, model or FPGA design.

## 2. Extract the package on Windows

Extract `hps_local_demo.zip` so that this file exists:

```text
C:\Users\tiffa\Downloads\hps_local_demo\scripts\build_hps_local.sh
```

If extraction creates an extra nested folder, adjust the paths below accordingly. The archive contains source, labels and fixed preprocessing fixtures, but no models or compiled ARM executable.

## 3. Compile only the inference service in WSL

Your existing ARM libraries were found here:

```text
/home/tiff/cpu_baseline_tools/build/ncnn-arm-install/lib/cmake/ncnn
```

In **Ubuntu/WSL**, run:

```sh
cd /mnt/c/Users/tiffa/Downloads/hps_local_demo
sh scripts/build_hps_local.sh /home/tiff/cpu_baseline_tools
```

The helper uses your existing arm-linux-gnueabihf-g++ toolchain and ARM ncnn installation. It builds a separate target and places `bin/camera_runner` in this package. It does not change the old benchmark executable or model files.

Expected `file` output identifies a 32-bit ARM executable. This recipe links statically to avoid transferring mismatched desktop shared libraries. If it reports missing static libraries, GLIBC requirements, unsupported compiler flags or other errors, stop and retain the log. Do not copy a host/x86 executable or remove linker flags without checking the ABI. The assistant's host build is not an ARM build.

## 4. Install preprocessing libraries on the board

In **PuTTY**:

```sh
sudo apt-get -o Acquire::ForceIPv4=true update
```

If updating succeeds, run:

```sh
sudo apt-get -o Acquire::ForceIPv4=true install python3-numpy python3-pil
```

Do not run an OS upgrade. Your camera FFmpeg installation is already working. Confirm:

```sh
python3 -c 'import sys, PIL, numpy; print(sys.version); print(PIL.__version__); print(numpy.__version__)'
ffmpeg -version
```

A repository or dependency error should be diagnosed before proceeding. The board has about 1.8 GB free and 640 MB available memory in your latest report; verify remaining space after installation. The preprocessing script uses Python 3.5-compatible syntax and supports the older Pillow bicubic constant, but actual old-library runtime behavior is tested next.

## 5. Transfer the package to HPS

In **Windows PowerShell**:

```powershell
cd "$env:USERPROFILE\Downloads"
scp -r .\hps_local_demo ubuntu@192.168.137.39:/home/ubuntu/
```

Replace the IP if it has changed. Use `scp -O -r` if SFTP is unavailable. On later transfers, ensure the folder is not accidentally nested inside itself.

In **PuTTY**:

```sh
cd /home/ubuntu/hps_local_demo
chmod +x bin/camera_runner
./bin/camera_runner
```

Expected: `Usage: camera_runner ...`. Do not start it manually with model arguments: its protocol is binary and the board application launches it automatically.

## 6. Validate HPS preprocessing before real inference

In **PuTTY**:

```sh
cd /home/ubuntu/hps_local_demo
python3 scripts/hps_local_demo.py --verify-fixtures fixtures
```

All three cases must report PASS. These fixed PNG/JPEG test files and expected tensors were prepared from the existing reference; no Windows preprocessing happens during this test or the demo. The check uses maximum absolute normalized-tensor error <= 1e-5 to detect changes in interpolation, crop rounding, normalization or JPEG decoding.

If a case fails, send its numerical error and Pillow/NumPy versions. Do not assume the prediction is equivalent or simply relax the threshold. The fixtures are a regression check, not proof of equivalence for every possible camera image.

## 7. Classify your saved camera image entirely on HPS

Confirm existing files:

```sh
ls -lh /home/ubuntu/camera.jpg /home/ubuntu/vit_camera/model/int8.param /home/ubuntu/vit_camera/model/int8.bin
```

Then, from `/home/ubuntu/hps_local_demo`, paste this as **one line in PuTTY**:

```sh
python3 scripts/hps_local_demo.py --runner ./bin/camera_runner --param /home/ubuntu/vit_camera/model/int8.param --model /home/ubuntu/vit_camera/model/int8.bin --labels imagenet_labels.txt --image /home/ubuntu/camera.jpg --snapshot-out snapshot_run1 --timings snapshot_run1.csv
```

The program loads the model once, preprocesses the JPEG on HPS and prints the prediction in PuTTY. It also saves `snapshot_run1/input.bin`, `logits.bin`, `crop.jpg` and `result.json`. The output directory and CSV must be new. For a retry use `snapshot_run2` and `snapshot_run2.csv`.

In your **Windows browser**, open:

```text
http://192.168.137.39:8080
```

You should see the saved image, classified crop and labels. Compare the same snapshot against the earlier demo; report discrepancies rather than assuming that a matching top label proves all logits are equivalent. The browser does not upload images/tensors or perform model preprocessing. Snapshot mode completes one classification and retains the result until Ctrl+C.

If the page does not open, in Windows PowerShell run `Test-NetConnection 192.168.137.39 -Port 8080`. Retain the result and PuTTY messages. Do not disable firewalls globally. The display server is intended for your direct Ethernet/private lab connection; it provides read-only endpoints and no authentication.

Press **Ctrl+C in PuTTY** to stop before starting live mode. It may wait for the current inference to finish or time out before closing its subprocesses.

## 8. Run live HPS-only mode

From `/home/ubuntu/hps_local_demo`, paste this as **one line in PuTTY**:

```sh
python3 scripts/hps_local_demo.py --runner ./bin/camera_runner --param /home/ubuntu/vit_camera/model/int8.param --model /home/ubuntu/vit_camera/model/int8.bin --labels imagenet_labels.txt --device /dev/video0 --size 640x480 --fps 30 --threads 2 --timings live_run1.csv
```

Refresh the same Windows browser page. No Windows Python viewer is involved. Defaults are runtime-int8 and blob names in0/out0, matching your validated model.

The HPS capture worker forwards compressed MJPEG into a latest-frame slot. The inference worker decodes only the frame it selects when idle. There is one inference per request, and the model stays loaded in HPS memory. Browser preview polls at most approximately 5 times/s; its image rendering is for display only. Result crop and labels travel together in one response and refer to the same frame.

Close the browser for a minute while leaving PuTTY open: classifications should continue printing and CSV rows should continue accumulating. Reopen it to see the latest result. This checks independence from Windows display. It does not mean the process survives closing the SSH/PuTTY session that launched it; service installation is a later deployment task.

For a bounded measurement with 100 warm requests plus the first cold one, add `--max-frames 101` and choose a new CSV. Capture/classification stop after that count and the page retains the last result until Ctrl+C. Do not mistake a retained snapshot for live capture.

## 9. Measure instead of assuming speedup

In another board shell, or after stopping the application:

```sh
python3 scripts/summarize_hps_local.py live_run1.csv
```

This prints mean, median and p95, excluding the first cold request by default, and whether at least 100 samples were selected. It also reports the completion-event rate between the first and last selected events; this is not a substitute for a controlled one-minute throughput test.

- decode_ms: HPS JPEG decoding/RGB conversion.
- preprocess_ms: HPS resize/crop/normalization/tensor serialization.
- infer_ms: HPS ncnn extractor creation, binding, execution and output extraction.
- pack_ms: tensor validation/packing in the C++ runner.
- ipc_overhead_ms: local pipe round-trip minus runner service duration; no Ethernet transfer is involved in inference.
- request_ms: selected-frame processing start through validated logits. Excludes one-time model load and subsequent display preparation.
- received_to_result_ms: complete JPEG receipt by the HPS app through validated logits. Does not start at camera exposure.
- publish_ms: postprocessing/label lookup/crop encoding preparation; does not include browser rendering or actual network delivery.

Your old video showed about 7.4-7.7 seconds/request while running two inferences and reloading the model. The new path removes both repeated work and the Windows tensor round trip, but adds preprocessing on HPS. It is expected to remove significant overhead, not guaranteed to hit 1 FPS or any particular latency. Record the result before claiming a speedup.

Keep 640x480 MJPEG at 30 fps for the first comparison. Then test a separate one-thread run (`--threads 1`) to assess contention. It can slow inference, so compare total time and stability. A smaller model is not part of this update; preserve proposal C4's fixed workload.

## 10. Integration with your teammates later

Your responsibility is now camera capture, HPS preprocessing, full-model CPU inference and display publication. No accelerator code is required to validate this path.

The application calls a backend with normalized FP32 CHW [3,224,224] plus frame ID and receives 1000 logits. Your teammates can supply a hybrid full-model backend behind this boundary. Internally they still need to integrate selected MSA/MLP operations into ncnn/model execution, agree tensor layouts/scales/tiling, allocate buffers and implement DMA/bridge completion/error handling. This is not achieved by calling an unrelated FPGA matrix program after preprocessing.

Timing fields here are CPU preprocessing/inference/local IPC; they are not FPGA kernel/DMA measurements. For final evaluation keep CPU and hybrid preprocessing, model, images and timing boundaries identical. Retain the historical benchmark and optimized CPU baseline separately. A final combined FPGA design must preserve whatever HPS boot/bridge configuration the team requires; the current LXDE image's FPGA configuration is not an integrated accelerator bitstream.

## Local validation and limitations

The assistant tested reference preprocessing equivalence, JPEG stream fragmentation, Python 3.5 syntax, native host persistent inference, snapshot output and the display-only endpoints with a small test network. Classification completed before any browser connection. Earlier persistent-runner tests compare logits exactly against the benchmark runner and cover invalid inputs/protocol/timeouts.

The ARM executable must be built in your WSL environment. The physical board's old Pillow version, real INT8 model, live USB capture and performance still require the steps above. A successful host test does not certify board speed or model accuracy. Do not overwrite your previous working viewer/model/SD card.
