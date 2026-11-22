# USB camera → DE1-SoC HPS TinyViT → Windows live display

Prepared 16 November 2026 for this repository and your existing board setup.

**Yes: leave the camera connected to the DE1-SoC's USB host port.** Enter Linux commands through PuTTY's serial console or an SSH terminal. To display video on your Windows computer, add an Ethernet connection between the board and your network/PC. The USB-UART connection provides a text console; it does not make the board-connected camera appear in Windows Camera.

This guide includes a working software integration candidate, `scripts/hps_camera_viewer.py`, using your existing ARM `model_runner`. It still needs testing on your physical board. The Amazon page could not be read reliably, so the camera's exact model, USB IDs and supported modes are **not verified**. Do not assume its advertised maximum resolution works with the board's old Linux kernel; use the device checks below.

## 1. What runs where

```text
USB camera ──USB──> DE1-SoC HPS Linux
                         │ FFmpeg captures camera JPEG frames
                         │
                         └──Ethernet / SSH──> Windows Python viewer
                                               │ live image display
                                               │ RGB resize/crop/normalization
                                               │
                          HPS model_runner <───┘ normalized FP32 tensor
                          │ full TinyViT inference on ARM CPUs
                          └──1000 logits──> Windows top-5 + matching image
```

All **model inference** runs on HPS. This initial version performs preprocessing and display on Windows, matching the architecture of your previous board evaluation, which also used desktop-preprocessed inputs. It is not a standalone, entirely board-local image-processing pipeline. Section 12 describes the changes needed for that later.

The viewer shows live video on the left, the exact crop used for the last completed prediction on the right, and top-5 scores underneath. Camera capture continues while inference runs. Only the newest available frame is selected for the next prediction; there is no growing inference queue.

Your recorded 1,000-image evaluation measured approximately **3.20 seconds per INT8 inference** and **4.48 seconds per FP32 inference**, excluding camera processing and transfers. The current runner also executes a separate initial inference even with zero warmups. Reusing it for one frame therefore executes the model **twice per request**, plus reloads the model. Expect prediction updates slower than roughly 6.4 seconds for INT8 or 9 seconds for FP32, potentially substantially slower under concurrent capture. These are baseline-derived expectations, not measured camera performance. Preview can update several times per second independently. See [the measured report](../results/reports/hps_accuracy1000_2026-11-08.md).

## 2. Connections and command windows

| Connection/window | Purpose |
|---|---|
| Camera → board's rectangular USB Type-A **HPS USB HOST** port | Camera input |
| Board **UART-to-USB** → Windows | PuTTY Linux console |
| Board Ethernet → router/switch shared with Windows, or directly to PC Ethernet | Video, tensor and prediction transfer |
| Board power adapter and known-working microSD | Existing Linux boot |
| Windows PowerShell | Install/run the viewer and SSH |
| Ubuntu/WSL, only if needed | Cross-compile missing board software |

DE1-SoC has two HPS USB 2.0 host ports. USB-Blaster is for FPGA programming/debug, not a Linux command console. Its VGA output belongs to the FPGA side and is unnecessary for this Windows display route. [Terasic board specifications](https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&No=836&PartNo=2)

If your USB stick still holds the model, keep it in the second host port. The camera and stick can coexist; identify the stick's actual partition before mounting it.

Open **Windows Device Manager → Ports (COM & LPT)** and identify the board UART COM port. In PuTTY choose **Serial**, that COM port, **115200 baud, 8 data bits, no parity, 1 stop bit, no flow control**. Press Enter after boot, then log in with the account used in your previous board work. Do not assume an unknown default password. If PuTTY already shows a working Linux prompt, keep that connection. Terasic documents 115200 for the board UART in its [setup manual](https://www.terasic.com.tw/attachment/archive/836/DE1SOC_OpenCL_v02.pdf).

Commands marked **BOARD** go into PuTTY/SSH. Commands marked **WINDOWS** go into PowerShell. Commands marked **WSL** go into Ubuntu. Do not paste Windows or WSL installation commands into the board console.

## 3. First check: does Linux see the camera?

**BOARD:**

```sh
uname -a
uname -m
cat /proc/meminfo | head
ls /dev/video*
ls /sys/class/video4linux
command -v lsusb
command -v v4l2-ctl
command -v ffmpeg
command -v dropbear
command -v sshd
dmesg | tail -80
```

Your last recorded system was `armv7l`, Linux `3.12.0-00307-g507abb4-dirty`, with BusyBox tools. A missing command is useful diagnostic information, not a reason to reinstall Linux immediately.

If available:

```sh
lsusb
```

Record the camera's **VID:PID**, product name, and messages containing `uvcvideo`, `USB Video`, or `video0`. If `lsusb` is absent, use:

```sh
for d in /sys/bus/usb/devices/*; do
    if [ -f "$d/idVendor" ]; then
        echo "$d"
        cat "$d/idVendor" "$d/idProduct"
        cat "$d/product" 2>/dev/null
    fi
done
```

If no video node exists, try the following only when UVC is a loadable module:

```sh
modprobe uvcvideo
dmesg | tail -60
ls /dev/video*
```

`modprobe: module not found` does not prove that UVC is absent: it might be built into the kernel. If available, inspect the running kernel configuration:

```sh
zcat /proc/config.gz | grep -E 'CONFIG_MEDIA_SUPPORT|CONFIG_VIDEO_DEV|CONFIG_USB_VIDEO_CLASS'
```

If `/proc/config.gz` does not exist, inspect the configuration from the exact BSP/kernel build. A missing UVC driver needs a matching kernel/module build with its dependencies. Do not copy a `uvcvideo.ko` from a different Linux version. Modern-camera compatibility with Linux 3.12 must be established experimentally. [Linux UVC project](https://ideasonboard.org/uvc/)

Stop here if there is no working capture device. The viewer cannot fix missing USB/driver support. A test on Windows with the camera temporarily plugged into the PC can check the camera/cable, but does not prove board compatibility; reconnect it to the board afterwards.

## 4. Choose a supported capture mode

**BOARD, if `v4l2-ctl` exists:**

```sh
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --all
v4l2-ctl -d /dev/video0 --list-formats-ext
```

A camera may expose several nodes. Choose the node that supports video capture, not a metadata-only node. Substitute that node throughout this guide.

If `v4l2-ctl` is absent but FFmpeg is present:

```sh
ffmpeg -hide_banner -f v4l2 -list_formats all -i /dev/video0
```

This enumeration command may exit with an error after printing formats because no output was requested. Use `v4l2-ctl` for detailed frame intervals where available.

Start with an **advertised** MJPEG mode, ideally 640×480 at 5 or 10 fps. If the camera only advertises 30 fps for that mode, request 30; do not assume it can capture at 5. MJPEG permits forwarding the camera's JPEG frames without decoding and re-encoding them on HPS. It still consumes USB/network/SSH resources. If only YUYV works, the script can encode it to JPEG on the board, at additional CPU cost. FFmpeg calls that raw format `yuyv422`. [FFmpeg V4L2 documentation](https://ffmpeg.org/ffmpeg-devices.html#video4linux2_002c-v4l2)

The following commands assume **MJPEG 640×480 at 5 fps was confirmed**. Replace all three settings consistently with your supported mode.

## 5. Capture one image before attempting live inference

**BOARD:**

```sh
ffmpeg -nostdin -hide_banner -f v4l2 -input_format mjpeg \
  -video_size 640x480 -framerate 5 -i /dev/video0 \
  -frames:v 1 -c:v copy -y /tmp/camera.jpg
ls -l /tmp/camera.jpg
```

This writes an actual JPEG; `-y` replaces only that temporary snapshot on repeated runs. For a YUYV-only camera, use:

```sh
ffmpeg -nostdin -hide_banner -f v4l2 -input_format yuyv422 \
  -video_size 640x480 -framerate 5 -i /dev/video0 \
  -frames:v 1 -c:v mjpeg -threads 1 -q:v 5 -y /tmp/camera.jpg
```

Once SSH works in section 6, copy the snapshot using **WINDOWS**:

```powershell
scp root@192.168.1.50:/tmp/camera.jpg .\camera.jpg
Start-Process .\camera.jpg
```

Replace `192.168.1.50` with the real board address. If modern `scp` reports a missing SFTP subsystem, try `scp -O` for the old SCP protocol. The viewer itself uses SSH execution channels and does **not** need SFTP or an SCP server.

Verify focus, recognizable objects and colors. The first frame may precede exposure/autofocus settling; repeat the snapshot or use the continuous preview before judging image quality. Do not proceed on corrupted images.

If FFmpeg is missing, go to section 11. If only `v4l2-ctl` is present, this diagnostic captures raw bytes:

```sh
v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=640,height=480,pixelformat=YUYV \
  --stream-mmap=4 --stream-count=1 --stream-to=/tmp/frame.yuyv
```

Use it only for an advertised YUYV mode. That file is **not a JPEG** and cannot be passed directly to the model. Raw decoding also needs the negotiated width, height, row stride and color interpretation.

## 6. Establish Ethernet and SSH

The easiest setup is board and Windows on the same router/switch. Windows may use Wi-Fi if that network permits communication with wired clients.

**BOARD:**

```sh
ip addr
```

If `ip` is unavailable:

```sh
ifconfig -a
```

Find the Ethernet interface and its IPv4 address. If it is `eth0`, has a link but no address, and your network uses DHCP:

```sh
udhcpc -i eth0
```

This changes that interface's network configuration; do not run it over an existing connection whose configuration you need to preserve. Use PuTTY serial during network setup.

**WINDOWS**, with the actual board IP:

```powershell
ping 192.168.1.50
Test-NetConnection 192.168.1.50 -Port 22
ssh root@192.168.1.50
```

Verify the host-key fingerprint through your trusted board/BSP setup before accepting the first connection. This creates Windows `%USERPROFILE%\.ssh\known_hosts`, which the viewer reads. A PuTTY saved host key alone does not populate that file. Use the same IP/hostname for the viewer as for this SSH connection.

If port 22 is closed, inspect on the **BOARD**:

```sh
ps | grep -E 'dropbear|sshd'
ls /etc/init.d
```

Start the existing image's SSH service using its actual init script, for example `/etc/init.d/dropbear start` **only if that script exists**. If no SSH server is installed, add a BSP-compatible Dropbear/OpenSSH build before using the viewer. An executable alone may still require host keys and configuration. Do not launch a second server when one is already running.

For a direct Ethernet cable without a router, use an otherwise unused private subnet. Example: set the Windows Ethernet adapter manually to `192.168.50.1`, mask `255.255.255.0`, no gateway/DNS; then in the serial console:

```sh
ifconfig eth0 192.168.50.2 netmask 255.255.255.0 up
```

Use `192.168.50.2` in all Windows commands. Choose another subnet if this conflicts with your Wi-Fi/VPN. Restore Windows DHCP afterwards. This board setting is temporary across reboot. Internet access on the board is not required for the demonstration.

## 7. Locate your existing HPS model

Your previous accuracy workflow copied the runner to `/home/root/hps_accuracy1000/build/model_runner`, while models remained on the USB bundle. Confirm, rather than assuming those files still exist.

**BOARD:**

```sh
ls -l /home/root/hps_accuracy1000/build/model_runner
ls -lh /mnt/usb/hps_accuracy1000/model
mount
df -h /tmp /home/root
```

Expected bundle model names are `int8.param`, `int8.bin`, `fp32.param`, `fp32.bin`. If the USB is not mounted, identify its partition using `cat /proc/partitions` and existing `mount` output. Only if the correct partition is `/dev/sda1` and it is not mounted:

```sh
mkdir -p /mnt/usb
mount -o ro /dev/sda1 /mnt/usb
```

Use your previously validated executable and model pair. `runtime-int8` selects native ncnn INT8 behavior; it is not your Python Q1 simulation. The input remains normalized **FP32**, even for this INT8 model.

Your runner's exact command structure is:

```text
model_runner eval MODEL.param MODEL.bin INPUTS.tsv NEW_OUT THREADS WARMUPS RUNS INPUT_BLOB OUTPUT_BLOB MIN_SECONDS PRECISION
```

Input is little-endian FP32, RGB CHW `[3,224,224]`, exactly **602,112 bytes**. `INPUTS.tsv` has no header and one line per image: `image_id<TAB>label<TAB>tensor_path`. The input/output blobs in the established workflow are `in0`/`out0`; use the actual converted model names if yours differ.

The viewer automatically sends the tensor, writes a one-image TSV, and executes this equivalent **BOARD** command in a unique temporary directory:

```sh
/home/root/hps_accuracy1000/build/model_runner eval \
  /mnt/usb/hps_accuracy1000/model/int8.param \
  /mnt/usb/hps_accuracy1000/model/int8.bin \
  "$job/inputs.tsv" "$job/result" 2 0 1 in0 out0 0 runtime-int8
```

This is explanatory, not a standalone command: `$job` is created by the viewer. `result` must not already exist; the runner creates it. Output `00000.logits.bin` contains 1,000 FP32 class logits. The viewer retrieves those bytes and removes its own temporary directory. Existing accuracy results/models are not modified.

A placeholder label `0` is used because a live scene has no supplied ground truth. **Ignore the runner's accuracy counters for live images.** The displayed predictions come from logits, not that placeholder.

## 8. Install and start the Windows viewer

Copy `scripts/hps_camera_viewer.py` from this guide's bundle to a Windows folder, for example `C:\Users\tiffa\Downloads\vit_camera`. Use your actual username/path. Install a standard Windows Python 3 distribution with Tcl/Tk support if Python is not already installed. Run the viewer in native Windows Python; WSL is not needed for its GUI.

**WINDOWS**, in that folder:

```powershell
py -3 -m venv .venv-camera
.\.venv-camera\Scripts\python.exe -m pip install numpy pillow paramiko
.\.venv-camera\Scripts\python.exe -m tkinter
```

Close the small Tk test window. Explicitly invoking the environment's Python avoids PowerShell activation-policy issues.

Stop any other camera capture command before starting the viewer. Then:

```powershell
.\.venv-camera\Scripts\python.exe .\hps_camera_viewer.py `
  --host 192.168.1.50 --user root --password `
  --runner /home/root/hps_accuracy1000/build/model_runner `
  --param /mnt/usb/hps_accuracy1000/model/int8.param `
  --model /mnt/usb/hps_accuracy1000/model/int8.bin `
  --precision runtime-int8 `
  --device /dev/video0 --format mjpeg --size 640x480 --fps 5
```

PowerShell uses a backtick for line continuation; do not put spaces after it. Enter the board account password at the prompt. Omit `--password` if SSH key authentication is already configured.

Change the address, paths, mode and rate to your verified values. If FFmpeg was installed at `/home/root/camera-tools/ffmpeg`, append:

```text
--ffmpeg /home/root/camera-tools/ffmpeg
```

For YUYV use `--format yuyv422`. For FP32 switch both model filenames to `fp32.*` and use `--precision fp32`. Use `--threads 1` to test whether reserving more CPU time for capture makes the preview smoother; model inference may become slower.

You should see:

```text
Live camera preview                 Last classified 224 × 224 crop
[updates continuously]              [changes when inference completes]

Result for frame 47; received 7.5s ago
Request duration ... s
class 281: ... %
class ...: ... %
```

The receive time is measured on Windows after JPEG arrival, **not the camera exposure timestamp**. Request duration includes Windows preprocessing, transfer, model loading, the initial inference and evaluation inference, and result retrieval. It is not model-only latency. Scores are softmax of the 1,000 logits; they are not calibrated certainty or measured accuracy.

The program starts remote FFmpeg itself and sends the JPEG stream over SSH. No browser server, separate video port, OpenCV, Python on the board, or FPGA reconfiguration is needed. The left and right panels deliberately distinguish current preview from the older classified image. Closing the window closes SSH channels; check `ps` on the board if an interrupted session leaves a process behind. If capture or inference fails, the window marks retained content **STOPPED / STALE**. Fix the cause and restart.

### Class names instead of numeric indices

Without `--labels`, the program correctly displays zero-based ImageNet class indices. To show names, supply a UTF-8 file containing exactly 1,000 lines in the checkpoint's class order and add `--labels .\imagenet_labels.txt`.

If using the standard ImageNet-1K order established in this project's export, your existing **WSL model environment** can produce a candidate name list without downloading weights:

```sh
python -c 'from torchvision.models import ResNet50_Weights; from pathlib import Path; Path("imagenet_labels.txt").write_text("\n".join(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])+"\n", encoding="utf-8")'
```

Copy it to the Windows viewer folder. These are ImageNet metadata names, not a ResNet model invocation. Confirm the order against your TinyViT checkpoint/export label mapping before presenting names; a length check alone cannot establish the mapping.

### Classify one saved camera snapshot

To isolate inference from live capture, append `--image .\camera.jpg` to the same viewer command. It displays and classifies that saved image once, using HPS for inference. It does not open the camera. This is a useful first test after section 5.

## 9. What happens to each input image

The script follows `configs/preprocessing.json` and `baseline/model_support.py`:

1. Decode JPEG and convert to RGB.
2. Resize the shorter side to 256 using Pillow bicubic interpolation, preserving aspect ratio.
3. Take the centered 224×224 crop, matching the reference's rounding.
4. Divide pixel values by 255.
5. Normalize RGB channels with mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`.
6. Serialize CHW little-endian float32 and send to HPS.

For a 640×480 frame, much of the left/right scene lies outside the center crop. Put the demonstration object near the center and check the classified-crop panel. Do not stretch the whole frame to 224×224 or normalize a second time. Image classification gives one set of whole-crop labels; it does not locate objects or create bounding boxes. [Torchvision resize behavior](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html)

## 10. Troubleshooting by symptom

| Symptom | Check/action |
|---|---|
| PuTTY has no Linux prompt | Confirm UART-to-USB rather than USB-Blaster, correct COM port, serial settings, board power and boot image. |
| USB device absent | Check camera cable/connector and `dmesg`; test the other host port. If logs suggest power trouble, test a suitable powered hub. |
| Camera enumerates, no `/dev/video*` | Inspect UVC/kernel configuration and module loading; compatibility with the old kernel is unresolved. |
| `Permission denied` opening camera | Use your board account's proper video-device permissions; the example uses the existing root lab account. |
| `Device or resource busy` | Stop the earlier FFmpeg/v4l2 test or another viewer. Only one capture process should own the node. |
| `Invalid argument`, unsupported format/rate | Enumerate modes and use an exact supported format/resolution/frame interval. |
| `ffmpeg: not found` | Install a board-compatible build, then set `--ffmpeg` to its absolute path. |
| Executable exists but says `not found` | Often a missing dynamic loader/library. Check ARM ABI and linkage, not just filename. |
| `GLIBC_* not found` / `Illegal instruction` | Wrong build for the board's libc/CPU; rebuild with the working BSP toolchain. |
| SSH timeout/refused | Check IP, Ethernet link, subnet, port 22 and server. Serial connectivity does not imply network connectivity. |
| Host key unknown/mismatch | Complete and verify the Windows SSH connection using the same hostname. Investigate changed keys; do not bypass validation. |
| Old SSH algorithm negotiation failure | Capture the exact error and inspect the board SSH server version. Prefer a compatible maintained server; do not globally enable obsolete algorithms. |
| Runner input/output binding failure | Verify model pair and blob names; use `--input-blob` / `--output-blob` if needed. |
| Model failed to load | Verify USB mount, absolute paths, binary permissions and bundle integrity. |
| Viewer appears paused before first result | The existing runner loads and executes twice. Allow time; default per-channel timeout is 120 s. |
| Frozen video after unplug | Viewer stops and marks stale content. Reconnect, recheck the video node, restart. |
| Odd labels or colors | Check RGB conversion, label order, classified crop, focus, lighting and supported scene classes. |
| Preview gets behind | Reduce supported resolution/rate, prefer camera MJPEG, try one inference thread, close other board workloads. SSH/capture can still buffer even though the inference handoff holds only the latest frame. |
| `mktemp` unavailable / temp storage full | Check BusyBox applets and `df -h /tmp`; viewer requires `mktemp`, `cat`, `wc`, `printf`, `rm`, and an existing writable `/tmp`. |

## 11. If the board has no FFmpeg

Do not assume `apt install` works on your BusyBox image. Prefer an existing BSP-compatible FFmpeg package/build. Needed features are V4L2 input, MJPEG parsing/decoding/encoding as applicable, image output, and pipe/file protocols. FFmpeg's V4L2 input does not inherently require a full OpenCV installation.

If you need a cross-build, do it in **WSL**, using the same ARMv7 hard-float toolchain/sysroot that produced your working runner. The following is an example build recipe, **not a board-tested binary or a guarantee that a modern Ubuntu compiler's libc will work with Linux 3.12**. FFmpeg 4.4.6 is a reproducible source example available from the official archive, not a claim about the latest release.

```sh
mkdir -p ~/camera-build
cd ~/camera-build
curl -LO https://ffmpeg.org/releases/ffmpeg-4.4.6.tar.xz
tar -xf ffmpeg-4.4.6.tar.xz
cd ffmpeg-4.4.6
./configure \
  --enable-cross-compile --target-os=linux --arch=arm --cpu=cortex-a9 \
  --cross-prefix=arm-linux-gnueabihf- \
  --disable-shared --enable-static --disable-autodetect \
  --disable-doc --disable-debug --disable-ffplay --disable-ffprobe \
  --extra-cflags='-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard' \
  --extra-ldflags='-static'
make -j2 ffmpeg
file ffmpeg
arm-linux-gnueabihf-readelf -l ffmpeg
```

Replace the compiler prefix and add `--sysroot=/actual/BSP/sysroot` as required by your working toolchain. Static linker inputs must exist in that toolchain. Static linking avoids board shared-library mismatches but does **not** guarantee old-kernel compatibility or support for every camera. Inspect configure failures rather than removing necessary capture features. Preserve the source version/build configuration. [Official FFmpeg source archive](https://ffmpeg.org/releases/)

Transfer the executable using your existing USB-stick process or SCP. On the **BOARD**, after placing it at the shown path:

```sh
chmod +x /home/root/camera-tools/ffmpeg
/home/root/camera-tools/ffmpeg -version
/home/root/camera-tools/ffmpeg -devices
/home/root/camera-tools/ffmpeg -encoders
```

Confirm V4L2 input and MJPEG output support, then repeat section 5 with this absolute executable path. A successful cross-build on WSL alone is not an acceptance test. If a matching driver or working SSH server is also missing, those are separate BSP tasks; copying FFmpeg will not supply them.

## 12. Next step: make preprocessing and continuous inference fully HPS-local

The supplied viewer achieves HPS-only **model execution** with Windows preprocessing. For an autonomous board application, implement a native C++ service:

```text
HPS V4L2 capture → MJPEG decode / YUYV-to-RGB → matching preprocessing
                → persistent ncnn Net → top-5 result + matching frame ID
                → Ethernet preview/result service → Windows display
```

Load the validated model once, use a fresh extractor for each selected frame, and remove the benchmark runner's extra initial inference from the repeated request path. Retain a latest-frame handoff and matching frame/result IDs. Match C++ resize/crop/normalization against the saved Python reference numerically before comparing accuracy. Bilinear resize is not a drop-in equivalent for the existing bicubic reference. Keep JPEG preview work separate from inference, and measure contention on the two Cortex-A9 cores.

This service is **not implemented by the current guide**; do not invent a `camera_runner` command or claim it exists. The existing `model_runner` cannot directly accept `/dev/video0` or a JPEG.

A monitor connected to the board's VGA port is another implementation path. It needs working FPGA video hardware, a known framebuffer/register interface and a combined compatible bitstream. A VGA cable alone does not provide Linux desktop output, and most laptop display connectors are outputs rather than video inputs. For your requested Windows screen, Ethernet plus the supplied viewer avoids that dependency. See the project's [peripheral integration plan](peripheral_setup_plan.md).

## 13. Acceptance checklist and what to collect if blocked

- Camera is identified by USB VID:PID; a capture node and an advertised mode are recorded.
- A JPEG captured on HPS opens correctly on Windows.
- Windows SSH reaches the board, and the existing model files are readable.
- `--image camera.jpg` returns finite HPS logits and displays the matching crop.
- Live preview continues while predictions update; the classified image and result remain paired.
- Run for ten minutes and watch memory, preview age and errors. Do not claim a classification rate until measured.
- Disconnect the camera once: confirm stale/error indication, then reconnect and restart successfully.
- Record model names/hashes, camera mode, board kernel, thread count and whether Windows preprocessing was used.

If blocked, collect the output of these **BOARD** commands plus the exact Windows error:

```sh
uname -a
ls /dev/video*
dmesg | tail -80
ifconfig -a
command -v ffmpeg
command -v v4l2-ctl
command -v dropbear
command -v sshd
ls -lh /mnt/usb/hps_accuracy1000/model
```

The guide and script were prepared without access to your physical board or Windows session. Local software checks are recorded separately; USB detection, SSH compatibility, capture mode negotiation and end-to-end hardware operation remain to be verified on your setup.

### Local validation completed

Three automated checks passed on 16 November 2026: the viewer's tensor matched the repository's torchvision preprocessing exactly on eight synthetic image shapes; a mock runner verified the generated shell command, quoted paths, input byte count and binary output separation; and a short tensor was rejected before launching inference. These checks do not exercise an actual SSH server, Windows GUI, webcam or ARM model.

Reproduce from the repository root using the existing model environment:

```sh
.venv-model/bin/python -m unittest discover -s tests -p test_hps_camera_viewer.py -v
```
