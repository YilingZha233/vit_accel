# TinyViT peripheral setup plan

Prepared for the project team by Yiling Zha | 26 October 2026

I propose to own USB camera capture and the user-facing classification demonstration on DE1-SoC. My first deliverable will capture an image on HPS Linux, pass it to the existing CPU inference path, and print the result. VGA is a second milestone, subject to integration and resource availability in the shared FPGA design.

The accelerator teammates will continue to own operation offloading, accelerator transfers and numerical correctness. We will agree on a software interface so camera and display development can proceed before hybrid inference is ready. This is a proposed work plan; camera and VGA operation have not yet been verified on our board.

### Proposed ownership

| Work item | Proposed owner |
| --- | --- |
| USB detection, V4L2 capture, RGB conversion and frame management | Me |
| Preprocessing agreement, inference input and result interface | Me and inference owner |
| Model execution, CPU or hybrid backend, quantization and FPGA offloading | Accelerator teammates |
| VGA application, text rendering and display tests | Me |
| VGA controller, memory, clocks, pin assignments and shared bitstream merge | Me with FPGA integration owner |
| Accelerator bridge configuration, address map and DMA if used | FPGA integration owner |

### What the onboard bridge means

The HPS–FPGA bridge hardware is built into the Cyclone V SoC. We do not buy or build a separate physical bridge. The project still has to expose and connect the required interfaces, assign addresses, supply clocks and resets, and enable access through the boot or driver flow. That work belongs with the shared FPGA integration. [S2]

USB camera capture through the HPS USB host does not require my own custom FPGA bridge design. VGA is connected to the FPGA side, so HPS-driven display updates need a configured access path to the display registers or framebuffer. The existing accelerator connection does not automatically create that display path. [S1, S3]

```text
USB camera -> HPS capture -> RGB frame -> inference adapter
                                      -> CPU or hybrid backend
Result + matching frame -> UART first -> optional VGA
Hybrid backend -> teammate-owned accelerator interface
```

## Equipment and camera bring up

Use the current working Linux image and ARM toolchain first. The project report records Linux 3.12, BusyBox 1.20.2 and statically built ARM software; a current desktop package installation recipe is not sufficient for this environment. [P1]

| Item | Selection and connection |
| --- | --- |
| Camera | Borrow a USB 2.0 UVC webcam with a supported YUYV mode; test the exact unit before buying. Connect to an HPS USB Type A host port. |
| Initial capture mode | Target 640 × 480 at 5–15 frames/s if advertised. Reduce resolution or rate if unsupported. MJPEG is a fallback requiring a decoder. |
| Console and files | Keep UART connected. Reuse microSD and the existing USB or Ethernet file-transfer method. |
| Optional display | Use a native VGA monitor and VGA cable. If only HDMI is available, verify an active VGA-input to HDMI-output converter; adapter direction matters. |

### Inventory before changing software

- [ ] Record board revision, kernel and boot image, Quartus version, camera make/model and USB vendor/product IDs.

- [ ] Back up the working boot files and record which FPGA image is loaded. Check available storage and memory.

```text
uname -a
cat /proc/meminfo
df -h
lsusb                    # if installed
dmesg | tail -60
ls /dev/video*
ls /sys/class/video4linux
```

### Check the camera driver and supported modes

- [ ] Confirm V4L2 support and CONFIG_USB_VIDEO_CLASS in the matching kernel configuration. Load uvcvideo only if built as a module; a built-in driver will not appear in lsmod. Use matching kernel modules, not modules copied from a different Linux version. [S4]

```text
modprobe uvcvideo         # only if the driver is modular
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --all
v4l2-ctl -d /dev/video0 --list-formats-ext
```

The video node and tools above are examples. Select the node that advertises video capture; a camera can expose multiple nodes. If v4l2-ctl is absent, cross-compile a compatible version or use the planned small V4L2 probe. Read back the negotiated format and frame interval instead of assuming the request was accepted.

Acceptance: the correct capture device is identified, a supported mode is recorded, and no persistent USB errors occur. If the driver is missing, prepare a matching module or kernel build with the BSP owner before continuing.

## Capture and preprocessing implementation

### First implement a single image capture

- [ ] Create a small C/C++ V4L2 capture utility for the board ABI. Start without a dependency on a full desktop OpenCV installation.

- [ ] Query capture and streaming capabilities; negotiate format; allocate and map buffers; queue buffers and start streaming. Use poll/select with a finite timeout, dequeue a completed buffer, copy or consume it safely, then requeue it. Stop streaming and release resources on exit. [S5]

- [ ] Honor returned bytesperline, sizeimage and bytesused. Convert YUYV to RGB using the reported color space and range; decode MJPEG if that mode is used. Check orientation and channel order with a recognizable color target.

- [ ] Save both capture metadata and a decoded RGB image, such as PPM. Requeue a camera buffer only after its content has been copied or is no longer in use.

If the installed tool supports these options, this is an optional raw-capture smoke test after confirming the mode. The output is raw YUYV, not a JPEG:

```text
v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=640,height=480,pixelformat=YUYV \
  --stream-mmap=4 --stream-count=1 --stream-to=frame.yuyv
```

### Match the existing model input

Use the saved CPU preprocessing as the authority: RGB, shorter side resized to 256 with bicubic interpolation, centered 224 × 224 crop, values scaled to [0,1], then channel normalization. Mean = [0.485, 0.456, 0.406]; standard deviation = [0.229, 0.224, 0.225]. Confirm interpolation and crop rounding against the reference. [P2]

- [ ] Save the normalized tensor from a captured image and compare it with the offline reference using a team-agreed numerical tolerance. Verify layout and dtype; normalization occurs exactly once.

- [ ] Keep INT8 scales, zero points and accelerator tile packing inside the inference backend. Do not reinterpret unsigned camera bytes as signed INT8 model activations.

### Then enable continuous capture

Use a bounded latest-frame handoff: a capture worker keeps the newest frame; an inference worker takes a stable copy when idle. Drop superseded frames instead of building a queue. Keep frame IDs and timestamps. Do not overwrite a frame while inference or display uses it.

Start with snapshot classification. A later live preview may update faster than classifications. The reported two-thread CPU INT8 pilot was about 3.2 s per image on preprocessed inputs; it does not establish camera-to-display performance. Capture, conversion and preview also compete for the two CPU cores. [P1]

Acceptance: recognizable saved images, matching offline preprocessing, successful CPU classification, bounded memory, and controlled timeout/disconnect behavior.

## VGA display implementation

VGA is optional until the camera-to-UART demonstration passes. A VGA cable alone is insufficient: the FPGA design must include a working timing and pixel-output path. First inspect the shared design for a usable display controller and a Linux framebuffer driver; do not assume /dev/fb0 exists. [S1, S3]

### Choose the simplest compatible display path

| Available system | Action |
| --- | --- |
| Working Linux framebuffer | Query the driver geometry, line length, pixel format and mapping. Render through its supported interface. |
| Existing memory-mapped video IP | Agree its register map, memory ownership, stride and access API with the FPGA owner. Use a suitable driver or controlled development mapping. |
| No video subsystem | Integrate a small reference video subsystem with the FPGA owner. Begin with a test pattern and text; add image buffering only after resource review. |

- [ ] Reuse board-revision-correct VGA pins and timing from a compatible reference. Include the required pixel clock, reset and clock-domain handling. Compile and check timing in the combined design.

- [ ] Display color bars, then a fixed result string written from HPS. Confirm colors, alignment and stable monitor synchronization.

- [ ] Add a framebuffer or character overlay as needed. The University Program reference demonstrates a 320 × 240 RGB565 image expanded to 640 × 480, plus a character buffer. Those features and addresses belong to that reference bitstream, not every DE1-SoC design. [S3]

### Agree on memory before integrating images

A packed 320 × 240 RGB565 frame is 153,600 bytes (150 KiB); double buffering needs 300 KiB before padding. The reference layout uses a 1,024-byte row stride, so a full 240-row allocation needs 245,760 bytes per buffer. These calculated sizes must be checked against accelerator RAM use and controller alignment. External memory or a smaller display may be needed.

- [ ] Document framebuffer base or device handle, accessible length, row stride, RGB format, cache policy and buffer-swap semantics. Never copy reference physical addresses into the custom project without verifying the generated map.

- [ ] Use front/back ownership and vertical-blank swapping if supported. Otherwise begin with an occasional static update; document any tearing. If the scanout path uses DMA, the integration owner must provide correct buffer allocation and cache handling. [S6]

### Show results without implying object detection

Display the classified frame or crop, class label, optional top-5 scores, backend, inference duration and frame ID. TinyViT here is an image classifier; do not draw invented detection boxes. If showing a separate live preview, label the result as belonging to the last classified frame and show its age.

Acceptance: stable test pattern, text updated by HPS, correct image colors and stride, matching frame/result pairs, and no regression in the accelerator design.

## Software handoff to the accelerator team

I propose one synchronous inference adapter used by both CPU and hybrid modes. The peripheral application supplies a stable RGB frame. The adapter owns preprocessing and model execution; an explicitly defined alternative may accept a normalized tensor, but we must choose one boundary and avoid duplicate normalization.

| Interface field | Agreement required |
| --- | --- |
| Frame | RGB888; width; height; stride_bytes; data length; frame_id; capture timestamp and clock domain. |
| Lifetime | Caller keeps the frame valid until the synchronous call returns. The backend cannot retain a borrowed pointer after return. |
| Preprocessing | RGB conversion occurs before the adapter. Adapter applies the agreed resize, crop, normalization and tensor layout once. |
| Backend | CPU and hybrid implement the same external API. Backend handles quantization, accelerator packing, transfers and completion. |
| Result | Matching frame_id; status; top-k class indices and scores; label-map revision; timing fields; backend identifier. |
| Scores | State whether outputs are logits or softmax probabilities. Do not label arbitrary logits as confidence percentages. |
| Failures | Explicit capture timeout, invalid frame, inference error and accelerator timeout. No silent CPU fallback presented as FPGA execution. |

### Conceptual API to agree before coding

```text
// Proposed interface only; not an existing project API.
int infer_frame(const FrameRGB *frame,
                Backend mode, Result *result);
int display_result(const FrameRGB *classified_frame,
                   const Result *result);
```

Bring up the peripheral application with the existing CPU backend first. A mock result may be used to test rendering, but the screen and logs must identify it as MOCK. Switch to hybrid execution only after the team supplies a validated adapter.

### Timing fields for a useful demonstration

Log frame ID, monotonic capture/dequeue time, preprocessing duration, model duration, render/submit duration, result age, backend and status. Keep model-only timing comparable with the existing baseline. A framebuffer submission timestamp is not the instant pixels become visible; call it capture-to-submit unless display presentation is measured.

Report capture frames/s, preview updates/s and completed classifications/s separately. A 60 Hz VGA signal does not mean 60 classifications/s. Retain a classified-frame thumbnail or frozen snapshot when a live preview is running.

### Shared FPGA integration rule

Keep application code here and synthesis integration in the teammate hardware repository. Loading a reference VGA bitstream replaces the accelerator configuration. The final demo requires one combined, versioned bitstream and documented address map.

## Milestones and acceptance checklist

Complete these milestones in dependency order. Camera development can proceed alongside accelerator work. VGA software can use mock results, but final VGA operation depends on the shared display hardware. These are work packages, not promised calendar dates.

| ID | Task and dependency | Evidence to retain |
| --- | --- | --- |
| P0 | Agree owners, API and CPU fallback; inventory board, BSP and camera. | Ownership note, versions and equipment list. |
| P1 | Detect camera and enumerate supported capture modes. Requires P0. | USB IDs, driver log and negotiated format. |
| P2 | Capture and decode a single image. Requires P1. | Raw sample, RGB sample and capture metadata. |
| P3 | Match preprocessing; classify captured image with CPU. Requires P2 and CPU adapter. | Tensor comparison, result and timing log. |
| P4 | Run bounded continuous capture and handle failures. Requires P3. | Ten-minute log, memory snapshots and restart results. |
| V1 | Review video IP/resources; integrate test pattern and HPS text. Can start after P0. | Combined build, timing/resource reports and display photo. |
| V2 | Display the classified image and result. Requires V1 and P3. | Screen photo matched to logged frame ID. |
| I1 | Enable hybrid adapter. Requires P3 and validated accelerator path. | Same-frame CPU/hybrid comparison and backend log. |
| D1 | Repeat full demo from documented startup. Requires P4 and I1; VGA only if V2 passes. | Runbook, versions, evidence bundle and limitations. |

### Functional acceptance tests

- [ ] Capture at least 10 recognizable frames; verify dimensions, channels, stride handling and saved-file readability.

- [ ] Run the same saved captured image through the peripheral and offline CPU paths. Compare preprocessing tensors and logits with agreed tolerances; matching top-1 alone is insufficient.

- [ ] Run continuous mode for 10 minutes. Require no crash, unbounded queue or progressive memory growth after warmup; record frames dropped and successful classifications.

- [ ] Disconnect and reconnect the camera. Require a clear error and controlled restart; do not leave an old prediction labelled as current.

- [ ] Verify VGA test pattern, text and image independently. Confirm a displayed result belongs to the displayed frame.

- [ ] Repeat tests after the VGA subsystem is merged with the accelerator. Check memory/address conflicts and renewed timing closure.

- [ ] Repeat a cold-start demo three times. Record any manual setup step still required.

## Demonstration and team discussion

### Proposed demonstration sequence

1. Boot the known image, connect UART and the camera, and identify the build and backend. Show that the capture device and mode match the recorded configuration.

2. Capture a single object or scene, save the frame, and run the CPU adapter. Print the frame ID, predicted class and duration. Explain that classification returns a whole-image label.

3. Start continuous capture with a latest-frame handoff. Show that preview and classification have separate rates. Keep the last classified image visible or clearly identify the result age.

4. If VGA is ready, show the same image and result on the monitor. If not, use UART and saved frames as the completed first milestone.

5. When the hybrid adapter is ready, classify the same saved frame in CPU and hybrid modes. Show the actual backend and measured timing; do not promise speedup before it is measured.

6. Unplug the camera once, show the error state, then demonstrate controlled recovery. Preserve the run log and the classified frames.

### Illustrative screen layout

```text
TINYVIT CLASSIFICATION     Backend: CPU / HYBRID

[ Image used for the last completed classification ]

Class: <label>             Frame: <frame_id>
Score: <value and type>    Model time: <measured ms>
Result age: <seconds>     State: READY / BUSY / ERROR

Capture rate: <fps>        Classifications: <per second>
```

This is a proposed layout, not a screenshot or measured result. A preview panel may be added separately if CPU and display bandwidth permit.

### What I plan to tell the team

“I will own USB camera capture, frame conversion and the classification demo application. I will first make snapshot capture work with our CPU model and UART output. Then I will add continuous capture with a bounded latest-frame buffer. VGA is my next milestone, but I need us to integrate its controller and memory into the shared FPGA project. I will use the same inference interface for CPU and hybrid modes, so your offloading work can be connected without rewriting camera capture.”

### Decisions needed at the next meeting

- [ ] Name the FPGA integration owner who will merge video IP and maintain the combined address map and bitstream.

- [ ] Agree who implements the inference adapter and approve the RGB-frame boundary, result structure and label mapping.

- [ ] Confirm camera availability, kernel driver support, monitor availability and whether VGA is required or a stretch objective.

- [ ] Agree the display memory budget and choose text-only, frozen-image, or live-preview scope after resource review.

## Dependencies risks and reference material

| Risk or dependency | Response |
| --- | --- |
| Camera unsupported by Linux 3.12 | Test an available UVC unit early. Inspect descriptors and driver logs; use a matching BSP driver build if necessary. |
| Camera mode or CPU load too high | Lower negotiated resolution/rate, measure conversion cost, and keep capture separate from inference. |
| Preprocessing mismatch | Use one canonical transform and compare saved tensors before judging the model or accelerator. |
| VGA consumes accelerator resources | Check combined RAM, logic, clock and bandwidth budgets. Start with text or a smaller static image. |
| Reference video map differs from custom design | Use generated addresses and documented stride. Integrate one combined design and rerun tests. |
| Old result displayed over a new scene | Bind every result to frame_id; show its source image and age. |
| Hybrid integration is delayed | Finish and demonstrate camera plus CPU inference independently; label the backend accurately. |

### Deliverables I will hand over

Camera probe/capture source and ARM build instructions; decoded image samples and metadata; preprocessing comparison; CPU/hybrid adapter agreement; display application and any agreed video IP integration changes; acceptance logs; wiring and startup instructions; and the combined build identity for the final demonstration. These are planned deliverables, not files already implemented by this report.

### Project evidence

[P1] results/reports/current_measurements_2026-10-15.md — reported Linux and ncnn setup, preprocessing and CPU pilot measurements. The report notes that some full-model raw evidence remains on the Windows/WSL machine.

[P2] configs/preprocessing.json — current preprocessing parameters. [P3] docs/teammate_review.md and configs/accelerator_contract.json — source-derived accelerator interface assumptions and unresolved integration mapping.

### Technical references

[S1] Terasic DE1-SoC specifications and board resources. Use the manual for the actual board revision. https://www.terasic.com.cn/cgi-bin/page/archive.pl?CategoryNo=167&Language=English&No=836&PartNo=2

[S2] Altera Cyclone V HPS reference, FPGA interface enables. https://www.intel.com/content/www/us/en/docs/programmable/683126/21-2/fpga-interface-enables.html

[S3] Intel FPGA University Program DE1-SoC Computer with ARM, video-out sections 4.2.1–4.2.4. https://fpgacademy.org/Downloads/DE1-SoC_Computer_ARM.pdf

[S4] Linux UVC driver documentation. https://www.kernel.org/doc/html/latest/userspace-api/media/drivers/uvcvideo.html

[S5] Linux V4L2 streaming with memory mapping. https://cdn.kernel.org/doc/html/latest/userspace-api/media/v4l/mmap.html

[S6] Linux DMA mapping guide. https://docs.kernel.org/core-api/dma-api-howto.html

The Linux references describe API concepts; implementation must use the headers, features and driver interfaces available in our deployed BSP. All acceptance targets and task ownership above are proposals for team agreement.
