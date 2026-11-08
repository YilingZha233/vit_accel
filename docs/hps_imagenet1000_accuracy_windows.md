# Actual HPS-only TinyViT accuracy: 1,000 ImageNet images

Prepared 2 November 2026. This corrects the earlier WSL-only Q1 workflow for the requested **DE1-SoC HPS execution**.

## Experiment definition

Both complete models run on the board's ARM CPU using your existing ncnn `model_runner`: FP32 and the native ncnn INT8 model you already converted. This is the model whose two-thread pilot took approximately 3.199 seconds/image. It is **not Q1 numerical emulation**, and native integer kernel coverage remains unverified. No FPGA fabric offload is involved.

Use exactly the same 1,000 ImageNet validation images, preprocessing and labels for both runs. Preserve the existing INT8 model and its original ten-image calibration: this answers how that deployed model performs on a larger evaluation set. Do not run `run_accuracy1000.sh` or recalibrate midway through this comparison. More representative recalibration can be a separately named follow-up experiment.

WSL prepares inputs and summarizes returned counters. **Every prediction used in the final accuracy report is made on the HPS.** Twenty batches of 100 evaluations (ten FP32, ten INT8) provide restart points. Model loading repeats per batch; timing from this workflow is not a replacement for the previous steady-state performance benchmarks.

## 1. Transfer the code ZIP to Windows

Copy `hps_imagenet1000_accuracy_tools.zip` into Windows Downloads. Open Ubuntu/WSL:

```bash
cd /home/tiff/cpu_baseline_tools
source .venv-model/bin/activate
python -m zipfile -e /mnt/c/Users/tiffa/Downloads/hps_imagenet1000_accuracy_tools.zip /home/tiff/cpu_baseline_tools
python -m pip install 'scipy==1.16.2'
python -m unittest discover -s tests -p 'test_*accuracy1000.py' -v
```

Use the Python 3.12 environment already configured. No ncnn rebuild is needed. The package adds scripts and instructions; it does not replace your model runner or model implementations.

## 2. Obtain the dataset, if not already downloaded

Use your account at [official ImageNet downloads](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php). Save these files in Windows Downloads:

- `ILSVRC2012_img_val.tar` (~6.3 GB, validation images).
- `ILSVRC2012_devkit_t12.tar.gz` (~2.5 MB, development kit Tasks 1 & 2, including labels).

Do not download the training archive or challenge test images for this workflow. The preparation code maps official devkit class IDs through WNIDs into checkpoint class indices; simply subtracting one from official IDs is incorrect. Downloads require your own authorized access. The ZIP contains no images.

Allow approximately 10 GB of free Windows/WSL space for the archive, subsets, staging copies and results. Allow at least 1 GB free on your USB drive. Do not reformat the drive.

If `data/imagenet_accuracy1000/manifest.json` was already successfully prepared using the earlier package, reuse it and skip the next command. Otherwise run in WSL:

```bash
python scripts/prepare_imagenet_accuracy1000.py \
  --val-tar /mnt/c/Users/tiffa/Downloads/ILSVRC2012_img_val.tar \
  --devkit /mnt/c/Users/tiffa/Downloads/ILSVRC2012_devkit_t12.tar.gz \
  --out data/imagenet_accuracy1000
```

The helper checks the official archive MD5 and selects a fixed-seed random subset. It prepares 1,000 `test` images plus 512 separate `calib` images. **This board workflow uses only the 1,000 `test` images; it does not use those 512 calibration images.** Here `test` is a local split name, not official challenge test data. This is a validation subset, not the full 50,000-image benchmark. If already inspected for model tuning, describe that honestly; do not claim it is a fresh untouched test set.

## 3. Prepare the USB folder in WSL

Check required existing artifacts:

```bash
ls build/model-runner-arm/model_runner
ls models/exports/smoke10/tinyvit_fp32.ncnn.param models/exports/smoke10/tinyvit_fp32.ncnn.bin
ls models/exports/tinyvit_int8_pilot10.ncnn.param models/exports/tinyvit_int8_pilot10.ncnn.bin
ls data/manifests/quant_pilot.json
```

`quant_pilot.json` identifies the original calibration images used for the native model. Keep its provenance association with the existing conversion log/table. The script checks that selected evaluation content does not overlap these calibration images. If it reports overlap, choose a new subset seed before any evaluation, using new output paths; do not ignore leakage.

Prepare inputs, models, runner, labels and checksums:

```bash
python scripts/prepare_hps_accuracy1000.py
```

This creates `transfer/hps_accuracy1000/`. It performs preprocessing only, not desktop model inference. Inputs are RGB, resize-short-edge 256, bicubic, center-crop 224, normalized, stored as little-endian float32 CHW. The same preprocessing config is recorded in the experiment metadata.

Copy the folder to Windows Downloads:

```bash
cp -r transfer/hps_accuracy1000 /mnt/c/Users/tiffa/Downloads/
```

Use a new destination if that folder already exists; avoid mixing old/new bundles. In Windows File Explorer, copy the whole `hps_accuracy1000` folder onto the USB drive's top level. Expect roughly 725 MiB including 574 MiB of inputs, both models and the runner. Do not copy these large inputs onto the board's small root partition.

If USB is currently on the board, first run in PuTTY:

```bash
cd /home/root
umount /mnt/usb
```

Wait for the prompt before moving it to Windows. After copying, safely eject the USB from Windows and reconnect it to the board.

## 4. Mount the USB and check the board

In **PuTTY**, run:

```bash
uname -m
cat /proc/partitions
mount
```

Expect `armv7l`. Identify the USB partition; in your previous sessions it was `/dev/sda1`. If it is still `/dev/sda1` and is not already mounted:

```bash
mkdir -p /mnt/usb
mount -o ro /dev/sda1 /mnt/usb
```

Then:

```bash
ls -lh /mnt/usb/hps_accuracy1000/model
ls /mnt/usb/hps_accuracy1000/inputs/batch_*.tsv
df -h /home/root /mnt/usb
command -v md5sum
```

Allow at least 100 MiB free under `/home/root` for executable copies and results. There are ten input TSV files. The USB remains read-only; logs and predictions are saved on the board filesystem. NTFS warnings about unrelated filenames do not establish failure of these files; the script checks every bundle file with MD5 checksums before evaluation. If file access/checksums fail, stop and repair the transfer.

## 5. Run both complete models on HPS

In PuTTY:

```bash
sh /mnt/usb/hps_accuracy1000/run.sh /home/root/hps_accuracy1000
```

The script:

1. Checks the USB bundle for transfer corruption.
2. Copies only the ARM executable to the board and marks it executable.
3. Evaluates 10 × 100 FP32 images on HPS, using two threads.
4. Evaluates the same 10 × 100 images with native ncnn INT8 on HPS.
5. Saves full logits and run JSON files, plus a small archive for accuracy reporting.

At prior measured speed, expect about **75 minutes FP32 + 53 minutes INT8**, plus input reading, model loading, checks and other overhead. Budget **2–3 hours**, possibly longer with slow USB. Keep the computer awake, PuTTY connected, USB attached and board powered. Avoid other board workloads. Each FP32 batch may take around 7.5 minutes and each INT8 batch around 5.5 minutes with no intermediate screen output.

Expected finish message:

```text
Finished both 1000-image evaluations: /home/root/hps_accuracy1000/accuracy1000_compact.tar.gz
```

Check:

```bash
ls -lh /home/root/hps_accuracy1000/accuracy1000_compact.tar.gz
```

### Interrupted runs

Do not launch two copies concurrently. After the prior process has stopped, the same command preserves batches with a `COMPLETE` marker and continues. It refuses an existing incomplete batch. Preserve that batch directory and rename it before retrying, for example (only if this exact batch is the one reported incomplete):

```bash
mv /home/root/hps_accuracy1000/results/fp32_batch_03 /home/root/hps_accuracy1000/results/fp32_batch_03_interrupted
```

Then rerun the original `sh .../run.sh ...` command. Never mark a failed batch complete by hand. Completed batches are tied to the bundle identity, and the WSL summarizer verifies counts, IDs, platform and precision. If a different model or input bundle is intended, use a new experiment/output folder.

Inspect the failed batch's `.stderr` file if the process reports an error. Leave model/input files unchanged during the experiment.

## 6. Return the compact evidence using your existing PuTTY method

The full logits (~8 MB total) remain on the board. The compact archive contains run counters and image IDs for all twenty batches, environment, hashes and experiment metadata. This is enough for top-1/top-5 accuracy; it does not contain individual logits for paired error analysis.

Enable PuTTY **Session → Logging → All session output**, filename:

```text
C:\Users\tiffa\Downloads\hps_accuracy1000_oct.log
```

In PuTTY, run each command separately, waiting for the prompt after `od`:

```bash
echo HPS_OCT_BEGIN
```

```bash
od -v -b /home/root/hps_accuracy1000/accuracy1000_compact.tar.gz
```

```bash
echo HPS_OCT_END
```

Set logging back to None and Apply to flush the log. In **WSL**:

```bash
python scripts/recover_hps_octal_archive.py \
  --log /mnt/c/Users/tiffa/Downloads/hps_accuracy1000_oct.log \
  --out results/raw/hps_accuracy1000_returned
```

The decoder checks byte offsets and gzip CRC before extraction. If it reports a gap, do not bypass the check. Retransmit the same archive into a new log; no inference rerun is needed. Do not recreate the archive when repairing an interrupted transfer.

## 7. Generate the board accuracy report in WSL

```bash
python scripts/summarize_hps_accuracy1000.py \
  --results results/raw/hps_accuracy1000_returned \
  --experiment transfer/hps_accuracy1000/experiment.json \
  --out results/reports/hps_accuracy1000
```

Then:

```bash
cat results/reports/hps_accuracy1000/REPORT.md
cat results/reports/hps_accuracy1000/accuracy_comparison.json
```

It requires all twenty eval batches to have completed 100 images each, zero failures, correct image order, ARMv7 platform, two threads and the requested precision. It sums **HPS-generated accuracy counters**. It reports FP32 and native INT8 top-1/top-5 counts, percentages, Wilson intervals and accuracy differences:

```text
Accuracy loss (percentage points) = FP32 accuracy % - native INT8 accuracy %
```

Each image contributes 0.1 percentage points. Negative loss means INT8 scored higher on this subset. Report the existing ten-image native calibration count as a limitation; no new accuracy value is promised. Do not infer a paired loss confidence interval by subtracting Wilson endpoints. Full logits retained on the board enable later paired analysis if transferred separately.

The WSL summarizer does not run either model. These are actual HPS full-graph predictions, including any FP32 fallback operations in the converted INT8 model. They are not FPGA-fabric results, not Q1 numerical-simulation results, and not application-level camera performance measurements.

## 8. Preserve results and stop safely

Keep the board's `/home/root/hps_accuracy1000/results` directory, the WSL prepared bundle/manifest and returned archive. Copy the finished report to Windows if desired:

```bash
cp results/reports/hps_accuracy1000/REPORT.md /mnt/c/Users/tiffa/Downloads/hps_accuracy1000_REPORT.md
```

In PuTTY, after evaluation and transfer have finished:

```bash
cd /home/root
sync
umount /mnt/usb
```

Only remove the USB after successful unmount. Shut down the board's Linux cleanly before powering off.

## Validation status

Helper tests exercise ImageNet label mapping, disjoint deterministic sampling, aggregation of 1,000 actual-run counters, rejection of a wrong execution platform, and serial-transfer corruption detection. Shell syntax has been checked. No 1,000-image board results have been generated by the assistant; those are produced by your runs.

Sources: [official ImageNet validation/devkit downloads](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php), [torchvision devkit mapping and archive checksums](https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py). Existing model, runner and previous board measurements come from this project and your supplied results.
