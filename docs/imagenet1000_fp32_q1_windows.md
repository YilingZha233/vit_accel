# Run FP32 versus Q1 accuracy on 1,000 ImageNet images

Prepared 2 November 2026 for the existing Windows WSL project at `/home/tiff/cpu_baseline_tools`.

## What this test measures

Your previous shared accuracy results used **10 images**. Counts of 20 were timed inference calls, not a 20-image accuracy benchmark. This workflow evaluates exactly **1,000 different images** with both FP32 (F0) and Q1, and calibrates Q1 on **512 separate images**.

**Q1 here means your existing Python numerical simulation**: symmetric per-tensor weight/activation scales on 40 selected QKV/projection/MLP Linear modules. Other operations, including Conv, attention matrix products and the classifier, remain FP32. This is not whole-model integer execution. FP32 is the unquantized reference.

All steps below run in WSL. The DE1-SoC connection is not needed for this numerical comparison. The previous ARM `model_runner` executes ncnn models and cannot execute this Q1 simulation. Native ncnn INT8 accuracy on the board is a separate experiment with a different quantization implementation; do not name it Q1. These runs do not produce new HPS speed, memory or FPGA measurements.

## 1. Transfer the small code package

Copy `imagenet1000_accuracy_tools.zip` from the Mac to **Windows Downloads**. In Ubuntu/WSL:

```bash
cd /home/tiff/cpu_baseline_tools
source .venv-model/bin/activate
python -m zipfile -e /mnt/c/Users/tiffa/Downloads/imagenet1000_accuracy_tools.zip /home/tiff/cpu_baseline_tools
python -m pip install 'scipy==1.16.2'
python -m unittest discover -s tests -p test_imagenet_accuracy1000.py -v
```

The ZIP adds a dataset-preparation script, a run script, this guide and three focused tests. It does not replace existing model scripts. Keep the existing Python 3.12 environment, trained checkpoint and pinned TinyViT source.

Check prerequisites:

```bash
python --version
ls external/Cream/TinyViT/models/tiny_vit.py models/checkpoints/tinyvit21m.pth
ls scripts/model_baseline.py scripts/compare_model_runs.py configs/preprocessing.json
```

If any file is missing, restore your existing CPU-baseline project before proceeding. Do not download a different TinyViT checkpoint.

## 2. Obtain the official validation archive and labels

In the Windows browser, open [ImageNet ILSVRC2012 downloads](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php). Log in/request access using your own ImageNet account if needed. Download these two files to Windows Downloads:

- **Validation images:** `ILSVRC2012_img_val.tar` (approximately 6.3 GB).
- **Image development kit, Tasks 1 & 2:** `ILSVRC2012_devkit_t12.tar.gz` (approximately 2.5 MB).

Do not download the 138 GB training archive or the challenge test images for this workflow. The official validation archive has 50,000 images, but the script extracts only 1,512. The official source distributes an archive rather than our selected 1,000-image subset. Allow space for the downloaded archive plus the selected images and outputs (roughly 8–10 GB free is a practical starting allowance).

The devkit supplies actual labels. Its numeric class IDs cannot be used directly as checkpoint indices: the script maps IDs through WordNet IDs into the checkpoint's sorted 1,000-class order, following the standard torchvision ImageNet organization. It checks image readability and hashes to reject duplicated content across calibration/evaluation.

This package contains no ImageNet images or account credentials. If access is unavailable, obtain an authorized copy from your institution; do not substitute random web pictures or silently use a different dataset.

Check downloads in WSL:

```bash
ls -lh /mnt/c/Users/tiffa/Downloads/ILSVRC2012_img_val.tar
ls -lh /mnt/c/Users/tiffa/Downloads/ILSVRC2012_devkit_t12.tar.gz
```

## 3. Freeze the 1,000-image evaluation subset

```bash
python scripts/prepare_imagenet_accuracy1000.py \
  --val-tar /mnt/c/Users/tiffa/Downloads/ILSVRC2012_img_val.tar \
  --devkit /mnt/c/Users/tiffa/Downloads/ILSVRC2012_devkit_t12.tar.gz \
  --out data/imagenet_accuracy1000
```

This first checks the full validation archive against its official MD5, then scans the archive and writes only selected images under the WSL filesystem. Both operations can take several minutes without frequent output. Expected final lines:

```text
Saved data/imagenet_accuracy1000/manifest.json
Evaluation: 1000; calibration: 512
```

Selection is fixed-seed uniform sampling without replacement, not the first 1,000 filenames or one image per class. Both models use the exact same evaluation IDs/order. A separate 512-image calibration subset comes from the remaining validation images; therefore describe this as a **1,000-image ImageNet validation-subset study with disjoint validation calibration**, not the standard full 50,000-image benchmark. It does not guarantee coverage of every class. These images were not used to train this workflow's weights; the checkpoint's original training provenance is unchanged.

The `test` field in our manifest is a local evaluation-split name, not the official ILSVRC challenge test set. Preserve `manifest.json`, `images.csv` and `provenance.json`. Do not regenerate seeds or adjust quantization based on these results and continue calling the same subset untouched. Existing ten-image calibration is not reused; 512-image calibration changes the scale configuration, so differences from the old pilot reflect both dataset and calibration changes.

## 4. Run calibration, FP32 and Q1

Keep the Windows computer awake and leave WSL open. The script prints evaluation progress every 25 images; calibration may be quiet. Q1 simulation performs extra numerical/diagnostic work, so it may be much slower than FP32. No fixed completion time is promised.

```bash
bash scripts/run_accuracy1000.sh
```

The script stops on the first failure and performs, in order:

1. Calibrate on 512 images.
2. FP32 evaluate on 1,000 images.
3. Q1 evaluate on those same 1,000 images.
4. Compute paired accuracy differences, agreement and logit error.
5. Write `REPORT.md`.

Default is two WSL CPU threads. To use another thread count before the first run:

```bash
THREADS=4 bash scripts/run_accuracy1000.sh
```

Outputs go to `results/raw/imagenet1000_q1/`. The script refuses to overwrite an existing output directory. For a deliberately new experiment:

```bash
bash scripts/run_accuracy1000.sh data/imagenet_accuracy1000/manifest.json results/raw/imagenet1000_q1_run2
```

### If a run stops

Preserve its log and failed directory. Do not delete results or treat a partial run as complete. Evaluation does not resume partway through an individual 1,000-image pass. If calibration and F0 completed successfully and Q1 alone failed, rerun only Q1 into a fresh directory:

```bash
python scripts/model_baseline.py evaluate \
  --source external/Cream/TinyViT \
  --checkpoint models/checkpoints/tinyvit21m.pth \
  --manifest data/imagenet_accuracy1000/manifest.json \
  --split test --limit 1000 --threads 2 --variant Q1 \
  --calibration results/raw/imagenet1000_q1/calibration/calibration.json \
  --out results/raw/imagenet1000_q1/Q1_retry
```

Then compare the successful retry:

```bash
python scripts/compare_model_runs.py \
  --reference results/raw/imagenet1000_q1/F0 \
  --candidate results/raw/imagenet1000_q1/Q1_retry \
  --out results/raw/imagenet1000_q1/comparison_retry.json
```

This manual retry produces the comparison JSON; the automated `REPORT.md` creation belongs to a successful full run. Diagnose the original error before retrying.

## 5. Read the accuracy result

```bash
cat results/raw/imagenet1000_q1/REPORT.md
cat results/raw/imagenet1000_q1/F0/accuracy.json
cat results/raw/imagenet1000_q1/Q1/accuracy.json
cat results/raw/imagenet1000_q1/comparison.json
```

Record top-1/top-5 percentages, N=1,000, calibration N=512, failures, confidence intervals, prediction agreement, logit RMSE and configuration hashes. The requested accuracy drop is:

```text
Top-1 loss in percentage points = FP32 top-1 % - Q1 top-1 %
```

For example only: 82.0% minus 80.8% is 1.2 percentage points, not a measured result here. A negative loss means Q1 performed better on this subset. With 1,000 images each image contributes 0.1 percentage points. Small losses still need uncertainty estimates; 1,000 images do not guarantee detection of a tiny accuracy difference. `all_logits_within_tolerance: false` does not itself mean classification failed.

Accuracy checks abort on unreadable images or invalid/nonfinite outputs instead of silently dropping them. Only accept a run that finishes with 1,000 images and completed status in both run manifests.

## 6. Copy the results to Windows

```bash
tar -czf /mnt/c/Users/tiffa/Downloads/imagenet1000_q1_results.tar.gz \
  results/raw/imagenet1000_q1 \
  data/imagenet_accuracy1000/manifest.json \
  data/imagenet_accuracy1000/images.csv \
  data/imagenet_accuracy1000/provenance.json
```

This archives numerical results and provenance, not the ImageNet images. Keep the dataset locally so hashes and predictions can be reproduced. Paste `REPORT.md` and `comparison.json` for interpretation.

## Board testing after this comparison

To measure deployed ncnn INT8 accuracy on these 1,000 images, use a separate export/native calibration/conversion/evaluation experiment. The HPS command's `runtime-int8` option does not select Q1. Do not send this Python Q1 script to PuTTY or substitute its numerical output for ncnn results.

Based on the earlier two-thread pilots, 1,000 native board inferences alone would take roughly 75 minutes for FP32 and 53 minutes for ncnn INT8, excluding transfer/load/preprocessing and allowing for variation. Storage must be planned: 1,000 raw 224×224×3 float32 inputs need about 574 MiB, in addition to executables, models and results; the board's small root partition may not have enough space. Board execution is unnecessary to answer the requested FP32-versus-Q1 numerical accuracy question.

## Validation and sources

The preparation helper has tests for reproducible disjoint sampling, invalid counts and mapping nontrivial official class IDs to checkpoint indices. Shell syntax is checked. This package has not been run against your ImageNet archive or evaluated on 1,000 images by the assistant; actual accuracy is produced when you run it.

- [Official ImageNet downloads and devkit](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)
- [Torchvision ImageNet implementation: archive checksums and devkit mapping](https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py)
