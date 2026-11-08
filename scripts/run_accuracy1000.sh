#!/usr/bin/env bash
# Run from project root with .venv-model activated. New output directory required.
set -euo pipefail
MANIFEST=${1:-data/imagenet_accuracy1000/manifest.json}
OUT=${2:-results/raw/imagenet1000_q1}
THREADS=${THREADS:-2}
if [ -e "$OUT" ]; then
  echo "Output exists: $OUT. Choose a new output path; existing results are preserved." >&2
  exit 1
fi
python - "$MANIFEST" <<'PY'
import json, sys
from collections import Counter
rows = json.load(open(sys.argv[1]))['images']
c = Counter(r['split'] for r in rows)
if c['test'] != 1000 or c['calib'] != 512:
    raise SystemExit('Need exactly 1000 test and 512 calibration images')
PY
mkdir -p "$OUT"
COMMON=(--source external/Cream/TinyViT --checkpoint models/checkpoints/tinyvit21m.pth --manifest "$MANIFEST" --threads "$THREADS")
python scripts/model_baseline.py calibrate "${COMMON[@]}" --split calib --limit 512 --out "$OUT/calibration" 2>&1 | tee "$OUT/calibration.log"
python scripts/model_baseline.py evaluate "${COMMON[@]}" --split test --limit 1000 --variant F0 --out "$OUT/F0" 2>&1 | tee "$OUT/F0.log"
python scripts/model_baseline.py evaluate "${COMMON[@]}" --split test --limit 1000 --variant Q1 --calibration "$OUT/calibration/calibration.json" --out "$OUT/Q1" 2>&1 | tee "$OUT/Q1.log"
python scripts/compare_model_runs.py --reference "$OUT/F0" --candidate "$OUT/Q1" --out "$OUT/comparison.json" 2>&1 | tee "$OUT/comparison.log"
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
a, b, c = [json.loads((p / f).read_text()) for f in ['F0/accuracy.json', 'Q1/accuracy.json', 'comparison.json']]
assert a['n'] == b['n'] == c['n'] == 1000
report = f'''# FP32 versus Q1: 1000 ImageNet validation images

Execution: WSL CPU. Q1 is selected-Linear numerical emulation, not native ncnn INT8 or FPGA execution.
Calibration: 512 disjoint validation images. Evaluation: fixed random 1000-image subset, not full ImageNet validation.

| Metric | FP32 | Q1 |
|---|---:|---:|
| Images | {a['n']} | {b['n']} |
| Top-1 % | {a['top1_pct']:.3f} | {b['top1_pct']:.3f} |
| Top-5 % | {a['top5_pct']:.3f} | {b['top5_pct']:.3f} |
| Top-1 Wilson 95% interval | {a['top1_wilson95']} | {b['top1_wilson95']} |
| Failures | {a['failed']} | {b['failed']} |

Top-1 loss (FP32 minus Q1): {c['top1_loss_pp']:.3f} percentage points.
Top-5 loss: {a['top5_pct'] - b['top5_pct']:.3f} percentage points.
Paired empirical-bootstrap top-1 loss 95% interval: {c['paired_loss_ci95_pp']} percentage points.
Prediction agreement: {c['prediction_agreement_pct']:.3f}%.
Logit RMSE: {c['logit_rmse']:.6f}. Mean cosine: {c['mean_cosine']:.6f}.

Intervals describe sampling uncertainty under their stated assumptions, not guarantees for unseen errors.
See manifest.json in each run for hashes, command, software versions and completion status.
Q1 uses per-tensor weight/activation scales on selected Linears; remaining operations use FP32.
'''
(p / 'REPORT.md').write_text(report)
print(report)
PY
