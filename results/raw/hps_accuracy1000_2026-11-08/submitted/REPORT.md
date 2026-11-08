# Full TinyViT accuracy: actual DE1-SoC HPS execution

Same 1000 ImageNet validation images; two HPS CPU threads. Existing ncnn INT8 model, not Q1 simulation.

| Metric | FP32 | Native INT8 |
|---|---:|---:|
| n | 1000 | 1000 |
| failed | 0 | 0 |
| top1_correct | 849 | 836 |
| top5_correct | 970 | 966 |
| top1_pct | 84.9 | 83.6 |
| top5_pct | 97.0 | 96.6 |
| top1_wilson95 | [82.54750432083057, 86.98538793687767] | [81.17726060331938, 85.7655812264513] |
| top5_wilson95 | [95.74965858524128, 97.89062611711653] | [95.28648097546889, 97.55686513367738] |

Top-1 accuracy loss: **1.300 percentage points**.

Top-5 accuracy loss: **0.400 percentage points**.

Existing INT8 calibration count: 10; no recalibration. A small calibration set limits conclusions about the best achievable INT8 accuracy. This is a 1000-image subset, not full ImageNet validation.

Native integer kernel coverage remains unverified. Accuracy is measured from full-graph HPS outputs. No FPGA fabric offload is used.

No paired loss interval or logit error is inferred from aggregate counts. Full logits remain on the board for optional later analysis.
