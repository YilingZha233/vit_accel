"""Aggregate actual HPS eval counts; no WSL predictions or inferred accuracy."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model_support import wilson, write_json


def aggregate(root, expected, variant):
    c1 = c5 = n = 0
    for batch in range(10):
        d = root / 'compact' / f'{variant}_batch_{batch:02d}'
        r = json.loads((d / 'run.json').read_text())
        ids = json.loads((d / 'image_ids.json').read_text())
        if ids != expected[batch*100:(batch+1)*100]:
            raise ValueError('Image IDs/order mismatch: ' + str(d))
        precision = 'fp32' if variant == 'fp32' else 'runtime-int8'
        if (r['mode'] != 'eval' or r['platform_machine'] != 'armv7l'
            or r['precision_request'] != precision or r['threads'] != 2
            or r['completed'] != 100 or r['failed'] != 0 or not r['accuracy_available']):
            raise ValueError('Incomplete or wrong-platform/configuration evaluation: ' + str(d))
        if not 0 <= r['top1_correct'] <= r['top5_correct'] <= 100:
            raise ValueError('Invalid accuracy counters')
        n += r['completed']; c1 += r['top1_correct']; c5 += r['top5_correct']
    return dict(n=n, failed=0, top1_correct=c1, top5_correct=c5,
                top1_pct=100*c1/n, top5_pct=100*c5/n,
                top1_wilson95=wilson(c1,n), top5_wilson95=wilson(c5,n))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--experiment', type=Path, default=Path('transfer/hps_accuracy1000/experiment.json'))
    p.add_argument('--out', type=Path, default=Path('results/reports/hps_accuracy1000'))
    a = p.parse_args()
    exp = json.loads(a.experiment.read_text())
    if exp != json.loads((a.results/'experiment.json').read_text()):
        raise ValueError('Returned experiment differs from prepared experiment')
    ids = [r['id'] for r in exp['images']]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError('Expected 1000 unique images')
    f, q = [aggregate(a.results, ids, v) for v in ('fp32','int8')]
    result = dict(execution='DE1-SoC HPS ARMv7 CPU, ncnn, two threads', fp32=f, native_int8=q,
                  top1_loss_pp=f['top1_pct']-q['top1_pct'], top5_loss_pp=f['top5_pct']-q['top5_pct'],
                  native_int8_calibration_n=len(exp['calibration_images']),
                  note='Existing ncnn INT8 pilot; not Q1. Paired loss CI and logit metrics require full per-image outputs, omitted from compact transfer.')
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out/'accuracy_comparison.json', result)
    report = '# Full TinyViT accuracy: actual DE1-SoC HPS execution\n\n'
    report += 'Same 1000 ImageNet validation images; two HPS CPU threads. Existing ncnn INT8 model, not Q1 simulation.\n\n'
    report += '| Metric | FP32 | Native INT8 |\n|---|---:|---:|\n'
    for key in ('n','failed','top1_correct','top5_correct','top1_pct','top5_pct','top1_wilson95','top5_wilson95'):
        report += f'| {key} | {f[key]} | {q[key]} |\n'
    report += f"\nTop-1 accuracy loss: **{result['top1_loss_pp']:.3f} percentage points**.\n"
    report += f"\nTop-5 accuracy loss: **{result['top5_loss_pp']:.3f} percentage points**.\n"
    report += f"\nExisting INT8 calibration count: {result['native_int8_calibration_n']}; no recalibration. "
    report += 'A small calibration set limits conclusions about the best achievable INT8 accuracy. This is a 1000-image subset, not full ImageNet validation.\n'
    report += '\nNative integer kernel coverage remains unverified. Accuracy is measured from full-graph HPS outputs. No FPGA fabric offload is used.\n'
    report += '\nNo paired loss interval or logit error is inferred from aggregate counts. Full logits remain on the board for optional later analysis.\n'
    (a.out/'REPORT.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
