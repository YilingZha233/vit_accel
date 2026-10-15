"""Produce a report index from completed desktop/native run directories."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model_support import write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output must be new')
    result = []
    for run in a.runs:
        row = dict(run=str(run.resolve()), target='', variant='', scope='', threads='', n='',
                   top1_pct='', top5_pct='', mean_ms='', median_ms='', p95_ms='',
                   images_per_inference_second='', peak_process_rss_mib='')
        if (run/'manifest.json').exists():
            m = json.loads((run/'manifest.json').read_text())
            if m['status'] != 'complete':
                raise ValueError('Incomplete run: '+str(run))
            row.update(target=m['target'], variant=m['variant'], scope=m.get('timing_scope', m['action']),
                       threads=m['threads'], peak_process_rss_mib=m.get('process_peak_rss_mib', ''))
            if (run/'accuracy.json').exists():
                scores = json.loads((run/'accuracy.json').read_text())
                for k in ('n', 'top1_pct', 'top5_pct'):
                    row[k] = scores[k]
            if (run/'timing.json').exists():
                timings = json.loads((run/'timing.json').read_text())
                for k in ('n', 'mean_ms', 'median_ms', 'p95_ms', 'images_per_inference_second'):
                    row[k] = timings[k]
            if m['variant'] != 'F0':
                row['scope'] += '; numerical Linear-only emulation, NOT native INT8 speed'
        else:
            m = json.loads((run/'run.json').read_text())
            row.update(target='native_cpu/'+m['platform_machine'], variant=m['precision_request'],
                       scope=m['mode']+'; '+m['scope'], threads=m['threads'], n=m['completed'],
                       peak_process_rss_mib=m['peak_process_rss_mib'])
            if m['mode'] == 'bench':
                for k in ('mean_ms', 'median_ms', 'p95_ms', 'images_per_inference_second'):
                    row[k] = m[k]
            if m['accuracy_available']:
                row.update(top1_pct=100*m['top1_correct']/m['completed'], top5_pct=100*m['top5_correct']/m['completed'])
        result.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(a.out, result)
    print('Saved', a.out)


if __name__ == '__main__':
    main()
