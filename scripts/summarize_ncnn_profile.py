"""Summarize NCNN_BENCHMARK layer logs from the supplied native runner.

Drops the runner's separate first inference and explicit warmups. Logged ncnn
layer intervals are execution intervals, not primitive PyTorch operator self-time.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model_support import write_csv, write_json

LINE = re.compile(r'^\s*(\S+)\s+(\S+)\s+([0-9]+(?:\.[0-9]+)?)ms\s*\|(.*)$')


def summarize(text, run, control=None):
    if run['mode'] != 'bench' or run['completed'] <= 0 or run.get('failed', 0):
        raise ValueError('Need a completed native benchmark run')
    skip = 1+run['warmups']
    count = run['completed']
    expected = skip+count
    records = []
    for line in text.splitlines():
        m = LINE.match(line)
        if m:
            typ, name, ms, detail = m.groups()
            records.append((typ, name, float(ms), detail.strip()))
    if not records:
        raise ValueError('No layer timing lines: check NCNN_BENCHMARK=ON and stderr redirection')

    # The first executed layer (normally the stem convolution) marks each
    # inference. Other nodes, especially Split, can execute repeatedly within
    # that pass. Split by passes BEFORE dropping initialization/warmups.
    anchor = records[0][:2]
    starts = [i for i, r in enumerate(records) if r[:2] == anchor]
    if len(starts) != expected:
        raise ValueError('First layer %s: found %d inference starts; expected %d. '
                         'Inspect log boundaries before summarizing.' % (anchor[1], len(starts), expected))
    ends = starts[1:]+[len(records)]
    passes = []
    for start, end in zip(starts, ends):
        current = defaultdict(list)
        for typ, name, ms, detail in records[start:end]:
            current[(typ, name)].append((ms, detail))
        passes.append(current)
    selected_passes = passes[skip:]
    inventory = {key: len(values) for key, values in selected_passes[0].items()}
    for i, current in enumerate(selected_passes):
        counts = {key: len(values) for key, values in current.items()}
        if counts != inventory:
            raise ValueError('Measured inference %d has missing/extra layer calls; expected stable call counts. '
                             'Inspect the log rather than averaging incomplete passes.' % i)

    rows, groups = [], defaultdict(lambda: dict(ms=0., layers=0, calls=0.))
    for (typ, name), calls in inventory.items():
        selected = [current[(typ, name)] for current in selected_passes]
        # Sum repeated invocations in each image, then average across images.
        times = [sum(v[0] for v in values) for values in selected]
        call_times = [v[0] for values in selected for v in values]
        mean = statistics.mean(times)
        rows.append(dict(op_type=typ, layer_name=name, calls_per_image=calls,
                         mean_layer_ms=mean, median_layer_ms=statistics.median(times),
                         mean_call_ms=statistics.mean(call_times),
                         input_output_detail=selected[0][0][1]))
        groups[typ]['ms'] += mean
        groups[typ]['layers'] += 1
        groups[typ]['calls'] += calls
    logged = sum(r['mean_layer_ms'] for r in rows)
    wall = run['mean_ms']
    if logged <= 0 or wall <= 0:
        raise ValueError('Need positive timing totals')
    for r in rows:
        r['share_of_logged_layer_time_pct'] = 100*r['mean_layer_ms']/logged
        r['fraction_of_profiled_wall_pct'] = 100*r['mean_layer_ms']/wall
    grouped = [dict(op_type=typ, distinct_layers=g['layers'], calls_per_image=g['calls'],
                    mean_layer_ms_per_image=g['ms'], share_of_logged_layer_time_pct=100*g['ms']/logged)
               for typ, g in groups.items()]
    summary = dict(images=count, skipped_inferences=skip, distinct_logged_layers=len(rows),
                   inference_passes=len(passes), inference_anchor=anchor[1],
                   logged_calls_per_image=sum(inventory.values()),
                   layers_with_multiple_calls_per_image=[dict(op_type=typ, layer_name=name, calls_per_image=calls)
                       for (typ, name), calls in inventory.items() if calls > 1],
                   profiled_wall_mean_ms=wall, logged_layer_mean_sum_ms=logged,
                   wall_minus_logged_layers_ms=wall-logged,
                   notes=['Printed layer times have 0.01 ms resolution; printed zero is not proof of zero cost.',
                          'Use same-run denominators. Layer sums exclude some runtime work and logging overhead.',
                          'Times are deployed-layer execution intervals; fused layers include their internal operations.',
                          'mean_layer_ms sums repeated invocations per image; mean_call_ms describes one invocation.',
                          'Do not substitute profiled wall time for headline uninstrumented model latency.'])
    if control is not None:
        for key in ('mode', 'threads', 'warmups', 'completed', 'precision_request'):
            if run[key] != control[key]:
                raise ValueError('Profiling/control settings differ: '+key)
        if control.get('failed', 0) or control['mean_ms'] <= 0:
            raise ValueError('Invalid control run')
        summary['uninstrumented_control_mean_ms'] = control['mean_ms']
        summary['observed_profiling_overhead_pct'] = 100*(wall/control['mean_ms']-1)
        summary['control_note'] = 'Verify same model/input hashes and host settings in the experiment manifest.'
    return sorted(rows, key=lambda r: -r['mean_layer_ms']), sorted(grouped, key=lambda r: -r['mean_layer_ms_per_image']), summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True, help='Profiled run.json')
    p.add_argument('--control', type=Path, help='Matching uninstrumented run.json')
    p.add_argument('--out', type=Path, required=True, help='New report directory')
    a = p.parse_args()
    if a.out.exists():
        p.error('Output directory must be new')
    control = json.loads(a.control.read_text()) if a.control else None
    rows, groups, summary = summarize(a.log.read_text(errors='replace'), json.loads(a.run.read_text()), control)
    a.out.mkdir(parents=True, exist_ok=False)
    write_csv(a.out/'layers.csv', rows)
    write_csv(a.out/'operator_groups.csv', groups)
    write_json(a.out/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    print('Saved', a.out)


if __name__ == '__main__':
    main()
