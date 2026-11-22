#!/usr/bin/env python3
"""Completed HPS-local requests; first cold request excluded by default."""
import argparse
import csv
import math
import statistics
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('csv');p.add_argument('--include-first',action='store_true')
a=p.parse_args()
with open(a.csv,newline='') as f:all_rows=list(csv.DictReader(f))
rows=[r for r in all_rows if a.include_first or r['first_request']=='0']
print('Completed: {}; selected: {}; >=100 selected: {}'.format(len(all_rows),len(rows),len(rows)>=100))
if not rows:p.exit(1,'No selected samples; use --include-first for a snapshot.\n')
print('{:26s} {:>10s} {:>10s} {:>10s}'.format('Metric (ms)','mean','median','p95'))
for key in ['decode_ms','preprocess_ms','infer_ms','pack_ms','ipc_overhead_ms','request_ms','received_to_result_ms','publish_ms']:
    values=sorted(float(r[key]) for r in rows)
    if not all(math.isfinite(v) and v>=0 for v in values):p.error('Invalid timing: '+key)
    print('{:26s} {:10.2f} {:10.2f} {:10.2f}'.format(key,statistics.mean(values),statistics.median(values),values[math.ceil(.95*len(values))-1]))
if len(rows)>1:
    duration=float(rows[-1]['completed_since_start_s'])-float(rows[0]['completed_since_start_s'])
    if duration>0:print('Completion-event rate across {:.2f}s: {:.4f} classifications/s'.format(duration,(len(rows)-1)/duration))
print('No camera exposure-to-screen measurement. No FPGA/DMA work in this run.')
