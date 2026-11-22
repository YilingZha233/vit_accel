#!/usr/bin/env python3
"""Summarize completed persistent-camera requests; excludes first/cold by default."""
import argparse
import csv
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--include-first', action='store_true')
    args = parser.parse_args()
    with args.csv.open(newline='', encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if args.include_first or r['first_request'] == '0']
    print(f'Completed requests: {len(all_rows)}; selected: {len(rows)}')
    if not rows:
        parser.exit(1, 'No selected samples; run live mode longer or use --include-first.\n')
    print('Metric (ms)                     median        p95')
    for key in ['decode_ms', 'preprocess_ms', 'infer_ms', 'pack_ms',
                'transport_overhead_ms', 'roundtrip_ms', 'request_ms', 'received_to_result_ms']:
        values = sorted(float(r[key]) for r in rows)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            parser.error(f'Invalid values in {key}')
        print(f'{key:30s} {statistics.median(values):9.2f} {values[math.ceil(.95*len(values))-1]:10.2f}')
    print('Transport overhead is round-trip minus server service, not pure wire time.')
    print('Receive-to-result starts at Windows JPEG receipt, not camera exposure.')


if __name__ == '__main__':
    main()
