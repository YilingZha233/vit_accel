"""Read-only numeric audit of exported HPS CSVs; writes a JSON audit separately."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
output = Path(sys.argv[2])
issues = []
audits = []
for suite, count in [('pilot', 3), ('report', 20)]:
    folder = root / suite
    rows = list(csv.DictReader((folder / 'summary.csv').open()))
    assert len(rows) == 22, (suite, len(rows))
    assert len({(r['case'], r['contract']) for r in rows}) == 22
    total = 0
    for row in rows:
        name = Path(row['case']).name
        samples = [s for s in csv.DictReader((folder / (name + '.samples.csv')).open())
                   if s['contract'] == row['contract']]
        assert len(samples) == count
        assert [int(s['sample']) for s in samples] == list(range(count))
        assert row['verified'] == 'true'
        assert int(row['repetitions']) == count
        per_case = list(csv.DictReader((folder / (name + '.csv')).open()))
        assert row in per_case
        assert 'compiler=15.2.0' in (folder / (name + '.log')).read_text()
        xs = [float(s['per_call_ms']) for s in samples]
        assert all(math.isfinite(x) and x > 0 for x in xs)
        for s in samples:
            assert int(s['calls_per_sample']) == int(row['calls_per_sample'])
            assert abs(float(s['group_ms']) / int(s['calls_per_sample']) - float(s['per_call_ms'])) < 1e-8
        mean = statistics.mean(xs)
        std = statistics.pstdev(xs)
        macs = int(row['k']) * int(row['n']) * int(row['m'])
        expected = dict(median_ms=statistics.median(xs), mean_ms=mean,
                        min_ms=min(xs), p95_ms=sorted(xs)[math.ceil(.95 * count)-1],
                        stddev_ms=std, cv=std/mean, macs=macs,
                        gmac_per_s=macs/(mean*1e6))
        for field, value in expected.items():
            tolerance = 1e-6 if field in ('cv', 'gmac_per_s') else 2e-8
            if abs(float(row[field]) - value) > tolerance:
                issues.append([suite, name, row['contract'], field, row[field], value])
        total += len(samples)
    audits.append(dict(suite=suite, rows=len(rows), samples=total))
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(root.rglob('*')) if p.is_file()}
result = dict(suites=audits, issues=issues, sha256=hashes,
              note='Checks CSV consistency, not a new execution or independent proof of golden correctness.')
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(suites=audits, issues=issues, files=len(hashes)), indent=2))
if issues:
    sys.exit(1)
