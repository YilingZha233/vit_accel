"""Run the C benchmark; retain raw samples, summary rows and host metadata."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=Path('build/bench_matmul'))
    p.add_argument('--vectors',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--target',choices=['desktop_cpu','hps_cpu'],required=True)
    p.add_argument('--runs',type=int,default=20)
    p.add_argument('--warmups',type=int,default=3)
    p.add_argument('--inner',type=int,default=1)
    a=p.parse_args()
    if a.target=='hps_cpu' and not (platform.system()=='Linux' and platform.machine().startswith('armv7')):
        p.error('hps_cpu requires ARMv7 Linux; use desktop_cpu for development')
    if not 1<=a.runs<=100000 or not 1<=a.inner<=100000 or not 0<=a.warmups<=100000:
        p.error('Invalid counts')
    cases=sorted(a.vectors.glob('*/manifest.json'))
    if not cases: p.error('No case manifests found; generate vectors first')
    binary=a.binary.resolve(strict=True)
    a.out.mkdir(parents=True,exist_ok=False)
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'target':a.target,
              'platform':platform.platform(),'machine':platform.machine(),
              'runs':a.runs,'warmups':a.warmups,'calls_per_sample':a.inner,
              'threads':1,'scope':'C GEMM kernels only, not full-model inference',
              'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
              'clock_note':'sysfs snapshot, not a continuously monitored clock',
              'configuration_files':{},'cases':[]}
    candidates=[Path('/proc/cpuinfo'),Path('/proc/version'),Path('/proc/meminfo')]
    candidates+=list(Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_*'))
    for file in candidates:
        try:metadata['configuration_files'][str(file)]=file.read_text()
        except OSError:pass
    (a.out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for manifest in cases:
        case=manifest.parent
        print(f'Running {case.name} ({a.runs} samples per arithmetic contract)',flush=True)
        env=dict(os.environ,BENCH_SAMPLES_PATH=str((a.out/(case.name+'.samples.csv')).resolve()))
        command=[str(binary),str(case.resolve()),str(a.runs),str(a.inner),str(a.warmups)]
        start=time.monotonic()
        result=subprocess.run(command,env=env,text=True,capture_output=True)
        (a.out/(case.name+'.csv')).write_text(result.stdout)
        (a.out/(case.name+'.log')).write_text(result.stderr)
        metadata['cases'].append({'case':case.name,'command':command,'returncode':result.returncode,
            'wall_seconds':time.monotonic()-start,'vector_manifest':json.loads(manifest.read_text())})
        (a.out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if result.returncode:raise SystemExit(f'{case.name} failed; inspect its log in {a.out}')
        rows=list(csv.DictReader(io.StringIO(result.stdout)))
        if len(rows)!=2 or any(row['verified']!='true' for row in rows):
            raise SystemExit('Missing or failed verification in benchmark output')
        summary=a.out/'summary.csv'
        first=not summary.exists()
        with summary.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            if first:writer.writeheader()
            writer.writerows(rows)
    print(f'Saved summary, samples, logs and run.json in {a.out}')


if __name__=='__main__':
    main()
