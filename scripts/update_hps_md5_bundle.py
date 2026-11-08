"""Update an existing prepared USB bundle without regenerating model inputs."""
from pathlib import Path
import hashlib
import shutil

root = Path('transfer/hps_accuracy1000')
if not (root / 'experiment.json').is_file():
    raise SystemExit('Run from cpu_baseline_tools; existing prepared bundle is required')
shutil.copyfile('scripts/run_hps_accuracy1000.sh', root / 'run.sh')
md5_rows, sha_rows = [], []
for f in sorted(root.rglob('*')):
    if not f.is_file() or f.relative_to(root).as_posix() in ('MD5SUMS', 'SHA256SUMS', 'CHECKSUMS.cksum'):
        continue
    md5, sha = hashlib.md5(), hashlib.sha256()
    with f.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            md5.update(chunk); sha.update(chunk)
    name = f.relative_to(root).as_posix()
    md5_rows.append(f'{md5.hexdigest()}  {name}\n')
    sha_rows.append(f'{sha.hexdigest()}  {name}\n')
(root / 'MD5SUMS').write_text(''.join(md5_rows))
(root / 'SHA256SUMS').write_text(''.join(sha_rows))
print('Updated', len(md5_rows), 'file checksums.')
print('Copy only run.sh, MD5SUMS and SHA256SUMS to the USB hps_accuracy1000 folder.')
