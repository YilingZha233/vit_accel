"""Prepare USB inputs/models for actual ncnn HPS evaluation; no desktop inference."""
import argparse
import json
from pathlib import Path
import shutil
import hashlib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model_support import records, read_image, transform, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/imagenet_accuracy1000/manifest.json'))
    p.add_argument('--calibration-manifest', type=Path, default=Path('data/manifests/quant_pilot.json'))
    p.add_argument('--out', type=Path, default=Path('transfer/hps_accuracy1000'))
    a = p.parse_args()
    rows = records(a.manifest, 'test')
    if len(rows) != 1000:
        raise ValueError('Need exactly 1000 frozen evaluation images')
    calib = [r for r in json.loads(a.calibration_manifest.read_text())['images'] if r['split'] == 'calib']
    if not calib:
        raise ValueError('Need the original native model calibration manifest')
    if {r['sha256'] for r in rows} & {r['sha256'] for r in calib}:
        raise ValueError('Evaluation overlaps existing calibration; prepare a new subset seed BEFORE evaluation')
    sources = {
        'build/model_runner': Path('build/model-runner-arm/model_runner'),
        'model/fp32.param': Path('models/exports/smoke10/tinyvit_fp32.ncnn.param'),
        'model/fp32.bin': Path('models/exports/smoke10/tinyvit_fp32.ncnn.bin'),
        'model/int8.param': Path('models/exports/tinyvit_int8_pilot10.ncnn.param'),
        'model/int8.bin': Path('models/exports/tinyvit_int8_pilot10.ncnn.bin'),
        'run.sh': Path('scripts/run_hps_accuracy1000.sh'),
    }
    for src in sources.values():
        if not src.is_file():
            raise FileNotFoundError(src)
    with sources['build/model_runner'].open('rb') as f:
        header = f.read(20)
    if header[:6] != b'\x7fELF\x01\x01' or int.from_bytes(header[18:20], 'little') != 40:
        raise ValueError('model_runner must be a little-endian 32-bit ARM ELF executable')
    a.out.mkdir(parents=True, exist_ok=False)
    for target, src in sources.items():
        dest = a.out / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    (a.out / 'inputs').mkdir()
    tf = transform(Path('configs/preprocessing.json'))
    for i, row in enumerate(rows):
        tensor = read_image(row, tf).numpy().astype('<f4')
        tensor.tofile(a.out / 'inputs' / f'{i:05d}.bin')
        if (i+1) % 100 == 0:
            print('Prepared', i+1, '/ 1000', flush=True)
    for chunk in range(10):
        subset = rows[chunk*100:(chunk+1)*100]
        text = ''.join(f"{r['id']}\t{r['label']}\t{i:05d}.bin\n"
                       for i, r in enumerate(subset, chunk*100))
        (a.out / 'inputs' / f'batch_{chunk:02d}.tsv').write_text(text)
    evidence = dict(images=rows, manifest_sha256=sha256(a.manifest),
        preprocessing=json.loads(Path('configs/preprocessing.json').read_text()),
        calibration_manifest_sha256=sha256(a.calibration_manifest),
        calibration_images=calib,
        precision='existing native ncnn INT8 pilot; not Q1; no recalibration',
        model_hashes={k: sha256(v) for k, v in sources.items() if k.startswith('model/')},
        runner_sha256=sha256(sources['build/model_runner']))
    (a.out / 'experiment.json').write_text(json.dumps(evidence, indent=2) + '\n')
    files = sorted(f for f in a.out.rglob('*') if f.is_file())
    # MD5 checks accidental transfer corruption using the board-supported tool.
    checks = []
    hashes = []
    for f in files:
        rel = f.relative_to(a.out).as_posix()
        digest = hashlib.md5()
        with f.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        checks.append(f'{digest.hexdigest()}  {rel}\n')
        hashes.append(f'{sha256(f)}  {rel}\n')
    (a.out / 'MD5SUMS').write_text(''.join(checks))
    (a.out / 'SHA256SUMS').write_text(''.join(hashes))
    print('USB folder:', a.out.resolve())
    print('Preserve experiment.json locally; it is needed for aggregation.')


if __name__ == '__main__':
    main()
