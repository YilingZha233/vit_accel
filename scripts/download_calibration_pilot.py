"""Download 10 DISJOINT repository-labelled examples for quantization workflow checks.

This deliberately small, fish/bird-heavy set is not representative calibration
for an accuracy claim. Expand/replace it before final quality evaluation.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model_support import freeze_data, write_csv, write_json

SAMPLES = [
    (2, 'n01484850_great_white_shark.JPEG'),
    (3, 'n01491361_tiger_shark.JPEG'),
    (4, 'n01494475_hammerhead.JPEG'),
    (5, 'n01496331_electric_ray.JPEG'),
    (6, 'n01498041_stingray.JPEG'),
    (7, 'n01514668_cock.JPEG'),
    (8, 'n01514859_hen.JPEG'),
    (10, 'n01530575_brambling.JPEG'),
    (11, 'n01531178_goldfinch.JPEG'),
    (12, 'n01532829_house_finch.JPEG'),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev-manifest', type=Path, default=Path('data/manifests/smoke10.json'))
    p.add_argument('--out', type=Path, default=Path('data/manifests/quant_pilot.json'))
    a = p.parse_args()
    csv_path = a.out.with_suffix('.csv')
    if a.out.exists() or csv_path.exists():
        p.error('Use new output paths; preserve existing calibration evidence')
    from PIL import Image
    dev = json.loads(a.dev_manifest.read_text())['images']
    if any(r['split'] != 'dev' for r in dev):
        raise ValueError('Expected the original dev-only smoke manifest')
    rows = [dict(id=r['id'], path=r['path'], label=r['label'], split='dev') for r in dev]
    dest = Path('data/images/calibration_pilot10')
    dest.mkdir(parents=True, exist_ok=True)
    base = 'https://raw.githubusercontent.com/nihui/imagenet-sample-images/master/'
    for label, name in SAMPLES:
        target = dest/name
        for attempt in range(3):
            try:
                request = urllib.request.Request(base+name, headers={'User-Agent': 'TinyViT-baseline-pilot'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    target.write_bytes(response.read())
                with Image.open(target) as im:
                    im.verify()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        rows.append(dict(id='calib_'+target.stem, path=str(target.resolve()), label=label, split='calib'))
        print('Downloaded', name, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows)
    freeze_data(csv_path, a.out)
    write_json(a.out.with_suffix('.provenance.json'), dict(source=base,
        scope='10-image nonrepresentative calibration WORKFLOW PILOT; original 10 images remain dev',
        note='Hash inventory in frozen manifest identifies actual downloads. Not a representative accuracy benchmark.'))
    print('Saved', a.out, '; development:', len(dev), '; calibration:', len(SAMPLES))


if __name__ == '__main__':
    main()
