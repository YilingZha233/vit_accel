"""Select validation-only images from ifigotin/imagenetmini-1000 for HPS evaluation."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import sys
from zipfile import ZipFile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model_support import freeze_data, sha256


def inventory(names):
    classes, validation = set(), []
    for name in names:
        parts = PurePosixPath(name).parts
        if len(parts) != 4 or parts[0] != 'imagenet-mini' or parts[1] not in ('train', 'val'):
            continue
        if not re.fullmatch(r'n[0-9]{8}', parts[2]) or not parts[3].lower().endswith('.jpeg'):
            continue
        classes.add(parts[2])
        if parts[1] == 'val':
            if not re.fullmatch(r'ILSVRC2012_val_[0-9]{8}\.JPEG', parts[3]):
                raise ValueError('Unexpected validation filename: ' + name)
            validation.append(name)
    ordered = sorted(classes)
    if len(ordered) != 1000 or ordered[:2] != ['n01440764', 'n01443537'] or ordered[-1] != 'n15075141':
        raise ValueError('Expected complete 1000-class ImageNet WNID folder inventory; cannot infer partial label order')
    if len(set(validation)) != len(validation) or len(validation) < 1000:
        raise ValueError('Duplicate validation names or fewer than 1000 validation images')
    return {w: i for i, w in enumerate(ordered)}, sorted(validation)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('data/kaggle_hps1000'))
    p.add_argument('--calibration-manifest', type=Path, default=Path('data/manifests/quant_pilot.json'))
    p.add_argument('--seed', type=int, default=4961000)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output exists; preserve it and choose a new --out')
    from PIL import Image
    previous = json.loads(a.calibration_manifest.read_text())['images']
    calibration_hashes = {r['sha256'] for r in previous if r['split'] == 'calib'}
    if not calibration_hashes:
        raise ValueError('Original native calibration image hashes are required')
    with ZipFile(a.zip) as z:
        mapping, names = inventory(z.namelist())
        print('Validation images:', len(names), '; mapped classes:', len(mapping), flush=True)
        random.Random(a.seed).shuffle(names)
        selected, hashes, excluded = [], set(), 0
        a.out.mkdir(parents=True)
        (a.out / 'images').mkdir()
        for name in names:
            raw = z.read(name)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in calibration_hashes or digest in hashes:
                excluded += 1
                continue
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            parts = PurePosixPath(name).parts
            dest = a.out / 'images' / parts[-1]
            if dest.exists():
                raise ValueError('Repeated image ID across classes')
            dest.write_bytes(raw)
            selected.append(dict(id=parts[-1], path='images/' + parts[-1],
                                 label=mapping[parts[2]], split='test'))
            hashes.add(digest)
            if len(selected) % 100 == 0:
                print('Selected', len(selected), '/ 1000', flush=True)
            if len(selected) == 1000:
                break
        if len(selected) != 1000:
            raise ValueError('Not enough nonoverlapping unique evaluation images')
    with (a.out / 'images.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'path', 'label', 'split'])
        w.writeheader(); w.writerows(selected)
    freeze_data(a.out / 'images.csv', a.out / 'manifest.json')
    print('Hashing downloaded ZIP for provenance...', flush=True)
    provenance = dict(source='https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000',
        dataset_scope='Kaggle ImageNet-mini validation subset; not full official validation benchmark',
        archive_sha256=sha256(a.zip), seed=a.seed, evaluation_n=1000,
        validation_candidates=len(names), excluded_duplicate_or_calibration_content=excluded,
        evaluation_classes=len({r['label'] for r in selected}),
        calibration_manifest_sha256=sha256(a.calibration_manifest),
        label_mapping='Sorted full 1000 WNID folder inventory, zero-based ImageNet checkpoint indices',
        class_map=mapping, calibration='Existing native INT8 model unchanged; no recalibration')
    (a.out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Saved', a.out / 'manifest.json')
    print('Evaluation images: 1000; native INT8 calibration unchanged.')


if __name__ == '__main__':
    main()
