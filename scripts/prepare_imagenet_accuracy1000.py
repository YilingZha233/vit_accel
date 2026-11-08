"""Select a reproducible ImageNet validation subset using official devkit labels."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model_support import freeze_data, sha256


def select_indices(total, evaluation, calibration, seed):
    if min(evaluation, calibration) < 1 or evaluation + calibration > total:
        raise ValueError('Invalid subset counts')
    indices = random.Random(seed).sample(range(total), evaluation + calibration)
    return indices[:evaluation], indices[evaluation:]


def checkpoint_labels(synsets, ground_truth):
    # ILSVRC numeric IDs are NOT checkpoint class indices. Map through WNIDs.
    id_to_wnid = {int(s.ILSVRC2012_ID): str(s.WNID)
                  for s in synsets if int(s.num_children) == 0}
    if len(id_to_wnid) != 1000 or len(set(id_to_wnid.values())) != 1000:
        raise ValueError('Expected 1000 unique leaf synsets')
    ordered = sorted(id_to_wnid.values())
    index = {wnid: i for i, wnid in enumerate(ordered)}
    if ordered[0] != 'n01440764' or ordered[-1] != 'n15075141':
        raise ValueError('Unexpected ImageNet class order')
    return [index[id_to_wnid[x]] for x in ground_truth], ordered


def member_bytes(archive, suffix):
    matches = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError('Expected exactly one ' + suffix)
    return archive.extractfile(matches[0]).read()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--val-tar', required=True, type=Path)
    p.add_argument('--devkit', required=True, type=Path)
    p.add_argument('--out', type=Path, default=Path('data/imagenet_accuracy1000'))
    p.add_argument('--seed', type=int, default=4961000)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output already exists; preserve it or choose a new --out')
    from scipy.io import loadmat
    from PIL import Image
    print('Checking official validation archive MD5 (reads the 6.3 GB archive)...', flush=True)
    h = hashlib.md5()
    with a.val_tar.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != '29b22e2961454d5413ddabcf34fc5622':
        raise ValueError('Validation archive does not match official ILSVRC2012 archive MD5')
    with tarfile.open(a.devkit, 'r:*') as t:
        synsets = loadmat(io.BytesIO(member_bytes(t, '/data/meta.mat')),
                          squeeze_me=True, struct_as_record=False)['synsets']
        truth = [int(x) for x in member_bytes(t, '/data/ILSVRC2012_validation_ground_truth.txt').split()]
    if len(truth) != 50000:
        raise ValueError('Expected 50000 validation labels')
    labels, wnids = checkpoint_labels(synsets, truth)
    test, calib = select_indices(50000, 1000, 512, a.seed)
    selected = {f'ILSVRC2012_val_{i+1:08d}.JPEG': (i, split)
                for split, indices in [('test', test), ('calib', calib)] for i in indices}
    a.out.mkdir(parents=True)
    images = a.out / 'images'
    images.mkdir()
    found = set()
    print('Extracting only 1512 selected images...', flush=True)
    with tarfile.open(a.val_tar, 'r|') as archive:
        for member in archive:
            name = Path(member.name).name
            if name not in selected or not member.isfile():
                continue
            if name in found:
                raise ValueError('Duplicate archive filename ' + name)
            contents = archive.extractfile(member).read()
            with Image.open(io.BytesIO(contents)) as im:
                im.verify()
            (images / name).write_bytes(contents)
            found.add(name)
    if found != set(selected):
        raise ValueError('Selected images missing from archive')
    csv_path = a.out / 'images.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'path', 'label', 'split'])
        for split, indices in [('test', test), ('calib', calib)]:
            for i in indices:
                name = f'ILSVRC2012_val_{i+1:08d}.JPEG'
                writer.writerow([name, 'images/' + name, labels[i], split])
    freeze_data(csv_path, a.out / 'manifest.json')
    provenance = dict(seed=a.seed, evaluation_n=1000, calibration_n=512,
        sampling='uniform random without replacement from official validation archive; disjoint subsets',
        label_mapping='ILSVRC ID -> leaf WNID -> lexicographically sorted WNID checkpoint index',
        validation_archive_md5=h.hexdigest(), devkit_sha256=sha256(a.devkit),
        class_wnids=wnids,
        scope='1000-image validation subset; calibration uses separate validation images. Not full 50000-image benchmark.')
    (a.out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Saved', a.out / 'manifest.json', '\nEvaluation: 1000; calibration: 512')


if __name__ == '__main__':
    main()
