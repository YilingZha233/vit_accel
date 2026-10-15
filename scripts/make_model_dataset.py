"""Index class-folder data with an explicit checkpoint class map; no dataset download."""
import argparse
import csv
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-root', required=True, type=Path)
    p.add_argument('--val-root', required=True, type=Path)
    p.add_argument('--class-map', required=True, type=Path, help='JSON {folder_name: checkpoint_index}; 1000 entries')
    p.add_argument('--calibration-count', type=int, default=512)
    p.add_argument('--dev-count', type=int, default=1000)
    p.add_argument('--seed', type=int, default=496)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    mapping = json.loads(a.class_map.read_text())
    if len(mapping) != 1000 or set(mapping.values()) != set(range(1000)):
        p.error('Class map must explicitly map 1000 names to unique checkpoint indices 0..999')
    if a.calibration_count < 1 or a.dev_count < 1 or a.out.exists():
        p.error('Use positive sample counts and a new CSV path')
    suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    def scan(root):
        rows = []
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            if folder.name not in mapping:
                raise ValueError('Unknown class folder: ' + folder.name)
            for f in sorted(folder.rglob('*')):
                if f.is_file() and f.suffix.lower() in suffixes:
                    rows.append((str(f.relative_to(root)), str(f.resolve()), mapping[folder.name]))
        if not rows:
            raise ValueError('No class-folder images at ' + str(root))
        return rows
    train, val = scan(a.train_root), scan(a.val_root)
    random.Random(a.seed).shuffle(train)
    if len(train) < a.calibration_count+a.dev_count:
        raise ValueError('Not enough training images for disjoint calibration/development sets')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'path', 'label', 'split'])
        for i, (name, path, label) in enumerate(train[:a.calibration_count+a.dev_count]):
            w.writerow(['train/'+name, path, label, 'calib' if i < a.calibration_count else 'dev'])
        for name, path, label in val:
            w.writerow(['val/'+name, path, label, 'test'])
    print('Wrote', a.out, '; next run freeze-data for file hashing and leakage checks.')


if __name__ == '__main__':
    main()
