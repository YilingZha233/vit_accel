"""Shared model experiment helpers. No network downloads or implicit checkpoints."""
import csv
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def timing_stats(xs):
    if not xs or any(not math.isfinite(x) or x <= 0 for x in xs):
        raise ValueError('Need positive finite timing samples')
    xs = list(xs)
    mean = statistics.mean(xs)
    return dict(n=len(xs), mean_ms=mean, median_ms=statistics.median(xs),
                p95_ms=sorted(xs)[math.ceil(.95 * len(xs))-1],
                min_ms=min(xs), max_ms=max(xs), stddev_ms=statistics.pstdev(xs),
                cv=statistics.pstdev(xs)/mean,
                images_per_inference_second=1000*len(xs)/sum(xs),
                percentile_method='nearest rank; population stddev')


def wilson(correct, n):
    if not 0 <= correct <= n or n <= 0:
        raise ValueError('Invalid accuracy counts')
    z = 1.959963984540054
    p = correct/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [100*(center-half), 100*(center+half)]


def freeze_data(csv_path, out):
    """CSV columns id,path,label,split; paths relative to CSV. Reject leakage."""
    csv_path, out = Path(csv_path), Path(out)
    if out.exists():
        raise FileExistsError(out)
    rows, ids, hashes = [], set(), set()
    with csv_path.open(newline='') as f:
        for r in csv.DictReader(f):
            if not {'id', 'path', 'label', 'split'} <= r.keys():
                raise ValueError('CSV needs id,path,label,split')
            ident = r['id']
            if not ident or any(c in ident for c in '\t\r\n') or ident in ids:
                raise ValueError('Empty, duplicate or invalid image ID: ' + ident)
            if r['split'] not in ('calib', 'dev', 'test'):
                raise ValueError('split must be calib, dev or test')
            label = int(r['label'])
            if not 0 <= label < 1000:
                raise ValueError('Labels must be checkpoint class indices 0..999')
            path = (csv_path.parent/r['path']).resolve(strict=True)
            digest = sha256(path)
            if digest in hashes:
                raise ValueError('Duplicate image content (possible split leakage): ' + str(path))
            ids.add(ident)
            hashes.add(digest)
            rows.append(dict(id=ident, path=str(path), label=label,
                             split=r['split'], sha256=digest))
    if not rows:
        raise ValueError('Empty dataset')
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, dict(schema=1, source_csv_sha256=sha256(csv_path), images=rows))


def records(path, split, limit=0, offset=0):
    data = json.loads(Path(path).read_text())
    rows = [r for r in data['images'] if r['split'] == split]
    rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError('No images in split ' + split)
    for r in rows:
        if sha256(r['path']) != r['sha256']:
            raise ValueError('Image changed since manifest: ' + r['path'])
    return rows


def metadata(args):
    import torch
    import torchvision
    import timm
    root = Path(args.source).resolve()
    rev = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
                command=sys.argv, target='desktop_cpu', platform=platform.platform(),
                machine=platform.machine(), threads=args.threads, batch_size=1,
                checkpoint_sha256=sha256(args.checkpoint),
                model_source_sha256=sha256(root/'models/tiny_vit.py'),
                upstream_commit=rev.stdout.strip() if rev.returncode == 0 else None,
                manifest_sha256=sha256(args.manifest),
                preprocessing=json.loads(Path(args.preprocessing).read_text()),
                versions=dict(torch=torch.__version__, torchvision=torchvision.__version__,
                              timm=timm.__version__), status='running')


def load_model(source, checkpoint):
    import torch
    path = Path(source)/'models/tiny_vit.py'
    spec = importlib.util.spec_from_file_location('project_tinyvit_reference', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Direct constructor avoids any pretrained download or implicit config selection.
    model = module.TinyViT(img_size=224, in_chans=3, num_classes=1000,
                          embed_dims=[96, 192, 384, 576], depths=[2, 2, 6, 2],
                          num_heads=[3, 6, 12, 18], window_sizes=[7, 7, 14, 7],
                          mlp_ratio=4., drop_path_rate=.2)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = state.get('model', state)
    state = {k: v for k, v in state.items() if not k.endswith('attention_bias_idxs')}
    model.load_state_dict(state, strict=True)
    return model.float().cpu().eval()


def transform(config):
    from torchvision import transforms as T
    c = json.loads(Path(config).read_text())
    if (c['input_size'], c['crop'], c['color'], c['interpolation']) != (224, 'center', 'RGB', 'bicubic'):
        raise ValueError('This workflow implements RGB/bicubic/center-crop 224 only')
    return T.Compose([T.Resize(c['resize_short_edge'], interpolation=T.InterpolationMode.BICUBIC),
                      T.CenterCrop(224), T.ToTensor(), T.Normalize(c['mean'], c['std'])])


def read_image(row, tf):
    from PIL import Image
    with Image.open(row['path']) as im:
        return tf(im.convert('RGB')).unsqueeze(0)


def boundary_modules(model):
    result = {'stem': (model.patch_embed, [1, 96, 56, 56])}
    shapes = [[1, 96, 56, 56], [1, 784, 192], [1, 196, 384], [1, 49, 576]]
    for i, layer in enumerate(model.layers):
        result['stage%d' % (i+1)] = (layer.blocks[-1], shapes[i])
        if layer.downsample is not None:
            result['transition%d' % (i+1)] = (layer.downsample, shapes[i+1])
    result['head_norm'] = (model.norm_head, [1, 576])
    result['logits'] = (model.head, [1, 1000])
    return result
