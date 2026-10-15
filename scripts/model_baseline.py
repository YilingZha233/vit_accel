"""Desktop TinyViT workflow. Use --help; never labels desktop runs as HPS."""
import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model_support import (boundary_modules, freeze_data, load_model, metadata,
    read_image, records, sha256, timing_stats, transform, wilson, write_csv, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    freeze = sub.add_parser('freeze-data', help='Freeze CSV id,path,label,split and detect duplicates')
    freeze.add_argument('--csv', required=True, type=Path)
    freeze.add_argument('--out', required=True, type=Path)
    for action in ('inspect', 'evaluate', 'benchmark', 'profile', 'calibrate', 'export'):
        a = sub.add_parser(action)
        a.add_argument('--source', required=True, type=Path, help='Path to upstream TinyViT directory')
        a.add_argument('--checkpoint', required=True, type=Path)
        a.add_argument('--manifest', required=True, type=Path)
        a.add_argument('--preprocessing', type=Path, default=Path('configs/preprocessing.json'))
        a.add_argument('--split', choices=['calib', 'dev', 'test'], default='dev')
        a.add_argument('--limit', type=int, default=0, help='0 = all images in split')
        a.add_argument('--offset', type=int, default=0, help='Skip this many images within the frozen split')
        a.add_argument('--threads', type=int, default=1)
        a.add_argument('--out', type=Path, required=True, help='New output directory')
        a.add_argument('--variant', choices=['F0', 'Q1', 'Q2', 'Q3', 'Q4'], default='F0')
        a.add_argument('--calibration', type=Path)
        if action == 'inspect':
            a.add_argument('--capture-linear', help='Full-match regex; save first input/weight/output of matching FP32 Linear modules')
        if action == 'benchmark':
            a.add_argument('--scope', choices=['model', 'application'], default='model')
            a.add_argument('--warmups', type=int, default=10)
            a.add_argument('--runs', type=int, default=100)
            a.add_argument('--seconds', type=float, default=0, help='Also run for at least this long')
        if action == 'calibrate':
            a.add_argument('--pattern', default=r'layers\.[123]\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.fc[12])')
    args = p.parse_args()
    if args.action == 'freeze-data':
        freeze_data(args.csv, args.out)
        return
    if args.threads < 1 or args.limit < 0 or args.offset < 0:
        p.error('threads must be positive; limit/offset nonnegative')
    if args.action == 'benchmark' and (args.runs < 1 or args.warmups < 0 or args.seconds < 0):
        p.error('Invalid benchmark counts')
    if args.variant != 'F0' and args.action not in ('evaluate', 'inspect'):
        p.error('Q1-Q4 are numerical simulations: only evaluate/inspect are supported')
    if args.action == 'inspect' and args.capture_linear and args.variant != 'F0':
        p.error('Capture real workloads from F0, then quantize with calibration')
    if args.action == 'calibrate' and args.split != 'calib':
        p.error('Calibration requires --split calib')
    if args.action in ('inspect', 'profile', 'export', 'benchmark') and not args.limit:
        p.error('Set an explicit small --limit for this diagnostic action')
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(496)
    torch.use_deterministic_algorithms(True)
    rows = records(args.manifest, args.split, args.limit, args.offset)
    args.out.mkdir(parents=True, exist_ok=False)
    meta = metadata(args)
    meta.update(action=args.action, split=args.split, variant=args.variant,
                images=[{k: r[k] for k in ('id', 'label', 'sha256')} for r in rows],
                execution='FP32' if args.variant == 'F0' else 'numerical_emulation_NOT_native_INT8')
    write_json(args.out/'manifest.json', meta)
    try:
        start = time.perf_counter()
        model = load_model(args.source, args.checkpoint)
        meta['model_construct_and_load_ms'] = (time.perf_counter()-start)*1000
        tf = transform(args.preprocessing)
        if args.variant != 'F0':
            if not args.calibration:
                raise ValueError('Q1-Q4 require --calibration')
            c = json.loads(args.calibration.read_text())
            for key in ('checkpoint_sha256', 'model_source_sha256', 'preprocessing'):
                if c[key] != meta[key]:
                    raise ValueError('Calibration does not match ' + key)
            if set(c['image_hashes']) & {r['sha256'] for r in rows}:
                raise ValueError('Calibration/evaluation overlap')
            from baseline.quant_experiment import apply_quantization
            model = apply_quantization(model, c, args.variant)
            meta['calibration_sha256'] = sha256(args.calibration)
            meta['precision_scope'] = c['precision_scope']
        with torch.inference_mode():
            if args.action == 'calibrate':
                from baseline.quant_experiment import calibrate
                c = {k: meta[k] for k in ('checkpoint_sha256', 'model_source_sha256', 'preprocessing')}
                c.update(image_hashes=[r['sha256'] for r in rows], pattern=args.pattern,
                         precision_scope='selected Linear only; Conv/attention matmuls/norm/nonlinear/bias FP32; bias after dequantization',
                         method='absmax static, symmetric [-127,127] inputs/weights; [-128,127] outputs',
                         scales=calibrate(model, (read_image(r, tf) for r in rows), args.pattern))
                write_json(args.out/'calibration.json', c)
            elif args.action == 'inspect':
                import re
                inventory, boundaries, handles = [], {}, []
                captured = set()
                def inspect_hook(name):
                    def hook(m, inputs, out):
                        inventory.append(dict(name=name, type=type(m).__name__,
                            input_shape=list(inputs[0].shape), output_shape=list(out.shape),
                            dtype=str(out.dtype)))
                        if (args.capture_linear and isinstance(m, torch.nn.Linear)
                                and re.fullmatch(args.capture_linear, name) and name not in captured):
                            np.savez(args.out/('linear_'+name+'.npz'), x=inputs[0].numpy(),
                                     weight=m.weight.numpy(), output=out.numpy(),
                                     bias=m.bias.numpy() if m.bias is not None else np.zeros(out.shape[-1], dtype=np.float32))
                            captured.add(name)
                    return hook
                def boundary_hook(name, expected):
                    def hook(m, inputs, out):
                        if list(out.shape) != expected or not torch.isfinite(out).all():
                            raise ValueError('Architecture boundary failed: ' + name)
                        boundaries[name] = out.numpy().copy()
                    return hook
                for name, m in model.named_modules():
                    if not list(m.children()):
                        handles.append(m.register_forward_hook(inspect_hook(name)))
                for name, (m, expected) in boundary_modules(model).items():
                    handles.append(m.register_forward_hook(boundary_hook(name, expected)))
                for i, row in enumerate(rows):
                    model(read_image(row, tf))
                    np.savez(args.out/('boundaries_%05d.npz' % i), **boundaries)
                for h in handles:
                    h.remove()
                if args.capture_linear and not captured:
                    raise ValueError('capture-linear matched no modules')
                write_json(args.out/'operators.json', inventory)
            elif args.action == 'evaluate':
                logits = np.lib.format.open_memmap(args.out/'logits.npy', mode='w+', dtype=np.float32,
                                                  shape=(len(rows), 1000))
                predictions = []
                for i, row in enumerate(rows):
                    y = model(read_image(row, tf)).reshape(-1)
                    if y.numel() != 1000 or not torch.isfinite(y).all():
                        raise ValueError('Invalid logits for ' + row['id'])
                    logits[i] = y.numpy()
                    top = y.topk(5).indices.tolist()
                    predictions.append(dict(id=row['id'], label=row['label'], top1=top[0],
                                            top5=json.dumps(top), correct1=int(top[0] == row['label']),
                                            correct5=int(row['label'] in top)))
                    if (i+1) % 25 == 0:
                        print('Evaluated', i+1, '/', len(rows), flush=True)
                logits.flush()
                write_csv(args.out/'predictions.csv', predictions)
                n = len(rows)
                c1, c5 = (sum(r[k] for r in predictions) for k in ('correct1', 'correct5'))
                write_json(args.out/'accuracy.json', dict(n=n, failed=0, top1_pct=100*c1/n,
                    top5_pct=100*c5/n, top1_wilson95=wilson(c1, n), top5_wilson95=wilson(c5, n)))
            elif args.action == 'benchmark':
                if len(rows) > 32:
                    raise ValueError('Benchmark preloads <=32 tensors; use --limit 10 or 32')
                tensors = [read_image(r, tf) for r in rows]
                decoded = []
                if args.scope == 'application':
                    from PIL import Image
                    for row in rows:
                        with Image.open(row['path']) as im:
                            decoded.append(im.convert('RGB').copy())
                t = time.perf_counter()
                first = model(tensors[0])
                meta['first_inference_ms'] = (time.perf_counter()-t)*1000
                if not torch.isfinite(first).all():
                    raise ValueError('Nonfinite first output')
                for i in range(args.warmups):
                    if args.scope == 'application':
                        model(tf(decoded[i % len(decoded)]).unsqueeze(0)).topk(5)
                    else:
                        model(tensors[i % len(tensors)])
                samples, n, start_loop = [], 0, time.perf_counter()
                while n < args.runs or time.perf_counter()-start_loop < args.seconds:
                    x = tensors[n % len(tensors)]
                    t = time.perf_counter()
                    if args.scope == 'application':
                        x = tf(decoded[n % len(decoded)]).unsqueeze(0)
                    y = model(x)
                    if args.scope == 'application':
                        y.topk(5)
                    ms = (time.perf_counter()-t)*1000
                    if not torch.isfinite(y).all():
                        raise ValueError('Nonfinite benchmark output')
                    samples.append(dict(sample=n, id=rows[n % len(rows)]['id'], latency_ms=ms))
                    n += 1
                meta['service_loop_seconds'] = time.perf_counter()-start_loop
                meta['service_loop_images_per_second'] = n/meta['service_loop_seconds']
                meta['timing_scope'] = ('preprocessed tensor ready -> logits' if args.scope == 'model'
                                        else 'decoded RGB image ready -> preprocessing -> logits -> top5; excludes file decode')
                meta['warmups'] = args.warmups
                write_csv(args.out/'latency.csv', samples)
                write_json(args.out/'timing.json', timing_stats([r['latency_ms'] for r in samples]))
            elif args.action == 'profile':
                if len(rows) > 32:
                    raise ValueError('Profile preloads <=32 tensors')
                tensors = [read_image(row, tf) for row in rows]
                x = tensors[0]
                for _ in range(3):
                    model(x)
                ranges, handles = {}, []
                modules = {'stem': model.patch_embed, 'head_norm': model.norm_head, 'classifier': model.head}
                for si, layer in enumerate(model.layers):
                    for bi, block in enumerate(layer.blocks):
                        modules['stage%d.block%d' % (si+1, bi)] = block
                    if layer.downsample is not None:
                        modules['transition%d' % (si+1)] = layer.downsample
                def pre(name):
                    def hook(m, inputs):
                        ranges[name] = torch.profiler.record_function('stage/'+name)
                        ranges[name].__enter__()
                    return hook
                def post(name):
                    def hook(m, inputs, output):
                        ranges.pop(name).__exit__(None, None, None)
                    return hook
                for name, m in modules.items():
                    handles.extend([m.register_forward_pre_hook(pre(name)), m.register_forward_hook(post(name))])
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                        record_shapes=True, profile_memory=True) as prof:
                    for x in tensors:
                        model(x)
                for h in handles:
                    h.remove()
                prof.export_chrome_trace(str(args.out/'trace.json'))
                write_csv(args.out/'operators.csv', [dict(op=e.key, calls=e.count,
                    self_cpu_us=e.self_cpu_time_total, cpu_total_us=e.cpu_time_total,
                    input_shapes=str(e.input_shapes)) for e in prof.key_averages(group_by_input_shape=True)])
                write_csv(args.out/'stage_profile.csv', [dict(component=e.key, calls=e.count,
                    inclusive_ms_per_image=e.cpu_time_total/1000/len(rows))
                    for e in prof.key_averages() if e.key.startswith('stage/')])
                meta['stage_scope'] = 'non-overlapping named stem/block/transition/head ranges; inclusive within each; pooling and dispatch outside these ranges'
                meta['profile_scope'] = 'desktop inference operator trace, preprocessed inputs; self times only; not HPS latency'
            elif args.action == 'export':
                from baseline.export_support import export_model
                convertible = export_model(model)
                entries = []
                for i, row in enumerate(rows):
                    x = read_image(row, tf)
                    x.numpy().astype('<f4').tofile(args.out/('%05d.input.bin' % i))
                    np.save(args.out/('%05d.input.npy' % i), x.numpy()[0])
                    model(x).numpy().astype('<f4').tofile(args.out/('%05d.logits.bin' % i))
                    entries.append('%s\t%d\t%05d.input.bin' % (row['id'], row['label'], i))
                (args.out/'inputs.tsv').write_text('\n'.join(entries)+'\n')
                (args.out/'npy_list.txt').write_text('\n'.join(str((args.out/('%05d.input.npy' % i)).resolve()) for i in range(len(rows)))+'\n')
                x = read_image(rows[0], tf)
                traced = torch.jit.trace(convertible, x, check_trace=True)
                for row in rows:
                    x = read_image(row, tf)
                    torch.testing.assert_close(traced(x), model(x), rtol=1e-4, atol=1e-5)
                traced.save(str(args.out/'tinyvit_fp32.pt'))
                meta['export_sha256'] = sha256(args.out/'tinyvit_fp32.pt')
                meta['export_rewrite'] = 'independent per-window attention via slices/cat; batch1 224 only; compared against original; conversion may duplicate shared weights and alter performance'
        if args.variant != 'F0':
            from baseline.quant_experiment import SimulatedLinear
            write_json(args.out/'quantization_diagnostics.json',
                       {n: m.report() for n, m in model.named_modules() if isinstance(m, SimulatedLinear)})
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        meta['process_peak_rss_mib'] = peak/(2**20 if platform.system() == 'Darwin' else 1024)
        meta['memory_scope'] = 'whole Python process, including load, diagnostics and export if selected'
        meta['status'] = 'complete'
    except Exception as e:
        meta['status'], meta['error'] = 'failed', str(e)
        raise
    finally:
        write_json(args.out/'manifest.json', meta)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
