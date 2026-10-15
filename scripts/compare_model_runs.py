"""Compare paired desktop evaluations, or exported FP32 goldens with ncnn outputs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model_support import write_json, wilson


def compare_arrays(ref, cand, labels, resamples=10000, seed=496):
    import numpy as np
    ref, cand = np.asarray(ref), np.asarray(cand)
    labels = np.asarray(labels)
    if ref.shape != cand.shape or ref.shape != (len(labels), 1000) or not len(labels):
        raise ValueError('Need matching nonempty [images,1000] logits')
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError('Nonfinite logits')
    rp, cp = ref.argmax(1), cand.argmax(1)
    rc, cc = rp == labels, cp == labels
    # Paired IID image bootstrap; sampling the three possible correctness
    # differences is equivalent to resampling image pairs for mean accuracy loss.
    diffs = rc.astype(int)-cc.astype(int)
    counts = np.array([(diffs == x).sum() for x in (-1, 0, 1)])
    draws = np.random.default_rng(seed).multinomial(len(labels), counts/len(labels), size=resamples)
    boot = 100*(draws[:, 2]-draws[:, 0])/len(labels)
    err = cand.astype(np.float64)-ref.astype(np.float64)
    norm = np.linalg.norm(ref.astype(np.float64), axis=1)*np.linalg.norm(cand.astype(np.float64), axis=1)
    valid = norm > 0
    cosine = (ref.astype(np.float64)*cand.astype(np.float64)).sum(1)[valid]/norm[valid]
    top5 = np.argpartition(cand, -5, axis=1)[:, -5:]
    return dict(n=len(labels), reference_top1_pct=100*rc.mean(), candidate_top1_pct=100*cc.mean(),
                candidate_top5_pct=100*(top5 == labels[:, None]).any(1).mean(),
                candidate_top1_wilson95=wilson(int(cc.sum()), len(labels)),
                top1_loss_pp=100*diffs.mean(), paired_loss_ci95_pp=np.quantile(boot, [.025, .975]).tolist(),
                loss_upper95_one_sided_pp=float(np.quantile(boot, .95)),
                prediction_agreement_pct=100*(rp == cp).mean(),
                logit_mae=float(np.abs(err).mean()), logit_rmse=float(np.sqrt((err*err).mean())),
                logit_max_abs_error=float(np.abs(err).max()),
                mean_cosine=float(cosine.mean()) if valid.any() else None,
                zero_norm_images=int((~valid).sum()), bootstrap_seed=seed, bootstrap_resamples=resamples,
                interval_scope='paired IID empirical bootstrap; not a guarantee for unseen rare errors or stratified sampling')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--native', action='store_true', help='reference is export folder; candidate is ncnn eval folder')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--resamples', type=int, default=10000)
    p.add_argument('--atol', type=float, default=1e-4)
    p.add_argument('--rtol', type=float, default=1e-3)
    args = p.parse_args()
    if args.resamples < 100 or args.out.exists():
        p.error('Use >=100 resamples and a new output file')
    import numpy as np
    rm = json.loads((args.reference/'manifest.json').read_text())
    if rm['status'] != 'complete':
        raise ValueError('Reference run incomplete')
    if args.native:
        nm = json.loads((args.candidate/'run.json').read_text())
        if nm['mode'] != 'eval' or nm['completed'] != len(rm['images']):
            raise ValueError('Native run incomplete or not evaluation')
        ids = (args.candidate/'image_ids.json').read_text()
        if json.loads(ids) != [r['id'] for r in rm['images']]:
            raise ValueError('Native image order mismatch')
        ref = np.stack([np.fromfile(args.reference/('%05d.logits.bin' % i), dtype='<f4') for i in range(len(rm['images']))])
        cand = np.stack([np.fromfile(args.candidate/('%05d.logits.bin' % i), dtype='<f4') for i in range(len(rm['images']))])
    else:
        cm = json.loads((args.candidate/'manifest.json').read_text())
        if cm['status'] != 'complete':
            raise ValueError('Candidate incomplete')
        for key in ('images', 'preprocessing', 'checkpoint_sha256', 'model_source_sha256'):
            if rm[key] != cm[key]:
                raise ValueError('Runs differ in ' + key)
        ref, cand = (np.load(d/'logits.npy', mmap_mode='r') for d in (args.reference, args.candidate))
    result = compare_arrays(ref, cand, [r['label'] for r in rm['images']], args.resamples)
    result.update(reference=str(args.reference.resolve()), candidate=str(args.candidate.resolve()),
                  all_logits_within_tolerance=bool(np.allclose(ref, cand, atol=args.atol, rtol=args.rtol)),
                  atol=args.atol, rtol=args.rtol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
