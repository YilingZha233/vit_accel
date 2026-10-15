"""Turn captured FP32 Linear tensors into real-data Q0.16 GEMM cases (bias excluded)."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model_support import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-columns', type=int, default=49, help='0 for all flattened tokens; 49 for small pilot')
    a = p.parse_args()
    if a.max_columns < 0:
        p.error('max-columns must be >=0')
    import numpy as np
    from scripts.generate_layer_vectors import references
    c = json.loads(a.calibration.read_text())
    meta = json.loads((a.capture/'manifest.json').read_text())
    if meta['status'] != 'complete' or meta['action'] != 'inspect' or meta['variant'] != 'F0':
        raise ValueError('Need a complete F0 inspection')
    for key in ('checkpoint_sha256', 'model_source_sha256', 'preprocessing'):
        if c[key] != meta[key]:
            raise ValueError('Capture/calibration mismatch: ' + key)
    files = sorted(a.capture.glob('linear_*.npz'))
    if not files:
        raise ValueError('No captured Linear tensors')
    a.out.mkdir(parents=True, exist_ok=False)
    for path in files:
        name = path.stem[len('linear_'):]
        scales = c['scales'][name]
        with np.load(path) as d:
            w, x = d['weight'].astype(np.float64), d['x'].astype(np.float64)
        x = x.reshape(-1, w.shape[1])
        if a.max_columns:
            x = x[:a.max_columns]
        sw = np.abs(w).max()/127 or 1.
        q16 = int(np.rint(sw*scales['sx']/scales['sy']*65536))
        if not 1 <= q16 <= 65535:
            raise ValueError('Unrepresentable multiplier: ' + name)
        wi = np.clip(np.rint(w/sw), -127, 127).astype(np.int8)
        xi = np.clip(np.rint(x/scales['sx']), -127, 127).astype(np.int8).T.copy()
        k, n, m = wi.shape[0], wi.shape[1], xi.shape[1]
        if n % 16 or max(k, n, m) > 4096:
            raise ValueError('Case exceeds current C benchmark shape support: ' + name)
        full, rtl = references(wi, xi, q16)
        folder = a.out/name
        folder.mkdir()
        with (folder/'input.txt').open('w') as f:
            f.write('%d %d %d %d\n' % (k, n, m, q16))
            np.savetxt(f, wi, fmt='%d'); np.savetxt(f, xi, fmt='%d')
        full.tofile(folder/'expected.full.bin'); rtl.tofile(folder/'expected.rtl.bin')
        write_json(folder/'manifest.json', dict(case=name, K_output_rows=k, N_contraction=n,
            M_columns=m, q16=q16, data_kind='checkpoint_weights_and_captured_FP32_activations',
            scope='GEMM only; bias excluded; first captured invocation, flattened tokens optionally truncated; not FPGA bank image',
            calibration_sha256=sha256(a.calibration), capture_sha256=sha256(path),
            checkpoint_sha256=meta['checkpoint_sha256'], max_columns=a.max_columns,
            sha256={f.name: sha256(f) for f in folder.iterdir() if f.is_file()}))
        print(name, (k, n, m), 'q16', q16)


if __name__ == '__main__':
    main()
