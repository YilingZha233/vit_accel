"""Run from project root: python3 scripts/generate_vectors.py."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.reference import (full_accumulation, rtl_compatibility, pack_inputs,
                                pack_output, row_major_bytes)

REVISION = "966e3459be039e9e51f74cff9ce833c085a0178a"


def write_case(root, name, w, x, q16, seed):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    k, n, m = len(w), len(w[0]), len(x[0])
    full, rtl = full_accumulation(w, x, q16), rtl_compatibility(w, x, q16)
    wi, xi = pack_inputs(w, x)
    payloads = {
        "weights.banked.bin": wi, "inputs.banked.bin": xi,
        "expected.full.bin": row_major_bytes(full),
        "expected.rtl.bin": row_major_bytes(rtl),
        "expected.rtl.banked.bin": pack_output(rtl),
    }
    for filename, content in payloads.items():
        (folder / filename).write_bytes(content)
    lines = [f"{k} {n} {m} {q16}"]
    lines += [" ".join(map(str, row)) for matrix in (w, x) for row in matrix]
    (folder / "input.txt").write_text("\n".join(lines) + "\n")
    metadata = {
        "schema_version": 1, "case": name, "seed": seed,
        "K_output_rows": k, "N_contraction": n, "M_columns": m,
        "q16": q16, "accelerator_revision": REVISION,
        "rtl_vs_full_mismatches": sum(a != b for ar, br in zip(full, rtl) for a, b in zip(ar, br)),
        "sha256": {f: hashlib.sha256((folder / f).read_bytes()).hexdigest()
                   for f in list(payloads) + ["input.txt"]},
        "note": "Software references only; no FPGA measurement. Banked files use accelerator-local byte layout; no CPU address scaling implied.",
    }
    (folder / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=496)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    cases = []
    def add(name, w, x, q):
        cases.append(write_case(args.out, name, w, x, q, args.seed))
    add("identity16", [[2 if r == p else 0 for p in range(16)] for r in range(16)],
        [[p * 16 + c - 128 for c in range(16)] for p in range(16)], 32768)
    add("rounding_gap", [[1 if p in (0, 16) else 0 for p in range(32)]], [[1] for _ in range(32)], 32768)
    add("saturation_cancel", [[127] * 16 + [-127] * 16], [[127] for _ in range(32)], 32768)
    add("zeros", [[0] * 16 for _ in range(16)], [[0] * 5 for _ in range(16)], 65535)
    for name, k, n, m, q in [("random_single",16,16,16,1024),
                              ("random_multi",32,32,32,1024),
                              ("partial_rows",25,16,18,1024),
                              ("layer_tile",16,128,16,256)]:
        add(name, [[rng.randint(-128,127) for _ in range(n)] for _ in range(k)],
            [[rng.randint(-128,127) for _ in range(m)] for _ in range(n)], q)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(cases, indent=2) + "\n")
    for case in cases:
        print(f"{case['case']}: RTL-contract vs full-sum differences = {case['rtl_vs_full_mismatches']}")


if __name__ == "__main__":
    main()
