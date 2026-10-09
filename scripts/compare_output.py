"""Compare a full 2048-byte FPGA output-bank dump to a selected reference."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.reference import unpack_output, row_major_bytes

parser = argparse.ArgumentParser()
parser.add_argument("case", type=Path)
parser.add_argument("dump", type=Path)
parser.add_argument("--reference", choices=["rtl", "full"], default="rtl")
args = parser.parse_args()
meta = json.loads((args.case / "manifest.json").read_text())
y = unpack_output(args.dump.read_bytes(), meta["K_output_rows"], meta["M_columns"])
actual = row_major_bytes(y)
expected = (args.case / f"expected.{args.reference}.bin").read_bytes()
if len(actual) != len(expected):
    raise SystemExit("Reference length mismatch")
bad = [i for i, (a,b) in enumerate(zip(actual, expected)) if a != b]
print(json.dumps({"case": meta["case"], "reference": args.reference,
                  "elements": len(actual), "mismatches": len(bad),
                  "first_mismatch_indices": bad[:10], "pass": not bad}))
raise SystemExit(bool(bad))
