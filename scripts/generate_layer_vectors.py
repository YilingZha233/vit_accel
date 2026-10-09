"""CPU-only, full-sized TinyViT-21M stage-1 GEMMs. Requires NumPy.

Synthetic INT8 data, not calibrated model tensors. No FPGA bank packing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SHAPES = {"stage1_qkv_window": (576,192,49),
          "stage1_mlp_fc1": (768,192,784),
          "stage1_mlp_fc2": (192,768,784)}


def references(w, x, q16):
    # Explicitly widen before multiplying: int8 matmul would overflow.
    w64, x64 = w.astype(np.int64), x.astype(np.int64)
    full = np.clip(((w64 @ x64)*q16) >> 16,-128,127).astype(np.int8)
    rtl = np.zeros(full.shape,dtype=np.int64)
    for start in range(0,w.shape[1],16):
        partial = w64[:,start:start+16] @ x64[start:start+16,:]
        term = np.clip((partial*q16) >> 16,-128,127)
        rtl = np.clip(rtl+term,-128,127)
    return full, rtl.astype(np.int8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('data/layers'))
    parser.add_argument('--seed',type=int,default=496)
    args=parser.parse_args()
    rng=np.random.default_rng(args.seed)
    for name,(k,n,m) in SHAPES.items():
        path=args.out/name
        path.mkdir(parents=True,exist_ok=True)
        w=rng.integers(-128,128,size=(k,n),dtype=np.int16).astype(np.int8)
        x=rng.integers(-128,128,size=(n,m),dtype=np.int16).astype(np.int8)
        q16=128  # Synthetic exercise scale, NOT a calibrated model scale.
        full,rtl=references(w,x,q16)
        with (path/'input.txt').open('w') as f:
            f.write(f'{k} {n} {m} {q16}\n')
            np.savetxt(f,w,fmt='%d');np.savetxt(f,x,fmt='%d')
        (path/'expected.full.bin').write_bytes(full.tobytes())
        (path/'expected.rtl.bin').write_bytes(rtl.tobytes())
        meta={'case':name,'model':'tiny_vit_21m_224','K_output_rows':k,
              'N_contraction':n,'M_columns':m,'q16':q16,'seed':args.seed,
              'numpy_version':np.__version__,'data_kind':'synthetic_int8',
              'scope':'GEMM only; excludes bias/nonlinearities; not an FPGA-sized transaction',
              'rtl_vs_full_mismatches':int(np.count_nonzero(full != rtl)),
              'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (path/'input.txt',path/'expected.full.bin',path/'expected.rtl.bin')}}
        (path/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(f'{name}: K={k}, N={n}, M={m}; MACs={k*n*m}; vectors ready')


if __name__=='__main__':
    main()
