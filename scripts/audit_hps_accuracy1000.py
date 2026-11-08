"""Audit a preserved attachments ZIP import against recovered HPS batch counters."""
import argparse
import hashlib
import json
from pathlib import Path
from summarize_hps_accuracy1000 import aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    recovered=a.root/'recovered'; submitted=a.root/'submitted'
    for name in ['experiment.json','environment.txt','MD5SUMS','SHA256SUMS','bundle.identity']:
        if (recovered/name).read_bytes() != (submitted/name).read_bytes():
            raise ValueError('Outer/inner evidence mismatch: '+name)
    exp=json.loads((recovered/'experiment.json').read_text())
    rows=exp['images']; ids=[r['id'] for r in rows]
    hashes={r['sha256'] for r in rows}
    if len(ids)!=1000 or len(set(ids))!=1000 or len(hashes)!=1000:
        raise ValueError('Expected 1000 unique image IDs and content hashes')
    if hashes & {r['sha256'] for r in exp['calibration_images']}:
        raise ValueError('Calibration/evaluation overlap')
    if any(r['split']!='test' or not 0<=r['label']<1000 for r in rows):
        raise ValueError('Invalid evaluation labels/split')
    submitted_result=json.loads((submitted/'accuracy_comparison.json').read_text())
    result={}
    per_batch=[]
    for variant,key in [('fp32','fp32'),('int8','native_int8')]:
        totals=aggregate(recovered,ids,variant)
        for metric,value in totals.items():
            actual=submitted_result[key][metric]
            if isinstance(value,list):
                assert all(abs(x-y)<1e-9 for x,y in zip(value,actual)), metric
            else: assert abs(value-actual)<1e-9, metric
        result[key]=totals
        runs=[]
        for batch in range(10):
            r=json.loads((recovered/'compact'/f'{variant}_batch_{batch:02d}'/'run.json').read_text())
            runs.append(r)
            per_batch.append(dict(variant=variant,batch=batch,completed=r['completed'],
                                  top1_correct=r['top1_correct'],top5_correct=r['top5_correct']))
        totals['evaluation_weighted_mean_ms']=sum(r['mean_ms']*r['completed'] for r in runs)/1000
        totals['max_batch_process_peak_rss_mib']=max(r['peak_process_rss_mib'] for r in runs)
        totals['summed_service_loop_seconds']=sum(r['service_loop_seconds'] for r in runs)
    result['top1_loss_pp']=result['fp32']['top1_pct']-result['native_int8']['top1_pct']
    result['top5_loss_pp']=result['fp32']['top5_pct']-result['native_int8']['top5_pct']
    for k in ['top1_loss_pp','top5_loss_pp']:
        assert abs(result[k]-submitted_result[k])<1e-9
    result.update(status='passed',evaluation_unique_classes=len({r['label'] for r in rows}),
                  calibration_n=len(exp['calibration_images']),per_batch=per_batch,
                  validation='20 batch counters, exact image order, ARMv7/two-thread/precision scope, 1000 unique IDs/hashes, calibration hash disjointness, supplied summary reconciliation',
                  limitations='Original images, binaries and full logits not supplied. Recorded hashes identify artifacts but cannot verify absent file contents. No paired loss CI or overall p95 can be reconstructed from aggregate counts.')
    result['preserved_file_sha256']={str(f.relative_to(a.root)):hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in sorted(a.root.rglob('*')) if f.is_file()}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_batch','preserved_file_sha256')},indent=2))


if __name__=='__main__': main()
