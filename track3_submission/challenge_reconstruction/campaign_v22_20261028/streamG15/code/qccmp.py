import json,sys,glob,numpy as np
from pathlib import Path
a,b=sys.argv[1],sys.argv[2]; tot={a:[],b:[]}
print('ep object | iou_med p10 low% zmed zp90 (A) -> (B)')
for pa in sorted(glob.glob(f'qc/{a}/episode_*/qc.json')):
    ep=Path(pa).parent.name; pb=f'qc/{b}/{ep}/qc.json'
    A=json.load(open(pa))['objects']; B=json.load(open(pb))['objects']
    for n in A:
        sa,sb=A[n]['summary'],B[n]['summary']
        f=lambda s:(s['iou_median'],s['iou_p10'],100*s['low_iou_fraction'],s['z_median_mm'],s['z_p90_mm'])
        tot[a].append(f(sa));tot[b].append(f(sb))
        print(f"{int(ep[-6:]):2d} {n:16s} | "+' '.join(f'{x:.3f}' if i<2 else f'{x:.1f}' for i,x in enumerate(f(sa)))+'  ->  '+' '.join(f'{x:.3f}' if i<2 else f'{x:.1f}' for i,x in enumerate(f(sb))))
for k,v in tot.items(): v=np.array(v,float); print(k,'mean iou_med %.3f p10 %.3f low%% %.2f zmed %.2f zp90 %.2f'%tuple(np.nanmean(v,0)))
