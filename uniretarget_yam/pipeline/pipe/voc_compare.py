"""No VOC against VOC by time, same starting grasp, scored with VOC off. Prints final scores, else the latest training check."""
import json,os
O='/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/out/';R='/scratch2/exaFLOPs/srl_yam_run/comparison/results/'
def fin(rn):
    f=R+f'upper_test_{rn}_s42.json'
    if not os.path.exists(f):return None
    b={}
    for x in json.load(open(f)):b.setdefault(x['steps'],[]).append(x)
    v={k:min(min(h['passed'] for h in y.get('holds',[y])) for y in u) for k,u in b.items()}
    k=max(v,key=lambda s:(v[s],-sum(y['cm'] if y['cm']==y['cm'] else 1e6 for y in b[s])))
    return f"FINAL {round(v[k]*16)}/16 @{k//10**6}M {b[k][0]['cm']:.2f}cm fr{b[k][0]['frontier']}"
def chk(ep,d):
    f=O+f'recon_{ep:06d}/{d}/upper_updates.jsonl'
    if not os.path.exists(f):return 'not started'
    x=[json.loads(l) for l in open(f)]
    if not x:return 'started'
    b=x[-1]['baseline'];return f"chk {x[-1]['step']//10**6}M {round(b['passed']*16)}/16 fr{b['frontier']} {b['cm']:.2f}cm {[round(h['passed']*16) for h in b.get('holds',[])]}"
rows=[(n,f'recon_{n:06d}_novoc','U-novoc-seed42',f'recon_{n:06d}_voc','U-voc-seed42') for n in (8,19,20,25,32,33)]
rows+=[(n,f'recon_{n:06d}','U-seed42',f'recon_{n:06d}_voc','U-voc-seed42') for n in (10,16,28,30,37)]
rows+=[(5,'recon_000005','U-seed42','recon_000005_g1_voc','U-g1-voc-seed42')]
for n,a,ad,b,bd in rows:print(f'ep{n}','| no VOC:',fin(a) or chk(n,ad),'| time:',fin(b) or chk(n,bd))
