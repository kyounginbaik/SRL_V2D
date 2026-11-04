"""K32-HG (port of streamK33 holdgrid.py for ep32): rigid-hold seed through the two spin bursts (3.5-5.0 s, 7.5-8.0 s), robot side only.
Freeze the spoon-in-palm relation at frame a and command the right wrist so the held spoon follows the target over [a,b], optional finger squeeze;
both bursts may be held in one table. Ranks full rollouts by spoon deg peak/mean over f60-175 with cup penalties. usage: E31_TGT=.. holdgrid32.py VAR TABLE OUT WORKERS"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,itertools,numpy as np,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).resolve().parent))
import mresH as mres
V,TB,OUT,WK=sys.argv[1],sys.argv[2],Path(sys.argv[3]),int(sys.argv[4]);OUT.mkdir(parents=True,exist_ok=True)
CFG=dict(ep=32,actor='/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH12_ep32/runs/save_e1f0/actor',meshes='/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep32_H12e1f0/episode_000032/meshes/episode_000032',table=TB,freeze=None)
E=None
def init():
    global E;E=mres.Env(CFG)
def job(a):
    name,tb=a;r=E.evaluate(tb,keep=True)
    if 'err' not in r:return dict(name=name,fail=True)
    e=mres.L.per_frame(32,E.poses(r['Q']),E.reg,E.tgt,want_mppe=True);d=e['wooden_spoon'][1];c=e['blue_cup']
    return dict(name=name,cup=[float(c[0].mean()),float(c[1].mean())],spoon=[float(e['wooden_spoon'][0].mean()),float(d.mean())],peak=float(d[60:175].max()),pkf=int(60+d[60:175].argmax()),m1=float(d[65:105].mean()),m2=float(d[135:170].mean()),mppe=float(e['_mppe'].mean()),cupmax=float(c[0].max()))
def pred(env,Q,a,b):
    Qf=Q.copy();oa=env.oa['wooden_spoon'];wq=env.wq['right'];Rw=R.from_euler('XYZ',Q[:,wq[3:]]);pw=Q[:,wq[:3]];Rs=R.from_quat(Q[:,oa+3:oa+7],scalar_first=True);ps=Q[:,oa:oa+3]
    Ih=Rw[a].inv()*Rs[a];ih=Rw[a].inv().apply(ps[a]-pw[a]);k=np.arange(a,b+1)
    Qf[k,oa:oa+3]=pw[k]+Rw[k].apply(ih);Qf[k,oa+3:oa+7]=(Rw[k]*Ih).as_quat(scalar_first=True);return Qf
if __name__=='__main__':
    env=mres.Env(CFG);t0=env.table0.copy();r0=env.evaluate(t0,keep=True);Q,A=r0['Q'],r0['A'];jobs=[('base',t0)]
    B1=[None,(62,96),(62,102),(66,96),(66,102),(70,100)];B2=[None,(136,162),(136,170),(141,162),(141,170),(146,166)]
    for w1,w2 in itertools.product(B1,B2):
        if w1 is None and w2 is None:continue
        for rp in (0.15,0.3):
            for sq in (0.0,0.2,0.4):
                t=t0.copy();nm=f"h{w1}_{w2}_r{rp}_s{sq}".replace(' ','').replace('(','').replace(')','').replace(',','-')
                for w in (w1,w2):
                    if w is None:continue
                    a,b=w;Qf=pred(env,Q,a,b);c=env.ilc_correction(Qf,A,'wooden_spoon','right',a/env.fps,b/env.fps,rp);t[:,:6]+=c
                    if sq:ww=mres.sm((env.t-a/env.fps)/rp)*(1-mres.sm((env.t-(b/env.fps-rp))/rp));t[:,6]+=sq*ww;t[:,7]+=sq*ww
                jobs.append((nm,t))
    print('candidates',len(jobs),flush=True)
    with ProcessPoolExecutor(max_workers=WK,mp_context=mp.get_context('spawn'),initializer=init) as p:res=list(p.map(job,jobs,chunksize=1))
    res=[r for r in res if not r.get('fail')];res.sort(key=lambda r:r['peak']/3+r['m1']+r['m2']+r['spoon'][1]+20*max(0,r['cup'][0]-1.2)+2*max(0,r['cup'][1]-11)+10*max(0,r['spoon'][0]-1.2))
    (OUT/'grid.json').write_text(json.dumps(res,indent=1));tbs=dict(jobs)
    for r in res[:10]:np.savez_compressed(OUT/f"{r['name']}.npz",table=tbs[r['name']])
    for r in [x for x in res if x['name']=='base']+res[:15]:print(json.dumps({k:(round(v,2) if isinstance(v,float) else ([round(x,2) for x in v] if isinstance(v,list) else v)) for k,v in r.items()}))
