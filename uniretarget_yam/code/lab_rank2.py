"""Placed grasps with over-closure: official-style mean error on the real goal, plus a fast lift."""
import argparse,glob,json,sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from grasp_lab import Lab
p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--goal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
rest=json.loads((a.fit/'result.json').read_text())['rest_frame'];lab=None
coverage={f"candidate_{r['index']:03d}":r.get('demand_coverage',0) for r in json.loads((a.fit/'attempts.json').read_text())['attempts'] if 'cost' in r}
def job(args):
    global lab
    if lab is None:lab=Lab(a.fit/'native_model.mjb',a.goal/'physics.npz',rest);lab.real=(lab.gp,lab.gr,lab.gp_raw,lab.gr_raw)
    path,grip,close,mode=args;z=np.load(path);star=z['qpos'];op=star[lab.qo:lab.qo+3].copy();o0=R.from_quat(star[lab.qo+3:lab.qo+7][[1,2,3,0]])
    if mode=='real':lab.gp,lab.gr,lab.gp_raw,lab.gr_raw=lab.real;h=86
    else:
        n=len(lab.real[0]);u=np.clip((np.arange(n)-lab.onset)/10.,0,1);s=u*u*(3-2*u);gp=op+np.outer(s*.08,lab.up);gr=R.concatenate([o0]*n);lab.gp=lab.gp_raw=gp;lab.gr=lab.gr_raw=gr;h=30
    r=lab.run(star,z['patch_positions'],z['patch_normals'],z['hand_geom_ids'],grip,horizon=h,overclose=close);r.update(candidate=Path(path).stem,grip=grip,close=close,mode=mode,coverage=coverage.get(Path(path).stem,0));return r
jobs=[(f,g,c,m) for f in sorted(glob.glob(str(a.fit/'candidate_*.npz'))) for g in (3.,8.) for c in (0.,.005,.01,.02) for m in ('fast','real')]
with Pool(a.workers) as pool:rows=pool.map(job,jobs,chunksize=1)
a.output.write_text(json.dumps(rows,indent=1)+'\n')
by={}
for r in rows:by.setdefault((r['candidate'],r['grip'],r['close']),{})[r['mode']]=r
print('jobs',len(rows),'candidate grip close cov | fast lift_mm cm deg | real mean_cm mean_deg within firstBreak contact lift')
ranked=sorted(by.items(),key=lambda kv:(kv[1]['real'].get('mean_cm',99)/3+kv[1]['real'].get('mean_deg',999)/30))
for (c,g,cl),m in ranked[:20]:
    f,r=m['fast'],m['real'];print(c,int(g),cl,round(r['coverage'],2),'|',round(f.get('lift_mm',0),1),round(f.get('mean_cm',0),1),round(f.get('mean_deg',0),1),'|',round(r.get('mean_cm',0),1),round(r.get('mean_deg',0),1),round(r.get('within',0),2),r.get('first_break'),r.get('contact_frames'),round(r.get('lift_mm',0),1))
lift=[k for k,m in by.items() if m['fast'].get('lift_mm',0)>60]
print('lifting >60mm:',len(lift),sorted(set((k[0],round(by[k]['real']['coverage'],2)) for k in lift)))
