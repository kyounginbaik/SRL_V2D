"""Rank every placed candidate by physics on gentle lift, fast lift and the real goal."""
import argparse,glob,json,sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from grasp_lab import Lab
p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--goal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
rest=json.loads((a.fit/'result.json').read_text())['rest_frame'];lab=None
def job(args):
    global lab
    if lab is None:lab=Lab(a.fit/'native_model.mjb',a.goal/'physics.npz',rest);lab.real=(lab.gp,lab.gr,lab.gp_raw,lab.gr_raw)
    path,grip,mode=args;z=np.load(path);star=z['qpos'];op=star[lab.qo:lab.qo+3].copy();o0=R.from_quat(star[lab.qo+3:lab.qo+7][[1,2,3,0]])
    if mode=='real':lab.gp,lab.gr,lab.gp_raw,lab.gr_raw=lab.real;h=86
    else:
        seconds=2. if mode=='gentle' else .5;n=len(lab.real[0]);u=np.clip((np.arange(n)-lab.onset)/(seconds*20),0,1);s=u*u*(3-2*u)
        gp=op+np.outer(s*.08,lab.up);gr=R.concatenate([o0]*n);lab.gp=lab.gp_raw=gp;lab.gr=lab.gr_raw=gr;h=int(seconds*20)+20
    r=lab.run(star,z['patch_positions'],z['patch_normals'],z['hand_geom_ids'],grip,horizon=h);r.update(candidate=Path(path).stem,grip=grip,mode=mode);return r
jobs=[(f,g,m) for f in sorted(glob.glob(str(a.fit/'candidate_*.npz'))) for g in (3.,8.) for m in ('gentle','fast','real')]
with Pool(a.workers) as pool:rows=pool.map(job,jobs,chunksize=1)
a.output.write_text(json.dumps(rows,indent=1)+'\n')
by={}
for r in rows:by.setdefault((r['candidate'],r['grip']),{})[r['mode']]=r
out=sorted(by.items(),key=lambda kv:-kv[1]['fast'].get('lift_mm',0))
print('candidate grip | gentle lift_mm deg | fast lift_mm cm deg | real within mean_cm deg firstBreak contact')
for (c,g),m in out[:22]:
    gg,ff,rr=m['gentle'],m['fast'],m['real']
    print(c,int(g),'|',round(gg.get('lift_mm',0),1),round(gg.get('mean_deg',0),1),'|',round(ff.get('lift_mm',0),1),round(ff.get('mean_cm',0),1),round(ff.get('mean_deg',0),1),'|',round(rr.get('within',0),2),round(rr.get('mean_cm',0),1),round(rr.get('mean_deg',0),1),rr.get('first_break'),rr.get('contact_frames'))
