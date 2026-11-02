"""Write searched samples as a plan the canonical evaluator can run from its real reset."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from antipodal_search import Antipodal
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--physics',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--top',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=6);p.add_argument('--rung',default='real');a=p.parse_args()
t=Antipodal(a.model,a.physics,a.plan/'plan.npz');t.set_rung(a.rung);rows=json.loads(a.top.read_text())[:a.count];a.output.mkdir(parents=True,exist_ok=False)
qs=[t.Q.copy()];ts=[t.T.copy()];names=['source_unchanged'];F=t.frames
for i,r in enumerate(rows):
    Q,T=t.compile_grasp(r['sample'],F-1);fq=t.Q.copy();ft=t.T.copy();fq[:len(Q)]=Q;ft[:len(T)]=T;qs.append(fq);ts.append(ft);names.append(f"top{i}_frontier{r['frontier']}_lift{r['lift_mm']:.0f}")
qs=np.asarray(qs,np.float32);ts=np.asarray(ts,np.float32)
qs[:,[-1]]=t.Q[-1:];np.savez_compressed(a.output/'plan.npz',joint_q=qs,joint_target=ts)
m=json.loads((a.plan/'manifest.json').read_text());m.update(candidates=[dict(name=n,controller='zero residual native targets') for n in names],source_plan=str(a.plan),rung=a.rung,no_object_force_or_state_assistance=True,physical_success_verified=False);(a.output/'manifest.json').write_text(json.dumps(m,indent=1)+'\n');print(names)
