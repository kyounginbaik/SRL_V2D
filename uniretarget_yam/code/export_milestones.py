"""Export a loop's milestone tapes (real goal) so canonical physics can judge them."""
import argparse,json,glob,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--physics',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=6);p.add_argument('--side',default='right');p.add_argument('--object',type=int,default=0);p.add_argument('--plan-file',type=Path);a=p.parse_args()
t=Loop(a.model,a.physics,a.plan_file or a.plan/'plan.npz',a.side,a.object);t.set_rung('real');a.output.mkdir(parents=True,exist_ok=False);files=sorted(glob.glob(str(a.run/'milestones/frontier_*.json')))
files=files[:1]+files[1:][-(a.limit-1):] if len(files)>a.limit else files
qs=[t.Q.copy()];ts=[t.T.copy()];names=['source_unchanged']
for f in files:
    r=json.loads(Path(f).read_text())['result'];Q,T=t.compile_tape(r['sample'],t.frames-1);fq=t.Q.copy();ft=t.T.copy();n=len(Q)
    fq[:n]=Q;ft[:n]=T;fq[-1]=t.Q[-1];ft[-1]=t.T[-1]  # Reset and terminal rows are task contract.
    # After the compiled horizon nothing is planned: hold the last wrist and finger command.
    qs.append(fq);ts.append(ft);names.append(Path(f).stem)
qs=np.asarray(qs,np.float32);ts=np.asarray(ts,np.float32);np.savez_compressed(a.output/'plan.npz',joint_q=qs,joint_target=ts)
m=json.loads((a.plan/'manifest.json').read_text());m.update(candidates=[dict(name=n,controller='zero residual native targets') for n in names],source_plan=str(a.plan),no_object_force_or_state_assistance=True,physical_success_verified=False);(a.output/'manifest.json').write_text(json.dumps(m,indent=1)+'\n');print(json.dumps(names))
