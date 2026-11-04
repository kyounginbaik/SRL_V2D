"""Episode 25 grip boost: during [T0, T1] add ALPHA x (closing direction) to one hand's finger commands, where the closing
direction = reference finger pose while holding (mean over [T0+0.3, T0+1.3]) minus the open pose before the grasp (at TOPEN).
Smooth 0.3 s ramps, clipped to joint limits. Then execute and score.
usage: grip_boost.py SRC_RUN OUT SIDE T0 T1 TOPEN ALPHA [SIDE2 T0 T1 TOPEN ALPHA]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(LEG/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
src,out=Path(sys.argv[1]),Path(sys.argv[2]); jobs=[sys.argv[i:i+5] for i in range(3,len(sys.argv),5)]
prev=json.loads((src/'result.json').read_text()); scene=prev['execution']['scene']; m=mujoco.MjModel.from_xml_path(scene)
ref=TR.load_reference(src/'reference.npz',m); Q=ref['qpos'].copy(); fps=ref['fps']; t=np.arange(len(Q))/fps
b=lambda x: np.clip(x,0,1)**3*(10-15*np.clip(x,0,1)+6*np.clip(x,0,1)**2)
for side,t0,t1,topen,alpha in jobs:
    t0,t1,topen,alpha=map(float,(t0,t1,topen,alpha))
    cols=[m.joint(i).qposadr[0] for i in range(m.njnt) if m.joint(i).name.startswith(f'robot_{side}_{side}_') and not any(s in m.joint(i).name for s in ('pos_','rot_'))]
    hold=Q[(t>=t0+0.3)&(t<=t0+1.3)][:,cols].mean(0); opn=Q[int(topen*fps)][cols]; dirn=hold-opn
    w=b((t-t0)/0.3)*b((t1-t)/0.3)
    jid=[i for i in range(m.njnt) if m.joint(i).name.startswith(f'robot_{side}_{side}_') and not any(s in m.joint(i).name for s in ('pos_','rot_'))]
    lim=m.jnt_range[jid]
    Q[:,cols]=np.clip(Q[:,cols]+alpha*w[:,None]*dirn[None,:],lim[:,0],lim[:,1])
out.mkdir(parents=True,exist_ok=True)
refp=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source='grip boost on finger columns; wrist/object targets unchanged',floating_runtime=True,no_retiming=True)
EP=Path(prev['spec']['manifest']).parent if 'spec' in prev else Path(prev['input_dir']); man=json.loads((EP/'input_manifest.json').read_text())
ex=execute_reference(scene,refp,out/'execution'); met=evaluate_files(man['source'],ex['trajectory'],EP/'input_manifest.json',ex['scene'],EP/'metric_geometry.json')
res=dict(episode='episode_000025',input_dir=str(EP),reference=str(refp),execution=ex,metrics=met,parent=str(src),grip_boost=jobs,spec=prev.get('spec'))
(out/'result.json').write_text(json.dumps(res,indent=1,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x)))
print(json.dumps(dict(out=str(out),mp_sr=met['reference_proxy_mp_sr'],objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in met['official_style_reference_proxy']['objects'].items()})),flush=True)
