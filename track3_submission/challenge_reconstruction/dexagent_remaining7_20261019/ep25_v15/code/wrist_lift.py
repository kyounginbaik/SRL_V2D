"""Add a smooth world-z lift to one wrist during [T0, T1] (ramp in 0.3 s, out over 0.5 s) and execute + score.
usage: wrist_lift.py SRC_RUN OUT SIDE T0 T1 DZ"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(LEG/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
src,out,side=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]; t0,t1,dz=map(float,sys.argv[4:7])
prev=json.loads((src/'result.json').read_text()); scene=prev['execution']['scene']; m=mujoco.MjModel.from_xml_path(scene)
ref=TR.load_reference(src/'reference.npz',m); Q=ref['qpos'].copy(); t=np.arange(len(Q))/ref['fps']
b=lambda x: np.clip(x,0,1)**3*(10-15*np.clip(x,0,1)+6*np.clip(x,0,1)**2); w=b((t-t0)/0.3)*b((t1-t)/0.5)
Q[:,m.joint(f'robot_{side}_{side}_pos_z').qposadr[0]]+=dz*w
out.mkdir(parents=True,exist_ok=True); refp=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source=f'{side} wrist z lift {dz} m in [{t0},{t1}]',floating_runtime=True,no_retiming=True)
EP=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/episodes/episode_000025'); man=json.loads((EP/'input_manifest.json').read_text())
ex=execute_reference(scene,refp,out/'execution'); met=evaluate_files(man['source'],ex['trajectory'],EP/'input_manifest.json',ex['scene'],EP/'metric_geometry.json')
res=dict(episode='episode_000025',input_dir=str(EP),reference=str(refp),execution=ex,metrics=met,parent=str(src),wrist_lift=dict(side=side,t0=t0,t1=t1,dz=dz),spec=prev.get('spec'))
(out/'result.json').write_text(json.dumps(res,indent=1,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x)))
print(json.dumps(dict(out=out.name,mp_sr=met['reference_proxy_mp_sr'],objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in met['official_style_reference_proxy']['objects'].items()})),flush=True)
