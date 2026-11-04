"""Stream F ep25: release-timing edit of a physical teacher. For each requested side, the hand's 28 commanded channels
(wrist pose + fingers) are frozen at their value at T_hold for every later frame (the hand keeps holding the board instead
of opening and withdrawing). Nothing else changes (other hand, timing, object targets). Executes in the teacher's frozen
scene and evaluates the official-style proxy. usage: hold_release.py SRC OUT [left=T] [right=T]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(LEG/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
src,out=Path(sys.argv[1]),Path(sys.argv[2]); holds=dict(a.split('=') for a in sys.argv[3:])
prev=json.loads((src/'result.json').read_text()); scene=prev['execution']['scene']; m=mujoco.MjModel.from_xml_path(scene)
ref=TR.load_reference(src/'reference.npz',m); Q=ref['qpos'].copy(); fps=ref['fps']
for side,t in holds.items():
    cols=[m.joint(i).qposadr[0] for i in range(m.njnt) if m.joint(i).name.startswith(f'robot_{side}_{side}_')]
    assert len(cols)==28, cols
    k=int(round(float(t)*fps)); Q[k:,cols]=Q[k,cols]
out.mkdir(parents=True,exist_ok=True)
refp=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source=f'hold (no release) {holds} from {src.name}',floating_runtime=True,no_retiming=True)
EP=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/episodes/episode_000025'); man=json.loads((EP/'input_manifest.json').read_text())
ex=execute_reference(scene,refp,out/'execution'); met=evaluate_files(man['source'],ex['trajectory'],EP/'input_manifest.json',ex['scene'],EP/'metric_geometry.json')
res=dict(episode='episode_000025',input_dir=str(EP),reference=str(refp),execution=ex,metrics=met,parent=str(src),hold=holds,spec=prev.get('spec'))
(out/'result.json').write_text(json.dumps(res,indent=1,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x)))
print(json.dumps(dict(out=out.name,hold=holds,objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in met['official_style_reference_proxy']['objects'].items()})),flush=True)
