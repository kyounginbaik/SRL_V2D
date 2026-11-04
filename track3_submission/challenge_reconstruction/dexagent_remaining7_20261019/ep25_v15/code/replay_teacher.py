"""Replay a teacher reference in a scene, score against the unchanged episode reference.
usage: replay_teacher.py SCENE REFERENCE OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json
from pathlib import Path
import numpy as np
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(LEG/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
scene,ref,out=sys.argv[1],sys.argv[2],Path(sys.argv[3]); out.mkdir(parents=True,exist_ok=True)
EP=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/episodes/episode_000025'); man=json.loads((EP/'input_manifest.json').read_text())
ex=execute_reference(scene,ref,out/'execution'); met=evaluate_files(man['source'],ex['trajectory'],EP/'input_manifest.json',ex['scene'],EP/'metric_geometry.json')
res=dict(episode='episode_000025',input_dir=str(EP),reference=str(ref),execution=ex,metrics=met,spec=dict(episode='episode_000025',scene=str(scene),source=man['source'],manifest=str(EP/'input_manifest.json'),geometry=str(EP/'metric_geometry.json'),original_reference=str(EP/'tracking_reference.npz')))
(out/'result.json').write_text(json.dumps(res,indent=1,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x)))
print(json.dumps(dict(out=str(out),mp_sr=met['reference_proxy_mp_sr'],objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in met['official_style_reference_proxy']['objects'].items()})),flush=True)
