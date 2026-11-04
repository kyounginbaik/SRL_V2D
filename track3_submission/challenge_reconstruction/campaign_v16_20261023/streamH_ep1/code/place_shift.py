"""Stream H ep1: windowed world shift of the RIGHT wrist (lid hand) on top of a teacher reference: smooth 0.4 s blend-in at T0,
held, optional smooth 0.4 s blend-out at T1 (persisting to the end if T1 < 0). Executes in the teacher's unchanged frozen scene and
scores against the corrected-track episode dir (and the original reference for the record).
usage: place_shift.py SRC OUT DX DY DZ T0 T1"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json, numpy as np, mujoco
from pathlib import Path
L=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(L/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
H=Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1'); EPDIR=H/'staging/episode_corrected/episode_000001'
ORIG=Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000001'); OEP=L/'refinement_20261018/episode_000001/reconstruction_repair/patch_height094/repaired_adapter'
src,out=Path(sys.argv[1]),Path(sys.argv[2]); dx,dy,dz,t0,t1=map(float,sys.argv[3:8])
if (out/'result.json').exists(): raise SystemExit('done')
r=json.load(open(src/'result.json')); sc=r['execution']['scene']; m=mujoco.MjModel.from_xml_path(sc); ref=TR.load_reference(src/'reference.npz',m); Q=ref['qpos'].copy()
s=lambda u: np.clip(u,0,1)**3*(10-15*np.clip(u,0,1)+6*np.clip(u,0,1)**2); t=np.arange(len(Q))/ref['fps']
w=s((t-t0)/.4)*(1-s((t-t1)/.4) if t1>0 else 1.0)
for j,dv in zip(('robot_right_right_pos_x','robot_right_right_pos_y','robot_right_right_pos_z'),(dx,dy,dz)): Q[:,int(m.joint(j).qposadr[0])]+=dv*w
out.mkdir(parents=True,exist_ok=True)
rp=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source=f'right-wrist world shift ({dx},{dy},{dz}) m over [{t0},{t1}] s (0.4 s blends) on top of {src}',floating_runtime=True)
ex=execute_reference(sc,rp,out/'execution')
man=json.load(open(EPDIR/'input_manifest.json'))
met=evaluate_files(man['source'],ex['trajectory'],EPDIR/'input_manifest.json',ex['scene'],EPDIR/'metric_geometry.json')
meto=evaluate_files(ORIG,ex['trajectory'],OEP/'input_manifest.json',ex['scene'],OEP/'metric_geometry.json')
res=dict(episode='episode_000001',input_dir=str(EPDIR),reference=str(rp),execution=ex,metrics=met,metrics_original=meto,source=str(src),spec=r.get('spec'),place_shift=dict(dx=dx,dy=dy,dz=dz,t0=t0,t1=t1,blend_s=.4,side='right'))
(out/'result.json').write_text(json.dumps(res,indent=1,default=str))
f=lambda mt:{k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1),round(v['max_position_error_m'][0]*100,1)) for k,v in mt['official_style_reference_proxy']['objects'].items()}
print(json.dumps(dict(out=out.name,mp_sr=met['reference_proxy_mp_sr'],corrected=f(met),original=f(meto))),flush=True)
