"""Hold re-plan on a teacher reference: from t_hold the carrying hand's wrist AND finger commands are frozen at their t_hold
values until the end of the episode (the robot keeps supporting the placed object instead of releasing it). Optionally the
wrist follows the object's reference pose with the fixed t_hold offset (identical when the object target is static).
Objects, clock and the other hand untouched. Chain on any reference dir (result.json + reference.npz).
usage: hold_variant.py --teacher DIR --episode-dir DIR --out DIR [--side right --object wooden_spoon --t-hold 4.75 --ramp 0.2]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
ROOT=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(ROOT/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR, hand as H
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
sm=lambda u:np.clip(u,0,1)**3*(10-15*np.clip(u,0,1)+6*np.clip(u,0,1)**2)
def dump(p,o): Path(p).write_text(json.dumps(o,indent=1,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
p=argparse.ArgumentParser(); p.add_argument('--teacher',required=True); p.add_argument('--episode-dir',required=True); p.add_argument('--out',required=True)
p.add_argument('--side',default='right'); p.add_argument('--object',default='wooden_spoon'); p.add_argument('--t-hold',type=float,default=4.75); p.add_argument('--ramp',type=float,default=0.2)
a=p.parse_args(); src=Path(a.teacher); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); epd=Path(a.episode_dir)
prev=json.loads((src/'result.json').read_text()); scene=prev['execution']['scene']; m=mujoco.MjModel.from_xml_path(scene)
ref=TR.load_reference(src/'reference.npz',m); Q0=ref['qpos']; fps=ref['fps']; T=len(Q0); t=np.arange(T)/fps; Q=Q0.copy()
rig=H.HandRig.build(m,f'robot_{a.side}_',a.side); wc=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')]); fc=np.array(list(rig.finger_qpos.values()))
k0=int(round(a.t_hold*fps)); w=sm((t-a.t_hold)/a.ramp); oc=int(m.joint(a.object+'_free').qposadr[0])
orig=TR.load_reference(epd/'tracking_reference.npz',m)['qpos']; p_ref=orig[:,oc:oc+3]; R_ref=R.from_quat(orig[:,oc+3:oc+7],scalar_first=True)
# wrist pose relative to the object target at t_hold, re-applied to the (static) object target afterwards
lp=R_ref[k0].inv().apply(Q0[k0,wc[:3]]-p_ref[k0]); lR=R_ref[k0].inv()*R.from_euler('XYZ',Q0[k0,wc[3:]])
for k in range(k0,T):
    g=w[k]; ph=p_ref[k]+R_ref[k].apply(lp); Rh=R_ref[k]*lR; rk=R.from_euler('XYZ',Q0[k,wc[3:]])
    Q[k,wc[:3]]=(1-g)*Q0[k,wc[:3]]+g*ph; rel=(Rh*rk.inv()).as_rotvec(); Q[k,wc[3:]]=continuous_euler_xyz(R.from_matrix((R.from_rotvec(g*rel)*rk).as_matrix()[None]),Q0[k,wc[3:]])[0]
    Q[k,fc]=(1-g)*Q0[k,fc]+g*Q0[k0,fc]
lim=m.jnt_range[m.actuator_trnid[:,0]]; qa=m.jnt_qposadr[m.actuator_trnid[:,0]]; Q[:,qa]=np.clip(Q[:,qa],lim[:,0],lim[:,1]); Q[0]=Q0[0]
refp=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source=f'hold re-plan {vars(a)} on {src}; immutable object targets/clock',arms_coupled=False,floating_runtime=True)
ex=execute_reference(scene,refp,out/'execution'); man=json.loads((epd/'input_manifest.json').read_text())
met=evaluate_files(man['source'],ex['trajectory'],epd/'input_manifest.json',ex['scene'],epd/'metric_geometry.json')
dump(out/'result.json',dict(episode=prev.get('episode',man.get('episode')),input_dir=str(epd),reference=str(refp),execution=ex,metrics=met,variant=vars(a),source=str(src),learned_policy=False))
o=met['official_style_reference_proxy']['objects']
print(json.dumps(dict(out=str(out),mp_sr=met['reference_proxy_mp_sr'],objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in o.items()},completed=ex['execution_completed'])),flush=True)
