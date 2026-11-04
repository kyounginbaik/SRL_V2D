"""Seated-grasp wrist re-plan: take the in-hand pose the object settles into (palm<-object at t_seat in the teacher execution)
and command the carrying wrist so that this in-hand pose placed on the immutable reference object track is realised from t_on
to t_off (blended). The fingers keep their reference commands. Objects, clock and the other hand untouched. Chainable.
usage: seated_grasp.py --teacher DIR --episode-dir DIR --out DIR [--side right --object wooden_spoon --t-seat 15.0 --t-on 0.7
       --t-off 18.8 --ramp 0.5 --seat-from EXEC_TRAJ_NPZ (default: teacher execution) --rot-only]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
ROOT=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(ROOT/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
sys.path.insert(0,str(Path(__file__).resolve().parent)); from curl_variant import inhand, dump
sm=lambda u:np.clip(u,0,1)**3*(10-15*np.clip(u,0,1)+6*np.clip(u,0,1)**2)
def T(pos,mat): t=np.eye(4); t[:3,:3]=np.asarray(mat).reshape(3,3); t[:3,3]=pos; return t

def palm_model(m,side):
    """Tw_palm = Trans(p) . R_XYZ(q_rot) . R_fixed  (verified: constant residual)."""
    d=mujoco.MjData(m); pre=f'robot_{side}_{side}_'; wc=[int(m.joint(pre+'pos_'+x).qposadr[0]) for x in 'xyz']; wr=[int(m.joint(pre+'rot_'+x).qposadr[0]) for x in 'xyz']
    palm=m.body(pre+'base_yaw').id; mujoco.mj_resetData(m,d); mujoco.mj_forward(m,d)
    Rf=R.from_euler('XYZ',d.qpos[wr]).as_matrix().T@d.xmat[palm].reshape(3,3)
    return wc,wr,palm,Rf

def build(m,Q0,fps,a,epd,seat_traj):
    wc,wr,palm,Rf=palm_model(m,a.side); T0=len(Q0); t=np.arange(T0)/fps
    orig=TR.load_reference(epd/'tracking_reference.npz',m)['qpos']; oc=int(m.joint(a.object+'_free').qposadr[0])
    z=np.load(seat_traj); ks=int(round(a.t_seat*fps)); d=mujoco.MjData(m); d.qpos[:]=z['qpos'][ks]; mujoco.mj_forward(m,d); ob=m.body(a.object).id
    Tp_o=np.linalg.inv(T(d.xpos[palm],d.xmat[palm]))@T(d.xpos[ob],d.xmat[ob])
    if a.rot_only:   # keep the teacher's own palm<-object translation at each frame, only impose the settled rotation
        pass
    w=sm((t-a.t_on)/a.ramp)*(1-sm((t-a.t_off)/a.ramp)); Q=Q0.copy(); invTp=np.linalg.inv(Tp_o)
    for k in range(T0):
        if w[k]<=0: continue
        Tw_o=T(orig[k,oc:oc+3],R.from_quat(orig[k,oc+3:oc+7],scalar_first=True).as_matrix()); Tw_p=Tw_o@invTp
        Rq=Tw_p[:3,:3]@Rf.T; p=Tw_p[:3,3]
        if a.rot_only:
            # translation: keep teacher wrist position; rotation: impose
            p=Q0[k,wc]
        R0=R.from_euler('XYZ',Q0[k,wr]); Rt=R.from_matrix(Rq); rel=(Rt*R0.inv()).as_rotvec()
        Q[k,wc]=(1-w[k])*Q0[k,wc]+w[k]*p; Q[k,wr]=continuous_euler_xyz(R.from_matrix((R.from_rotvec(w[k]*rel)*R0).as_matrix()[None]),Q0[k,wr])[0]
    return Q, Tp_o

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--teacher',required=True); p.add_argument('--episode-dir',required=True); p.add_argument('--out',required=True)
    p.add_argument('--side',default='right'); p.add_argument('--object',default='wooden_spoon'); p.add_argument('--seat-from'); p.add_argument('--rot-only',action='store_true')
    for k,v in (('t_seat',15.),('t_on',.7),('t_off',18.8),('ramp',.5)): p.add_argument('--'+k.replace('_','-'),type=float,default=v)
    a=p.parse_args(); src=Path(a.teacher); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); epd=Path(a.episode_dir)
    prev=json.loads((src/'result.json').read_text()); scene=prev['execution']['scene']; m=mujoco.MjModel.from_xml_path(scene)
    ref=TR.load_reference(src/'reference.npz',m); Q0=ref['qpos']; fps=ref['fps']; seat=a.seat_from or prev['execution']['trajectory']
    Q,Tp_o=build(m,Q0,fps,a,epd,seat); lim=m.jnt_range[m.actuator_trnid[:,0]]; qa=m.jnt_qposadr[m.actuator_trnid[:,0]]; Q[:,qa]=np.clip(Q[:,qa],lim[:,0],lim[:,1]); Q[0]=Q0[0]
    refp=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source=f'seated-grasp wrist re-plan {vars(a)} on {src}; immutable object targets/clock',arms_coupled=False,floating_runtime=True)
    ex=execute_reference(scene,refp,out/'execution'); man=json.loads((epd/'input_manifest.json').read_text())
    met=evaluate_files(man['source'],ex['trajectory'],epd/'input_manifest.json',ex['scene'],epd/'metric_geometry.json'); ih=inhand(m,np.load(ex['trajectory']),a.object,a.side)
    dump(out/'result.json',dict(episode=prev.get('episode',man.get('episode')),input_dir=str(epd),reference=str(refp),execution=ex,metrics=met,inhand=ih,variant=vars(a),seat_palm_from_object=Tp_o,source=str(src),learned_policy=False))
    o=met['official_style_reference_proxy']['objects']
    print(json.dumps(dict(out=str(out),mp_sr=met['reference_proxy_mp_sr'],objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in o.items()},inhand=ih,completed=ex['execution_completed'])),flush=True)
