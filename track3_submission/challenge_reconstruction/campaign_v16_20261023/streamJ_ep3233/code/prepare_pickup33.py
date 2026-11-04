"""Pickup-search specs (physics_pickup_search_v2) for episode 33's wooden spoon (recon world). Anchor = the base teacher's
ACHIEVED right-hand hold at T_HELD (palm<-spoon pose from the executed trajectory, closed fingers = commanded targets at T_HELD)
transplanted to the spoon's initial reference pose, so the grasp the hand actually settles into is planned from the start and
the rigid-grasp transport follows the immutable spoon track; release at the base reference's right-hand release time (the
reference spoon dives 3 cm below the cup floor at 14.5-15 s, so the plan lets go before that). Left hand = base reference.
usage: prepare_pickup33.py BASE_TEACHER_DIR OUT_DIR RELEASE_S T_HELD[,T_HELD...]"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=['/data1/SRL_V2D/third_party/DexAgent',str(LEG/'code')]
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import dump
TE=Path(sys.argv[1]); out=Path(sys.argv[2]); REL=float(sys.argv[3]); HELD=[float(x) for x in sys.argv[4].split(',')]
EP=LEG/'episodes/episode_000033'; out.mkdir(parents=True,exist_ok=True)
r=json.loads((TE/'result.json').read_text()); scene=r['execution']['scene']; man=json.loads((EP/'input_manifest.json').read_text())
m=mujoco.MjModel.from_xml_path(scene); rig=H.HandRig.build(m,'robot_right_','right'); wc=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')])
fnames=list(rig.finger_qpos); fcols=np.array([rig.finger_qpos[k] for k in fnames]); facts=np.array([rig.finger_act[k] for k in fnames])
ref=TR.load_reference(TE/'reference.npz',m); Qb=ref['qpos']; fps=ref['fps']
orig=TR.load_reference(EP/'tracking_reference.npz',m)['qpos']; oc=int(m.joint('wooden_spoon_free').qposadr[0])
p0=orig[0,oc:oc+3]; R0=R.from_quat(orig[0,oc+3:oc+7],scalar_first=True)
tr=np.load(r['execution']['trajectory']); opened=Qb[0,fcols].copy()
def anchor(lp,lR,fingers):
    a=orig[0].copy(); a[wc[:3]]=p0+R0.apply(lp); a[wc[3:]]=continuous_euler_xyz(R.from_matrix((R0*lR).as_matrix()[None]),orig[0,wc[3:]])[0]; a[fcols]=fingers; return a
specs=[]
def spec_for(name,pose,extra):
    ap=out/f'anchor_{name}.npz'; np.savez_compressed(ap,qpos=pose)
    times=dict(start=1.55,hover=1.95,contact=2.2,closed=2.5,transport=2.8,release=REL)
    s=dict(episode='episode_000033',scene=scene,reference=str(TE/'reference.npz'),original_reference=str(EP/'tracking_reference.npz'),source=man['source'],geometry=str(EP/'metric_geometry.json'),
           manifest=str(EP/'input_manifest.json'),object_name='wooden_spoon',side='right',anchor_pose=str(ap),open_fingers=opened,finger_names=fnames,times=times,
           hover_offset=[0,0,.06],horizon_s=17.3,score_hold_start_s=2.5,score_track_start_s=3.0,right_hand_release_s=REL,base_teacher=str(TE),**extra)
    dump(out/f'{name}.json',s); specs.append(str(out/f'{name}.json'))
for th in HELD:
    k=round(th*fps); q=tr['qpos'][k]; ctrl=tr['ctrl'][k]; po=q[oc:oc+3]; Ro=R.from_quat(q[oc+3:oc+7],scalar_first=True)
    local_p=Ro.inv().apply(q[wc[:3]]-po); local_R=Ro.inv()*R.from_euler('XYZ',q[wc[3:]]); closed=ctrl[facts].copy()
    spec_for(f'held{th:g}',anchor(local_p,local_R,closed),dict(grasp_generator=f'base teacher achieved hold at {th} s transplanted to the initial spoon pose; closed = commanded fingers'))
    print('held',th,'local_p',np.round(local_p,3))
# commanded grasp of the base reference at its closure (frame nearest 2.5 s), transplanted the same way (control: should reproduce the base plan's grasp)
k=round(2.5*fps); q=Qb[k]; po=orig[k,oc:oc+3]; Ro=R.from_quat(orig[k,oc+3:oc+7],scalar_first=True)
spec_for('cmd2.5',anchor(Ro.inv().apply(q[wc[:3]]-po),Ro.inv()*R.from_euler('XYZ',q[wc[3:]]),q[fcols].copy()),dict(grasp_generator='base reference commanded wrist/fingers at 2.5 s re-expressed at the initial spoon pose'))
json.dump(specs,open(out/'all_specs.json','w'),indent=1); print(len(specs),'specs')
