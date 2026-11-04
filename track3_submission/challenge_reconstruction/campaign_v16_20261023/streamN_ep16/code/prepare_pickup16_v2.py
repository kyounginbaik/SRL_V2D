"""v2: same anchors, plus release-timing variants (release at T_REL s instead of holding: the placed spoon is let go), for the
re-targeted spoon world where a released spoon can rest near the reference (rest probe 1.19 cm / 7.4 deg).
Pickup-search specs (physics_pickup_search_v2) for episode 16's wooden spoon in the Stream-N recon-rack world, after
streamE_ep15/code/prepare_pickup15.py. Anchors: the donor pinch of the pre-scan world (collision_v2_donor28_park1); the
teacher's own achieved hold at T_HELD re-expressed at the spoon's initial pose (closed fingers = commanded targets at T_HELD);
that hold shifted along body x (spoon long axis), body z (ep15's winning family), rotated about the long axis. All 'hold'
timing (release beyond the horizon: the right hand stays on the placed spoon until the end). Left hand parked at its initial pose.
Immutable object targets and clock. usage: prepare_pickup16.py WORLD TEACHER_DIR OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=['/data1/SRL_V2D/third_party/DexAgent',str(LEG/'code')]
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import dump
DONOR=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/spoon_transfer/episode_000016/collision_v2_donor28_park1')
W,TE,out=(Path(x) for x in sys.argv[1:4]); out.mkdir(parents=True,exist_ok=True); T_HELD=2.2
EP=W/'episodes/episode_000016'; scene=str(W/'scene/controller_scene.xml'); man=json.loads((EP/'input_manifest.json').read_text())
r=json.loads((TE/'result.json').read_text())
dspec=json.loads((DONOR/'spec.json').read_text()); seed_ref=dspec['reference']; opened=dspec['open_fingers']
m=mujoco.MjModel.from_xml_path(scene); rig=H.HandRig.build(m,'robot_right_','right'); wc=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')])
fnames=list(rig.finger_qpos); fcols=np.array([rig.finger_qpos[k] for k in fnames]); facts=np.array([rig.finger_act[k] for k in fnames])
orig=TR.load_reference(EP/'tracking_reference.npz',m)['qpos']; oc=int(m.joint('wooden_spoon_free').qposadr[0])
p0=orig[0,oc:oc+3]; R0=R.from_quat(orig[0,oc+3:oc+7],scalar_first=True)
tr=np.load(r['execution']['trajectory']); k=round(T_HELD*20); q=tr['qpos'][k]; ctrl=tr['ctrl'][k]
po=q[oc:oc+3]; Ro=R.from_quat(q[oc+3:oc+7],scalar_first=True)
local_p=Ro.inv().apply(q[wc[:3]]-po); local_R=Ro.inv()*R.from_euler('XYZ',q[wc[3:]]); closed=ctrl[facts].copy()
def anchor(lp,lR,fingers):
    a=orig[0].copy(); a[wc[:3]]=p0+R0.apply(lp); a[wc[3:]]=continuous_euler_xyz(R.from_matrix((R0*lR).as_matrix()[None]),orig[0,wc[3:]])[0]; a[fcols]=fingers; return a
specs=[]
def spec_for(name,pose,extra,rel=None):
    ap=out/f'anchor_{name}.npz'; np.savez_compressed(ap,qpos=pose)
    times=dict(start=0.2,hover=0.8,contact=1.1,closed=1.4,transport=1.8,release=(9.5 if rel is None else rel))
    s=dict(episode='episode_000016',scene=scene,reference=seed_ref,original_reference=str(EP/'tracking_reference.npz'),source=man['source'],geometry=str(EP/'metric_geometry.json'),
           manifest=str(EP/'input_manifest.json'),object_name='wooden_spoon',side='right',anchor_pose=str(ap),open_fingers=opened,finger_names=fnames,times=times,
           hover_offset=[0,0,.06],horizon_s=8.6,score_hold_start_s=1.4,score_track_start_s=1.9,left_hand_held_at_initial_pose=True,
           right_hand_holds_until_end=rel is None,teacher=str(TE),**extra)
    tag='hold' if rel is None else f'rel{rel}'; dump(out/f'{name}_{tag}.json',s); specs.append(str(out/f'{name}_{tag}.json'))
hp=anchor(local_p,local_R,closed)
for dz in (-0.04,-0.05,-0.03):
    v=np.array([0,0,dz]); spec_for(f'held22_z{int(round(dz*100)):+d}',anchor(local_p+v,local_R,closed),dict(grasp_generator=f'held22 shifted {dz*100:+.0f} cm along body z'))
for rel in (6.5,7.0):
    spec_for('held22_z-4',anchor(local_p+np.array([0,0,-0.04]),local_R,closed),dict(grasp_generator='held22 shifted -4 cm along body z; release variant'),rel=rel)
spec_for('held22',hp,dict(grasp_generator=f'teacher achieved hold at {T_HELD} s transplanted to the initial spoon pose'))
json.dump(specs,open(out/'all_specs.json','w'),indent=1); print(len(specs),'specs'); print('local_p',np.round(local_p,3),'closed',np.round(closed,2))
