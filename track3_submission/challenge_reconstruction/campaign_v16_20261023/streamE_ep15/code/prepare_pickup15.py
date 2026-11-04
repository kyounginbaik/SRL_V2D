"""Pickup-search specs (physics_pickup_search_v2) for episode 15's wooden spoon. Anchors: the donor pinch (as shipped), the
teacher's own achieved hold at T_HELD transplanted to the spoon's initial pose (closed fingers = commanded targets at T_HELD),
and that hold shifted along the spoon axis / rotated about it / with the middle finger closed. Timing variants: 'hold' keeps the
grasp (release beyond the horizon: the right hand stays on the placed spoon until the end) and 'rel' releases at 4.8 s.
Left hand parked at its initial pose (donor seed). Immutable object targets and clock.
usage: prepare_pickup15.py OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=['/data1/SRL_V2D/third_party/DexAgent',str(LEG/'code')]
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import dump
D=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/spoon_transfer/episode_000015')
TE=D/'retention_window32_44'; DONOR=D/'collision_v2_donor28_park1'; EP=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_collision_v2/episodes/episode_000015')
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True); T_HELD=2.2
r=json.loads((TE/'result.json').read_text()); scene=r['execution']['scene']; man=json.loads((EP/'input_manifest.json').read_text())
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
def spec_for(name,pose,timing,extra):
    ap=out/f'anchor_{name}.npz'; np.savez_compressed(ap,qpos=pose)
    hold=timing=='hold'; times=dict(start=0.25,hover=0.85,contact=1.15,closed=1.45,transport=1.85,release=(9.0 if hold else 4.8))
    s=dict(episode='episode_000015',scene=scene,reference=seed_ref,original_reference=str(EP/'tracking_reference.npz'),source=man['source'],geometry=str(EP/'metric_geometry.json'),
           manifest=str(EP/'input_manifest.json'),object_name='wooden_spoon',side='right',anchor_pose=str(ap),open_fingers=opened,finger_names=fnames,times=times,
           hover_offset=[0,0,.06],horizon_s=(8.25 if hold else 5.3),score_hold_start_s=1.45,score_track_start_s=1.95,left_hand_held_at_initial_pose=True,
           right_hand_holds_until_end=hold,teacher=str(TE),**extra)
    dump(out/f'{name}_{timing}.json',s); specs.append(str(out/f'{name}_{timing}.json'))
donor=np.load(DONOR/'anchor_pose.npz')['qpos']
for timing in ('hold','rel'):
    spec_for('donor',donor,timing,dict(grasp_generator='donor ep3 pinch anchor as shipped (collision_v2_donor28_park1)'))
    spec_for('held22',anchor(local_p,local_R,closed),timing,dict(grasp_generator=f'teacher achieved hold at {T_HELD} s transplanted to the initial spoon pose; closed = commanded fingers'))
for dz in (-0.04,-0.02,0.02):
    spec_for(f'held22_z{int(dz*100):+d}',anchor(local_p+np.array([0,0,dz]),local_R,closed),'hold',dict(grasp_generator=f'held22 shifted {dz*100:+.0f} cm along the spoon axis (body z; +z toward the bowl)'))
for ang in (40,-40):
    Rz=R.from_rotvec([0,0,np.radians(ang)]); spec_for(f'held22_r{ang:+d}',anchor(Rz.apply(local_p),Rz*local_R,closed),'hold',dict(grasp_generator=f'held22 rotated {ang:+d} deg about the spoon axis (different pinch direction)'))
mid=closed.copy()
for jn in rig.flexion_joints('middle'): lo,hi=rig.finger_range[jn]; mid[fnames.index(jn)]=lo+0.9*(hi-lo)
spec_for('held22_mid',anchor(local_p,local_R,mid),'hold',dict(grasp_generator='held22 with the middle finger flexed to 0.9 of range (third contact)'))
json.dump(specs,open(out/'all_specs.json','w'),indent=1); print(len(specs),'specs'); print('local_p',np.round(local_p,3),'closed',np.round(closed,2))
