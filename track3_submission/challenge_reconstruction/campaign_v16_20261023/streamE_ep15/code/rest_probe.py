"""Physics floor probes for ep15: (a) settle the spoon from the reference pose at frame K with no hands moving (hands held at
frame 0), report the settled pose error vs the reference at K; (b) print fingertip positions in the spoon frame during the hold.
usage: rest_probe.py TEACHER_DIR EPISODE_DIR"""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR, hand as H
te,epd=Path(sys.argv[1]),Path(sys.argv[2]); r=json.loads((te/'result.json').read_text()); scene=r['execution']['scene']
m=mujoco.MjModel.from_xml_path(scene); ref=TR.load_reference(epd/'tracking_reference.npz',m)['qpos']; fps=20.
oc=int(m.joint('wooden_spoon_free').qposadr[0]); rc=int(m.joint('plastic_dish_rack_free').qposadr[0]); ob=m.body('wooden_spoon').id
print('spoon mass',m.body_mass[ob],'rack mass',m.body_mass[m.body('plastic_dish_rack').id])
qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
def settle(k,seconds=2.0,rack_from_sim=None):
    d=mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m,d,0); d.qpos[:]=ref[0]; d.qpos[oc:oc+7]=ref[k,oc:oc+7]
    if rack_from_sim is not None: d.qpos[rc:rc+7]=rack_from_sim
    # park the hands far away so they cannot touch anything
    for s in ('left','right'):
        for ax,v in (('x',0.6*(1 if s=='right' else -1)),('y',-0.6),('z',0.5)): d.qpos[int(m.joint(f'robot_{s}_{s}_pos_{ax}').qposadr[0])]=v
    d.qvel[:]=0; mujoco.mj_forward(m,d); d.ctrl[:]=d.qpos[qa]
    for _ in range(int(seconds/m.opt.timestep)): mujoco.mj_step(m,d)
    p=d.qpos[oc:oc+3]; q=d.qpos[oc+3:oc+7]; e=np.linalg.norm(p-ref[k,oc:oc+3])*100; er=np.degrees((R.from_quat(q,scalar_first=True)*R.from_quat(ref[k,oc+3:oc+7],scalar_first=True).inv()).magnitude())
    dz=(p[2]-ref[k,oc+2])*100
    return e,er,dz,p.copy()
for k in (0,165):
    e,er,dz,p=settle(k); print(f'settle from reference frame {k} ({k/fps:.2f}s): err {e:.2f} cm / {er:.1f} deg, dz {dz:+.2f} cm, pos {np.round(p,3)}')
# final rest with the rack where the teacher left it
tr=np.load(r['execution']['trajectory']); i=list(tr['object_names']).index('plastic_dish_rack'); rq=np.r_[tr['object_positions'][-1,i],tr['object_quaternions_wxyz'][-1,i]]
e,er,dz,p=settle(165,rack_from_sim=rq); print(f'settle from reference final with rack at teacher final pose: err {e:.2f} cm / {er:.1f} deg, dz {dz:+.2f} cm')
# fingertip positions in the spoon frame during the hold
d=mujoco.MjData(m); rig=H.HandRig.build(m,'robot_right_','right'); Q=tr['qpos']
for k in range(28,66,4):
    d.qpos[:]=Q[k]; mujoco.mj_forward(m,d); Rm=d.xmat[ob].reshape(3,3); op=d.xpos[ob]
    tips={f:np.round((Rm.T@(d.site_xpos[rig.fingertip_sites[f]]-op))*100,1).tolist() for f in ('thumb','index','middle')}
    palm=np.round((Rm.T@(d.xpos[m.body('robot_right_right_base_yaw').id]-op))*100,1).tolist()
    print(f'{k/fps:.2f}s tips (spoon frame, cm) {tips} palm {palm}')
# spoon mesh extent along its principal axis for reference
v=m.mesh_vert[m.mesh_vertadr[m.geom_dataid[[g for g in range(m.ngeom) if m.geom_bodyid[g]==ob][0]]]:][:m.mesh_vertnum[m.geom_dataid[[g for g in range(m.ngeom) if m.geom_bodyid[g]==ob][0]]]]
print('spoon mesh extent (body frame) min',np.round(v.min(0)*100,1),'max',np.round(v.max(0)*100,1),'com',np.round(m.body_ipos[ob]*100,1))
