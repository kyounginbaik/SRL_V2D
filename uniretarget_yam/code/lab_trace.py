"""Trace one placed grasp through squeeze and lift; plus a lab sanity control."""
import sys,json,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from grasp_lab import Lab,FPS,SUBSTEPS
root=Path(__file__).parent.parent/'runs'
lab=Lab(root/'surface_anchor_fit_v6/native_model.mjb',root/'cpu_baseline_jacobians_v1/physics.npz',19)
m,d=lab.m,lab.d;print('onset',lab.onset,'timestep',m.opt.timestep)
slide=[int(m.jnt_qposadr[j]) for j in lab.slides];right={int(m.jnt_bodyid[j]) for j in lab.joints}
def contacts():
    out={}
    for c in d.contact[:d.ncon]:
        a,b=m.geom_bodyid[c.geom]
        if lab.obj in (a,b):
            o=int(b if a==lab.obj else a);f=np.zeros(6);mujoco.mj_contactForce(m,d,list(d.contact[:d.ncon]).index(c) if False else 0,f) if False else None
            out[m.body(o).name.replace('right_sharpa_wave_right_','R_')]=out.get(m.body(o).name.replace('right_sharpa_wave_right_','R_'),0)+1
    return out
for name in ['candidate_015','candidate_009','candidate_004']:
    z=np.load(root/f'surface_anchor_fit_v6/{name}.npz');star=z['qpos'];P,N,G=z['patch_positions'],z['patch_normals'],z['hand_geom_ids']
    print('==',name,'geoms',[m.body(m.geom_bodyid[g]).name[23:] for g in G],'normals',np.round(N,2).tolist(),'pairwise n dots',np.round(N@N.T,2)[np.triu_indices(3,1)].tolist())
    mujoco.mj_resetData(m,d);d.qpos[:]=star;mujoco.mj_forward(m,d);print(' t0 contacts',contacts(),'ncon',d.ncon)
    # Static hold: targets = fitted pose, no lift; does the placed hand even stay put?
    for j,a in lab.pos.items():d.ctrl[a]=star[int(m.jnt_qposadr[j])]
    t=R.from_quat(star[int(m.jnt_qposadr[lab.ball]):int(m.jnt_qposadr[lab.ball])+4][[1,2,3,0]]).as_rotvec();lt=R.from_quat(star[int(m.jnt_qposadr[lab.lball]):int(m.jnt_qposadr[lab.lball])+4][[1,2,3,0]]).as_rotvec()
    for f in range(12):
        for _ in range(SUBSTEPS):
            d.qfrc_applied[:]=0;lab.ball_torque(lab.ball,t);lab.ball_torque(lab.lball,lt);mujoco.mj_step(m,d)
        if f in (0,3,7,11):print(f' hold f{f} wrist drift cm',round(100*np.linalg.norm(d.qpos[slide]-star[slide]),2),'max finger drift rad',round(float(np.abs(d.qpos[36:58]-star[36:58]).max()),3),'obj moved mm',round(1e3*float(np.linalg.norm(d.qpos[lab.qo:lab.qo+3]-star[lab.qo:lab.qo+3])),2),contacts())
# Control: a kinematically welded grasp must carry, proving the goal path and metric are sane.
z=np.load(root/'surface_anchor_fit_v6/candidate_015.npz');star=z['qpos'];wp=[]
print('control: goal displacement frames onset..onset+85 (cm)',np.round(100*np.linalg.norm(lab.gp[lab.onset:lab.onset+85:17]-lab.gp[lab.onset],axis=1),1).tolist())
print('finger kp/kv',m.actuator_gainprm[8,0],m.actuator_gainprm[9,0],'force range',m.actuator_forcerange[8],'geom solref',m.geom_solref[35],'solimp',m.geom_solimp[35][:3],'condim',m.geom_condim[35],m.geom_condim[53],'margin',m.geom_margin[35],'gap',m.geom_gap[35],'fric hand',m.geom_friction[35],'obj',m.geom_friction[53])
print('contact pair types',mujoco.mjtGeom(m.geom_type[35]).name,mujoco.mjtGeom(m.geom_type[53]).name)
