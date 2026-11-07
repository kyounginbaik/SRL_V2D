"""Do As I Do rollout at their 30 fps, resampled to our clock, with overlap and actuator check. Writes daid_view.npz with world vertices' transforms handled by render_d.py."""
import numpy as np, json, re, mujoco, wmetric as W
from pathlib import Path
O=Path('/lustre/exaflops/retarget_bench/repos/do-as-i-do/retargeting/outputs/sharpa/right/write_wb/0'); c=(O/'config.yaml').read_text(); g=lambda k:float(re.search(rf'^{k}: *([-\d.e]+)',c,re.M).group(1)); wu=int(g('warmup_steps')); rs=int(round(g('ref_dt')/g('sim_dt')))
m=mujoco.MjModel.from_xml_path(str(O/'scene.xml')); d=mujoco.MjData(m); r=np.load(O/'trajectory_mjwp.npz'); q=r['qpos']
if q.ndim==3:
    if 'sim_step' in r.files and len(r['sim_step'])==q.shape[0]: q=q[np.argsort(r['sim_step'].ravel())]
    q=q.reshape(-1,m.nq)
print('nq',m.nq,'nu',m.nu,'warmup',wu,'stride',rs,'frames',len(q),'finger kp',m.actuator_gainprm[6:10,0],'armature',m.dof_armature[6:8],'friction pair',m.pair_friction[0][:2])
a=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'right_object_joint')]; print('pen z over warmup',np.round(q[[0,wu//2,wu-1,wu],a+2],3).tolist())
post=q[wu:][::rs]; PG={i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('right_object_') and 'visual' not in mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i)}; BG=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'board'); D=[];N=[]
for f in range(0,len(post),3):
    d.qpos[:]=post[f]; mujoco.mj_forward(m,d); x=[d.contact[i].dist for i in range(d.ncon) if (d.contact[i].geom1 in PG)!=(d.contact[i].geom2 in PG) and BG not in (d.contact[i].geom1,d.contact[i].geom2)]; D.append(min(x) if x else np.nan); N.append(len(x))
print('overlap cm p5-worst',round(float(-np.nanpercentile(D,5))*100,2),'median worst',round(float(-np.nanmedian(D))*100,2),'contacts',np.median(N))
np.save('daid_post.npy',post)
