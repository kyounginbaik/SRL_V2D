import numpy as np,mujoco
from pathlib import Path
root=Path(__file__).parent.parent/'runs';m=mujoco.MjModel.from_binary_path(str(root/'surface_anchor_fit_v7/native_model.mjb'));d=mujoco.MjData(m)
P=np.load(root/'loop_r1_eval/references.npz')['joint_q'][0];Z=np.load(root/'loop_r1_eval/physics.npz')
for j in range(29,52):print(j,m.joint(j).name.replace('right_sharpa_wave_right_',''),m.jnt_type[j],np.round(m.jnt_range[j],2).tolist(),'q0src',round(float(P[0,m.jnt_qposadr[j]]),2) if m.jnt_type[j]==3 else '','body',m.body(m.jnt_bodyid[j]).name.replace('right_sharpa_wave_right_',''),'axis',np.round(m.jnt_axis[j],1).tolist())
Q0=Z['native_qpos'][0,0].copy();d.qpos[:]=Q0;mujoco.mj_kinematics(m,d)
for b in range(27,53):print(b,m.body(b).name.replace('right_sharpa_wave_right_',''),np.round(d.xpos[b],3).tolist(),'parent',m.body_parentid[b],'geoms',[int(g) for g in range(m.ngeom) if m.geom_bodyid[g]==b])
print('tendon/equality',m.ntendon,m.neq,'exclude',m.nexclude)
print('source finger q at f0 stats',np.round(P[0,36:58],2).tolist());print('self-collide? contype/conaff',set((int(m.geom_contype[g]),int(m.geom_conaffinity[g])) for g in range(27,53)))
