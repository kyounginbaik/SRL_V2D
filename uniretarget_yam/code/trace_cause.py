"""Trace back from the tolerance break to the last healthy grip (heuristic, unconfirmed)."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
from flash_chord.retargeting.native_contact_analysis import object_contact_snapshot
def loop_frontier(ep,er,total):
    # The loop's own frontier: object lost, or the official mean can no longer be met.
    n=np.arange(1,len(ep)+1);limit=.7*n+.3*total
    over=np.flatnonzero((ep>.15)|(er>np.radians(150.))|(np.cumsum(ep)>.03*limit)|(np.cumsum(er)>np.radians(30.)*limit))
    return int(over[0]) if len(over) else -1
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--world',type=int,default=0);p.add_argument('--star',type=Path);p.add_argument('--output',type=Path);p.add_argument('--object',type=int,default=0);p.add_argument('--side',default='right');a=p.parse_args()
m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m);z=np.load(a.run/'physics.npz');ref=np.load(a.run/'references.npz')
cap=int(z['native_capture_world_ids'][a.world]);Q=z['native_qpos'][:,a.world];pose=z['object_pose'][:,cap,a.object];grp=int(z['world_reference_group'][cap]);free=[j for j in range(m.njnt) if m.jnt_type[j]==0];obj=int(m.jnt_bodyid[min(free,key=lambda j:float(np.linalg.norm(z['native_qpos'][0,0][m.jnt_qposadr[j]:m.jnt_qposadr[j]+3]-z['object_pose'][0,0,a.object,:3])))])
init=pose[0];al=R.from_quat(init[3:7])*R.from_quat(init[10:14]).inv();off=init[:3]-al.apply(init[7:10]);gp=al.apply(pose[:,7:10])+off;gr=al*R.from_quat(pose[:,10:14])
ep=np.linalg.norm(pose[:,:3]-gp,axis=1);er=(R.from_quat(pose[:,3:7]).inv()*gr).magnitude();brk=loop_frontier(ep,er,len(ep)+1)
mg=float(m.body_mass[obj]*np.linalg.norm(m.opt.gravity));oid=int(z['native_object_body_ids'][a.world,0]);og=np.flatnonzero(m.geom_bodyid==obj)
right=[g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(a.side+'_') and m.geom_contype[g]+m.geom_conaffinity[g]>0]
rows=[];end=brk if brk>=0 else len(Q)-1;off=z['native_contact_offsets'];worlds=z['native_contact_worldid'];geoms=z['native_contact_geom'];forces=z['native_contact_newton_force'];body_of=m.geom_bodyid;hand_bodies={int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.body(m.jnt_bodyid[j]).name.startswith(a.side+'_')}
for f in range(end+1):
    # Hand force on THIS object, read from the solver's contacts of this world and frame.
    a0,b0=off[f],off[f+1];sel=np.arange(a0,b0)[worlds[a0:b0]==cap];force=0.
    for i in sel:
        x,y=body_of[geoms[i]]
        if (x==obj and y in hand_bodies) or (y==obj and x in hand_bodies):force+=float(np.linalg.norm(forces[i,:3]))
    d.qpos[:]=Q[f];mujoco.mj_kinematics(m,d);gap=min(mujoco.mj_geomDistance(m,d,h,o,.1,None) for h in right for o in og);rows.append((float(force),float(gap)))
force=np.array([r[0] for r in rows]);gap=np.array([r[1] for r in rows]);healthy=np.flatnonzero(force>=mg)
if len(healthy):cause=min(int(healthy[-1])+1,end);why='last frame with hand force >= object weight'
else:
    near=np.flatnonzero(gap<.02);cause=int(near[0]) if len(near) else end;why='grip never reached object weight; cause = grasp formation (hand first within 2 cm)'
out=dict(world=a.world,group=grp,break_frame=brk,cause_frame=cause,why=why,fps=float(z['fps']),peak_hand_force_over_mg=float(force.max()/mg),status='heuristic, not counterfactually confirmed')
print(json.dumps(out));print('frame force/mg gap_mm wristErr_cm fingErrMax_rad');rq=ref['joint_q'][grp]
for f in range(max(0,cause-4),end+1,2):print(f,round(force[f]/mg,2),round(1e3*gap[f],1),round(100*float(np.linalg.norm(Q[f,29:32]-rq[f,29:32])),2),round(float(np.abs(Q[f,36:58]-rq[f,36:58]).max()),3))
if a.star:
    st=np.load(a.star)['qpos'];k=min(end,33);print('frame',k,'vs fit: wrist cm',round(100*float(np.linalg.norm(Q[k,29:32]-st[29:32])),2),'ball deg',round(float(np.degrees((R.from_quat(Q[k,32:36][[1,2,3,0]]).inv()*R.from_quat(st[32:36][[1,2,3,0]])).magnitude())),1),'finger max rad',round(float(np.abs(Q[k,36:58]-st[36:58]).max()),3),'object mm',round(1e3*float(np.linalg.norm(Q[k,58:61]-st[58:61])),1))
    d.qpos[:]=st;mujoco.mj_kinematics(m,d);print('fit gap mm',round(1e3*min(mujoco.mj_geomDistance(m,d,h,o,.1,None) for h in right for o in og),2))
if a.output:a.output.write_text(json.dumps(out,indent=1)+'\n')
