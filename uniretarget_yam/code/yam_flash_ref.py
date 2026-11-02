"""Write a motion_v1 reference for flash_chord: robot joints from a refined tape or the fixed human reference, object goal copied from the source parquet."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np,mujoco,pyarrow as pa,pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).parent))
import yam_loop as Y,yam_lower_only as L
p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--side',required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--tape',type=Path);p.add_argument('--fixed',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--onset-floor',type=float,default=2.);a=p.parse_args();Y.ONSET_FLOOR=a.onset_floor
t=Y.YamTask(a.episode,a.side,0);t.set_stage('real');m=t.m;H=t.frames-1
if a.fixed:
    b=L.Lower(a.episode,a.side,0);C=b.tape(np.zeros(b.size));t=b.t;m=t.m;C=np.vstack([C,C[-1:]]);t.last_Q=np.vstack([t.last_Q,t.last_Q[-1:]])
else:
    s=json.loads(a.tape.read_text());s=s[0] if isinstance(s,list) else s;C,_=t.compile(s['sample'],H)
adr=m.jnt_qposadr[m.actuator_trnid[:,0]];q=np.tile(t.base_q[adr],(len(C),1))
names=[]
for a_ in range(m.nu):
    n=m.joint(int(m.actuator_trnid[a_,0])).name;side=n.split('_')[0];rest=n.split('_arm_',1)[1];names.append(f"{'L' if side=='left' else 'R'}_arm_j{rest[-1]}" if rest.startswith('joint') else rest)
    j=int(m.actuator_trnid[a_,0]);lo,hi=m.jnt_range[j];q[:,a_]=np.clip(C[:,a_],lo,hi)
# Arm targets in C carry a velocity feed-forward; the reference is the position, and flash_chord derives velocity itself.
q[:,t.act_arm]=t.last_Q
src=pq.read_table(a.source).to_pylist()[0];F=len(q);a.output.mkdir(parents=True,exist_ok=True)
# The schema needs named frames with full rank: the palms, from this model's kinematics.
d=mujoco.MjData(m);pb=[m.body(f'{x}_arm_{x}_hand_C_MC').id for x in ('left','right')];E=np.zeros((F,2,7))
for f in range(F):
    d.qpos[:]=t.base_q;d.qpos[adr]=q[f];mujoco.mj_kinematics(m,d);E[f,:,:3]=d.xpos[pb];E[f,:,3:]=d.xquat[pb]
row=dict(schema_version='motion_v1',motion_kind='single_robot',source_dataset='',raw_motion_file=str(a.tape or 'fixed_human_reference'),fps=float(Y.FPS),coord_frame='world',robot_joint_names=names,robot_root_position=np.zeros((F,3)).tolist(),robot_root_wxyz=np.tile([1.,0,0,0],(F,1)).tolist(),robot_joint_positions=q.tolist(),ee_link_names=['left_hand_C_MC','right_hand_C_MC'],ee_pose_w=E.tolist(),hand_sides=['left','right'],object_name=src['object_name'],object_body_names=src['object_body_names'],object_mesh_paths=src['object_mesh_paths'],object_urdf_paths=src['object_urdf_paths'],object_mesh_radius=src['object_mesh_radius'],object_articulation=np.zeros((F,1)).tolist(),object_body_position=src['object_body_position'][:F],object_body_wxyz=src['object_body_wxyz'][:F],sequence_id=a.output.name,robot_name='yam_sharpa')
pq.write_table(pa.Table.from_pylist([row]),a.output/'reference.parquet')
for n in ('scene_overrides.json','reference_support.usda'):
    if (a.source.parent/n).exists():shutil.copy(a.source.parent/n,a.output/n)
print(json.dumps(dict(frames=F,joints=len(names),arm_right_range=np.ptp(q[:,[names.index(f'R_arm_j{i}') for i in range(1,7)]],axis=0).round(2).tolist(),first=names[:3])))
