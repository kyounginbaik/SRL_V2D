"""Read-only object collision-geometry clearance and velocity during release."""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from learned_feedback_policy import sha

def audit(run,object_name,start_s,end_s):
 run=Path(run);result=json.loads((run/'result.json').read_text());scene=Path(result['execution']['scene']);trajectory=Path(result['execution']['trajectory']);m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);ob=m.body(object_name).id;table=m.geom('g_table').id;of=m.joint(object_name+'_free');dof=int(of.dofadr[0]);geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==ob and m.geom_contype[g] and m.geom_dataid[g]>=0];rows=[]
 with np.load(trajectory) as z:
  mask=(z['substep_time_s']>=start_s)&(z['substep_time_s']<=end_s)
  for t,q,v in zip(z['substep_time_s'][mask],z['substep_qpos'][mask],z['substep_qvel'][mask]):
   d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d);vertices=[]
   for g in geoms:
    mid=m.geom_dataid[g];p=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]];vertices.extend(p@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
   vertices=np.asarray(vertices);local=(vertices-d.geom_xpos[table])@d.geom_xmat[table].reshape(3,3);kind=m.geom_type[table]
   if kind==mujoco.mjtGeom.mjGEOM_CYLINDER:inside=np.linalg.norm(local[:,:2],axis=1)<=m.geom_size[table,0];top=m.geom_size[table,1]
   elif kind==mujoco.mjtGeom.mjGEOM_BOX:inside=np.all(abs(local[:,:2])<=m.geom_size[table,:2],axis=1);top=m.geom_size[table,2]
   else:raise ValueError('Unsupported table geom')
   rows.append(dict(time_s=float(t),collision_bottom_clearance_global_m=float(local[:,2].min()-top),collision_bottom_clearance_over_table_m=float(local[inside,2].min()-top) if np.any(inside) else None,world_origin_z_m=float(d.xpos[ob,2]),world_linear_velocity_z_m_s=float(v[dof+2]),world_angular_speed_rad_s=float(np.linalg.norm(v[dof+3:dof+6]))))
 out=run/'audit';out.mkdir(exist_ok=True);report=dict(scene=str(scene),scene_sha256=sha(scene),trajectory=str(trajectory),trajectory_sha256=sha(trajectory),object=object_name,interval_s=[start_s,end_s],interpretation='Collision-mesh lowest vertex relative to finite table top; not object-origin height. No forces inferred from these geometry snapshots.',rows=rows);path=out/f'{object_name}_release_clearance_{start_s:g}_{end_s:g}.json';path.write_text(json.dumps(report,indent=2)+'\n');print(path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--object-name',required=True);p.add_argument('--start-s',type=float,required=True);p.add_argument('--end-s',type=float,required=True);audit(**vars(p.parse_args()))
