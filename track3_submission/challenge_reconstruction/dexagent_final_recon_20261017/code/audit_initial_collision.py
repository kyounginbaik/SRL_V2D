"""Read-only visual/collision audit of episode1's initial pot/lid penetration."""
from pathlib import Path
import json
import mujoco,numpy as np,trimesh
from scipy.spatial import ConvexHull

B=Path(__file__).resolve().parents[1]
source=B.parent/'final_recon/episode_000001/assets'
scene=B/'runs/pilot_v2/episode_000001/seed_44/baseline/scene_physics.xml'
m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
objects={}
for name in ('white_pot','white_pot_lid'):
    body=m.body(name).id
    visual=trimesh.load(source/f'{name}_visual.obj',force='mesh',process=False)
    parts=[trimesh.load(p,force='mesh',process=False) for p in sorted((source/f'{name}_convex').glob('*.obj'))]
    objects[name]=dict(visual=visual,parts=parts,R=d.xmat[body].reshape(3,3),p=d.xpos[body].copy())

contacts=[]
for ci in np.argsort([c.dist for c in d.contact]):
    c=d.contact[ci]; names=[m.geom(g).name for g in (c.geom1,c.geom2)]
    if not (names[0].startswith('white_pot_collision_') and names[1].startswith('white_pot_lid_collision_')):continue
    normal=c.frame[:3];wpts=np.array([c.pos,c.pos+.5*c.dist*normal,c.pos-.5*c.dist*normal])
    row=dict(geom_names=names,penetration_m=float(-c.dist),world_contact_point=c.pos.tolist(),world_normal=normal.tolist(),visual_surface_distances={})
    for name,obj in objects.items():
        local=(wpts-obj['p'])@obj['R']
        closest,dist,tri=trimesh.proximity.closest_point_naive(obj['visual'],local)
        row['visual_surface_distances'][name]=dict(local_points=local.tolist(),unsigned_distance_m=dist.tolist(),closest_points=closest.tolist())
    contacts.append(row)
    if len(contacts)>=10:break

geometry={}
for name,obj in objects.items():
    allv=np.vstack([p.vertices for p in obj['parts']]);geometry[name]=dict(collision_parts=len(obj['parts']),
        visual_bounds=obj['visual'].bounds.tolist(),collision_bounds=[allv.min(0).tolist(),allv.max(0).tolist()],
        visual_watertight=bool(obj['visual'].is_watertight),visual_volume_m3=float(obj['visual'].volume),
        sum_convex_part_volumes_m3=sum(p.volume for p in obj['parts']))
    # Sample points along pot opening axis: local X is the pot height direction.
    if name=='white_pot':
        probes=np.array([[x,0.,-.01] for x in np.linspace(-.04,.08,13)])
        inside=np.zeros(len(probes),bool)
        for p in obj['parts']:
            hull=ConvexHull(p.vertices)
            inside|=np.all(probes@hull.equations[:,:3].T+hull.equations[:,3]<=1e-8,axis=1)
        geometry[name]['cavity_axis_probe_points']=probes.tolist();geometry[name]['inside_collision_union']=inside.tolist()

result=dict(scene=str(scene),initial_contacts_total=d.ncon,geometry=geometry,deepest_pot_lid_contacts=contacts)
out=B/'refinement_20261017/visual_audit/episode_000001_initial_collision.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
