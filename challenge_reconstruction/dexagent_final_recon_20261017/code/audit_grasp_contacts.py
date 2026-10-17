"""Geometric contact audit of recorded physics and original kinematic poses."""
import argparse,json,sys
from pathlib import Path
import mujoco,numpy as np

B=Path(__file__).resolve().parents[1]

def inspect(episode):
    episode=f'episode_{episode:06d}';root=B/'runs/pilot_v2'/episode/'seed_44'
    model=mujoco.MjModel.from_xml_path(str(root/'baseline/scene_physics.xml'))
    ref=dict(np.load(B/'episodes'/episode/'tracking_reference.npz'))
    manifest=json.loads((B/'episodes'/episode/'input_manifest.json').read_text())
    objects=[o['id'] for o in manifest['objects']]
    object_geom={o:[g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name==o and model.geom_group[g]==3] for o in objects}
    if not all(object_geom.values()):
        object_geom={o:[g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name==o and (model.geom_contype[g] or model.geom_conaffinity[g])] for o in objects}
    handgeom={s:[g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith(f'robot_{s}_') and model.geom_group[g]==3] for s in ['left','right']}
    rows=[]
    for mode,path in [('reference',None),('baseline',root/'baseline/trajectory.npz'),('refined',root/'refined/trajectory.npz')]:
        q=ref['qpos'] if path is None else np.load(path)['qpos']
        data=mujoco.MjData(model)
        for t in range(len(q)):
            data.qpos[:]=q[t];mujoco.mj_forward(model,data)
            pairs={o:{s:[] for s in ['left','right']} for o in objects}
            for i in range(data.ncon):
                c=data.contact[i]
                n1=model.body(model.geom_bodyid[c.geom1]).name;n2=model.body(model.geom_bodyid[c.geom2]).name
                for o in objects:
                    hand=n2 if n1==o else n1 if n2==o else ''
                    if hand.startswith('robot_'):
                        side='left' if hand.startswith('robot_left_') else 'right'
                        pairs[o][side].append(dict(body=hand,penetration_m=max(0.,-float(c.dist))))
            row=dict(mode=mode,time_s=t/float(ref['fps']),contacts=pairs)
            if t%10==0 and t<=80:
                row['closest_surface_distance_m']={}
                for o in objects:
                    row['closest_surface_distance_m'][o]={}
                    for s in ['left','right']:
                        best=(float('inf'),'')
                        for hg in handgeom[s]:
                            for og in object_geom[o]:
                                dist=mujoco.mj_geomDistance(model,data,hg,og,.5,None)
                                if dist<best[0]:best=(float(dist),model.body(model.geom_bodyid[hg]).name)
                        row['closest_surface_distance_m'][o][s]=dict(distance=best[0],body=best[1])
            rows.append(row)
    out=B/'refinement_20261017/visual_audit';out.mkdir(exist_ok=True)
    path=out/f'{episode}_contacts.json';path.write_text(json.dumps(rows,indent=2))
    for mode in ['reference','baseline','refined']:
        for o in objects:
            for s in ['left','right']:
                selected=[r for r in rows if r['mode']==mode and r['contacts'][o][s]]
                initial=[r for r in selected if r['time_s']<=3]
                print(mode,o,s,'contact_frames',len(selected),'first3s',len(initial),'times',[r['time_s'] for r in initial])
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);a=p.parse_args();print(inspect(a.episode))
