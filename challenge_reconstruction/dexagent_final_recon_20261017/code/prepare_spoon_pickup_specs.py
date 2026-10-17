from pathlib import Path
import json,sys
import numpy as np,mujoco
B=Path(__file__).resolve().parents[1];R=B/'refinement_20261017'
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H

scene=R/'cem_v2/episode_000003/execution/scene_physics.xml'
model=mujoco.MjModel.from_xml_path(str(scene));rig=H.HandRig.build(model,'robot_right_','right')
source=R/'spoon_collision_grasps/episode_000003';report=json.loads((source/'report.json').read_text())
out=B/'refinement_20261018/episode_000003/pickup_inputs';out.mkdir(parents=True,exist_ok=True)
for rank in [0,1]:
    data=mujoco.MjData(model);pose=source/f'pose_{rank:02d}.npz';data.qpos[:]=np.load(pose)['qpos'];mujoco.mj_forward(model,data)
    geometry=next(a['geometry'] for a in report['exported'] if a['rank']==rank);patches=geometry['actual_contacts']['patches'];best=None
    for a in patches['thumb']:
        for b in patches['index']:
            na,nb=np.array(a['normal']),np.array(b['normal']);v=np.array(a['position'])-np.array(b['position']);v/=np.linalg.norm(v)
            margin=min(-.5-na@nb,na@v-.6401844,nb@(-v)-.6401844)
            if best is None or margin>best[0]:best=(margin,na,nb)
    tips={f:data.site_xpos[rig.fingertip_sites[f]].copy() for f in ['thumb','index']}
    normals=dict(thumb=best[1],index=best[2]);H.finger_ik(model,data,rig,{f:tips[f]+.004*normals[f] for f in tips},iters=120)
    spec=dict(scene=str(scene),reference=str(R/'cem_v2/episode_000003/reference.npz'),original_reference=str(B/'episodes/episode_000003/tracking_reference.npz'),
        source=str(B.parents[1]/'challenge_reconstruction/final_recon/episode_000003'),geometry=str(B/'episodes/episode_000003/metric_geometry.json'),
        manifest=str(B/'episodes/episode_000003/input_manifest.json'),object_name='wooden_spoon',side='right',anchor_pose=str(pose),
        open_fingers=[float(data.qpos[c]) for c in rig.finger_qpos.values()],finger_names=list(rig.finger_qpos),
        times=dict(start=.8,hover=1.25,contact=1.8,closed=2.15,transport=2.9,release=14.45),horizon_s=4.2,hover_offset=[0.,0.,.10],
        measured_symmetric_opening_m={f:float((data.site_xpos[rig.fingertip_sites[f]]-tips[f])@normals[f]) for f in tips})
    spec['scale']=np.r_[[.01]*3,[.10]*3,[.02]*3,[.12]*3,[.10]*22,[.10]*22,[.012]*3,[.08]*3,[.10,.07,.10]].tolist()
    (out/f'pose_{rank:02d}.json').write_text(json.dumps(spec,indent=2)+'\n');print(out/f'pose_{rank:02d}.json')
