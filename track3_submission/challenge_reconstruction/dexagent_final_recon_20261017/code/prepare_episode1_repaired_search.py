"""Transfer the actual previously optimized grasp to a fixed repaired scene."""
import json
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,OUT,EP
from physics_pickup_clearance import ClearanceExperiment
from physics_pickup_search import dump

def run():
    old=B/'refinement_20261018/episode_000001/pickup_cem_v1/pose_01'
    report=json.loads((old/'result.json').read_text());spec=report['spec'];x=np.array(report['parameters']);exp=ClearanceExperiment(spec)
    root=OUT/'patch_height094';dest=root/'search_input';dest.mkdir(exist_ok=True)
    scene=root/'pickup_4s/scene.xml';m=mujoco.MjModel.from_xml_path(str(scene));q=m.key_qpos[0].copy()
    obj=q[exp.oc:exp.oc+7];R=Rotation.from_quat(obj[3:],scalar_first=True)
    gp=exp.local_p+x[:3];gr=exp.local_R*Rotation.from_rotvec(x[3:6]);q[exp.wcols[:3]]=obj[:3]+R.apply(gp)
    q[exp.wcols[3:]]=(R*gr).as_euler('XYZ');q[exp.fcols]=np.clip(exp.closed+x[34:56],exp.limits[6:,0],exp.limits[6:,1])
    anchor=dest/'anchor.npz';np.savez_compressed(anchor,qpos=q)
    _,timing=exp.commands(x);spec=dict(spec);spec.update(scene=str(scene),reference=str(old/'reference.npz'),anchor_pose=str(anchor),
        open_fingers=np.clip(exp.opened+x[12:34],exp.limits[6:,0],exp.limits[6:,1]).tolist(),
        hover_offset=(np.array(spec.get('hover_offset',[0.,0.,.1]))+x[6:9]).tolist(),horizon_s=4.,
        source=str(B.parent/'final_recon/episode_000001'),manifest=str(EP/'input_manifest.json'))
    spec['times']=dict(spec['times'],contact=timing['contact'],closed=timing['closed'],transport=timing['transport'])
    spec['scale']=np.r_[[.006]*3,[.07]*3,[.012]*3,[.08]*3,[.12]*22,[.12]*22,[.008]*3,[.06]*3,[.07,.05,.07]].tolist()
    spec['provenance']=dict(original_search=str(old),reconstruction_candidate=str(root),unchanged_original_object_targets=True,
        fixed_scene_initial_state=True,no_object_actuation=True,scope='Relearn contact acquisition on corrected source-supported collision/registration; full metric required after prefix search')
    dump(dest/'spec.json',spec);print(dest/'spec.json',flush=True)

if __name__=='__main__':run()
