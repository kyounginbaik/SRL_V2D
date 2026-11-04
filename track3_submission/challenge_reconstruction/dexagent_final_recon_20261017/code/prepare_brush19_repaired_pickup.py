"""Prepare robot-only pickup tests on a source-reviewed bristle candidate."""
from pathlib import Path
import json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from physics_pickup_search_v2 import PickupExperiment,dump
from wrist_reference import continuous_euler_xyz
B=Path(__file__).resolve().parents[1]

def run():
    root=B/'refinement_20261018/expanded_batch';repair=root/'brush19_support/alpha_1.00'
    evidence=json.loads((repair/'report.json').read_text())
    if evidence['brush_initial_tilt_at_point6_s_degrees']>1:raise RuntimeError('Candidate did not fix pre-contact support')
    outputs=[]
    for pose in range(3):
        old=root/f'handle_geometry/episode_000019/mini_sweeper_right/spec_{pose:02d}.json'
        spec=json.loads(old.read_text());spec['scene']=str(repair/'execution/scene_physics.xml')
        spec['reconstruction_candidate']=dict(report=str(repair/'report.json'),source_review=str(repair/'source_evidence.json'),
            status='Provisional source-supported lower bristle envelope; original pose and metric anchor unchanged')
        out=root/f'brush19_repaired_pickup_v2/pose_{pose:02d}';out.mkdir(parents=True,exist_ok=True)
        # Old static seeds were planned around the OLD fallen brush at .5 s.
        # Preserve their object-relative grasp, but relocate them to the actual
        # upright rest pose of this versioned candidate. Never reuse the old
        # absolute wrist location for a differently settled object.
        m=mujoco.MjModel.from_xml_path(spec['scene']);oc=int(m.joint('mini_sweeper_free').qposadr[0])
        wc=np.array([int(m.joint('robot_right_right_'+key).qposadr[0]) for key in ['pos_x','pos_y','pos_z','rot_x','rot_y','rot_z']])
        q=np.load(spec['anchor_pose'])['qpos'].copy()
        z=np.load(repair/'execution/trajectory.npz');rest=z['qpos'][round(.5*float(z['fps']))]
        old_R=Rotation.from_quat(q[oc+3:oc+7],scalar_first=True);new_R=Rotation.from_quat(rest[oc+3:oc+7],scalar_first=True)
        delta=new_R*old_R.inv();q[wc[:3]]=rest[oc:oc+3]+delta.apply(q[wc[:3]]-q[oc:oc+3])
        q[wc[3:]]=continuous_euler_xyz(Rotation.from_matrix((delta*Rotation.from_euler('XYZ',q[wc[3:]])).as_matrix()[None]),q[wc[3:]])[0]
        q[oc:oc+7]=rest[oc:oc+7];np.savez_compressed(out/'anchor_pose.npz',qpos=q)
        spec['anchor_pose']=str(out/'anchor_pose.npz');spec['anchor_rest_pose_source']=str(repair/'execution/trajectory.npz')
        dump(out/'spec.json',spec);exp=PickupExperiment(spec);zero=exp.evaluate([0.]*exp.dim)
        dump(out/'zero_proposal.json',zero);outputs.append(dict(pose=pose,spec=str(out/'spec.json'),zero=zero))
    dump(root/'brush19_repaired_pickup_v2/cases.json',outputs);print(json.dumps(outputs),flush=True)

if __name__=='__main__':run()
