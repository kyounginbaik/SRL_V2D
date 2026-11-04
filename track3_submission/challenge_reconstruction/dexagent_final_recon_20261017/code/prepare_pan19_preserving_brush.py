"""Refine the left dustpan grasp while preserving the successful right commands."""
from pathlib import Path
import json
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from physics_pickup_search_v2 import B,dump
from wrist_reference import continuous_euler_xyz

def main():
    root=B/'refinement_20261018/expanded_batch';base=root/'brush19_repaired_pickup_v2/pose_02/search';result=json.loads((base/'result.json').read_text())
    brush=result['metrics']['official_style_reference_proxy']['objects']['mini_sweeper']
    if not brush['success_per_world'][0]:raise ValueError('No physically validated brush pass')
    m=mujoco.MjModel.from_xml_path(result['execution']['scene']);oc=int(m.joint('mini_dust_pan_free').qposadr[0]);wc=np.array([int(m.joint('robot_left_left_'+s).qposadr[0]) for s in ['pos_x','pos_y','pos_z','rot_x','rot_y','rot_z']])
    recorded=np.load(result['execution']['trajectory']);rest=recorded['qpos'][1]
    for i in range(3):
        spec=json.loads((root/f'handle_geometry/episode_000019/mini_dust_pan_left/spec_{i:02d}.json').read_text());q=np.load(spec['anchor_pose'])['qpos'].copy()
        oldR=Rotation.from_quat(q[oc+3:oc+7],scalar_first=True);newR=Rotation.from_quat(rest[oc+3:oc+7],scalar_first=True);delta=newR*oldR.inv()
        q[wc[:3]]=rest[oc:oc+3]+delta.apply(q[wc[:3]]-q[oc:oc+3]);q[wc[3:]]=continuous_euler_xyz(Rotation.from_matrix((delta*Rotation.from_euler('XYZ',q[wc[3:]])).as_matrix()[None]),q[wc[3:]])[0];q[oc:oc+7]=rest[oc:oc+7]
        out=root/f'pan19_preserving_brush_inputs/pose_{i:02d}';out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'anchor.npz',qpos=q)
        spec.update(scene=result['execution']['scene'],reference=result['reference'],anchor_pose=str(out/'anchor.npz'),preserved_right_hand_commands=str(base/'reference.npz'),
            provisional_reconstruction_candidate=str(root/'brush19_support/alpha_1.00/report.json'),original_object_targets_and_clock_unchanged=True)
        dump(out/'spec.json',spec)
    print(root/'pan19_preserving_brush_inputs',flush=True)

if __name__=='__main__':main()
