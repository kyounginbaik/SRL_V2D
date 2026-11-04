"""Prepare immutable-input episode-1 physical pickup search specifications.

The collision-qualified pose defines a wrist/finger grasp proposal only. It
never replaces the scene keyframe or the original object target trajectory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import mujoco
from scipy.spatial import cKDTree

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B.parents[1] / 'third_party/DexAgent'))
from tool_library.common import hand as H
from lid_collision_grasps import collision_surface


def prepare(output=None, opening_m=.003):
    output = Path(output or B/'refinement_20261018/episode_000001/pickup_inputs')
    output.mkdir(parents=True, exist_ok=True)
    reference = B/'refinement_20261017/continuous_references/episode_000001.npz'
    original = B/'episodes/episode_000001/tracking_reference.npz'
    source = B.parent/'final_recon/episode_000001'
    specs = []
    for candidate in range(4):
        old = B/f'refinement_20261017/lid_collision_feedback/episode_000001/candidate_{candidate:02d}_open10'
        scene = old/'execution/scene_physics.xml'
        anchor = B/f'refinement_20261017/lid_collision_grasps/episode_000001/pose_{candidate:02d}.npz'
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        data.qpos[:] = np.load(anchor)['qpos']
        mujoco.mj_forward(model, data)
        rig = H.HandRig.build(model, 'robot_right_', 'right')
        body = model.body('white_pot_lid').id
        points, normals, _ = collision_surface(model, data, body)
        position = data.xpos[body].copy()
        rotation = data.xmat[body].reshape(3, 3).copy()
        tree = cKDTree(points)
        targets = {}
        original_sites = {}
        selected_normals = {}
        for finger in ('thumb', 'index'):
            point = data.site_xpos[rig.fingertip_sites[finger]].copy()
            closest = tree.query((point-position)@rotation)[1]
            normal = rotation@normals[closest]
            targets[finger] = point + opening_m*normal
            original_sites[finger] = point
            selected_normals[finger] = normal
        residual = H.finger_ik(model, data, rig, targets, iters=120)
        achieved = {finger: float((data.site_xpos[rig.fingertip_sites[finger]]-original_sites[finger])@selected_normals[finger])
                    for finger in targets}
        finger_names = list(rig.finger_qpos)
        open_fingers = data.qpos[list(rig.finger_qpos.values())].copy()
        spec = dict(scene=str(scene), reference=str(reference), original_reference=str(original),
                    source=str(source), geometry=str(B/'episodes/episode_000001/metric_geometry.json'),
                    manifest=str(B/'episodes/episode_000001/input_manifest.json'),
                    object_name='white_pot_lid', side='right', anchor_pose=str(anchor),
                    open_fingers=open_fingers.tolist(), finger_names=finger_names,
                    times=dict(start=.9, hover=1.35, contact=1.75, closed=2.05,
                               transport=2.15, release=4.45), horizon_s=4.,
                    episode='episode_000001', requested_opening_m=opening_m,
                    measured_opening_m=achieved, opening_ik_residual_m=residual,
                    original_initial_state='exact frozen scene keyframe',
                    object_targets='immutable original reference on original clock',
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (scene, reference, original, anchor)})
        destination = output/f'pose_{candidate:02d}.json'
        destination.write_text(json.dumps(spec, indent=2)+'\n')
        specs.append(str(destination))
    (output/'specifications.json').write_text(json.dumps(specs, indent=2)+'\n')
    print(json.dumps(specs, indent=2))
    return specs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--opening-m', type=float, default=.003)
    prepare(**vars(parser.parse_args()))
