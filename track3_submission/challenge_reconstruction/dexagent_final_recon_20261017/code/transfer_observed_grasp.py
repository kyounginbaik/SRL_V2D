"""Test an observed robot grasp on the identical object in another episode.

Only robot joint commands are transferred. Each target episode is simulated
from its own unchanged initial state, with its own objects and original clock.
The grasp is a command template, not a weld or an injected object trajectory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

B = Path(__file__).resolve().parents[1]
R = B / 'refinement_20261017'
sys.path.insert(0, str(B.parents[1] / 'third_party/DexAgent'))
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from refine_grasp_cem import contact_envelopes
from wrist_reference import continuous_euler_xyz


def hand_joints(model, side):
    prefix = f'robot_{side}_{side}_'
    wrist = [model.joint(prefix + k).id for k in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z')]
    fingers = [j for j in range(model.njnt) if model.joint(j).name.startswith(prefix) and j not in wrist]
    return wrist + fingers


def run(episode, anchor, blend_s):
    if episode == 10:
        side, obj, source_ep = 'right', 'beige_cup', 8
    elif episode == 3:
        side, obj, source_ep = 'right', 'wooden_spoon', 5
    else:
        side, obj, source_ep = 'left', 'white_pot', 3
    name, source_name = f'episode_{episode:06d}', f'episode_{source_ep:06d}'
    inp, sinp = B/'episodes'/name, B/'episodes'/source_name
    manifest = json.loads((inp/'input_manifest.json').read_text())
    source_manifest = json.loads((sinp/'input_manifest.json').read_text())
    assets = lambda m: next(o for o in m['objects'] if o['id'] == obj)
    # Object-local grasp transfer is valid only for the exact same asset/frame.
    signature = lambda a: (a['source_urdf_sha256'], [(m['kind'],m['sha256']) for m in a['meshes']])
    assert signature(assets(manifest)) == signature(assets(source_manifest)), 'Object geometry/frame differs'
    src, base_dir = R/'block_polish'/source_name, R/'block_polish'/name
    sm = mujoco.MjModel.from_xml_path(str(src/'execution/scene_physics.xml'))
    model = mujoco.MjModel.from_xml_path(str(base_dir/'execution/scene_physics.xml'))
    sr = TR.load_reference(src/'reference.npz', sm)
    ref = TR.load_reference(base_dir/'reference.npz', model)
    original = TR.load_reference(inp/'tracking_reference.npz', model)
    measured = dict(np.load(src/'execution/trajectory.npz'))
    sj, tj = hand_joints(sm,side), hand_joints(model,side)
    sc, tc = sm.jnt_qposadr[sj].astype(int), model.jnt_qposadr[tj].astype(int)
    assert [sm.joint(j).name for j in sj] == [model.joint(j).name for j in tj]
    sa, ta = int(sm.joint(obj+'_free').qposadr[0]), int(model.joint(obj+'_free').qposadr[0])
    k = round(anchor*sr['fps'])
    observed = measured['qpos'][k]
    orot = Rotation.from_quat(observed[sa+3:sa+7],scalar_first=True)
    wrot = Rotation.from_euler('XYZ',sr['qpos'][k,sc[3:6]])
    local_pos = orot.inv().apply(sr['qpos'][k,sc[:3]]-observed[sa:sa+3])
    local_rot = orot.inv()*wrot
    Q = ref['qpos'].copy(); T = len(Q); t = np.arange(T)/ref['fps']
    objrot = Rotation.from_quat(original['qpos'][:,ta+3:ta+7],scalar_first=True)
    target = Q[:,tc].copy()
    target[:,:3] = original['qpos'][:,ta:ta+3]+objrot.apply(np.tile(local_pos,(T,1)))
    target[:,6:] = sr['qpos'][k,sc[6:]]
    envelope = contact_envelopes(manifest['source'],T,ref['fps'])[side]
    active = np.flatnonzero(envelope > .5)
    first, last = int(active[0]), int(active[-1])
    start, stop = max(0.,t[first]-.15), min(t[-1],t[last]+.15)
    weight = np.minimum(np.clip((t-start)/blend_s,0,1),np.clip((stop-t)/blend_s,0,1))
    weight = weight*weight*(3-2*weight)
    Q[:,tc[:3]] = (1-weight[:,None])*Q[:,tc[:3]]+weight[:,None]*target[:,:3]
    base_rot = Rotation.from_euler('XYZ',ref['qpos'][:,tc[3:6]])
    goal_rot = objrot*local_rot
    edited_rot = base_rot*Rotation.from_rotvec((base_rot.inv()*goal_rot).as_rotvec()*weight[:,None])
    Q[:,tc[3:6]] = continuous_euler_xyz(edited_rot,initial_angles=ref['qpos'][0,tc[3:6]])
    Q[:,tc[6:]] = (1-weight[:,None])*Q[:,tc[6:]]+weight[:,None]*target[:,6:]
    for j in tj:
        if model.jnt_limited[j]:
            a = int(model.jnt_qposadr[j]); Q[:,a] = np.clip(Q[:,a],*model.jnt_range[j])
    Q[0] = original['qpos'][0]
    untouched = np.setdiff1d(np.arange(model.nq),tc)
    assert np.array_equal(Q[:,untouched],ref['qpos'][:,untouched])
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = int(model.jnt_qposadr[j]); assert np.array_equal(Q[:,a:a+7],original['qpos'][:,a:a+7])
    out = R/'grasp_transfer'/name/f'anchor{anchor:.1f}_blend{blend_s:.1f}'
    out.mkdir(parents=True,exist_ok=True)
    path = TR.save_reference(out/'reference.npz',model,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],
        source=f'Observed robot grasp template from {source_name} at {anchor}s; target objects unmodified',floating_runtime=True)
    execution = execute_reference(base_dir/'execution/scene_physics.xml',path,out/'execution')
    metrics = evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    report = dict(episode=name,source_episode=source_name,source_anchor_seconds=anchor,object=obj,side=side,
        blend_s=blend_s,active_interval_seconds=[start,stop],reference=path,execution=execution,metrics=metrics,
        learned_policy=False,object_targets_modified=False,source_recording_sha256=hashlib.sha256((src/'execution/trajectory.npz').read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--episode',type=int,choices=[1,3,5,8,10],required=True)
    p.add_argument('--anchor',type=float,required=True);p.add_argument('--blend-s',type=float,required=True)
    run(**vars(p.parse_args()))
