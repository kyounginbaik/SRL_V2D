#!/usr/bin/env python3
"""Import supplied reconstruction into an isolated floating-Sharpa DexAgent setup.

This is input preparation, not a DexAgent success certificate. Source trajectories
remain kinematic targets; objects in scene.xml have unactuated free joints.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import trimesh
import yaml
from scipy.optimize import least_squares


ROOT = Path('/data1/SRL_V2D')
WORKSPACE = Path(__file__).resolve().parents[1]
ROBOT = ROOT / 'robotic_grounding/source/robotic_grounding/robotic_grounding/assets'
DEX = ROOT / 'third_party/DexAgent'
COLORS = {'white_pot': '.88 .88 .85 1', 'white_pot_lid': '.88 .88 .85 1',
          'beige_cup': '.82 .76 .6 1', 'blue_cup': '.16 .35 .65 1',
          'wooden_spoon': '.7 .49 .29 1'}


def values(x):
    return ' '.join(f'{float(v):.12g}' for v in np.asarray(x).ravel())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def xmlwrite(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def prefixed(root, prefix):
    root = copy.deepcopy(root)
    for el in root.iter():
        for k in ('name', 'joint', 'joint1', 'joint2', 'mesh', 'material',
                  'texture', 'class', 'childclass', 'body1', 'body2', 'geom1', 'geom2'):
            if k in el.attrib:
                el.set(k, prefix + el.get(k))
    return root


def base_xml(model):
    root = ET.Element('mujoco', model=model)
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', timestep='.002', gravity='0 0 -9.81',
                  integrator='implicitfast', iterations='50', cone='elliptic')
    custom = ET.SubElement(root, 'custom')
    ET.SubElement(custom, 'text', name='dexagent_robot_schema', data='dexagent.floating_sharpa.v1')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1014', offheight='760')
    ET.SubElement(visual, 'headlight', ambient='.5 .5 .5', diffuse='.7 .7 .7')
    ET.SubElement(root, 'asset')
    ET.SubElement(root, 'worldbody')
    ET.SubElement(root, 'actuator')
    return root


def prepare_hand(side, target):
    """Use reconstruction's exact robot geometry/inertias, plus six servo joints."""
    source = ROBOT / 'xmls/sharpawave' / f'{side}_sharpawave.xml'
    urdf = ROBOT / 'urdfs/sharpawave' / f'{side}_sharpa_wave.urdf'
    src = ET.parse(source).getroot()
    src_meshdir = (source.parent / src.find('compiler').get('meshdir')).resolve()
    out = base_xml(f'floating_{side}_sharpa')
    asset = out.find('asset')
    provenance = []
    for a in src.find('asset'):
        a = copy.deepcopy(a)
        f = a.get('file')
        if f:
            dest = target / 'assets' / side / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_meshdir / f, dest)
            a.set('file', str(dest.resolve()))
            provenance.append({'source': str(src_meshdir / f), 'sha256': sha(dest)})
        asset.append(a)
    defaults = copy.deepcopy(src.find('default'))
    out.insert(2, defaults)
    body = copy.deepcopy(src.find('worldbody/body'))
    # Keep exactly the source collision mesh. Naming alone adds HandRig discovery.
    for geom_index, geom in enumerate(body.iter('geom')):
        if geom.get('contype', '1') != '0' or geom.get('conaffinity', '1') != '0':
            geom.set('name', 'collision_hand_' + side + '_' + geom.get('name', geom.get('mesh')) + f'_{geom_index}')
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
            geom.set('friction', '1 .005 .0001')
            geom.set('condim', '4')
    # These exact source collision meshes already provide convex hand surfaces.
    # Canonical names prevent DexAgent adding a duplicate visual-mesh collider.
    for parent in body.iter('body'):
        tail = parent.get('name', '')
        if tail in [f'{side}_{link}' for link in ('hand_C_MC', 'thumb_MC', 'thumb_DP')]:
            collision = [g for g in parent.findall('geom') if g.get('name', '').startswith('collision_hand_')]
            if collision:
                collision[0].set('name', 'collision_hand_' + tail + '_hull')
    # Existing exact source sites remain; aliases satisfy DexAgent's public API.
    for parent in body.iter():
        for site in list(parent):
            if site.tag == 'site' and site.get('name', '').endswith('_tip_site'):
                alias = copy.deepcopy(site)
                alias.set('name', site.get('name')[:-5])
                parent.append(alias)
    cursor = out.find('worldbody')
    wrist_names = []
    for idx, (suffix, axis, kind, bname) in enumerate([
        ('pos_x', '1 0 0', 'slide', 'base_tx'),
        ('pos_y', '0 1 0', 'slide', 'base_ty'),
        ('pos_z', '0 0 1', 'slide', 'base_tz'),
        ('rot_x', '1 0 0', 'hinge', 'base_roll'),
        ('rot_y', '0 1 0', 'hinge', 'base_pitch'),
        ('rot_z', '0 0 1', 'hinge', 'base_yaw')]):
        cursor = ET.SubElement(cursor, 'body', name=f'{side}_{bname}')
        ET.SubElement(cursor, 'inertial', pos='0 0 0', mass='.001', diaginertia='.000001 .000001 .000001')
        name = f'{side}_{suffix}'
        limits = '-5 5' if kind == 'slide' else '-12.5663706144 12.5663706144'
        ET.SubElement(cursor, 'joint', name=name, type=kind, axis=axis,
                      range=limits, damping='0', armature='.001')
        ET.SubElement(out.find('actuator'), 'position', name=name + '_position',
                      joint=name, kp='1000' if kind == 'slide' else '50',
                      kv='80' if kind == 'slide' else '5', ctrlrange=limits,
                      forcerange='-50 50' if kind == 'slide' else '-5 5')
        wrist_names.append(name)
    cursor.append(body)
    limits = {j.get('name'): j.find('limit').attrib for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
    joint_elems = {j.get('name'): j for j in body.iter('joint')}
    actuator_report = {}
    for act in src.find('actuator'):
        act = copy.deepcopy(act)
        j = act.get('joint')
        cap = float(limits[j]['effort'])
        act.set('name', j + '_position')
        act.attrib.pop('inheritrange', None)
        act.set('ctrlrange', joint_elems[j].get('range'))
        act.set('forcerange', values([-cap, cap]))
        out.find('actuator').append(act)
        actuator_report[j] = {'effort_cap_nm': cap, 'source_velocity_limit_rad_s': float(limits[j]['velocity']),
                              'kp': float(act.get('kp')), 'dampratio': float(act.get('dampratio'))}
    path = target / 'robot' / f'sharpa_{side}.xml'
    path.parent.mkdir(parents=True, exist_ok=True)
    xmlwrite(out, path)
    return out, {'xml': str(path), 'source_xml': str(source), 'source_xml_sha256': sha(source),
                 'source_urdf': str(urdf), 'source_urdf_sha256': sha(urdf), 'meshes': provenance,
                 'wrist_joint_names': wrist_names, 'finger_actuators': actuator_report,
                 'wrist_servo': {'translation_kp_n_m': 1000, 'translation_kv_ns_m': 80,
                                  'translation_force_cap_n': 50, 'rotation_kp_nm_rad': 50,
                                  'rotation_kv_nms_rad': 5, 'rotation_torque_cap_nm': 5},
                 'velocity_limit_scope': 'URDF velocity recorded; MuJoCo position servos impose force caps, not hard velocity caps.',
                 'added_wrist_chain_mass_kg': .006}


def object_xml(name, source, target):
    src = ET.parse(source / 'assets' / f'{name}.urdf').getroot().find('link')
    target.mkdir(parents=True, exist_ok=True)
    root = base_xml(name)
    root.remove(root.find('actuator'))
    body = ET.SubElement(root.find('worldbody'), 'body', name=name)
    ET.SubElement(body, 'freejoint', name=name + '_free')
    inertia = src.find('inertial')
    origin = inertia.find('origin')
    ii = inertia.find('inertia').attrib
    # MuJoCo fullinertia uses the same tensor convention as URDF.
    ET.SubElement(body, 'inertial', pos=origin.get('xyz', '0 0 0'),
                  mass=inertia.find('mass').get('value'),
                  fullinertia=' '.join(ii[k] for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')))
    if origin.get('rpy', '0 0 0') != '0 0 0':
        raise ValueError('Nonidentity URDF inertial rotation needs explicit tensor rotation')
    files = []
    for kind in ('visual', 'collision'):
        for idx, el in enumerate(src.findall(kind)):
            mesh = el.find('geometry/mesh')
            if mesh is None:
                raise ValueError(f'{name}: expected mesh URDF {kind}')
            rel = Path(mesh.get('filename'))
            srcpath = source / 'assets' / rel
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(srcpath, dest)
            files.append({'source': str(srcpath), 'sha256': sha(dest), 'kind': kind})
            meshname = f'{name}_{kind}_{idx}'
            ET.SubElement(root.find('asset'), 'mesh', name=meshname, file=str(dest.resolve()),
                          scale=mesh.get('scale', '1 1 1'))
            geom = ET.SubElement(body, 'geom', name=meshname, type='mesh', mesh=meshname,
                                mass='0', contype='0' if kind == 'visual' else '1',
                                conaffinity='0' if kind == 'visual' else '1', group='1' if kind == 'visual' else '3',
                                rgba=COLORS.get(name, '.7 .7 .7 1'), friction='1 .005 .0001', condim='4')
            local = el.find('origin')
            if local is not None:
                geom.set('pos', local.get('xyz', '0 0 0'))
                geom.set('quat', values(Rotation.from_euler('xyz', np.fromstring(local.get('rpy', '0 0 0'), sep=' ')).as_quat(scalar_first=True)))
    path = target / f'{name}.xml'
    xmlwrite(root, path)
    return root, {'id': name, 'mass_kg': float(inertia.find('mass').get('value')),
                  'source_urdf': str(source / 'assets' / f'{name}.urdf'),
                  'source_urdf_sha256': sha(source / 'assets' / f'{name}.urdf'), 'meshes': files,
                  'asset_xml': str(path), 'friction_assumption': [1, .005, .0001]}


def euler_continuous(quats):
    from wrist_reference import continuous_euler_xyz
    return continuous_euler_xyz(Rotation.from_quat(quats, scalar_first=True))


def infer_table(first, K, world_from_camera, destination):
    """Fit the visible wooden circular edge on the already inferred support plane."""
    hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([4, 65, 35], np.uint8), np.array([35, 255, 255], np.uint8))
    mask[:int(.35 * len(mask))] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = cv2.convexHull(max(contours, key=cv2.contourArea))[:, 0].astype(float)
    # Glare and the white pot/handle can cut holes through the colour component;
    # the convex external envelope follows the circular tabletop boundary.
    contour = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b-a) / 3)))
                              for a, b in zip(contour, np.roll(contour, -1, axis=0))])
    height, width = first.shape[:2]
    contour = contour[(contour[:, 1] < height - 5) & (contour[:, 0] > 5) & (contour[:, 0] < width - 5)]
    rays_camera = np.c_[contour, np.ones(len(contour))] @ np.linalg.inv(K).T
    rays_world = rays_camera @ world_from_camera[:3, :3].T
    origin = world_from_camera[:3, 3]
    distance = -origin[2] / rays_world[:, 2]
    points = origin + rays_world * distance[:, None]
    xy = points[distance > 0, :2]
    if len(xy) < 100:
        raise ValueError('Too few calibrated tabletop edge points')
    a = np.c_[2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))]
    b = (xy * xy).sum(1)
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    initial = [cx, cy, np.sqrt(max(.001, c + cx * cx + cy * cy))]
    fit = least_squares(lambda p: np.linalg.norm(xy - p[:2], axis=1) - p[2], initial,
                        loss='soft_l1', f_scale=.005)
    error = abs(np.linalg.norm(xy - fit.x[:2], axis=1) - fit.x[2])
    inliers = error < .015
    if inliers.sum() > 100:
        fit = least_squares(lambda p: np.linalg.norm(xy[inliers] - p[:2], axis=1) - p[2], fit.x,
                            loss='soft_l1', f_scale=.003)
    radius = float(fit.x[2])
    if not .15 < radius < .8 or float(np.median(error)) > .025:
        raise ValueError(f'Tabletop circle fit failed quality checks: radius={radius}, median={np.median(error)}')
    angle = np.linspace(0, 2 * np.pi, 361)
    xyz = np.c_[fit.x[0] + radius * np.cos(angle), fit.x[1] + radius * np.sin(angle), np.zeros(len(angle))]
    cv = (xyz - origin) @ world_from_camera[:3, :3]
    pixels = cv @ K.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    overlay = first.copy()
    cv2.polylines(overlay, [np.round(pixels).astype(np.int32)], True, (0, 255, 0), 3)
    cv2.putText(overlay, f'Calibrated tabletop edge fit: radius {radius:.3f} m', (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(str(destination), overlay)
    return {'center_xy_m': fit.x[:2].tolist(), 'radius_m': radius, 'thickness_m': .04,
            'method': 'brown connected-component outer contour projected onto support plane with calibrated camera, robust circle fit',
            'edge_samples': len(xy), 'median_radial_residual_m': float(np.median(error)),
            'inlier_fraction_15mm': float(np.mean(inliers)), 'review_image': str(destination),
            'limitation': 'circle fit assumes a horizontal round wooden tabletop; thickness assumed 4cm'}


def prepare(number, setup_from_source=None, support_evidence=None):
    episode = f'episode_{number:06d}'
    source = ROOT / 'challenge_reconstruction/final_recon' / episode
    output = WORKSPACE / 'episodes' / episode
    setup = output / 'setup'
    stage2 = output / 'stage2'
    setup.mkdir(parents=True, exist_ok=True)
    stage2.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / 'config.json').read_text())
    row = pq.read_table(source / 'reference.parquet').to_pylist()[0]
    obj = np.load(source / 'object_poses.npz')
    cam = np.load(source / 'camera_motion.npz')
    names = obj['names'].tolist()
    positions = np.array(row['object_body_position'])
    quats = np.array(row['object_body_wxyz'])
    count = len(positions)
    fps = float(row['fps'])
    assert np.allclose(positions, obj['world_from_object'][:, :, :3, 3])
    bottoms = {}
    for k, name in enumerate(names):
        mesh = trimesh.load(source / 'assets' / f'{name}.obj', process=False)
        points = mesh.vertices @ obj['world_from_object'][0, k, :3, :3].T + positions[0, k]
        bottoms[name] = float(points[:, 2].min())
    support_z = min(bottoms.values())
    transform = np.eye(4)
    transform[2, 3] = -support_z
    calibrated_support = setup_from_source is not None
    if calibrated_support:
        transform = np.asarray(setup_from_source, dtype=float)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8)
                or np.linalg.det(transform[:3, :3]) < .999999):
            raise ValueError('Explicit setup_from_source must be a rigid transform')
        if support_evidence is None:
            raise ValueError('A calibrated support transform requires its measured evidence')
    transformed_objects = transform @ obj['world_from_object']
    cameras = transform @ cam['world_from_rgb']
    positions = transformed_objects[:, :, :3, 3]
    if calibrated_support:
        quats = Rotation.from_matrix(transformed_objects[:, :, :3, :3].reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(count, len(names), 4)
    source_video = Path(config['source']) / 'anycalib/undistorted.mp4'
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise FileNotFoundError(source_video)
    ok, first = cap.read()
    assert ok
    width, height = first.shape[1], first.shape[0]
    source_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    cv2.imwrite(str(stage2 / 'frame0.jpg'), first)
    intr = json.loads((source / 'stereo_calibration/rgb_intrinsics.json').read_text())
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    K[0] *= width / intr['width']
    K[1] *= height / intr['height']
    table_geometry = infer_table(first, K, cameras[0], stage2 / 'table_fit.jpg')
    camera_quats = Rotation.from_matrix(cameras[:, :3, :3] @ np.diag([1., -1., -1.])).as_quat(scalar_first=True)
    camera = {'width': width, 'height': height, 'fx': K[0, 0], 'fy': K[1, 1],
              'cx': K[0, 2], 'cy': K[1, 2], 'fovy_deg': math.degrees(2 * math.atan(height / (2 * K[1, 1]))),
              'pos': cameras[0, :3, 3].tolist(), 'quat_wxyz': camera_quats[0].tolist(),
              'source': 'given', 'frame': 'source reconstruction world rigidly transformed to declared support frame',
              'moving_camera': True, 'trajectory_npz': str(output / 'camera_trajectory.npz'),
              'static_pose_scope': 'frame 0 only; all later renders must update camera from trajectory',
              'R_cv': cameras[0, :3, :3].tolist()}
    dump(stage2 / 'camera.json', camera)
    np.savez_compressed(output / 'camera_trajectory.npz', timestamps_seconds=np.arange(count) / fps,
                        world_from_rgb=cameras, positions=cameras[:, :3, 3], quaternions_wxyz=camera_quats,
                        K=K, width=width, height=height, fps=fps)
    scene = base_xml(episode)
    defaults = ET.Element('default')
    scene.insert(2, defaults)
    robot_reports = {}
    stations = {}
    for side in ('left', 'right'):
        hand, robot_reports[side] = prepare_hand(side, setup)
        prefix = f'robot_{side}_'
        hand = prefixed(hand, prefix)
        for child in hand.find('asset'):
            scene.find('asset').append(child)
        default = hand.find('default')
        default.set('class', prefix + 'defaults')
        defaults.append(default)
        for child in hand.find('worldbody'):
            child.set('childclass', prefix + 'defaults')
            scene.find('worldbody').append(child)
        for child in hand.find('actuator'):
            scene.find('actuator').append(child)
        stations[side] = {'arm': None, 'hand': 'floating Sharpa Wave, exact reconstruction asset',
                          'base_body': prefix + side + '_base_tx', 'arm_joints': [],
                          'hand_root_body': prefix + side + '_base_yaw',
                          'hand_xml': f'robot/sharpa_{side}.xml',
                          'wrist_joints': [prefix + n for n in robot_reports[side]['wrist_joint_names']],
                          'finger_joint_prefix': prefix + side + '_', 'home_arm_rad': []}
    world = scene.find('worldbody')
    ET.SubElement(world, 'light', pos='0 0 2', dir='0 0 -1', directional='true')
    table = ET.SubElement(world, 'body', name='table')
    ET.SubElement(table, 'geom', name='g_table', type='cylinder',
                  pos=values([*table_geometry['center_xy_m'], -.02]), size=values([table_geometry['radius_m'], .02]),
                  rgba='.55 .40 .25 1', friction='1 .005 .0001')
    for name in ('ego', 'video_cam'):
        # MuJoCo camera optical axis -z and image-up +y. principalpixel is an
        # offset from image centre; image y is down, hence the vertical sign.
        ET.SubElement(world, 'camera', name=name, pos=values(camera['pos']), quat=values(camera['quat_wxyz']),
                      resolution=f'{width} {height}', sensorsize=f'{width} {height}',
                      focalpixel=values([K[0, 0], K[1, 1]]),
                      principalpixel=values([width / 2 - K[0, 2], height / 2 - K[1, 2]]))
    # Setup contains only robot and support; scene2 adds the exact task assets.
    setup_scene = copy.deepcopy(scene)
    object_reports = []
    recon_objects = []
    for index, name in enumerate(names):
        asset, report = object_xml(name, source, stage2 / 'assets' / name)
        object_reports.append(report)
        for child in asset.find('asset'):
            scene.find('asset').append(child)
        body = asset.find('worldbody/body')
        body.set('pos', values(positions[0, index]))
        body.set('quat', values(quats[0, index]))
        world.append(body)
        recon_objects.append({'id': name, 'type': 'rigid', 'skill': 'supplied_final_recon_mesh',
                              'skill_origin': 'external_input', 'asset_xml': report['asset_xml'],
                              'root_body': name, 'bodies': [name],
                              'pose': {'x': float(positions[0, index, 0]), 'y': float(positions[0, index, 1]),
                                       'z': float(positions[0, index, 2]), 'yaw_deg': float(Rotation.from_quat(quats[0, index], scalar_first=True).as_euler('xyz', degrees=True)[2]),
                                       'quat_wxyz': quats[0, index].tolist()},
                              'params': {'mass': report['mass_kg'], 'exact_supplied_mesh': True},
                              'verification_plan': {}, 'verifiers': {},
                              'properties_covered': {}, 'import_status': 'input asset; not a DexAgent reconstruction certificate'})
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))
    q = np.tile(model.qpos0, (count, 1))
    robot_columns = []
    for side in ('left', 'right'):
        prefix = f'robot_{side}_'
        wrist = np.array(row[f'robot_{side}_wrist_position']) @ transform[:3, :3].T + transform[:3, 3]
        wrist_quats = np.array(row[f'robot_{side}_wrist_wxyz'])
        if calibrated_support:
            wrist_quats = (Rotation.from_matrix(transform[:3, :3]) * Rotation.from_quat(wrist_quats, scalar_first=True)).as_quat(scalar_first=True)
        angles = euler_continuous(wrist_quats)
        for k, name in enumerate(robot_reports[side]['wrist_joint_names']):
            col = model.joint(prefix + name).qposadr[0]
            q[:, col] = np.c_[wrist, angles][:, k]
            robot_columns.append(int(col))
        finger_values = np.array(row[f'robot_{side}_finger_joints'])
        for k, name in enumerate(row[f'{side}_robot_finger_joint_names']):
            col = model.joint(prefix + name).qposadr[0]
            q[:, col] = finger_values[:, k]
            robot_columns.append(int(col))
    for k, name in enumerate(names):
        col = model.joint(name + '_free').qposadr[0]
        q[:, col:col + 3] = positions[:, k]
        q[:, col + 3:col + 7] = quats[:, k]
    ctrl = q[:, np.asarray(model.jnt_qposadr)[model.actuator_trnid[:, 0]]]
    if not np.isfinite(q).all():
        raise ValueError('Non-finite reference')
    home = ET.SubElement(scene, 'keyframe')
    ET.SubElement(home, 'key', name='home', qpos=values(q[0]), ctrl=values(ctrl[0]))
    setup_model = mujoco.MjModel.from_xml_string(ET.tostring(setup_scene, encoding='unicode'))
    setup_home = ET.SubElement(setup_scene, 'keyframe')
    ET.SubElement(setup_home, 'key', name='home', qpos=values(q[0, :setup_model.nq]), ctrl=values(ctrl[0]))
    xmlwrite(scene, stage2 / 'scene.xml')
    xmlwrite(setup_scene, setup / 'scene.xml')
    sys.path.insert(0, str(DEX))
    from tool_library.common.episode import joint_names
    from tool_library.common.hand import HandRig
    model = mujoco.MjModel.from_xml_path(str(stage2 / 'scene.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    rigs = {s: HandRig.build(model, f'robot_{s}_', s) for s in ('left', 'right')}
    assert all(len(r.finger_act) == 22 and len(r.wrist_act) == 6 for r in rigs.values())
    assert model.nu == 56
    assert np.all(model.actuator_forcelimited) and np.all(model.actuator_ctrllimited)
    for frame in np.linspace(0, count - 1, min(count, 12)).astype(int):
        data.qpos[:] = q[frame]
        mujoco.mj_forward(model, data)
        for side in ('left', 'right'):
            palm = model.body(f'robot_{side}_{side}_hand_C_MC').id
            target = transform[:3, :3] @ np.array(row[f'robot_{side}_wrist_position'][frame]) + transform[:3, 3]
            assert np.linalg.norm(data.xpos[palm] - target) < 1e-8
            rot = transform[:3, :3] @ Rotation.from_quat(row[f'robot_{side}_wrist_wxyz'][frame], scalar_first=True).as_matrix()
            assert np.max(abs(data.xmat[palm].reshape(3, 3) - rot)) < 1e-8
    # Bare forward dynamics only: no grasp/task success claim.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(25):
        mujoco.mj_step(model, data)
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    assert finite
    np.savez_compressed(output / 'reference_trajectory.npz', qpos=q, ctrl=ctrl, fps=fps,
                        joint_names=np.asarray(joint_names(model)), object_names=np.asarray(names),
                        timestamps_seconds=np.arange(count) / fps, object_positions=positions,
                        object_quaternions_wxyz=quats, robot_qpos_columns=np.asarray(robot_columns),
                        setup_from_source=transform, source_from_setup=np.linalg.inv(transform),
                        kind='kinematic_reference_not_physical_execution')
    from tool_library.common.tracking import save_reference
    save_reference(output / 'tracking_reference.npz', model, q,
                   ['both: reference tracking'] * count, fps, np.zeros(count, dtype=int),
                   [{'stage1_indices': [0], 'motion': 'track_reference', 'side': 'both'}],
                   source='supplied final_recon human-derived Sharpa IK; kinematic reconstruction, not physical execution',
                   arms_coupled=False, timestamps_seconds=np.arange(count) / fps)
    setup_config = {'name': episode + '_floating_sharpa', 'scene': 'scene.xml',
                    'frame': {'origin': 'source world transformed by explicit setup_from_source',
                              'x': 'declared setup x', 'y': 'declared setup y',
                              'z': 'measured tabletop normal' if calibrated_support else 'assumed source gravity-aligned up',
                              'setup_from_source': transform.tolist(), 'tabletop_z': 0.},
                    'table': {'body': 'table', 'geom': 'g_table', 'center': [*table_geometry['center_xy_m'], -.02],
                              'half_extents': [table_geometry['radius_m'], table_geometry['radius_m'], .02],
                              'geometry': 'finite cylinder from calibrated source image', 'fit': table_geometry,
                              'work_area': {'x': [-1., 1.], 'y': [-1., 1.]}},
                    'stations': stations, 'robot': {'body_prefix': 'robot_', 'attachment': 'floating six-axis servo chain',
                                                   'nq': 56, 'reach': None},
                    'runtime_robot': {'schema': 'dexagent.floating_sharpa.v1', 'compiler': 'floating_sharpa_adapter',
                                      'actuators': '56 bounded position servos; no gravity compensation or object actuation'},
                    'home': {'keyframe': 'home', 'description': 'source reconstruction frame zero'},
                    'cameras': {'ego': camera}, 'physics': {'timestep': .002, 'integrator': 'implicitfast'}}
    (setup / 'setup.yaml').write_text(yaml.safe_dump(json.loads(json.dumps(setup_config)), sort_keys=False))
    dump(stage2 / 'scene_recon.json', {'scene_semantic': '../stage1/scene_semantics.json',
          'scene_xml': 'scene.xml', 'camera_json': 'camera.json', 'frame0': 'frame0.jpg',
          'simulator': 'mujoco', 'objects': recon_objects, 'scene_checks': {},
          'source': 'supplied final_recon; input conversion, not DexAgent reconstruction',
          'notes': ['Moving calibrated camera stored separately; static camera pose is only frame0.',
                    'No reconstruction-mask IoU or task-success verifier results have been fabricated.',
                    'Measured calibrated stereo tabletop normal/height defines setup frame.' if calibrated_support else
                    'Gravity-aligned source frame assumed; support-plane height inferred from initial mesh lower bound.']})
    report = {'episode': episode, 'source': str(source), 'setup': str(setup), 'scene_recon': str(stage2 / 'scene_recon.json'),
              'source_reference': str(source / 'reference.parquet'), 'source_reference_sha256': sha(source / 'reference.parquet'),
              'source_object_poses_sha256': sha(source / 'object_poses.npz'),
              'source_video': str(source_video), 'source_video_frames': source_video_frames, 'source_video_fps': video_fps,
              'frames': count, 'fps': fps, 'last_reference_timestamp_seconds': (count - 1) / fps,
              'setup_from_source': transform.tolist(), 'source_from_setup': np.linalg.inv(transform).tolist(),
              'support': {'source_z_m': None if calibrated_support else support_z, 'initial_object_bottoms_m': bottoms,
                          'table_geometry': table_geometry,
                          'method': 'calibrated stereo tabletop plane; entire scene rigidly aligned to its normal' if calibrated_support else
                                    'minimum world-z vertex over all source visual meshes at frame0',
                          'evidence': support_evidence,
                          'limitation': 'Stereo/camera uncertainty remains; 4cm tabletop thickness assumed' if calibrated_support else
                                        'assumes at least one object initially rests on a horizontal support; not a calibrated table estimate'},
              'objects': object_reports, 'robot': robot_reports, 'camera': camera,
              'verification': {'compiled': True, 'nq': model.nq, 'nv': model.nv, 'nu': model.nu,
                               'both_hands_have_6_wrist_and_22_finger_controls': True,
                               'wrist_fk_matches_source_sampled_frames': True, 'all_actuators_force_and_ctrl_bounded': True,
                               'finite_50ms_forward_dynamics': finite, 'task_success_tested': False},
              'limitations': ['Physical robot is the explicitly requested floating Sharpa adaptation, not native DexAgent YAM arms.',
                              'Source reconstructed trajectories are references, not measured object ground truth.',
                              'Initial support is inferred; friction is assumed; source masses and inertias are copied.']}
    dump(output / 'input_manifest.json', report)
    print(json.dumps({'episode': episode, 'output': str(output), 'nq': model.nq, 'nu': model.nu, 'frames': count,
                      'objects': {r['id']: r['mass_kg'] for r in object_reports}, 'support_z': support_z}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, nargs='+', default=[1, 3, 5, 8, 10])
    args = parser.parse_args()
    results = [prepare(i) for i in args.episodes]
    dump(WORKSPACE / 'prepared_episodes.json', [{'episode': r['episode'], 'manifest': str(WORKSPACE / 'episodes' / r['episode'] / 'input_manifest.json')} for r in results])
