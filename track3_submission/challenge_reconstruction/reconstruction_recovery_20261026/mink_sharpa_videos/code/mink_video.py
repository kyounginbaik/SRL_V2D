#!/usr/bin/env python3
"""Kinematic Mink IK of reconstructed MANO hands onto floating Sharpa Wave hands,
rendered from the physical ego RGB camera next to the original undistorted RGB.

Phases (each reads the previous phase's files under OUT/<bundle_name>/episode_XXXXXX):
  decode  MANO FK of the HaWoR hands in the OLD camera frame   (docker v2d_task_library_loader)
  map     old camera -> physical RGB camera -> new bundle world (dexagent venv)
  ik      exact floating Sharpa Wave Mink IK (mink_sharpa_exact) (docker robotic-grounding:mink-1.3.0)
  render  MuJoCo render from world_from_rgb + intrinsics.json, side by side with rgb.mp4 (venv, node100)
No physics. Object poses are read verbatim from the bundle.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, shlex
from pathlib import Path
import numpy as np

ROOT = Path('/data1/SRL_V2D')
RECOVERY = ROOT / 'challenge_reconstruction/reconstruction_recovery_20261026'
OUT_ROOT = Path(os.environ.get('MINK_VIDEO_OUT', str(RECOVERY / 'mink_sharpa_videos')))
MANO_DIR = ROOT / 'reconstruction/data/weights/wilor/pretrained_models'
LOADER = ROOT / 'reconstruction/modules/v2d_task_library_loader/lib/read_mano.py'
MINK_EXACT = ROOT / 'challenge_reconstruction/dexagent_remaining7_20261019/code/mink_sharpa_exact.py'
SHARPA_XML = ROOT / 'robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave'
CALIB = Path('/data1/video_to_data_challenge/track_3/public/meta/camera_calibration.json')
SIDES = ('left', 'right')
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]


def load(p): return json.loads(Path(p).read_text())
def dump(p, d): Path(p).write_text(json.dumps(d, indent=2) + '\n')
def ep_name(e): return f'episode_{int(e):06d}'


def episodes_of(bundle: Path, eps):
    if eps: return [int(e) for e in eps]
    return sorted(int(p.name.split('_')[1]) for p in bundle.glob('episode_*') if (p / 'object_poses.npz').is_file())


def out_dir(bundle: Path, e: int) -> Path:
    d = OUT_ROOT / bundle.name / ep_name(e); d.mkdir(parents=True, exist_ok=True); return d


def record(bundle, e): return load(bundle / ep_name(e) / 'prepared.json')['record']


def hand_source(rec):
    root = Path(rec['source']) / rec['names'][0]
    return root, root / 'result_slam_gravity_aligned/result.npz'


# ----------------------------------------------------------------------------- decode
def decode(bundle, e):
    import torch
    torch.set_num_threads(2)
    import importlib.util
    spec = importlib.util.spec_from_file_location('local_read_mano', LOADER); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mano = mod.MANO(str(MANO_DIR), device=torch.device('cpu'), flat_hand_mean=True, center_idx=None)
    rec = record(bundle, e); src_root, src = hand_source(rec); d = np.load(src)
    out = {'old_camera_to_world': d['camera_to_world_transform'].astype(float), 'source': str(src)}
    for side in SIDES:
        aa = d[f'hand_{side}_wrist_orient_in_camera'].astype(float).copy()
        fp = d[f'hand_{side}_finger_pose'].astype(float).reshape(-1, 15, 3).copy()
        trans = d[f'hand_{side}_wrist_trans_in_camera'].astype(float)
        scale = d[f'hand_{side}_scale'].astype(float)
        betas = d[f'hand_{side}_betas'].astype(float)
        if side == 'left':  # right-parameter-space storage -> native left MANO (negate yz once)
            aa[:, 1:] *= -1; fp[:, :, 1:] *= -1
        T = len(aa); tens = lambda x: torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            r = mano.forward(side, tens(betas), tens(aa), torch.zeros((T, 3)), tens(fp.reshape(T, 45)))
        v = r['vertices'].numpy().astype(float); j = r['joints'].numpy().astype(float)
        c = v.mean(axis=1, keepdims=True)
        # HaWoR depth-aligned contract (prepare_hawor_hands.decode): scale about vertex centroid, then cam_t.
        v = (v - c) * scale[:, None, None] + c + trans[:, None]
        j = (j - c) * scale[:, None, None] + c + trans[:, None]
        out[f'{side}_joints_cam'] = j; out[f'{side}_verts_cam'] = v.astype(np.float32)
        out[f'{side}_wxyz_cam'] = r['joints_wxyz'].numpy().astype(float)
        out[f'{side}_faces'] = r['faces'].numpy().astype(np.int32)
        out[f'{side}_scale'] = scale; out[f'{side}_valid'] = d[f'hand_{side}_is_valid'].astype(bool)
    np.savez_compressed(out_dir(bundle, e) / 'mano_oldcam.npz', **out)
    print('decoded', e, flush=True)


# ----------------------------------------------------------------------------- map
def quat_to_mat(wxyz):
    from scipy.spatial.transform import Rotation as R
    s = wxyz.shape[:-1]
    return R.from_quat(wxyz.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(s + (3, 3))


def mat_to_quat(m):
    from scipy.spatial.transform import Rotation as R
    s = m.shape[:-2]
    return R.from_matrix(m.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(s + (4,))


def ray_rotation(a, b):
    """Batch rotation taking unit ray a onto unit ray b (hand_priors.py v2 formula)."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
    x, y, z = np.cross(a, b).T; n = len(a); K = np.zeros((n, 3, 3))
    K[:, 0, 1] = -z; K[:, 0, 2] = y; K[:, 1, 0] = z; K[:, 1, 2] = -x; K[:, 2, 0] = -y; K[:, 2, 1] = x
    return np.eye(3)[None] + K + K @ K / np.maximum(1e-8, 1 + np.sum(a * b, axis=1))[:, None, None]


def anycalib_pixels(src_root, rec_intr, pts):
    """Old (stable virtual pinhole) camera points -> pixels of the physical undistorted RGB (intrinsics.json)."""
    import cv2
    virtual = load(src_root / 'anycalib/undistorted_intrinsics.json'); stable = load(src_root / 'intrinsics_stable.json')
    any_k = load(src_root / 'anycalib/intrinsics.json'); any_d = np.array(load(src_root / 'anycalib/distortion.json')['params'])
    cal = load(CALIB)['cameras']['ego_cam_a']; pk = np.array(cal['intrinsic_matrix']); pd = np.array(cal['distortion_coefficients'])
    raw_k = np.array([[any_k['fx'], 0, any_k['cx']], [0, any_k['fy'], any_k['cy']], [0, 0, 1.]])
    tk = np.array([[rec_intr['fx'], 0, rec_intr['cx']], [0, rec_intr['fy'], rec_intr['cy']], [0, 0, 1.]])
    ray = pts / np.maximum(pts[:, 2:3], 1e-6)
    uv = np.c_[ray[:, 0] * stable['fx'] + stable['cx'], ray[:, 1] * stable['fy'] + stable['cy']]
    nrm = np.c_[(uv[:, 0] - virtual['cx']) / virtual['fx'], (uv[:, 1] - virtual['cy']) / virtual['fy']]
    raw = cv2.fisheye.distortPoints(nrm[:, None].astype(float), raw_k, any_d)
    return cv2.undistortPoints(raw, pk, pd, P=tk)[:, 0]


def interp_invalid(x, valid):
    """Fill invalid frames of x [T,...] by linear interpolation / endpoint hold over valid frames."""
    if valid.all() or not valid.any(): return x
    T = len(x); ix = np.flatnonzero(valid); flat = x.reshape(T, -1)
    return np.stack([np.interp(np.arange(T), ix, flat[ix, k]) for k in range(flat.shape[1])], 1).reshape(x.shape)


def stereo_depth(canon, rec, label, frame, uv_center, radius=40):
    """Median metric depth of the hand mask (or a window if no mask) near a pixel in the canonical RGB-D."""
    import cv2
    enc = cv2.imread(str(canon / 'depth' / f'{frame:06d}.png'), -1)
    if enc is None: return np.nan
    m = cv2.imread(str(canon / 'objects' / rec['names'][0] / 'masks' / str(label) / f'{frame:06d}.png'), 0)
    H, W = enc.shape; u, v = int(round(uv_center[0])), int(round(uv_center[1]))
    if not (0 <= u < W and 0 <= v < H): return np.nan
    y0, y1, x0, x1 = max(0, v - radius), min(H, v + radius), max(0, u - radius), min(W, u + radius)
    patch = enc[y0:y1, x0:x1].astype(float)
    sel = (patch > 0) & (patch < 65535)
    if m is not None: sel &= m[y0:y1, x0:x1] > 0
    z = 65535 / np.maximum(patch[sel], 1) - 1; z = z[(z > .1) & (z < 2.)]
    return float(np.median(z)) if len(z) >= 20 else np.nan


def map_hands(bundle, e, mode='anycalib', depth='stereo'):
    from scipy.signal import savgol_filter
    o = out_dir(bundle, e); m = np.load(o / 'mano_oldcam.npz'); rec = record(bundle, e)
    src_root, _ = hand_source(rec); intr = load(bundle / ep_name(e) / 'intrinsics.json')
    poses = np.load(bundle / ep_name(e) / 'object_poses.npz'); W = poses['world_from_rgb'].astype(float)
    T = len(W); assert len(m['old_camera_to_world']) == T, (e, len(m['old_camera_to_world']), T)
    canon = Path(rec['output']); manifest = load(Path(hand_source(rec)[1]).parent / 'manifest.json')
    labels = {sd: (Path(manifest['sources'][f'{sd}_hand_dir']).name if manifest['sources'].get(f'{sd}_hand_dir') else None) for sd in SIDES}
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    res = {'fps': 20., 'world_from_rgb': W}; report = {'episode': e, 'mode': mode, 'depth': depth, 'labels': labels, 'sides': {}}
    for side in SIDES:
        j = m[f'{side}_joints_cam'].copy(); v = m[f'{side}_verts_cam'].astype(float); rot = quat_to_mat(m[f'{side}_wxyz_cam'])
        valid = m[f'{side}_valid'].copy(); s = m[f'{side}_scale']
        if depth in ('metric', 'stereo'):
            # Same image projection, metric MANO size: scale whole hand about the camera centre by 1/s.
            j = j / s[:, None, None]; v = v / s[:, None, None]
        if mode == 'anycalib':
            new_uv = anycalib_pixels(src_root, intr, j[:, 0])
            nr = np.c_[(new_uv[:, 0] - intr['cx']) / intr['fx'], (new_uv[:, 1] - intr['cy']) / intr['fy'], np.ones(T)]
            Rc = ray_rotation(j[:, 0], nr)
            j = np.einsum('tij,tkj->tki', Rc, j); v = np.einsum('tij,tkj->tki', Rc, v); rot = np.einsum('tij,tkjl->tkil', Rc, rot)
        zst = np.full(T, np.nan); zh = j[:, [0, 5, 9, 13, 17]].mean(1)[:, 2]
        if depth == 'stereo' or True:
            cen = j[:, [0, 5, 9, 13, 17]].mean(1); uv = (cen / cen[:, 2:3]) @ K.T
            for t in range(T):
                if valid[t]: zst[t] = stereo_depth(canon, rec, labels[side], t, uv[t, :2])
        ratio = zst / zh; good = np.isfinite(ratio) & valid
        if depth == 'stereo' and good.sum() >= 5:
            # Robust smooth per-frame depth correction along the camera ray, clamped to +-30 %.
            r = np.clip(ratio, .7, 1.3); r = interp_invalid(np.where(good, r, np.nan)[:, None], good)[:, 0]
            win = min(21, (T // 2) * 2 - 1); r = savgol_filter(r, win, 2) if win >= 5 else r
            cen = j[:, [0, 5, 9, 13, 17]].mean(1, keepdims=True)
            shift = cen * (r[:, None, None] - 1)  # translate hand (rigid, metric size) along its ray
            j = j + shift; v = v + shift
        else:
            r = np.ones(T)
        R_ = W[:, :3, :3]; t_ = W[:, :3, 3]
        jw = np.einsum('tij,tkj->tki', R_, j) + t_[:, None]; vw = np.einsum('tij,tkj->tki', R_, v) + t_[:, None]
        rw = np.einsum('tij,tkjl->tkil', R_, rot)
        if valid.any() and not valid.all():
            jw = interp_invalid(jw, valid); vw = interp_invalid(vw, valid)
            ix = np.flatnonzero(valid); near = ix[np.abs(np.arange(T)[:, None] - ix[None]).argmin(1)]
            rw = rw[near]
        res[f'{side}_joints_world'] = jw; res[f'{side}_wxyz_world'] = mat_to_quat(rw); res[f'{side}_verts_world'] = vw.astype(np.float32)
        res[f'{side}_faces'] = m[f'{side}_faces']; res[f'{side}_valid'] = valid
        res[f'{side}_joints_cam'] = j
        report['sides'][side] = {'valid_frames': int(valid.sum()), 'frames': T, 'hawor_scale_mean': float(s.mean()),
                                 'stereo_over_hand_depth_median': float(np.nanmedian(ratio[good])) if good.any() else None,
                                 'stereo_frames': int(good.sum()), 'applied_depth_ratio_range': [float(r.min()), float(r.max())]}
    np.savez_compressed(o / 'hands_world.npz', **res); dump(o / 'hands_map_report.json', report)
    print('mapped', e, json.dumps(report['sides']), flush=True)


def project(Pw, Wc, K):
    cam = (Pw - Wc[:3, 3]) @ Wc[:3, :3]; p = cam @ K.T
    return p[:, :2] / p[:, 2:3], cam[:, 2] > .01


def check_projection(bundle, e, frames=None):
    """Overlay mapped MANO joints and object mesh silhouettes on rgb.mp4 frames (alignment check)."""
    import cv2, trimesh
    o = out_dir(bundle, e); h = np.load(o / 'hands_world.npz'); rec = record(bundle, e)
    intr = load(bundle / ep_name(e) / 'intrinsics.json'); K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    poses = np.load(bundle / ep_name(e) / 'object_poses.npz'); W = poses['world_from_rgb']; T = len(W)
    meshes = [trimesh.load(bundle / ep_name(e) / 'meshes' / f'{n}.obj', force='mesh', process=False) for n in poses['names']]
    cap = cv2.VideoCapture(str(Path(rec['output']) / 'rgb.mp4')); frames = frames or [0, T // 4, T // 2, 3 * T // 4, T - 1]; tiles = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); assert ok
        for k, mesh in enumerate(meshes):
            Pw = trimesh.transform_points(mesh.vertices[::max(1, len(mesh.vertices) // 3000)], poses['world_from_object'][f, k])
            uv, ok2 = project(Pw, W[f], K)
            for p in uv[ok2].astype(int): cv2.circle(im, tuple(p), 1, (0, 255, 0) if k == 0 else (255, 0, 255), -1)
        for side, col in zip(SIDES, [(255, 160, 30), (30, 160, 255)]):
            uv, ok2 = project(h[f'{side}_joints_world'][f], W[f], K)
            for ch in CHAINS:
                for a, b in zip(ch, ch[1:]):
                    if ok2[a] and ok2[b]: cv2.line(im, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), col, 2, cv2.LINE_AA)
        cv2.putText(im, f'ep{e} f{f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (507, 380)))
    while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(str(o / 'projection_check.jpg'), sheet); print(o / 'projection_check.jpg')


# ----------------------------------------------------------------------------- ik
def ik(bundle, e):
    import importlib.util, torch
    from scipy.signal import savgol_filter
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('mink_sharpa_exact', MINK_EXACT); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    import mujoco
    o = out_dir(bundle, e); h = np.load(o / 'hands_world.npz'); T = len(h['world_from_rgb'])
    data = {'fps': np.array(20.)}; report = {'episode': e, 'bundle': str(bundle), 'solver': 'Mink floating Sharpa Wave (mink_sharpa_exact.MinkSharpa), max_iter 80', 'sides': {}}
    for side in SIDES:
        valid = h[f'{side}_valid']
        obj = mod.MinkSharpa(side=side, robot_asset_path=str(SHARPA_XML / f'{side}_sharpawave.xml'), source_model='mano', max_iter=80, frequency=200.)
        names = [mujoco.mj_id2name(obj.mm, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(obj.mm.njnt)]
        if not valid.any():
            data[f'{side}_qpos_mj'] = np.zeros((T, obj.mm.nq)); data[f'{side}_present'] = np.zeros(T, bool)
            data[f'{side}_mj_joint_names'] = np.array(names); report['sides'][side] = {'skipped': 'no valid frames'}; continue
        j = h[f'{side}_joints_world'].copy(); q = h[f'{side}_wxyz_world'].copy(); w = 9
        for t in range(1, T): q[t][(q[t] * q[t - 1]).sum(-1) < 0] *= -1
        if T > w: j = savgol_filter(j, w, 2, axis=0); q = savgol_filter(q, w, 2, axis=0)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        results = []; prev = None
        for t in range(T):
            if prev is None:
                prev = obj.robot.q0.copy(); prev[:3] = j[t, 0]; prev[3:7] = q[t, 0, [1, 2, 3, 0]]
            r = obj.compute(j[t], q[t], source_to_robot_scale=1., qpos=prev); prev = r['q']; results.append(r)
            if t % 50 == 0: print('IK', e, side, t, flush=True)
        pose = np.array([r['q'] for r in results])
        for t in range(1, T):
            if np.dot(pose[t, 3:7], pose[t - 1, 3:7]) < 0: pose[t, 3:7] *= -1
        if T > 11: pose = savgol_filter(pose, 11, 2, axis=0)
        pose[:, 3:7] /= np.linalg.norm(pose[:, 3:7], axis=1, keepdims=True)
        pose[:, 7:] = np.clip(pose[:, 7:], obj.robot.model.lowerPositionLimit[7:], obj.robot.model.upperPositionLimit[7:])
        qmj = np.array([obj.to_mj(p) for p in pose]); errs = np.array([r['frame_task_errors'] for r in results])
        data[f'{side}_qpos_mj'] = qmj; data[f'{side}_qpos_pin_xyzw'] = pose; data[f'{side}_present'] = np.ones(T, bool)
        data[f'{side}_valid'] = valid; data[f'{side}_mj_joint_names'] = np.array(names)
        data[f'{side}_frame_task_errors'] = errs; data[f'{side}_frame_task_names'] = np.array(list(obj.mt))
        report['sides'][side] = {'task_error_mean_m': float(errs.mean()), 'task_error_p95_m': float(np.percentile(errs, 95)),
                                 'wrist_error_mean_m': float(errs[:, list(obj.mt).index(side + '_hand_C_MC')].mean()), 'valid_frames': int(valid.sum())}
    data['qpos_layout'] = np.array('per side: MuJoCo qpos of <side>_sharpawave.xml + freejoint on root body: [x y z qw qx qy qz, finger joints in mj order]')
    np.savez_compressed(o / 'sharpa_ik_qpos.npz', **data); dump(o / 'ik_report.json', report)
    print('ik done', e, json.dumps(report['sides']), flush=True)


# ----------------------------------------------------------------------------- render
def sharpa_scene_xml(objects):
    """Two exact Sharpa Wave hands (floating) + mocap object meshes in one MJCF."""
    import xml.etree.ElementTree as ET
    root = ET.Element('mujoco', model='mink_sharpa_video'); ET.SubElement(root, 'compiler', angle='radian')
    vis = ET.SubElement(root, 'visual'); ET.SubElement(vis, 'global', offwidth='2048', offheight='2048')
    ET.SubElement(vis, 'headlight', ambient='.35 .35 .35', diffuse='.5 .5 .5', specular='.1 .1 .1')
    ET.SubElement(vis, 'quality', shadowsize='0')
    asset = ET.SubElement(root, 'asset'); default = ET.SubElement(root, 'default'); world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', directional='true', pos='0 0 3', dir='0 0 -1', diffuse='.5 .5 .5')
    ET.SubElement(world, 'camera', name='ego', pos='0 0 0', fovy='60')
    for side in SIDES:
        x = ET.parse(SHARPA_XML / f'{side}_sharpawave.xml').getroot(); meshdir = (SHARPA_XML / x.find('compiler').get('meshdir')).resolve()
        for el in x.iter():
            if el.get('class') is not None: el.set('class', f'{side}__{el.get("class")}')
            if el.tag == 'mesh' and el.get('file'): el.set('file', str(meshdir / el.get('file'))); el.set('name', f'{side}__{el.get("name")}')
            elif el.get('mesh') is not None: el.set('mesh', f'{side}__{el.get("mesh")}')
            if el.tag == 'geom' and el.get('group') == '1': el.set('rgba', '.42 .42 .48 1' if side == 'left' else '.5 .45 .4 1')
        for a in x.find('asset'): asset.append(a)
        for d in x.find('default'):
            if d.tag == 'default': default.append(d)
        body = x.find('worldbody/body'); body.insert(0, ET.Element('freejoint', name=f'{side}__floating_root'))
        world.append(body)
    palette = ['.85 .55 .25 1', '.35 .75 .45 1', '.6 .45 .85 1']
    for k, (name, path) in enumerate(objects):
        ET.SubElement(asset, 'mesh', name=f'obj_{k}', file=str(path))
        b = ET.SubElement(world, 'body', name=f'obj_{k}_{name}', mocap='true')
        ET.SubElement(b, 'geom', type='mesh', mesh=f'obj_{k}', rgba=palette[k % 3], contype='0', conaffinity='0', group='1')
    return ET.tostring(root, encoding='unicode')


def render(bundle, e, ghost=True, stride=1, max_frames=None):
    import mujoco, cv2, imageio_ffmpeg
    o = out_dir(bundle, e); rec = record(bundle, e); bdir = bundle / ep_name(e)
    poses = np.load(bdir / 'object_poses.npz'); intr = load(bdir / 'intrinsics.json'); W = poses['world_from_rgb'].astype(float); T = len(W)
    h = np.load(o / 'hands_world.npz'); ikd = np.load(o / 'sharpa_ik_qpos.npz')
    names = [str(n) for n in poses['names']]
    xml = sharpa_scene_xml([(n, bdir / 'meshes' / f'{n}.obj') for n in names])
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    # Symmetric-frustum render on a padded canvas, cropped so that (cx, cy) is the principal point.
    fx, fy, cx, cy, Wd, Ht = intr['fx'], intr['fy'], intr['cx'], intr['cy'], intr['width'], intr['height']
    hw = int(np.ceil(max(cx, Wd - cx))); hh = int(np.ceil(max(cy, Ht - cy)))
    RW, RH = 2 * hw, 2 * hh; f = .5 * (fx + fy)
    cam = model.cam('ego').id; model.cam_fovy[cam] = np.degrees(2 * np.arctan(hh / f))
    # aspect: MuJoCo uses fovy + viewport aspect, so horizontal focal == vertical focal == f.
    renderer = mujoco.Renderer(model, height=RH, width=RW)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    ox, oy = int(round(hw - cx)), int(round(hh - cy))
    # Map IK-model qpos (freejoint first, then hinges in order) onto the combined render model by joint name.
    qmaps = {}
    for side in SIDES:
        jn = [str(x) for x in ikd[f'{side}_mj_joint_names']]; assert jn[0] == 'floating_root', jn[:2]
        src = [0] + [6 + i for i in range(1, len(jn))]
        dst = [model.jnt_qposadr[model.joint(f'{side}__floating_root' if n == 'floating_root' else n).id] for n in jn]
        qmaps[side] = (np.array(src[1:]), np.array(dst[1:]), dst[0])
    cap = cv2.VideoCapture(str(Path(rec['output']) / 'rgb.mp4'))
    frames = list(range(0, T, stride))[:max_frames] if max_frames else list(range(0, T, stride))
    writer = imageio_ffmpeg.write_frames(str(o.parent / f'epi{e}.mp4') if not max_frames else str(o / f'preview_epi{e}.mp4'),
                                         (2 * Wd, Ht), fps=20 / stride, codec='libx264', quality=7, macro_block_size=2, pix_fmt_out='yuv420p')
    writer.send(None); sheet_frames = set(np.linspace(0, T - 1, 8).astype(int).tolist()); sheet = []
    flip = np.diag([1., -1., -1.])
    blue = np.array([.55, .8, 1., .35], np.float32)
    for t in frames:
        if stride != 1: cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, bgr = cap.read()
        if not ok: bgr = np.zeros((Ht, Wd, 3), np.uint8)
        for side in SIDES:
            qm = ikd[f'{side}_qpos_mj'][t]; srci, dsti, root = qmaps[side]
            data.qpos[root:root + 7] = qm[:7]; data.qpos[dsti] = qm[srci]
            if not bool(ikd[f'{side}_present'][t]): data.qpos[root:root + 7] = [0, 0, -50, 1, 0, 0, 0]
        for k in range(len(names)):
            M = poses['world_from_object'][t, k]; U, _, Vt = np.linalg.svd(M[:3, :3]); Rm = U @ Vt
            data.mocap_pos[k] = M[:3, 3]; q = np.zeros(4); mujoco.mju_mat2Quat(q, Rm.reshape(-1)); data.mocap_quat[k] = q
        Rc = W[t, :3, :3] @ flip; q = np.zeros(4); mujoco.mju_mat2Quat(q, Rc.reshape(-1))
        model.cam_pos[cam] = W[t, :3, 3]; model.cam_quat[cam] = q
        mujoco.mj_kinematics(model, data); mujoco.mj_camlight(model, data)
        renderer.update_scene(data, camera='ego', scene_option=opt)
        if ghost:
            scn = renderer.scene
            for side in SIDES:
                if not h[f'{side}_valid'][t]: continue
                J = h[f'{side}_joints_world'][t]
                for ch in CHAINS:
                    for a, b in zip(ch, ch[1:]):
                        if scn.ngeom >= scn.maxgeom: break
                        g = scn.geoms[scn.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), blue)
                        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, .007, J[a], J[b]); scn.ngeom += 1
        img = renderer.render()[oy:oy + Ht, ox:ox + Wd]
        right = img[:, :, ::-1].copy()
        cv2.putText(bgr, f'RGB  ep{e}  f{t}  {t / 20:.2f}s', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        lab = 'Sharpa Mink IK + objects' + ('  (blue: MANO)' if ghost else '')
        cv2.putText(right, lab, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        miss = [s_ for s_ in SIDES if not h[f'{s_}_valid'][t]]
        if miss: cv2.putText(right, 'unobserved (interpolated/hidden): ' + ','.join(miss), (12, Ht - 15), cv2.FONT_HERSHEY_SIMPLEX, .7, (80, 80, 255), 2, cv2.LINE_AA)
        frame = np.hstack([bgr, right]); writer.send(np.ascontiguousarray(frame[:, :, ::-1]))
        if t in sheet_frames: sheet.append(cv2.resize(frame, (Wd, Ht // 2)))
    writer.close(); renderer.close(); cap.release()
    if not max_frames and sheet:
        while len(sheet) % 2: sheet.append(np.zeros_like(sheet[0]))
        cv2.imwrite(str(o / 'contact_sheet.jpg'), np.vstack([np.hstack(sheet[i:i + 2]) for i in range(0, len(sheet), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('rendered', e, flush=True)




# ----------------------------------------------------------------------------- docker / main
def docker(phase, bundle, eps, extra):
    image = 'v2d_task_library_loader:latest' if phase == 'decode' else 'robotic-grounding:mink-1.3.0'
    entry = 'python' if phase == 'decode' else '/workspace/isaaclab/_isaac_sim/python.sh'
    cmd = ['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}', '--group-add', '1234', '-e', 'HOME=/tmp',
           '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'PYTHONUNBUFFERED=1', '-e', 'OMP_NUM_THREADS=2',
           '-e', f'MINK_VIDEO_OUT={OUT_ROOT}', '--entrypoint', entry, '-v', '/data1:/data1']
    if phase == 'ik':
        cmd += ['-v', str(ROOT / 'robotic_grounding') + ':/workspace/video_to_data/robotic_grounding', '-w', '/workspace/video_to_data/robotic_grounding']
    cmd += [image, str(Path(__file__).resolve()), '--phase', phase, '--bundle', str(bundle), '--in-container', '--episodes', *map(str, eps)] + extra
    subprocess.run(['sg', 'docker', '-c', shlex.join(cmd)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--phase', choices=['decode', 'map', 'check', 'ik', 'render', 'preview'], required=True)
    p.add_argument('--episodes', nargs='*', default=None)
    p.add_argument('--map-mode', choices=['naive', 'anycalib'], default='anycalib')
    p.add_argument('--depth', choices=['hawor', 'metric', 'stereo'], default='stereo')
    p.add_argument('--no-ghost', action='store_true'); p.add_argument('--in-container', action='store_true')
    a = p.parse_args(); bundle = a.bundle.resolve(); eps = episodes_of(bundle, a.episodes)
    if a.phase in ('decode', 'ik') and not a.in_container:
        return docker(a.phase, bundle, eps, [])
    failed = []
    for e in eps:
        try:
            if a.phase == 'decode': decode(bundle, e)
            elif a.phase == 'map': map_hands(bundle, e, a.map_mode, a.depth); check_projection(bundle, e)
            elif a.phase == 'check': check_projection(bundle, e)
            elif a.phase == 'ik': ik(bundle, e)
            elif a.phase == 'render': render(bundle, e, ghost=not a.no_ghost)
            elif a.phase == 'preview': render(bundle, e, ghost=not a.no_ghost, max_frames=40)
        except Exception as ex:  # keep going over episodes; report failures at the end
            import traceback; traceback.print_exc(); failed.append(e)
    if failed: print('FAILED', a.phase, failed, flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    # EGL teardown can hang on node100; exit hard once outputs are flushed.
    os._exit(1 if failed else 0)


if __name__ == '__main__':
    main()
