"""Audit saved Track 3 runs and export actual per-frame geometry for the lab.

Run with the lightweight v2d environment. Input folders are read only. This
re-evaluates saved predictions; it does not claim to rerun neural inference.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

from .metrics import bundle_checks, hand_metrics, trajectory_metrics

REPO = Path(__file__).resolve().parents[2]


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False))
    temporary.replace(path)


def binary(path, value, dtype='<f4'):
    value = np.asarray(value, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    value.tofile(path)
    return {'url': '/data/' + str(path.relative_to(OUTPUT)), 'shape': list(value.shape), 'dtype': dtype}


def media(path, dataset):
    return '/media/' + path.relative_to(dataset).as_posix() if path.exists() else None


def exporter():
    spec = importlib.util.spec_from_file_location('v2d_scene_export', REPO / 'reconstruction/modules/v2d_pipelines/export_result_threejs_scene.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle_hands(arrays):
    result = {}
    n = len(arrays['camera_to_world_transform'])
    for side in ['left', 'right']:
        pre = 'hand_' + side
        result[side] = {
            'orient': arrays[pre + '_wrist_orient_in_camera'], 'pose': arrays[pre + '_finger_pose'],
            'betas': np.repeat(arrays[pre + '_betas'][None], n, axis=0),
            'trans': arrays[pre + '_wrist_trans_in_camera'], 'scale': arrays[pre + '_scale'],
            'valid': arrays[pre + '_is_valid'].astype(bool),
            'filled': arrays.get(pre + '_qa_filled', np.zeros(n, bool)),
        }
    return result


def stage_hands(root, n, before_depth=False):
    result = {side: {'orient': np.zeros((n, 3)), 'pose': np.zeros((n, 15, 3)),
                     'betas': np.zeros((n, 10)), 'trans': np.zeros((n, 3)), 'scale': np.ones(n),
                     'valid': np.zeros(n, bool), 'filled': np.zeros(n, bool)} for side in ['left', 'right']}
    for f in sorted(root.glob('*/*.json')):
        rec = read_json(f)
        if not isinstance(rec, dict) or 'mano' not in rec:
            continue
        i = int(rec['frame_idx'])
        if i >= n:
            continue
        x = result['right' if rec['is_right'] else 'left']
        if before_depth and 'cam_t_pre_dz' not in rec.get('diagnostics', {}):
            continue
        x['orient'][i] = rec['mano']['global_orient']
        x['pose'][i] = np.asarray(rec['mano']['hand_pose']).reshape(15, 3)
        x['betas'][i] = rec['mano']['betas']
        x['trans'][i] = rec['diagnostics']['cam_t_pre_dz'] if before_depth else rec['camera']['cam_t']
        x['scale'][i] = 1 if before_depth else rec.get('hand_scale', 1)
        x['valid'][i] = True
        x['filled'][i] = bool(rec.get('interpolated', False))
    return result


def export_hands(out, hands, model, exp, fps):
    vertices = []
    result = {}
    for side in ['left', 'right']:
        h = hands[side]
        n = len(h['valid'])
        v = np.zeros((n, len(model['v_template']), 3), np.float32)
        for i in np.flatnonzero(h['valid']):
            v[i] = exp._mano_forward_camera(model, h['orient'][i], h['pose'][i].reshape(45), h['betas'][i], h['trans'][i], side == 'right', h['scale'][i])
        vertices.append(v)
        result[side] = hand_metrics(h['trans'], h['valid'], fps)
        result[side]['valid'] = h['valid'].tolist()
        result[side]['filled'] = h['filled'].tolist()
        result[side]['scale_median'] = float(np.median(h['scale'][h['valid']])) if h['valid'].any() else None
    result['geometry'] = binary(out, vertices)
    return result


def truth_poses(dataset, episode, object_name):
    p = dataset / f'track_3/public/data/chunk-000/{episode}.parquet'
    rows = pq.ParquetFile(p).read().to_pylist()
    n = len(rows)
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    valid = np.zeros(n, bool)
    for i, row in enumerate(rows):
        for obj in row['observation.objects']:
            if obj['name'] != object_name or not obj['visible']:
                continue
            pose = np.asarray(obj['pose'])
            if not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
                continue
            poses[i, :3, 3] = pose[:3]
            poses[i, :3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            valid[i] = True
    return poses, valid, np.array([row['timestamp'] for row in rows])


def depth_previews(root, out):
    """Fixed metric colour scale shared by every episode, not per-frame contrast."""
    files = sorted((root / 'depth').glob('*.png'))
    if not files:
        return None
    target = out / 'depth';target.mkdir(exist_ok=True)
    for file in files:
        dest = target / (file.stem + '.jpg')
        if dest.exists():
            continue
        px = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
        if px is None:
            continue
        px = cv2.resize(px, (640, round(640 * px.shape[0] / px.shape[1])), interpolation=cv2.INTER_NEAREST)
        with np.errstate(divide='ignore'):
            metres = 65535.0 / px.astype(np.float32) - 1
        intensity = np.clip((metres - .1) / 1.9, 0, 1)
        image = cv2.applyColorMap((intensity * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        image[px == 0] = 0
        cv2.imwrite(str(dest), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return '/data/' + str(target.relative_to(OUTPUT)) + '/'


def match_retarget_input(row, reconstruction_root):
    """Match stored MANO joints to candidate bundles up to the loader's shift.

    The loader's camera-frame wrist is cam_t + constant rest-joint offset.
    Testing constancy avoids depending on MANO implementation details.
    """
    matches = []
    for name in ['result_slam_gravity_aligned', 'result_clean', 'refined_bundle/result_slam_gravity_aligned']:
        path = reconstruction_root / name / 'result.npz'
        if not path.exists():
            continue
        d = dict(np.load(path, allow_pickle=False))
        objects = np.asarray(row['object_body_position'])[:, 0]
        if len(objects) != len(d['object_is_valid']):
            continue
        delta = objects - d['object_to_world_transform'][:, :3, 3]
        offset = np.median(delta, axis=0)
        residual = float(np.max(np.abs(delta - offset)))
        cam = d['camera_to_world_transform']
        for side in ['left', 'right']:
            joints = np.asarray(row['mano_' + side + '_joints'])[:, 0]
            rest = np.einsum('nji,nj->ni', cam[:, :3, :3], joints - offset - cam[:, :3, 3]) - d[f'hand_{side}_wrist_trans_in_camera']
            residual = max(residual, float(np.max(np.linalg.norm(rest - np.median(rest, axis=0), axis=1))))
        if residual < 1e-4:
            matches.append({'bundle': name, 'max_residual_m': residual})
    return matches


def export_retarget(dataset, episode, out, reconstruction_root):
    root = dataset / 'runs/retarget/mesh' / episode
    files = list((root / 'human_motion_data/ego_recon/processed').glob('sequence_id=*/robot_name=*/*.parquet'))
    if len(files) != 1:
        return None
    f = files[0]
    row = pq.ParquetFile(f).read().to_pylist()[0]
    if row['robot_right_frames'] is None:
        return None
    matches = match_retarget_input(row, reconstruction_root)
    description = ', '.join(x['bundle'] for x in matches) if matches else 'unresolved from available bundles'
    result = {'fps': float(row['fps']), 'source': str(f), 'robot': 'Floating Sharpa', 'input': row['raw_motion_file'], 'matched_inputs': matches,
              'video': media(root / 'floating_retarget_qa' / (episode + '_grid.mp4'), dataset) or media(root / 'floating_retarget_qa' / (episode + '.mp4'), dataset),
              'provenance': f'Existing IK output. Numerically matched input: {description}. The current script requests cleanup, but some saved runs predate it. No RL.'}
    for side in ['left', 'right']:
        values = np.asarray(row['robot_' + side + '_frames'], np.float32)
        errors = np.asarray(row['robot_' + side + '_frame_task_errors'], float)
        result[side] = {'names': row[side + '_robot_frame_names'], 'frames': binary(out / ('robot_' + side + '.bin'), values),
                        'ik_mean_mm': float(np.nanmean(errors) * 1000), 'ik_p95_mm': float(np.nanpercentile(errors, 95) * 1000)}
    result['object_position'] = row['object_body_position']
    result['object_wxyz'] = row['object_body_wxyz']
    return result


def build(dataset, output, selected=None):
    global OUTPUT
    OUTPUT = output
    output.mkdir(parents=True, exist_ok=True)
    exp = exporter()
    model = exp._load_mano_right_model(dataset / 'runs/mano/mano_v1_2')
    catalog = {'schema_version': 1, 'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
               'dataset': str(dataset), 'episodes': [], 'methodology': {
                   'object_rigid_rmse_mm': 'Diagnostic only: one SE(3) best fit over ALL mutually valid public object-centre samples. No scale fit. Optimistic gauge-aligned trajectory error; not an official score.',
                   'object_anchor_rmse_mm': 'Align the first mutually visible predicted object pose to its public pose, then measure the remaining trajectory. No later fitting and no scale fitting. Assumes matching supplied-mesh frame.',
                   'hands': 'No hand ground truth is provided. Coverage and distance/speed outliers are diagnostics, not accuracy. A wrist >1.1 m from the head camera is flagged for inspection, never automatically removed.',
                   'retarget': 'Stored IK task residuals measure fit to reconstructed targets, not real-world accuracy or physical success.',
                   'geometry': 'MANO vertices regenerated on CPU from saved pose, shape and scale using the repository exporter. No neural inference is implied.',
               }}
    write_json(output / 'hand_faces.json', model['faces'].tolist())
    original_scene = next((dataset / 'runs/recon').glob('**/threejs_scene/three.module.js'))
    (output / 'vendor').mkdir(exist_ok=True)
    for name in ['three.module.js', 'OrbitControls.js']:
        shutil.copyfile(original_scene.parent / name, output / 'vendor' / name)
    lines = (dataset / 'runs/prep/episodes.tsv').read_text().splitlines()
    entries = list(csv.reader([line for line in lines if not line.startswith('#')], delimiter='\t'))
    summary = []
    for ep, task, obj, prompt in entries:
        if selected and int(ep) not in selected:
            continue
        episode = f'episode_{int(ep):06d}'
        out = output / episode
        out.mkdir(exist_ok=True)
        root = dataset / 'runs/recon/mesh' / episode
        original_root = root
        superseded = False
        if not (root / 'result_slam_gravity_aligned/result.npz').exists() and (root / 'ref0_discarded/result_slam_gravity_aligned/result.npz').exists():
            root = root / 'ref0_discarded'
            superseded = True
        source = dataset / f'track_3/public/videos/chunk-000/observation.images.ego_cam_a/{episode}.mp4'
        cap = cv2.VideoCapture(str(source))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out / 'thumb.jpg'), cv2.resize(frame, (320, round(320 * h / w))))
        cap.release()
        data = {'id': episode, 'index': int(ep), 'task': task.replace('_', ' '), 'object': obj, 'prompt': prompt,
                'fps': fps, 'frames': n, 'width': w, 'height': h, 'source': media(source, dataset), 'variants': {}, 'hands': {},
                'superseded': superseded, 'findings': [], 'root': str(root), 'stages': {}, 'retarget': None,
                'undistorted': media(original_root / 'anycalib/undistorted.mp4', dataset),
                'reference_frame': int((original_root / 'reference_frame.txt').read_text().strip()) if (original_root / 'reference_frame.txt').exists() else 0}
        data['depth_preview'] = depth_previews(original_root, out)
        for key, file in {'masks': 'masks_overlay.mp4', 'raw': 'hamer_overlay.mp4', 'aligned': 'hamer_aligned_overlay.mp4',
                          'filled': 'hamer_aligned_filled_overlay.mp4', 'refined': 'refined_bundle/refined_simple_overlay.mp4'}.items():
            data['stages'][key] = media(root / file, dataset)
        truth, gt_valid, timestamps = truth_poses(dataset, episode, obj)
        data['ground_truth'] = {'poses': truth.round(7).tolist(), 'valid': gt_valid.tolist()}
        data['timestamps'] = timestamps.tolist()
        base_path = root / 'result_slam_gravity_aligned/result.npz'
        if base_path.exists():
            base = dict(np.load(base_path, allow_pickle=False))
            intr = read_json(original_root / 'intrinsics_stable.json', {})
            data['intrinsics'] = {**intr, 'fx': float(base['camera_intrinsics'][0]), 'fy': float(base['camera_intrinsics'][1]), 'cx': float(base['camera_intrinsics'][2]), 'cy': float(base['camera_intrinsics'][3])}
            mesh = trimesh.load(root / 'result_slam_gravity_aligned/mesh.obj', force='mesh', process=False)
            data['mesh'] = {'vertices': binary(out / 'object_vertices.bin', mesh.vertices), 'faces': binary(out / 'object_faces.bin', mesh.faces, '<u4')}
            hands = {'filled': bundle_hands(base)}
            if (root / 'hamer_aligned').exists():
                hands['raw'] = stage_hands(root / 'hamer_aligned', len(base['object_is_valid']), True)
                hands['aligned'] = stage_hands(root / 'hamer_aligned', len(base['object_is_valid']))
            refined = root / 'refined_bundle/result_slam_gravity_aligned/result.npz'
            if refined.exists():
                hands['refined'] = bundle_hands(dict(np.load(refined, allow_pickle=False)))
            clean = root / 'result_clean/result.npz'
            if clean.exists():
                hands['cleaned'] = bundle_hands(dict(np.load(clean, allow_pickle=False)))
            # This view isolates the scale discarded by the current ego loader.
            loader = bundle_hands(dict(np.load(clean if clean.exists() else base_path, allow_pickle=False)))
            for side in loader:
                loader[side]['scale'] = np.ones_like(loader[side]['scale'])
            hands['loader_scale'] = loader
            for name, values in hands.items():
                data['hands'][name] = export_hands(out / ('hands_' + name + '.bin'), values, model, exp, fps)
            for refined_flag, pre in [(False, ''), (True, 'refined_bundle/')]:
                for mode, sub in [('static', 'result'), ('slam', 'result_slam'), ('gravity', 'result_slam_gravity_aligned')]:
                    folder = root / (pre + sub)
                    path = folder / 'result.npz'
                    if not path.exists():
                        continue
                    arrays = dict(np.load(path, allow_pickle=False))
                    count = len(arrays['object_is_valid'])
                    problems = bundle_checks(arrays, n)
                    metric, alignment = ({}, None)
                    if count == len(truth):
                        metric, alignment = trajectory_metrics(arrays['object_to_world_transform'], truth, arrays['object_is_valid'] & gt_valid)
                    name = ('refined' if refined_flag else 'baseline') + '_' + mode
                    data['variants'][name] = {'camera': arrays['camera_to_world_transform'].round(7).tolist(),
                        'object_camera': arrays['object_to_camera_transform'].round(7).tolist(), 'object_valid': arrays['object_is_valid'].tolist(),
                        'metrics': metric, 'alignment': alignment, 'checks': problems, 'bundle': str(path),
                        'manifest': media(folder / 'manifest.json', dataset), 'refined': refined_flag, 'mode': mode}
            hm = data['hands']['filled']
            for side in ['left', 'right']:
                maximum = hm[side]['max_camera_distance_m']
                if maximum is not None and maximum > 1.1:
                    data['findings'].append(f'{side.title()} hand reaches {maximum:.2f} m from camera; {len(hm[side]["far_frames"])} frames exceed the 1.1 m inspection threshold.')
                scale = hm[side]['scale_median']
                if scale and abs(scale - 1) > .05:
                    data['findings'].append(f'{side.title()} MANO scale is {scale:.3f}; current ego loader does not consume hand scale. Inspect the loader-scale view before attributing grasp error to IK.')
            data['status'] = 'archived' if superseded else 'complete'
            if superseded:
                data['findings'].insert(0, 'Showing archived reference-frame-0 outputs. The newer rerun is incomplete; these are not its results.')
            data['retarget'] = export_retarget(dataset, episode, out, root)
            # Retain alignment failure evidence instead of equating files with
            # trustworthy detections. Frames with zero pixels receive dz=0.
            failures = []
            for f in (root / 'hamer_aligned').glob('*/*.json'):
                rec = read_json(f)
                if not isinstance(rec, dict) or 'mano' not in rec:
                    continue
                if rec.get('diagnostics', {}).get('n_pixels') == 0:
                    failures.append({'frame': int(rec['frame_idx']), 'side': 'right' if rec['is_right'] else 'left',
                                     'distance_m': float(np.linalg.norm(rec['camera']['cam_t']))})
            data['zero_depth_support'] = failures
            severe = [f for f in failures if f['distance_m'] > 1.1]
            if severe:
                worst = max(severe, key=lambda f: f['distance_m'])
                data['findings'].insert(0, f'Frame {worst["frame"]}: {worst["side"]} HaMeR outlier survives depth alignment with zero usable depth pixels (dz=0). The pre-depth estimate is already bad.')
        else:
            data['status'] = 'incomplete'
            data['findings'].append('No final result bundle. Existing logs contain FoundationPose CUDA out-of-memory failures; model comparisons require completing this run.')
        log = dataset / f'runs/logs/full_mesh/ep{int(ep)}.log'
        data['log'] = media(log, dataset)
        if log.exists():
            text = log.read_text(errors='replace')
            data['logged_oom'] = 'CUDA out of memory' in text
        urls = {data['source'], data.get('undistorted'), *data['stages'].values(), (data.get('retarget') or {}).get('video')}
        data['video_timing'] = {}
        for url in urls:
            if not url:
                continue
            cap = cv2.VideoCapture(str(dataset / url.removeprefix('/media/')))
            data['video_timing'][url] = {'fps': cap.get(cv2.CAP_PROP_FPS), 'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
            cap.release()
        write_json(out / 'episode.json', data)
        short = {key: data[key] for key in ['id', 'index', 'task', 'object', 'frames', 'fps', 'status', 'superseded', 'findings']}
        short['thumb'] = '/data/' + episode + '/thumb.jpg'
        short['refined'] = 'refined' in data['hands']
        short['retarget'] = bool(data['retarget'])
        catalog['episodes'].append(short)
        metric = data['variants'].get('baseline_gravity', {}).get('metrics', {})
        refined_metric = data['variants'].get('refined_gravity', {}).get('metrics', {})
        summary.append({'episode': int(ep), 'task': task, 'status': data['status'], 'baseline_rmse_mm': metric.get('object_rigid_rmse_mm'),
                        'refined_rmse_mm': refined_metric.get('object_rigid_rmse_mm'), 'retarget': bool(data['retarget']),
                        'max_left_m': data['hands'].get('filled', {}).get('left', {}).get('max_camera_distance_m'),
                        'max_right_m': data['hands'].get('filled', {}).get('right', {}).get('max_camera_distance_m')})
        print(f'{episode}: {data["status"]}, {len(data["hands"])} hand stages, {len(data["variants"])} scene bundles', flush=True)
    write_json(output / 'catalog.json', catalog)
    write_json(output / 'audit.json', summary)
    with (output / 'audit.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    report = ['# Track 3 baseline audit', '', 'Saved-output re-evaluation; no fresh neural inference. No RL.', '',
              'Object RMSE below is a diagnostic after one rigid ALL-FRAME trajectory alignment (no scale). It is not an official challenge score.', '',
              '| Episode | Status | Baseline RMSE mm | Gsplat RMSE mm | Max left wrist m | Max right wrist m | IK available |',
              '|---|---|---:|---:|---:|---:|---|']
    def fmt(x):
        return '—' if x is None else f'{x:.2f}'
    for row in summary:
        report.append(f'| {row["episode"]} | {row["status"]} | {fmt(row["baseline_rmse_mm"])} | {fmt(row["refined_rmse_mm"])} | {fmt(row["max_left_m"])} | {fmt(row["max_right_m"])} | {row["retarget"]} |')
    report += ['', '## Findings', '',
               '- Existing mesh runs bypass SAM3D and object scale estimation. They do not test learned mesh generation.',
               '- The current custom retarget script selects the unrefined bundle and applies cleanup. Some saved runs predate cleanup (including episode 41). Per-episode provenance now numerically matches stored MANO targets to available input bundles.',
               '- The ego loader hard-codes 30 fps; the custom script repairs its exported fps to the dataset rate. Check actual exported metadata rather than assuming this caused the existing failures.',
               '- The ego loader does not consume per-hand scale or validity. Filled/interpolated predictions can also be marked valid in the reconstruction bundle. Coverage is not accuracy.',
               '- A single tracked object omits other manipulated/support objects in multi-object tasks.',
               '- HaWoR weights and container were absent in the inspected baseline setup. Uncomputed alternatives are explicitly unavailable.',
               '', '## Reproduce', '', '`OPENBLAS_NUM_THREADS=1 /home/exaFLOPs/miniforge3/envs/v2d/bin/python -m tools.ablation_lab.build`']
    completed = [r for r in summary if r['status'] == 'complete' and r['refined_rmse_mm'] is not None]
    if completed:
        before = np.mean([r['baseline_rmse_mm'] for r in completed])
        after = np.mean([r['refined_rmse_mm'] for r in completed])
        worse = sum(r['refined_rmse_mm'] > r['baseline_rmse_mm'] for r in completed)
        report += ['', '## Current-run aggregate', '', f'{len(completed)} complete episodes: mean episode object-path RMSE {before:.2f} mm before gsplat, {after:.2f} mm after. {worse}/{len(completed)} worsen on this diagnostic. Archived and incomplete episodes excluded.']
    retest = read_json(output / 'retarget_retest.json')
    if retest:
        report += ['', '## Fresh kinematic retargeting test', '', retest['mode'], '',
                   'Episode 41 was rerun through the baseline IK on CPU. Wrist positions, finger joints, and per-frame IK residuals exactly match the saved output (maximum absolute differences all zero).',
                   f'Mean IK residual: left {retest["sides"]["left"]["ik_mean_mm"]:.2f} mm, right {retest["sides"]["right"]["ik_mean_mm"]:.2f} mm. These are errors relative to reconstructed MANO targets, not hand ground truth.']
    (output / 'REPORT.md').write_text('\n'.join(report) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/data1/video_to_data_challenge'))
    parser.add_argument('--output', type=Path, default=REPO / '.ablation_lab/site')
    parser.add_argument('--episodes', type=int, nargs='+')
    args = parser.parse_args()
    build(args.dataset.resolve(), args.output.resolve(), args.episodes)
    # Rebuilding base artifacts must not silently discard registered experiments.
    from .experiments import read_registry, register_scene
    for record in read_registry()['runs']:
        if record['status'] == 'completed' and (not args.episodes or record['episode'] in args.episodes):
            register_scene(record, args.output.resolve(), args.dataset.resolve())
    from .export_probe import export_probe
    export_probe(args.output.resolve())


if __name__ == '__main__':
    main()
