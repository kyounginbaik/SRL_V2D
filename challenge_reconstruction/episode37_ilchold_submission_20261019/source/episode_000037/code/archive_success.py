"""Archive genuine full-horizon VOC0 rollouts passing a local MP-SR proxy.

No simulation or trajectory synthesis occurs here. Public-grid world0 contains
one selected complete rollout, with its original world ID retained in provenance.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def valid_poses(poses):
    return (np.isfinite(poses).all()
            and np.all(np.abs(np.linalg.norm(poses[..., 3:], axis=-1) - 1.) < 1e-3))


def orientation_error(actual, reference, symmetry=None):
    """Radians; symmetries act in the declared object's local coordinates."""
    a = Rotation.from_quat(actual)
    r = Rotation.from_quat(reference)
    symmetry = symmetry or {'type': 'none'}
    kind = symmetry['type']
    if kind == 'none':
        return (r.inv() * a).magnitude()
    if kind == 'continuous':
        axis = np.asarray(symmetry['axis'], dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
            raise ValueError('Invalid declared symmetry axis')
        axis /= np.linalg.norm(axis)
        dot = np.sum(a.apply(axis) * r.apply(axis), axis=-1)
        if symmetry.get('axis_sign_equivalent', False):
            dot = np.abs(dot)
        return np.arccos(np.clip(dot, -1., 1.))
    if kind == 'discrete':
        qs = np.asarray(symmetry['quaternions_xyzw'], dtype=float)
        if qs.ndim != 2 or qs.shape[1] != 4 or not len(qs) or not np.isfinite(qs).all() or not np.allclose(np.linalg.norm(qs, axis=1), 1., atol=1e-6):
            raise ValueError('Invalid declared symmetry rotations')
        # Identity is always admissible; declarations cannot make scoring stricter.
        errors = [(r.inv() * a).magnitude()]
        errors += [(r.inv() * a * Rotation.from_quat(q)).magnitude() for q in qs]
        return np.min(errors, axis=0)
    raise ValueError(f'Unknown symmetry type: {kind}')


def score_worlds(actual, reference, names, symmetries=None):
    """Every object's time-mean errors must pass strict 3cm/30deg thresholds."""
    actual, reference = np.asarray(actual, float), np.asarray(reference, float)
    if actual.ndim != 4 or actual.shape[-1] != 7 or min(actual.shape[:3]) < 1:
        raise ValueError('actual must have nonempty [T,W,O,7] shape')
    if reference.shape != (actual.shape[0], actual.shape[2], 7):
        raise ValueError('reference shape mismatch')
    if len(names) != actual.shape[2] or len(set(names)) != len(names) or not valid_poses(reference):
        raise ValueError('Invalid object roster or canonical reference poses')
    result = []
    for world in range(actual.shape[1]):
        poses = actual[:, world]
        if not valid_poses(poses):
            result.append({'world_id': world, 'valid': False, 'passed': False,
                           'reason': 'Nonfinite pose or nonunit quaternion'})
            continue
        # Conservative local-origin alignment. The latest host additionally registers
        # meshes to hidden anchors/axes; those transformations cannot be reproduced here.
        axes=[]
        for name in names:
            declaration=(symmetries or {}).get(name,{'type':'none'})
            axis=np.asarray(declaration.get('axis',[0.,0.,0.]),dtype=float)
            if declaration['type']=='none':axis=np.zeros(3)
            if axis.shape!=(3,) or not np.isfinite(axis).all():raise ValueError('Invalid symmetry axis')
            if np.any(axis):axis=axis/np.linalg.norm(axis)
            axes.append(axis)
        fixed=[i for i,axis in enumerate(axes) if not np.any(axis)]
        if fixed or len(names)==1:
            anchor=fixed[0] if fixed else 0
            align=Rotation.from_quat(reference[0,anchor,3:])*Rotation.from_quat(poses[0,anchor,3:]).inv()
        else:
            def basis(axis,direction):
                direction=direction-axis*np.dot(axis,direction)
                if np.linalg.norm(direction)<1e-8:
                    direction=np.eye(3)[np.argmin(np.abs(axis))];direction-=axis*np.dot(axis,direction)
                direction/=np.linalg.norm(direction)
                return np.column_stack((direction,np.cross(axis,direction),axis))
            ar=Rotation.from_quat(reference[0,0,3:]).apply(axes[0])
            aa=Rotation.from_quat(poses[0,0,3:]).apply(axes[0])
            align=Rotation.from_matrix(basis(ar,reference[0,1,:3]-reference[0,0,:3])@basis(aa,poses[0,1,:3]-poses[0,0,:3]).T)
        positions = align.apply((poses[..., :3] - poses[0, 0, :3]).reshape(-1, 3)).reshape(poses.shape[:2] + (3,)) + reference[0, 0, :3]
        rotations = (align * Rotation.from_quat(poses[..., 3:].reshape(-1, 4))).as_quat().reshape(poses.shape[:2] + (4,))
        errors = []
        for obj, name in enumerate(names):
            symmetry = (symmetries or {}).get(name, {'type': 'none', 'reason': 'No verified declaration; conservative fallback'})
            position = float(np.linalg.norm(positions[:, obj] - reference[:, obj, :3], axis=-1).mean())
            angle = float(np.rad2deg(orientation_error(rotations[:, obj], reference[:, obj, 3:], symmetry)).mean())
            errors.append({'object': name, 'mean_position_m': position,
                           'mean_orientation_deg': angle, 'symmetry': symmetry,
                           'passed': position < .03 and angle < 30.})
        result.append({'world_id': world, 'valid': True,
                       'passed': all(e['passed'] for e in errors), 'objects': errors,
                       'selection_cost': max(max(e['mean_position_m']/.03, e['mean_orientation_deg']/30.) for e in errors)})
    return result


def checked_file(manifest, key):
    path = Path(manifest[key + '_path'])
    expected = manifest[key + '_sha256']
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f'{key} is missing or changed')
    return path


def validate_input(trajectory, manifest, contract):
    import pyarrow.parquet as pq
    for key in ('voc_scale', 'voc_force_limit', 'voc_torque_limit', 'reset_count_after_start', 'reset_assist_steps'):
        if key not in manifest or manifest[key] != 0:
            raise ValueError(f'Unassisted continuous rollout requires {key}=0')
    if manifest.get('full_horizon') is not True or manifest.get('object_pose_forcing') is not False or manifest.get('auto_reset') is not False:
        raise ValueError('Require full horizon, no object forcing, no automatic resets')
    checkpoint = checked_file(manifest, 'checkpoint')
    source = checked_file(manifest, 'source_reference')
    if manifest.get('reference_snapshot_path'):
        checked_file(manifest, 'reference_snapshot')
    episode = manifest['episode']
    roster = contract['episodes'][episode]
    with np.load(trajectory, allow_pickle=False) as data:
        actual, reference, frames = data['actual'].copy(), data['reference'].copy(), data['frame_indices'].copy()
        if 'object_names' in data and data['object_names'].tolist() != manifest['object_names']:
            raise ValueError('NPZ object names disagree with manifest')
        if 'fps' in data and float(data['fps']) != float(manifest['fps']):
            raise ValueError('NPZ FPS disagrees with manifest')
    if manifest['object_names'] != roster['objects'] or actual.ndim != 4 or actual.shape[0] != roster['frames']:
        raise ValueError('Episode names/frame count do not match public roster')
    if frames.dtype.kind not in 'iu' or frames.shape != (roster['frames'],) or not np.array_equal(frames, np.arange(roster['frames'])):
        raise ValueError('Require every original frame exactly once, starting at0')
    if manifest.get('frames') != len(frames) or manifest.get('world_count') != actual.shape[1] or len(manifest.get('world_seeds', [])) != actual.shape[1]:
        raise ValueError('Manifest frame/world/seed counts disagree with trajectory')
    row = pq.read_table(source).to_pylist()[0]
    canonical = np.concatenate((np.asarray(row['object_body_position']), np.asarray(row['object_body_wxyz'])[..., [1, 2, 3, 0]]), axis=-1)
    if row['safe_object_body_names'] != roster['objects'] or float(row['fps']) != float(manifest['fps']):
        raise ValueError('Canonical reference names/FPS mismatch')
    if reference.shape != canonical.shape or not np.allclose(reference[..., :3], canonical[..., :3], atol=2e-6, rtol=0) or not np.allclose(np.abs(np.sum(reference[..., 3:]*canonical[..., 3:], axis=-1)), 1., atol=2e-6, rtol=0):
        raise ValueError('Exported reference differs from canonical parquet')
    return actual, reference, checkpoint, source, row


def _export(path, actual, reference, manifest, source, exporter_path):
    spec = importlib.util.spec_from_file_location('archive_trajectory_exporter', exporter_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rollout = SimpleNamespace(achieved_pose_w=actual[:, None], reference_pose_w=reference,
        object_body_names=tuple(manifest['object_names']), body_object_ids=np.arange(actual.shape[1]),
        non_finite_worlds=[], start_frame=0, control_fps=float(manifest['fps']))
    script = Path(manifest.get('evaluation_script_path') or Path(__file__).with_name('render_parallel_checkpoint.py'))
    if not script.is_file():
        raise ValueError('Actual evaluation script is required for export provenance')
    if manifest.get('evaluation_script_sha256') and digest(script) != manifest['evaluation_script_sha256']:
        raise ValueError('Evaluation script changed')
    code_hash = manifest.get('eval_code_sha256')
    if not code_hash:
        # Honest fallback identifies the actual exporter+renderer; never official pinned code.
        code_hash = hashlib.sha256((digest(script)+digest(exporter_path)).encode()).hexdigest()
    module.write_object_trajectories_parquet(path, rollout,
        checkpoint_sha256=manifest['checkpoint_sha256'], eval_script_sha256=digest(script),
        eval_code_sha256=code_hash, reference_parquet=source)


def archive(trajectory, manifest_path, archive_root, *, video=None, contract_path=None, exporter_path=None):
    trajectory, manifest_path, archive_root = map(Path, (trajectory, manifest_path, archive_root))
    base = Path(__file__).resolve().parents[1]
    contract_path = Path(contract_path or base/'reproduction/archive_public_contract.json')
    exporter_path = Path(exporter_path or base/'flash_chord/src/flash_chord/evaluation/submission.py')
    manifest = json.loads(manifest_path.read_text()); contract = json.loads(contract_path.read_text())
    actual, reference, checkpoint, source, row = validate_input(trajectory, manifest, contract)
    symmetries=contract.get('symmetries',{})
    symmetry_path=base/'reproduction/proxy_symmetry.json'
    proxy_symmetry=None
    if symmetry_path.exists():
        proxy_symmetry=json.loads(symmetry_path.read_text())
        for name,item in proxy_symmetry['objects'].items():
            if item.get('source_mesh_path') and digest(item['source_mesh_path'])!=item['source_mesh_sha256']:
                raise ValueError('Estimated symmetry mesh changed')
        symmetries=proxy_symmetry['objects']
    scores = score_worlds(actual, reference, manifest['object_names'], symmetries)
    report = {'episode': manifest['episode'], 'environment_steps': manifest.get('environment_steps'),
        'metric': 'Local reconstructed-reference MP-SR proxy; all-object time-mean strict3cm/30deg',
        'alignment': 'One rigid transform per world: initial first nonsymmetric object orientation, otherwise estimated cup axis plus interobject direction; translation anchored at object0; raw exported poses unchanged',
        'worlds': scores, 'any_success': any(w['passed'] for w in scores),
        'official_heldout_success': None, 'cd_o': None,
        'cd_o_reason': 'No official ground-truth reconstruction meshes available',
        'symmetry_contract': contract.get('symmetry_policy'),
        'local_estimated_symmetry':proxy_symmetry,
        'local_estimated_symmetry_sha256':digest(symmetry_path) if proxy_symmetry else None,
        'official_metric_limitations': ['Host mesh registration, anchor locations and numeric symmetry axes are not public',
            'Local origin/orientation errors are a proxy; no official heldout success guarantee'],
        'checkpoint_sha256': digest(checkpoint), 'trajectory_sha256': digest(trajectory),
        'source_reference_sha256': digest(source), 'contract_sha256': digest(contract_path)}
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root/'scores').mkdir(exist_ok=True)
    score_id = report['checkpoint_sha256'][:12] + '-' + report['trajectory_sha256'][:12]
    atomic_json(archive_root/'scores'/f'{score_id}.json', report)
    created = []
    with (archive_root/'.archive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for score in scores:
            if not score['passed']:
                continue
            world = score['world_id']; name = f'{score_id}-world{world:03d}'
            target = archive_root/name
            if not target.exists():
                temporary = Path(tempfile.mkdtemp(prefix='.pending-', dir=archive_root))
                try:
                    shutil.copy2(trajectory, temporary/'raw_all_worlds.npz')
                    shutil.copy2(manifest_path, temporary/'rollout_manifest.json')
                    shutil.copy2(checkpoint, temporary/'policy.safetensors')
                    shutil.copy2(source, temporary/'canonical_reference.parquet')
                    shutil.copy2(contract_path, temporary/'public_contract.json')
                    if proxy_symmetry:shutil.copy2(symmetry_path,temporary/'proxy_symmetry.json')
                    if manifest.get('reference_snapshot_path'):
                        shutil.copy2(manifest['reference_snapshot_path'], temporary/'robot_reference_snapshot.npz')
                    # Preserve the complete source asset closure, including URDF collision meshes.
                    assets = source.parent/'assets'
                    if not assets.is_dir(): assets = source.parent/'assets_dexagent_repaired'  # repaired-scene roots
                    if not assets.is_dir():
                        raise ValueError('Canonical object asset closure is missing')
                    shutil.copytree(assets, temporary/'assets')
                    support = source.parent/'reference_support.usda'
                    if support.exists(): shutil.copy2(support, temporary/'reference_support.usda')
                    video_path = Path(video or manifest.get('video_path', ''))
                    video_saved = video_path.is_file() and video_path.stat().st_size > 0
                    if video_saved: shutil.copy2(video_path, temporary/'parallel_rollout.mp4')
                    _export(temporary/(manifest['episode']+'.parquet'), actual[:, world], reference, manifest, source, exporter_path)
                    from package_success import pack_component
                    try:
                        packed=pack_component(temporary/'submission_component.parquet',actual[:,world],
                            manifest['object_names'],row['object_mesh_paths'],manifest,
                            base/'reproduction/submission_kit',manifest.get('code_commit_url',''))
                    except Exception as error:
                        # Keep the genuine successful rollout even when packaging is blocked.
                        packed={'ready_component':False,'error':str(error),
                                'complete_official_submission':False}
                    atomic_json(temporary/'packaging.json',packed)
                    info = {**report, 'selected_world': score, 'selected_original_world_id': world,
                        'selected_world_seed': manifest['world_seeds'][world], 'submitted_world_id': 0,
                        'world_reindexing': 'Exactly one complete actual rollout renamed to public world0; no frame/world padding',
                        'action_context': manifest.get('action_context'), 'video_saved': video_saved,
                        'schema_component_ready': packed.get('schema_and_public_episode_grid_valid',False),
                        'submission_component_ready':packed['ready_component'],
                        'packaging':packed,'complete_official_submission': False,
                        'official_evaluator_compliance_verified': False,
                        'missing_public_episodes': sorted(set(contract['episodes'])-{manifest['episode']}),
                        'selected_for': 'Full-episode local proxy success; not official leaderboard performance'}
                    atomic_json(temporary/'success.json', info)
                    (temporary/'README.txt').write_text(
                        'Successful unassisted complete physics rollout against the reconstructed local reference.\n'
                        'submission_component.parquet combines actual poses and reconstructed meshes using the latest public kit when packaging.json reports success.\n'
                        'All20 episodes, a published code commit, and official pinned-evaluator compliance are still required for a complete official submission.\n'
                        'Raw poses are actual simulation outputs; no reset, VOC, pose forcing, frame filling, or duplicated worlds.\n'
                        'Selected original world is recorded in success.json; the public row roster calls it world0.\n'
                        'MP-SR is a LOCAL PROXY. Official heldout MP-SR and CD-O are unknown.\n'
                        'parallel_rollout.mp4 shows all original worlds; consult selected_original_world_id.\n'
                        'Checkpoint/reference retain their original paths; this archive preserves evidence and is not a patched relocatable checkpoint.\n')
                    hashes = {str(p.relative_to(temporary)):digest(p) for p in sorted(temporary.rglob('*')) if p.is_file()}
                    atomic_json(temporary/'sha256.json', hashes)
                    os.rename(temporary, target)
                finally:
                    if temporary.exists(): shutil.rmtree(temporary)
            else:
                for rel, expected in json.loads((target/'sha256.json').read_text()).items():
                    if digest(target/rel) != expected: raise ValueError('Existing immutable success archive changed')
            pointer = {'archive': name, 'world_id': world, 'selection_cost': score['selection_cost'],
                       'checkpoint_sha256': report['checkpoint_sha256'], 'environment_steps': manifest.get('environment_steps')}
            first = archive_root/'first_success.json'
            if not first.exists(): atomic_json(first, pointer)
            best = archive_root/'best_success.json'
            if not best.exists() or pointer['selection_cost'] < json.loads(best.read_text())['selection_cost']:
                atomic_json(best, pointer)
            created.append(str(target))
    return {**report, 'archives': created}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--archive-root',type=Path,required=True)
    parser.add_argument('--video',type=Path)
    args=parser.parse_args()
    result=archive(args.trajectory,args.manifest,args.archive_root,video=args.video)
    print(json.dumps({'any_success':result['any_success'],'archives':result['archives']}),flush=True)


if __name__=='__main__': main()
