#!/usr/bin/env python3
"""Calibrated-table preparation and honest full-physics baselines for new episodes.

Original five adapters and all source reconstructions are immutable. No baseline
MPC/reference execution is labelled a learned policy or submission-ready result.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

B=Path(__file__).resolve().parents[1]
ROOT=B.parents[1]
NEW=[6,19,32,33,35,36,37,38]
ALL=[1,3,5,6,8,10,19,32,33,35,36,37,38]
MISSING=[14,15,16,20,25,28,30]
OUT=B/'refinement_20261018/expanded_batch'
sys.path.insert(0,str(ROOT/'third_party/DexAgent'))


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)


def inventory():
    rows=[]
    for ep in ALL:
        name=f'episode_{ep:06d}';source=ROOT/'challenge_reconstruction/final_recon'/name
        m=read(source/'manifest.json');prepared=B/'episodes'/name
        raw=OUT/'baseline'/name/'result.json'
        row=dict(episode=name,source=str(source),frames=m['frames'],fps=m['fps'],objects=m['objects'],
            source_reference_sha256=sha(source/'reference.parquet'),source_object_poses_sha256=sha(source/'object_poses.npz'),
            source_manifest_sha256=sha(source/'manifest.json'),source_unchanged=True,
            scope='new calibrated-table baseline' if ep in NEW else 'existing original adapter preserved',
            input_manifest=str(prepared/'input_manifest.json') if (prepared/'input_manifest.json').exists() else None,
            baseline_result=str(raw) if raw.exists() else None)
        if raw.exists():
            r=read(raw);row.update(reference_proxy_mp_sr=r['metrics']['reference_proxy_mp_sr'],
                physics_verified=r['execution']['physics_verified'],learned_policy=False)
        rows.append(row)
    result=dict(episodes=rows,source_folder_episode_count=13,official_roster_count=20,
        missing_official_episodes=[f'episode_{i:06d}' for i in MISSING],
        full_submission_ready=False,uploaded=False,
        metric_ground_truth='Immutable source reconstruction proxy; no confidential official GT',
        rule='Source target hashes never change to improve a score. Repaired reconstruction candidates must be separately versioned and scored against originals.')
    write(OUT/'inventory.json',result)
    return result


def prepare_one(ep):
    if ep not in NEW:raise ValueError('This preparation wrapper cannot overwrite the original five')
    name=f'episode_{ep:06d}';target=B/'episodes'/name
    already_prepared=target.exists()
    if already_prepared:
        if not (target/'input_manifest.json').exists():raise FileExistsError(f'Partial preparation needs explicit review: {target}')
        current=read(target/'input_manifest.json')
        if current['support']['method']!='calibrated stereo tabletop plane; entire scene rigidly aligned to its normal':
            raise FileExistsError('Refusing to replace differently prepared episode')
    support_path=OUT/'source_review'/name/'table_support.json';support=read(support_path)
    if not support['tilted_plane_quality_pass']:raise ValueError('Measured table-plane quality failed')
    source=ROOT/'challenge_reconstruction/final_recon'/name
    before={f:sha(source/f) for f in ['reference.parquet','object_poses.npz','manifest.json']}
    if support['source_pose_sha256']!=before['object_poses.npz']:raise ValueError('Source poses changed after support measurement')
    from prepare_episodes import prepare
    from evaluate_mp_sr import geometry_from_reference
    evidence=dict(path=str(support_path),sha256=sha(support_path),
        source_normal=support['fitted_plane_normal_source'],normal_spread_degrees=support['normal_spread_degrees'],
        median_plane_residual_m=support['plane_median_absolute_residual_m'],
        initial_mesh_bottoms_in_setup_m=support['aligned_initial_mesh_bottoms_m'])
    manifest=current if already_prepared else prepare(ep,np.asarray(support['proposed_setup_from_source']),evidence)
    write(target/'metric_geometry.json',geometry_from_reference(source))
    original=np.load(source/'object_poses.npz');reference=np.load(target/'reference_trajectory.npz')
    R=Rotation.from_quat(reference['object_quaternions_wxyz'].reshape(-1,4),scalar_first=True).as_matrix().reshape(-1,len(original['names']),3,3)
    T=np.broadcast_to(np.eye(4),original['world_from_object'].shape).copy();T[:,:,:3,:3]=R;T[:,:,:3,3]=reference['object_positions']
    restored=np.asarray(manifest['source_from_setup'])@T
    difference=float(np.max(abs(restored-original['world_from_object'])))
    # Source float32 rotation matrices have ~3e-7 orthogonality residual;
    # conversion to unit quaternion projects that storage noise onto SO(3).
    position_difference=float(np.max(abs(restored[:,:,:3,3]-original['world_from_object'][:,:,:3,3])))
    if difference>1e-6 or position_difference>1e-10:
        raise ValueError('World-frame transform changes source object targets beyond source precision')
    if any(sha(source/f)!=h for f,h in before.items()):raise ValueError('Source inputs changed during adapter preparation')
    write(target/'calibrated_frame_validation.json',dict(source_hashes=before,all_original_object_frames_roundtrip_max_error=difference,
        position_roundtrip_max_error_m=position_difference,
        rotation_roundtrip_tolerance=1e-6,rotation_precision_note='Original float32 matrices normalized to unit-quaternion rotations; no fitted per-object adjustment',
        full_rigid_frame_applied_to=['objects positions/orientations','robot wrist positions/orientations','moving cameras'],
        velocity_semantics='Controller derives motion from transformed full-timing qpos; no source linear/angular velocity arrays are reused.',
        gravity='Setup -Z, aligned with independently measured tabletop normal',
        initial_geometry_warnings={k:v for k,v in support['aligned_initial_mesh_bottoms_m'].items() if v<-.002},
        original_object_targets_unchanged=True))
    return manifest


def baseline(ep):
    if ep not in NEW:raise ValueError('New-episode baseline only')
    name=f'episode_{ep:06d}';inp=B/'episodes'/name;manifest=read(inp/'input_manifest.json')
    out=OUT/'baseline'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return read(out/'result.json')
    from floating_runtime import _freeze,execute_reference
    from calibrate_controller import controller_scene
    from evaluate_mp_sr import evaluate_files
    base=out/'frozen_initial';base.mkdir(exist_ok=True)
    _,frozen=_freeze(inp/'stage2/scene.xml',base)
    scene=out/'controller_scene.xml'
    controller=controller_scene(frozen,scene,'kp5000_r200_d5')
    reference=inp/'tracking_reference.npz';begin=time.monotonic()
    execution=execute_reference(scene,reference,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    result=dict(episode=name,learned_policy=False,method='Source robot reference, continuous wrist Euler branches, calibrated measured tabletop frame, standard bounded PD',
        controller=controller,reference=str(reference),reference_sha256=sha(reference),
        input_manifest_sha256=sha(inp/'input_manifest.json'),metric_geometry_sha256=sha(inp/'metric_geometry.json'),
        support_evidence=manifest['support'],execution=execution,metrics=metrics,wall_seconds=time.monotonic()-begin,
        source_targets_unchanged=True,submission_ready=False)
    write(out/'result.json',result)
    print(json.dumps({'episode':name,'physics_verified':execution['physics_verified'],
        'reference_proxy_mp_sr':metrics['reference_proxy_mp_sr'],'objects':metrics['official_style_reference_proxy']['objects']}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','baseline','inventory']);p.add_argument('--episodes',type=int,nargs='+',default=NEW)
    args=p.parse_args()
    if args.mode=='prepare':
        for ep in args.episodes:prepare_one(ep)
    elif args.mode=='baseline':
        for ep in args.episodes:baseline(ep)
    inventory()
