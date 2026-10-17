#!/usr/bin/env python3
"""Prepare one genuine learned-policy episode for the official Track 3 packer.

No upload, no fake provenance, no modified 20-episode roster. `prepare` stages
actual meshes only. `stage` requires a freshly evaluated learned-policy recording.
The schema name follows the official reader; it does not claim FlashCHORD execution.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from evaluate_mp_sr import (apply_world_transform, evaluate_files, load_reference,
                           load_trajectory)
from grasp_stability import audit_grasp_stability
from submission_readiness import public_roster, trajectory_info

B = Path(__file__).resolve().parents[1]
KIT = B/'refinement_20261017/submission_readiness/latest_official/v2d_submission_kit'
WRITER = B.parents[1]/'robotic_grounding/flash_chord/src/flash_chord/evaluation/submission.py'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def prepare(episode, output, episode_dir=None):
    name = f'episode_{episode:06d}'
    source = Path(episode_dir).resolve() if episode_dir else B/'episodes'/name
    manifest = read(source/'input_manifest.json')
    roster = public_roster(KIT)[episode]
    assert [o['id'] for o in manifest['objects']] == roster['object_names']
    original = Path(manifest['source'])
    for file, key in [('reference.parquet','source_reference_sha256'),
                      ('object_poses.npz','source_object_poses_sha256')]:
        if sha(original/file) != manifest[key]:
            raise ValueError(f'Original source changed: {original/file}')
    sys.path.insert(0, str(KIT))
    # The repository also has a regular `tools` package. Bind this process's
    # toolkit namespace explicitly so the official relative helper is selected.
    kit_tools = ModuleType('tools')
    kit_tools.__path__ = [str(KIT/'tools')]
    sys.modules['tools'] = kit_tools
    from v2dlb.track3_pose import pack_object_meshes, validate_packed_meshes
    mesh_manifest = {}
    for obj in manifest['objects']:
        visuals = [a for a in obj['meshes'] if a['kind'] == 'visual']
        if len(visuals) != 1:
            raise ValueError('Expected one source visual mesh in exact body coordinates')
        visual = visuals[0]
        if sha(visual['source']) != visual['sha256']:
            raise ValueError('Source visual mesh has changed')
        destination = output/'meshes'/name/(obj['id']+Path(visual['source']).suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha(destination) != visual['sha256']:
            raise ValueError(f'Refusing to overwrite a different staged mesh: {destination}')
        if not destination.exists():
            shutil.copy2(visual['source'], destination)
        mesh_manifest[obj['id']] = dict(source=visual['source'], staged=str(destination),
            sha256=sha(destination), body_coordinates_preserved=True,
            provenance='Participant-supplied final_recon asset; reconstruction origin must remain documented separately')
    # Exercise the actual official mesh packer/decoder without emitting fake pose rows.
    mesh_rows = pd.DataFrame({'row_id':[f'track_3/e{episode:06d}/w000/f000000/o{i}'
                                      for i in range(len(roster['object_names']))]})
    packed = pack_object_meshes(mesh_rows, output/'meshes', {str(episode):roster['object_names']})
    decoded = validate_packed_meshes(packed)
    budget = {roster['object_names'][slot]:dict(padded_vertices=len(vertices), padded_faces=len(faces),
              finite=bool(np.isfinite(vertices).all()),
              nonzero_area_faces=int(np.sum(np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],
                  vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)>1e-12)))
              for (_,slot),(vertices,faces) in decoded.items()}
    result = dict(episode=name, kind='partial episode input preparation; not a submission',
        official_kit=str(KIT), official_kit_manifest_sha256=sha(KIT/'KIT_MANIFEST.json'),
        source_manifest=str(source/'input_manifest.json'), source_manifest_sha256=sha(source/'input_manifest.json'),
        source_reference_sha256=manifest['source_reference_sha256'],
        source_object_poses_sha256=manifest['source_object_poses_sha256'],
        original_object_targets_unchanged=True, fps=manifest['fps'], frames=manifest['frames'],
        expected_trajectory_rows=len(roster['rows']), object_names=roster['object_names'],
        meshes=mesh_manifest, official_mesh_pack_decode_validation=budget,
        learned_policy_checkpoint=None, policy_rollout=None, episode_ready_for_packaging_review=False,
        complete_submission_ready=False, official_score=None, uploaded=False,
        limitations=['This is one episode of the official 20; the public roster is not changed.',
            'The source-reference MP-SR is a proxy, not confidential held-out scoring.',
            'A genuine learned-policy checkpoint, fresh online rollout and actual producing code commit are required.',
            'Custom evaluator provenance must identify the actual implementation; no official-runtime certification is implied.'])
    write(output/'readiness.json', result)
    return result, manifest, roster


def stage(args, prepared, manifest, roster):
    run = args.run.resolve()
    root = B/'refinement_20261018'
    if not run.is_relative_to(root):
        raise ValueError('Expected a new learned-policy evaluation under refinement_20261018')
    report_path = run/'result.json' if (run/'result.json').exists() else run/'report.json'
    report = read(report_path)
    checkpoint = args.checkpoint.resolve()
    training = args.training_report.resolve()
    if not report.get('learned_policy', False):
        raise ValueError('Recording is not identified as a learned-policy online evaluation')
    if not checkpoint.is_file() or not training.is_file():
        raise ValueError('Actual checkpoint and training evidence are required')
    checkpoint_hash = sha(checkpoint)
    if report.get('policy_checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Recorded learned-policy checkpoint hash does not match actual weights')
    if read(training).get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Training evidence does not identify the same actual checkpoint')
    if report.get('evaluator_script_sha256') != sha(args.evaluator_script):
        raise ValueError('Evaluator source changed since this rollout was produced')
    if report.get('policy_code_sha256') not in {sha(p) for p in args.evaluation_code}:
        raise ValueError('Actual producing policy module must be included in evaluation-code files')
    execution = report['execution']
    if not (execution.get('physics_verified') and execution.get('execution_completed') and
            execution.get('independent_control_replay',{}).get('ok')):
        raise ValueError('Full-horizon physical recording and independent control replay must pass')
    episode = f'episode_{args.episode:06d}'
    directory = args.episode_dir.resolve() if args.episode_dir else B/'episodes'/episode
    metrics = evaluate_files(manifest['source'], execution['trajectory'], directory/'input_manifest.json',
                             execution['scene'], directory/'metric_geometry.json')
    write(args.output/'validation/reference_proxy_mp_sr.json', metrics)
    if metrics['reference_proxy_mp_sr'] != 1.:
        raise ValueError('New learned-policy rollout does not pass the requested reference-proxy MP-SR')
    grasp = audit_grasp_stability(execution['scene'], execution['trajectory'], frames_only=True)
    write(args.output/'validation/grasp_stability.json', grasp)
    achieved = apply_world_transform(load_trajectory(execution['trajectory']),
                                    np.asarray(manifest['source_from_setup']))
    reference = load_reference(manifest['source'])
    if achieved['object_names'] != roster['object_names'] or achieved['object_positions'].shape[1] != 1:
        raise ValueError('Require one world and exact official object order')
    if (not np.allclose(achieved['timestamps_seconds'],reference['timestamps_seconds'],rtol=0,atol=1e-6)
            or len(achieved['timestamps_seconds']) != manifest['frames']):
        raise ValueError('Full original timestamp grid must be recorded without retiming')
    output = args.output/'results'/f'{episode}.parquet'
    code_files = list(dict.fromkeys([args.evaluator_script.resolve(), *[p.resolve() for p in args.evaluation_code],
                                    Path(__file__).resolve(), WRITER.resolve()]))
    code_manifest = [{'path':str(p), 'sha256':sha(p)} for p in code_files]
    # Explicit concatenation convention matches the organizer harness; order is recorded.
    digest = hashlib.sha256()
    for file in code_files:
        digest.update(file.read_bytes())
    spec = importlib.util.spec_from_file_location('official_schema_writer', WRITER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    poses = lambda a: np.concatenate([a['object_positions'], a['object_quaternions_wxyz'][...,[1,2,3,0]]],axis=-1)
    rollout = SimpleNamespace(achieved_pose_w=poses(achieved), reference_pose_w=poses(reference)[:,0],
        object_body_names=roster['object_names'], body_object_ids=np.arange(len(roster['object_names'])),
        non_finite_worlds=(), start_frame=0, control_fps=manifest['fps'])
    module.write_object_trajectories_parquet(output, rollout, checkpoint_sha256=checkpoint_hash,
        eval_script_sha256=sha(args.evaluator_script), eval_code_sha256=digest.hexdigest(),
        reference_parquet=Path(manifest['source'])/'reference.parquet')
    table = pq.read_table(output)
    metadata = dict(table.schema.metadata or {})
    metadata.update({b'participant.evaluator':b'custom frozen-scene MuJoCo learned feedback policy',
        b'participant.checkpoint_format':checkpoint.suffix.encode(),
        b'participant.training_evidence_sha256':sha(training).encode(),
        b'participant.timestamp_semantics':b'initial state at t=0; original 20Hz timestamp grid; 500Hz policy physics',
        b'participant.source_from_setup':json.dumps(manifest['source_from_setup']).encode(),
        b'participant.evaluation_code_manifest':json.dumps(code_manifest).encode()})
    pq.write_table(table.replace_schema_metadata(metadata), output)
    expected = dict(roster, fps=manifest['fps'])
    validation = trajectory_info(output, expected, KIT, {'sha256':checkpoint_hash})
    if validation['problems']:
        raise ValueError(validation['problems'])
    write(args.output/'validation/official_reader.json', validation)
    result = dict(prepared, kind='genuine learned-policy episode component for packaging review',
        learned_policy_checkpoint=dict(path=str(checkpoint),sha256=checkpoint_hash),
        training_evidence=dict(path=str(training),sha256=sha(training)),
        policy_evaluation_report=dict(path=str(report_path),sha256=sha(report_path)),
        policy_rollout=validation, evaluation_code_manifest=code_manifest,
        evaluation_code_sha256=digest.hexdigest(), evaluator_script_sha256=sha(args.evaluator_script),
        reference_proxy_mp_sr=metrics['reference_proxy_mp_sr'],
        episode_ready_for_packaging_review=True, complete_submission_ready=False,
        provenance_scope='Actual custom learned-policy evaluator; official schema-compatible, not bundled-harness execution')
    write(args.output/'readiness.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','stage'])
    p.add_argument('--episode',type=int,default=8,
                   help='Official roster episode; each requires an existing verified adapter and fresh policy evaluation')
    p.add_argument('--episode-dir',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--run',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--training-report',type=Path)
    p.add_argument('--evaluator-script',type=Path)
    p.add_argument('--evaluation-code',type=Path,action='append',default=[])
    args=p.parse_args()
    args.output = (args.output or B/f'refinement_20261018/policy_handoff/episode_{args.episode:06d}').resolve()
    if not args.output.is_relative_to(B/'refinement_20261018'):
        p.error('Output must remain in the new isolated experiment directory')
    if args.mode == 'stage' and any(getattr(args,k) is None for k in
            ('run','checkpoint','training_report','evaluator_script')):
        p.error('stage requires --run, --checkpoint, --training-report, --evaluator-script')
    result, manifest, roster = prepare(args.episode,args.output,args.episode_dir)
    if args.mode == 'stage':
        result=stage(args,result,manifest,roster)
    print(json.dumps({'output':str(args.output), 'episode_ready_for_packaging_review':result['episode_ready_for_packaging_review'],
        'complete_submission_ready':False,'expected_rows':result['expected_trajectory_rows']}))


if __name__=='__main__':
    main()
