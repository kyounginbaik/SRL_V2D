#!/usr/bin/env python3
"""[v22 stream F copy of streamV_C19/code/stage_ep19C_component.py: target_final_v5 world with the physical twin sweeper; paths/notes only] Stage the episode-19 actor as a frozen-build component whose scoring reference is the compose-target corrected copy
(Stream V, 2026-10-24; adapted from streamH_ep1/code/stage_ep1_component.py, same flow as refinement_20261021_ep36/code/stage_ep36_component.py). The frozen runtime, the physics scene
and the reconstruction meshes are unchanged except the home keyframe object poses; the reference object tracks are the compose-target corrected copy.

Flow (unchanged frozen evaluator build of 2026-10-19):
  fresh re-evaluation (exactness check) -> shared component + portable bundle -> portable replay
  -> common dispatcher execution -> evidence + portable archive -> staged.json
Only new output paths are written. Nothing is uploaded.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, tarfile
from pathlib import Path
from types import SimpleNamespace
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path('/data1/SRL_V2D')
LEGACY = ROOT / 'challenge_reconstruction/dexagent_final_recon_20261017'
COMMON = ROOT / 'challenge_reconstruction/episode37_ilchold_submission_20261019'
KIT = LEGACY / 'refinement_20261017/submission_readiness/latest_official/v2d_submission_kit'
H = ROOT / 'challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19'
PREVIOUS = ROOT / 'challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/ep19/staging/components/episode_000019'   # v21 component
EXPECTED_MESHES = {'mini_sweeper': 'cf7bf340cfaddef30fa4c9b2d33ffc64959a7299487c9f49999ae7ed7843b9cb',
                   'mini_dust_pan': '1d298c012d68712ce49ff3b73a36f1b8f76e6cdb6bba3ee1357c467c3e61075f'}


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda: s.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')
def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def command(script, *arguments, cwd=None):
    subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True, cwd=cwd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--episode', type=int, default=19); ap.add_argument('--episode-dir', type=Path, required=True); ap.add_argument('--distilled', type=Path, required=True, help='dir with training/policy.npz and evaluation/result.json')
    ap.add_argument('--base', type=Path, required=True, help='staging base (new directory tree)'); ap.add_argument('--video', type=Path)
    ap.add_argument('--lineage-file', type=Path, action='append', default=[]); ap.add_argument('--teacher', type=Path, required=True, help='teacher dir with result.json + reference.npz')
    ap.add_argument('--teacher-note', default='open-loop servo reference teacher: v21 ep19 ILC teacher (C_ep19_ilc lineage), object-relative wrist port (port19.py, sweeper grasp re-mapped through the half-turn twin) to the target_final_v5 targets; teacher only, not shipped')
    a = ap.parse_args(); a.base = a.base.resolve(); a.distilled = a.distilled.resolve()
    name = f'episode_{a.episode:06d}'
    checkpoint = a.distilled / 'training/policy.npz'; training_report = a.distilled / 'training/training_report.json'
    evaluation = a.distilled / 'evaluation'; report = read(evaluation / 'result.json')
    if (report.get('episode') != name or not report.get('learned_policy') or not report.get('reference_proxy_success') or
            report.get('policy_checkpoint_sha256') != sha(checkpoint) or report.get('metrics', {}).get('reference_proxy_mp_sr') != 1. or
            not report.get('execution', {}).get('physics_verified') or not report['execution'].get('independent_control_replay', {}).get('ok')):
        raise ValueError('Only a successful distilled actor with independent physics replay may be staged')
    code = LEGACY / 'code'
    common = module_at('frozen_common_staging_validation', COMMON / 'package_code/common_evaluator.py')
    build = common.verify_build(COMMON / 'build_manifest.json')
    for entry in build['sources']:
        label, _, relative = entry['key'].partition('/')
        if label not in ('legacy_code', 'legacy_dexagent'): continue
        root = code if label == 'legacy_code' else ROOT / 'third_party/DexAgent'
        if sha(root / relative) != entry['sha256']: raise ValueError('Legacy runtime source changed: ' + entry['key'])
    episode_dir = Path(report['input_dir']).resolve(strict=True)
    if episode_dir != a.episode_dir.resolve(): raise ValueError('Actor was not evaluated against the stream F target episode dir')
    manifest = read(episode_dir / 'input_manifest.json'); corr = Path(manifest['compose_correction']['record'])
    cr = read(corr)
    if sha(Path(manifest['source']) / 'object_poses.npz') != manifest['source_object_poses_sha256']: raise ValueError('Corrected source changed')
    if sha(corr) != manifest['compose_correction']['record_sha256']: raise ValueError('Correction record changed')
    component = a.base / 'components' / name; staged = component / 'staged.json'
    if staged.exists():
        print(json.dumps(read(staged), indent=2)); return
    component.mkdir(parents=True, exist_ok=True)
    teacher_variant = a.teacher.resolve(strict=True); teacher_files = [teacher_variant / 'result.json', teacher_variant / 'reference.npz']
    teacher_rollout = dict(result=str(teacher_variant / 'result.json'), sha256=sha(teacher_variant / 'result.json'), reference_sha256=sha(teacher_variant / 'reference.npz'), controller=a.teacher_note)
    stereo = [Path(manifest['compose_correction']['scene_change']), ROOT / 'challenge_reconstruction/recon_v15_20261022/episode_000019/change.json']
    script = Path(manifest['compose_correction']['script'])
    lineage_record = dict(schema='episode19_target_final_v5_twin_lineage_v1', episode=a.episode,
        previous_component=str(PREVIOUS), previous_checkpoint_sha256=sha(PREVIOUS / 'portable/original/policy.npz') if (PREVIOUS / 'portable/original/policy.npz').exists() else None,
        teacher_rollout=teacher_rollout,
        distillation=dict(training_report=str(training_report), sha256=sha(training_report), architecture=report.get('policy_architecture', 'learned_feedback_policy.LearnedPolicy')),
        track_correction=dict(record=str(corr), sha256=sha(corr), script=str(script), script_sha256=sha(script), original_object_poses_sha256=cr['original_object_poses_sha256'],
            corrected_object_poses_sha256=manifest['source_object_poses_sha256'], registration={k: dict(chamfer_mm=v['chamfer_mm']) for k, v in cr['registration'].items()}, world_transform=cr['world_transform'],
            sweeper_face=cr['sweeper_face'], compose_tracks=cr['compose_tracks'], evidence=[str(s) for s in stereo],
            rationale='v22: object targets are target_final_v5 (our reconstruction recovery selected20_final_v5; our meshes, scorer-exact registration, one rigid transform anchored on the physically settled standing twin of the sweeper); the scene differs from the shipped frozen scene only in the home keyframe sweeper pose (the half-turn twin face of target_final_v5, settled in physics).'),
        original_object_targets_byte_identical=False, scene_unchanged=False, scene_change=manifest['compose_correction']['scene_change'], meshes_unchanged=True,
        package_postprocess_instruction='do NOT apply the episode-19 sweeper half-turn relabel (postprocess_v16 G): the face is physical in this rollout', source_pose_proxy_not_official_ground_truth=True, uploaded=False)
    lineage_path = component / 'provenance/episode19_compose_target_lineage.json'; write(lineage_path, lineage_record)
    lineage = [lineage_path, training_report, *teacher_files, corr, script, *stereo, *a.lineage_file]
    fresh = a.base / 'refinement_20261018' / name / 'evaluation'
    if not (fresh / 'shared_evaluator_validation.json').is_file():
        command(code / 'reevaluate_component.py', '--baseline-report', evaluation / 'result.json', '--output', fresh, '--episode-dir', episode_dir)
    if not read(fresh / 'shared_evaluator_validation.json')['all_physical_states_actions_exact']:
        raise ValueError('Fresh re-evaluation differs from the distilled actor recording')
    sys.path[:0] = [str(code), str(ROOT / 'third_party/DexAgent')]
    import stage_policy_component, prepare_shared_component
    stage_policy_component.B = a.base; stage_policy_component.WRITER = Path(build['source_roots']['schema_writer'])
    stage_args = SimpleNamespace(run=fresh, checkpoint=checkpoint, training_report=training_report,
        base_training_report=None, episode_dir=episode_dir, output=component, quality_dir=None, lineage_file=lineage)
    if not (component / 'portable/bundle_manifest.json').is_file():
        prepare_shared_component.run(stage_args)
    for label, directory in [('legacy_code', component / 'portable/code'), ('legacy_dexagent', component / 'portable/third_party/DexAgent')]:
        common.verify_equivalent_tree(build, label, directory)
    meshes = {p.stem: sha(p) for p in (component / 'meshes' / name).iterdir()}
    if meshes != EXPECTED_MESHES: raise ValueError('Staged meshes differ from the shipped reconstruction meshes: ' + json.dumps(meshes))
    validation = component / 'portable_validation'
    if not (validation / 'portable_validation.json').is_file():
        command(component / 'portable/code/reproduce_policy_component.py', 'replay', '--bundle', component / 'portable', '--output', validation, cwd='/tmp')
    checked = read(validation / 'portable_validation.json')
    if not all(checked[k] for k in ['all_physical_states_actions_exact', 'compiled_model_arrays_options_exact', 'original_weights_unchanged', 'source_object_targets_unchanged']) or checked['reference_proxy_mp_sr'] != 1.:
        raise ValueError('Portable physical/controller validation did not pass')
    common_output = a.base / 'common_components' / name
    if not (common_output / 'completed.json').is_file():
        command(COMMON / 'package_code/common_evaluator.py', 'evaluate', '--backend', 'legacy', '--episode', a.episode, '--legacy-component', component,
                '--build-manifest', COMMON / 'build_manifest.json', '--kit', KIT, '--output', common_output, cwd='/tmp')
    common_record = read(common_output / 'completed.json')
    if common_record['checkpoint']['sha256'] != sha(checkpoint) or common_record['eval_code_sha256'] != build['eval_code_sha256'] or not common_record['exact_previous_physical_states_actions']:
        raise ValueError('Common evaluator did not reproduce the distilled actor under this build')
    evidence = component / 'portable/release_evidence'; evidence.mkdir(exist_ok=True)
    for src, dst in [(component / 'readiness.json', 'readiness.json'), (lineage_path, 'episode19_compose_target_lineage.json'), (corr, 'compose_correction.json'),
                     (validation / 'portable_validation.json', 'portable_validation.json'), (common_output / 'completed.json', 'common_execution.json'), *[(s, s.name) for s in stereo]]:
        shutil.copy2(src, evidence / dst)
    action = evidence / 'actual_robot_actions.npz'
    if not action.exists(): shutil.copy2(common_record['actions']['path'], action)
    if sha(action) != common_record['actions']['sha256']: raise ValueError('Copied action evidence differs from the common replay')
    if a.video:
        vd = component / 'videos'; vd.mkdir(exist_ok=True)
        if a.video.is_dir(): shutil.copytree(a.video, vd, dirs_exist_ok=True)
        else: shutil.copy2(a.video, vd / a.video.name)
    archive = a.base / 'portable_archives' / (name + '_portable.tar.gz'); archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        tmp = Path(str(archive) + '.partial')
        with tarfile.open(tmp, 'w:gz', compresslevel=3) as tar: tar.add(component / 'portable', arcname=name + '_portable')
        tmp.replace(archive)
    common.verify_build(COMMON / 'build_manifest.json')
    result = dict(episode=name, checkpoint_sha256=sha(checkpoint), component=str(component), common_component=str(common_output),
        common_build_manifest_sha256=sha(COMMON / 'build_manifest.json'), actual_robot_actions_sha256=common_record['actions']['sha256'],
        archive=dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive)), portable_physical_states_actions_exact=True,
        source_targets_unchanged=False, corrected_track=True, corrected_track_record=str(corr), meshes_sha256=meshes, reference_proxy_mp_sr=1., source_pose_proxy_not_official_ground_truth=True,
        video_directory=str(component / 'videos') if (component / 'videos').is_dir() else None, uploaded=False)
    write(staged, result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
