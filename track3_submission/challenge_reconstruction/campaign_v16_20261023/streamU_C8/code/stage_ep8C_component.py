#!/usr/bin/env python3
"""Stream U (phase 6, v21C hedge, 2026-10-24): stage an episode-8 actor as a frozen-build component whose scoring reference is the
coworker's compose_eval track converted into OUR mesh bodies / source world (corrected-source copy, worldC2/source_compose) and whose
physics world is rebuilt on the table plane implied by the compose resting poses (worldC2/world). Derived from
streamH_ep1/code/stage_ep1_component.py (corrected-reference wrapper pattern). Runtime code and the reconstruction meshes are unchanged.

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
U = ROOT / 'challenge_reconstruction/campaign_v16_20261023/streamU_C8'; WORLD = U / 'worldC2'
PREVIOUS = LEGACY / 'refinement_20261020_pot/staging_v2/components/episode_000008'   # v21 component (our evidence-based set)
EXPECTED_MESHES = {'beige_cup': '2c041e69fbd3be79fa953609276ffc4a54a513ec2ca7e7f31020a78ab6871431',
                   'white_pot': '8cbd35431829ae37eddb9ad8fc573f823836d880f21519f99d18c7420726d8f8'}


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
    ap.add_argument('--episode', type=int, default=8); ap.add_argument('--distilled', type=Path, required=True, help='dir with training/policy.npz and evaluation/result.json')
    ap.add_argument('--base', type=Path, required=True, help='staging base (new directory tree)'); ap.add_argument('--video', type=Path)
    ap.add_argument('--lineage-file', type=Path, action='append', default=[]); ap.add_argument('--teacher', type=Path, required=True, help='teacher dir with result.json + reference.npz')
    ap.add_argument('--teacher-note', default='open-loop servo reference teacher: streamB level-world ep8 teacher (retention CEM + pot ILC) ported object-relatively into worldC2 (retarget_teacher.py), then reference-level carry ILC on the pot (left) and the cup (right, spin-free); teacher only, not shipped')
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
    if episode_dir != (WORLD / 'world/episodes' / name).resolve(): raise ValueError('Actor was not evaluated against the Stream U compose-target episode dir')
    manifest = read(episode_dir / 'input_manifest.json')
    if sha(Path(manifest['source']) / 'object_poses.npz') != manifest['source_object_poses_sha256']: raise ValueError('Compose-target source changed')
    corr = WORLD / 'lineage.json'; cr = read(corr); plane = WORLD / 'table_plane.json'; diag = Path(manifest['compose_target']['diagnostics'])
    if sha(diag) != manifest['compose_target']['diagnostics_sha256']: raise ValueError('compose reference diagnostics changed')
    component = a.base / 'components' / name; staged = component / 'staged.json'
    if staged.exists():
        print(json.dumps(read(staged), indent=2)); return
    component.mkdir(parents=True, exist_ok=True)
    teacher_variant = a.teacher.resolve(strict=True); teacher_files = [teacher_variant / 'result.json', teacher_variant / 'reference.npz']
    teacher_rollout = dict(result=str(teacher_variant / 'result.json'), sha256=sha(teacher_variant / 'result.json'), reference_sha256=sha(teacher_variant / 'reference.npz'), controller=a.teacher_note)
    stereo = [plane, diag, U / 'code/compose_reference.py', U / 'code/build_world.py', U / 'code/retarget_teacher.py', U / 'code/ilc_carry_sym.py']
    script = U / 'code/build_world.py'
    lineage_record = dict(schema='episode8_compose_target_lineage_v1', episode=a.episode, hedge_set='v21C (phase 6): scored against the coworker compose_eval track',
        previous_component=str(PREVIOUS), previous_checkpoint_sha256=sha(PREVIOUS / 'portable/original/policy.npz'),
        previous_staged=read(PREVIOUS / 'staged.json'), teacher_rollout=teacher_rollout,
        distillation=dict(training_report=str(training_report), sha256=sha(training_report), architecture=report.get('policy_architecture', 'learned_feedback_policy.LearnedPolicy')),
        track_correction=dict(record=str(corr), sha256=sha(corr), script=str(script), script_sha256=sha(script), compose_inputs=cr['compose_inputs'], W_source_from_compose=cr['W_source_from_compose'],
            scoring_object_poses_sha256=cr['scoring_object_poses_sha256'], physical_object_poses_sha256=cr['physical_object_poses_sha256'], v21_reference_sha256=cr['v21_reference_sha256'],
            shift_vs_v21={k: {kk: vv for kk, vv in v.items() if kk != 'cm_per_frame'} for k, v in cr['shift_vs_v21'].items()}, table_plane=cr['table_plane'],
            rationale='Phase-6 hedge: the compose poses are adopted as given (body conversion = the kit scorer registration of our packed meshes onto theirs; cup spin about the scorer symmetry axis re-chosen closest to our track, score-invariant; one rigid camera-chain transform into our source world). The compose resting poses imply a table 6.5 deg off our stereo plane (pot and cup rest axes agree within 2.2 deg), so the physics world is rebuilt on the compose pot rest plane; resting frames of the physical targets are projected onto it.'),
        original_object_targets_byte_identical=False, scene_rebuilt=True, meshes_unchanged=True, source_pose_proxy_not_official_ground_truth=True, uploaded=False)
    lineage_path = component / 'provenance/episode8_compose_target_lineage.json'; write(lineage_path, lineage_record)
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
    for src, dst in [(component / 'readiness.json', 'readiness.json'), (lineage_path, 'episode8_compose_target_lineage.json'), (corr, 'compose_target_lineage.json'),
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
