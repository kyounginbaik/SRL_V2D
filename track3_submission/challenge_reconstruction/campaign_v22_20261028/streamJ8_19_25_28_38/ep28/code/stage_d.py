#!/usr/bin/env python3
"""Stream D (phase 7, v22; copied from streamA_ep35/streamT_C356 stage_compose_component.py, episodes 1 and 20): target = campaign_v22_20261028/target_final_v5 (our reconstruction selected20_final_v5). Original doc: stage a pot+spoon actor (episodes 3, 5, 6) as a frozen-build component whose scoring reference is the
Stream T compose-referenced copy of the source (object targets = coworker compose_eval_20261021 poses in our mesh frames, one rigid
transform G per episode; streamT_C356/epE/world/source_compose/episode_X/compose_reference.json). Adapted from
streamP_ep33/code/stage_ep33_roll_component.py (same flow as streamH_ep1/code/stage_ep1_component.py). Frozen runtime, robot,
controller, table and reconstruction meshes unchanged; object targets and the objects' frame-0 home poses differ."""
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
ST = ROOT / 'challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep28'
EXPECTED = {1: {'white_pot': 'a55317043909ef4e2e299fd786e4b5121cc896dabeaa8af980210b514a4f182d',
                 'white_pot_lid': 'bc1443c51ad9634ab73582c2bef132307d00d1f0a9e76e18a9b2032eaedfeba2'},
            20: {'blue_cup': 'ff3784a8fa2527b55c4e77584ffc37d842c40fb1afd5c5ef0abc3ce5f59ad2fd',
                 'water_pitcher': '08e6ab820cf67665b9e5879e33c352fc680c35297631b5d3431bae23c8d9b9b7'},
            28: {'plastic_dish_rack': '00c513765e744bd1dbafe2efe0ee58f68943084e117d9cb9383684e163c2c741',
                 'white_pot': '69677910d854cc15affe19d7b885b06da2642b49b27777f0af2641f441c54e9e'}}
V21 = {1: 'campaign_v16_20261023/streamH_ep1/staging/stage_prev_s128_fg0.003_c512/components/episode_000001', 20: 'dexagent_remaining7_20261019/components/episode_000020', 28: 'campaign_v16_20261023/streamO_ep28/staging_6prong_C710/components/episode_000028'}


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
    ap.add_argument('--episode', type=int, required=True); ap.add_argument('--distilled', type=Path, required=True, help='dir with training/policy.npz and evaluation/result.json')
    ap.add_argument('--base', type=Path, required=True, help='staging base (new directory tree)'); ap.add_argument('--video', type=Path)
    ap.add_argument('--lineage-file', type=Path, action='append', default=[]); ap.add_argument('--teacher', type=Path, required=True, help='teacher dir with result.json + reference.npz')
    ap.add_argument('--teacher-note', default='open-loop servo reference: v21 (O6_ep28_C710) teacher ported into the target_final_v5-referenced world (object-relative wrist retarget, campaign_v22_20261028/streamJ8_19_25_28_38/ep28/code/port_d.py) and re-planned as documented in campaign_v22_20261028/streamJ8_19_25_28_38/ep28/README.md; teacher only, not shipped')
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
    if episode_dir != (ST / os.environ.get('WORLD', 'world') / 'episode_target' / name).resolve(): raise ValueError('Actor was not evaluated against the Stream D target-referenced episode dir')
    manifest = read(episode_dir / 'input_manifest.json'); corr = Path(manifest['target_correction']['record'])
    cr = read(corr)
    if sha(Path(manifest['source']) / 'object_poses.npz') != manifest['source_object_poses_sha256'] or sha(Path(manifest['source']) / 'object_poses.npz') != cr['corrected_object_poses_sha256']: raise ValueError('Corrected source changed')
    PREVIOUS = ROOT / 'challenge_reconstruction' / V21[a.episode]
    component = a.base / 'components' / name; staged = component / 'staged.json'
    if staged.exists():
        print(json.dumps(read(staged), indent=2)); return
    component.mkdir(parents=True, exist_ok=True)
    teacher_variant = a.teacher.resolve(strict=True); teacher_files = [teacher_variant / 'result.json'] + ([teacher_variant / 'reference.npz'] if (teacher_variant / 'reference.npz').exists() else [])
    teacher_rollout = dict(result=str(teacher_variant / 'result.json'), sha256=sha(teacher_variant / 'result.json'), controller=a.teacher_note)
    stereo = [ST / os.environ.get('WORLD', 'world') / 'target_reference.json']
    script = Path(cr['script'])
    lineage_record = dict(schema='streamD_target_final_v5_referenced_lineage_v1', episode=a.episode,
        previous_component=str(PREVIOUS), previous_checkpoint_sha256=read(PREVIOUS / 'staged.json')['checkpoint_sha256'],
        previous_staged=read(PREVIOUS / 'staged.json'), teacher_rollout=teacher_rollout,
        distillation=dict(training_report=str(training_report), sha256=sha(training_report), architecture=report.get('policy_architecture', 'learned_feedback_policy.LearnedPolicy')),
        track_correction=dict(record=str(corr), sha256=sha(corr), script=str(script), script_sha256=sha(script),
            original_object_poses_sha256=cr['original_object_poses_sha256'], corrected_object_poses_sha256=manifest['source_object_poses_sha256'],
            target_sha256=cr['target_sha256'], G_setup_from_target=cr['G_setup_from_target'], anchor=cr['anchor'], axisymmetric_spin_kept_from_v21_reference=cr.get('axisymmetric_spin_kept_from_v21_reference'), Gt_tilt_deg=cr['Gt_tilt_deg'], shift_vs_v21_reference=cr['shift_vs_v21_reference_setup'],
            rationale='v22 (PHASE7.md): the scoring reference is our own reconstruction selected20_final_v5 (campaign_v22_20261028/target_final_v5), registered onto our shipped meshes with the scorer registration and placed in our physics world by one rigid transform (anchor frame-0 + table-plane fit of the resting contacts).'),
        original_object_targets_byte_identical=False, scene_unchanged=False, scene_change='object bodies / home keyframe moved to the target frame-0 poses (ep20: left wrist home moved rigidly with the held cup); robot model, controller, table, meshes unchanged', meshes_unchanged=True, source_pose_proxy_not_official_ground_truth=True, uploaded=False)
    lineage_path = component / 'provenance/streamD_target_referenced_lineage.json'; write(lineage_path, lineage_record)
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
    if meshes != EXPECTED[a.episode]: raise ValueError('Staged meshes differ from the shipped reconstruction meshes: ' + json.dumps(meshes))
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
    for src, dst in [(component / 'readiness.json', 'readiness.json'), (lineage_path, 'streamD_target_referenced_lineage.json'), (corr, 'target_reference.json'),
                     (validation / 'portable_validation.json', 'portable_validation.json'), (common_output / 'completed.json', 'common_execution.json')]:
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
