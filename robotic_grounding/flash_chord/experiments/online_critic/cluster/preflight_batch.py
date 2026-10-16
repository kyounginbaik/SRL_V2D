"""Verify every prepared public episode and CPU-test the frozen source once.

This certifies inputs, configuration composition, CPU scene assembly and tests.
It does not certify GPU integration, learned improvement or hidden scores.
"""
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

os.environ['JAX_PLATFORMS'] = 'cpu'
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT/'flash_chord'
TESTS = (
    'tests/retargeting/test_wrench_hypergradient.py',
    'tests/retargeting/test_wrench_hypergradient_editor.py',
    'tests/retargeting/test_lower_adaptation.py',
    'tests/retargeting/test_reference_relabel.py',
    'tests/retargeting/test_actor_response.py',
    'tests/retargeting/test_finite_response_acceptance.py',
    'tests/retargeting/test_critic_audit.py',
    'tests/retargeting/test_demo_archive.py',
    'tests/retargeting/test_reference_snapshot.py',
    'tests/runtime/test_wrist_limits.py',
    'tests/runtime/test_wrist_residual_slew.py',
    'tests/data/test_hand_validity.py',
    'tests/objectives/test_masked_hand_rewards.py',
    'tests/objectives/test_masked_contact_supervision.py',
    'tests/lifecycle/test_hand_validity_termination.py',
    'tests/retargeting/test_masked_contact_support.py',
    'tests/scene/test_measured_support.py',
    'tests/scene/test_initial_object_poses.py',
    'tests/scene/test_physics_gate.py',
    'tests/scene/test_setup.py',
    'tests/retargeting/test_fixed_start.py',
    '../cluster/test_initial_state.py',
    'tests/training/flash_sac/test_eigen_exploration.py',
    'tests/visualization/test_policy_viewer_resets.py',
    'tests/training/flash_sac/test_flash_sac_runner.py::test_reference_capture_hooks_run_before_donation_and_publish_metrics',
    'tests/evaluation/test_submission.py',
    '../submission/test_contract.py',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_manifests():
    bindings = {}
    for name, base in (('source_manifest.json', WORKSPACE),
                       ('launcher_manifest.json', ROOT), ('data_manifest.json', ROOT)):
        path = ROOT/name
        records = json.loads(path.read_text())
        require(isinstance(records, dict) and bool(records), f'Empty checksum manifest: {path}')
        for relative, expected in records.items():
            target = base/relative
            require(target.is_file() and digest(target) == expected,
                    f'Missing or changed frozen file: {target}')
        bindings[name.removesuffix('.json')+'_sha256'] = digest(path)
    bindings['batch_manifest_sha256'] = digest(ROOT/'batch_manifest.json')
    return bindings


def validate_inputs(batch):
    import numpy as np
    import pyarrow.parquet as pq
    from flash_chord.assets.registry import resolve_asset_path
    from flash_chord.data.mano_sharpa import load_mano_sharpa
    episodes = batch['episodes']
    records = batch['records']
    require(bool(episodes) and len(episodes) == len(set(episodes)),
            'Expected a nonempty selection of unique public episodes')
    require(batch['episode_count'] == len(episodes), 'Declared episode count differs from selected inventory')
    require(set(records) == set(episodes), 'Batch records and episode list differ')
    roster_path = ROOT/batch['public_roster']
    require(digest(roster_path) == batch['public_roster_sha256'], 'Public roster checksum mismatch')
    roster = json.loads(roster_path.read_text())['episodes']
    public_episodes = {f'episode_{int(key):06d}' for key in roster}
    require(set(episodes) <= public_episodes, 'Prepared episode set contains a non-public episode')
    missing = sorted(public_episodes-set(episodes))
    require(batch['public_episode_count'] == len(public_episodes)
            and batch['missing_public_episodes'] == missing
            and batch['public_coverage_complete'] == (not missing),
            'Public coverage metadata does not match the selected episode subset')
    inventory_path = ROOT/'input_inventory.json'
    require(digest(inventory_path) == batch['input_inventory_sha256'], 'Selected inventory checksum mismatch')
    selected = [item['episode'] for item in json.loads(inventory_path.read_text())['episodes']]
    require(len(selected) == len(set(selected)) and set(selected) == set(episodes),
            'Prepared episode set differs from the explicitly selected inventory')
    inputs = {}
    for episode in episodes:
        record = records[episode]
        require(record['episode'] == episode, f'{episode}: frozen record has the wrong identity')
        frozen = ROOT/'frozen'/episode
        parquet = frozen/'reference.parquet'
        require(json.loads((ROOT/episode/'frozen_manifest.json').read_text()) == [record],
                f'{episode}: episode manifest differs from batch record')
        require(digest(parquet) == record['runtime_sha256'], f'{episode}: runtime parquet changed')
        for relative, checksum in record['asset_sha256'].items():
            require(digest(frozen/relative) == checksum, f'{episode}: asset changed: {relative}')
        table = pq.read_table(parquet)
        require(table.num_rows == 1, f'{episode}: reference must contain one sequence row')
        source = frozen/'source_reference.parquet'
        require(digest(source) == record['source_sha256'], f'{episode}: source reference changed')
        if record.get('support_asset'):
            support_report = json.loads((frozen/'assets/support_report.json').read_text())
            require(support_report.get('physics_passed') is True
                    and support_report.get('source_reference_sha256') == record['source_sha256']
                    and support_report.get('support_sha256') == digest(frozen/record['support_asset']),
                    f'{episode}: measured support physics/provenance check failed')
        original = pq.read_table(source)
        invariant_columns = [name for name in original.column_names
                             if name not in ('object_mesh_paths', 'object_urdf_paths')]
        require(original.select(invariant_columns).equals(table.select(invariant_columns)),
                f'{episode}: freezing changed motion, object targets or contact labels')
        row = table.to_pylist()[0]
        reference = load_mano_sharpa(str(parquet))
        names = list(reference.object_body_names())
        frames, fps = reference.num_frames, reference.fps
        require(frames == record['frames'] and fps == record['fps'],
                f'{episode}: frame count/FPS disagree with frozen manifest')
        require(names == record['objects'], f'{episode}: object order disagrees with frozen manifest')
        public = roster[str(int(episode.split('_')[-1]))]
        require(names == public['object_names'], f'{episode}: objects differ from public roster')
        require(public['frame_indices'] == list(range(frames)), f'{episode}: public frame grid differs')
        require(public['world_indices'] == [0], f'{episode}: expected one public demonstration world')
        require(public['row_count'] == frames*len(names), f'{episode}: public row count differs')
        declared_assets = {(frozen/relative).resolve() for relative in record['asset_sha256']}
        for field in ('object_mesh_paths', 'object_urdf_paths'):
            for stored in row.get(field, []):
                target = Path(resolve_asset_path(stored, reference_path=parquet))
                if not target.is_absolute():
                    target = frozen/target
                target = target.resolve()
                require(target in declared_assets and target.is_file(),
                        f'{episode}: undeclared or missing {field} asset: {target}')
                if field == 'object_urdf_paths':
                    for mesh in ET.parse(target).findall('.//mesh'):
                        filename = mesh.get('filename')
                        require(bool(filename), f'{episode}: empty URDF mesh filename')
                        dependency = (target.parent/filename).resolve()
                        require(dependency in declared_assets and dependency.is_file(),
                                f'{episode}: URDF mesh outside the frozen closure: {dependency}')
        limits = record['controller_limits']
        measured_speeds = {}
        for side in ('left', 'right'):
            position = np.asarray(row[f'robot_{side}_wrist_position'], dtype=np.float64)
            quaternion = np.asarray(row[f'robot_{side}_wrist_wxyz'], dtype=np.float64)
            quaternion = quaternion/np.linalg.norm(quaternion, axis=1, keepdims=True)
            measured_speeds[side] = dict(
                linear=np.linalg.norm(np.diff(position, axis=0), axis=1)*fps,
                angular=2*np.arccos(np.clip(np.abs(np.sum(quaternion[1:]*quaternion[:-1], axis=1)), 0., 1.))*fps)
        for axis, suffix in (('linear', 'm_s'), ('angular', 'rad_s')):
            bound = limits[f'{axis}_speed_limit_{suffix}']
            measured = limits[f'reference_{axis}_speed_max_{suffix}']
            floor, ceiling = (1., 2.) if axis == 'linear' else (12., 20.)
            expected = min(max(floor, measured*limits['margin']), ceiling)
            require(np.isfinite([bound, measured, limits['margin']]).all() and measured >= 0
                    and limits['margin'] == 1.2 and np.isclose(bound, expected, rtol=1e-12, atol=0.),
                    f'{episode}: physical {axis} speed cap differs from the bounded reference-derived rule')
            require(limits[f'{axis}_hard_ceiling_{suffix}'] == ceiling
                    and np.isclose(measured, max(values[axis].max() for values in measured_speeds.values())),
                    f'{episode}: declared {axis} peak/ceiling does not match the reference')
            for side, values in measured_speeds.items():
                count = int(np.count_nonzero(values[axis] > bound))
                require(limits['hands'][side][f'{axis}_reference_exceedance_count'] == count,
                        f'{episode}: wrong {side} {axis} reference exceedance count')
            if measured > bound:
                print(f'[batch-preflight] {episode}: reference {axis} peak {measured:g} exceeds '
                      f'physical cap {bound:g}; raw reference is preserved and coverage is limited', flush=True)
        covered = all(limits[f'reference_{axis}_speed_max_{suffix}'] <= limits[f'{axis}_speed_limit_{suffix}']
                      for axis, suffix in (('linear', 'm_s'), ('angular', 'rad_s')))
        require(limits['reference_peak_covered'] == covered and limits['reference_motion_modified'] is False,
                f'{episode}: reference coverage metadata is misleading')
        inputs[episode] = dict(frames=frames, fps=fps, objects=names,
            runtime_sha256=record['runtime_sha256'], asset_files=len(declared_assets),
            controller_limits=limits)
        print(f'[batch-preflight] {episode}: {frames} frames, {len(names)} objects, inputs verified', flush=True)
    return inputs


def run_dir(episode, phase, seed):
    return ROOT/episode/'runs'/phase/episode/'chunk'/f'seed_{seed}'


def compose(episode, phase, seed):
    run = run_dir(episode, phase, seed)
    subprocess.run([sys.executable, str(ROOT/episode/'code/run_mixed.py'), '--phase', phase,
        '--seed', str(seed), '--dry-run'], check=True, cwd=WORKSPACE, stdout=subprocess.DEVNULL)
    recipe = json.loads((run/'run_config.json').read_text())
    resolved = subprocess.check_output([sys.executable, *recipe['training_command'],
        '--cfg', 'job', '--resolve'], text=True, cwd=WORKSPACE)
    import yaml
    cfg = yaml.safe_load(resolved)
    (run/'launch_resolved.yaml').write_text(resolved)
    return recipe, cfg, digest(run/'launch_resolved.yaml')


def physical_settings(cfg):
    return {key: deepcopy(cfg[key]) for key in ('task', 'sim', 'scene', 'collision',
        'embodiment', 'action', 'observation', 'objective', 'reset', 'env')}


def validate_config(episode, phase, recipe, cfg, record, bindings):
    require(recipe['episode'] == record, f'{episode}: wrong episode in composed command')
    require(recipe['source_manifest_sha256'] == bindings['source_manifest_sha256'],
            f'{episode}: command source hash mismatch')
    require(cfg['scene']['world_count'] == cfg['training']['world_count'] == 2048,
            f'{episode}: wrong vector world count')
    require(cfg['sim']['fps'] == record['fps'],
            f'{episode}: control/reference FPS mismatch requires an explicit resampled media/export contract')
    total, checkpoint = (1048576, 1048576) if phase == 'smoke' else (100663296, 5017600)
    require(cfg['training']['total_environment_steps'] == total, f'{episode}: wrong {phase} main budget')
    require(cfg['training']['checkpoint']['save_interval_environment_steps'] == checkpoint,
            f'{episode}: wrong {phase} checkpoint cadence')
    replay = cfg['training']['replay']
    require((replay['capacity'], replay['minimum_size'], replay['batch_size']) == (1048576, 102400, 1024),
            f'{episode}: lower replay settings changed')
    upper = cfg['bilevel']
    scope = json.loads((ROOT/'batch_manifest.json').read_text()).get('upper_edit_scope', 'full')
    require(upper['edit_scope'] == recipe['upper_edit_scope'] == scope
            and recipe['upper_action_dimension'] == (12 if scope == 'wrists_only' else 56),
            f'{episode}: upper refinement scope/dimension mismatch')
    require(bool(cfg['scene']['support']) == bool(record.get('support_asset')),
            f'{episode}: measured support setting mismatch')
    require(cfg['collision'].get('ground_plane') == record.get('ground_plane'),
            f'{episode}: measured floor setting mismatch')
    require(cfg['scene'].get('object_initial_pose_overrides') == record.get('object_initial_pose_overrides'),
            f'{episode}: initial object pose configuration mismatch')
    if record.get('object_initial_pose_overrides') is not None:
        from initial_state import validate_initial_state_report
        import pyarrow.parquet as pq
        initial_row = pq.ParquetFile(ROOT/'frozen'/episode/'reference.parquet').read().to_pylist()[0]
        report = validate_initial_state_report(ROOT/'frozen'/episode/record['initial_state_report'],
            episode=episode, source_sha256=record['source_sha256'],
            frozen_directory=ROOT/'frozen'/episode, object_names=record['objects'],
            object_urdf_paths=initial_row['object_urdf_paths'])
        require(all(record.get(key) == value for key, value in report.items()),
                f'{episode}: initial-state record differs from validated evidence')
    require(upper['engine'] == 'wrench_hypergradient' and upper['warmup'] == upper['interval'] == 501760,
            f'{episode}: wrong upper engine or schedule')
    require(upper['critic_online_interval_vector_steps'] == 16 and upper['critic_online_updates'] == 4
            and upper['diagnostic_save_trials'] is True, f'{episode}: critic schedule/evidence changed')
    hyper = upper['hypergradient']
    for key, expected in dict(inner_optimizer='adam', inner_steps=3, cost_scale=.01,
            trial_mc_weight=1., trial_critic_updates=32, inner_adam_eps_root=1e-12).items():
        require(hyper[key] == expected, f'{episode}: wrong hypergradient {key}')
    adaptation = upper['adaptation']
    for key, expected in dict(acceptance_mode='finite_response', maximum_confirmations=2,
            budget_mode='bounded_window', minimum_vector_steps=64,
            checkpoint_stride_vector_steps=32, maximum_vector_steps=448,
            window_tail_vector_steps=16).items():
        require(adaptation[key] == expected, f'{episode}: wrong adaptation {key}')
    for key, expected in dict(wrist_pos_clip=.05, wrist_ori_clip=.5,
            wrist_residual_linear_speed_limit_m_s=.15,
            wrist_residual_angular_speed_limit_rad_s=1.5).items():
        require(cfg['action'][key] == cfg['env']['action'][key] == expected,
                f'{episode}: action setting did not reach the resolved environment: {key}')
    require(cfg['observation']['object_body_velocity']['enabled'] is True
            and cfg['env']['observation']['object_body_velocity']['enabled'] is True,
            f'{episode}: lower actor must observe object velocity')
    for axis, suffix in (('linear', 'm_s'), ('angular', 'rad_s')):
        key = f'wrist_{axis}_speed_limit_{suffix}'
        require(cfg['env'][key] == record['controller_limits'][f'{axis}_speed_limit_{suffix}'],
                f'{episode}: physical speed cap mismatch: {key}')
    require(cfg['reset']['fixed_initial_state'] and cfg['reset']['always_reset_to_first_frame']
            and cfg['reset']['voc_decay_steps'] == 0 and cfg['env']['voc_scale'] == 1.,
            f'{episode}: initial state/curriculum changed')


def check_scene(episode, cfg, expected_frames):
    import warp as wp
    import flash_chord.embodiments.sharpa_hands  # Register the embodiment.
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from flash_chord.scene.setup import setup_scene
    with wp.ScopedDevice('cpu'):
        setup = setup_scene(parquet=ROOT/'frozen'/episode/'reference.parquet',
            control_fps=cfg['sim']['fps'], motion_speed=cfg['task']['motion_speed'],
            embodiment=instantiate(OmegaConf.create(cfg['embodiment'])),
            collision=instantiate(OmegaConf.create(cfg['collision'])), world_count=1,
            include_support=cfg['scene']['support'], decompose_objects=cfg['scene']['decompose_objects'],
            object_initial_pose_overrides=cfg['scene'].get('object_initial_pose_overrides'))
        require(setup.reference.num_frames == expected_frames and
                setup.scene.robot_reference.joint_q.shape[0] == expected_frames,
                f'{episode}: assembled scene changed the reference length')
        report = dict(passed=True, worlds=1, frames=expected_frames,
                      scope='CPU scene assembly and full referenced robot/object asset closure')
    del setup
    gc.collect()
    return report


def cpu_tests(bindings):
    report_path = ROOT/'reports/cpu_tests.json'
    if report_path.is_file():
        previous = json.loads(report_path.read_text())
        if (previous.get('passed') and previous.get('tests') == list(TESTS)
                and all(previous.get(key) == value for key, value in bindings.items())):
            print('[batch-preflight] Reusing successful CPU tests for the exact frozen manifests', flush=True)
            return previous
    for relative in TESTS:
        require((WORKSPACE/relative.split('::', 1)[0]).is_file(), f'Missing required CPU test: {relative}')
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *TESTS]
    started = time.monotonic()
    log = ROOT/'reports/cpu_tests.log'
    print('[batch-preflight] Running shared CPU critic, acceptance, wrist-control and submission tests', flush=True)
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=WORKSPACE, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'JAX_PLATFORMS': 'cpu', 'CUDA_VISIBLE_DEVICES': '',
                 'NVIDIA_VISIBLE_DEVICES': 'void', 'PYTHONDONTWRITEBYTECODE': '1'})
    report = dict(passed=result.returncode == 0, returncode=result.returncode,
        seconds=time.monotonic()-started, tests=list(TESTS), command=command,
        log=str(log), log_sha256=digest(log), summary=log.read_text().splitlines()[-8:], **bindings)
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print('\n'.join(report['summary']), flush=True)
    require(report['passed'], f'CPU tests failed; see {log}')
    return report


def main():
    (ROOT/'reports').mkdir(exist_ok=True)
    bindings = verify_manifests()
    batch = json.loads((ROOT/'batch_manifest.json').read_text())
    seed = int(batch['seed'])
    # Never overwrite the configuration of a run that has already launched.
    for episode in batch['episodes']:
        for phase in ('smoke', 'pilot'):
            run = run_dir(episode, phase, seed)
            require(not (run/'stage.json').exists()
                    and not list((run/'training').glob('policy_*.safetensors'))
                    and not (run/'training/upper_updates.jsonl').exists(),
                    f'Fresh preflight refuses to modify a launched run: {run}')
    inputs = validate_inputs(batch)
    checks = []
    from flash_chord.retargeting.lower_adaptation import AdaptationConfig
    for episode in batch['episodes']:
        pair, hashes = {}, {}
        for phase in ('smoke', 'pilot'):
            recipe, cfg, hashes[phase] = compose(episode, phase, seed)
            validate_config(episode, phase, recipe, cfg, batch['records'][episode], bindings)
            pair[phase] = cfg
        require(physical_settings(pair['smoke']) == physical_settings(pair['pilot']),
                f'{episode}: smoke/full physical settings differ')
        require(pair['smoke']['training']['network'] == pair['pilot']['training']['network'],
                f'{episode}: smoke/full network settings differ')
        require(pair['smoke']['bilevel'] == pair['pilot']['bilevel'],
                f'{episode}: smoke/full upper optimization differs')
        frames = inputs[episode]['frames']
        adaptation = AdaptationConfig(**pair['pilot']['bilevel']['adaptation'])
        late = adaptation.budget_report(2048, frames,
            region=dict(start_frame=max(0, frames-20), stop_frame_exclusive=frames))
        scene = check_scene(episode, pair['pilot'], frames)
        checks.append(dict(episode=episode, seed=seed, inputs=inputs[episode],
            resolved_config_sha256=hashes, physical_settings_match=True, scene=scene,
            end_of_episode_adaptation_budget=late))
        print(f'[batch-preflight] {episode}: smoke/full config and CPU scene verified; '
              f'late-window nominal coverage={late["final_checkpoint_can_reach_scoring_end"]}', flush=True)
    tests = cpu_tests(bindings)
    # Catch edits during composition/testing before certifying a frozen batch.
    require(verify_manifests() == bindings, 'Frozen manifests changed during preflight')
    report = dict(passed=True, scope='CPU input/configuration/scene/test validation; GPU integration still required',
        episode_count=len(checks), seed=seed, checks=checks, cpu_tests_report='reports/cpu_tests.json',
        public_episode_count=batch['public_episode_count'],
        missing_public_episodes=batch['missing_public_episodes'],
        public_coverage_complete=batch['public_coverage_complete'],
        cpu_tests_log_sha256=tests['log_sha256'], **bindings)
    (ROOT/'reports/batch_preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'[batch-preflight] Passed all {len(checks)} selected public episodes; '
          f'{len(batch["missing_public_episodes"])} public episodes were not selected', flush=True)


if __name__ == '__main__':
    main()
