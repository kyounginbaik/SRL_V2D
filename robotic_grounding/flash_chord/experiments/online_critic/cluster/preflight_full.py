"""CPU-only configuration checks; this does not certify a GPU smoke result."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
root = Path(__file__).resolve().parents[1]
base = root/'episode_000001'
workspace = root/'flash_chord'


def output_dir(phase, seed):
    return base/f'runs/{phase}/episode_000001/chunk/seed_{seed}'


def compose(phase, seed):
    subprocess.run([sys.executable, str(base/'code/run_mixed.py'), '--phase', phase,
                    '--seed', str(seed), '--dry-run'], check=True, stdout=subprocess.DEVNULL,
                   cwd=workspace)
    run = output_dir(phase, seed)
    config = json.loads((run/'run_config.json').read_text())
    composed = subprocess.check_output([sys.executable, *config['training_command'],
                                       '--cfg', 'job', '--resolve'], text=True, cwd=workspace)
    import yaml
    cfg = yaml.safe_load(composed)
    (run/'launch_resolved.yaml').write_text(composed)
    return run, config, cfg, composed


def physical_settings(cfg):
    # Vary only the requested seed and number of training worlds. Compare the
    # fully resolved nested environment too, so a config alias cannot drift.
    keys = ('task', 'sim', 'scene', 'collision', 'embodiment', 'action',
            'observation', 'objective', 'reset', 'env')
    physics = {key: deepcopy(cfg[key]) for key in keys}
    physics['scene'].pop('world_count')
    physics['reset'].pop('seed')
    physics['env']['reset'].pop('seed')
    return physics


def assert_current_hypergradient(cfg):
    assert cfg['bilevel']['critic_online_interval_vector_steps'] == 16
    assert cfg['bilevel']['critic_online_updates'] == 4
    assert cfg['bilevel']['diagnostic_save_trials'] is True
    options = cfg['bilevel']['hypergradient']
    assert options['inner_optimizer'] == 'adam'
    assert cfg['bilevel']['adaptation']['acceptance_mode'] == 'finite_response'
    assert cfg['bilevel']['adaptation']['maximum_confirmations'] == 2
    assert options['cost_scale'] == .01
    assert options['trial_mc_weight'] == 1.
    assert options['inner_adam_eps_root'] == 1e-12
    assert options['trial_critic_updates'] == 32


def main():
    assert (workspace/'tests/retargeting/test_actor_response.py').is_file(), 'Missing Adam response tests'
    assert (workspace/'tests/retargeting/test_finite_response_acceptance.py').is_file()
    # Check every destination before writing any dry-run config. A repeated
    # preflight must never overwrite the configuration of an active run.
    for phase, seed in [('smoke', 43), ('pilot', 43), ('pilot', 44), ('pilot', 45)]:
        run = output_dir(phase, seed)
        training = run/'training'
        if ((run/'stage.json').exists() or list(training.glob('policy_*.safetensors'))
                or (training/'upper_updates.jsonl').exists()):
            raise RuntimeError(f'Fresh-run preflight refuses to touch launched run: {run}')
    _, smoke, smoke_cfg, _ = compose('smoke', 43)
    physics = physical_settings(smoke_cfg)
    assert_current_hypergradient(smoke_cfg)
    assert smoke_cfg['scene']['world_count'] == smoke_cfg['training']['world_count'] == 2048
    assert smoke_cfg['training']['total_environment_steps'] == 1048576
    assert smoke_cfg['training']['replay']['batch_size'] == 1024
    assert smoke_cfg['training']['replay']['capacity'] == 1048576
    assert smoke_cfg['training']['replay']['minimum_size'] == 102400
    assert smoke_cfg['bilevel']['interval'] == smoke_cfg['bilevel']['warmup'] == 501760
    assert smoke['episode']['pot_mass_kg'] == .2
    assert smoke['episode']['lid_mass_kg'] == .1
    assert smoke_cfg['bilevel']['engine'] == 'wrench_hypergradient'
    assert smoke_cfg['bilevel']['adaptation']['budget_mode'] == 'bounded_window'
    assert smoke_cfg['bilevel']['adaptation']['maximum_vector_steps'] == 448
    assert smoke_cfg['bilevel']['hypergradient']['trial_critic_updates'] == 32
    checks = []
    from flash_chord.retargeting.lower_adaptation import AdaptationConfig
    for seed in (43, 44, 45):
        _, config, cfg, composed = compose('pilot', seed)
        assert_current_hypergradient(cfg)
        assert config['experiment_name'] == smoke['experiment_name'] == 'wrench-stable-control'
        assert config['episode'] == smoke['episode']
        for key in ('wrist_control', 'eigen_basis_sha256', 'fixed_initial_state',
                    'hand_keypoint_reward_weight', 'finger_joint_tracking_reward_weight',
                    'voc_scales', 'source_manifest_sha256'):
            assert config[key] == smoke[key], key
        assert physical_settings(cfg) == physics, f'physical settings drift for seed {seed}'
        assert cfg['training']['network'] == smoke_cfg['training']['network']
        assert cfg['bilevel']['engine'] == 'wrench_hypergradient'
        assert cfg['scene']['world_count'] == cfg['training']['world_count'] == 2048
        assert cfg['training']['total_environment_steps'] == 100663296
        assert cfg['training']['replay']['capacity'] == 1048576
        assert cfg['training']['replay']['minimum_size'] == 102400
        assert cfg['training']['replay']['batch_size'] == 1024
        assert cfg['training']['checkpoint']['save_interval_environment_steps'] == 5017600
        assert cfg['bilevel']['eval_worlds'] == cfg['bilevel']['upper_trials'] == 2048
        assert cfg['bilevel']['interval'] == cfg['bilevel']['warmup'] == 501760
        adaptation = cfg['bilevel']['adaptation']
        assert adaptation == smoke_cfg['bilevel']['adaptation']
        assert adaptation['budget_mode'] == 'bounded_window'
        assert adaptation['minimum_vector_steps'] == 64
        assert adaptation['checkpoint_stride_vector_steps'] == 32
        assert adaptation['maximum_vector_steps'] == 448
        assert adaptation['window_tail_vector_steps'] == 16
        assert cfg['bilevel']['hypergradient']['trial_critic_updates'] == 32
        assert cfg['bilevel']['hypergradient'] == smoke_cfg['bilevel']['hypergradient']
        assert cfg['reset']['fixed_initial_state'] is True
        assert cfg['reset']['always_reset_to_first_frame'] is True
        assert cfg['reset']['voc_decay_steps'] == 0
        assert cfg['env']['voc_scale'] == 1.
        settings = AdaptationConfig(**adaptation)
        frames = config['episode']['frames']
        early = settings.budget_report(2048, frames,
            region=dict(start_frame=20, stop_frame_exclusive=53))
        late = settings.budget_report(2048, frames,
            region=dict(start_frame=frames-20, stop_frame_exclusive=frames))
        assert early['environment_steps'] == [141312, 206848, 272384]
        assert late['final_checkpoint_can_reach_scoring_end']
        assert late['final_checkpoint_includes_requested_tail']
        assert max(late['vector_steps']) <= 448
        checks.append(dict(seed=seed, phase='pilot', worlds=2048, main_steps=100663296,
            upper_interval=501760, adaptation_early_example=early,
            adaptation_end_of_episode_example=late, trial_critic_updates=32,
            inner_optimizer='adam', inner_adam_eps_root=1e-12,
            upper_cost_scale=.01, trial_mc_weight=1.,
            physical_settings_equal_to_seed43_smoke=True,
            resolved_config_sha256=hashlib.sha256(composed.encode()).hexdigest()))
        print(f'seed {seed}: full configuration verified; physics matches seed43 smoke config', flush=True)
    subprocess.run([sys.executable, str(root/'cluster/check_scene.py')], check=True, cwd=workspace)
    report = dict(passed=True, scope='CPU configuration composition and scene assembly; GPU smoke has not been certified',
                  smoke_reference_seed=43, checks=checks)
    (root/'reports').mkdir(parents=True, exist_ok=True)
    (root/'reports/full_launch_preflight.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
