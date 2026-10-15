"""CPU-only checks for portable recipes and both W&B upper-record schemas."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, PACKAGE/'templates'/f'{name}.py')
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class LaunchRecipesTest(unittest.TestCase):
    def test_generated_hypergradient_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root/'episode_000001'
            code = base/'code'
            frozen = base/'frozen/episode_000001'
            for folder in (code, frozen, base/'priors'):
                folder.mkdir(parents=True)
            reference = frozen/'reference.parquet'
            reference.write_bytes(b'only hashing is needed by launcher dry run')
            (base/'priors/sharpa_bimanual_residual_basis.npz').write_bytes(b'prior')
            (root/'source_manifest.json').write_text('{}')
            (base/'frozen_manifest.json').write_text(json.dumps([dict(
                runtime_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), asset_sha256={})]))
            for name in ('run_mixed.py', 'hypergradient_launch.py'):
                (code/name).write_text((PACKAGE/'templates'/name).read_text().replace(
                    '__UPPER_OPTIMIZER__', 'wrench_hypergradient'))
            result = subprocess.run([sys.executable, str(code/'run_mixed.py'), '--dry-run',
                                     '--phase', 'smoke', '--seed', '45'], check=True,
                                    text=True, capture_output=True)
            config = json.loads(result.stdout)
            command = config['training_command']
            self.assertEqual(config['worlds'], 64)
            self.assertEqual(config['planned_upper_search_rounds'], 1)
            self.assertEqual(config['upper_action_dimension'], 56)
            self.assertEqual(config['maximum_proposals_per_round'], 4)
            self.assertNotIn('upper_trials', config)
            self.assertNotIn('lower_surrogate', config)
            self.assertIn('+bilevel=wrench_hypergradient', command)
            self.assertIn('bilevel.interval=65536', command)
            self.assertIn('training.total_environment_steps=131072', command)
            self.assertIn('training.replay.capacity=262144', command)
            self.assertIn('training.replay.minimum_size=8192', command)
            self.assertIn('training.replay.batch_size=1024', command)
            self.assertIn('reset.fixed_initial_state=true', command)
            self.assertFalse(any(arg.startswith('+bilevel.') for arg in command))
            self.assertEqual(len(command), len(set(arg.split('=')[0] for arg in command)))

    def test_full_recipe_keeps_budget_and_replay(self):
        command = ['scripts/train_flash_sac.py', '+bilevel.engine=response_bilevel',
                   '+bilevel.upper_proposal=flashsac', 'training.replay.capacity=100']
        config = dict(lower_response_error_tolerance=1e-5, upper_trials=2048)
        module('hypergradient_launch').configure_hypergradient(command, config, 'pilot', 2048, Path('/tmp'))
        self.assertIn('training.total_environment_steps=100663296', command)
        self.assertIn('training.replay.capacity=1048576', command)
        self.assertIn('bilevel.interval=501760', command)
        self.assertEqual(config['adaptation_minimum_reference_passes'], [1, 2, 4])
        self.assertNotIn('lower_response_error_tolerance', config)

    def test_upper_records_accept_both_engines_and_skipped_windows(self):
        summary = module('wandb_sync').candidate_summary
        result = summary([dict(cost=2., accepted=True), dict(skipped='no replay'),
                          dict(cost=4., accepted=False)])
        self.assertEqual(result['candidate_count'], 2)
        self.assertEqual(result['skipped_window_count'], 1)
        self.assertEqual(result['gate_pass_fraction'], .5)
        self.assertEqual(result['candidate_cost_mean'], 3.)
        self.assertEqual(summary([dict(cost=1., gate_passed=True)])['gate_pass_fraction'], 1.)

    def test_launch_cli_supports_hypergradient(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(PACKAGE/'launch.py'), '--dry-run',
                '--phase', 'smoke', '--upper', 'wrench_hypergradient', '--run-root', directory],
                check=True, text=True, capture_output=True)
            self.assertIn('--upper wrench_hypergradient', result.stdout)
            self.assertIn('--phase smoke', result.stdout)


if __name__ == '__main__':
    unittest.main()
