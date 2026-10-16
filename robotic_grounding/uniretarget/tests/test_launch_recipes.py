"""CPU-only checks for portable recipes and both W&B upper-record schemas."""
import hashlib
import copy
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
            self.assertEqual(config['worlds'], 2048)
            self.assertEqual(config['planned_upper_search_rounds'], 1)
            self.assertEqual(config['upper_action_dimension'], 56)
            self.assertEqual(config['maximum_proposals_per_round'], 4)
            self.assertNotIn('upper_trials', config)
            self.assertNotIn('lower_surrogate', config)
            self.assertIn('+bilevel=wrench_hypergradient', command)
            self.assertIn('bilevel.hypergradient.inner_optimizer=adam', command)
            self.assertIn('bilevel.hypergradient.cost_scale=0.01', command)
            self.assertIn('bilevel.hypergradient.trial_mc_weight=1.0', command)
            self.assertIn('bilevel.adaptation.acceptance_mode=finite_response', command)
            self.assertIn('bilevel.interval=501760', command)
            self.assertIn('+adaptive_voc.interval=501760', command)
            self.assertIn('training.total_environment_steps=524288', command)
            self.assertIn('training.replay.capacity=1048576', command)
            self.assertIn('training.replay.minimum_size=102400', command)
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
        self.assertIn('bilevel.adaptation.budget_mode=bounded_window', command)
        self.assertIn('bilevel.adaptation.acceptance_mode=finite_response', command)
        self.assertIn('bilevel.adaptation.maximum_vector_steps=448', command)
        self.assertEqual(config['adaptation_maximum_transitions_per_branch'], 917504)
        self.assertEqual(config['experiment_name'], 'wrench-finite-response')
        self.assertEqual(config['adaptation_acceptance_mode'], 'finite_response')
        self.assertEqual(config['proposal_inner_optimizer'], 'adam')
        self.assertEqual(config['upper_cost_scale'], .01)
        self.assertEqual(config['trial_mc_weight'], 1.)
        self.assertNotIn('adaptation_minimum_reference_passes', config)
        self.assertNotIn('lower_response_error_tolerance', config)

    def test_smoke_rejects_disconnected_feedback_or_exceeded_cap(self):
        validate = module('hypergradient_smoke').validate_feedback_and_budget
        config = dict(worlds=2048, adaptation_maximum_vector_steps=256,
                      adaptation_maximum_transitions_per_branch=524288)
        budgets = [131072, 196608, 262144]
        row = dict(adaptation_checkpoints=budgets,
            adaptation_budget=dict(mode='bounded_window', world_count=2048,
                maximum_vector_steps=256, environment_steps=budgets, vector_steps=[64, 96, 128]),
            candidates=[dict(gradient_norm=1., accepted=False, window=dict(window_id=0),
                adaptation=dict(checkpoints=[dict(environment_steps=count) for count in budgets]))],
            critic_feedback=[dict(branch=branch, environment_steps=count,
                trial_feedback_rows=40, trial_feedback_complete_rows=40,
                trial_critic_skipped=0, trial_critic_updates=32,
                trial_critic_parameter_delta_l2=.1, trial_critic_cost_scale=.01)
                for branch, count in [('baseline', 0)]+
                    [(branch, count) for branch in ('noedit', 'window_0') for count in budgets]])
        self.assertEqual(validate(row, config)['critic_feedback_complete_rows'], 280)
        missing = copy.deepcopy(row)
        missing['critic_feedback'] = [r for r in missing['critic_feedback'] if r['branch'] == 'noedit']
        with self.assertRaisesRegex(AssertionError, 'Missing baseline/candidate/control'):
            validate(missing, config)
        missing_baseline = copy.deepcopy(row)
        missing_baseline['critic_feedback'] = [r for r in missing_baseline['critic_feedback']
                                               if r['branch'] != 'baseline']
        with self.assertRaisesRegex(AssertionError, 'Missing baseline/candidate/control'):
            validate(missing_baseline, config)
        unchanged = copy.deepcopy(row)
        unchanged['critic_feedback'][-1]['trial_critic_parameter_delta_l2'] = 0
        with self.assertRaisesRegex(AssertionError, 'did not change'):
            validate(unchanged, config)
        unbounded = copy.deepcopy(config)
        unbounded['adaptation_maximum_transitions_per_branch'] = 200000
        with self.assertRaises(AssertionError):
            validate(row, unbounded)

    def test_smoke_rejects_stale_generation_or_reset_adam_state(self):
        validate = module('hypergradient_smoke').validate_current_proposal
        row = dict(upper_round=3, outer_validation_generation=3)
        candidate = dict(validation_generation=3, inner_optimizer='adam',
                         inner_optimizer_state_preserved=True, critic_cost_scale=.01)
        validate(row, candidate)
        with self.assertRaisesRegex(AssertionError, 'Stale candidate'):
            validate(row, {**candidate, 'validation_generation': 2})
        with self.assertRaisesRegex(AssertionError, 'Stale outer'):
            validate({**row, 'outer_validation_generation': 2}, candidate)
        with self.assertRaisesRegex(AssertionError, 'Adam state'):
            validate(row, {**candidate, 'inner_optimizer_state_preserved': False})
        with self.assertRaisesRegex(AssertionError, 'cost scale'):
            validate(row, {**candidate, 'critic_cost_scale': 1.})

    @staticmethod
    def finite_acceptance_row():
        budgets = [131072, 196608, 262144]
        confirmation = dict(attempted=True, passed=True, environment_steps=budgets[0], window_id=0)
        candidate = dict(gradient_norm=1., accepted=True, window=dict(window_id=0),
            physical_passed_at_selected_checkpoint=True, rejection_reasons=[],
            confirmation=copy.deepcopy(confirmation),
            adaptation=dict(acceptance_mode='finite_response', stable=False,
                selected_environment_steps=budgets[0],
                checkpoints=[dict(environment_steps=budget,
                    candidate=dict(training=dict(environment_steps=budget)),
                    control=dict(training=dict(environment_steps=budget))) for budget in budgets]))
        return dict(accepted=True, adaptation_checkpoints=budgets, candidates=[candidate],
                    confirmation=dict(attempted=True, passed=True, attempts=[confirmation]))

    def test_finite_acceptance_allows_confirmed_early_checkpoint_without_plateau(self):
        validate = module('hypergradient_smoke').validate_finite_acceptance
        row = self.finite_acceptance_row()
        validate(row)
        # Another physically useful proposal is not published if the selected
        # pair already confirmed. Acceptance counts actual reference changes.
        not_selected = copy.deepcopy(row['candidates'][0])
        not_selected.update(accepted=False, window=dict(window_id=1))
        not_selected.pop('confirmation')
        row['candidates'].append(not_selected)
        validate(row)
        rejected = self.finite_acceptance_row()
        rejected.update(accepted=False, confirmation=dict(attempted=False, passed=False, attempts=[]))
        rejected['candidates'][0].update(accepted=False, physical_passed_at_selected_checkpoint=False,
                                        rejection_reasons=['no_local_improvement'])
        rejected['candidates'][0].pop('confirmation')
        validate(rejected)

    def test_finite_acceptance_rejects_unconfirmed_or_unmatched_publication(self):
        validate = module('hypergradient_smoke').validate_finite_acceptance
        unconfirmed = self.finite_acceptance_row()
        unconfirmed['candidates'][0]['confirmation']['passed'] = False
        with self.assertRaisesRegex(AssertionError, 'lacks frozen confirmation'):
            validate(unconfirmed)
        mismatched = self.finite_acceptance_row()
        mismatched['candidates'][0]['confirmation']['environment_steps'] = 262144
        with self.assertRaisesRegex(AssertionError, 'different adaptation checkpoint'):
            validate(mismatched)
        unmatched_training = self.finite_acceptance_row()
        unmatched_training['candidates'][0]['adaptation']['checkpoints'][0]['control']['training']['environment_steps'] = 262144
        with self.assertRaisesRegex(AssertionError, 'Unmatched selected training budgets'):
            validate(unmatched_training)
        unchecked = self.finite_acceptance_row()
        unchecked['candidates'][0]['physical_passed_at_selected_checkpoint'] = False
        with self.assertRaisesRegex(AssertionError, 'failed physical checks'):
            validate(unchecked)

    def test_finite_acceptance_rejects_partial_budget_or_multiple_publications(self):
        validate = module('hypergradient_smoke').validate_finite_acceptance
        incomplete = self.finite_acceptance_row()
        incomplete['candidates'][0]['adaptation']['checkpoints'].pop()
        with self.assertRaisesRegex(AssertionError, 'all matched budgets'):
            validate(incomplete)
        duplicate = self.finite_acceptance_row()
        duplicate['candidates'].append(copy.deepcopy(duplicate['candidates'][0]))
        with self.assertRaisesRegex(AssertionError, 'Published candidate count'):
            validate(duplicate)

    def test_feedback_panels_include_rejected_trial_learning(self):
        summary = module('wandb_sync').feedback_summary
        result = summary([dict(branch='window_0', accepted=False, trial_critic_skipped=0,
                               trial_feedback_rows=30, trial_feedback_complete_rows=20,
                               trial_feedback_discarded_rows=10, trial_feedback_trajectories=1,
                               trial_critic_parameter_delta_l2=.2,
                               trial_critic_heldout_return_mae_before=3.,
                               trial_critic_heldout_return_mae_after=1.,
                               trial_critic_heldout_return_mae_after_normalized=1.,
                               trial_critic_heldout_return_mae_after_raw=100.,
                               trial_critic_cost_scale=.01),
                          dict(branch='noedit', trial_critic_skipped=1,
                               trial_feedback_rows=1, trial_feedback_discarded_rows=1)])
        self.assertEqual(result['fitted_branch_checkpoints'], 1)
        self.assertEqual(result['trial_feedback_complete_rows'], 20)
        self.assertEqual(result['trial_feedback_discarded_rows'], 11)
        self.assertEqual(result['trial_critic_parameter_delta_l2_max'], .2)
        self.assertEqual(result['trial_critic_heldout_return_mae_after'], 1.)
        self.assertEqual(result['trial_critic_heldout_return_mae_after_normalized'], 1.)
        self.assertEqual(result['trial_critic_heldout_return_mae_after_raw'], 100.)
        self.assertEqual(result['trial_critic_cost_scale'], .01)

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
