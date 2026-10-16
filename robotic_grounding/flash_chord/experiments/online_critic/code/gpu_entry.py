"""Validate the nested search, then start full training in the same allocation."""
from pathlib import Path
import argparse,json,subprocess,sys,os
import numpy as np

b=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=43)
p.add_argument('--phase',choices=['smoke','pilot','full'],default='full');a=p.parse_args()
if a.phase in ('smoke','full'):
 subprocess.run([sys.executable,'-m','pytest','tests/retargeting/test_response_bilevel.py',
  'tests/retargeting/test_response_policy.py','tests/retargeting/test_response_metrics.py','tests/retargeting/test_local_acceptance.py','tests/retargeting/test_fixed_start.py',
  'tests/retargeting/test_chord_upper.py','tests/retargeting/test_response_upper.py','tests/evaluation/test_track3_estimates.py','tests/test_adaptive_windows.py',
  'tests/test_failure_consensus.py','tests/test_local_correction.py','tests/test_rejected_search_callback.py',
  'tests/runtime/test_wrist_limits.py','-q','-p','no:cacheprovider'],
  env={**os.environ,'JAX_PLATFORMS':'cpu'},check=True)
 if 'wrench_hypergradient' == 'wrench_hypergradient':
  subprocess.run([sys.executable,'-m','pytest','tests/retargeting/test_lower_adaptation.py',
   'tests/retargeting/test_reference_relabel.py','tests/retargeting/test_actor_response.py',
   'tests/retargeting/test_finite_response_acceptance.py',
   'tests/retargeting/test_wrench_hypergradient.py',
   'tests/retargeting/test_wrench_hypergradient_editor.py','-q','-p','no:cacheprovider'],
   env={**os.environ,'JAX_PLATFORMS':'cpu'},check=True)
for phase in (('smoke','pilot') if a.phase=='full' else (a.phase,)):
 subprocess.run([sys.executable,str(b/'code/run_sequence.py'),'--seed',str(a.seed),'--phase',phase],check=True)
 if phase=='smoke':
  run=b/'runs/smoke/episode_000001/chunk'/f'seed_{a.seed}'
  config=json.loads((run/'run_config.json').read_text())
  assert config['contact_proximity'] is False
  manifest=json.loads((run/'training/ttsa_manifest.json').read_text())
  assert manifest['contact_location_penalty'] is False and manifest['contact_location_gate'] is False
  rows=[json.loads(line) for line in (run/'training/upper_updates.jsonl').read_text().splitlines()]
  if config.get('engine') == 'wrench_hypergradient':
   from hypergradient_smoke import validate
   evidence=validate(run,config,manifest,rows)
  else:
   assert rows and all(row['lower_response_solved_for_every_candidate'] for row in rows)
   if 'wrench_hypergradient' == 'flashsac':
    assert all(row['upper_proposal']=='flashsac' and row['upper_optimizer_updates_this_round']>0 and row['upper_actor_parameter_max_change']>0 for row in rows)
   assert all(row['lower_parameter_error_bound_max']<=1e-5 for row in rows)
   for row in rows:
    assert row['acceptance_mode']=='local_nominal' and row['confirmation_reset_count']==0
    assert row['confirmation_attempts']==[]
    region=row['scoring_region']
    assert region['stop_frame_exclusive']>region['start_frame']
    if row['accepted']:
     assert row['local_cost_gain']>1e-4
     chosen=row['candidates'][row['selected_trial']]
     assert chosen['rejection_reasons']==[]
     assert chosen['cost']==min(c['cost'] for c in row['candidates'] if c['gate_passed'])
    failure=row['failure_context']
    assert failure['detector']=='fixed_start_nominal' and failure['baseline_worlds']==1
    assert failure['baseline_probe_translation_m']==0. and failure['baseline_probe_finger_degrees']==0.
    assert row['fixed_initial_state'] and len(row['initial_state_sha256'])==64
    probe=np.load(run/'training/search_batches'/f'step_{row["step"]}'/'nominal_baseline.npz')
    assert probe['position_m'].shape[1:]==(1,len(config['episode']['objects']))
    assert np.isfinite(probe['position_m']).all() and np.isfinite(probe['orientation_rad']).all()
    assert not list((run/'training/search_batches').glob('*/baseline_ensemble.npz'))
    for candidate in row['candidates']:
     assert np.isfinite(candidate['cost'])
     assert candidate['metrics']['cws_frames']>0
     assert candidate['metrics']['local_cws_frames']==region['stop_frame_exclusive']-region['start_frame']
     assert candidate['metrics']['local_start_frame']==region['start_frame']
     assert candidate['metrics']['local_stop_frame_exclusive']==region['stop_frame_exclusive']
     assert len(candidate['metrics']['per_object_cws_tolerance_error'])==len(config['episode']['objects'])
     assert 'contact_proximity_mean_m' not in candidate['metrics']
   audits=[json.loads(line) for line in (run/'training/initial_state_checks.jsonl').read_text().splitlines()]
   assert len({v['initial_state_sha256'] for v in audits})==1
   assert all(v['frame']==0 and v['joint_velocity_max_error']==0 and v['joint_position_max_error']<=1e-6 for v in audits)
   assert {'training_start','candidate_batch','nominal_rollout'} <= {v['context'] for v in audits}
   best=json.loads((run/'training/best_demo.json').read_text())
   assert Path(best['policy']).is_file() and Path(best['reference']).is_file()
   progress=[json.loads(line) for line in (run/'training/progress_checks.jsonl').read_text().splitlines()]
   assert np.all(np.diff([v['best_success_prefix_seconds'] for v in progress])>=0)
   assert any(v['origin']=='checkpoint' for v in progress)
   evidence=dict(searches=len(rows),accepted=sum(row['accepted'] for row in rows),
    maximum_lower_error=max(row['lower_parameter_error_bound_max'] for row in rows))
  (b.parent/'reports'/f'smoke_validation_{b.name}_s{a.seed}.json').write_text(json.dumps(dict(
   passed=True,seed=a.seed,**evidence,
   next_phase='pilot' if a.phase=='full' else None),indent=2)+'\n')
  print('[uniretarget] GPU integration validation passed; '+
        ('starting 2048-world full training' if a.phase=='full' else 'smoke complete'),flush=True)
