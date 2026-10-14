"""Validate the nested search, then start full training in the same allocation."""
from pathlib import Path
import argparse,json,subprocess,sys,os
import numpy as np

b=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=43)
p.add_argument('--phase',choices=['smoke','pilot','full'],default='full');a=p.parse_args()
if a.phase in ('smoke','full'):
 subprocess.run([sys.executable,'-m','pytest','tests/retargeting/test_response_bilevel.py',
  'tests/retargeting/test_response_policy.py','tests/retargeting/test_response_metrics.py',
  'tests/retargeting/test_chord_upper.py','tests/retargeting/test_response_upper.py','tests/evaluation/test_track3_estimates.py','tests/test_adaptive_windows.py',
  'tests/test_local_correction.py','tests/test_rejected_search_callback.py',
  'tests/runtime/test_wrist_limits.py','-q','-p','no:cacheprovider'],
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
  assert rows and all(row['lower_response_solved_for_every_candidate'] for row in rows)
  assert all(row['lower_parameter_error_bound_max']<=1e-5 for row in rows)
  for row in rows:
   for candidate in row['candidates']:
    assert np.isfinite(candidate['cost'])
    assert candidate['metrics']['cws_frames']>0
    assert len(candidate['metrics']['per_object_cws_tolerance_error'])==len(config['episode']['objects'])
    assert 'contact_proximity_mean_m' not in candidate['metrics']
  (b.parent/'reports'/f'smoke_validation_{b.name}_s{a.seed}.json').write_text(json.dumps(dict(
   passed=True,seed=a.seed,searches=len(rows),accepted=sum(row['accepted'] for row in rows),
   maximum_lower_error=max(row['lower_parameter_error_bound_max'] for row in rows),
   next_phase='pilot' if a.phase=='full' else None),indent=2)+'\n')
  print('[chord-response] GPU integration validation passed; '+
        ('starting 2048-world full training' if a.phase=='full' else 'smoke complete'),flush=True)
