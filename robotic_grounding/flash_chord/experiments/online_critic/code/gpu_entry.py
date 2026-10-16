"""Run a seed/episode integration check and optionally its full training phase."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

base = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--phase', choices=['smoke', 'pilot', 'full'], default='full')
args = parser.parse_args()
# The prepared batch preflight runs CPU tests once for this source snapshot.
# Standalone full/smoke commands retain their own checks.
test_report = base.parent/'reports/cpu_tests.json'
certified = False
if test_report.is_file():
    report = json.loads(test_report.read_text())
    certified = report.get('passed') is True and all(
        report.get(name.replace('.json', '_sha256')) == hashlib.sha256((base.parent/name).read_bytes()).hexdigest()
        for name in ('source_manifest.json', 'launcher_manifest.json', 'data_manifest.json'))
if args.phase in ('smoke', 'full') and not certified:
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        'tests/retargeting/test_wrench_hypergradient.py',
        'tests/retargeting/test_wrench_hypergradient_editor.py',
        'tests/retargeting/test_lower_adaptation.py',
        'tests/retargeting/test_reference_relabel.py',
        'tests/retargeting/test_actor_response.py',
        'tests/retargeting/test_finite_response_acceptance.py',
        'tests/runtime/test_wrist_limits.py',
        'tests/runtime/test_wrist_residual_slew.py'],
        env={**os.environ, 'JAX_PLATFORMS': 'cpu'}, check=True)
if args.phase == 'pilot' and (base.parent/'batch_manifest.json').exists():
    certification = base.parent/'reports'/f'smoke_validation_{base.name}_s{args.seed}.json'
    if not certification.is_file() or not json.loads(certification.read_text()).get('passed'):
        raise RuntimeError(f'Full batch training requires passed integration: {certification}')
for phase in (('smoke', 'pilot') if args.phase == 'full' else (args.phase,)):
    subprocess.run([sys.executable, str(base/'code/run_sequence.py'),
                    '--seed', str(args.seed), '--phase', phase], check=True)
    if phase == 'smoke':
        run = base/'runs'/phase/base.name/'chunk'/f'seed_{args.seed}'
        config = json.loads((run/'run_config.json').read_text())
        manifest = json.loads((run/'training/ttsa_manifest.json').read_text())
        rows = [json.loads(line) for line in (run/'training/upper_updates.jsonl').read_text().splitlines()]
        assert config['contact_proximity'] is False
        assert manifest['contact_location_penalty'] is False and manifest['contact_location_gate'] is False
        from hypergradient_smoke import validate
        evidence = validate(run, config, manifest, rows)
        report = dict(passed=True, seed=args.seed, episode=base.name, **evidence,
                      next_phase='pilot' if args.phase == 'full' else None)
        (base.parent/'reports'/f'smoke_validation_{base.name}_s{args.seed}.json').write_text(json.dumps(report, indent=2)+'\n')
        print('[uniretarget] GPU integration validation passed', flush=True)

    if (base.parent/'batch_manifest.json').is_file():
        subprocess.run([sys.executable, str(base.parent/'cluster/pack_phase.py'), phase], check=True)
