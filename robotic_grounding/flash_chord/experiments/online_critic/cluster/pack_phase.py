"""Produce a submission only when all public episode contracts validate."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
phase = sys.argv[1]
assert phase in ('smoke', 'pilot')
revision = json.loads((root/'reports/preparation.json').read_text())['git_revision']
batch = json.loads((root/'batch_manifest.json').read_text())
selected = {int(episode.split('_')[-1]) for episode in batch['episodes']}
command = [sys.executable, str(root/'submission/pack_submission.py'),
           '--results', str(root/'submission_results'/phase),
           '--kit', str(root/'submission_kit'),
           '--code-commit', f'https://github.com/kyounginbaik/SRL_V2D/commit/{revision}',
           '--output', str(root/'submissions'/f'track3_{phase}.csv')]
if phase == 'pilot':
    command += ['--fallback-results', str(root/'submission_results/smoke')]
if len(selected) < 20:
    command += ['--allow-incomplete']
subprocess.run(command, check=True)
readiness = root/'submissions'/f'track3_{phase}.readiness.json'
report = json.loads(readiness.read_text())
missing_selected = selected & {int(episode) for episode in report['missing_episodes']}
report.update(selected_episodes=sorted(selected), selected_episodes_ready=not missing_selected,
              missing_selected_episodes=sorted(missing_selected))
temporary = readiness.with_suffix('.tmp')
temporary.write_text(json.dumps(report, indent=2)+'\n')
temporary.replace(readiness)
if missing_selected:
    raise RuntimeError(f'Selected batch episodes failed export validation: {sorted(missing_selected)}')
print(f'All {len(selected)} selected episode rollouts validate; combined submission ready={report["ready"]}', flush=True)
