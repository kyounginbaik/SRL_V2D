"""Queue actual backend evaluations and finalize only after both succeed."""
from pathlib import Path
import json
import shlex
import subprocess

BASE = Path(__file__).resolve().parents[1]
LEGACY = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
KIT = LEGACY / 'refinement_20261017/submission_readiness/latest_official/v2d_submission_kit'
PYTHON = LEGACY / '.venv/bin/python'


def submit(arguments):
    output = subprocess.check_output(['sbatch', '--parsable', *map(str, arguments)], text=True).strip()
    job = output.split(';')[0]
    if not job.isdigit():
        raise ValueError('Unexpected Slurm response: ' + output)
    return int(job)


def main():
    if (BASE / 'jobs.json').exists():
        raise FileExistsError('Jobs already submitted; inspect jobs.json instead of duplicating them')
    provenance = json.loads((BASE / 'git_provenance.json').read_text())
    manifest = json.loads((BASE / 'build_manifest.json').read_text())
    if not provenance['pushed'] or manifest['code_commit_url'] != provenance['code_commit_url']:
        raise ValueError('Require pushed, verified producing source before submission')
    (BASE / 'slurm').mkdir(exist_ok=True)
    cpu_command = [PYTHON, BASE / 'package_code/run_legacy_batch.py',
                   '--build-manifest', BASE / 'build_manifest.json', '--kit', KIT,
                   '--output-root', BASE / 'common_components',
                   '--legacy-root', LEGACY / 'refinement_20261018/common_release_v2_20261018', '--workers', '4']
    jobs = {}
    jobs['legacy_twelve_cpu'] = submit(['--job-name=ep37-common12', '--partition=debug', '--nodelist=node01',
        '--cpus-per-task=8', '--mem=24G', '--time=02:00:00',
        '--output=' + str(BASE / 'slurm/cpu-%j.log'), '--wrap', shlex.join(map(str, cpu_command))])
    (BASE / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    gpu_command = [BASE / 'package_code/common_evaluator.py', 'evaluate',
        '--build-manifest', BASE / 'build_manifest.json', '--kit', KIT,
        '--output', BASE / 'common_components/episode_000037', '--backend', 'ilchold', '--episode', '37',
        '--exporter', BASE / 'code/export_ilchold_rollout.py', '--run-root', BASE / 'source/episode_000037',
        '--world-count', '32', '--world-index', '28',
        '--mesh', 'wooden_spoon=' + str(BASE / 'review/meshes/episode_000037/wooden_spoon.obj')]
    jobs['episode37_gpu'] = submit(['--time=00:20:00', BASE / 'release_tools/gpu_replay.sbatch', *gpu_command])
    (BASE / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    finalize = [PYTHON, BASE / 'release_tools/finalize_release.py', '--base', BASE,
                '--build-manifest', BASE / 'build_manifest.json',
                '--components-root', BASE / 'common_components', '--kit', KIT]
    jobs['finalize_after_both'] = submit(['--job-name=ep37-package', '--partition=debug', '--nodelist=node01',
        '--cpus-per-task=4', '--mem=16G', '--time=00:30:00',
        f"--dependency=afterok:{jobs['legacy_twelve_cpu']}:{jobs['episode37_gpu']}",
        '--output=' + str(BASE / 'slurm/package-%j.log'), '--wrap', shlex.join(map(str, finalize))])
    (BASE / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    status = json.loads((BASE / 'release_status.json').read_text())
    status.update(stage='evaluations_submitted', jobs=jobs, producing_code_commit=provenance['commit'])
    (BASE / 'release_status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(jobs, indent=2))


if __name__ == '__main__':
    main()
