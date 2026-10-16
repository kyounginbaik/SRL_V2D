#!/usr/bin/env python3
"""Queue one seed per episode, first complete coverage, then longer training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--partition', default='debug')
    parser.add_argument('--concurrency', default=3, type=int)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.concurrency < 1:
        parser.error('--concurrency must be positive')
    subprocess.run(['python3', str(root/'cluster/check_source.py')], check=True)
    report = json.loads((root/'reports/batch_preflight.json').read_text())
    if report.get('passed') is not True:
        parser.error('CPU batch preflight must pass before submission')
    for name in ('source_manifest', 'launcher_manifest', 'data_manifest', 'batch_manifest'):
        if report.get(name+'_sha256') != hashlib.sha256((root/(name+'.json')).read_bytes()).hexdigest():
            parser.error(f'CPU batch preflight is stale for {name}')
    manifest = json.loads((root/'batch_manifest.json').read_text())
    record = root/'reports/submitted_jobs.json'
    if record.exists():
        parser.error(f'Submission record exists; inspect it before submitting again: {record}')
    jobs = {}
    # Persist every returned ID immediately, including partial submissions.
    def submit(name, script, phase, *options):
        command = ['sbatch', '--parsable', f'--partition={args.partition}',
                   f'--job-name=ur-{name}', f'--output={root}/slurm/{name}_%A_%a.out',
                   f'--error={root}/slurm/{name}_%A_%a.err', *options,
                   str(root/'cluster'/script), str(root), phase]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        jobs[name] = dict(job_id=job, command=command)
        record.write_text(json.dumps(jobs, indent=2)+'\n')
        print(f'{name}: {job}', flush=True)
        return job
    array = f'--array=0-{len(manifest["episodes"])-1}%{args.concurrency}'
    first = submit('first-pass', 'train_batch.sbatch', 'smoke', array, '--time=1-00:00:00')
    submit('pack-first', 'pack_batch.sbatch', 'smoke', f'--dependency=afterany:{first}')
    full = submit('full', 'train_batch.sbatch', 'pilot', array, f'--dependency=afterany:{first}')
    submit('pack-full', 'pack_batch.sbatch', 'pilot', f'--dependency=afterany:{full}')


if __name__ == '__main__':
    main()
