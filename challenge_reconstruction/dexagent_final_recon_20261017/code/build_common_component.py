"""Re-evaluate a frozen episode under one real shared producing-code build.

This never rewrites the selected experiment or changes the official roster.
The new actor execution must exactly match the reviewed source recording.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'

from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

from learned_feedback_policy import sha


def read(path):
    return json.loads(Path(path).read_text())


def run(source_handoff, release_root, worktree, commit):
    source = Path(source_handoff).resolve()
    release = Path(release_root).resolve()
    code = Path(__file__).resolve().parent
    before = read(source/'readiness.json')
    portable = read(source/'portable_validation/portable_validation.json')
    producing = read(source/'provenance/producing_code.json')
    if not (before['episode_ready_for_packaging_review'] and
            portable['all_physical_states_actions_exact'] and
            portable['source_object_targets_unchanged']):
        raise ValueError('Source component has not passed independent validation')
    cp = before['learned_policy_checkpoint']
    if sha(cp['path']) != cp['sha256']:
        raise ValueError('Source checkpoint changed')
    output = release/before['episode']
    if output.exists():
        raise FileExistsError('Preserve existing common-build attempt: '+str(output))
    output.mkdir(parents=True)
    full_commit = subprocess.check_output(
        ['git', '-C', str(worktree), 'rev-parse', commit], text=True).strip()
    def command(script, *args, cwd=None):
        subprocess.run([sys.executable, str(code/script), *map(str, args)],
                       check=True, cwd=cwd)
    adapter = Path(before['source_manifest']).parent
    evaluation = output/'evaluation'
    command('reevaluate_component.py', '--baseline-report', before['policy_evaluation_report']['path'],
            '--episode-dir', adapter, '--output', evaluation)
    component = output/'component'
    base_training = source/'portable/original/base_training_report.json'
    if not base_training.exists():
        base_training = source/'portable/original/training_report.json'
    options = ['--run', evaluation, '--checkpoint', cp['path'],
               '--training-report', before['training_evidence']['path'],
               '--base-training-report', base_training,
               '--episode-dir', adapter, '--output', component]
    quality = source/'validation/physical_quality'
    if quality.exists():
        options += ['--quality-dir', quality]
    for evidence in sorted((source/'portable/lineage').glob('*')):
        if evidence.is_file():
            options += ['--lineage-file', evidence]
    options += ['--lineage-file', source/'provenance/producing_code.json']
    command('prepare_shared_component.py', *options)
    comparison = evaluation/'shared_evaluator_validation.json'
    shutil.copy2(comparison, component/'validation/reviewed_recording_exact.json')
    shutil.copy2(source/'README.md', component/'provenance/selected_component_notes.md')
    if (source/'videos').exists():
        shutil.copytree(source/'videos', component/'videos')
    lineage = dict(source_handoff=str(source), source_readiness_sha256=sha(source/'readiness.json'),
        original_producing_commit=producing['commit'], common_producing_commit=full_commit,
        checkpoint_sha256=cp['sha256'],
        full_reviewed_recording_exact=read(comparison)['all_physical_states_actions_exact'],
        physical_review='Carried forward only because every state and action matches exactly',
        original_object_targets_unchanged=True, original_component_preserved=True)
    (component/'provenance/common_build_lineage.json').write_text(json.dumps(lineage, indent=2)+'\n')
    (component/'README.md').write_text(
        '# '+before['episode']+' common-build component\n\n'
        'The selected learned checkpoint was freshly executed under the recorded common\n'
        'code build. Every state and action matches its physically reviewed recording.\n'
        'The original component remains unchanged. Exact checkpoint, model and source\n'
        'lineage are recorded under `provenance/`.\n\n'
        'Read `provenance/selected_component_notes.md` for the motion, metrics, repair\n'
        'evidence and remaining grasp/joint limitations. Read the separate portable\n'
        'validation report for the independent execution from `/tmp`.\n\n'
        'To reproduce, install `portable/requirements.txt` in Python 3.11 and run:\n\n'
        '```sh\npython portable/code/reproduce_policy_component.py replay '
        '--bundle portable --output /tmp/reproduction\n```\n\n'
        'This is an episode component, not a complete official 20-episode submission.\n'
        'Reference-proxy MP-SR is not confidential-ground-truth scoring. No upload is\n'
        'performed. Videos remain beside the archive; the archive includes the\n'
        'portable actor, geometry, trajectory, evidence and actual source provenance.\n')
    subprocess.run([sys.executable, str(component/'portable/code/reproduce_policy_component.py'),
                    'replay', '--bundle', str(component/'portable'),
                    '--output', str(component/'portable_validation')], check=True, cwd='/tmp')
    command('finalize_component_provenance.py', '--handoff', component,
            '--worktree', Path(worktree).resolve(), '--commit', full_commit,
            '--source-code', Path(__file__).resolve(),
            '--source-code', code/'reevaluate_component.py', '--archive')
    readiness = read(component/'readiness.json')
    result = dict(episode=before['episode'], component=str(component),
        checkpoint_sha256=cp['sha256'], producing_commit=full_commit,
        eval_code_sha256=readiness['evaluation_code_sha256'],
        readiness_sha256=sha(component/'readiness.json'),
        original_reviewed_execution_exact=True, portable_execution_exact=True,
        reference_proxy_mp_sr=readiness['reference_proxy_mp_sr'],
        archive=read(component/'archive.json'), complete_submission_ready=False, uploaded=False)
    (output/'completed.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source-handoff', 'release-root', 'worktree', 'commit']:
        p.add_argument('--'+name, required=True)
    run(**vars(p.parse_args()))
