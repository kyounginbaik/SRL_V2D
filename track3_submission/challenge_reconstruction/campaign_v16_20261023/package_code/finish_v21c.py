"""Assemble and score the completed v21C set locally. Does not upload anything."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path('/data1/SRL_V2D')
C = ROOT / 'challenge_reconstruction/campaign_v16_20261023'
BUILD = ROOT / 'challenge_reconstruction/evaluator_build_v2_20261023'
OUT = ROOT / 'challenge_submissions/track3_v21C_20261024/all_20_episodes'
SRL = ROOT / 'challenge_submissions_SRL/track3_v21C_20261024/all_20_episodes'
CHANGED = {3, 5, 6, 8, 14, 19, 25, 30, 32, 33, 35, 36}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def run(script, *args, log):
    print('Running', Path(script).name, flush=True)
    with Path(log).open('x') as stream:
        subprocess.run([sys.executable, '-B', '-u', str(script), *map(str, args)],
                       cwd='/tmp', stdout=stream, stderr=subprocess.STDOUT, check=True)


def inventory(folder):
    values = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
              if p.is_file() and p != folder / 'SHA256SUMS.json'}
    (folder / 'SHA256SUMS.json').write_text(json.dumps(values, indent=2) + '\n')


def main():
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    args = shlex.split((C / 'assemble_v21C.args').read_text().replace('\\\n', '\n'))
    components = {}
    assembly_args = []
    for i in range(0, len(args), 2):
        flag, value = args[i:i + 2]
        episode, path = value.split('=', 1)
        episode = int(episode)
        if flag == '--component':
            components[episode] = Path(path)
        if flag == '--video' and episode in CHANGED:
            continue  # v21 videos show different robot executions for these episodes.
        assembly_args += [flag, value]
    for episode in sorted(CHANGED):
        video_dir = C / 'videos_v21C' / f'episode_{episode:06d}'
        report = read(video_dir / 'render_report.json')
        execution = read(components[episode] / 'backend/evaluation/result.json')['execution']
        if sha(report['refined_trajectory']) != sha(execution['trajectory']):
            raise ValueError(f'Video is from a different rollout: {episode}')
        if not Path(report['video']).is_file():
            raise ValueError(f'Video missing: {episode}')
        assembly_args += ['--video', f'{episode}={video_dir}']
    run(C / 'package_code/assemble_twenty_v2.py',
        '--build-manifest', BUILD / 'build_manifest.json',
        '--common-evaluator', BUILD / 'package_code/common_evaluator.py',
        *assembly_args, '--output', OUT,
        '--reason', 'v21C: twenty physics policy rollouts targeted to the coworker compose reconstruction; '
        'episode 33 uses a power grasp with cup-rim clearance, teacher ILC and learned finger margins. '
        'All components reproduce exactly under evaluator build v2. Our reconstruction meshes only; '
        'no ep19 sweeper relabel or ep15 mesh replacement. These are reconstruction proxy scores.',
        log=C / 'logs/assemble_v21C.log')
    roster = read(OUT / 'manifest.json')['available_episodes']
    score_parts = OUT / 'compose_score_parts'
    score_parts.mkdir()
    def score_episode(episode):
        result = score_parts / f'episode_{episode:06d}.json'
        run(C / 'compose_target/score_vs_compose.py', OUT / 'track3_submission.parquet',
            OUT / 'results', C / 'compose_target', result, episode,
            log=C / f'logs/score_v21C_ep{episode:02d}.log')
        return read(result)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(score_episode, roster))
    scores = {str(ep): part[str(ep)] for ep, part in zip(roster, parts)}
    (OUT / 'compose_scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    if sorted(map(int, scores)) != roster or any(v.get('scores', {}).get('MP-SR') != 1.0 for v in scores.values()):
        raise ValueError('The packed v21C set does not pass all twenty episodes against compose')
    import pandas as pd
    previous = ROOT / 'challenge_submissions/track3_v21_20261024/all_20_episodes_post/track3_submission.parquet'
    before, after = (pd.read_parquet(p).set_index('row_id').sort_index()
                     for p in (previous, OUT / 'track3_submission.parquet'))
    if not before.index.equals(after.index):
        raise ValueError('v21 and v21C rosters differ')
    columns = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'object_mesh_b64']
    episode_ids = after.index.str.extract(r'/e(\d+)/')[0].astype(int).to_numpy()
    comparison = {}
    for episode in roster:
        mask = episode_ids == episode
        same = before.loc[mask, columns].equals(after.loc[mask, columns])
        if episode not in CHANGED and not same:
            raise ValueError(f'Reused episode changed: {episode}')
        comparison[str(episode)] = dict(poses_and_meshes_identical_to_v21=same,
                                       compose_scores_v21=read(C / 'compose_crosscheck/v21_vs_compose.json')[str(episode)],
                                       compose_scores_v21C=scores[str(episode)])
    (OUT / 'v21_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    margin_misses = {ep: {name: error for name, error in value['objects'].items()
                          if error[0] > 2.4 or error[1] > 24}
                     for ep, value in scores.items()}
    margin_misses = {ep: objects for ep, objects in margin_misses.items() if objects}
    summary = dict(episodes_passed=20, episodes_total=20, exact_build_v2_replay=True,
                   thresholds=dict(position_cm=3, orientation_deg=30),
                   aspirational_margin=dict(position_cm=2.4, orientation_deg=24,
                                             misses=margin_misses),
                   scores_are_against_coworker_reconstruction=True,
                   official_leaderboard_scores=False, uploaded=False,
                   evaluator_code_publication_pending=dict(
                       branch='evaluator-build-v2',
                       commit='5cc5ca516fc791f745449b279620c67e748856cd',
                       commit_url_returned_http=404,
                       branch_missing_from_remote_on='2026-10-24'))
    (OUT / 'compose_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    note = '\n## v21C (2026-10-24)\n\n'
    note += ('All 20 episodes pass against the coworker compose reconstruction using the kit scorers '
             '(3 cm / 30 deg), with exact build-v2 replay for every component. '
             'This is a reconstruction proxy result; hidden ground truth scores are unavailable. '
             'The 2.4 cm / 24 deg refinement target is reported separately in compose_validation.json. '
             'Episode 19 is packed without the sweeper relabel. All submitted meshes are our reconstructions. '
             'Episode 25 uses slotted versions of our board meshes.\n\n')
    note += '| Episode | Object errors (cm / deg) |\n| --- | --- |\n'
    for ep, value in scores.items():
        errors = '; '.join(f'{name}: {error[0]:.2f} / {error[1]:.1f}' for name, error in value['objects'].items())
        note += f'| {int(ep)} | {errors} |\n'
    note += '\nVideos for the 12 retargeted episodes compare v21 with the actual v21C rollout. Nothing uploaded.\n'
    note += ('\nBefore submission, publish the existing evaluator build-v2 commit. Its recorded GitHub URL '
             'returned HTTP 404 and the branch was absent from the remote on 2026-10-24. '
             'The frozen runtime is already committed locally. Command: '
             '`git -C /data1/SRL_V2D/.local/worktree_evaluator_build_v2 push kyoungin evaluator-build-v2`.\n')
    note_file = C / 'srl_note_v21C.md'
    note_file.write_text(note)
    with (OUT / 'README.md').open('a') as stream:
        stream.write(note)
    inventory(OUT)
    run(ROOT / 'challenge_submissions_SRL/make_srl_folder.py', OUT, SRL, note_file,
        log=C / 'logs/srl_v21C.log')
    for name in ('compose_scores.json', 'compose_validation.json', 'v21_comparison.json'):
        (SRL / name).write_bytes((OUT / name).read_bytes())
    if sha(OUT / 'track3_submission.parquet') != sha(SRL / 'track3_submission.parquet'):
        raise ValueError('SRL submission differs from the validated package')
    inventory(SRL)
    knockout_dir = ROOT / 'challenge_submissions_SRL/ablation_knockout_v21C_20261024'
    if knockout_dir.exists():
        raise FileExistsError(knockout_dir)
    knockout_dir.mkdir()
    run(ROOT / 'challenge_submissions_SRL/ablation_knockout_20261020/make_knockouts.py',
        '--source', SRL / 'track3_submission.parquet', '--out', knockout_dir,
        '--episodes', *roster,
        '--group', 'retargeted12=' + ','.join(map(str, sorted(CHANGED))),
        '--group', 'rack5=14,15,16,28,30',
        '--group', 'spoon10=3,5,6,15,16,32,33,35,36,37',
        log=C / 'logs/knockouts_v21C.log')
    for variant in read(knockout_dir / 'index.json')['variants']:
        episodes = ', '.join(map(str, variant['episodes']))
        (knockout_dir / variant['folder'] / 'README.md').write_text(
            f'# v21C diagnostic knockout: episodes {episodes}\n\n'
            f'Source: `{SRL / "track3_submission.parquet"}`.\n\n'
            f'{variant["rows_changed"]} of {variant["rows_total"]} rows were intentionally perturbed '
            'for diagnostic comparison. Frame 0 and all other episodes are unchanged. '
            'These diagnostic poses do not represent physics policy rollouts. '
            'The main v21C package contains the original validated policy rollouts.\n\n'
            'Compare any future leaderboard results with the v21C baseline: an MP-SR decrease '
            'of 0.05 corresponds to one previously passing episode. The 20/20 reconstruction '
            'proxy result is not a leaderboard baseline. Nothing uploaded.\n')
    print(json.dumps(dict(summary=summary, package=str(OUT), srl_package=str(SRL),
                          knockouts=str(knockout_dir),
                          submission_sha256=sha(OUT / 'track3_submission.parquet')), indent=2), flush=True)


if __name__ == '__main__':
    main()
