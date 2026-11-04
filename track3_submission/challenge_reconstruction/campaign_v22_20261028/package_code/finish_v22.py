"""Assemble and score the v22 set (phase 7: retargeted onto our reconstruction selected20_final_v5). Does not upload.

usage: finish_v22.py render-args   -> writes videos_v22/render_args_epNN.json for every changed episode
       finish_v22.py assemble      -> package + scores vs target_final_v5 + SRL copy + knockouts
Selection: selection_v22.json {episode: tag | "v21" | "v21C"}; a tag is a build-v2 output in
campaign_v16_20261023/common_v2_candidates/<tag>/episode_XXXXXX whose completed.json names its legacy component.
"""
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
C16 = ROOT / 'challenge_reconstruction/campaign_v16_20261023'
C = ROOT / 'challenge_reconstruction/campaign_v22_20261028'
BUILD = ROOT / 'challenge_reconstruction/evaluator_build_v2_20261023'
CAND = C16 / 'common_v2_candidates'
TARGET = C / 'target_final_v5'
OUT = ROOT / 'challenge_submissions/track3_v22_20261028/all_20_episodes'
SRL = ROOT / 'challenge_submissions_SRL/track3_v22_20261028/all_20_episodes'
PY = ROOT / 'challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python'


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


def package_args(name):
    args = shlex.split((C16 / name).read_text().replace('\\\n', '\n'))
    out = {}
    for i in range(0, len(args), 2):
        flag, value = args[i:i + 2]
        episode, path = value.split('=', 1)
        out.setdefault(int(episode), {})[flag[2:]] = path
    return out


def selection():
    v21, v21c = package_args('assemble_v21.args'), package_args('assemble_v21C.args')
    chosen = {}
    for key, tag in read(C / 'selection_v22.json').items():
        episode = int(key); name = f'episode_{episode:06d}'
        if tag in ('v21', 'v21C'):
            chosen[episode] = dict(tag=tag, changed=False, **(v21 if tag == 'v21' else v21c)[episode])
            continue
        out = CAND / tag / name; done = read(out / 'completed.json')
        if done.get('exact_previous_physical_states_actions') is not True:
            raise ValueError(f'{tag}: build-v2 replay not exact')
        legacy = Path(done['legacy_component'])
        portable = legacy.parents[1] / 'portable_archives' / f'{name}_portable.tar.gz'
        if not portable.is_file():
            raise FileNotFoundError(portable)
        chosen[episode] = dict(tag=tag, changed=True, component=str(out), portable=str(portable),
                               legacy=str(legacy), baseline=v21[episode]['component'])
    if sorted(chosen) != sorted(v21):
        raise ValueError('selection must cover all twenty episodes')
    return chosen


def world_dir(legacy):
    found = subprocess.run(['grep', '-rhoE', '"input_dir": "[^"]*"', str(legacy)], capture_output=True, text=True).stdout
    dirs = sorted({line.split('"')[3] for line in found.splitlines() if 'portable_validation' not in line})
    if len(dirs) > 1:  # prefer the world inside the component's own stream dir
        stream = Path(legacy).parents[2] if 'campaign_v22' in str(legacy) else Path(legacy).parents[3]
        dirs = [d for d in dirs if d.startswith(str(stream))] or dirs
    if len(dirs) != 1:
        raise ValueError(f'{legacy}: world dir ambiguous {dirs}')
    return dirs[0]


def render_args():
    videos = C / 'videos_v22'; (videos / 'logs').mkdir(parents=True, exist_ok=True)
    changed = []
    for episode, item in sorted(selection().items()):
        if not item['changed']:
            continue
        name = f'episode_{episode:06d}'; changed.append(episode)
        execution = lambda d: str(Path(d) / 'backend/evaluation/execution/execution_report.json')
        args = dict(episode_dir=world_dir(item['legacy']), baseline_report=execution(item['baseline']),
                    refined_report=execution(item['component']), out_dir=str(videos / name),
                    reference_scene=str(Path(item['component']) / 'backend/evaluation/execution/scene_physics.xml'),
                    width=640, height=480, stride=1, include_source=True,
                    titles=['Our reconstruction final_v5 target (kinematic)' if 'campaign_v22' in world_dir(item['legacy'])
                            else 'Component reference (kinematic)', 'v21 (recorded physics)',
                            f'v22 {item["tag"]} (recorded physics)'],
                    video_name=f'episode_{episode:02d}_v21_vs_v22.mp4')
        (videos / f'render_args_ep{episode}.json').write_text(json.dumps(args, indent=2) + '\n')
    (videos / 'changed.txt').write_text(' '.join(map(str, changed)) + '\n')
    print('changed', changed)


def assemble():
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    chosen = selection(); assembly_args = []
    for episode, item in sorted(chosen.items()):
        assembly_args += ['--component', f'{episode}={item["component"]}', '--portable', f'{episode}={item["portable"]}']
        if item['changed']:
            video_dir = C / 'videos_v22' / f'episode_{episode:06d}'
            report = read(video_dir / 'render_report.json')
            execution = read(Path(item['component']) / 'backend/evaluation/result.json')['execution']
            if sha(report['refined_trajectory']) != sha(execution['trajectory']):
                raise ValueError(f'Video is from a different rollout: {episode}')
            assembly_args += ['--video', f'{episode}={video_dir}']
        else:
            assembly_args += ['--video', f'{episode}={item["video"]}']
    (C / 'logs').mkdir(exist_ok=True)
    run(C16 / 'package_code/assemble_twenty_v2.py',
        '--build-manifest', BUILD / 'build_manifest.json',
        '--common-evaluator', BUILD / 'package_code/common_evaluator.py',
        *assembly_args, '--output', OUT,
        '--reason', 'v22: twenty physics policy rollouts targeted to our own reconstruction selected20_final_v5 '
        '(validated 10/10 public + 10/10 held-out public). All components reproduce exactly under evaluator build v2. '
        'Our reconstruction meshes only. These are reconstruction proxy scores.',
        log=C / 'logs/assemble_v22.log')
    roster = read(OUT / 'manifest.json')['available_episodes']
    parts_dir = OUT / 'target_score_parts'; parts_dir.mkdir()

    def score_episode(episode):
        result = parts_dir / f'episode_{episode:06d}.json'
        run(C16 / 'compose_target/score_vs_compose.py', OUT / 'track3_submission.parquet', OUT / 'results', TARGET,
            result, episode, log=C / f'logs/score_v22_ep{episode:02d}.log')
        return read(result)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(score_episode, roster))
    scores = {str(ep): part[str(ep)] for ep, part in zip(roster, parts)}
    (OUT / 'target_final_v5_scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    failed = [ep for ep, v in scores.items() if v.get('scores', {}).get('MP-SR') != 1.0]
    margin_misses = {ep: {n: e for n, e in v['objects'].items() if e[0] > 2.4 or e[1] > 24} for ep, v in scores.items()}
    margin_misses = {ep: o for ep, o in margin_misses.items() if o}
    summary = dict(episodes_passed=20 - len(failed), episodes_total=20, failed=failed, exact_build_v2_replay=True,
                   thresholds=dict(position_cm=3, orientation_deg=30),
                   aspirational_margin=dict(position_cm=2.4, orientation_deg=24, misses=margin_misses),
                   target='reconstruction_recovery_20261026/selected20_final_v5 (ours, GT-free)',
                   selection={str(ep): item['tag'] for ep, item in chosen.items()},
                   official_leaderboard_scores=False, uploaded=False)
    (OUT / 'target_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    note = '\n## v22 (2026-10-28)\n\n'
    note += (f'{summary["episodes_passed"]}/20 episodes pass against our reconstruction selected20_final_v5 with the kit scorers '
             '(3 cm / 30 deg); every component replays exactly under evaluator build v2. Reconstruction proxy only; '
             'hidden-GT scores are unavailable. Our meshes only.\n\n| Episode | Tag | Object errors (cm / deg) |\n| --- | --- | --- |\n')
    for ep, value in scores.items():
        errors = '; '.join(f'{n}: {e[0]:.2f} / {e[1]:.1f}' for n, e in value['objects'].items())
        note += f'| {int(ep)} | {chosen[int(ep)]["tag"]} | {errors} |\n'
    note += ('\nBefore submission, publish the evaluator build-v2 commit: '
             '`git -C /data1/SRL_V2D/.local/worktree_evaluator_build_v2 push kyoungin evaluator-build-v2`. Nothing uploaded.\n')
    note_file = C / 'srl_note_v22.md'; note_file.write_text(note)
    with (OUT / 'README.md').open('a') as stream:
        stream.write(note)
    inventory(OUT)
    run(ROOT / 'challenge_submissions_SRL/make_srl_folder.py', OUT, SRL, note_file, log=C / 'logs/srl_v22.log')
    for name in ('target_final_v5_scores.json', 'target_validation.json'):
        (SRL / name).write_bytes((OUT / name).read_bytes())
    if sha(OUT / 'track3_submission.parquet') != sha(SRL / 'track3_submission.parquet'):
        raise ValueError('SRL submission differs from the validated package')
    inventory(SRL)
    knockout_dir = ROOT / 'challenge_submissions_SRL/ablation_knockout_v22_20261028'
    knockout_dir.mkdir()
    changed = sorted(ep for ep, item in chosen.items() if item['changed'])
    run(ROOT / 'challenge_submissions_SRL/ablation_knockout_20261020/make_knockouts.py',
        '--source', SRL / 'track3_submission.parquet', '--out', knockout_dir, '--episodes', *roster,
        '--group', 'retargeted=' + ','.join(map(str, changed)),
        '--group', 'spoon10=3,5,6,15,16,32,33,35,36,37', log=C / 'logs/knockouts_v22.log')
    print(json.dumps(dict(summary=summary, package=str(OUT), srl_package=str(SRL), knockouts=str(knockout_dir),
                          submission_sha256=sha(OUT / 'track3_submission.parquet')), indent=2), flush=True)


if __name__ == '__main__':
    {'render-args': render_args, 'assemble': assemble}[sys.argv[1]]()
