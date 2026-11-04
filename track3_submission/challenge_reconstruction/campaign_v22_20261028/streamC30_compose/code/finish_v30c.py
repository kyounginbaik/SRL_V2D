"""C30 (2026-11-01): assemble and score a v30c<K> package = v29's chain with the selection selection_v30c<K>.json: the twenty
components (v29 components where their rollout already fits the coworker compose_v<K> transplant target, C30 re-retargeted build-v2
tags elsewhere). Target: campaign_v22_20261028/target_compose_v<K>_on_v14 (compose_v<K> changes transplanted onto selected20_final_v14_g17).
Also scored vs target_final_v14_g17. No knockouts. Does not upload.

usage: finish_v30c.py K render-args  -> streamC30_compose/videos/render_args_<tag>_epNN.json for every component without a matching video
       finish_v30c.py K assemble     -> package + scores + SRL copy
(The rest of this file is v29/finish_v29.py; original docstring follows.)
v29: Assemble and score the v29 set (v28 with only the submitted wooden_spoon mesh replaced by the G18 fix mesh in the 10 spoon episodes; mesh-only swap, rollouts identical, videos reused from v28) (v28: v27 retargeted onto final_v12b_g15: RGB-clock retime + camera depth sync, G15) (v27: v26 + ep32 onto final_v10_g12, the G12 official-consistent spoon roll) (v26: v25 + G10 spoon roll tracks ep32/ep3: final_v8_g10b, both re-retargeted) (v25: v24 + ep32 spoon in-hand roll fix: final_v7_g9rot, ep32 re-retargeted) (v24: v23b + phase-9 fixes onto our reconstruction final_v6: half-turn mesh swaps, ep32 anchor, tightened rollouts) (phase 8; same as v22b assembly) (phase 7: retargeted onto our reconstruction selected20_final_v5). Does not upload.

usage: finish_v22b.py render-args  -> writes /home/exaFLOPs/v22_overflow_20261028/v22b/videos/render_args_epNN.json for every changed episode
       finish_v29.py assemble      -> package + scores vs target_final_v14_g17 + SRL copy + knockouts
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
K = int(sys.argv[1]); X = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose')
V = X / f'v30c{K}'; V.mkdir(exist_ok=True); VID = X / 'videos'
V22B = Path('/home/exaFLOPs/v22_overflow_20261028/v22b/selection_v22b.json')
CAND_OVERFLOW = Path('/home/exaFLOPs/v22_overflow_20261028/common_v2_candidates')
BUILD = ROOT / 'challenge_reconstruction/evaluator_build_v2_20261023'
CAND = C16 / 'common_v2_candidates'
TARGET = C / f'target_compose_v{K}_on_v14'; TARGET14 = C / 'target_final_v14_g17'
OUT = ROOT / f'challenge_submissions/track3_v30c{K}_20261101/all_20_episodes'
SRL = ROOT / f'challenge_submissions_SRL/track3_v30c{K}_20261101/all_20_episodes'
PY = ROOT / 'challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def same_rollout(video_trajectory, trajectory):
    """v29: a G18 mesh-only copy re-runs build-v2 at a new path, so trajectory.npz differs only in the path-dependent scene_sha256
    string (the scene XML names the component's asset paths); every recorded array (qpos/qvel/ctrl, substeps, object poses) must be equal."""
    if sha(video_trajectory) == sha(trajectory):
        return True
    import numpy as np
    a, b = np.load(video_trajectory, allow_pickle=False), np.load(trajectory, allow_pickle=False)
    if sorted(a.files) != sorted(b.files):
        return False
    return all(np.array_equal(a[k], b[k]) for k in a.files if k != 'scene_sha256')


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
    previous = read(V22B)  # videos compare against the submitted v22b component

    def baseline(episode):
        tag = previous[str(episode)]
        if tag in ('v21', 'v21C'):
            return (v21 if tag == 'v21' else v21c)[episode]['component']
        name = f'episode_{episode:06d}'
        return str((CAND / tag / name) if (CAND / tag / name).is_dir() else (CAND_OVERFLOW / tag / name))
    for key, tag in read(C / f'selection_v30c{K}.json').items():
        episode = int(key); name = f'episode_{episode:06d}'
        if tag in ('v21', 'v21C'):
            chosen[episode] = dict(tag=tag, changed=False, **(v21 if tag == 'v21' else v21c)[episode])
            continue
        out = (CAND / tag / name) if (CAND / tag / name).is_dir() else (CAND_OVERFLOW / tag / name); done = read(out / 'completed.json')
        if done.get('exact_previous_physical_states_actions') is not True:
            raise ValueError(f'{tag}: build-v2 replay not exact')
        legacy = Path(done['legacy_component'])
        portable = legacy.parents[1] / 'portable_archives' / f'{name}_portable.tar.gz'
        if not portable.is_file():
            raise FileNotFoundError(portable)
        chosen[episode] = dict(tag=tag, changed=True, component=str(out), portable=str(portable),
                               legacy=str(legacy), baseline=baseline(episode))
    if sorted(chosen) != sorted(v21):
        raise ValueError('selection must cover all twenty episodes')
    return chosen


def world_dir(legacy):
    found = subprocess.run(['grep', '-rhoE', '"input_dir": "[^"]*"', str(legacy)], capture_output=True, text=True).stdout
    dirs = sorted({line.split('"')[3] for line in found.splitlines() if 'portable_validation' not in line})
    if len(dirs) > 1:  # prefer the world inside the component's own stream dir
        stream = Path(legacy).parents[2] if ('campaign_v22' in str(legacy) or 'v22_overflow' in str(legacy)) else Path(legacy).parents[3]
        dirs = [d for d in dirs if d.startswith(str(stream))] or dirs
        dirs = [d for d in dirs if 'campaign_v22' in d or 'v22_overflow' in d] or dirs  # v22 world over the phase-6 one
    if len(dirs) > 1:  # v24: prefer the world in the same stream dir as the component
        own = [d for d in dirs if any(str(Path(legacy).parents[k]) + '/' in d + '/' for k in (2, 3) if 'stream' in Path(legacy).parents[k].name)]
        dirs = own or dirs
    if len(dirs) > 1 and 'streamH13' in str(legacy):  # v28: H13 actors reuse their v27 source world
        prev = read(C / 'v27/videos' / f'render_args_ep{int(Path(legacy).name.split("_")[-1])}.json')['episode_dir']
        dirs = [d for d in dirs if d == prev] or dirs
    if len(dirs) > 1 and 'streamG18v29' in str(legacy):  # v29: G18 mesh-only copies of the v28 components reuse the v28 world
        prev = read(C / 'v28/videos' / f'render_args_ep{int(Path(legacy).name.split("_")[-1])}.json')['episode_dir']
        dirs = [d for d in dirs if d == prev] or dirs
    if len(dirs) > 1 and 'streamC30_compose' in str(legacy):  # C30: re-retargets of v29 actors reuse the v29 (v28/v27) source world
        for v in ('v28', 'v27'):
            f = C / v / 'videos' / f'render_args_ep{int(Path(legacy).name.split("_")[-1])}.json'
            if f.is_file():
                prev = read(f)['episode_dir']; dirs = [d for d in dirs if d == prev] or dirs; break
    if len(dirs) != 1:
        raise ValueError(f'{legacy}: world dir ambiguous {dirs}')
    return dirs[0]


VIDEO_DIRS = [C / v / 'videos' for v in ('v28', 'v27', 'v26', 'v25', 'v24', 'v23b', 'v23')] + [C / 'videos_v22']


def find_video(episode, item):
    """first existing rendered video whose refined trajectory equals this component's rollout (v29 same_rollout rule)"""
    name = f'episode_{episode:06d}'
    execution = read(Path(item['component']) / 'backend/evaluation/result.json')['execution']
    for d in [VID / item['tag'] / name] + [v / name for v in VIDEO_DIRS]:
        if (d / 'render_report.json').is_file() and same_rollout(read(d / 'render_report.json')['refined_trajectory'], execution['trajectory']):
            return d
    return None


def render_args():
    """C30: write render args (shared across versions, keyed by tag) for every component whose rollout has no matching video."""
    (VID / 'logs').mkdir(parents=True, exist_ok=True); todo = []
    for episode, item in sorted(selection().items()):
        if not item['changed'] or find_video(episode, item) is not None:
            continue
        name = f'episode_{episode:06d}'
        execution = lambda d: str(Path(d) / 'backend/evaluation/execution/execution_report.json')
        args = dict(episode_dir=world_dir(item['legacy']), baseline_report=execution(item['baseline']),
                    refined_report=execution(item['component']), out_dir=str(VID / item['tag'] / name),
                    reference_scene=str(Path(item['component']) / 'backend/evaluation/execution/scene_physics.xml'),
                    width=640, height=480, stride=1, include_source=True,
                    titles=['Component reference (kinematic)', 'v22b submitted (recorded physics)', f'v30c {item["tag"]} (recorded physics)'],
                    video_name=f'episode_{episode:02d}_v22b_vs_{item["tag"]}.mp4')
        f = VID / f'render_args_{item["tag"]}_ep{episode}.json'
        if not f.exists(): f.write_text(json.dumps(args, indent=2) + '\n')
        todo.append(str(f))
    print('TODO', ' '.join(todo))


def assemble():
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    chosen = selection(); assembly_args = []
    for episode, item in sorted(chosen.items()):
        assembly_args += ['--component', f'{episode}={item["component"]}', '--portable', f'{episode}={item["portable"]}']
        if item['changed']:
            video_dir = find_video(episode, item)
            if video_dir is None:
                raise ValueError(f'no video with the same rollout for episode {episode} ({item["tag"]}); run render-args + render')
            assembly_args += ['--video', f'{episode}={video_dir}']
        else:
            assembly_args += ['--video', f'{episode}={item["video"]}']
    (V / 'logs').mkdir(exist_ok=True)
    run(C16 / 'package_code/assemble_twenty_v2.py',
        '--build-manifest', BUILD / 'build_manifest.json',
        '--common-evaluator', BUILD / 'package_code/common_evaluator.py',
        *assembly_args, '--output', OUT,
        '--reason', f'v30c{K}: twenty physics policy rollouts (v29 components plus C30 re-retargets) targeted to the coworker compose_v{K} changes '
        'transplanted onto our reconstruction selected20_final_v14_g17 (target_compose_v{K}_on_v14). All components reproduce exactly under evaluator build v2. '
        'Our reconstruction meshes only (v29 meshes; G18 spoon mesh in the 10 spoon episodes). These are reconstruction proxy scores.',
        log=V / f'logs/assemble_v30c{K}.log')
    roster = read(OUT / 'manifest.json')['available_episodes']
    parts_dir = OUT / 'target_score_parts'; parts_dir.mkdir()

    def score_episode(episode):
        result = parts_dir / f'episode_{episode:06d}.json'
        run(C16 / 'compose_target/score_vs_compose.py', OUT / 'track3_submission.parquet', OUT / 'results', TARGET,
            result, episode, log=V / f'logs/score_v30c{K}_ep{episode:02d}.log')
        return read(result)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(score_episode, roster))
    scores = {str(ep): part[str(ep)] for ep, part in zip(roster, parts)}
    (OUT / f'target_compose_v{K}_on_v14_scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    parts14 = OUT / 'target_score_parts_v14'; parts14.mkdir()

    def score14(episode):
        result = parts14 / f'episode_{episode:06d}.json'
        run(C16 / 'compose_target/score_vs_compose.py', OUT / 'track3_submission.parquet', OUT / 'results', TARGET14,
            result, episode, log=V / f'logs/score14_v30c{K}_ep{episode:02d}.log')
        return read(result)
    with ThreadPoolExecutor(max_workers=4) as pool:
        p14 = list(pool.map(score14, roster))
    (OUT / 'target_final_v14_g17_scores.json').write_text(json.dumps({str(ep): p[str(ep)] for ep, p in zip(roster, p14)}, indent=2) + '\n')
    failed = [ep for ep, v in scores.items() if v.get('scores', {}).get('MP-SR') != 1.0]
    margin_misses = {ep: {n: e for n, e in v['objects'].items() if e[0] > 2.4 or e[1] > 24} for ep, v in scores.items()}
    margin_misses = {ep: o for ep, o in margin_misses.items() if o}
    summary = dict(episodes_passed=20 - len(failed), episodes_total=20, failed=failed, exact_build_v2_replay=True,
                   thresholds=dict(position_cm=3, orientation_deg=30),
                   aspirational_margin=dict(position_cm=2.4, orientation_deg=24, misses=margin_misses),
                   target=f'campaign_v22_20261028/target_compose_v{K}_on_v14 (coworker compose_v{K} transplanted onto our selected20_final_v14_g17; GT-free)',
                   selection={str(ep): item['tag'] for ep, item in chosen.items()},
                   official_leaderboard_scores=False, uploaded=False)
    (OUT / 'target_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    note = f'\n## v30c{K} (2026-11-01)\n\n'
    note += (f'{summary["episodes_passed"]}/20 episodes pass against our reconstruction selected20_final_v14_g17 with the kit scorers '
             '(3 cm / 30 deg); every component replays exactly under evaluator build v2. Reconstruction proxy only; '
             'hidden-GT scores are unavailable. Our meshes only (the v29 meshes, incl. the G18 spoon mesh); see campaign_v22_20261028/streamC30_compose/README.md.\n\n| Episode | Tag | Object errors (cm / deg) |\n| --- | --- | --- |\n')
    for ep, value in scores.items():
        errors = '; '.join(f'{n}: {e[0]:.2f} / {e[1]:.1f}' for n, e in value['objects'].items())
        note += f'| {int(ep)} | {chosen[int(ep)]["tag"]} | {errors} |\n'
    note += ('\nBefore submission, publish the evaluator build-v2 commit: '
             '`git -C /data1/SRL_V2D/.local/worktree_evaluator_build_v2 push kyoungin evaluator-build-v2`. Nothing uploaded.\n')
    note_file = V / f'srl_note_v30c{K}.md'; note_file.write_text(note)
    with (OUT / 'README.md').open('a') as stream:
        stream.write(note)
    inventory(OUT)
    run(ROOT / 'challenge_submissions_SRL/make_srl_folder.py', OUT, SRL, note_file, log=V / f'logs/srl_v30c{K}.log')
    for name in (f'target_compose_v{K}_on_v14_scores.json', 'target_final_v14_g17_scores.json', 'target_validation.json'):
        (SRL / name).write_bytes((OUT / name).read_bytes())
    if sha(OUT / 'track3_submission.parquet') != sha(SRL / 'track3_submission.parquet'):
        raise ValueError('SRL submission differs from the validated package')
    inventory(SRL)
    print(json.dumps(dict(summary=summary, package=str(OUT), srl_package=str(SRL), 
                          submission_sha256=sha(OUT / 'track3_submission.parquet')), indent=2), flush=True)


if __name__ == '__main__':
    {'render-args': render_args, 'assemble': assemble}[sys.argv[2]]()
