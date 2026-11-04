"""[E31 v31a/v31b: v29 chain re-pointed: selection_v31{a,b}.json (v29 + E31 re-retargets of the episodes that fail MP vs the cowork2 E14 target), target_E14_HGA/HGB (+ v14_g17 cross-score), new package dirs, new renders only for changed episodes (others reuse the v29 video dirs), no knockouts. usage: finish_v31.py {a|b} {render-args|assemble}]
Assemble and score the v29 set (v28 with only the submitted wooden_spoon mesh replaced by the G18 fix mesh in the 10 spoon episodes; mesh-only swap, rollouts identical, videos reused from v28) (v28: v27 retargeted onto final_v12b_g15: RGB-clock retime + camera depth sync, G15) (v27: v26 + ep32 onto final_v10_g12, the G12 official-consistent spoon roll) (v26: v25 + G10 spoon roll tracks ep32/ep3: final_v8_g10b, both re-retargeted) (v25: v24 + ep32 spoon in-hand roll fix: final_v7_g9rot, ep32 re-retargeted) (v24: v23b + phase-9 fixes onto our reconstruction final_v6: half-turn mesh swaps, ep32 anchor, tightened rollouts) (phase 8; same as v22b assembly) (phase 7: retargeted onto our reconstruction selected20_final_v5). Does not upload.

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
VAR = sys.argv[1]; assert VAR in ('a', 'b', 'acw', 'bcw'); VN = {'a': 'HGA', 'b': 'HGB'}[VAR[0]]; CWM = VAR.endswith('cw')   # cw: coworker bundle meshes (E31 cwmesh_v5), robot runs identical to v31a/b
V = Path(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/v31{VAR}')
V22B = Path('/home/exaFLOPs/v22_overflow_20261028/v22b/selection_v22b.json')
CAND_OVERFLOW = Path('/home/exaFLOPs/v22_overflow_20261028/common_v2_candidates')
BUILD = ROOT / 'challenge_reconstruction/evaluator_build_v2_20261023'
CAND = C16 / 'common_v2_candidates'
TARGET = C / f'target_E14_{VN}'; TARGET14 = C / 'target_final_v14_g17'
PKG = f'track3_v31{VAR[0]}_{VN}_cwmesh_20261101' if CWM else f'track3_v31{VAR}_{VN}_20261101'
OUT = ROOT / f'challenge_submissions/{PKG}/all_20_episodes'
SRL = ROOT / f'challenge_submissions_SRL/{PKG}/all_20_episodes'
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
    for key, tag in read(C / f'selection_v31{VAR}.json').items():
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
    if len(dirs) > 1 and 'streamE31' in str(legacy):  # v31: E31 re-retargets reuse the world of the v29 component's video (v28 -> v27 ... render args)
        ep_ = int(Path(legacy).name.split('_')[-1])
        for v_ in ('v28', 'v27', 'v26', 'v25', 'v24', 'v23b', 'v23'):
            f_ = C / v_ / 'videos' / f'render_args_ep{ep_}.json'
            if f_.is_file():
                prev = read(f_)['episode_dir']; dirs = [d for d in dirs if d == prev] or dirs; break
    if len(dirs) != 1:
        raise ValueError(f'{legacy}: world dir ambiguous {dirs}')
    return dirs[0]


VIDEO_CHAIN = ('v31a', 'v28', 'v27', 'v26', 'v25', 'v24', 'v23b', 'v23')


def find_video(episode, item):
    """v31: first video dir (own v31 render, then v31a for v31b, then the v29 reuse chain v28 ...) whose refined trajectory equals this component's rollout."""
    name = f'episode_{episode:06d}'
    execution = read(Path(item['component']) / 'backend/evaluation/result.json')['execution']
    for d in [V / 'videos' / name] + [C / v / 'videos' / name for v in VIDEO_CHAIN] + [C / 'videos_v22' / name]:
        if (d / 'render_report.json').is_file() and same_rollout(read(d / 'render_report.json')['refined_trajectory'], execution['trajectory']):
            return d
    return None


def render_args():
    """v31: episodes whose rollout already has a video (v29 reuse chain, or v31a for v31b) reuse it (reuse.json); the others get
    render_args_epNN.json for v31/render.sbatch (node100)."""
    videos = V / 'videos'; (videos / 'logs').mkdir(parents=True, exist_ok=True); reuse = {}; todo = []
    v29 = read(C / 'selection_v29.json')
    for episode, item in sorted(selection().items()):
        d = find_video(episode, item)
        reuse[episode] = dict(tag=item['tag'], same_as_v29=item['tag'] == v29[str(episode)], video_dir=str(d) if d else None)
        if d is None:
            if item['tag'] == v29[str(episode)]:
                raise ValueError(f'v29 rollout without its video: {episode}')
            name = f'episode_{episode:06d}'; todo.append(episode)
            execution = lambda d: str(Path(d) / 'backend/evaluation/execution/execution_report.json')
            args = dict(episode_dir=world_dir(item['legacy']), baseline_report=execution(item['baseline']),
                        refined_report=execution(item['component']), out_dir=str(videos / name),
                        reference_scene=str(Path(item['component']) / 'backend/evaluation/execution/scene_physics.xml'),
                        width=640, height=480, stride=1, include_source=True,
                        titles=['Our reconstruction final_v6 target (kinematic)' if 'campaign_v22' in world_dir(item['legacy'])
                                else 'Component reference (kinematic)', 'v22b submitted (recorded physics)',
                                f'v31{VAR} {item["tag"]} (recorded physics)'],
                        video_name=f'episode_{episode:02d}_v22b_vs_v31{VAR}.mp4')
            (videos / f'render_args_ep{episode}.json').write_text(json.dumps(args, indent=2) + '\n')
    (videos / 'reuse.json').write_text(json.dumps(reuse, indent=2) + '\n')
    (videos / 'to_render.txt').write_text(' '.join(map(str, todo)) + '\n')
    print(json.dumps(reuse, indent=1)); print('to_render', todo)


def assemble():
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    chosen = selection(); assembly_args = []
    for episode, item in sorted(chosen.items()):
        assembly_args += ['--component', f'{episode}={item["component"]}', '--portable', f'{episode}={item["portable"]}']
        if item['changed']:
            video_dir = find_video(episode, item)
            if video_dir is None:
                raise ValueError(f'No video of this rollout: {episode} (run render-args + render.sbatch)')
            assembly_args += ['--video', f'{episode}={video_dir}']
        else:
            assembly_args += ['--video', f'{episode}={item["video"]}']
    (V / 'logs').mkdir(exist_ok=True)
    run(C16 / 'package_code/assemble_twenty_v2.py',
        '--build-manifest', BUILD / 'build_manifest.json',
        '--common-evaluator', BUILD / 'package_code/common_evaluator.py',
        *assembly_args, '--output', OUT,
        '--reason', f'v31{VAR}: twenty physics policy rollouts: v29 components plus E31 re-retargets onto the reconstruction recon_E14_best_B2A_S1_{VN} '
        '(selected20_final_v14_g17 format; B2 consensus + A14 smoothing, annotation-fitted spoon roll eps 3/32/33). All components reproduce exactly under evaluator build v2. '
        + ('Meshes: the coworker bundle recon_E14_best_B2A_S1 own meshes (byte-exact where the component body frame agrees, else rigidly re-expressed in the component body frame; streamK33_ep33/cwmesh_v5). ' if CWM else
           'Meshes: the v29 component meshes (G18 spoon mesh in the 10 spoon episodes). ') + 'These are reconstruction proxy scores.',
        log=V / f'logs/assemble_v31{VAR}.log')
    roster = read(OUT / 'manifest.json')['available_episodes']
    allscores = {}
    for label, target in ((f'target_E14_{VN}', TARGET), ('target_final_v14_g17', TARGET14)):
        parts_dir = OUT / f'{label}_score_parts'; parts_dir.mkdir()

        def score_episode(episode):
            result = parts_dir / f'episode_{episode:06d}.json'
            run(C16 / 'compose_target/score_vs_compose.py', OUT / 'track3_submission.parquet', OUT / 'results', target,
                result, episode, log=V / f'logs/score_{label}_ep{episode:02d}.log')
            return read(result)
        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(score_episode, roster))
        allscores[label] = {str(ep): part[str(ep)] for ep, part in zip(roster, parts)}
        (OUT / f'{label}_scores.json').write_text(json.dumps(allscores[label], indent=2) + '\n')
    scores = allscores[f'target_E14_{VN}']
    failed = [ep for ep, v in scores.items() if v.get('scores', {}).get('MP-SR') != 1.0]
    failed14 = [ep for ep, v in allscores['target_final_v14_g17'].items() if v.get('scores', {}).get('MP-SR') != 1.0]
    margin_misses = {ep: {n: e for n, e in v['objects'].items() if e[0] > 2.4 or e[1] > 24} for ep, v in scores.items()}
    margin_misses = {ep: o for ep, o in margin_misses.items() if o}
    summary = dict(episodes_passed=20 - len(failed), episodes_total=20, failed=failed, exact_build_v2_replay=True,
                   thresholds=dict(position_cm=3, orientation_deg=30),
                   aspirational_margin=dict(position_cm=2.4, orientation_deg=24, misses=margin_misses),
                   target=f'cowork2 recon_E14_best_B2A_S1_{VN} -> campaign_v22_20261028/target_E14_{VN} (GT-free)',
                   cross_target_v14_g17=dict(episodes_passed=20 - len(failed14), failed=failed14),
                   selection={str(ep): item['tag'] for ep, item in chosen.items()},
                   official_leaderboard_scores=False, uploaded=False)
    (OUT / 'target_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    note = f'\n## {PKG} (2026-11-01)\n\n'
    note += (f'{summary["episodes_passed"]}/20 episodes pass against the reconstruction recon_E14_best_B2A_S1_{VN} (target_E14_{VN}) with the kit scorers '
             f'(3 cm / 30 deg); {20 - len(failed14)}/20 against selected20_final_v14_g17. Every component replays exactly under evaluator build v2. Reconstruction proxy only; '
             'hidden-GT scores are unavailable. ' + ('Meshes: the coworker bundle meshes (cwmesh_v5: byte-exact or rigidly re-expressed in the component body frame, per-object table in streamK33_ep33/README.md); robot rollouts identical to the v29-mesh package. ' if CWM else
             'Meshes: v29 component meshes (G18 spoon mesh in eps 3, 5, 6, 15, 16, 32, 33, 35, 36, 37). ') + 'v31 = v29 with E31 re-retargets of the episodes that fail MP vs the E14 target (3, 32, 33).\n\n'
             f'| Episode | Tag | Object errors vs target_E14_{VN} (cm / deg) | vs v14_g17 |\n| --- | --- | --- | --- |\n')
    for ep, value in scores.items():
        errors = '; '.join(f'{n}: {e[0]:.2f} / {e[1]:.1f}' for n, e in value['objects'].items())
        e14 = '; '.join(f'{n}: {e[0]:.2f} / {e[1]:.1f}' for n, e in allscores['target_final_v14_g17'][ep]['objects'].items())
        note += f'| {int(ep)} | {chosen[int(ep)]["tag"]} | {errors} | {e14} |\n'
    note += ('\nBefore submission, publish the evaluator build-v2 commit: '
             '`git -C /data1/SRL_V2D/.local/worktree_evaluator_build_v2 push kyoungin evaluator-build-v2`. Nothing uploaded.\n')
    note_file = V / f'srl_note_v31{VAR}.md'; note_file.write_text(note)
    with (OUT / 'README.md').open('a') as stream:
        stream.write(note)
    inventory(OUT)
    run(ROOT / 'challenge_submissions_SRL/make_srl_folder.py', OUT, SRL, note_file, log=V / f'logs/srl_v31{VAR}.log')
    for name in (f'target_E14_{VN}_scores.json', 'target_final_v14_g17_scores.json', 'target_validation.json'):
        (SRL / name).write_bytes((OUT / name).read_bytes())
    if sha(OUT / 'track3_submission.parquet') != sha(SRL / 'track3_submission.parquet'):
        raise ValueError('SRL submission differs from the validated package')
    inventory(SRL)
    print(json.dumps(dict(summary=summary, package=str(OUT), srl_package=str(SRL),
                          submission_sha256=sha(OUT / 'track3_submission.parquet')), indent=2), flush=True)


if __name__ == '__main__':
    {'render-args': render_args, 'assemble': assemble}[sys.argv[2]]()
