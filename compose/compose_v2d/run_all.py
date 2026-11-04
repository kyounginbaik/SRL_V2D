"""Batch driver: run the whole ComPose-on-V2D pipeline for a list of episodes on one GPU, idempotently.

Per episode: FoundationStereo depth -> rect-left mono frames -> per object: SAM3 RGB video masks (+ transfer) -> prep ->
stereo VO -> WiLoR -> track_compose (--mesh team --tag TAG). Steps whose outputs exist are skipped; failures are recorded
in outputs/batch/status_<name>.json and the driver moves on. Objects without a team mesh (wooden_spoon) stop after WiLoR.
usage (inside the container): envs/v2d/bin/python -u compose_v2d/run_all.py --gpu 0 --name g0 --episodes 0 2 11 ...
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, split_of, work_dir, mono_masks_dir
from v2d_io import episode_objects

PY = str(ROOT / 'envs/v2d/bin/python')
PROMPTS = {
    'white_pot': ['white pot', 'white saucepan'], 'white_pot_lid': ['white pot lid', 'lid'],
    'beige_cup': ['beige cup', 'cup'], 'blue_cup': ['blue cup', 'cup'],
    'plastic_dish_rack': ['dish rack', 'plastic dish rack'], 'wooden_spoon': ['wooden spoon', 'spoon'],
    'mini_sweeper': ['small hand broom', 'brush', 'small broom'], 'mini_dust_pan': ['dust pan', 'dustpan', 'small plastic dustpan'],
    'water_pitcher': ['white jug', 'white watering can', 'white pitcher', 'plastic jug', 'watering can with long spout', 'water pitcher', 'watering can'],
    'wooden_piece_1': ['wooden block', 'wooden piece'], 'wooden_piece_2': ['wooden block', 'wooden piece'],
}
TEAM_MESH = ROOT / 'data/drive_results/V2D/extracted/public_meshes_20261008/selected_public'


def run(cmd, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'a') as f:
        f.write(f'\n$ {" ".join(cmd)}\n'); f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', required=True); ap.add_argument('--name', required=True); ap.add_argument('--episodes', nargs='+', type=int, required=True)
    ap.add_argument('--tag', default='_v3'); ap.add_argument('--only', default='', help='comma list of steps to run (default all)')
    ap.add_argument('--track_args', default='', help="extra args for track_compose, e.g. '--icp_jump_gate_deg 25 --adaptive_trim --grasp_sym'")
    ap.add_argument('--force_prep_multi', action='store_true', help='re-run prep (with --exclude_objs) for multi-object episodes even if pc_cache exists')
    a = ap.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    status_p = ROOT / 'outputs/batch' / f'status_{a.name}.json'; status_p.parent.mkdir(parents=True, exist_ok=True)
    status = json.load(open(status_p)) if status_p.exists() else {}
    def save(): json.dump(status, open(status_p, 'w'), indent=1)
    def mark(key, step, ok, t0):
        status.setdefault(key, {})[step] = dict(ok=ok, sec=round(time.time() - t0), at=time.strftime('%H:%M:%S')); save()
        print(f'[{a.name}] {key} {step}: {"ok" if ok else "FAIL"} ({time.time() - t0:.0f}s)', flush=True)
    steps = set(a.only.split(',')) if a.only else None
    want = lambda s: steps is None or s in steps

    for ep in a.episodes:
        split = split_of(ep); objs = episode_objects(ep); key = f'ep{ep}'
        logd = ROOT / 'logs/batch'
        print(f'\n===== episode {ep} ({split}) objects {objs}', flush=True)
        # 1. depth
        if want('depth') and not (ROOT / 'depth' / split / f'episode_{ep:06d}' / 'depth_mm.npz').exists():
            t0 = time.time(); rc = run([PY, '-u', 'scripts/run_foundation_stereo.py', '--vit', 'vitl', '--episodes', str(ep)], logd / f'{key}_depth.log', env)
            mark(key, 'depth', rc == 0, t0)
            if rc != 0: continue
        # 2. mono frames
        mf = ROOT / 'outputs' / f'mono_frames_ep{ep}'
        if want('mono') and not (mf.exists() and len(list(mf.glob('*.jpg'))) > 10):
            t0 = time.time(); rc = run([PY, '-u', 'compose_v2d/make_mono_frames.py', '--ep', str(ep)], logd / f'{key}_mono.log', env); mark(key, 'mono', rc == 0, t0)
        # 3. masks for every object first (prep subtracts the other objects' masks, e.g. the lid sitting on the pot)
        mask_ok = {}
        for obj in objs:
            k = f'ep{ep}_{obj}'; md = mono_masks_dir(ep, obj)
            if want('masks') and not (md / 'track_stats.json').exists():
                inst = ['--instance', {'wooden_piece_1': 'left', 'wooden_piece_2': 'right'}.get(obj, 'all')]
                t0 = time.time(); rc = run([PY, '-u', 'compose_v2d/sam3_track_rgb.py', '--ep', str(ep), '--obj', obj, *inst, '--prompts', *PROMPTS.get(obj, [obj.replace('_', ' ')])], logd / f'{k}_masks.log', env)
                mark(k, 'masks', rc == 0, t0)
            mask_ok[obj] = (mono_masks_dir(ep, obj) / 'track_stats.json').exists()   # re-resolve: the dir may have just been created
        for obj in objs:
            k = f'ep{ep}_{obj}'; wd = work_dir(ep, obj)
            if not mask_ok.get(obj): continue
            others = [o for o in objs if o != obj and mask_ok.get(o)]
            # 4. prep
            if want('prep') and (not (wd / 'pc_cache.npz').exists() or (a.force_prep_multi and len(objs) > 1 and not (wd / 'prep_excl.done').exists())):
                skip_rgb = ['--skip_rgb'] if (wd / 'rgb_undist').exists() and len(list((wd / 'rgb_undist').glob('*.jpg'))) > 10 else []
                t0 = time.time(); rc = run([PY, '-u', 'compose_v2d/prep_episode.py', '--ep', str(ep), '--obj', obj, *skip_rgb, '--exclude_objs', *others], logd / f'{k}_prep.log', env); mark(k, 'prep', rc == 0, t0)
                if rc == 0 and others: (wd / 'prep_excl.done').touch()
                if rc != 0: continue
            # 5. VO
            if want('vo') and not (wd / 'ego_cam0_T_cam.npz').exists():
                t0 = time.time(); rc = run([PY, '-u', 'compose_v2d/stereo_vo.py', '--ep', str(ep), '--obj', obj], logd / f'{k}_vo.log', env); mark(k, 'vo', rc == 0, t0)
                if rc != 0: continue
            # 6. WiLoR
            if want('wilor') and not (wd / 'wilor_hands.npz').exists():
                t0 = time.time(); rc = run([PY, '-u', 'compose_v2d/run_wilor_v2d.py', '--ep', str(ep), '--obj', obj], logd / f'{k}_wilor.log', env); mark(k, 'wilor', rc == 0, t0)
                if rc != 0: continue
            # 7. track
            if not list((TEAM_MESH / obj).glob('*.obj')):
                status.setdefault(k, {})['track'] = dict(ok=False, note='no team mesh'); save(); print(f'[{a.name}] {k}: no team mesh, tracking skipped', flush=True); continue
            if want('track') and not (wd / f'track{a.tag}' / 'poses.npz').exists():
                t0 = time.time(); rc = run([PY, '-u', 'compose_v2d/track_compose.py', '--ep', str(ep), '--obj', obj, '--mesh', 'team', '--tag', a.tag, *a.track_args.split()], logd / f'{k}_track{a.tag}.log', env)
                mark(k, 'track', rc == 0, t0)
    print(f'[{a.name}] done', flush=True)


if __name__ == '__main__':
    main()
