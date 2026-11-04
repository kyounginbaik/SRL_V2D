"""Export the best ComPose tracking result per sequence into compose-sequences/{public|evaluation}/episode_XXXXXX/.

Per object: <obj>_trajectory.json (+ .npz: T_world_obj (N,4,4)), <obj>_mesh_metric.obj, <obj>_viz.mp4 (rendered if missing).
Per episode: camera_motion.json (+ .npz: cam0_T_cam (N,4,4)), README.md (chosen tag/variant, scores, screening).
World frame = rectified-left camera at frame 0 (L0); pose = world_T_object with the object's body frame = the exported mesh.

Selection
  public      : highest in-script ADD-AUC over tags/variants; kept only if AUC >= --min_auc (GT available).
  evaluation  : no GT -> GT-free screening (compose_v2d/screen_depth.py: rendered-mesh depth vs stereo depth, silhouette
                spill) must pass; among passing candidates the preferred tag order per object class and the ComPose blend
                when it agrees with the mesh-ICP track (else the mesh-ICP track).
  Objects whose team mesh is known to be wrong/unverifiable (water_pitcher, wooden pieces, mini_sweeper) go to _unverified/,
  everything that fails goes to _rejected/ (same layout, README says why) so nothing is silently dropped.
usage: export_sequences.py [--min_auc 0.8] [--episodes ...] [--no_viz]
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, work_dir, split_of, load_meta
from v2d_io import episode_objects, PUBLIC_EPISODES, EVAL_EPISODES

PY = str(ROOT / 'envs/v2d/bin/python')
TAG_ORDER = ['_v9lid', '_v9', '_v7', '_v7ng', '_v7nj', '_v8', '_v6', '_v5b', '_v4', '_v3', '_meshicp', '_team2', '_team']
VARIANTS = ['compose_meshicp_smooth', 'compose_meshicp', 'icp_track_6dof']
# evaluation split: tag preference per object class, decided from the public results (outputs/COMPOSE_V2D_RESULTS.md)
EVAL_PREF = {'white_pot': ['_v9lid', '_v9', '_v7nj', '_v7', '_v3'], 'blue_cup': ['_v9', '_v7', '_v4', '_v3'], 'beige_cup': ['_v9', '_v7', '_v3'],
             'white_pot_lid': ['_v9', '_v3'], 'plastic_dish_rack': ['_v9', '_v3'], 'mini_dust_pan': ['_v9', '_v7nj', '_v3'],
             'mini_sweeper': ['_v9', '_v5b', '_v3'], 'wooden_piece_1': ['_v9', '_v3'], 'wooden_piece_2': ['_v9', '_v3'], 'water_pitcher': ['_v9', '_v3']}
EVAL_VARIANT = 'compose_meshicp_smooth'; EVAL_FALLBACK = 'icp_track_6dof'; AGREE_DEG = 15.0
MESH_ISSUE = {'water_pitcher': 'team mesh has the wrong shape (CD-O 0.65 on public) -> needs re-reconstruction',
              'wooden_piece_1': 'team piece_1 mesh has the slot on the wrong side -> 180-deg twin registration',
              'wooden_piece_2': 'two identical pieces; twin ambiguity',
              'mini_sweeper': 'team mesh never matches the stereo depth (median 18 mm even on public runs with AUC 0.83) -> cannot be verified without GT'}
SCREEN_PATH = ROOT / 'outputs/screen_depth2.json'
SCREEN = json.load(open(SCREEN_PATH)) if SCREEN_PATH.exists() else {}


def screen_pass(ep, obj, tag, variant):
    """GT-free verdict from screen_depth.py (calibrated on the public split: flipped/misplaced poses and pot handle slips fail,
    every run with AUC >= 0.8 passes except the thin-bar dish rack which gets a looser depth limit)."""
    r = SCREEN.get(f'{ep}:{obj}:{tag}:{variant}')
    if r is None: return None, 'not screened'
    bad_lim = 0.4 if obj == 'plastic_dish_rack' else 0.15
    spill_ok = True if obj in ('blue_cup', 'beige_cup') else (r['spill_p90'] <= 0.12 and r['spill_frames'] <= 0.05)   # spill = handle-slip test; cups have no handle
    ok = r['bad_frames'] <= bad_lim and r['bad_last_q'] <= bad_lim and r['cover_med'] >= 0.8 and spill_ok
    return ok, f"depth-bad {r['bad_frames']:.2f}/{r['bad_last_q']:.2f}, cover {r['cover_med']:.2f}, spill p90 {r['spill_p90']:.2f} ({r['spill_frames']:.2f})"


def variant_disagreement(td):
    z = np.load(td / 'poses.npz')
    if EVAL_VARIANT not in z.files or EVAL_FALLBACK not in z.files: return None
    A, B = z[EVAL_VARIANT][:, :3, :3], z[EVAL_FALLBACK][:, :3, :3]
    ang = np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum('nij,nkj->nik', A, B)).as_rotvec(), axis=1))
    return float(np.percentile(ang, 90))


def candidates(ep, obj):
    out = []
    for tag in TAG_ORDER:
        td = work_dir(ep, obj) / f'track{tag}'
        if not (td / 'poses.npz').exists(): continue
        res = json.load(open(td / 'results.json')).get('results', {}) if (td / 'results.json').exists() else {}
        for v in VARIANTS:
            sc = res.get(v, {}).get('d0', {})
            out.append(dict(tag=tag, variant=v, auc=sc.get('auc'), pos_cm=sc.get('pos_cm'), ori_deg=sc.get('ori_deg'), dir=td))
    return out


def choose(ep, obj, split, min_auc):
    """Returns (best or None, bucket, reason)."""
    c = candidates(ep, obj)
    if not c: return None, 'missing', 'no tracking result'
    if split == 'public':
        scored = [x for x in c if x['auc'] is not None]
        if not scored: return None, 'missing', 'no GT score'
        best = max(scored, key=lambda x: x['auc'])
        best['note'] = f"best in-script AUC {best['auc']:.3f}"
        return best, ('ok' if best['auc'] >= min_auc else 'rejected'), (f"AUC {best['auc']:.3f} < {min_auc}" if best['auc'] < min_auc else '')
    order = EVAL_PREF.get(obj, ['_v3']) + TAG_ORDER
    tags = sorted({x['tag'] for x in c}, key=lambda t: order.index(t) if t in order else 99)
    tried = []
    for tag in tags:
        td = [x for x in c if x['tag'] == tag][0]['dir']; dis = variant_disagreement(td)
        first = EVAL_VARIANT if (dis is None or dis <= AGREE_DEG) else EVAL_FALLBACK
        for v in [first] + [x for x in [EVAL_VARIANT, EVAL_FALLBACK] if x != first]:
            cand = [x for x in c if x['tag'] == tag and x['variant'] == v]
            if not cand: continue
            ok, why = screen_pass(ep, obj, tag, v); tried.append(f'{tag}/{v}: {why}')
            if ok:
                b = dict(cand[0]); b['note'] = f"screen pass ({why}); blend/ICP p90 disagreement {dis:.0f} deg" if dis is not None else f'screen pass ({why})'
                return b, ('unverified' if obj in MESH_ISSUE else 'ok'), (MESH_ISSUE.get(obj, ''))
    # nothing passed: report the preferred candidate so it can still be inspected
    b = dict([x for x in c if x['tag'] == tags[0]][0]); b['note'] = 'screen FAIL; tried: ' + ' | '.join(tried)
    return b, 'rejected', 'GT-free screen failed for every tag/variant: ' + ' | '.join(tried)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'compose-sequences')); ap.add_argument('--min_auc', type=float, default=0.8)
    ap.add_argument('--episodes', nargs='*', type=int, default=None); ap.add_argument('--no_viz', action='store_true')
    a = ap.parse_args()
    eps = a.episodes or (PUBLIC_EPISODES + EVAL_EPISODES)
    out = Path(a.out); summary = []
    for ep in eps:
        split = split_of(ep); objs = episode_objects(ep); chosen = {}
        for obj in objs:
            best, bucket, reason = choose(ep, obj, split, a.min_auc)
            if best is None: summary.append((ep, split, obj, None, None, bucket, reason)); continue
            chosen[obj] = (best, bucket, reason)
        if not chosen: continue
        for sub in ('', '_rejected', '_unverified'):                    # one episode folder per bucket; stale copies elsewhere are removed
            d = out / sub / split / f'episode_{ep:06d}' if sub else out / split / f'episode_{ep:06d}'
            if d.exists() and not any(b == {'': 'ok', '_rejected': 'rejected', '_unverified': 'unverified'}[sub] for _, b, _ in chosen.values()): shutil.rmtree(d)
        for bucket in sorted({b for _, b, _ in chosen.values()}):
            od = (out / {'ok': '', 'rejected': '_rejected', 'unverified': '_unverified'}[bucket] / split / f'episode_{ep:06d}') if bucket != 'ok' else out / split / f'episode_{ep:06d}'
            od.mkdir(parents=True, exist_ok=True)
            for f_ in od.glob('*'):                                      # objects may move between buckets between runs
                if f_.is_file() and not f_.name.startswith('camera_motion'): f_.unlink()
            objs_b = {o: v for o, v in chosen.items() if v[1] == bucket}
            o0 = next(iter(objs_b)); meta = load_meta(ep, o0); E = np.load(work_dir(ep, o0) / 'ego_cam0_T_cam.npz')['cam0_T_cam']
            np.savez(od / 'camera_motion.npz', cam0_T_cam=E, K_rect=np.array(meta['K_rect']), K_a_undist=np.array(meta['K_a_undist']), T_a_from_rectL=np.array(meta['T_a_from_rectL']))
            json.dump(dict(frame='world = rectified-left mono camera at frame 0 (L0); cam0_T_cam[f] maps frame-f rect-left camera coords into world',
                           n_frames=int(len(E)), K_rect=meta['K_rect'], K_a_undist=meta['K_a_undist'], T_a_from_rectL=meta['T_a_from_rectL'],
                           cam0_T_cam=[dict(frame=f, pos=E[f, :3, 3].round(5).tolist(), quat_xyzw=Rot.from_matrix(E[f, :3, :3]).as_quat().round(6).tolist()) for f in range(len(E))]),
                      open(od / 'camera_motion.json', 'w'))
            lines = [f'# episode {ep} ({split}) - {bucket.upper()}', '', 'world frame = rectified-left camera at frame 0; quaternions xyzw; metres; body frame = <obj>_mesh_metric.obj', '']
            for obj, (b, _, reason) in objs_b.items():
                z = np.load(b['dir'] / 'poses.npz'); T = z[b['variant']]
                np.savez(od / f'{obj}_trajectory.npz', T_world_obj=T, frames=np.arange(len(T)))
                json.dump(dict(object=obj, tag=b['tag'], variant=b['variant'], n_frames=int(len(T)), selection=b.get('note', ''),
                               poses=[dict(frame=f, pos=T[f, :3, 3].round(5).tolist(), quat_xyzw=Rot.from_matrix(T[f, :3, :3]).as_quat().round(6).tolist()) for f in range(len(T))]),
                          open(od / f'{obj}_trajectory.json', 'w'))
                shutil.copy(b['dir'] / 'mesh_metric.obj', od / f'{obj}_mesh_metric.obj')
                if not a.no_viz:
                    mp4 = b['dir'] / f'viz_{b["variant"]}.mp4'
                    if not mp4.exists():
                        subprocess.run([PY, '-u', 'compose_v2d/viz_track.py', '--ep', str(ep), '--obj', obj, '--tag', b['tag'], '--variants', b['variant']] + (['--gt'] if split == 'public' else []),
                                       cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        if (b['dir'] / 'viz.mp4').exists(): shutil.move(b['dir'] / 'viz.mp4', mp4)
                    if mp4.exists(): shutil.copy(mp4, od / f'{obj}_viz.mp4')
                sc = f"AUC {b['auc']:.3f}, pos {b['pos_cm']:.2f} cm, ori {b['ori_deg']:.1f} deg (in-script, GT)" if b['auc'] is not None else 'no GT (evaluation split)'
                lines.append(f'- {obj}: tag {b["tag"]}, variant {b["variant"]} -> {sc}; {b.get("note", "")}' + (f'; **{reason}**' if reason else ''))
                summary.append((ep, split, obj, b['tag'], b['variant'], bucket, reason or b.get('note', '')))
            open(od / 'README.md', 'w').write('\n'.join(lines) + '\n')
        print(f'episode {ep} ({split}): ' + ', '.join(f'{o}:{b["tag"]}/{b["variant"]}[{bk}]' + (f'({b["auc"]:.2f})' if b['auc'] is not None else '') for o, (b, bk, _) in chosen.items()), flush=True)
    # top-level screening summary
    L = ['# compose-sequences screening summary', '', 'ok = exported in {public,evaluation}/; rejected = _rejected/ (failed GT score or GT-free screen); unverified = _unverified/ (known mesh issue)', '',
         '| episode | split | object | tag | variant | bucket | reason / selection |', '|---|---|---|---|---|---|---|']
    for ep, split, obj, tag, var, bucket, reason in summary: L.append(f'| {ep} | {split} | {obj} | {tag} | {var} | {bucket} | {reason} |')
    open(out / 'SCREENING.md', 'w').write('\n'.join(L) + '\n')
    n_ok = sum(1 for s in summary if s[5] == 'ok'); print(f'\n{n_ok} object sequences exported as ok, {sum(1 for s in summary if s[5]=="rejected")} rejected, {sum(1 for s in summary if s[5]=="unverified")} unverified -> {a.out}')


if __name__ == '__main__':
    main()
