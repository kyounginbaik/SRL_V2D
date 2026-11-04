"""Hedge rule on one eval episode on top of selected20_final_v10_g12 + worst-case diagnostics. GT-free.
Sources: our trackers (raw, multimap, pointmap_cotracker, bundle_adjusted), fro (final_recon for 13 eps / final_recon_ours for 14,15,16,20,25,28,30)
and compose21 (only spoon / rack: team-mesh objects without scan templates). A source object is admitted only if its mesh is OUR team mesh
(byte-identical, or same mesh resampled: sorted extents within 0.3 cm, or same vertex count and extents within 1.0 cm).
Writes out/eval_<TAG>/ep<EP>.npz (hedged world_from_object) and ep<EP>.json (info + per-hypothesis time-mean errors before/after).
usage: make_eval.py TAG GP GR MP MR SMOOTH EP [EP ...]"""
import sys, json, os, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(__file__))
import hedge as H
R = '/data1/SRL_V2D/challenge_reconstruction'; RR = f'{R}/reconstruction_recovery_20261026'
BASE = f'{RR}/selected20_final_v10_g12'; CAN = f'{RR}/canonical/evaluation'
FRO7 = (14, 15, 16, 20, 25, 28, 30)
C21_OK = ('wooden_spoon', 'plastic_dish_rack')   # compose objects tracked with our team mesh, no scan template
TAG = sys.argv[1]; GP, GR, MP, MR = map(float, sys.argv[2:6]); SM = int(sys.argv[6]); MODE = os.environ.get('MODE', 'frame'); EPS = [int(x) for x in sys.argv[7:]]
O = Path(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14/out/eval_{TAG}'); O.mkdir(parents=True, exist_ok=True)

def mesh_ok(p, ours):
    a = trimesh.load(p, force='mesh', process=False); b = trimesh.load(ours, force='mesh', process=False)
    if len(a.vertices) == len(b.vertices) and np.abs(np.asarray(a.vertices) - np.asarray(b.vertices)).max() < 1e-7: return True, 'identical'
    d = np.abs(np.sort(a.extents) - np.sort(b.extents)).max() * 100
    if d <= 0.3: return True, f'resampled ext {d:.2f}cm'
    if len(a.vertices) == len(b.vertices) and d <= 1.0: return True, f'same-count ext {d:.2f}cm'
    return False, f'different mesh ext {d:.2f}cm n{len(a.vertices)}/{len(b.vertices)}'

def sources(ep, b, names, meshes):
    F = len(b['world_from_object']); W = b['world_from_rgb']; out = []; adm = {}
    def put(tag, Cdict, mdict):
        T = np.full((F, len(names), 4, 4), np.nan); ms = [None] * len(names)
        for k, n in enumerate(names):
            if n not in Cdict: continue
            ok, why = mesh_ok(mdict[n], meshes[k]); adm[f'{tag}:{n}'] = why
            if not ok: continue
            C = Cdict[n]; Fa = min(F, len(C)); T[:Fa, k] = W[:Fa] @ C[:Fa]; ms[k] = mdict[n]
        if any(m is not None for m in ms): out.append((tag, T, ms))
    for s in ('raw', 'multimap', 'pointmap_cotracker', 'bundle_adjusted'):
        p = f'{CAN}/episode_{ep:06d}/exports/{s}/object_poses.npz'
        if not os.path.exists(p): continue
        a = np.load(p); an = [str(x) for x in a['names']]
        put(s, {n: a['camera_from_object'][:, an.index(n)] for n in names if n in an},
            {n: f'{CAN}/episode_{ep:06d}/objects/{n}/mesh.obj' for n in names})
    fd = f'{R}/dexagent_remaining7_20261019/source/final_recon_ours/episode_{ep:06d}' if ep in FRO7 else f'{R}/final_recon/episode_{ep:06d}'
    pf = np.load(f'{fd}/object_poses.npz', allow_pickle=True); nf = [str(x) for x in pf['names']]
    Cf = np.linalg.inv(pf['world_from_rgb'].astype(float))[:, None] @ pf['world_from_object']
    put('fro', {n: Cf[:, nf.index(n)] for n in names if n in nf}, {n: f'{fd}/assets/{n}.obj' for n in names})
    D = f'{R}/campaign_v16_20261023/compose_target/episode_{ep:06d}'
    if os.path.exists(f'{D}/camera_motion.npz'):
        cm = np.load(f'{D}/camera_motion.npz'); Cc = {}
        for n in names:
            if n in C21_OK and os.path.exists(f'{D}/{n}_trajectory.npz'):
                Cc[n] = cm['T_a_from_rectL'] @ np.linalg.inv(cm['cam0_T_cam']) @ np.load(f'{D}/{n}_trajectory.npz')['T_world_obj']
        put('c21', Cc, {n: f'{D}/{n}_mesh_metric.obj' for n in names})
    return out, adm

def tm_errors(X7, Al, axes, orders):
    """time-mean pos cm / ori deg of reference X (anchor 7-vec, already in the base gauge, same f0) vs aligned hypothesis Al."""
    res = []
    for s in range(X7.shape[1]):
        ok = np.isfinite(Al[:, s]).all(-1)
        if ok.sum() == 0: res.append(None); continue
        p = np.linalg.norm(X7[ok, s, :3] - Al[ok, s, :3], axis=-1) * 100
        o = np.degrees(M._symmetry_orientation_errors(X7[ok, s][:, None, None], Al[ok, s][:, None, None], [axes[s]], [orders[s]]))[:, 0, 0]
        res.append((float(p.mean()), float(o.mean())))
    return res
M = H.M
for ep in EPS:
    d = f'{BASE}/episode_{ep:06d}'; b = dict(np.load(f'{d}/object_poses.npz')); names = [str(x) for x in b['names']]
    mj = json.load(open(f'{d}/meshes.json')); meshes = [f'{d}/' + mj[n]['file'] for n in names]
    alts, adm = sources(ep, b, names, meshes)
    Tn, info = H.hedge_clip(b['world_from_object'], names, meshes, alts, GP, GR, MP, MR, SM, ep, MODE)
    # diagnostics: each full hypothesis (alt where admitted, base elsewhere), aligned onto base as the kit would
    infos = [H.mesh_info(m, n) for m, n in zip(meshes, names)]; axes = [i['axis'] for i in infos]; orders = [i['order'] for i in infos]
    B7 = np.stack([H.to_anchor(b['world_from_object'][:, s], infos[s]['anc']) for s in range(len(names))], 1)
    N7 = np.stack([H.to_anchor(Tn[:, s], infos[s]['anc']) for s in range(len(names))], 1)
    diag = {}
    for tag, T, ms in alts:
        A = B7.copy(); have = []
        for s in range(len(names)):
            if ms[s] is None: continue
            Ts = T[:, s].copy(); ok = np.isfinite(Ts).all((1, 2)); Ts[~ok] = b['world_from_object'][~ok, s]
            a, ang = H.body_convert(Ts, ms[s], meshes[s], names[s], 1_000_003 * ep + 101 * s)
            if ang > 20: continue
            a[~ok] = np.nan; A[:, s] = a; have.append(s)
        if not have: continue
        Af = np.where(np.isfinite(A), A, B7); Al = H.align(Af, B7, axes); Al[~np.isfinite(A).all(-1)] = np.nan
        for s in range(len(names)):
            if s not in have: Al[:, s] = np.nan
        eb, en = tm_errors(B7, Al, axes, orders), tm_errors(N7, Al, axes, orders)
        diag[tag] = {names[s]: dict(before=eb[s], after=en[s]) for s in have}
    np.savez(O / f'ep{ep}.npz', world_from_object=Tn, names=np.array(names))
    json.dump(dict(ep=ep, params=dict(MODE=MODE, GP=GP, GR=GR, MP=MP, MR=MR, SMOOTH=SM), admitted=adm, info=info, per_hyp=diag), open(O / f'ep{ep}.json', 'w'), indent=1)
    print(ep, {k: (round(v['move_cm_mean'], 2), round(v['move_deg_mean'], 1)) for k, v in info['objects'].items()}, flush=True)
    for tag, dd in diag.items():
        print('   ', tag, {n: (tuple(round(x, 2) for x in v['before']), tuple(round(x, 2) for x in v['after'])) for n, v in dd.items()}, flush=True)
