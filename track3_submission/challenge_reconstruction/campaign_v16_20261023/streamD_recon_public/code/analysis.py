"""Stream D analysis: anchor sensitivity, static-window anchor, per-object bias (object frame / viewing ray) with LOO correction,
rotation bias. Public mocap GT is used only inside the kit-style scorer."""
import json, numpy as np, common as C
from scipy.spatial.transform import Rotation as R
OUT = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamD_recon_public/results'
res = {}
scenes = {}
for ep in C.CLIPS:
    tr, c2w = C.final_tracks(ep); sc = C.Scene(ep, tr); scenes[ep] = (sc, tr, c2w)
    e, pos, ori, al = sc.evaluate(); a = sc.anchor; r = dict(names=sc.names, anchor=sc.names[a], n=sc.n, base={o: list(v) for o, v in e.items()})
    # Q1 anchor perturbation (frame 0 only)
    q1 = {}
    for mag in (1, 2, 3):
        d = []
        for axis in (np.array([0, 0, 1.]), np.array([1., 0, 0]), np.array([0, 1., 0])):
            for s in (1, -1):
                e2 = sc.evaluate(C.perturb_frame0(sc.ach, a, rotvec_world=np.radians(mag) * s * axis))[0]
                d.append([e2[o][0] - e[o][0] for o in sc.names])
        d = np.array(d); q1[f'rot{mag}deg'] = dict(mean_abs_dpos_cm=np.abs(d).mean(0).round(3).tolist(), max_dpos_cm=np.abs(d).max(0).round(3).tolist())
    for mag in (0.005, 0.01):
        d = []
        for axis in np.eye(3):
            for s in (1, -1):
                e2 = sc.evaluate(C.perturb_frame0(sc.ach, a, dpos=mag * s * axis))[0]; d.append([e2[o][0] - e[o][0] for o in sc.names])
        d = np.array(d); q1[f'pos{mag*100:g}cm'] = dict(mean_abs_dpos_cm=np.abs(d).mean(0).round(3).tolist(), max_dpos_cm=np.abs(d).max(0).round(3).tolist())
    r['q1_anchor_perturbation'] = q1
    # Q2 static-window anchor estimate
    import pp as PP
    tw, c, w = PP.principal_twins(C.MESH(sc.names[a])); k = C.static_window(tr[sc.names[a]], c)
    q2 = dict(static_frames=int(k))
    if k:
        Tk = np.tile(np.eye(4), (k, 1, 1)); Tk[:, :3, :3] = R.from_quat(sc.ach[:k, 0, a, 3:]).as_matrix(); Tk[:, :3, 3] = sc.ach[:k, 0, a, :3]
        m = C.mean_pose(Tk); ach2 = sc.ach.copy(); ach2[0, 0, a, :3] = m[:3, 3]; ach2[0, 0, a, 3:] = R.from_matrix(m[:3, :3]).as_quat()
        e2 = sc.evaluate(ach2)[0]; q2['static_anchor'] = {o: list(v) for o, v in e2.items()}
        q2['frame0_vs_window_mean_cm_deg'] = [float(np.linalg.norm(sc.ach[0, 0, a, :3] - m[:3, 3]) * 100), float(np.degrees((R.from_quat(sc.ach[0, 0, a, 3:]) * R.from_matrix(m[:3, :3]).inv()).magnitude()))]
        # oracle: what if frame 0 of the anchor were exactly the GT pose (upper bound of anchor-only gains)
        ach3 = sc.ach.copy(); ach3[0, 0, a] = sc.ref[0, 0, a]; q2['oracle_gt_frame0_anchor'] = {o: list(v) for o, v in sc.evaluate(ach3)[0].items()}
    if len(sc.names) == 2:
        sc_r = C.Scene(ep, tr, order=sc.names[::-1]); e_r = sc_r.evaluate()[0]; q2['anchor_on_other_object'] = dict(anchor=sc_r.names[sc_r.anchor], errors={o: list(v) for o, v in e_r.items()})
    r['q2_static_anchor'] = q2
    # Q3/Q4 bias vectors per object (object frame = GT scan body frame; camera ray in our world rotated by the scene alignment)
    Ral = sc.align_rotation(); q3 = {}
    for i, o in enumerate(sc.names):
        ev = (al[:, 0, i, :3] - sc.ref[:, 0, i, :3]); Rr = R.from_quat(sc.ref[:, 0, i, 3:]).as_matrix(); eo = np.einsum('nji,nj->ni', Rr, ev)
        cen = tr[o][:sc.n, :3, 3]; d = cen - c2w[:sc.n, :3, 3]; d /= np.linalg.norm(d, axis=1, keepdims=True); dg = d @ Ral.T
        along = np.einsum('ni,ni->n', ev, dg); across = np.linalg.norm(ev - along[:, None] * dg, axis=1)
        Ra = R.from_quat(al[:, 0, i, 3:]).as_matrix(); rv = R.from_matrix(Ra @ np.transpose(Rr, (0, 2, 1))).as_rotvec(); ro = np.einsum('nji,nj->ni', Rr, rv)
        vis = sc.gt_vis[o]
        q3[o] = dict(mean_err_world_cm=(ev.mean(0) * 100).round(2).tolist(), mean_err_objframe_cm=(eo.mean(0) * 100).round(2).tolist(),
                     objframe_bias_norm_cm=float(np.linalg.norm(eo.mean(0)) * 100), mean_along_ray_cm=float(along.mean() * 100), mean_across_cm=float(across.mean() * 100),
                     rms_pos_cm=float(np.sqrt((ev ** 2).sum(1).mean()) * 100), rot_mean_rotvec_objframe_deg=np.degrees(ro.mean(0)).round(2).tolist(),
                     rot_bias_norm_deg=float(np.degrees(np.linalg.norm(ro.mean(0)))), rot_mean_deg=float(ori[:, i].mean()), gt_visible_frac=float(vis.mean()))
    r['q3_bias'] = q3; res[ep] = r
    print(ep, json.dumps(dict(base=r['base'], anchor=r['anchor'], q2=q2.get('static_anchor'), oracle=q2.get('oracle_gt_frame0_anchor'))), flush=True)
json.dump(res, open(OUT + '/analysis.json', 'w'), indent=1)
# LOO bias correction per object type (object-frame bias) and pooled along-ray bias
groups = {}
for ep, r in res.items():
    for o in r['names']: groups.setdefault(o, []).append(ep)
loo = {}
for o, eps in groups.items():
    if len(eps) < 2: continue
    for ep in eps:
        others = [x for x in eps if x != ep]; b = np.mean([res[x]['q3_bias'][o]['mean_err_objframe_cm'] for x in others], 0) / 100
        sc, tr, c2w = scenes[ep]; i = sc.names.index(o); ach = sc.ach.copy(); Ra = R.from_quat(ach[:, 0, i, 3:]).as_matrix()
        ach[:, 0, i, :3] -= np.einsum('nij,j->ni', Ra, b); e2 = sc.evaluate(ach)[0]
        loo[f'{ep}:{o}'] = dict(bias_applied_cm=(b * 100).round(2).tolist(), before=res[ep]['base'][o], after_objframe_bias=list(e2[o]))
        # along-ray (depth) bias, pooled over the other clips of the same object
        s = np.mean([res[x]['q3_bias'][o]['mean_along_ray_cm'] for x in others]) / 100
        Ral = sc.align_rotation(); cen = tr[o][:sc.n, :3, 3]; d = cen - c2w[:sc.n, :3, 3]; d /= np.linalg.norm(d, axis=1, keepdims=True)
        ach = sc.ach.copy(); ach[:, 0, i, :3] -= d * s; e3 = sc.evaluate(ach)[0]
        loo[f'{ep}:{o}'].update(ray_bias_applied_cm=float(s * 100), after_ray_bias=list(e3[o]))
        print('LOO', ep, o, loo[f'{ep}:{o}'], flush=True)
# pooled along-ray bias over ALL objects (sensor-level hypothesis), LOO by clip
pool = {}
for ep, r in res.items():
    for o in r['names']:
        others = [(x, p) for x, rr in res.items() for p in rr['names'] if x != ep]; s = np.mean([res[x]['q3_bias'][p]['mean_along_ray_cm'] for x, p in others]) / 100
        sc, tr, c2w = scenes[ep]; i = sc.names.index(o); cen = tr[o][:sc.n, :3, 3]; d = cen - c2w[:sc.n, :3, 3]; d /= np.linalg.norm(d, axis=1, keepdims=True)
        ach = sc.ach.copy(); ach[:, 0, i, :3] -= d * s; pool[f'{ep}:{o}'] = dict(ray_bias_cm=float(s * 100), before=res[ep]['base'][o], after=list(sc.evaluate(ach)[0][o]))
json.dump(dict(loo=loo, pooled_ray=pool), open(OUT + '/loo.json', 'w'), indent=1)
print('done')
