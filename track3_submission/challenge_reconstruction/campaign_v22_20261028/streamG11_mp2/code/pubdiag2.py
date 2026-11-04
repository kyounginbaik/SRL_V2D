"""SCORER-SIDE diagnosis only (public GT read only through the kit scorer path): structure of the per-object error
of a public export after the kit's registration + anchor alignment.
usage: pubdiag.py EXPORT_DIR [EXPORT_DIR...]   prints JSON lines
Per object: kit mean pos/ori; mean error vector (cm) in the GT body frame and in the world (aligned) frame; std;
frame-0 error; rotation error rotvec (GT body frame) mean; for symmetric objects the spin about the axis
(order-2: reduced mod 180) and the tilt; registration seed spread (spin of our mesh's registration across seeds)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020/code/boards')
import bscore
from bscore import K
from v2dlb.mesh_budget import budget_mesh
from scipy.spatial.transform import Rotation as Rot


def run(folder):
    folder = Path(folder); root = folder.parents[1]
    rec = json.loads((root / 'prepared.json').read_text())['record']; ep = rec['episode']
    d = np.load(folder / 'object_poses.npz'); names0 = [str(n) for n in d['names']]
    mj = json.loads((folder / 'meshes.json').read_text()) if (folder / 'meshes.json').exists() else {}
    tracks = {n: (d['world_from_object'][:, i], mj.get(n, str(root / 'objects' / n / 'mesh.obj'))) for i, n in enumerate(names0)}
    M = bscore.mods()['MP-SR']; order, nt = bscore.gt_order(ep); objs = [o for o in order if o in tracks]
    n = min([len(tracks[o][0]) for o in objs] + [nt]); sol_rows, sub_rows = [], []
    for slot, obj in enumerate(objs):
        gt, vis = K.gt_world_T_obj(ep, obj); g = K.pose_cols(gt[:n]); o = K.pose_cols(tracks[obj][0][:n])
        mv, mf = budget_mesh(tracks[obj][1], 4096, 4096); payload = K.encode(vertices=np.asarray(mv, np.float64), faces=np.asarray(mf, np.float64))
        for f in range(n):
            rid = f'track_3/e{ep:06d}/w000/f{f:06d}/o{slot}'
            sol_rows.append(dict(row_id=rid, episode_index=ep, world_index=0, frame_index=f, object_slot=slot, object_name=obj, **{'ref_' + k: float(v[f]) for k, v in g.items()}, reference_geometry_b64=''))
            sub_rows.append(dict(row_id=rid, **{k: float(v[f]) for k, v in o.items()}, **K.PROV, object_mesh_b64=payload if f == 0 else ''))
    sol = pd.DataFrame(sol_rows); sub = pd.DataFrame(sub_rows); sol.loc[0, 'reference_geometry_b64'] = K.encode(**K.reference_geometry(objs, bscore.mods()['AUC']))
    geometry = M._decode_archive(sol['reference_geometry_b64'].iloc[0], 16_000_000); lookup = {str(x): i for i, x in enumerate(geometry['names'])}
    merged = sol.merge(sub[['row_id', *M._REQUIRED_SUBMISSION_COLUMNS, 'object_mesh_b64']], on='row_id')
    ach, ref, _, nm = M._episode_tensors(merged)
    ach = M._normalise_quaternions(ach, 's', ValueError).copy(); ref = M._normalise_quaternions(ref, 's', ValueError).copy()
    axes, orders, regs = [], [], {}
    for slot, name in enumerate(nm):
        rows = merged[merged.object_slot == slot].sort_values(['world_index', 'frame_index']); enc = rows.object_mesh_b64.fillna('').astype(str)
        cand = M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0]); i = lookup[str(name)]
        mesh = (geometry[f'v{i}'], geometry[f'f{i}'].astype(np.int64)); anchor = geometry[f'anchor{i}']
        seed0 = 1_000_003 * ep + 101 * slot
        ca, rot = M._register_object_body(cand, mesh, anchor, seed0)
        # seed spread of the registration (relative rotation to the kit seed)
        spread = []
        for s in range(6):
            ca2, rot2 = M._register_object_body(cand, mesh, anchor, seed0 + 7919 * (s + 1))
            rv = Rot.from_matrix(rot2 @ rot.T).as_rotvec(); spread.append([float(np.degrees(np.linalg.norm(rv))), float(np.linalg.norm(ca2 - ca) * 100)])
        regs[str(name)] = dict(anchor_cm=(ca * 100).round(2).tolist(), seed_spread_deg_cm=spread)
        ach[:, :, slot] = M._convert_object_body(ach[:, :, slot], ca, rot)
        ref[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, slot, 3:]), anchor)
        axes.append(geometry[f'axis{i}']); orders.append(int(geometry[f'order{i}']))
    fixed = [i for i, a in enumerate(axes) if not np.any(a)]
    ach = M._align_registered_scene(ach, ref, axes)
    out = dict(ep=ep, export=str(folder), order=[str(x) for x in nm], rot_anchor_slot=(fixed[0] if fixed else 0), trans_slot=0, objects={})
    for slot, name in enumerate(nm):
        a = ach[:, 0, slot]; r = ref[:, 0, slot]
        e = (a[:, :3] - r[:, :3]) * 100
        Rr = M._rotation_matrix(r[:, 3:]); Ra = M._rotation_matrix(a[:, 3:])
        eb = np.einsum('tji,tj->ti', Rr, e)  # in GT body frame
        Rerr = np.einsum('tji,tjk->tik', Rr, Ra)  # GT-body rotation error
        rv = Rot.from_matrix(Rerr).as_rotvec()
        np.save(f"/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG11_mp2/out/eb_{ep:02d}_{name}.npy", np.stack([eb, e], 0))
        o = dict(pos_mean=float(np.linalg.norm(e, axis=1).mean()), vec_world_mean=e.mean(0).round(2).tolist(), vec_world_std=e.std(0).round(2).tolist(),
                 vec_body_mean=eb.mean(0).round(2).tolist(), vec_body_std=eb.std(0).round(2).tolist(), f0_vec_world=e[0].round(2).tolist(),
                 rotvec_body_mean_deg=np.degrees(rv.mean(0)).round(1).tolist(), rot_geodesic_mean=float(np.degrees(np.linalg.norm(rv, axis=1)).mean()),
                 reg=regs[str(name)])
        ax = axes[slot]
        if np.any(ax):
            # spin about the symmetry axis (swing-twist decomposition of Rerr), and tilt
            q = Rot.from_matrix(Rerr).as_quat()  # x y z w
            proj = q[:, :3] @ ax; tw = 2 * np.degrees(np.arctan2(proj, q[:, 3])); tw = (tw + 180) % 360 - 180
            if orders[slot] == 2: tw = (tw + 90) % 180 - 90
            aa = np.einsum('tij,j->ti', Ra, ax); ar = np.einsum('tij,j->ti', Rr, ax)
            tilt = np.degrees(np.arccos(np.clip(np.sum(aa * ar, 1), -1, 1)))
            o.update(order=orders[slot], spin_mean=float(tw.mean()), spin_std=float(tw.std()), spin_pcts=np.percentile(tw, [10, 50, 90]).round(1).tolist(),
                     spin_f0=float(tw[0]), tilt_mean=float(tilt.mean()), axis=ax.round(3).tolist())
        ori = np.degrees(M._symmetry_orientation_errors(ach, ref, axes, orders))[:, 0, slot]
        o['ori_mean'] = float(ori.mean())
        out['objects'][str(name)] = o
    return out


if __name__ == '__main__':
    for f in sys.argv[1:]:
        print(json.dumps(run(f)), flush=True)
