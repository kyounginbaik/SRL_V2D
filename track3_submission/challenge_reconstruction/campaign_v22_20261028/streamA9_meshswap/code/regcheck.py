"""A9 scorer-side (scans used ONLY here, inside the kit registration, as Y9 reg_compare / twin_margin):
for one (ep, obj): write v6 mesh mapped into the submitted body frame (T_icp and its twin T_twin; our meshes only),
then kit-register (exact kit rule, kit seed, and extra seeds) the submitted mesh, v6@T_icp and v6@T_twin against the scan and
report the relative rotation between registrations (~0 = same twin, ~180 = opposite half-turn twin).
Usage: regcheck.py AUDIT.json OUT.json"""
import sys, json, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamY9_mesh')
import twin_margin as TM
from a9lib import *
OUTM = CAMP / 'streamA9_meshswap/meshes_audit'

def tmesh(src, T, out):
    m = trimesh.load(src, force='mesh', process=False); m.apply_transform(np.asarray(T)); out.parent.mkdir(parents=True, exist_ok=True); m.export(out); return str(out)

def reg(path, ref, seed):
    mv, mf = TM.budget_mesh(path, 4096, 4096); cand = (np.asarray(mv, float), np.asarray(mf, np.int64))
    sols, _ = TM.solutions(cand, ref, seed, 0, np.random.default_rng(0))
    best = min(sols, key=lambda s: s[2]); return best[0], best[2]

def main():
    A = json.loads(Path(sys.argv[1]).read_text()); res = []
    for r in A:
        ep, obj = r['ep'], r['obj']
        try: g = TM.K.reference_geometry([obj], TM.AUC)
        except Exception as e:
            res.append(dict(ep=ep, obj=obj, note=f'no public scan ({e}); kit registration not checkable')); print(ep, obj, 'no scan'); continue
        ref = (g['v0'], g['f0'].astype(np.int64)); names = [str(n) for n in src_poses(ep)[1]['names']]; slot = names.index(obj)
        kit_seed = 1_000_003 * ep + 101 * slot; seeds = [kit_seed] + [int(s) for s in np.random.default_rng(ep * 31 + slot).integers(1, 2**31 - 1, 7)]
        sub = str(comp_component(ep) / 'meshes' / ename(ep) / f'{obj}.obj'); v6 = str(V6 / ename(ep) / 'meshes' / f'{obj}.obj')
        mi = tmesh(v6, r['Ticp'], OUTM / f'ep{ep:02d}' / f'{obj}_v6_icp.obj')
        mt = tmesh(v6, r['Ttwin_icp'], OUTM / f'ep{ep:02d}' / f'{obj}_v6_twin.obj') if 'Ttwin_icp' in r else None
        rows = []
        for s in seeds:
            Rs, es = reg(sub, ref, s); Ri, ei = reg(mi, ref, s); row = dict(seed=s, sub_vs_v6icp_deg=angdeg(Rs, Ri), sub_score_cm=es * 100, v6icp_score_cm=ei * 100)
            if mt: Rt, et = reg(mt, ref, s); row.update(sub_vs_v6twin_deg=angdeg(Rs, Rt), v6twin_score_cm=et * 100, v6icp_vs_v6twin_deg=angdeg(Ri, Rt))
            rows.append(row)
        agree = [x['sub_vs_v6icp_deg'] < 45 for x in rows]
        out = dict(ep=ep, obj=obj, kit_seed=rows[0], frac_seeds_sub_same_twin_as_v6=float(np.mean(agree)),
                   v6icp_vs_twin_distinct=float(np.mean([x.get('v6icp_vs_v6twin_deg', 0) > 135 for x in rows])), seeds=rows)
        res.append(out); print(ep, obj, 'kit seed: sub vs v6icp %.1f deg, sub vs v6twin %s deg; frac seeds same twin %.2f; v6icp/v6twin distinct %.2f' % (
            rows[0]['sub_vs_v6icp_deg'], round(rows[0].get('sub_vs_v6twin_deg', float('nan')), 1), out['frac_seeds_sub_same_twin_as_v6'], out['v6icp_vs_twin_distinct']), flush=True)
        Path(sys.argv[2]).write_text(json.dumps(res, indent=1, default=float) + '\n')
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1, default=float) + '\n')

if __name__ == '__main__':
    main()
