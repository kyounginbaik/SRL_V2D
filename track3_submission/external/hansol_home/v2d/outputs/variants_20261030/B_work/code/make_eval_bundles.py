"""Variant B: apply rules B1/B2 (variants.py, identical code to the public run) to eval with base = selected20_final_v8_g10b.
Bundle = byte-identical copy of v8_g10b; only object_poses.npz of changed episodes is rewritten (same keys/dtypes, compressed).
ep32 is never changed. Spoon has no public counterpart -> no rule -> base. Writes QC input folders too. No GT anywhere."""
import sys, os, json, shutil, hashlib, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__)); import variants as V
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
V8 = RR / 'selected20_final_v8_g10b'; CAN = RR / 'canonical' / 'evaluation'
OUT = Path('/home/hansol/v2d/outputs/variants_20261030'); W = OUT / 'B_work'
NAMES = {'B1': 'B1_family_select', 'B2': 'B2_family_consensus'}
rules = V.type_rules(json.load(open(W / 'per_type_fam_X.json'))['per_type'])
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
summary = {}
for var, bname in NAMES.items():
    dst = OUT / bname
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(V8, dst, copy_function=shutil.copy2)
    summary[var] = {}
    for ep_dir in sorted(V8.glob('episode_*')):
        ep = int(ep_dir.name[-6:])
        if ep == 32: continue
        b = dict(np.load(ep_dir / 'object_poses.npz')); names = [str(x) for x in b['names']]
        mj = json.loads((ep_dir / 'meshes.json').read_text()); meshes = {n: str(ep_dir / mj[n]['file']) for n in names}
        # sanity: v8 camera_from_object = inv(world_from_rgb) @ world_from_object
        rel = np.abs(np.linalg.inv(b['world_from_rgb'])[:, None] @ b['world_from_object'] - b['camera_from_object']).max()
        T, info = V.apply(b, str(CAN / ep_dir.name / 'exports'), meshes, rules, var)
        if np.abs(T - b['world_from_object']).max() <= 1e-9: continue
        d = dict(b); d['world_from_object'] = T.astype(b['world_from_object'].dtype)
        d['camera_from_object'] = (np.linalg.inv(b['world_from_rgb'])[:, None] @ T).astype(b['camera_from_object'].dtype)
        np.savez_compressed(dst / ep_dir.name / 'object_poses.npz', **d)
        objs = {}
        for k, n in enumerate(names):
            dp, dr = V.dist(T[:, k], b['world_from_object'][:, k])
            if dp.max() < 1e-7 and dr.max() < 1e-5: continue
            st_b = V.dist(b['world_from_object'][1:, k], b['world_from_object'][:-1, k]); st_n = V.dist(T[1:, k], T[:-1, k])
            objs[n] = dict(info.get(n, {}), move_vs_v8=dict(mean_cm=float(dp.mean()), max_cm=float(dp.max()), mean_deg=float(dr.mean()), max_deg=float(dr.max())),
                           steps_v8=dict(max_cm=float(st_b[0].max()), max_deg=float(st_b[1].max()), over_2cm=int((st_b[0] > 2).sum()), over_30deg=int((st_b[1] > 30).sum()), mean_cm=float(st_b[0].mean()), mean_deg=float(st_b[1].mean())),
                           steps_new=dict(max_cm=float(st_n[0].max()), max_deg=float(st_n[1].max()), over_2cm=int((st_n[0] > 2).sum()), over_30deg=int((st_n[1] > 30).sum()), mean_cm=float(st_n[0].mean()), mean_deg=float(st_n[1].mean())))
        summary[var][ep] = dict(objects=objs, v8_relation_err=float(rel), npz_sha256=sha(dst / ep_dir.name / 'object_poses.npz'))
        for tag, arr in [('v8', b), (var, d)]:
            q = W / 'qc' / tag / ep_dir.name
            if tag == 'v8' and (q / 'object_poses.npz').exists(): continue
            q.mkdir(parents=True, exist_ok=True); np.savez(q / 'object_poses.npz', **arr); (q / 'meshes.json').write_text(json.dumps(meshes))
        print(var, ep, {n: (o.get('source'), round(o['move_vs_v8']['mean_cm'], 2), round(o['move_vs_v8']['max_cm'], 2), round(o['move_vs_v8']['max_deg'], 1)) for n, o in objs.items()}, flush=True)
(W / 'eval_variant_summary.json').write_text(json.dumps(summary, indent=1))
lines = [f'{CAN}/{q.name} {q}' for tag in ['v8', 'B1', 'B2'] for q in sorted((W / 'qc' / tag).glob('episode_*')) if not (q / 'qc.json').exists()]
(W / 'lists' / 'qc.txt').write_text('\n'.join(lines) + '\n'); print(len(lines), 'qc jobs')
