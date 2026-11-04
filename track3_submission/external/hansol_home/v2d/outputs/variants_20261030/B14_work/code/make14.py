"""Variant B on v14: build public/held-out exports (base g15_camrt) in our mirror and the eval bundles (base selected20_final_v14_g17).
usage: make14.py public | eval        GT is not read here (public GT enters only the unchanged scorer, run separately)."""
import sys, os, json, shutil, hashlib, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__)); import v14lib as L
V = L.V
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026'); SRC = RR / 'canonical'
OUT = Path('/home/hansol/v2d/outputs/variants_20261030'); BW = OUT / 'B_work'; W = OUT / 'B14_work'
V8, V14 = RR / 'selected20_final_v8_g10b', RR / 'selected20_final_v14_g17'
mx = lambda a: float(np.abs(a).max())
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
CV = json.load(open(BW / 'cv_rules.json'))


def lag_rows(tag, base14, C_pre, ex, cam, d, rules):
    rows = []
    names = [str(x) for x in base14['names']]; F = len(base14['world_from_object'])
    for k, n in enumerate(names):
        for f in sorted(set(rules[1].get(n, []) + ([rules[0][n]] if n in rules[0] else []))):
            C = L.family_cam(ex, f, n, F)
            if C is None: continue
            ok = np.isfinite(C).all((1, 2)); Cf = np.where(ok[:, None, None], C, C_pre[:, k]); bp = base14['world_from_object'][:, k, :3, 3]
            r = {}
            for rt in [False, True]:
                Wf, _, ok2 = L.g15(Cf, ok, cam, d, rt); r['retimed' if rt else 'naive'] = L.best_lag(Wf[:, :3, 3], ok2, bp)
            rows.append(dict(clip=tag, object=n, family=f, delta=d if np.ndim(d) == 0 else sorted(set(np.asarray(d).tolist())), **r))
    return rows


def public():
    lines = {}; moves = {}; lags = []
    for split in ['public', 'public_heldout']:
        for root in sorted((SRC / split).glob('episode_*')):
            e = int(root.name[-6:]); gd = root / 'exports' / 'g15_camrt'; ex = str(root / 'exports')
            b = dict(np.load(gd / 'object_poses.npz')); Cpre = np.load(root / 'exports' / 'r8_base' / 'object_poses.npz')['camera_from_object']
            mj = json.loads((gd / 'meshes.json').read_text()); names = [str(x) for x in b['names']]; F = len(b['world_from_object'])
            cam = np.load(L.G15 / 'cam' / f'public_{e}' / 'camera.npz')['world_from_camera'][:F]; d = L.DELTA[f'public_{e}']
            for k in range(len(names)):   # the operator reproduces the team base exactly
                Wk, Ck, _ = L.g15(Cpre[:, k], np.ones(F, bool), cam, d); assert mx(Wk - b['world_from_object'][:, k]) < 1e-12 and mx(Ck - b['camera_from_object'][:, k]) < 1e-12
            mroot = BW / 'canonical' / split / root.name / 'exports'
            o = mroot / 'v14_base'
            if not o.exists():
                o.mkdir(parents=True); shutil.copy2(gd / 'object_poses.npz', o); shutil.copy2(gd / 'meshes.json', o); shutil.copy2(gd / 'public_score.json', o / 'score_team.json')
                lines.setdefault('base', []).append(str(o))
            jobs = [('v14_B1', L.SHIPPED, 'B1', True), ('v14_B2', L.SHIPPED, 'B2', True), ('v14_B1n', L.SHIPPED, 'B1', False), ('v14_B2n', L.SHIPPED, 'B2', False),
                    ('v14_B2cv', (CV['heldout_only' if split == 'public' else 'public_only']['B1'], CV['heldout_only' if split == 'public' else 'public_only']['B2_good']), 'B2', True)]
            jobs += [(f'fam14_{f}', ({n: f for n in names}, {}), 'B1', True) for f in V.FAMS]
            for tag, rules, var, rt in jobs:
                T, Cm, info = L.apply14(b, Cpre, ex, cam, d, mj, rules, var, rt)
                out = mroot / tag
                if out.exists(): shutil.rmtree(out)
                ch = mx(T - b['world_from_object']) > 1e-9
                if tag.startswith('fam14_') and not any(v.get('coverage') for v in info.values()): continue
                out.mkdir(parents=True)
                if not ch:
                    for f in ['object_poses.npz', 'meshes.json']: shutil.copy2(gd / f, out / f)
                    shutil.copy2(gd / 'public_score.json', out / 'public_score.json'); (out / 'source.txt').write_text('unchanged = g15_camrt (team score copied; re-scored base checks determinism)\n'); continue
                dd = dict(b); dd['world_from_object'] = T; dd['camera_from_object'] = Cm
                np.savez(out / 'object_poses.npz', **dd); shutil.copy2(gd / 'meshes.json', out / 'meshes.json')
                (out / 'source.json').write_text(json.dumps(info, indent=1)); lines.setdefault(tag.split('_')[0] if tag.startswith('fam14') else tag, []).append(str(out))
                if not tag.startswith('fam14_'):
                    dp, dr = V.dist(T, b['world_from_object']); moves[f'{split}/{e}/{tag}'] = dict(info={n: v.get('source') for n, v in info.items()}, max_cm=float(dp.max()), max_deg=float(dr.max()), mean_cm=float(dp.mean()))
                    print(split, e, tag, {n: v.get('source') for n, v in info.items()}, f'max move {dp.max():.2f} cm {dr.max():.1f} deg', flush=True)
            lags += lag_rows(f'{split}/{e}', b, Cpre, ex, cam, d, L.SHIPPED)
    W.mkdir(exist_ok=True); (W / 'lists').mkdir(exist_ok=True)
    allv = [x for k, v in lines.items() for x in v]
    (W / 'lists' / 'public14.txt').write_text('\n'.join(allv) + '\n')
    (W / 'public_moves14.json').write_text(json.dumps(moves, indent=1)); (W / 'lag_public.json').write_text(json.dumps(lags, indent=1))
    print({k: len(v) for k, v in lines.items()}, len(allv), 'to score')


def evaluation():
    NAMES = {'B1': 'B14_family_select', 'B2': 'B14_family_consensus'}; summary = {}; lags = []
    for var, bname in NAMES.items():
        dst = OUT / bname
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(V14, dst, copy_function=shutil.copy2); summary[var] = {}
        for ep_dir in sorted(V14.glob('episode_*')):
            e = int(ep_dir.name[-6:])
            if e == 32: continue
            b = dict(np.load(ep_dir / 'object_poses.npz')); names = [str(x) for x in b['names']]; F = len(b['world_from_object'])
            Cpre = np.load(V8 / ep_dir.name / 'object_poses.npz')['camera_from_object']
            cam = np.load(ep_dir / 'camera.npz')['world_from_camera'][:F]; d = L.DELTA[f'evaluation_{e}']
            mj = json.loads((ep_dir / 'meshes.json').read_text()); meshes = {n: str(ep_dir / mj[n]['file']) for n in names}
            ex = str(SRC / 'evaluation' / ep_dir.name / 'exports')
            for k, n in enumerate(names):   # exact base reproduction for every slot a rule can touch
                if n in L.SHIPPED[0] or L.SHIPPED[1].get(n):
                    Wk, Ck, _ = L.g15(Cpre[:, k], np.ones(F, bool), cam, d); assert mx(Wk - b['world_from_object'][:, k]) < 1e-12 and mx(Ck - b['camera_from_object'][:, k]) < 1e-12, (e, n)
            T, Cm, info = L.apply14(b, Cpre, ex, cam, d, meshes, L.SHIPPED, var, True)
            if var == 'B2': lags += lag_rows(f'evaluation/{e}', b, Cpre, ex, cam, d, L.SHIPPED)
            if mx(T - b['world_from_object']) <= 1e-9: continue
            dd = dict(b); dd['world_from_object'] = T.astype(b['world_from_object'].dtype); dd['camera_from_object'] = Cm.astype(b['camera_from_object'].dtype)
            np.savez_compressed(dst / ep_dir.name / 'object_poses.npz', **dd)
            objs = {}
            for k, n in enumerate(names):
                dp, dr = V.dist(T[:, k], b['world_from_object'][:, k])
                if dp.max() < 1e-7 and dr.max() < 1e-5: continue
                st_b = V.dist(b['world_from_object'][1:, k], b['world_from_object'][:-1, k]); st_n = V.dist(T[1:, k], T[:-1, k])
                st = lambda s: dict(max_cm=float(s[0].max()), max_deg=float(s[1].max()), over_2cm=int((s[0] > 2).sum()), over_30deg=int((s[1] > 30).sum()), mean_cm=float(s[0].mean()), mean_deg=float(s[1].mean()))
                objs[n] = dict(info.get(n, {}), move_vs_v14=dict(mean_cm=float(dp.mean()), max_cm=float(dp.max()), mean_deg=float(dr.mean()), max_deg=float(dr.max())), steps_v14=st(st_b), steps_new=st(st_n))
            summary[var][e] = dict(objects=objs, npz_sha256=sha(dst / ep_dir.name / 'object_poses.npz'),
                                   rel_incons_new=mx(np.einsum('tij,tnjk->tnik', dd['world_from_rgb'], dd['camera_from_object']) - dd['world_from_object']),
                                   rel_incons_v14=mx(np.einsum('tij,tnjk->tnik', b['world_from_rgb'], b['camera_from_object']) - b['world_from_object']))
            for tag, arr in [('v14', b), (var, dd)]:
                q = W / 'qc' / tag / ep_dir.name
                if tag == 'v14' and (q / 'object_poses.npz').exists(): continue
                q.mkdir(parents=True, exist_ok=True); np.savez(q / 'object_poses.npz', **arr); (q / 'meshes.json').write_text(json.dumps(meshes))
            print(var, e, {n: (o.get('source'), round(o['move_vs_v14']['mean_cm'], 2), round(o['move_vs_v14']['max_cm'], 2), round(o['move_vs_v14']['max_deg'], 1)) for n, o in objs.items()}, flush=True)
    (W / 'eval_variant_summary14.json').write_text(json.dumps(summary, indent=1)); (W / 'lag_eval.json').write_text(json.dumps(lags, indent=1))
    ql = [f'{SRC}/evaluation/{q.name} {q}' for tag in ['v14', 'B1', 'B2'] for q in sorted((W / 'qc' / tag).glob('episode_*')) if not (q / 'qc.json').exists()]
    (W / 'lists' / 'qc14.txt').write_text('\n'.join(ql) + '\n'); print(len(ql), 'qc jobs')


if __name__ == '__main__':
    W.mkdir(exist_ok=True); (W / 'lists').mkdir(exist_ok=True)
    public() if sys.argv[1] == 'public' else evaluation()
