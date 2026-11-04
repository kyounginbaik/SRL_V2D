"""Variant B: apply rules B1/B2 (variants.py) to the 20 public/held-out clips (base r8_base) in our mirror; write exports/<V>
for changed clips only and the scoring list. GT is not read here."""
import sys, os, json, shutil, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__)); import variants as V
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical')
W = Path('/home/hansol/v2d/outputs/variants_20261030/B_work')
rules = V.type_rules(json.load(open(W / 'per_type_fam_X.json'))['per_type'])
print('B1 per-type family:', rules[0]); print('B2 good families:', rules[1])
(W / 'rules.json').write_text(json.dumps(dict(B1=rules[0], B2_good=rules[1]), indent=1))
lines, summ = [], {}
for split in ['public', 'public_heldout']:
    for root in sorted((SRC / split).glob('episode_*')):
        bd = root / 'exports' / 'r8_base'; b = dict(np.load(bd / 'object_poses.npz')); mj = json.loads((bd / 'meshes.json').read_text())
        for var in ['B1', 'B2']:
            T, info = V.apply(b, str(root / 'exports'), mj, rules, var)
            dp, dr = V.dist(T, b['world_from_object'])
            ch = bool(np.abs(T - b['world_from_object']).max() > 1e-9)
            summ[f'{split}/{root.name}/{var}'] = dict(changed=ch, info=info, max_move_cm=float(dp.max()), max_move_deg=float(dr.max()), mean_move_cm=float(dp.mean()), mean_move_deg=float(dr.mean()))
            out = W / 'canonical' / split / root.name / 'exports' / var
            if out.exists(): shutil.rmtree(out)
            out.mkdir(parents=True)
            if not ch:
                for f in ['object_poses.npz', 'meshes.json', 'public_score.json']: shutil.copy2(bd / f, out / f)
                (out / 'source.txt').write_text('unchanged = r8_base (score copied; scorer is deterministic)\n'); continue
            d = dict(b); d['world_from_object'] = T; d['camera_from_object'] = np.linalg.inv(b['world_from_rgb'])[:, None] @ T
            np.savez(out / 'object_poses.npz', **d); shutil.copy2(bd / 'meshes.json', out / 'meshes.json')
            (out / 'source.json').write_text(json.dumps(info, indent=1)); lines.append(str(out))
            print(split, root.name, var, {k: v.get('source') for k, v in info.items()}, f'max move {dp.max():.2f} cm {dr.max():.1f} deg', flush=True)
(W / 'lists').mkdir(exist_ok=True); (W / 'lists' / 'variants_public.txt').write_text('\n'.join(lines) + '\n')
(W / 'public_variant_moves.json').write_text(json.dumps(summ, indent=1)); print(len(lines), 'to score')
