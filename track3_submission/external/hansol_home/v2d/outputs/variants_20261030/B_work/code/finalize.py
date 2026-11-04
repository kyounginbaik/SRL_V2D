"""Variant B: verify bundles against v8_g10b (byte identity of unchanged files, meshes, ep32), write manifest.json + README.md
into each bundle, and zip it. GT-free (scores were produced earlier by the unchanged public scorer)."""
import json, hashlib, os, zipfile, numpy as np
from pathlib import Path
OUT = Path('/home/hansol/v2d/outputs/variants_20261030'); W = OUT / 'B_work'
V8 = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v8_g10b')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
pub = json.load(open(W / 'public_summary.json')); rules = json.load(open(W / 'rules.json')); cv = json.load(open(W / 'cv_rules.json'))
ev = json.load(open(W / 'eval_variant_summary.json')); qc = json.load(open(W / 'eval_qc_summary.json'))
pertype = json.load(open(W / 'per_type_fam_X.json'))['per_type']
cvs = json.load(open(W / 'cv_summary.json')) if (W / 'cv_summary.json').exists() else {}
BUNDLES = {'B1': 'B1_family_select', 'B2': 'B2_family_consensus'}
RULE_TEXT = {
 'B1': 'Per object type, the whole track of the pose family that beats the base on public+held-out in BOTH mean position and mean '
       'orientation (paired on the same objects), with zero public MP failures, >= 3 public objects and dpos/3cm + dori/30deg <= -0.01; '
       'otherwise the base (= v8_g10b on eval, r8_base on public). Result: plastic_dish_rack -> pointmap_cotracker; every other type -> base.',
 'B2': 'Per frame, soft trust-region weighted SE(3) mean of the base and the type\'s good families (zero public MP failures for the type, '
       'paired dpos <= +0.25 cm, dori <= +3 deg). Candidates are aligned to the base first (half-turn twins about our mesh principal '
       'axes for sweeper/wooden pieces). Weight exp(-d^2/2), d = max(dp/1.5cm, dr/15deg) vs the base; base weight 1. Cups/lid: position '
       'only (rotation = base). Spoon (no public counterpart), blue cup and lid (no good family): base unchanged.'}
for var, bname in BUNDLES.items():
    B = OUT / bname; files = {}; ident, changed = [], []
    for p in sorted(B.rglob('*')):
        if p.is_dir() or p.name in ('manifest.json', 'README.md'): continue
        rel = str(p.relative_to(B)); h = sha(p); files[rel] = h
        (ident if (V8 / rel).exists() and sha(V8 / rel) == h else changed).append(rel)
    v8files = sorted(str(p.relative_to(V8)) for p in V8.rglob('*') if p.is_file())
    assert sorted(files) == v8files, 'file list differs from v8_g10b'
    assert all(c.endswith('object_poses.npz') for c in changed), changed
    assert not any(c.startswith('episode_000032') for c in changed)
    meshes_ok = all(r in ident for r in files if r.endswith('.obj'))
    ch_eps = sorted(int(c.split('/')[0][-6:]) for c in changed)
    objs = {str(ep): {n: dict(source=o.get('source'), move_vs_v8=o['move_vs_v8'], steps_v8=o['steps_v8'], steps_new=o['steps_new'])
                      for n, o in ev[var][str(ep)]['objects'].items() if o.get('source')} for ep in ch_eps}
    pa = pub[var]; pb = pub['fam_base']
    man = dict(variant=bname, base='selected20_final_v8_g10b (byte-identical copy; only object_poses.npz of changed episodes rewritten)',
               rule=RULE_TEXT[var], rule_params=dict(B1=rules['B1']) if var == 'B1' else dict(B2_good_families=rules['B2_good']),
               ground_truth_used=False, eval_gt_read=False, scanned_gt_meshes_used=False, meshes_unchanged=meshes_ok, episode_32='v8_g10b unchanged',
               changed_episodes=ch_eps, changed_objects=objs,
               public_score_of_shipped_rule={k: {kk: pa[k][kk] for kk in ['clips', 'mp', 'sp', 'pos', 'ori', 'MPPE', 'AUC']} for k in ['all', 'public', 'heldout']},
               public_score_r8_base={k: {kk: pb[k][kk] for kk in ['clips', 'mp', 'sp', 'pos', 'ori', 'MPPE', 'AUC']} for k in ['all', 'public', 'heldout']},
               cross_validation=dict(rules_by_fold=cv, scores=cvs.get(var)),
               eval_qc_gt_free=qc[var], files_sha256=files, unchanged_files_byte_identical_to_v8=len(ident), changed_files=changed,
               scorer='tools/reconstruction_audit/score_bundle.py -> bscore (unchanged), run from our mirror B_work/canonical',
               work_dir=str(W))
    (B / 'manifest.json').write_text(json.dumps(man, indent=1))
    fmt = lambda a: f"{a['mp']}/{a['clips']} | {a['sp']}/{a['clips']} | {a['pos']:.3f} | {a['ori']:.2f} | {a['MPPE']:.4f} | {a['AUC']:.4f}"
    lines = [f'# {bname} (Track 3 recon variant {var}, 2026-10-30)', '',
             f'Base: `selected20_final_v8_g10b`, same format. The bundle is a byte-identical copy of v8_g10b; only `object_poses.npz` of the changed episodes is rewritten '
             '(same keys and dtypes, compressed npz, `camera_from_object = inv(world_from_rgb) @ world_from_object`). Meshes are unchanged, ep32 is v8_g10b. No GT, no eval GT, no scanned meshes.', '',
             f'## Rule (the same code ran on public and eval: `B_work/code/variants.py`)', RULE_TEXT[var], '',
             '## Public / held-out score of exactly this rule (unchanged scorer; GT only inside it)', '',
             '| set | MP-SR | SP-SR | pos cm | ori deg | MPPE | AUC |', '|---|---|---|---|---|---|---|']
    for k in ['all', 'public', 'heldout']:
        lines += [f'| r8_base {k} | {fmt(pb[k])} |', f'| **{var} {k}** | {fmt(pa[k])} |']
    if cvs.get(var): lines += ['', f"2-fold check (rule chosen on one split, scored on the other): {cvs[var]['text']}"]
    lines += ['', f'## Changed eval episodes: {ch_eps}', '', '| ep | object | source | mean/max move vs v8 cm | max deg | IoU median v8->new | paired dIoU | mean rot step deg v8->new | max rot step |', '|---|---|---|---|---|---|---|---|---|']
    for ep in ch_eps:
        for n, o in objs[str(ep)].items():
            q = qc[var].get(f'{ep}/{n}', {})
            lines.append(f"| {ep} | {n} | {o['source']} | {o['move_vs_v8']['mean_cm']:.2f}/{o['move_vs_v8']['max_cm']:.2f} | {o['move_vs_v8']['max_deg']:.1f} | "
                         f"{q.get('iou_median', [0, 0])[0]:.3f}->{q.get('iou_median', [0, 0])[1]:.3f} | {q.get('d_iou_paired', 0):+.4f} | {o['steps_v8']['mean_deg']:.2f}->{o['steps_new']['mean_deg']:.2f} | {o['steps_v8']['max_deg']:.1f}->{o['steps_new']['max_deg']:.1f} |")
    lines += ['', 'Details: `manifest.json` (sources, sha256 of every file, scores, QC). Work folder: `' + str(W) + '`.']
    (B / 'README.md').write_text('\n'.join(lines) + '\n')
    z = OUT / f'recon_varB_{bname}.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(B.rglob('*')):
            if p.is_file(): zf.write(p, f'{bname}/{p.relative_to(B)}')
    print(var, 'changed eps', ch_eps, 'unchanged identical', len(ident), 'changed', len(changed), 'meshes ok', meshes_ok, 'zip', z, sha(z)[:12], os.path.getsize(z))
