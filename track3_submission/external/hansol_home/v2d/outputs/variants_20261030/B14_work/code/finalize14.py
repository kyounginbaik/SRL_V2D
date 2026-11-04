"""Variant B on v14: verify bundles against selected20_final_v14_g17 (byte identity of every file except the rewritten object_poses.npz,
meshes, ep32; untouched slots and non-pose keys equal), write manifest.json + README.md, zip. GT-free."""
import json, hashlib, os, zipfile, numpy as np
from pathlib import Path
OUT = Path('/home/hansol/v2d/outputs/variants_20261030'); W = OUT / 'B14_work'; BW = OUT / 'B_work'
V14 = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
pub = json.load(open(W / 'public_summary14.json')); rules = json.load(open(BW / 'rules.json')); rules14 = json.load(open(W / 'rules14.json'))
cv = json.load(open(BW / 'cv_rules.json')); ev = json.load(open(W / 'eval_variant_summary14.json')); qc = json.load(open(W / 'eval_qc_summary14.json'))
lags = {k: json.load(open(W / f'lag_{k}.json')) for k in ['public', 'eval']}
BUNDLES = {'B1': ('B14_family_select', 'recon_varB14_B1_family_select.zip'), 'B2': ('B14_family_consensus', 'recon_varB14_B2_family_consensus.zip')}
KEYS = ['clips', 'mp', 'sp', 'pos', 'ori', 'MPPE', 'AUC']
RULE_TEXT = {
 'B1': 'Shipped variant-B rule, unchanged: per object type, the whole track of the pose family that beat the base on public+held-out in BOTH mean '
       'position and orientation with zero MP failures (n >= 3, dpos/3cm + dori/30deg <= -0.01); otherwise the base. Result: plastic_dish_rack -> '
       'pointmap_cotracker, every other type -> v14_g17.',
 'B2': 'Shipped variant-B rule, unchanged: per frame, soft trust-region weighted SE(3) mean of the base and the type\'s good families '
       '(B_work/rules.json). Weight exp(-d^2/2), d = max(dp/1.5cm, dr/15deg) vs the base, base weight 1; half-turn twins for sweeper/pieces; '
       'cups/lid position only. Spoon, blue cup, lid, piece_2: base. The same weights are applied to world_from_object and camera_from_object.'}
CLOCK = ('Every family track is first mapped onto the v14 clock with the exact G15 operator the team used for the base: '
         'world = retime(G15 camera @ family camera_from_object, delta_ep), camera = retime(family camera_from_object, delta_ep) '
         '(streamG15/code/retime.py shift, out/sync_delta_t.json; ep20 piecewise 0.925/1.85). Checked: the same operator on v8_g10b reproduces '
         'v14_g17 exactly for every slot a rule touches (max abs diff 0). Missing family frames: B1 filled with the pre-G15 base, B2 weight 0 '
         '(mask carried through the shift).')


def lag_text(rows):
    r = [x for x in rows if x['naive'] and x['retimed']]
    n = np.array([x['naive'][0] for x in r]); t = np.array([x['retimed'][0] for x in r])
    e0 = np.mean([x['naive'][2] for x in r]); e1 = np.mean([x['retimed'][2] for x in r])
    return dict(pairs=len(r), best_lag_naive_median=float(np.median(n)), best_lag_retimed_median=float(np.median(t)),
                frac_retimed_within_0p125=float(np.mean(np.abs(t) <= 0.125)), moving_frame_gap_cm_naive=float(e0), moving_frame_gap_cm_retimed=float(e1))


for var, (bname, zname) in BUNDLES.items():
    B = OUT / bname; files = {}; ident, changed = [], []
    for p in sorted(B.rglob('*')):
        if p.is_dir() or p.name in ('manifest.json', 'README.md'): continue
        rel = str(p.relative_to(B)); h = sha(p); files[rel] = h
        (ident if (V14 / rel).exists() and sha(V14 / rel) == h else changed).append(rel)
    v14files = sorted(str(p.relative_to(V14)) for p in V14.rglob('*') if p.is_file())
    assert sorted(files) == v14files, 'file list differs from v14_g17'
    assert all(c.endswith('object_poses.npz') for c in changed), changed
    assert not any(c.startswith('episode_000032') for c in changed)
    meshes_ok = all(r in ident for r in files if r.endswith('.obj'))
    ch_eps = sorted(int(c.split('/')[0][-6:]) for c in changed)
    slot_check = {}
    for ep in ch_eps:
        a = np.load(V14 / f'episode_{ep:06d}' / 'object_poses.npz'); b = np.load(B / f'episode_{ep:06d}' / 'object_poses.npz')
        assert sorted(a.files) == sorted(b.files) and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape for k in a.files)
        assert all(np.array_equal(a[k], b[k]) for k in a.files if k not in ('world_from_object', 'camera_from_object'))
        names = [str(x) for x in a['names']]; touched = set(ev[var][str(ep)]['objects'])
        for k, n in enumerate(names):
            if n not in touched: assert np.array_equal(a['world_from_object'][:, k], b['world_from_object'][:, k]) and np.array_equal(a['camera_from_object'][:, k], b['camera_from_object'][:, k]), (ep, n)
        slot_check[ep] = dict(changed_objects=sorted(touched), other_slots_and_keys_identical=True)
    objs = {str(ep): {n: dict(source=o.get('source'), move_vs_v14=o['move_vs_v14'], steps_v14=o['steps_v14'], steps_new=o['steps_new'])
                      for n, o in ev[var][str(ep)]['objects'].items()} for ep in ch_eps}
    pa, pb = pub[f'v14_{var}'], pub['v14_base']; pn = pub[f'v14_{var}n']
    cvs = None
    if var == 'B2':
        c = pub['v14_B2cv']; cvs = {k: dict(fold=f, base={x: pb[k][x] for x in KEYS}, cv={x: c[k][x] for x in KEYS}) for k, f in [('public', 'rule from held-out only'), ('heldout', 'rule from public only')]}
    else:
        cvs = dict(public=dict(fold='rule from held-out only (= shipped rule: rack -> pointmap_cotracker)', base={x: pb['public'][x] for x in KEYS}, cv={x: pa['public'][x] for x in KEYS}),
                   heldout=dict(fold='rule from public only selects nothing (2 public racks < n 3) -> base', base={x: pb['heldout'][x] for x in KEYS}, cv={x: pb['heldout'][x] for x in KEYS}))
    man = dict(variant=bname, base='selected20_final_v14_g17 (byte-identical copy; only object_poses.npz of changed episodes rewritten)',
               rule=RULE_TEXT[var], clock=CLOCK, rule_params=dict(B1=rules['B1']) if var == 'B1' else dict(B2_good_families=rules['B2_good']),
               rule_rederived_on_v14_clock=dict(B1=rules14['B1'], B2_good=rules14['B2_good'], note='differs from the shipped rule only in bundle_adjusted for mini_dust_pan (-) and water_pitcher (+); eval has bundle_adjusted only in eps 3/5/8/33/36 (no dust pan/pitcher), so the eval bundle would be identical'),
               ground_truth_used=False, eval_gt_read=False, scanned_gt_meshes_used=False, meshes_unchanged=meshes_ok, episode_32='v14_g17 unchanged (G12 spoon roll kept)',
               changed_episodes=ch_eps, changed_objects=objs, slot_check=slot_check,
               public_reference='g15_camrt (= the public counterpart of v14_g17; v14 differs from v12b_g15 only by the ep3 spoon, which has no public counterpart); re-scored 20/20 identical to the team scores',
               public_score_of_shipped_rule={k: {kk: pa[k][kk] for kk in KEYS} for k in ['all', 'public', 'heldout']},
               public_score_v14_reference={k: {kk: pb[k][kk] for kk in KEYS} for k in ['all', 'public', 'heldout']},
               public_per_clip_mppe_delta=pa['per_clip_delta'], public_worst_object_deltas=pa['worst_object_deltas'],
               retime_check=dict(gt_free_lag=dict(public=lag_text(lags['public']), eval=lag_text(lags['eval'])),
                                 public_score_without_family_retime={k: {kk: pn[k][kk] for kk in KEYS} for k in ['all', 'public', 'heldout']}),
               cross_validation=cvs, eval_qc_gt_free=qc[var], files_sha256=files, unchanged_files_byte_identical_to_v14=len(ident), changed_files=changed,
               scorer='tools/reconstruction_audit/score_bundle.py --metrics -> bscore (unchanged), run from our mirror B_work/canonical (exports v14_*)', work_dir=str(W))
    (B / 'manifest.json').write_text(json.dumps(man, indent=1))
    fmt = lambda a: f"{a['mp']}/{a['clips']} | {a['sp']}/{a['clips']} | {a['pos']:.3f} | {a['ori']:.2f} | {a['MPPE']:.4f} | {a['AUC']:.4f}"
    lt = man['retime_check']['gt_free_lag']
    lines = [f'# {bname} (Track 3 recon variant {var} on v14_g17, 2026-10-30)', '',
             'Base: `selected20_final_v14_g17`, same format. Byte-identical copy of v14_g17; only `object_poses.npz` of the changed episodes is rewritten '
             '(same keys and dtypes, compressed; untouched objects and all other keys are identical to v14). Meshes unchanged; ep32 is v14_g17. No GT, no eval GT, no scanned meshes.', '',
             '## Rule (identical to the v8-based variant B; code `B14_work/code/v14lib.py` + `B_work/code/variants.py`)', RULE_TEXT[var], '', '## Clock', CLOCK, '',
             f"GT-free check: without the retime, family tracks lag the v14 base by a median {lt['eval']['best_lag_naive_median']} frames on eval ({lt['public']['best_lag_naive_median']} public); "
             f"after it, {lt['eval']['best_lag_retimed_median']} ({lt['public']['best_lag_retimed_median']}), and the moving-frame gap to the base drops "
             f"{lt['eval']['moving_frame_gap_cm_naive']:.2f} -> {lt['eval']['moving_frame_gap_cm_retimed']:.2f} cm on eval ({lt['public']['moving_frame_gap_cm_naive']:.2f} -> {lt['public']['moving_frame_gap_cm_retimed']:.2f} public).", '',
             '## Public / held-out score of exactly this rule (unchanged scorer; reference g15_camrt)', '',
             '| set | MP-SR | SP-SR | pos cm | ori deg | MPPE | AUC |', '|---|---|---|---|---|---|---|']
    for k in ['all', 'public', 'heldout']:
        lines += [f'| v14 reference {k} | {fmt(pb[k])} |', f'| **{var} {k}** | {fmt(pa[k])} |', f'| {var} without family retime {k} | {fmt(pn[k])} |']
    lines += ['', '2-fold check: ' + '; '.join(f"{k} ({v['fold']}): MPPE {v['base']['MPPE']:.4f} -> {v['cv']['MPPE']:.4f}, AUC {v['base']['AUC']:.4f} -> {v['cv']['AUC']:.4f}, MP {v['cv']['mp']}/10" for k, v in cvs.items())]
    lines += ['', f'## Changed eval episodes: {ch_eps}', '', '| ep | object | source | mean/max move vs v14 cm | max deg | IoU median v14->new | paired dIoU | mean rot step deg v14->new | max rot step |', '|---|---|---|---|---|---|---|---|---|']
    for ep in ch_eps:
        for n, o in objs[str(ep)].items():
            q = qc[var].get(f'{ep}/{n}', {})
            lines.append(f"| {ep} | {n} | {o['source']} | {o['move_vs_v14']['mean_cm']:.2f}/{o['move_vs_v14']['max_cm']:.2f} | {o['move_vs_v14']['max_deg']:.1f} | "
                         f"{q.get('iou_median', [0, 0])[0]:.3f}->{q.get('iou_median', [0, 0])[1]:.3f} | {q.get('d_iou_paired', 0):+.4f} | {o['steps_v14']['mean_deg']:.2f}->{o['steps_new']['mean_deg']:.2f} | {o['steps_v14']['max_deg']:.1f}->{o['steps_new']['max_deg']:.1f} |")
    lines += ['', 'Not retargeted: the v29 rollouts target v14_g17; these bundles only matter if components are re-retargeted onto them.',
              'Details: `manifest.json` (sources, sha256 of every file, scores, QC, lag check). Work folder: `' + str(W) + '`.']
    (B / 'README.md').write_text('\n'.join(lines) + '\n')
    z = OUT / zname
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(B.rglob('*')):
            if p.is_file(): zf.write(p, f'{bname}/{p.relative_to(B)}')
    with zipfile.ZipFile(z) as zf: bad = zf.testzip(); nz = len(zf.namelist())
    print(var, 'changed eps', ch_eps, 'unchanged identical', len(ident), 'changed', len(changed), 'meshes ok', meshes_ok, 'zip', z, sha(z)[:12], os.path.getsize(z), 'entries', nz, 'testzip', bad)
