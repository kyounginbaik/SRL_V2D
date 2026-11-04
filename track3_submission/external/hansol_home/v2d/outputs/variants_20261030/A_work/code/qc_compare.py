"""GT-free QC comparison of a smoother variant vs its base, on the frames the variant changed.
usage: qc_compare.py eval TAG          (A_work/eval/TAG vs A_work/eval/base)
       qc_compare.py public TAG        (pub/*/exports/A_TAG vs r8_base)
Per object: changed frames (> 0.5 deg or 2 mm at the mesh centroid), mean IoU / |median depth residual| / ball px on
those frames before -> after, frames whose IoU drops by > 0.1, and steps > 30 / > 15 deg before -> after."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import slib
W = Path('/home/hansol/v2d/outputs/variants_20261030/A_work')
part, tag = sys.argv[1], sys.argv[2]
if part in ('eval', 'eval14'):
    pairs = [(W / part / 'base' / e.name, e) for e in sorted((W / part / tag).glob('episode_*'))]
else:
    pairs = [(e / 'exports' / 'r8_base', e / 'exports' / f'A_{tag}') for s in ['public', 'public_heldout'] for e in sorted((W / 'pub' / s).glob('episode_*'))]


def nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')


res = {}
tot = dict(chg=0, iou_b=[], iou_a=[], z_b=[], z_a=[], drop=0, ball_b=[], ball_a=[])
for b, a in pairs:
    if not (a / 'qc.json').exists() or not (b / 'qc.json').exists():
        print('missing qc', a); continue
    qb, qa = json.loads((b / 'qc.json').read_text()), json.loads((a / 'qc.json').read_text())
    db, da = np.load(b / 'object_poses.npz'), np.load(a / 'object_poses.npz')
    meshes = json.loads((a / 'meshes.json').read_text())
    names = [str(n) for n in da['names']]
    clip = a.name if part != 'public' else f'{a.parent.parent.parent.name}/{a.parent.parent.name}'
    res[clip] = {}
    for k, n in enumerate(names):
        info = slib.mesh_info(n, meshes[n])
        Tb, Ta = db['world_from_object'][:, k], da['world_from_object'][:, k]
        dr = slib.rel_angle_deg(Ta[:, :3, :3], Tb[:, :3, :3], info)
        dp = np.linalg.norm(slib.centers(Ta, info) - slib.centers(Tb, info), axis=-1) * 100
        ch = (dr > 0.5) | (dp > 0.2)
        pb, pa = qb['objects'][n]['per_frame'], qa['objects'][n]['per_frame']
        ib, ia = np.array(pb['iou'], float), np.array(pa['iou'], float)
        zb, za = np.abs(np.array(pb['median_z_mm'], float)), np.abs(np.array(pa['median_z_mm'], float))
        sb, sa = slib.summarize(slib.motion_stats(Tb, info)), slib.summarize(slib.motion_stats(Ta, info))
        r = dict(changed=int(ch.sum()), iou_before=nanmean(ib[ch]), iou_after=nanmean(ia[ch]), absz_before=nanmean(zb[ch]), absz_after=nanmean(za[ch]),
                 iou_drop_gt0p1=int(np.nansum((ia - ib)[ch] < -0.1)), iou_gain_gt0p1=int(np.nansum((ia - ib)[ch] > 0.1)),
                 iou_clip_median_before=nanmean([np.nanmedian(ib)]), iou_clip_median_after=nanmean([np.nanmedian(ia)]),
                 rot_gt30=[sb['rot_gt30'], sa['rot_gt30']], rot_gt15=[sb['rot_gt15'], sa['rot_gt15']], pos_gt3=[sb['pos_gt3'], sa['pos_gt3']],
                 racc_gt20=[sb['racc_gt20'], sa['racc_gt20']])
        tot['chg'] += r['changed']; tot['drop'] += r['iou_drop_gt0p1']
        for key, arr in [('iou_b', ib), ('iou_a', ia), ('z_b', zb), ('z_a', za)]:
            tot[key] += list(arr[ch])
        if n == 'wooden_spoon' and (a / 'ballqc.json').exists() and (b / 'ballqc.json').exists():
            bb = np.array([np.nan if x is None else x for x in json.loads((b / 'ballqc.json').read_text()).get('per_frame', [])], float)
            ba = np.array([np.nan if x is None else x for x in json.loads((a / 'ballqc.json').read_text()).get('per_frame', [])], float)
            if len(bb):
                r.update(ball_changed_before=nanmean(bb[ch]), ball_changed_after=nanmean(ba[ch]),
                         ball_clip_median=[float(np.nanmedian(bb)), float(np.nanmedian(ba))], ball_clip_p90=[float(np.nanpercentile(bb, 90)), float(np.nanpercentile(ba, 90))])
                tot['ball_b'] += list(bb[ch]); tot['ball_a'] += list(ba[ch])
        res[clip][n] = r
        bl = f" ball(chg) {r['ball_changed_before']:.1f}->{r['ball_changed_after']:.1f} clip med {r['ball_clip_median'][0]:.2f}->{r['ball_clip_median'][1]:.2f} p90 {r['ball_clip_p90'][0]:.1f}->{r['ball_clip_p90'][1]:.1f}" if 'ball_clip_median' in r else ''
        print(f"{clip[-14:]:14s} {n:18s} chg {r['changed']:3d} IoU {r['iou_before']:.3f}->{r['iou_after']:.3f} (drop>.1 {r['iou_drop_gt0p1']} gain>.1 {r['iou_gain_gt0p1']}) "
              f"|z| {r['absz_before']:.1f}->{r['absz_after']:.1f}mm steps>30 {r['rot_gt30']} >15 {r['rot_gt15']} pos>3 {r['pos_gt3']}{bl}")
summ = dict(changed=tot['chg'], iou_before=nanmean(tot['iou_b']), iou_after=nanmean(tot['iou_a']), absz_before=nanmean(tot['z_b']), absz_after=nanmean(tot['z_a']),
            iou_drop_gt0p1=tot['drop'], ball_before=nanmean(tot['ball_b']), ball_after=nanmean(tot['ball_a']))
print('TOTAL', {k: round(v, 3) for k, v in summ.items()})
(W / f'qc_{part}_{tag}.json').write_text(json.dumps(dict(summary=summ, objects=res), indent=1))
