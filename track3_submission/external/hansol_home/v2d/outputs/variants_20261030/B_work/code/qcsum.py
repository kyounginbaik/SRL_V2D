"""Variant B: GT-free eval QC comparison (team qc.py outputs) v8_g10b vs B1/B2 on changed objects, plus step counts."""
import json, glob, os, numpy as np
W = '/home/hansol/v2d/outputs/variants_20261030/B_work'
ev = json.load(open(f'{W}/eval_variant_summary.json')); out = {}
for var in ['B1', 'B2']:
    out[var] = {}; print(f'== {var}: ep object | IoU med v8->new (paired mean dIoU) | p10 | low<0.5 | |z| med mm | z p90 | rot max step | >30deg steps | trans max step cm | move mean/max cm, max deg')
    for ep, e in sorted(ev[var].items(), key=lambda x: int(x[0])):
        a = json.load(open(f'{W}/qc/v8/episode_{int(ep):06d}/qc.json'))['objects']; b = json.load(open(f'{W}/qc/{var}/episode_{int(ep):06d}/qc.json'))['objects']
        for n, o in e['objects'].items():
            if o['move_vs_v8']['max_cm'] < 1e-6 and o['move_vs_v8']['max_deg'] < 0.01: continue
            sa, sb = a[n]['summary'], b[n]['summary']
            ia = np.array(a[n]['per_frame']['iou'], float); ib = np.array(b[n]['per_frame']['iou'], float); m = np.isfinite(ia) & np.isfinite(ib)
            za = np.abs(np.array(a[n]['per_frame']['median_z_mm'], float)); zb = np.abs(np.array(b[n]['per_frame']['median_z_mm'], float)); mz = np.isfinite(za) & np.isfinite(zb)
            r = dict(source=o.get('source'), iou_median=(sa['iou_median'], sb['iou_median']), iou_p10=(sa['iou_p10'], sb['iou_p10']), low_iou=(sa['low_iou_fraction'], sb['low_iou_fraction']),
                     d_iou_paired=float((ib[m] - ia[m]).mean()), absz_median=(float(np.median(za[mz])), float(np.median(zb[mz]))), z_p90=(sa['z_p90_mm'], sb['z_p90_mm']),
                     rot_max_step=(sa['max_rotation_step_deg'], sb['max_rotation_step_deg']), rot_over30=(sa['rotation_steps_over_30deg'], sb['rotation_steps_over_30deg']),
                     trans_max_step_cm=(o['steps_v8']['max_cm'], o['steps_new']['max_cm']), trans_over2cm=(o['steps_v8']['over_2cm'], o['steps_new']['over_2cm']),
                     mean_step=(o['steps_v8']['mean_cm'], o['steps_new']['mean_cm'], o['steps_v8']['mean_deg'], o['steps_new']['mean_deg']), move=o['move_vs_v8'])
            out[var][f'{ep}/{n}'] = r
            f = lambda t, p=3: f'{t[0]:.{p}f}->{t[1]:.{p}f}'
            print(f"{var} {int(ep):2d} {n:17s} | {f(r['iou_median'])} ({r['d_iou_paired']:+.4f}) | {f(r['iou_p10'])} | {f(r['low_iou'])} | {f(r['absz_median'],1)} | {f(r['z_p90'],1)} | {f(r['rot_max_step'],1)} | {r['rot_over30'][0]}->{r['rot_over30'][1]} | {f(r['trans_max_step_cm'],2)} | {o['move_vs_v8']['mean_cm']:.2f}/{o['move_vs_v8']['max_cm']:.2f} {o['move_vs_v8']['max_deg']:.1f}")
    v = list(out[var].values())
    print(f"{var} aggregate over {len(v)} objects: mean dIoU {np.mean([x['d_iou_paired'] for x in v]):+.4f}; IoU-median up {sum(x['iou_median'][1] > x['iou_median'][0] for x in v)}/{len(v)}; |z| median v8 {np.mean([x['absz_median'][0] for x in v]):.2f} -> {np.mean([x['absz_median'][1] for x in v]):.2f} mm; rot>30 steps {sum(x['rot_over30'][0] for x in v)} -> {sum(x['rot_over30'][1] for x in v)}; mean step cm {np.mean([x['mean_step'][0] for x in v]):.3f} -> {np.mean([x['mean_step'][1] for x in v]):.3f}, deg {np.mean([x['mean_step'][2] for x in v]):.2f} -> {np.mean([x['mean_step'][3] for x in v]):.2f}")
json.dump(out, open(f'{W}/eval_qc_summary.json', 'w'), indent=1)
