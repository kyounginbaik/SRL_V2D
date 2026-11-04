"""Add the evidence summary to C_ep20/manifest.json, then zip the bundle as recon_C_ep20_selected20_final_v8_g10b/ and verify the zip.
usage: finish_C.py VARIANTS_DIR"""
import sys, os, json, hashlib, zipfile
V = sys.argv[1]; OUT = f'{V}/C_ep20'; W = f'{V}/CD_work'
TOP = 'recon_C_ep20_selected20_final_v8_g10b'; ZIP = f'{V}/{TOP}.zip'
sha = lambda b: hashlib.sha256(b).hexdigest()
m = json.load(open(f'{OUT}/manifest.json'))
m['evidence_summary'] = dict(
    hypothesis=('eval20 RGB frame drop at f53: ego_cam_a lags the stereo/GT clock by 0.925 frame on f0-52 and 1.85 frame from f53, so the '
                'v8_g10b ep20 poses (RGB clock) trail the GT clock by 1.85 frames during the fast pitcher pour. Unique among the 20 eval clips.'),
    camera_flow_lag=('G15 camsync_fine (optical-flow xcorr ego_cam_a vs b/c, 24-frame windows): -0.85..-1.0 up to f44, -1.8..-1.95 from f56 '
                     'to the end in all 4 pair/axis series, r >= 0.99. ep33: constant ~0.875.'),
    eval20_depth_v8_vs_C_moving_frames_median_abs_dz_mm={'pitcher f0-52': [7.82, 5.37], 'pitcher f53-201': [6.93, 5.62],
                                                         'cup f0-52': [3.88, 3.85], 'cup f53-201': [4.80, 3.42]},
    eval20_depth_shift_scan_best_s={'cup f0-52': 0.5, 'cup f53-201': 1.4, 'pitcher f0-52': 0.9, 'pitcher f53-201': '1.4 (flat 0.5-1.85)'},
    public_validation_unchanged_scorer=dict(rule='G15 g15_camrt (retime + clock-consistent camera)', n_clips=20, mp_pass='20/20 -> 20/20',
                                            mppe='1.1714 -> 1.0641', auc='0.9312 -> 0.9437',
                                            pub21_pitcher_cm='1.49 -> 1.11', held22_pitcher_cm='1.64 -> 1.43', worst_clip_mppe_loss='held27 +0.034',
                                            rescored_myself='pub21 r8_base/g15_camrt and held22 r8_base/g15_camrt identical to stored public_score.json'),
    kit_style_change_vs_v8='cup 0.79 cm / 0.8 deg tilt, pitcher 1.22 cm / 2.9 deg (max 8.6 cm), largest of the 20 eval episodes',
    rejected_hint=('G12 "eval20 pitcher 2.3 % deeper than stereo" was measured on v9_g11 (contains G11 pitcher shift); on v8_g10b the same '
                   'estimator gives r = 0.9998, dz -0.07 mm, as public 21/22 (1.000/0.996)'),
    not_claimed='expected pitcher change about 1.2 cm; ep20 flips MP only if the current rollout is within about 1 cm of 3 cm; a flat upload does not prove ep20 passes',
    retarget_note='team H13a already retargeted ep20 onto the same ep20 recon (v12b): H13_ep20_b2, MPPE 0.448 vs v12b',
)
m['zip'] = dict(path=ZIP, top_folder=TOP, layout='same as recon_v26_selected20_final_v8_g10b.zip minus videos_sharpa/ (v26 renders)')
json.dump(m, open(f'{OUT}/manifest.json', 'w'), indent=1)

if os.path.exists(ZIP): raise SystemExit(f'{ZIP} exists; refusing to overwrite')
files = sorted(os.path.relpath(os.path.join(d, f), OUT) for d, _, fs in os.walk(OUT) for f in fs)
dirs = sorted({os.path.dirname(f) for f in files if os.path.dirname(f)} | {os.path.dirname(os.path.dirname(f)) for f in files if os.path.dirname(os.path.dirname(f))})
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr(TOP + '/', b'')
    for d in dirs: z.writestr(f'{TOP}/{d}/', b'')
    for f in files: z.write(f'{OUT}/{f}', f'{TOP}/{f}')
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None
    bad = [f for f in files if sha(z.read(f'{TOP}/{f}')) != sha(open(f'{OUT}/{f}', 'rb').read())]
    assert not bad, bad
    n = len([i for i in z.infolist() if not i.filename.endswith('/')])
zs = sha(open(ZIP, 'rb').read())
open(ZIP + '.sha256', 'w').write(f'{zs}  {os.path.basename(ZIP)}\n')
print('zip', ZIP, 'files', n, 'size', os.path.getsize(ZIP), 'sha256', zs)
