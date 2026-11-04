"""Write verdicts.json: final per-object verdicts (judgement recorded here) + the automatic 1-s window table from summary.json."""
import json
S = json.load(open('evidence/summary.json')); C = json.load(open('markers/canon.json'))['eps']
import numpy as np
def mk(ep):
    z = C[str(ep)]; o, n = np.array(z['ours']), np.array(z['new'])
    return dict(canon_marker_median_px=[round(float(np.median(o)), 1), round(float(np.median(n)), 1)], canon_marker_inlier10=[round(float(np.mean(o < 10)), 2), round(float(np.mean(n < 10)), 2)], n_detections=len(o))
V = {
 3: {'wooden_spoon': ('OURS', 'high', 'No 180-deg disagreement in the camera frame (the coworker\'s roll fix corrected THEIR earlier track; ours already had that roll). Markers: ours explains the balls at rest (start inl15 0.97-1.00 vs 0.62-0.72; end f260-312 0.74-1.00 vs 0.00). NEW is 1-6 cm off (along the handle/view ray) f0-180; IoU 0.80 vs 0.71.'),
     'white_pot': ('UNDECIDABLE', 'low', 'Tracks differ 1.1 cm / 6.7 deg; different pot meshes (ours 12.5x18.8x34.1 cm, new 19.7x10.1x33.9); IoU 0.61/0.58, free-space 0.26/0.21; no markers used. Keep OURS (no evidence to switch).')},
 5: {'wooden_spoon': ('OURS', 'low', 'Differences 2-4.7 cm are mostly along the camera ray (calibration-chain depth scale, see README 4). Markers overall equal (inl10 0.44/0.45, inl20 0.75/0.72); IoU 0.78/0.73 for ours; NEW better only f0-19 and f140-159 (markers). No roll disagreement.'),
     'white_pot': ('UNDECIDABLE', 'low', 'Different meshes; free-space 0.32/0.22 favours NEW mildly, IoU equal (0.57/0.55); 1.3 cm / 7.8 deg. Keep OURS unless h2h_other pot evidence says otherwise.')},
 6: {'wooden_spoon': ('UNDECIDABLE', 'low', 'Markers mildly favour NEW (canonical inl10 0.63 vs 0.52, median 7.6 vs 9.8 px), IoU favours ours (0.81/0.77); windows alternate. Difference 2.6 cm / 11 deg, mostly along the view ray. Keep OURS (ep6 already passes).'),
     'white_pot': ('UNDECIDABLE', 'low', 'As ep5: free-space 0.31/0.18 for NEW, IoU equal; different meshes.')},
 15: {'wooden_spoon': ('NEW', 'high', 'OUR ep15 reference is the coworker\'s earlier delivery. It is 2-4 cm off along the handle: canonical-ball reprojection 76.7 px vs 10.2 px (inl10 0.00 vs 0.50), free-fit balls land 3 cm down the handle from the cross-episode geometry, its size-consistent depth is 9.5% closer (IoU-optimal scale 1.095 vs 1.010), IoU 0.60 vs 0.73.'),
      'plastic_dish_rack': ('TIE', 'high', 'Tracks within 0.4 cm / 0.1 deg (static). Mesh change (6 prongs) is separate.')},
 16: {'wooden_spoon': ('NEW', 'high', 'Same as ep15: canonical-ball 44.2 vs 9.0 px (inl10 0.01 vs 0.55), IoU 0.62 vs 0.75, depth scale 1.100 vs 1.015.'),
      'plastic_dish_rack': ('TIE', 'high', 'Within 0.5 cm / 0.1 deg.')},
 32: {'wooden_spoon': ('MIXED', 'medium', 'OURS outside f89-176. Inside the roll-ambiguous swing f89-176 (tracks 107-175 deg apart) the balls follow NEW at f90-99, f115-129, f150-172 (also visual: figures/zoom32_roll*.jpg) and OURS at f140-149 (there NEW is the flipped twin) and from f176 on. End rest f240-258 markers 1.00 vs 0.65 for ours.'),
      'blue_cup': ('OURS', 'high', 'NEW cup occupies free space (fs 0.32 vs 0.10) and IoU 0.62 vs 0.77 in 12 of 13 windows; offset ~2 cm visible in overlays.')},
 33: {'wooden_spoon': ('MIXED', 'medium', 'OURS except f140-158 (NEW). Rest windows strongly ours (markers 0.78-0.94 vs 0.12-0.27 at f0-59; 1.00 vs 0.72 at f320-339). At f145-154 the balls match OUR 180-deg twin and NEW (10-36 px vs 119-125 px), confirmed visually (figures/zoom33_roll.jpg). This contradicts the prior "ep33 roll matches the balls" only for this window; the prior check covered 2-6 s.'),
      'blue_cup': ('OURS', 'high', 'fs 0.12 vs 0.27-0.48, IoU 0.74 vs 0.60 (0.37-0.57 during the scoops); overlays show the NEW cup offset upward by several cm (figures/cup33.jpg).')},
 35: {'wooden_spoon': ('MIXED', 'low', 'Short alternating windows: NEW at f60-74, f135-149, f280-289, f305-319 (markers); OURS at f0-19, f75-89, f180-199, f270-279, f340-399. Overall canonical inl10 0.37 vs 0.34, inl20 0.74 vs 0.66, IoU 0.79/0.77. No sustained window -> keep OURS.')},
 36: {'wooden_spoon': ('OURS', 'medium-high', 'Both 180-deg disagreements go to ours: f72-89 (NEW\'s edge-on roll re-estimate) - balls 13-21 px ours vs 33-36 px NEW with NEW\'s twin at 8-11 px, and the edge-on bowl profile matches ours (figures/zoom36_bowl_profile.jpg); f250-294 (our marker-based correction) - ours 7-11 px vs NEW 62-77 px, NEW twin 13-26 px. NEW slightly better only f90-110 (6-16 vs 21-26 px).')},
 37: {'wooden_spoon': ('NEW', 'medium', 'Markers favour NEW in every window (canonical median 5.8 vs 13.1 px, inl10 0.75 vs 0.39); IoU equal (0.81/0.79); difference 1.8 cm / 11 deg, no roll disagreement. Note ep37 already passes on the legacy reference.')},
}
out = {}
for ep, objs in V.items():
    out[ep] = {}
    for o, (v, c, n) in objs.items():
        out[ep][o] = dict(verdict=v, confidence=c, notes=n, windows=S[str(ep)][o])
        if o == 'wooden_spoon': out[ep][o]['markers_ours_new'] = mk(ep)
json.dump(out, open('verdicts.json', 'w'), indent=1)
print('ok')
