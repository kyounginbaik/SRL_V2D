"""Assemble verdicts.json: hand verdicts (README) + the window tables from summary.json / rdepth_summary.json."""
import json
from pathlib import Path
B = Path(__file__).resolve().parents[1]; S = json.load(open(B / 'results/summary.json')); RD = json.load(open(B / 'results/rdepth_summary.json'))
V = {
 1: {'white_pot': ('OURS', 'low', 'SAM2 mask for white_pot is byte-identical to the lid mask (eval_hawor outputs ep1), so masked depth/IoU are invalid for the pot; mask-free rendered-depth agreement ours 0.48 vs new 0.38 (new worse f240-333: 0.28-0.29 vs 0.37-0.44). Tracks differ 2.0 cm / 4 deg median (max 2.9 cm / 12 deg).'),
     'white_pot_lid': ('MIXED', 'medium', 'OURS f40-279 (new 12-18 mm off-tolerant depth vs ours 4-6; mask-free agree 0.56-0.71 vs 0.19-0.46); f0-39 and f280-333 conflicting (new better depth 2.3-2.9 vs 4.2-5.3 mm and mask-free agree 0.57-0.69 vs 0.47-0.59; ours better IoU 0.80-0.90 vs 0.67-0.68) -> keep ours. Tracks differ 3.2 cm / 19 deg (half-turn symmetric) almost everywhere.')},
 8: {'beige_cup': ('OURS', 'high', 'Every window: depth 2.9-4.3 vs 6.6-8.0 mm, IoU 0.56-0.82 vs 0.50-0.74, mask-free agree 0.53 vs 0.41. The 3.3 cm v20-vs-new gap is the NEW track being off: track difference 2.3 cm median (max 4.1), new also needs a +0.54 cm vertical world offset.'),
     'white_pot': ('UNDECIDABLE', 'low', 'Tracks within 1.1 cm / 6 deg; raw depth 4.5/5.3 mm, offset-tolerant 4.1/2.6 (new), IoU 0.74/0.75, mask-free agree 0.55/0.53. Equivalent; keep ours.')},
 10: {'blue_cup': ('OURS', 'medium', 'Tracks differ 1.8 cm / 3 deg; IoU ours 0.42 vs 0.32 every window, offset-tolerant depth 4.8 vs 5.6; mask-free equal (0.32/0.31). Raw depth favours new 9.8/7.6 (our world offset 0.8 cm absorbs it).'),
      'beige_cup': ('OURS', 'medium', 'f0-39 clearly ours (depth 2.7 vs 7.7 mm off-tol, IoU 0.81 vs 0.61, tracks 3.4 cm apart); later windows equal or ours. Overall 1.7 cm / 4 deg apart.')},
 14: {'blue_cup': ('OURS', 'high', 'Depth 6.2 vs 9.6 mm, IoU 0.78 vs 0.60 in every window; mask-free agree 0.47 vs 0.33 (except f120-156). Tracks 2.3 cm apart.'),
      'plastic_dish_rack': ('OURS', 'high', 'Static rack, each with its own 6-prong mesh: constant 2.5 cm offset between tracks; depth 3.2 vs 5.9 mm, IoU 0.78 vs 0.66, mask-free agree 0.55 vs 0.32 in every window. The 2.9 cm v20-vs-new gap is the NEW rack being off.')},
 19: {'mini_sweeper': ('OURS', 'high', 'Marker-ball face cue (scan balls via the exact kit registration): ours 0.28 vs new 0.11 (twins 0.01 / 0.05), ours wins every 2 s window; marker sides 166-178 deg apart for the whole clip (= the v14+ relabel, which the video supports). Also depth 5.2 vs 8.7 mm, IoU 0.58 vs 0.38, mask-free agree 0.50 vs 0.12.'),
      'mini_dust_pan': ('MIXED', 'medium', 'NEW f0-119 and f160-234 (depth 4-7 vs 10-13 mm, IoU 0.65-0.73 vs 0.50-0.58; mask-free f160+: 0.61 vs 0.21-0.27); f120-159 (the low-passed fast swing) ours equal/better (mask-free 0.49 vs 0.34). Tracks are only 1.5 cm / 6 deg apart (max 9.7 cm in f140-164). The 40 cm / 177 deg v20-vs-new gap is NOT a track gap: it is the scorer anchor alignment on the frame-0 sweeper face (reproduced: 179.4 deg anchor rotation, 20.4 cm sweeper-dustpan separation -> 40.0 cm / 176.4 deg).')},
 20: {'blue_cup': ('OURS', 'high', 'Depth 2.2 vs 7.3 mm, IoU 0.78 vs 0.67, mask-free agree 0.60 vs 0.31, every window. Tracks 1.5 cm apart.'),
      'water_pitcher': ('MIXED', 'low', 'NEW f0-29 (IoU 0.84-0.90 vs 0.70-0.77, depth similar) and f114-200 (depth 5.5-6.9 vs 7-8.7 mm; IoU mixed); OURS f30-113 pour (new depth 8-25 mm vs 5-13, mask-free agree 0.28-0.29 vs 0.34-0.45). Different meshes (new = kit public scan, ours = reconstruction); tracks 3.0 cm / 3 deg apart (max 11 cm / 31 deg around f96-108).')},
 25: {'wooden_piece_1': ('OURS', 'high', 'Slot side agrees (both tracks 1-7 deg; video: opens right when lying, up when standing). Position: f0-119 roughly equal (new depth 2.4-4.5 vs 3.5-7.3 mm raw, ours IoU 0.68-0.87 vs 0.59-0.80); f120-355 ours far better (depth 1.4-2.1 vs 11-29 mm, IoU 0.44-0.74 vs 0.19-0.58, mask-free agree 0.65-0.88 vs 0.05-0.14: the new standing board 1 is 2-3.6 cm off, often over background).'),
      'wooden_piece_2': ('MIXED', 'high', 'SLOT SIDE: NEW is right for the whole clip; OUR reference has board 2 rotated 180 deg about its normal relative to its slot-cut collision (our collision slot vs the video 178 deg, f0 and f160 images show the notch opening LEFT where NEW puts it). Our visual mesh has no slot, and the exact kit registration to the scan still maps ours onto the correct slot side (1.5 deg), so the scored rotation is not wrong - but the simulated board 2 has its slot on the wrong half (likely why board 2 never seats). Pose: NEW f0-199 (depth 1.2-6.3 vs 10-16 mm, mask-free agree 0.75-0.99 vs 0.02-0.27; ours ~1 cm too low); OURS/equal f200-355 (mask-free 0.73-0.84 vs 0.31-0.75). Tracks 1.5 cm / 2 deg apart (max 4.4 cm).')},
 28: {'plastic_dish_rack': ('OURS', 'medium', 'Static, 1.2 cm / 1 deg apart; depth 3.1 vs 4.6 mm, IoU 0.86 vs 0.80, mask-free agree 0.60 vs 0.47.'),
      'white_pot': ('OURS', 'low', 'Tracks within 0.9 cm / 2 deg; masked evidence equal (6.1/6.1 mm, IoU 0.71/0.69; no mask after f~150); mask-free agree 0.60 vs 0.47. Effectively equivalent.')},
 30: {'mini_sweeper': ('OURS', 'high', 'Both tracks share the face during the rack rest f0-51 (marker sides 1 deg apart); NEW flips to the other face at the lift (f52) and stays flipped (marker sides 169 deg apart, = the 140 deg v20-vs-new gap). Marker cue f60-209: ours 0.62 vs new 0.35 (our twin 0.10, new twin 0.39 = new is on the wrong face); depth 2.1-7.3 vs 3.1-10.7 mm, IoU 0.41-0.63 vs 0.24-0.49 (6 of 7 windows ours). Rest pose f0-39: ours IoU 0.63 vs 0.24 - the new "standing in the rack" re-fit is not supported. End f240-262: both mediocre (4.2 cm apart), ours slightly better depth/IoU, mask-free favours new (0.43 vs 0.19) -> undecided.'),
      'plastic_dish_rack': ('UNDECIDABLE', 'low', 'Static, 1.3 cm / 2 deg apart; identical evidence (depth 4.4/4.3 mm, IoU 0.85/0.82, mask-free 0.43/0.46). Keep ours.')},
 38: {'mini_dust_pan': ('OURS', 'medium', 'Depth 2.2 vs 4.7 mm, IoU 0.83 vs 0.78, mask-free agree 0.76 vs 0.60; ours wins f0-159, equal after. Tracks 1.2 cm / 4 deg apart (max 3.4 cm / 26 deg at f44-50).')},
}
EPV = {1: 'OURS', 8: 'OURS', 10: 'OURS', 14: 'OURS', 19: 'MIXED', 20: 'MIXED', 25: 'MIXED', 28: 'OURS', 30: 'OURS', 38: 'OURS'}
out = {}
for ep, objs in V.items():
    out[str(ep)] = {'episode_verdict': EPV[ep]}
    for k, (v, c, note) in objs.items():
        so = S[str(ep)]['objects'][k]; rd = RD[str(ep)][k]
        wins = []
        for w in so['windows']:
            rw = next((x for x in rd['windows'] if x['frames'][0] == w['frames'][0]), None)
            wins.append(dict(frames=w['frames'], depth_mm_ours_new=w['depth_mm'], depth_offset_tolerant_mm_ours_new=w['depth_off_mm'], iou_ours_new=w['iou'],
                             maskfree_agree_ours_new=rw['agree'] if rw else None, track_diff_cm=w['d_cen_cm'], track_diff_deg=w['d_rot_deg'], auto=w['auto']))
        out[str(ep)][k] = dict(verdict=v, confidence=c, notes=note, differ_frames_gt2cm_or_20deg=so['differ_frames'], overall=so['overall'], maskfree_overall=rd['overall'], windows=wins)
out['25']['wooden_piece_2']['slot_check'] = json.load(open(B / 'results/scanslot25.json'))['wooden_piece_2']
out['19']['mini_dust_pan']['anchor_artefact'] = json.load(open(B / 'results/ep19_anchor.json'))
json.dump(out, open(B / 'verdicts.json', 'w'), indent=1); print('ok')
