# Stream G9 (phase 9): locate the gross official failure of v24 (GT-free) and build reconstruction fixes

Official v24: MP-SR 0.80, SP-SR 0.95 (one SP-failing episode X), AUC 0.855, MPPE 1.95. Proxy vs target_final_v6: 20/20, MPPE 0.69.
Rules followed: public scans are used only inside scorer-side targets (`scorer_only_scan_target_*`, never in a bundle or mesh);
our meshes only; no poses hand-edited in any package; nothing earlier overwritten (new bundles/targets only); no upload, no commit.

## Bottom line
1. **No mesh/registration (twin) fault is left in v24.** v24 scored against a scorer-only "scan-in-v6" target (public scan registered
   into our final_v6 body frame by the kit rule, trajectories = final_v6) passes 20/20 (worst ep25 piece_1 1.09 cm / 12.9 deg).
   The same target reproduces exactly the two known v22b twin failures (ep25 piece_2 176.7 deg, ep30 sweeper/rack 174 deg) and nothing else.
   Spoon: no public scan; our spoon meshes (5fcd334c in 8 episodes, e6c98973 in 15/16) register onto two independent spoon meshes
   (compose21/27) in ONE basin for all 10 spoon-episode kit seeds + 14 random seeds (max spread 1.8 deg; 10.5 deg for e6c9), and onto each other.
   So no mesh-only swap is needed (task 3: nothing staged).
2. Since v22b fails only ep25 + ep30 against that target but scored 3 official SP failures, **X is a reconstruction-vs-GT gross error that is
   present in both v22b and v24** (both follow final_v5/v6 poses), not a registration twin.
3. **Top suspect for X: ep32 spoon in-hand roll (f61-155).** Among all spoon episodes ep32 is the only one where our OWN 2D ball evidence
   prefers the coworker track on the disagreement frames: compose27 differs from final_v6 by > 90 deg on 34 % of frames (roll-type), and on
   21 of those frames compose27 fits the 2D mocap balls > 2x better with IoU no worse, 0 frames the other way (every other spoon episode is mixed or favours ours).
   Our silhouette collapses there (IoU 0.17-0.33 at f90-91, f152-154). Z9 had already ranked ep32 the riskiest spoon (flip-better frames f121-125, 149-150).
   If the truth is like compose27, v24 ep32 = spoon 74.8 deg + cup 8.6 deg -> mean 41.7 deg > 28.6 -> SP fail; if it is like our G9 fix, v24 ep32 = spoon 43.9 deg -> MP fail only.
4. Count calibration against the three official results (v21C 4/9 SP/MP failures, v22b 3/7, v24 1/4) with per-episode labels {ours, compose, intermediate}
   is **not decisive** (795 consistent assignments with compose21, none with compose27): the official counts need intermediate truths on several spoon
   episodes. It does rule out "compose is right everywhere" and "ours is right everywhere".

## Ranked suspects (GT-free)
| rank | ep | mechanism | evidence | est. official effect |
|---|---|---|---|---|
| 1 (X) | 32 | spoon roll flipped in hand, f61-96 / f108-139 / f142-155 | c27 disagreement 34 % of frames (all roll-type, long-axis tilt 4.5 deg); balls on usable frames: ours 26 px vs c27 ~11 px (40/45 frames c27 better); our IoU collapses f90-91, 152-154; Z9 flip-better f121-125, 149-150; c21 disagrees on 21 % | SP fail (if truth ~c27) or MP fail (if ~G9 fix: spoon 43.9 deg); fixing: +0.05 MP (+0.05 SP), MPPE about -0.1 |
| 2 | 36 | single-object spoon, roll windows f71-89, f251-317 | vs c21 51.8 deg (19 % >90), vs c27 41.3; balls favour ours (47 vs 21 frames) but IoU favours compose (55 vs 4) | single object: SP fails at 28.6 deg; half the windows wrong would be borderline SP / MP |
| 3 | 3 | spoon roll windows f50-63, f104-152, f167-184 | 26 % >90 vs both composes; balls tied (7 vs 7 strong frames, many in-hand misdetections), IoU favours ours | borderline: SP if compose right ((5+55)/2 = 30 deg), MP likely if intermediate |
| 4 | 35 | single-object spoon, f139-167 (c21) / f67-187 (c27) | c21 7 % >90, balls favour ours (3.4 vs 6.6 px), IoU compose; c27 has a 17 cm position disagreement (compose lost the track) | MP/SP only if compose21-like truth (33.5 deg) |
| 5 | 33 | spoon hand-over windows | balls favour ours vs c21 (12 vs 1 strong frames); c27 grossly off (26 cm) | low |
| MP-only 1 | 1 | white_pot_lid spin (order-2 axis) | our pipeline's public lid error is 21-29 deg (pub0/pub2, after lid_up), i.e. right at the 30-deg MP line; compose differs 26-44 deg; IoU mixed (c27 better 177 vs 24 frames, c21 worse), depth favours ours | MP fail likely (SP needs > 52 deg) |
| MP-only 2 | 20 | water_pitcher placement (+cup anchor) | constant 3.5-4.0 cm offset vs BOTH composes (they tracked the pitcher with the scan template); mask IoU favours compose (129 vs 20 frames) even with compose's camera offset; our pitcher IoU median only 0.49 | MP fail likely |
| MP-only 3 | 8 | beige cup mid-clip | 4-5 cm vs both composes in 20-80 % of the clip (mean 2.9-3.0); image evidence favours ours (IoU 138 vs 38 frames, depth 4.7 vs 9.1 mm) | possible |
| low | 19 | sweeper half-turn twin (scene flip, dust pan 39 cm) | compose disagrees, but compose's sweeper (be217ca0) was ICP-tracked without balls; our pose puts the side-A ball pair (2 tight clusters, 291 detections) at body +x where our pub31-validated mesh has it; twin would need them at -x | SP if wrong, but evidence strongly for ours |
| low | 30 | sweeper twin | fixed in v24 (A9); 171-0 ball votes | - |

Frame-0 anchor: R8/Z9 audits stand (only ep32 was an outlier, already fixed in v6); no other anchor can produce >= 28.6 deg.

## Fix built: spoon segment adoption rule (G9)
`code/g9_spoon_adopt.py` (one rule for all 10 spoon episodes): where final_v6 and compose27 disagree > 90 deg on a contiguous segment, adopt compose27 only if,
on the segment's frames with usable balls (min px < 40), its median 2D ball reprojection is < 0.6x ours over >= 4 frames and its median mask IoU is not lower by > 0.02.
Compose poses enter our camera through the kit registration of OUR spoon mesh onto the compose spoon mesh plus one constant camera-frame correction fitted on frames
where the two agree (< 20 deg; ep32: 3.8 deg / 37 mm). Grown to < 45 deg, blended over 3 frames. Meshes untouched.
**Fires only on ep32** (f61-96, f108-139, f142-155); rejected everywhere else (ep3 f104-128: balls 14 vs 35 px but IoU 0.67 vs 0.80; ep35/36 balls favour ours).

| bundle (new) | ep32 change | balls, usable frames (median px) | IoU mean / median / frames < 0.4 | depth \|z\| median | v24 scored vs it (ep32) | delta vs v6 (kit) |
|---|---|---|---|---|---|---|
| final_v6 | - | 26.2 | 0.702 / 0.727 / 5 | 6.7 mm | cup 0.53/2.9, spoon 0.85/5.2 (pass) | - |
| **selected20_final_v7_g9** (compose pose) | 96 frames | **11.2** (40 of 45 frames better) | 0.696 / 0.703 / **0** | 11.1 mm (worse) | cup 0.53/2.9, spoon 1.10/**43.9** -> MP fail | spoon 0.56 cm / 40.9 deg, 22.8 % frames > 90 |
| selected20_final_v7_g9rot (compose rotation about our centroid, our position) | 96 frames | **11.3** (30 of 45 better) | 0.702 / 0.727 / 1 | 8.7 mm (worse) | cup 0.53/2.9, spoon 0.85/**43.9** -> MP fail | spoon 0.00 cm / 40.9 deg, 22.8 % frames > 90 |

- Disagreement with compose27 drops from spoon 74.6 deg (v6) to 35.6 deg (v7_g9); against compose21 it was 46.8 deg.
- **Validation:** the rule is spoon-only and public 10 + held-out 10 contain no spoon, so it cannot fire there: the public/held-out exports are unchanged
  (r8_base: 20/20, pos 1.114, ori 5.23, MPPE 1.171, AUC 0.931). This validation is vacuous, as for Z9's spoon work.
- **Eval evidence is mixed:** ball reprojection improves strongly, IoU collapse frames vanish, but the stereo depth residual gets 2-4 mm worse
  (the spoon is in the fist; stereo on the spoon has the known 5-10 mm toward-camera bias). Per the stream rule ("evidence must get better, not worse")
  this is a **candidate, not an adopted fix**; main decides. If adopted: target_final_v7_g9 (or _g9rot) and **re-retarget ep32 only** (all other episodes byte-identical to v6).

## Files
- code/: `pkg_score.sbatch` (+ `pkg_jobs*.txt`; score any package vs any target), `build_scan_target.py` (scorer-only scan-in-recon targets),
  `perframe.py` (kit-faithful per-frame errors), `spoon_xreg.py`, `spoon_axis.py`, `spoon_window.py` (ours vs compose image evidence, any object),
  `g9_spoon_adopt.py` (the rule), `evid_compare.py`, `calib.py` (official-count calibration), `lid_public_diag.py` (scorer-side, unfinished).
- out/pkg/<pkg>_<target>/epN.json: v24/v22b/v21C vs v6, scan-in-v6, scan-in-c21, compose21, compose27, v7_g9; out/pf/: per-frame recon-vs-recon errors;
  out/spoon_window_*.json, out/obj_window_*.json, out/evid_ep32_*.json, out/spoon_xreg.json.
- scorer_only_scan_target_v6/, scorer_only_scan_target_c21/: contain public scans (registered); validation only, never to be used for a submission or reconstruction.
- Bundles: reconstruction_recovery_20261026/selected20_final_v7_g9, selected20_final_v7_g9rot (copies of final_v6; only episode_000032/object_poses.npz differs).
  Targets: campaign_v22_20261028/target_final_v7_g9, target_final_v7_g9rot.
