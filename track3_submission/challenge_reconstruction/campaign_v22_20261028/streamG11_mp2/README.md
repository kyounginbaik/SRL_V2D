# Stream G11 (phase 9): MP-only failures after v26 (official MP 0.80)

Rules followed: our meshes only, and no mesh was changed. Public GT and scans were read only through the kit scorer path (`pubdiag*.py`, `spinscore.py`,
`scalecheck.py`, `pitcher_bias.py` are scorer-side diagnostics). Compose tracks were used only as hypotheses, and their scan-template poses were not copied.
No submitted pose was edited. Nothing was overwritten. No upload and no commit. CPU jobs ran on node01 (debug) and each took under 50 min.

## 0. New official fact: v26 = MP 0.80, SP 1.00, MPPE 1.924, AUC 0.85479, RPE 1.742 (v25: 0.80 / 1.00 / 1.932 / 0.85558 / 1.733)
Only eps 3 and 32 changed between v25 and v26. G10's prediction (MP 0.90) is wrong.

### ep3 / ep32 hypotheses vs v24, v25 and v26 together (`out/hyp_ep3_ep32.txt`, `out/pkg_v2?_*.json`)
I scored the official packages (eps 3 and 32, kit scorers incl. RPE, `code/score_pkg_rpe.py`) against 5 candidate truths each and
compared the predicted sum of per-episode deltas (v26 minus v25) with the official deltas x 20: MPPE -0.16, AUC -0.016, RPE +0.18.

| ep3 truth | ep32 truth | dMPPE sum | dAUC sum | dRPE sum | MP count v24/25/26 for these 2 eps | verdict |
|---|---|---|---|---|---|---|
| **v8_g10b (flip real)** | **compose27-like** | **-0.13** | **-0.0125** | **+0.14** | 1/1/1 (constant, OK) | **best fit** (dist 0.57). v26 SP ep32 is marginal under c27 (cup 9.5 includes compose's cup error) |
| v6 (no flip) | v8_g10 (G10) | +0.36 | -0.027 | +0.36 | 1/1/1 | count-consistent but **MPPE has the wrong sign** (+0.018 official predicted vs -0.008 observed). Rejected (dist 4.0) |
| v8_g10b | v8_g10 | -0.55 | -0.011 | +0.36 | 1/1/2 | rejected: MP would rise to 0.85 |
| v8_g10b | v6 / v7_g9rot | -0.08 / +0.05 | -0.019 / -0.024 | +0.36 | count changes | rejected (also v24 SP fail unexplained) |
| c21/c27 | any | | | | ep3 SP fail in v24/v25 | rejected by official SP 1.00 in v25 |

Conclusion: **ep3 passes in v24, v25 and v26.** Its in-hand flip is real (H10_ep03_H2f0 improves MPPE). **ep32 fails MP in all three.** Its truth
is further from our tracks than v8_g10 and nearer compose27 (spoon error vs c27: v24 74.8, v25 44.0, v26 53.5). The 30-54 deg band of v25 and v26 pins it.
So v26 has **ep32 + 3 other MP failures, not ep3.**

**v27 recommendation:** ep3 = H10_ep03_H2f0 (keep v26). ep32 = **H9_ep32_h19f0** (the v25 component): it was officially SP-pass, it is closest to the
c27-like truth (44.0 vs 53.5 deg; c21 36.1 vs 35.7), and its MPPE is lower (2.85 vs 3.08 vs c27). RPE is 0.87 vs 1.20 on our recon.
Expected vs v26: MPPE about -0.01, RPE about -0.015, MP and SP unchanged. No existing ep32 component passes MP under a c27-like truth.

## 1. ep1: white_pot_lid spin (order 2). Exonerated by a validated image estimator
- The lid mesh is the same file (3590f16b) in public ep0, public ep2 and eval ep1. Our lid mesh has the bar handle (`spinscan.py` finds 900+ handle points).
- Scorer-side: the public lid error is a spin about the knob axis. `spinscore.py` (constant body spin delta) gives the optimum spin at **pub0 -17 deg
  (21.4 -> 10.4 deg)** and **pub2 -26 deg (28.7 -> 6.9 deg)**. Registration seed spread is <= 1.6 deg, so the spin is stable.
- GT-free estimator `spinscan.py`: stereo depth residual on the projected handle pixels (hd) per spin delta, pooled over about 115-125 frames.
  hd optimum: **pub0 -4.3, pub2 -13.1**, so the image-vs-scorer offset is **-13.0 and -12.9**, a constant mesh/scan handle offset.
  Leave-one-out with the hd + offset rule: pub2 predicted -26 (error 28.7 -> about 7), pub0 predicted -17 (21.4 -> about 10.5).
  Edge chamfer and silhouette costs are not consistent (edge offsets +25 / -11), so they were not used.
- **eval ep1: hd optimum +11.9 (bootstrap 10-15). The rule predicts a correction of -1 deg**, so our ep1 lid is already at the
  public-calibrated optimum. Expected hidden lid ori is about 7-11 deg plus the rollout's 4.9. **ep1 is unlikely to be an MP failure.** No bundle was built (the correction is -1 deg).

## 2. ep20: pitcher/cup placement
- The cup is slot 0, so it sets the alignment translation, and it is **in hand at f0** (frames 0-50 checked: it is held off the table, beyond the table edge).
  In public 21 and 22 the cup rests on the table at f0. The pitcher sets the alignment rotation.
- The difference vs compose27 is set at f0: pitcher (-1.7, -3.2, -1.25) cm in camera-0 coordinates, mostly **perpendicular** to both rays
  (along the cup ray only 1.25). Moving each object along its own ray cannot produce it, so in our camera either track
  is about 35 px off in 2D at f0. Our overlays (`visual/ov20.jpg`) fit the cup and pitcher silhouettes at f0/20/60/100/150/200. Compose27 centroids
  projected with our camera are 15-25 px off. Conclusion: the compose discrepancy is a compose camera/track artefact, not evidence against ours.
- Public pitcher body-frame bias (scorer-side, same mesh 88458e8f): pub21 [0.60, -0.28, 0.24] cm and held22 [0.99, 0.09, 0.61] cm (GT body).
  Leave-one-out with the other clip's bias: pub21 1.49 -> 1.39, held22 1.64 -> 1.32 cm (`pitcher_bias.py`, see section 5). This is too small to turn an MP failure into a pass.
- Scorer-side finding, not used: our pitcher mesh is about 6 % larger than the scan (kit scale 0.940; cups 0.99, pot 1.00, lid 1.03).
  Silhouette fits of an oversized mesh in hand place it about 6 % too far (1-2 cm at 30 cm). A GT-free size estimate would be needed before acting on it.

## 3. ep8: beige cup mid-clip
- The compose27 difference is mostly **along the viewing ray** in the in-hand frames (2-4 cm, f50-160). Scorer-side scale check: our beige cup mesh is
  correctly sized (scan/ours 0.992), while compose's beige template is 11 % smaller (G10 meshreg scale 0.892). A smaller mesh
  fitted to the same silhouette sits closer to the camera, which explains the compose offset. Stereo depth favours ours (G9: 5.2 vs 10.2 mm). Overlays fit (`visual/ov8.jpg`).
  **ep8 is unlikely to fail on the cup.**

## 4. Where the 3 non-ep32 MP failures could be (open)
v22b official: SP 3 (25, 30, 32) and MP 7, so 4 MP-only failures. v22b vs our recon fails MP only on 30 and 32. The 3 persistent v24-v26 failures
therefore need recon-vs-GT errors >= about 3 cm or 30 deg that our own evidence does not show, and ep3 and ep1 are now unlikely.
The remaining candidates are in-hand phases where only the silhouette constrains depth: pitcher (ep20, oversized mesh), cups in hand (10, 14, 33), the wooden pieces (25: piece_1 12.5 deg vs recon), and sweeper ep19.
A per-episode official calibration (as G9 calib.py, with the v26 result added) is the next step.

## 5. Results files
- out/: pd_*_r8base.jsonl (public error structure), spin_*.npz/log (GT-free spin scans), spinscore_*.log, pkg_v2?_*.json/log, hyp_ep3_ep32.txt,
  rvr_ep20_c27.txt, rvrcam_ep8_c27.txt, eb_*.npy, pitcher_bias.json/log.
- code/: pubdiag.py, pubdiag2.py, spinscan.py, spinscore.py, score_pkg_rpe.py, rvr.py, rvr_cam.py, overlay.py, scalecheck.py, pitcher_bias.py, cmd.sbatch, py.sh.

## 6. Bundle built: selected20_final_v9_g11 (optional, small MPPE gain)
- `reconstruction_recovery_20261026/selected20_final_v9_g11` = v8_g10b with one change: the ep20 water_pitcher track gets a constant **body-frame
  translation t = (-0.79, +0.09, -0.43) cm** (`W @ T(t)`; camera_from_object recomputed). This is the mean public pitcher bias, mapped
  into our body frame by the kit registration rotation (Rb is the same in 21 and 22 to within 0.3 deg). No scan geometry enters the mesh or the poses. Meshes are byte-identical
  (`diff -rq`: only ep20 object_poses.npz differs, plus G11_CHANGES.json). Target: `campaign_v22_20261028/target_final_v9_g11`.
- Public validation (bscore, GT only in the scorer). The leave-one-out (loo) bias is learned on the other clip:

| clip | pitcher cm base / loo / both | MPPE base / loo / both | AUC base / loo / both |
|---|---|---|---|
| pub21 | 1.49 / 1.39 / 1.26 | 1.191 / 1.143 / 1.078 | 0.9373 / 0.9387 / 0.9434 |
| held22 | 1.64 / 1.32 / 1.16 | 1.424 / 1.265 / 1.187 | 0.9269 / 0.9367 / 0.9464 |

  No other public or held-out clip has a pitcher, so they are unchanged. GT-free eval evidence: a constant 0.9 cm shift is below silhouette resolution, so it is neutral.
  The rule does not fix MP (it is about 0.3 cm). It is an MPPE/AUC hedge for ep20 only.
- Re-retarget: ep20 F9_ep20_b4 vs target_final_v9_g11: cup 0.25/1.4, **pitcher 1.06/1.7**, MPPE 0.678, AUC 0.981, MP 1
  (vs v8_g10b: pitcher 0.39/1.7, MPPE 0.347, AUC 0.996). Optional; low priority given only one upload.
