# Stream G10 (phase 9): ep32 truth from the two official results + MP-only failures

Rules followed: our meshes only (no mesh changed at all); compose tracks used only as hypotheses/calibration (spoon);
public GT/scans not read (compose27 meshes used only scorer-side in `meshreg.py`/`errframe.py` diagnostics); no pose hand-edited
in any package (only the RECONSTRUCTION spoon track changes); nothing overwritten (new bundles/targets); no upload, no commit;
CPU on node01 (debug), all jobs < 50 min.

## Bottom line
1. **ep32: new target `target_final_v8_g10`** (bundle `reconstruction_recovery_20261026/selected20_final_v8_g10` = final_v6 + ep32 spoon roll only).
   The spoon is rolled about its long axis by a **GT-free 2D-ball Viterbi track** (`code/rollscan.py` + `code/balltrack.py`, lambda 0.05, px<20, no gap prior):
   ramp f49-65 (45->135 deg), held flipped ~135-165 deg f65-151 (dip to ~90 deg f90-101 where the balls are uninformative), back to ours at f152.
   Scored with the official scorer path (`code/score_any.sh`): **v24's ep32 component -> SP 0 / MP 0 (spoon 56.6 deg, mean ori 29.7 > 28.6);
   v25's ep32 component -> SP 1 / MP 0 (spoon 31.9 deg)** = exactly the two official outcomes. final_v6, v7_g9rot and compose21 each contradict one of them.
2. **ep3: same failure mode found** (in-hand roll flip that the balls prefer): bundle `selected20_final_v8_g10b` / `target_final_v8_g10b` = v8_g10 + ep3 spoon
   flip f103-159 (~-135 deg) and f50-56 (~105 deg). Balls on usable frames 20.4 -> 9.7 px (24 frames better vs 4 worse), IoU 0.759 -> 0.769, both compose tracks
   also flip there (c21 f104-185, c27 f99-158). D9_ep03_b3 (in v24 and v25) vs it: spoon 28.8 deg, MP pass only by 1.2 deg -> with hidden noise ep3 is plausibly one of the 3 unexplained MP failures. Consistent with SP 1.00 either way.
3. No other spoon episode shows the mode (5, 6, 15, 16, 33, 35, 36, 37: no ball-supported flip). **ep35/36/37 cannot be MP-only failures by orientation at all**
   (single object: MP fail needs ori >= 30 but SP pass needs ori <= 28.6) -> only a >= 3 cm spoon position error could, which nothing indicates.
4. MP-only ranking for the remaining slots (official: ep32 + 3 others fail MP): **ep3 (above) > ep20 (pitcher/cup world offset 3.2 cm vs both composes) > ep1 (lid spin ~25 deg, public prior 21-29) > ep8**. No reconstruction fix built for 1/8/20 (evidence mixed; see below).

## 1. ep32 official-constraint calibration (`code/base32.py`, `code/cands.py`, `code/balltrack.py`; out/base32.json, cands_ep32_*.json, balltrack_ep32_*.json)
Official: v24 ep32 = SP fail (so MP fail); v25 ep32 = SP pass, MP fail. Kit SP uses the MEAN over objects (cup + spoon) of time-mean ori <= 28.6 deg.
Kit-faithful proxy (registration with kit seed, spoon frame-0 anchor alignment, our meshes as reference):

| candidate truth for ep32 | v24 cup / spoon deg (mean) | v24 SP/MP | v25 cup / spoon deg (mean) | v25 SP/MP | consistent? |
|---|---|---|---|---|---|
| final_v6 | 2.9 / 5.2 (4.0) | 1/1 | 2.4 / 40.7 (21.5) | 1/0 | no (v24 passes) |
| v7_g9rot (= v25's target) | 2.9 / 43.9 (23.4) | 1/0 | 2.4 / 12.3 (7.3) | 1/1 | no (both) |
| compose21 | 7.2 / 45.9 (26.5) | 1/0 | 7.4 / 36.1 | 1/0 | no (v24 SP pass) |
| compose27 (own world) | 8.6 / 74.8 (41.7) | 0/0 | 9.0 / 44.0 (26.5) | 1/0 | yes |
| c27 spoon rotation adopted on segment subsets (32 subsets) | | | | | only subsets containing 61-96 + 108-139 + **188-248** (balls reject 188-248: ours 5 px vs c27 12 px) |
| ball track, gap prior 0.003 / 0.01 (flip only where balls insist: 116-151) | 2.9 / 25.8 ; 17.9 | 1/1 | 2.4 / 30.1 ; 32.2 | 1/0 | no (v24 passes) |
| **ball track l0.05 (v8_g10)** | **2.9 / 56.6 (29.7)** | **0/0** | **2.4 / 31.9 (17.1)** | **1/0** | **yes** (official scorer path too) |
| ball track l0.035 / l0.07 / px15 / px30 | 55.6-57.4 (29.2-30.2) | 0/0 | 29.7-33.1 | 1/0 (l0.035: 1/1 at 29.7) | yes (marginal) |
| ball track l0.1 (flip only 94-151) | 37.5 | 1/0 | 34.1 | 1/0 | no |

So the official pair selects, among ball-consistent tracks, the one that **holds the flip through the ball-poor gaps (f60-151)**, not the one that
flips only where the balls insist, and not v7_g9rot's fast reversals (500 deg/s, which H9 found the hand cannot follow anyway).

Hidden-noise robustness (`code/noise_mc.py`, 30 draws, smooth body-frame noise on both objects; P(v24 SP fail & v25 SP pass & v25 MP fail)):

| noise deg / cm | final_v6 | v7_g9rot | v8_g10 | v8 l0.035 | l0.1 | gap prior 0.003 |
|---|---|---|---|---|---|---|
| 0 / 0 | 0 | 0 | **1.00** | 1.00 | 0 | 0 |
| 5 / 0.5 | 0 | 0.30 | **0.97** | 1.00 | 0.13 | 0 |
| 8 / 0.8 | 0 | 0.90 | **0.80** | 0.77 | 0.27 | 0 |
| 12 / 1.0 | 0 | 0.67 | 0.17 | 0.07 | 0.03 | 0 |

(public recon-vs-GT is 5.2 deg mean ori, so 5-8 deg is the realistic range; v7_g9rot is consistent only if the hidden noise is large.)

GT-free image evidence, ep32 spoon, changed frames (`streamG9_gross/code/evid_compare.py`; out/evid_ep32_*.json):

| | balls, frames with a usable detection (min px < 20): median px, frames better | visible IoU median | stereo depth median |
|---|---|---|---|
| final_v6 -> v8_g10 (46 frames) | **16.3 -> 7.4**, 32 vs 7 | 0.723 -> 0.709 | 6.7 -> 8.9 mm (worse, the known in-fist stereo bias, as for G9) |
| v7_g9rot -> v8_g10 (50 frames) | **9.5 -> 7.5**, 22 vs 10 | 0.723 -> 0.712 | 7.8 -> 8.9 mm |

IoU is not roll-informative here: on the static frames f0-21 it prefers the 165-deg-rolled spoon (0.82 vs 0.75) while balls prefer ours by 25 px, so IoU was not used for roll.
Visual sheets `visual/ep32_v7rot_vs_v8g10_*.png` (left v7_g9rot, right v8_g10; red = projected ball model, yellow = detections): inconclusive at this resolution.
The spoon ball model's body frame is rolled -15 deg vs our mesh on static frames (constant, removed before use; a constant body offset is absorbed by the kit registration).

## 2. Other spoon episodes (the ep32 mode re-examined; `code/balltrack.py`, `code/cands.py`)
| ep | ball-Viterbi flips (no gap prior) | v24/v25 spoon vs it | compose flip windows | verdict |
|---|---|---|---|---|
| 3 | f50-56 (~105), **f103-159 (~-135)** | 28.2 (official path, smoothed: 28.8) | c21 104-185, c27 99-158 | **adopt (v8_g10b)**: balls 24:4 frames, 20.4 -> 9.7 px; IoU 0.759 -> 0.769; depth 6.7 -> 8.0 mm |
| 5 | f73-76, 151-154 (60, 4 frames) | 6.9 | c27 135-144, 202-214 | no |
| 6 | none | 7.5 | c27 133-159 | no |
| 15 | f72-165 at 75 deg rests on 3 single-ball frames; vanishes with any gap prior | (43.5 artefact) | none | no |
| 16, 35, 37 | none (37: 3 frames) | 5.2 / 7.7-8.0 / 3.5 | | no |
| 33 | f153-156 (4 frames) | 6.4 | c21 146-231 (balls favour ours 74:20) | no |
| 36 | f294-317 at -60 (collapses to 2 frames with gap prior) | 7.0 | c21 246-319, c27 295-318 | no (R8 already rescued f295-305) |
Official consistency: compose-like truths are ruled out for 3, 33, 35, 36 (they make v25 SP fail, but official SP = 1.00).

## 3. MP-only failures (3 slots besides ep32), GT-free ranking
Compose-predicted v24 outcomes (G9 pkg scores; out/base*.json): MP-only failures only in ep8 (cup 3.04 cm), ep20 (pitcher 4.04, cup 2.89), ep1 (c21 lid 44 deg; c27 26.5) and ep32.
| rank | ep | mechanism | evidence (this stream) | fix |
|---|---|---|---|---|
| 1 | 3 | spoon roll flip f103-159 | above; v24/v25 28.8 deg vs v8_g10b | in v8_g10b (re-retarget) |
| 2 | 20 | pitcher vs cup placement | vs both composes a WORLD-frame offset (pitcher -3.2 cm world y, cup -2.1; body-frame std large, `errframe.py`) - not a mesh/registration offset: our pitcher registers onto the compose (scan-template) mesh with a 0.5 cm anchor shift, 0.43 cm chamfer, all seeds same basin (`meshreg.py`). G9: IoU favours compose 111:26 frames, depth favours ours 5.6 vs 9.4 mm | none (mixed; compose pitcher is scan-template, pose not usable) |
| 3 | 1 | white_pot_lid spin | compose lid differs by a near-constant 22-37 deg (c27) / 30-56 (c21) through the whole clip (constant spin offset); our public lid error is 21-29 deg; lid registration seeds spread +-7 deg; lid ball model is built from our own poses, so it cannot see a constant spin | none |
| 4 | 8 | beige cup mid-clip (f45-170) | compose 4-5 cm off in mid-clip; image evidence favours ours (IoU 57:22, depth 5.2 vs 10.2 mm) | none |

Predictions for the pending `ablation_knockout_v25_singles_20261029` uploads (MP drop 0.05 = episode was passing): ep35, ep36 -> drop (passing);
ep3 -> no drop if the flip is real; of ep1/8/20, two are expected not to drop (ep20 most likely, then ep1).

## 4. Validation
- Public 10 + held-out 10 contain no spoon (heldout/public manifests: 0 wooden_spoon), so this spoon-only rule cannot fire there: exports unchanged,
  r8_base stays 20/20, pos 1.114, ori 5.23, MPPE 1.171, AUC 0.931. This validation is vacuous (as for Z9/G9).
- Bundles: only `episode_000032/object_poses.npz` (and ep3 in v8_g10b) differ from final_v6 (`diff -rq`); meshes byte-identical.
- GT-free eval evidence: balls clearly better (ep32 and ep3); IoU -0.01 (ep32) / +0.01 (ep3); stereo depth 1-2 mm worse (known in-fist bias).

## 5. Re-retarget list
| ep | current component (v25) | vs current target | vs new target | action |
|---|---|---|---|---|
| **32** | H9_ep32_h19f0 | v7_g9rot: cup 0.55/2.4, spoon 0.73/12.3, MP 1 | **target_final_v8_g10: cup 0.55/2.4, spoon 0.73/31.9, MPPE 1.29, AUC 0.953, MP 0** | re-retarget (hold the ~150-165 deg roll f65-151; one slow unroll at f151-155) |
| 32 (ref) | B9_ep32_s6f0 (v24) | v6: 0.85/5.2 | v8_g10: 0.85/56.6, SP 0 | - |
| **3** (if v8_g10b) | D9_ep03_b3 | v6: pot 0.43/3.8, spoon 0.72/5.7 | **target_final_v8_g10b: pot 0.43/3.8, spoon 0.72/28.8, MPPE 1.15, AUC 0.973, MP 1 (1.2 deg margin)** | re-retarget |
| others | unchanged | | byte-identical target | none |

Confidence: ep32 v8_g10 - moderate-high that the truth holds the flip through f60-151 (the only ball-consistent family that reproduces both official results at realistic noise); the exact roll in f49-65 and f90-101 is weakly determined. ep3 - moderate (balls + two composes vs no official proof). 1/8/20 - ranking only.

## Files
- code/: g10lib.py (multi-candidate kit scorer), base32.py, implied.py, cands.py, rollscan.py, rollana.py, balltrack.py, build_v8.py, mk_tmp_target.py, noise_mc.py,
  meshreg.py, errframe.py, visual_v7rot_v8.py, py.sh, arr.sbatch, cmd.sbatch, jobs_*.txt.
- out/: base<ep>.json (+_pf.npz), cands_ep*_c2?.json, implied_ep*_c2?.npz, rollscan_ep*.npz, rollana_ep32.*, balltrack_ep*_*.json/_roll.npy, evid_*.json, noise_mc_ep32_s*.json, meshreg_*.txt, errframe_*.txt.
- tmp_targets/: scratch one-episode calibration targets (not deliverables). visual/: ep32 sheets.
- Deliverables: reconstruction_recovery_20261026/selected20_final_v8_g10, selected20_final_v8_g10b; campaign_v22_20261028/target_final_v8_g10, target_final_v8_g10b;
  scores/cand/G10_*/score.json.
