# Stream G16: wooden-spoon audit of the v12b reconstruction (eval spoon episodes 3, 5, 6, 15, 16, 32, 33, 35, 36, 37)

Rules followed: GT-free evidence only (2D gray-ball detections, stereo ball points, our masks/depth, our meshes); no public
scan/GT mesh read anywhere (no spoon exists in public anyway); official-outcome calibration uses only the official
episode-level SP/MP results (as G10/G12 did). Nothing overwritten: new bundle `selected20_final_v13_g16`, new target
`campaign_v22_20261028/target_final_v13_g16`, this directory. No upload, no commit. No shared code path changed
(all tools are new files here), so the public 20/20 result is untouched; the spoon rule cannot fire on public (no spoon).
CPU jobs on node01 (debug), each < 50 min.

Note: the spoon is in 10 eval episodes (the task text listed 8; 15 and 33 also contain it).

## Clock
v12b `camera_from_object(t)` = v10_g12 `camera_from_object(t + delta)` exactly (G15 retime; verified max diff 0.0 on all 10
episodes). So every RGB-frame test here (ball detections, masks) uses the v10_g12 spoon pose at the same frame, i.e. v12b
resampled at t - delta (delta 0.875-0.9). Stereo ball points / depth are on the b/c clock (~0.9 frame ahead) and are used as a
secondary cue, mainly on the early (slow) frames.

## 1. Per-frame roll / end-flip scan (`code/g16scan.py`, `code/g16ana.py`)
Hypotheses per frame: adopted pose x {roll about the long PCA axis through the surface centroid, 5 deg step (balls) / 15 deg
(silhouette IoU + stereo depth)} x {no flip, 180 deg tip<->handle flip}. Ball cost = median nearest-ball px (shared ball
model `spoon_roll_v8`). Informative frame = >= 2 detections and best roll-only px < 20. Roll-only Viterbi (lambda 0.05/deg).
"Adopted" = roll 0 corrected by the episode's early-window offset OFF (column "early roll").

| ep | informative/total | single-frame best roll >60 deg from adopted (frac of informative) | Viterbi path >60 deg (frac of all) | windows | ball px median before -> after | IoU adopted / best end-flipped |
|---|---|---|---|---|---|---|
| 3 | 143/313 | 3 (0.021) | 0 | - | 7.17 -> 6.49 | 0.76 / 0.57 |
| 5 | 94/288 | 1 (0.011) | 0 | - | 3.23 -> 3.17 | 0.75 / 0.57 |
| 6 | 123/243 | 0 | 0 | - | 1.79 -> 1.74 | 0.79 / 0.58 |
| 15 | 52/166 | 1 (0.019) | 0 | - | 6.11 -> 6.11 | 0.80 / 0.60 |
| 16 | 62/173 | 0 | 0 | - | 5.34 -> 5.34 | 0.79 / 0.60 |
| 32 | 147/259 | 13 (0.088) | 0 | - | 6.56 -> 4.92 | 0.72 / 0.57 |
| 33 | 139/348 | 1 (0.007) | 0 | - | 2.35 -> 2.27 | 0.74 / 0.57 |
| 35 | 291/421 | 8 (0.027) | 0 | - | 6.53 -> 6.39 | 0.79 / 0.57 |
| 36 | 185/407 | 15 (0.081) | 0 | - | 3.53 -> 3.48 | 0.76 / 0.59 |
| 37 | 118/151 | 3 (0.025) | 0.033 | f50-54 (3 informative, 15-23 px) | 3.12 -> 3.05 | 0.79 / 0.58 |

"before" = adopted at roll 0 (raw), "after" = roll-only Viterbi path.

- **No remaining roll flips.** The roll-only Viterbi path never departs > 60 deg from the adopted v12b track except ep37 f50-54
  (5 frames; only 3 informative, best fits 15-23 px = poor; < 2 deg time-mean effect even if real) -> not changed. Single-frame
  disagreements are isolated frames with 1-10 px margins (noise). The G10/G12 flip windows of ep3 and ep32 are confirmed (the
  path follows the adopted track through them).
- **No end (tip/handle) flips.** Allowing the flip, the balls "prefer" it only where the detections lie 100-160 px from the
  adopted balls (spurious in-hand detections; ep33 f60-66/f246-281, ep36 f98-122, ep3 f87-125); there the silhouette rejects the
  flip by 0.2-0.3 IoU (e.g. ep33 0.80 vs 0.55). The few frames where flipped IoU wins (ep32 f174-182, ep36 f46-49/f81-85) are
  occlusion frames with IoU < 0.45 and no balls. Median IoU adopted 0.72-0.80 vs best flipped 0.57-0.60 in every episode.
- Position glitch noted (not a roll issue): ep36 f82-83 stereo depth residual 99/118 mm (spoon nearly hidden; 2 frames).

## 2. Frame-0 / constant roll bias (`code/offs.py`, `offs2.py`, `offs3.py`, `f0fit.py`, `f0sil.py`)
The ten spoon episodes use the SAME spoon mesh (md5 5fcd334c) and the same ball model, so a constant body-frame roll between
our track and the balls must be ~equal in all episodes if it were a model/mesh offset. It is not:

| ep | 2D roll scan f0-23 | 2D + free image shift | 3D stereo balls | 3D + free shift | 6-DoF fit 2D (rot+trans) | 6-DoF fit 3D | whole clip 2D (CI90) |
|---|---|---|---|---|---|---|---|
| 3 | -20 | -10 | -15 | -10 | -12.4 | -18.5 | -20 [-25, -15] |
| 5 | +10 | +5 | +5 | +5 | -2.7 | +12.0 | 0 [-5, 0] |
| 6 | +5 | +5 | +5 | +5 | +5.6 | +5.4 | 0 [0, 0] |
| 15 | 0 | +45 (unstable) | -5 | 0 | +39.0 (ill-posed) | +1.6 | 0 [-5, 0] |
| 16 | 0 | +45 (unstable) | 0 | 0 | +40.1 (ill-posed) | +2.8 | -5 [-5, 0] |
| **32** | **-15** | **-10** | **-15** | **-15** | **-11.5** | **-12.6** | **-15 [-20, -15]** |
| 33 | 0 | +5 | +15 | +10 | +4.0 | -4.2 | 0 [0, 0] |
| 35 | 0 | +5 | 0 | 0 | +5.3 | +5.5 | 0 [0, 0] |
| 36 | -5 | -10 | -10 | -10 | -8.1 | -5.4 | 0 [0, 0] |
| 37 | 0 | 0 | -5 | 0 | +0.4 | +0.1 | 0 [0, 0] |

- 8 of 10 episodes sit at 0 +- 5-10 deg (the model frame matches the mesh); **ep32 (-15) and ep3 (-13.5) are outliers by
  every estimator**, CI excluding 0, over the static window and the whole clip; ep32's tilt of the long axis is 2-3.5 deg only.
- The silhouette, which is not roll-informative by itself, shows the same shift: in 8 episodes IoU peaks at roll -15, in
  ep3/ep32 at <= -30 (out/f0_sil_depth.json).
- G10 had removed this offset as a "ball-model offset" (it only looked at ep32); the other 8 episodes show it is
  episode-specific, i.e. a tracking bias of ep3/ep32 (both clips likely share the same tracker init failure).
- ep36 (-5..-10, median -8) is consistent in sign but below the 10 deg rule threshold; ep33 estimators disagree in sign (0..+15
  vs -4..-6) -> unchanged.

**Why this matters (kit mechanics).** `_align_registered_scene` takes the rotation from the anchor (first non-symmetric slot)
but the translation pivot from slot 0. In ep32/ep33 slot 0 is the (symmetric) blue cup and the spoon is the anchor, d = 21 cm:
a 15 deg spoon f0 error rotates the spoon about the cup -> 5.5 cm spoon position error and ~15 deg cup tilt.
Single-object episodes (35/36/37) lever the f0 error on the spoon's own displacement (ep36: 9 deg -> 1.8 cm).
In ep3/5/6 (pot) and 15/16 (rack) the other object is the anchor, so only the spoon's own orientation changes.

## 3. Bundle v13_g16 and target
`selected20_final_v13_g16` = v12b + one constant body roll of the spoon on every frame of ep32 (-15 deg) and ep3 (-13.5 deg)
(`code/build_v13.py`; rule: all six early-window estimators agree in sign, |median| >= 10, CI excludes 0). Everything else
byte-identical (diff -rq: only the two object_poses.npz + json notes). Target `target_final_v13_g16` built with
`campaign_v22_20261028/code/build_recon_target.py` (same builder as v12b); only the two spoon trajectories differ.

### Impact (kit-faithful, our meshes as reference; `code/g16impact.py`, official scorer path `code/score_any.sh`)
| ep | v12b -> v13 time-mean delta | v28 component vs v12b | v28 component vs v13 |
|---|---|---|---|
| 32 | spoon 5.49 cm / 7.7 deg, cup 1.15 cm / 14.7 deg (MP of v12b-vs-v13 = fail, by position) | MP 1: cup 1.05/3.9, spoon 1.31/22.4, MPPE 1.548, AUC 0.923 | **MP 0**: cup 1.77/16.3, spoon 5.93/25.4, MPPE 4.17 |
| 3 | spoon 0 cm / 12.5 deg | MP 1: pot 0.35/3.2, spoon 0.66/9.8, MPPE 0.685 | MP 1: pot 0.35/3.2, spoon 0.66/17.6, MPPE 0.875 |
| (36, -9 deg, not built) | spoon 1.79 cm / 10.3 deg | spoon 0.79/4.9 | spoon 2.08/12.1 |

Old packages vs v13 (proxy): ep3 v24/v25 spoon 34.4 deg (MP fail), v26 19.3 (pass); ep32 v24 SP0/MP0, v25 SP0 (28.7, limit 28.6), v26 SP1/MP0.

### Official-outcome consistency for ep32 (`code/g16mc.py`; official ep32: v24 SP0/MP0, v25 SP1/MP0, v26 SP1/MP0)
| hidden noise | P(official) truth v12b | truth v13 (-15) | truth v10 (RGB clock) | v10 -15 | P(v28 ep32 MP pass) v12b / v13 |
|---|---|---|---|---|---|
| 0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 / 0.0 |
| 5 deg / 0.5 cm | 0.95 | 0.275 | 0.95 | 0.375 | 0.675 / 0.025 |
| 8 deg / 0.8 cm | 0.50 | 0.125 | 0.60 | 0.275 | 0.125 / 0.0 |

Roll profile (truth = v12b rolled 0/-5/-10/-15/+5/-20 deg; job 65730, `code/jobs_mc2.txt`): still queued on node01 at hand-back; results will land in `out/mc_ep32_*_rolls*.json` (not yet read).

### Confidence
- ep32: GT-free image evidence for -15 deg is consistent across six cues but small in px/mm; the official outcomes (which also contain
  any kit registration roll offset of our spoon mesh vs the GT mesh) favour v12b by ~3.5-4x. A half-way hedge does not work
  (7.5 deg x 21 cm lever = 2.75 cm + tracking error). Even under v12b the ep32 margin is thin (v28 passes MP in 0.675 of draws at 5 deg).
- ep3: moderate; safe either way (current v28 ep3 component passes both targets).

### Decision (main, 2026-10-30)
**ep3 roll adopted; ep32 roll NOT adopted (official outcomes favour v12b).** Note: `selected20_final_v13_g16` / `target_final_v13_g16`
contain BOTH changes as built; an ep3-only bundle would be v12b + ep3 -13.5 deg (`code/build_v13.py OUT 3:-13.5`).

## Files
- code/: g16scan.py (scan), g16ana.py (analysis), peek.py, offs.py / offs2.py / offs3.py / f0fit.py / f0sil.py (constant-roll
  estimators), g16impact.py (kit-faithful deltas + package scores vs candidate truths), g16mc.py (official-outcome noise MC),
  build_v13.py, arr.sbatch / cmd.sbatch / py.sh, jobs_*.txt
- out/: scan_ep*_*.npz, ana_ep*_l0.05_p20_b2.{json,npz} (flip allowed), ana_ep*_l0.05_p20_b2_r1.{json,npz} (roll only), roll_offsets.json, offs2_ep*.json, offs3_ep*.json,
  f0fit_ep*_w24.json, f0_sil_depth.json, impact_ep*_r*.json, mc_ep32_*.json
