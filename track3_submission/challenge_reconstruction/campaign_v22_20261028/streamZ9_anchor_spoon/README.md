# Stream Z9 (phase 9): anchor frame 0 + spoon audit (GT-free)

Rules from PHASE9.md: no public scans or GT meshes anywhere in preparation. Public GT is used only inside the unchanged kit
scorer (`tools/reconstruction_audit/score_bundle.py` -> `bscore`). Nothing earlier was overwritten. No meshes were changed,
so this stream does not conflict with Y9's sweeper/piece/rack/cup meshes.

**Result:** candidate bundle `reconstruction_recovery_20261026/selected20_final_v6_z9/` = `selected20_final_v5` with exactly one change:
**ep32 wooden_spoon frame 0** (the scorer's anchor) is replaced by its static-window consensus pose. The rotation changes by 5.89 deg
(roll about the long axis) and the centroid moves 4.5 mm. Scored shift vs v5 (`target_delta.py`, same mesh both sides):
spoon 2.19 cm / 5.87 deg mean (max 2.76 cm), blue cup 0.48 cm / 5.79 deg mean (max 1.45 cm).
The other 19 episodes and every mesh are byte-identical to final_v5.
On public 10 + held-out 10 the rule fires nowhere, so the result is bit-identical: 20/20, pos 1.114, ori 5.23, MPPE 1.171, AUC 0.9312.
A 6-deg frame-0 corruption test shows the rule repairs real outliers (below).

## Tools (code/)
- `z9_anchor.py`: the anchor frame-0 rule (modes `final` = the adopted rule; `consensus`, `evidence` and `none` are the explored variants;
  `--inject DEG` = synthetic public test). `make_cmds.py`, `cmd_array.sbatch`, `make_scores.py` (scorer reuse when the poses equal an already scored folder), `score_array.sbatch`.
- `spoon_audit.py`: spoon length (2D mask vs mesh along the projected axis), ball layout (2D only), roll-flip and IoU checks.
- `length_audit.py`: the same 2D length measurement for every non-symmetric object (calibration on public meshes).
- `body_refit.py`: constant body-frame silhouette refit (rejected, see below). `flipcheck.py`, `overlay.py`: diagnostics.
- Outputs are in `out/` (per-clip JSON reports, `spoon_risk.json`, `delta32_v6z9_vs_v5.json`, `delta_bodyfit_vs_v6z9.json`, `flipcheck.txt`) and `viz/` (overlays).
- Exports (new folders only): `canonical/{evaluation,public,public_heldout}/episode_*/exports/z9_anchor_final` (adopted),
  `z9_anchor_ev`, `z9_anchor_cs`, `z9_inj6_{none,cs,ev,final}` (public only), `z9_bodyfit` (public), `z9_bodyfit_spoon` (eval, rejected).

## Task 1: anchor frame-0 rule (one rule for every clip)
For the scorer's anchor (first non-symmetric slot in roster order):
1. Static window W = frames before sustained motion (`anchor_audit.static_window`). Consensus C = robust mean pose of W (R8's robust rotation mean, median centroid).
2. Candidate only if frame 0 is an outlier against its own window: rot dev >= max(2 deg, 3 x the window's median rot deviation), or pos dev >= max(10 mm, 3 x the median pos deviation).
3. The window must be static by our own evidence: |W| >= 8, median spread <= 2.5 deg / 5 mm (pose), C projected as one static world pose through
   every window frame's camera keeps silhouette IoU >= 0.85 x the per-frame tracked IoU on >= 80 % of frames (mask motion), and the median stereo residual stays within 10 mm of the tracked pose's (stereo).
4. Image evidence must reject frame 0. The window's summed cost (10 x (1 - IoU) + min(ball px, 40)/2, with balls only for the spoon, 2D only) is profiled along the geodesic P0 -> C.
   Frame 0 is rejected if the minimiser s* > 0 in >= 95 % of 500 frame bootstraps.
5. The leading frames that are outliers (step 2) get C. Every other frame is untouched.

### Validation (public 10 + held-out 10; GT only in the scorer)
| variant | fires on | MP-SR | mean pos cm | mean ori deg | MPPE | AUC |
|---|---|---|---|---|---|---|
| r8_base (final policy) | - | 20/20 | 1.114 | 5.23 | 1.171 | 0.9312 |
| **z9_anchor_final (adopted)** | none (bit-identical) | 20/20 | 1.114 | 5.23 | 1.171 | 0.9312 |
| z9_anchor_ev (no 3x-spread gate, target = evidence optimum s*) | pub21 (10 frames) | 20/20 | 1.120 | 5.22 | 1.176 | 0.9306 |
| 6-deg frame-0 corruption of every anchor (seeded random axis) | - | 18/20 | 1.654 | 8.23 | 1.743 | 0.8658 |
| corruption + z9 rule (final) | 14/19 anchors | 20/20 | 1.222 | 6.11 | 1.305 | 0.9161 |
| corruption + evidence-optimum target | 14/19 | 20/20 | 1.235 | 6.19 | 1.325 | 0.9141 |

On the 14 clips where the rule fired after corruption: MPPE base 1.203, corrupted 1.833, repaired 1.207. The 5 that did not fire (21, 22, 31, 41, 44)
are windows that are not static, or pots/pitchers where the evidence cannot see the injected axis. These are misses, not false fires.
- The 3x-spread gate is what keeps pub21 (2.25 deg dev, 1.70 spread) from firing. Without it the rule makes pub21 pitcher worse (1.49 -> 1.67 cm).
- Target choice: the image-evidence optimum s* is noisy and shrinks toward frame 0 (39/40: s* 0.65 when the truth is s = 1). The consensus C is better on public (1.305 vs 1.325), so C is the target.

### All 20 eval episodes
| ep | anchor | f0 dev deg / mm | gate deg | window frames | static | s* [p5, p95] | fired | change |
|---|---|---|---|---|---|---|---|---|
| 1 | white_pot | 1.24 / 3.5 | 4.17 | 80 | yes | - | no | - |
| 3 | white_pot | 0.79 / 3.3 | 2.05 | 65 | yes | - | no | - |
| 5 | white_pot | 0.49 / 1.3 | 2.00 | 28 | yes | - | no | - |
| 6 | white_pot | 0.39 / 3.8 | 4.37 | 37 | yes | - | no | - |
| 8 | white_pot | 0.70 / 1.4 | 2.00 | 80 | yes | - | no | - |
| 10 | none (all symmetric) | - | - | - | - | - | no | - |
| 14 | rack | 0.51 / 0.6 | 2.62 | 80 | yes | - | no | - |
| 15 | rack | 0.51 / 2.9 | 2.00 | 80 | yes | - | no | - |
| 16 | rack | 0.28 / 1.3 | 2.02 | 80 | yes | - | no | - |
| 19 | mini_sweeper | 0.93 / 1.0 | 5.67 | 32 | yes | - | no | - |
| 20 | water_pitcher | 0.33 / 0.4 | 2.00 | 29 | yes | - | no | - |
| 25 | wooden_piece_1 | 0.11 / 0.6 | 2.00 | 40 | yes | - | no | - |
| 28 | rack | 0.40 / 2.0 | 2.00 | 26 | yes | - | no | - |
| 30 | mini_sweeper | 0.30 / 0.5 | 3.94 | 48 | yes | - | no | - |
| **32** | **wooden_spoon** | **5.89 / 4.5** | 3.63 | 24 | yes | 0.56 [0.49, 0.67] | **yes, frame 0 only** | **5.89 deg / 4.5 mm; scored spoon 2.19 cm / 5.87 deg, cup 0.48 cm / 5.79 deg** |
| 33 | wooden_spoon | 1.22 / 0.6 | 10.95 | 49 | no (3.7 deg spread) | - | no | - |
| 35 | wooden_spoon | 2.02 / 0.7 | 3.05 | 11 | yes | - | no | - |
| 36 | wooden_spoon | 2.20 / 1.1 | 4.30 | 34 | yes | - | no | - |
| 37 | wooden_spoon | 0.30 / 1.3 | 2.00 | 34 | yes | - | no | - |
| 38 | mini_dust_pan | 1.52 / 0.9 | 2.00 | 44 | yes | - | no | - |

ep32 evidence: the frame-0 deviation is a pure roll of +5.8 deg about the spoon long axis. Over the 24 static frames (each frame with its own camera),
mean IoU is 0.729 for P0 vs 0.736 for C, and the ball median is 5.90 vs 5.51 px. Even on frame 0 itself C fits better (IoU 0.749 vs 0.740).
The profile minimum is s* = 0.56, so the image rejects frame 0 but places the truth about halfway to C. Expected residual about 0.44 x 5.9 = 2.6 deg (~1 cm).
The roll is weakly observable: the per-frame tracked roll jitters +-3-6 deg inside the window.

## Task 2: spoon episodes (no public spoon; GT-free audit)
Our spoon mesh (one mesh, md5 5fcd334c, used in all 10 spoon episodes): length 30.8 cm, bowl width 6.3 cm, bowl z-extent 2.7 cm, handle about 1.8 x 1.7 cm.
The ball model (from spoon_roll_v8; tip, near, far) puts the tip ball 6.9 mm inside the mesh's handle end.

### a) Size: mesh vs 2D mask, ball layout, stereo
Clean frames only (no hand or other object within 15 px, fully in view). Length ratio = 2D mask extent along the axis / projected mesh extent.
| ep | spoon anchor? | clean frames | length ratio | handle end mm | bowl end mm | ball tip-near / tip-far 2D ratio | ball px median | stereo abs residual mm | IoU median |
|---|---|---|---|---|---|---|---|---|---|
| 3 | no (pot) | 27 | 0.971 | +3.2 | -16.2 | 1.006 / 0.987 | 9.0 | 5.8 | 0.73 |
| 5 | no (pot) | 22 | 0.965 | -4.4 | -6.7 | 0.989 / 0.979 | 5.0 | 8.9 | 0.72 |
| 6 | no (pot) | 24 | 0.979 | -0.3 | -7.8 | 0.998 / 0.996 | 2.6 | 8.7 | 0.70 |
| 15 | no (rack) | 8 | 0.984 | +2.7 | -7.7 | 1.031 / 1.027 | 6.3 | 4.1 | 0.74 |
| 16 | no (rack) | 8 | 0.981 | +1.4 | -7.2 | 1.021 / 1.020 | 6.0 | 4.4 | 0.73 |
| 32 | yes | 21 | 0.954 | -0.7 | -17.0 | 0.999 / 0.977 | 8.5 | 5.9 | 0.66 |
| 33 | yes | 21 | 0.978 | -0.3 | -7.7 | 1.007 / 1.005 | 8.6 | 7.0 | 0.67 |
| 35 | yes | 17 | 0.980 | -2.2 | -3.6 | 1.031 / 1.032 | 8.2 | 4.8 | 0.75 |
| 36 | yes | 22 | 0.971 | -3.7 | -5.5 | 1.026 / 1.018 | 7.7 | 6.7 | 0.69 |
| 37 | yes | 25 | 0.970 | -3.9 | -5.4 | 1.029 / 1.017 | 3.3 | 4.3 | 0.69 |

Calibration on public (`length_audit.py`, our meshes that the scorer accepts): wooden pieces give 0.948-0.965, rack 0.976-0.982, pitcher 0.905-0.924, dust pan 0.885-0.934.
The mask-vs-mesh measurement therefore reads 2-5 % short on correct meshes. The spoon's 0.954-0.984 (median 0.973) is inside that band,
so there is **no evidence that the spoon mesh is too long or too wide. Mesh not changed.** The 2D ball layout ratios (0.98-1.03) and the stereo residual (4-9 mm,
the same toward-camera stereo bias R8 found on every object) show no scale or bowl-depth inconsistency either.
The exception is **eps 3 and 32, which are axially asymmetric.** The mesh overshoots the mask by 16-17 mm at the bowl and is flush at the handle, on frames at both the start and the end.
The same appears in spoon_roll_v8 before the joint refine, so it is a pose offset (bowl side), not a mesh error. See `viz/ep32_f10_v5_vs_bodyfit.png`.

### b) Roll flips / half turns (2D balls + IoU)
Per sampled frame (stride 2), the pose is compared with the same pose rolled 180 deg about the handle line (`out/spoon*.json`, `out/flipcheck.txt`):
| ep | frames where the flip fits balls >= 2x better | ... where it fits better at all | notes |
|---|---|---|---|
| 3 | 1.3 % (f126-128) | 12.7 % | f126-128 in hand: flip 17 vs 37 px, IoU equal |
| 5 | 0 | 10.4 % | |
| 6 | 0 | 2.5 % | |
| 15 / 16 | 0 | 1 % | |
| 32 | 5.4 % (f121-125, 149-150) | 21.5 % | in hand: flip 6-13 vs 26-35 px with 3 balls, IoU equal. f90-91 and 152-154 have IoU < 0.25 (pose lost in the fist) |
| 33 | 0.6 % | 12.6 % | f158, 190, 202 IoU 0.11-0.30 (hand-over); 189-191 ball px > 120 for both hypotheses |
| 35 | 0 | 4.3 % | |
| 36 | 0 | 10.8 % | f84-86 IoU 0.24-0.28, no balls |
| 37 | 2.6 % (f50-52) | 7.9 % | the flip loses IoU 0.81 -> 0.70, likely ball misdetections |

**Not fixed.** The in-hand windows mix flip-better, neither-fits and pose-better frames. Isolated flip segments (ep32 f121-125, ep3 f126-128)
would need two 180-deg roll steps within 50 ms. The roll_v8 Viterbi has already weighed these balls against continuity, and there is no public spoon to validate a re-weighting.

### c) Frame 0 where the spoon is the anchor (32, 33, 35, 36, 37)
- ep32: fixed (task 1).
- ep33: not static by our evidence (3.7 deg window spread). Frame 0 is 1.2 deg from the window mean, ~0.3 cm scored.
- ep35: 2.0 deg dev on an 11-frame window (gate 3.05). Evidence keeps frame 0 (s* = -1).
- ep36: 2.2 deg (gate 4.3). Evidence slightly prefers frame 0 (s* = -0.09).
- ep37: 0.3 deg, fine.
- Spoon is not the anchor in 3, 5, 6 (pot) or 15, 16 (rack). There its frame-0 deviation (0.5-2.4 deg) only affects the spoon itself.

### Rejected: constant body-frame silhouette refit (`body_refit.py`)
This optimises one 6-DoF body offset per clip (|r| <= 5 deg, |t| <= 2 cm) to maximise multi-frame mask IoU, accepting only if the gain is >= 0.02, positive on >= 75 % of frames, and balls are no worse.
On the spoon it looks convincing: ep32 IoU 0.729 -> 0.774 with balls 8.6 -> 7.4 px, and the overlay fits. But the roll term sits at the +5 deg bound in all eight spoon fits, which is suspicious.
On public it is clearly worse: 17/20, pos 1.468, MPPE 1.518, AUC 0.890. The thin objects are mixed: dust pan pub18 2.03 -> 4.70 cm although its IoU rose 0.86 -> 0.90, sweeper 1.01 -> 1.05, dust pan pub40 0.86 -> 0.72, piece_2 pub23 1.08 -> 1.06.
A silhouette IoU gain is not evidence of a better 3D pose, so the rule was rejected for every object, including the spoon.
Its eval deltas measure the spoon's silhouette-vs-current disagreement: 3: 0.62 cm / 7.3 deg; 5: 0.22 / 6.1; 32: 1.63 / 3.9 (cup 0.56 / 6.5); 33: 1.13 / 3.8; 35: 1.02 / 4.0; 37: 0.49 / 0.1 (36 and 6 were not accepted).

### Per-episode spoon risk (GT-free; linear and conservative)
risk = 1.2 cm prior (public thin objects with the final policy: sweeper 1.00, pieces 1.25-1.33) + residual anchor frame-0 shift (spoon anchors) + half the silhouette-refit disagreement.
The orientation risk from roll-flip windows is given separately (fraction of frames x 180 deg; strong-weak range).
| ep | anchor f0 term cm | silhouette disagreement cm | **spoon risk cm** | roll-flip ori risk deg | main driver |
|---|---|---|---|---|---|
| 3 | - | 0.62 | **1.5** | 2-23 | bowl-side axial offset (16 mm), in-hand flips f126-128 |
| 5 | - | 0.22 | **1.3** | 0-19 | |
| 6 | - | n/a | **1.2** | 0-4 | |
| 15 / 16 | - | n/a (8 clean frames) | **1.2** | 0-2 | |
| 32 | 0.96 (after fix; s* = 0.56) | 1.63 | **3.0** | 10-39 | anchor roll, bowl-side offset, in-hand flips f121-125 / lost f90-91, 152-154 |
| 33 | 0.28 | 1.13 | **2.0** | 1-23 | hand-over windows f158-202 (IoU 0.1-0.3, balls > 120 px) |
| 35 | 0.47 | 1.02 | **2.2** | 0-8 | short 11-frame anchor window |
| 36 | 0.47 | n/a | **1.7** | 0-19 | anchor f0 2.2 deg (kept by evidence) |
| 37 | 0.14 | 0.49 | **1.6** | 5-14 | f50-52 in-hand ambiguity |

## Bundle
`reconstruction_recovery_20261026/selected20_final_v6_z9/` (copy of final_v5. Only `episode_000032/object_poses.npz` changed, plus
`episode_000032/z9_anchor.json` and the `bundle_manifest.json` note). Merge with Y9: take Y9's meshes/poses for sweeper/piece/rack/cup, and
take ep32 wooden_spoon frame 0 from this bundle. Y9 does not touch the spoon, so the two do not conflict.
