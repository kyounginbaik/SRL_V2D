# Stream H9 (phase 9): episode 32 re-planned against target_final_v7_g9rot

What changed: target_final_v7_g9rot (from reconstruction_recovery_20261026/selected20_final_v7_g9rot, stream G9) differs from target_final_v6
only in the ep32 wooden_spoon track. On f58-157 the spoon is rolled about its long axis by up to 180 deg and more
(compose27 rotation about our centroid, our position). The cup track, both meshes and frame 0 are byte-identical to v6.
The goal was every object <= 1.0 cm / 10 deg vs v7_g9rot, MPPE <= 0.9 and AUC >= 0.95.

Rules followed:
- Our meshes only. Submitted meshes are byte-identical to the v24 ep32 component (B9_ep32_s6f0): cup 3c3c5328..., spoon c9ab88e3....
- Poses come only from build-v2 policy rollouts. Nothing was hand-edited.
- CPU work ran on node01 and every job was <= 50 min. Nothing was overwritten, uploaded or committed. Public GT was never read.

## 1. Visual check (`code/visual_v6_v7rot.py`; `visual/ep32_v6_vs_v7rot_{0..3}.png`, `visual/ball_px.json`)
- Frames 55-165, every 5th. Each frame shows final_v6 (green, left) beside v7_g9rot (blue, right).
- Overlays: the mesh silhouette and outline, the projected spoon ball model (red circles), and the detected 2D balls (yellow crosses).
- Method: plain cv2 projection with the canonical intrinsics and each bundle's camera_from_object.

Verdict per window:
- **f61-96: mostly v7_g9rot.**
  - f90: the v6 silhouette points diagonally away from the horizontal spoon. v7 lies on the spoon (ball 14 vs 44 px only because the one detection sits on the other side).
  - f65: balls 30 -> 11 px. f85: 84 -> 69 px.
  - f75 favours v6 (2 vs 24 px). The silhouettes are near-identical, as expected for a roll about the long axis.
- **f108-139: v7_g9rot.**
  - f120 and f125: the projected balls land on the visible gray neck balls (26 -> 9 px, 35 -> 7 px). v6 puts them on the wrong side of the neck.
- **f142-155: v7_g9rot (weaker).**
  - f150: 27 -> 14 px, and the bowl goes into the cup instead of sticking out. f145: 85 -> 71 px. f155 is tied.
- Frames with ball px > 60 are dominated by detections on other objects (cup balls), so they carry no information.

## 2. World W7 (`code/build_world_b.py`, the unchanged streamB/B9 recipe; target dir = target_final_v7_g9rot)
- Inputs, same as B9: base scene streamR regrasp/pick_full/power0.6, base episode dir repaired_export_inputs/episode_000032, shipped meshes = B9_ep32_s6f0's, `--short 0.0178`.
- W7 matches B9 W6 in everything except the spoon reference on f58-157:
  - scene `W7/scene/scene_physics.xml` 36779927... is byte-identical to W6;
  - the world transform X is identical (max diff 0.0);
  - tilt 0.469 deg, lift 0.117 cm, cup yaw fix 21.14 deg, as in W6;
  - target spoon track 59a996f3....

## 3. Port and searches
- The `code/port.py` copy took the B9_ep32_s6f0 actor (B9 `runs/save_s6f0/actor`) into W7 as `runs/port_s6f0`. Weights are unchanged and the scene is identical.
  - The rollout is unchanged: cup 0.53/2.9, spoon 0.85/**43.9** vs v7_g9rot.
- **Physical limit found (`code/diag_roll.py`, `diag_roll2.py`):**
  - A one-shot rigid-hold ILC that rolls the wrist 180-300 deg drops or loses the spoon in every alpha/sigma/squeeze combination.
  - The cause is torsional slip of the thin handle in the grasp.
  - So the roll can only be won window by window from exact cached states.
- **Search code:**
  - `code/mres.py` (H9 copy): `ilc_correction` is made time-continuous. It tries both XYZ-Euler branches and unwraps by 2pi to the previous frame, so rolls past 180 deg do not flip sign. It also gains a `roll_axis` (twist-only) mode.
  - `code/wbeam3.py`: the B9 copy, with the target changed to v7_g9rot.
  - `code/wbeam4.py`: wbeam3 plus spoon roll-only twist ILC candidates about the mesh PCA axis, plus ILC+squeeze candidates.
- Chain (full-rollout result vs v7_g9rot, cup / spoon cm/deg):

| run | init | windows | beam/n/look, tol, cup_w | result |
|---|---|---|---|---|
| h1 / h2 / h3 | port | 0.5 s, 2.4-12.9 | 6/32/1.5 (h3 1.0), 0.8,10 (h3 1.0,12), 0.5 | 0.70/4.1, 1.28/33.5 · **0.60/3.4, 1.23/28.8** · 0.77/4.8, 1.36/35.1 |
| h4 / h5 / h6 (wbeam4) | h2 / h2 / h1 | 0.5 s | 6/24 | 0.64/3.1, 1.00/28.2 · **0.64/3.8, 1.23/23.6** · 0.50/3.0, 1.08/33.0 |
| h7 / h8 | h5 / h5+h4 | 0.5 s | 6/24/1.0 · 8/16 tol 1.3,8 | **0.50/2.5, 1.02/22.3** · 0.75/4.9, 1.13/21.5 |
| h9 / h10 | h7 / h7+h8 | 0.25 s on 3.2-8.2 + 0.5 s tail · 0.5 s | 6/20/0.8 · 8/16 | **0.57/3.0, 0.98/18.9** · 0.56/3.5, 1.36/22.3 |
| h11 / h12 | h9 | 0.2 s on 2.8-8.2 · 0.25 s, tol 1.3,8 | 6/16/0.6 · 6/20/0.8 | **0.69/3.9, 0.89/15.4** · 0.52/2.8, 1.00/17.3 |
| h13 / h14 | h11 | 0.2 s · 0.15 s | 6/16 · 5/14 | 0.71/3.6, 1.00/16.3 · 0.74/4.4, 0.98/16.0 (no gain) |
| h15 / h16 | h11+h13+h14 / h11 | 0.2 s on the reversals only (3.3-5.1, 6.4-8.2) + 0.5 s on 5.1-6.4 and the tail | 10/12/0.6 per-parent 3 · 6/16/1.0 tol 1.3,8 | **0.57/3.1, 0.83/12.7** · 0.55/3.8, 0.93/14.8 |
| h17 / h18 | h15 | as h15 | 10/12/0.6 · 10/12/1.0 tol 1.3,8 | **0.54/2.5, 0.79/12.4** · 0.62/3.1, 0.76/12.1 |
| h19 | h17+h18 | 0.15 s on the reversals | 10/10/0.6 | **0.55/2.4, 0.73/12.3** |

Rollouts were saved with `code/save_chain.sh` (`mres.py save`, official evaluator in W7). They were staged with `code/stage_b.py`/`stage_b.sbatch`:
- an H9 copy of the B9 stager;
- world W7; previous component B9_ep32_s6f0; lineage `episode_target_final_v7_g9rot_lineage.json`;
- each component carries `portable_archives/episode_000032_portable.tar.gz` (the parents[1] layout the assembler expects).

Build-v2 ran with `campaign_v16_20261023/common_v2_candidates/run_one.sbatch 32 <component> <TAG>`. Scoring used `code/score_any.sh 32 <build-v2 dir> <label> <target>`.

## Results (score_any, build-v2 re-execution; every completed.json has exact_previous_physical_states_actions = true)

| tag | component (streamH9_ep32v7/...) | vs v7_g9rot: cup / spoon | MPPE / AUC | MP | vs target_final_v6: cup / spoon | MPPE / AUC | MP |
|---|---|---|---|---|---|---|---|
| **H9_ep32_h19f0** (recommended) | staging/ep32_h19f0/components/episode_000032 | **0.55/2.4, 0.73/12.3** | **0.832 / 0.963** | 1 | 0.55/2.4, 0.73/40.7 | 1.434 / 0.949 | **0** |
| H9_ep32_h17f0 | staging/ep32_h17f0/components/episode_000032 | 0.54/2.5, 0.79/12.4 | 0.853 / 0.959 | 1 | 0.54/2.5, 0.79/41.6 | 1.465 / 0.946 | 0 |
| H9_ep32_h15f0 | staging/ep32_h15f0/components/episode_000032 | 0.57/3.1, 0.83/12.7 | 0.903 / 0.957 | 1 | 0.57/3.1, 0.83/40.8 | 1.492 / 0.943 | 0 |
| H9_ep32_h16f0 | staging/ep32_h16f0/components/episode_000032 | 0.55/3.8, 0.93/14.8 | 0.967 / 0.954 | 1 | 0.55/3.8, 0.93/41.2 | 1.515 / 0.941 | 0 |
| H9_ep32_h11f0 | staging/ep32_h11f0/components/episode_000032 | 0.69/3.9, 0.89/15.4 | 1.043 / 0.948 | 1 | 0.69/3.9, 0.89/40.6 | 1.564 / 0.937 | 0 |
| H9_ep32_h9f0 | staging/ep32_h9f0/components/episode_000032 | 0.57/3.0, 0.98/18.9 | 1.068 / 0.945 | 1 | 0.57/3.0, 0.98/34.0 | 1.391 / 0.943 | 0 |
| H9_ep32_h7f0 | staging/ep32_h7f0/components/episode_000032 | 0.50/2.5, 1.02/22.3 | 1.122 / 0.947 | 1 | 0.50/2.5, 1.02/32.3 | 1.317 / 0.948 | 0 |
| H9_ep32_h5f0 | staging/ep32_h5f0/components/episode_000032 | 0.64/3.8, 1.23/23.6 | 1.304 / 0.932 | 1 | 0.64/3.8, 1.23/30.1 | 1.419 / 0.934 | 0 |
| B9_ep32_s6f0 (v24, before) | streamB9_ep32/staging/ep32_s6f0/... | 0.53/2.9, 0.85/43.9 | 1.519 / 0.948 | **0** | 0.53/2.9, 0.85/5.2 | 0.752 / 0.967 | 1 |

- H9_ep32_h19f0:
  - build-v2 output: `campaign_v16_20261023/common_v2_candidates/H9_ep32_h19f0/episode_000032`;
  - checkpoint sha256 491e1b63c5fc7c9d...;
  - portable archive `staging/ep32_h19f0/portable_archives/episode_000032_portable.tar.gz`, sha256 cfd6cb0f598bbd66....
- Meshes in the staged component and the build-v2 output: cup 3c3c5328..., spoon c9ab88e3..., byte-identical to B9_ep32_s6f0.
- Scores: `campaign_v22_20261028/scores/cand/H9_ep32_<x>/score.json` (vs v7_g9rot) and `.../H9_ep32_<x>_vs_v6/score.json`.

Goal status for H9_ep32_h19f0 vs v7_g9rot:
- Met: cup 0.55 cm / 2.4 deg; spoon position 0.73 cm; MPPE 0.832 <= 0.9; AUC 0.963 >= 0.95.
- **Not met: spoon orientation 12.3 deg vs the 10 deg goal.**
  - The residual sits in the target's fast roll reversals. At 3.5-5.0 s (f70-99) the roll swings 180 -> 67 -> 170 -> 0 deg. At 6.5-8.0 s (f130-157) it goes 175 -> 17 -> 172 -> 0 deg.
  - Those swings run at up to about 25 deg/frame (500 deg/s), against a grasp that slips in torsion.
  - The last three rounds gained 0.4, 0.3 and 0.1 deg: a plateau at about 12 deg.

## Caveat (important for selection)
- Every v7-tracking component **fails MP-SR against target_final_v6** (spoon 30-42 deg). B9_ep32_s6f0 fails MP-SR against v7_g9rot (43.9 deg).
- The ep32 choice is therefore a bet on which reconstruction is right:
  - the 2D ball / visual evidence above favours v7_g9rot on f108-155 and mostly on f61-96;
  - G9 notes the stereo depth residual is 2-4 mm worse for v7.
- Proxy scores are against our reconstructions, not hidden GT.

## Files
- code/:
  - visual_v6_v7rot.py; build_world_b.py, blib.py, mlib.py (target v7_g9rot);
  - port.py, mres.py (continuous / roll ILC), wbeam3.py, wbeam4.py;
  - diag_roll.py, diag_roll2.py, save_chain.sh, stage_b.py, stage_b.sbatch, res.sbatch.
- W7/: the world. runs/: port_s6f0, h1-h19, save_*, eval_*, init_*.npz. staging/: ep32_h{5,7,9,11,15,16,17,19}f0. visual/. logs/.
