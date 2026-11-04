# Camera-path check for the legacy eval episodes (2026-10-22)

Inputs: stereo-PnP camera path (code/pnp_egomotion_general.py: SIFT + stereo depth + PnP-RANSAC, hands/objects masked) vs the
current world_from_rgb (= stage-A monocular SLAM path up to a fixed transform). Object world tracks = camera path @ the same
camera-relative object poses. Per-episode numbers: camera_path_eval.json.

## Decision rule (GT-free)
Use PnP only if healthy (<= 3 frames with < 40 inliers, 5th-percentile inliers >= 80), material (> 1 cm mean change on any
object) and the static-object cue prefers it on every object with >= 20 rest frames (rest = world speed < 6 cm/s under both
paths; PnP must give >= 10% less rest-frame jitter). Public check of the cue: picks the better path on ep44 (PnP) and ep39
(SLAM); ep41 has no rest frames. Health alone does not separate them (all three public paths healthy).

## Result: no episode switches; tracks in episode_XXXXXX/object_poses.npz are the current tracks (unchanged).
Mean SLAM-vs-PnP difference 0.3-3.8 cm; largest in 6 (3.75), 32 (2.23), 35 (2.25), 3 (2.12), 5 (1.63). The cue prefers SLAM or
is undetermined everywhere except 38 (change only 0.74 cm). Cue differences are small (0.2-0.4 cm), i.e. weak evidence.

## Sensitivity: shipped v13/v14 actors vs current track and vs the PnP-variant track (episode_XXXXXX/object_poses_pnp_variant.npz)
| ep | current track (worst obj) | PnP-variant (worst obj) |
|---|---|---|
| 1 | pass, lid 2.90/12.8 | FAIL, lid 3.01/12.8 |
| 3 | pass, spoon 1.92/23.2 | FAIL, spoon 3.06/23.1 |
| 5 | pass, spoon 1.96/12.2 | pass, spoon 2.52/12.2 |
| 6 | pass, spoon 1.27/14.5 | FAIL, spoon 3.82/14.1, pot 3.24 |
| 8 | pass, cup 1.97/14.0 | pass, cup 2.50/14.0 |
| 10 | pass, 1.13/6.2 | pass, 1.35/4.2 |
| 32 | pass, spoon 1.94/16.0 | FAIL, cup 3.13/9.0 |
| 33 | pass, spoon 1.99/15.7 | pass, 1.96/15.7 |
| 35 | pass, spoon 1.98/20.1 | pass, spoon 2.53/20.0 |
| 36 | pass, 2.13/16.5 | pass, 2.19/16.5 |
| 37 | pass, 2.24/11.7 | pass, 2.30/11.7 |
| 38 | pass, 0.61/3.8 | pass, 1.07/3.8 |
| 19 | v14 rows (sweeper face relabelled) vs the un-relabelled local track: fail (dust pan 41.8/178.7) — expected, the local reference
still carries the old face of the anchor; not a regression vs official GT. |
So if the PnP camera were right, 1, 3, 6 and 32 would fail; the camera path is a real ±1-4 cm uncertainty on the legacy world
tracks that the robot inherits. Evidence is split (public GT: PnP better on 2 of 3 pot/dust-pan clips; eval static cue: SLAM).
No re-retarget performed (no track changed).

## Public GT, 10 clips (public/compare/epNN.json; code/compare_paths_public.py)
Mean object position error (cm) with the SAME camera-relative tracks placed through each camera path:
| clip | slam | pnp | slamR_pnpT | pnp_scaled | avg |
|---|---|---|---|---|---|
| 41 pot | 2.76 | 2.09 | 1.94 | 2.45 | 2.24 |
| 44 pot | 2.13 | 1.42 | 1.64 | 1.80 | 1.65 |
| 39 dust pan | 1.47 | 1.98 | 2.03 | 1.86 | 1.66 |
| 11 cups | 2.36 | 2.29 | 2.29 | 2.30 | 2.32 |
| 12 blue cup* | 18.46 | 18.57 | 18.61 | 18.57 | 18.50 |
| 18 sweeper+dust pan* | 28.53 | 28.21 | 28.18 | 28.18 | 28.35 |
| 21 blue cup | 4.44 | 5.06 | 5.03 | 4.87 | 4.70 |
| 23 boards | 1.78 | 1.91 | 1.91 | 1.57 | 1.47 |
| 24 boards | 3.99 | 2.01 | 2.01 | 2.03 | 2.10 |
| 31 sweeper* | 16.08 | 15.21 | 15.20 | 15.17 | 15.58 |
(* gross tracking failures of the object track itself; the camera comparison is still like-for-like.)
vs SLAM: pnp wins 6/10 (mean -0.32 cm), slamR_pnpT 6/10 (-0.32), pnp_scaled 7/10 (-0.32), avg 7/10 (-0.34; 5/7 and -0.40 on the
clips without gross failures; worst loss +0.26 cm, best gain -1.89 cm). MP-SR over the 10 clips: every alternative 5, SLAM 4.
Static cue predicted the per-clip winner on 4/10 clips (chance level) -> no evidence-only rule.
Decision: avg meets the >= 70% / lower-mean criterion (borderline, 10 clips) and is the lowest-risk choice -> applied.

## Applied: chosen/episode_XXXXXX/object_poses.npz (avg camera path; ep19 sweeper column relabelled by G to match the v14 rows)
Object change vs current: 0.1-1.6 cm mean (largest 6: 1.6, 3: 1.05, 32: 1.0, 35: 0.9).
Shipped v13 actors (v14 rows for 19) vs the chosen tracks (chosen/shipped_actor_scores.txt): all 13 pass MP-SR.
Worst objects: 1 lid 2.94/12.8 (thin), 3 spoon 2.33/23.1, 6 spoon 2.33/14.3, 37 2.26/11.7, 32 cup 2.19, 8 cup 2.16, 5 2.13,
36 2.13, 35 2.06/20.1, 33 1.91, 19 sweeper 1.73/10.5, 10 1.13, 38 0.79.
Only ep1 is past the 2.4 cm margin; its lid error is the reconstruction's 12 deg lid-on pot tilt (stereo-verified; vs the
flat-seated reference the same actor scores lid 1.76 cm, R/refinement_20261021_ep1/v13) which a re-plan cannot follow -> not
re-retargeted. No other episode fails or is thin -> no re-retargeting, nothing staged.
