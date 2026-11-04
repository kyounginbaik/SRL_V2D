# Episodes 36/37 spoon: marker tracker + camera-path check (2026-10-22)

## Marker-ball tracker (code/marker_track.py, code/public_validate.py)
Public GT ep41 white_pot (init = stage B metric mesh): init 2.76 cm / 5.7 deg; marker 2.88 / 5.1; marker+smooth 2.86 / 5.0.
Only 2 of the pot's balls learned, ~25% frames accepted; grey bolts on the black truss and floor are confusers (dark-mount cue
not discriminative; depth-to-surface filter helps but not enough). Not better than stage B -> not applied to 36/37.

## Camera path (code/pnp_egomotion_general.py = parameterised copy of boards' SIFT+stereo-depth+PnP-RANSAC; code/compare_camera.py)
(a) final_recon world_from_rgb for 36/37 equals the stage-A monocular SLAM path (eval_hawor_latest_20261008/.../result_slam_gravity_aligned)
    up to a fixed transform (relative motion identical to 0.0 cm / 0.0 deg). Same construction for all legacy episodes.
(b) Public GT, same stage-B metric object-in-camera track, world via SLAM vs PnP camera:
| ep | object | SLAM camera | PnP camera | camera-path diff (max) |
|---|---|---|---|---|
| 41 | white_pot | 2.76 / 5.7 | **2.09 / 5.6** | 4.9 cm |
| 44 | white_pot | 2.13 / 2.7 | **1.42 / 2.7** | 3.5 cm |
| 39 | mini_dust_pan | **1.47 / 4.4** | 1.98 / 4.5 | 2.6 cm |
(c) Eval 36/37 (PnP camera, no frames held, min inliers 158/318): camera-path difference mean 0.63/0.34 cm, max 1.89/0.87 cm;
    spoon world track change mean 0.62/0.27 cm, max 1.88/0.84 cm (tracks: episode_0000NN/object_poses.npz, details camera_path_eval.json).
Shipped v13 actors vs old -> new track: ep36 2.13/16.5 -> 2.19/16.5; ep37 2.24/11.7 -> 2.30/11.7 (all MP-SR 1). No re-retarget needed.
