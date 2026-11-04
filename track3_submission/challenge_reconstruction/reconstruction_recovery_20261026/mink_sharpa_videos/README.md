# Kinematic Mink IK: reconstructed hands -> floating Sharpa Wave, ego-camera videos

Kinematic only: no physics and no RL. Object poses and meshes are read unchanged from the bundle.

## Layout
```
mink_sharpa_videos/
  code/mink_video.py          pipeline (phases: decode, map, ik, render; plus check, preview)
  code/run_episode.sbatch     one Slurm array task per episode, runs all phases on node100
  <bundle_name>/
    epi<N>.mp4                side by side at 20 fps: left = canonical undistorted rgb.mp4, right = MuJoCo render
                              from world_from_rgb[t] with intrinsics.json (Sharpa hands + object meshes;
                              light-blue capsules = mapped MANO skeleton, drawn only on observed frames)
    episode_XXXXXX/
      sharpa_ik_qpos.npz      <side>_qpos_mj [T, 7+22] = MuJoCo qpos of <side>_sharpawave.xml with a freejoint on
                              the root body ([x y z qw qx qy qz, fingers]); <side>_mj_joint_names;
                              <side>_qpos_pin_xyzw (Pinocchio layout); <side>_frame_task_errors; <side>_valid; <side>_present
      hands_world.npz         MANO joints, joint quaternions (wxyz), vertices and faces in the bundle world
      mano_oldcam.npz         MANO FK in the old HaWoR camera frame
      contact_sheet.jpg       8 evenly spaced frames from the video
      projection_check.jpg    mapped MANO joints and object mesh points drawn on rgb.mp4 frames
      hands_map_report.json, ik_report.json
```

## Rerun with a new bundle (for example, the corrected-spoon-roll bundle)
```bash
cd /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/mink_sharpa_videos/code
NEW=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/<new_bundle_dir>
sbatch --array=1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38%8 run_episode.sbatch "$NEW"
```
Outputs go to `mink_sharpa_videos/<new_bundle_dir name>/`. Logs go to
`../logs/mink_sharpa_video-<job>_<episode>.log`. The array index is the episode number.

To run a single phase by hand (decode and ik start their own docker containers, so run them on node100):
```bash
B=<bundle_dir>; C=.../mink_sharpa_videos/code/mink_video.py
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
srun -p debug -w node100 python3 $C --bundle $B --phase decode --episodes 1   # docker v2d_task_library_loader:latest
$PY $C --bundle $B --phase map --episodes 1 [--map-mode anycalib|naive] [--depth stereo|metric|hawor]
srun -p debug -w node100 python3 $C --bundle $B --phase ik --episodes 1       # docker robotic-grounding:mink-1.3.0
srun -p debug -w node100 env MUJOCO_GL=egl $PY $C --bundle $B --phase render --episodes 1   # needs node100 GL
```

## Method
1. **decode**: HaWoR MANO comes from `<record.source>/<names[0]>/result_slam_gravity_aligned/result.npz`.
   Left-hand parameters are converted from right-hand parameter space by negating the y and z axis-angle
   components. MANO FK uses flat_hand_mean and center_idx=None. The HaWoR `hand_scale` is applied about the
   vertex centroid, then `cam_t` is added. This is the same contract as
   `dexagent_remaining7_20261019/code/prepare_hawor_hands.py`.
2. **map** (default `--map-mode anycalib --depth stereo`):
   - The whole hand is scaled by 1/hand_scale about the old camera centre. This keeps its projection the
     same and gives it metric MANO size.
   - The hand is rotated about the camera centre so that its wrist ray lands on the right pixel of the
     physical undistorted RGB. The pixel comes from the AnyCalib path: stable virtual pinhole -> AnyCalib
     fisheye raw pixel -> physical rational-model undistortion. This is the same path as
     `tools/reconstruction_audit/hand_priors.py` (v2).
   - The hand is moved rigidly along its ray so that its depth matches the median canonical stereo depth
     inside that hand's SAM2 mask. The per-frame ratio is clamped to 0.7-1.3 and Savitzky-Golay smoothed.
   - The result is mapped into the new world with `world_from_rgb[t]`.

   Without the stereo step, HaWoR hands sit 20-50 % too far from the camera. On episode 1, the left hand was
   within 1 cm of an object on 6 % of frames before the step and 83 % after it.
   Frames where a hand is invalid are linearly interpolated in the world frame. A hand that is never
   observed is not solved and is parked out of view.
3. **ik**: `dexagent_remaining7_20261019/code/mink_sharpa_exact.py` (`MinkSharpa`, the tuned Mink setup with
   the exact Sharpa Wave XML and a floating base), max_iter 80. Pre- and post-smoothing and joint-limit
   clipping follow `prepare_hawor_hands.ik`, with `source_to_robot_scale=1`.
4. **render**: MuJoCo 3.10 with EGL. The camera is a symmetric pinhole with focal length (fx+fy)/2 on a
   padded canvas, cropped so that (cx, cy) is the principal point. Projection was checked numerically
   against mesh points.

## Status: selected20_fused_v3_gap (2026-10-27)
All 20 episodes finished. Each video's frame count matches its bundle. Mean IK frame-task error is 1.4-2.1 cm,
and the wrist (hand_C_MC) error is about 2.5-5 cm because Sharpa's proportions differ from the MANO hand.
Visual review of the contact sheets: hands sit on or around the right objects, and the render camera matches the RGB.
Known issues:
- ep14: HaWoR has no right-hand track at all (right_hand_dir is null), so the right Sharpa hand is hidden.
  The single "left" track follows the real right hand during the cup grasp (a HaWoR identity swap), so a
  LEFT Sharpa hand grasps the cup.
- ep20: the right hand is unobserved for 70 frames (about 2.5-5 s) while it is out of view. These frames are
  interpolated and labelled in the video.
- ep37 and ep38: the left hand stays idle the whole time, so it has no object contact. This is expected.
- Stereo depth snapping pulls the hand toward the visible skin surface, so fingers may slightly penetrate
  objects in some grasps. This is kinematic only; no penetration or contact constraint is applied.
