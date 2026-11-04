# Campaign v16 (started 2026-10-23): all 20 Track 3 episodes pass MP-SR

Goal (user, 2026-10-22): every episode passes MP-SR (3 cm / 30 deg vs hidden mocap GT) by the Nov 4 2026 freeze.
Leaderboard 2026-10-21 (v13): MP-SR 0.45, SP-SR 0.80, MPPE 5.13 cm. v15 is built (eps 19, 25 changed), not uploaded.
User decisions: bundle every fix into v16 (no isolating uploads); the 9 episodes believed passing (6, 8, 10, 14, 16, 20, 28, 30, 38)
keep their v15 poses unless the user chooses otherwise, so improved runs for them are kept as candidates only.

## Corrected diagnosis (2026-10-23, replaces the "9 deg tilt in 8 episodes" claim of HANDOFF_track3_remaining_episodes.md)
- Only eps 3, 8 (legacy min-z worlds, 9.7 / 9.0 deg) and 37 (UniRetarget frozen scene, 8.4 deg) are in tilted physics worlds.
  All other components already use the depth-fitted table normal (checked within 0.6 deg / 0.4 cm).
- Official failures are recon-vs-GT error plus robot-vs-recon error on borderline objects. Robot error (local) is 1.3-2.9 cm on the
  failing episodes; the spoon (10 episodes) is the dominant object and its recon error cannot be validated (no public spoon clip).
- Wrist-level ILC (refinement_20261021_ep38/code/ilc_carry_generic.py) converges on rigid grasps (dust pan: 0.6 cm) and diverges on
  every spoon attempt (32, 33, 35) because the spoon pinch is fragile. So: stable spoon grasp first, then wrist correction.

## Streams
- A (streamA_spoon): make the spoon hold rigid enough that ILC converges; pilot ep35 (single object), then 37 (legacy MuJoCo actor in
  the calibrated world), then 3, 5, 6, 32, 33, 36, 15, 16. Target per object <= 1.2 cm / 12 deg vs our track.
- B (streamB_tilt38): table-aligned rebuild of eps 3 and 8 (pattern code/prepare_episode10_calibrated.py), carry the shipped teacher
  over by the rigid frame change, replay, distill, evaluate, stage.
- C (streamC_rigid_ilc): ILC + re-distill for rigid-grasp objects still above 1.3 cm: ep30 sweeper, ep19 sweeper, ep14 blue cup,
  ep28 pot, ep25 boards (candidates), ep1 lid/pot (previous ILC a/b/c got worse; needs a different window).
- Integrator (main session): ep37 backend decision (user), recon audit, v16 assembly from v15 via
  dexagent_remaining7_20261019/package_code/assemble_twenty_swap.py, SRL folder via make_srl_folder.py, no upload.

## Hard rules (user)
Never upload. No DexAgent mention inside challenge_submissions_SRL (build SRL folders only with make_srl_folder.py). No scanned meshes
(gt_best_per_object) in new work. Submitted poses only from policy rollouts under the frozen evaluator build (build manifest bd66b08f...,
episode37_ilchold_submission_20261019). Never edit poses by hand. Slurm partition debug; video renders only with --nodelist=node100.
Never cancel other people's jobs (ur-tl-* jobs belong to the user's UniRetarget thread). Never delete or overwrite existing outputs:
write only new directories under this campaign folder or new refinement_* subdirectories.

## Status log
See logs/STATUS.md (append-only, one line per milestone with date, stream, episode, numbers).
