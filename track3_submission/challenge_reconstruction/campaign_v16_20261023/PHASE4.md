# Phase 4 (2026-10-24): physics-consistency correction of the reference for episodes 33, 25, 32

User: "do it" (after the proposal below). v19 is the current package (challenge_submissions/track3_v19_20261024, NOT uploaded).

Idea: the official error is dominated by reconstruction-vs-GT. Where our reconstruction is physically impossible (object through
the cup / rack / table, object floating or tilting with no support, insertion geometrically impossible), the real GT cannot be
like that. Project those poses to the nearest physically valid pose, CONFIRM with independent evidence (stereo depth / ICP,
RGB silhouette overlay, hand-object contact), keep the correction only where the evidence agrees, then re-plan the robot on
the corrected reference. Same class and same lineage requirements as ep1 (refinement_20261021_ep1/v13/code/lid_on_correction.py,
streamH_ep1/code/stage_ep1_component.py) and ep36 (refinement_20261021_ep36/code/correct_spoon_track.py, stage_ep36_component.py).

Rules for corrections (non-negotiable):
- Corrections change the REFERENCE object poses (object_poses.npz of a corrected source copy) with a script + JSON record
  (windows, per-frame deltas, evidence numbers, sha256 of input/output), never the submitted rollout poses. Submitted poses
  still come only from a policy rollout under evaluator build v2.
- Every corrected frame must be supported by evidence that the ORIGINAL pose is worse: stereo depth residual / ICP fit,
  silhouette IoU on RGB, or a hard physical contradiction (penetration > 5 mm into a rigid body that the video shows intact).
  Minimal change: smallest rigid adjustment that resolves the contradiction; smooth blend in/out (>= 0.25 s).
- Report how much the correction moves the reference (mean/max cm, deg) separately from the robot error vs the corrected
  reference, so the integrator can judge plausibility.
- Meshes: our reconstruction meshes only (same as v19 for that episode). No gt_best_per_object.
- Stage with a corrected-track wrapper carrying the correction lineage (pattern stage_ep1_component.py / stage_ep36_component.py /
  rackscan ep30_v8 corrected-track wrapper for Compose episodes), verify with
  common_v2_candidates/run_one.sbatch EP COMPONENT TAG (exact, completed.json).
- Usual: partition debug; distill/stage/eval --nodelist=node01; jobs <= 50 min; never touch ur-tl-* jobs; renders only node100;
  never delete/overwrite; never upload; STATUS.md lines; README in the stream dir.

Streams:
| stream | ep | v19 (local, cm/deg) | what is impossible in the reconstruction |
|---|---|---|---|
| P | 33 | cup 1.05/12.6, spoon 1.99/15.7 | spoon 3 cm below the cup floor at 14.5-15 s; scoops 6-10 s 5-8 cm too high vs where a scoop in this cup must be; in-hand 17-20 deg from 2.5 s |
| Q | 25 | piece_1 1.93/9.7, piece_2 2.17/13.2 | piece_2 long axis 62 deg from the slot it must seat in (slot +3 mm), board stays 5 cm high; board 1 orientation during insertion |
| R | 32 | cup 1.29/9.6, spoon 1.54/14.1 | cup lifts ~1 cm and tilts 10-12 deg with ~1 N hand force (unsupported); spoon bowl misses the cup 2.6-7 cm at 3-6 s |

## Phase 4b (user, 2026-10-24): rebuild every rack episode on the user's NEW rack reconstruction

New mesh: /data1/SRL_V2D/challenge_reconstruction/plastic_dish_rack_6prong_noGT.obj (sha256 1aa7b34a7af13c56..., 59399 verts,
watertight, our reconstruction, no GT). Same body frame as the old per-episode recon racks (ICP < 0.1 deg, < 0.5 mm), shape differs
~2.6 mm mean / 7 mm p95 (six-prong refinement). Per-episode copies scaled about the body origin to each episode's previous recon rack
size: campaign_v16_20261023/rack6prong_meshes/plastic_dish_rack_6prong_ep{14,15,16,28,30}.obj + provenance.json
(scales 1.0001, 0.9614, 0.9989, 0.9544, 1.0234). Use THESE files as --rack-visual. The packer reduces meshes to the kit's 4096-vertex
budget itself. Collision must be re-decomposed (new mesh), so every world needs the full re-plan. Streams: I ep30, M ep14, N ep16,
O ep28 (resumed with their context), S ep15 (new, streamS_ep15). Tags: <stream>6_ep<EP>_<name> (e.g. I6_ep30_x). New dirs only
(e.g. world_6prong, staging_6prong). Targets: beat or match the v19 recon-rack components; rack <= 1.0 / 5.
