# Phase 6 (2026-10-24): v21C — a full 20-episode hedge set targeted at the coworker's reconstruction

User: "prepare also the 20 set that succeeds all in the new recon; run the retargeting based on that again".
Goal: a second package (v21C) in which EVERY episode passes MP-SR with margin when scored against the coworker's compose_eval
tracks + meshes as ground truth. v21 (our evidence-based set) stays untouched; v21C is assembled separately.

Target data (shared disk): campaign_v16_20261023/compose_target/episode_XXXXXX/<obj>_trajectory.npz (T_world_obj, world =
rectified-left camera at frame 0), <obj>_mesh_metric.obj, camera_motion.npz, README.md. Scorer:
`python compose_target/score_vs_compose.py SUBMISSION_PARQUET RESULTS_DIR compose_target OUT.json [EP]` (kit scorers, our mesh
registered onto theirs, anchor frame-0 alignment = the official procedure). For a single staged component, build a 1-episode
submission parquet from common_v2_candidates/<TAG>/episode_XXXXXX/results/episode_XXXXXX.parquet + meshes (the packer's
row format; see the v21 package track3_submission.parquet rows) or adapt the script — document what you did.

Pass criterion for v21C: every object <= 2.4 cm mean position and <= 24 deg mean orientation vs the compose target (margin
under the 3 cm / 30 deg thresholds), MP-SR 1, build-v2 re-execution exact.

v21 vs compose today (cm/deg): ep3 spoon 5.00/29.9; ep5 spoon 3.48/17.7; ep6 spoon 2.98/24.7; ep8 beige_cup 3.26/12.5;
ep14 rack 2.86/2.2; ep19 sweeper 5.29/50.7, dust pan 40/177 (scene rotated by the sweeper face flip); ep25 piece_2 2.80/171.8
(our slotless board mesh registers the wrong way onto their slotted mesh); ep30 sweeper 1.49/139.6 (their face flip at f52);
ep32 cup 2.48/17.8, spoon 5.10/33.9; ep33 cup 4.83/15.5, spoon 7.53/36.3; ep35 spoon 4.04/27.8; ep36 spoon 2.88/50.6.
Already passing with margin: 1, 10, 15, 16, 20, 28, 37, 38 (keep v21 components).

Recipe (per episode):
1. Reference: take the compose poses as the new reference. Keep OUR meshes (what we submit); convert the compose poses into our
   mesh body frames by the fixed mesh-to-mesh registration (h2h_spoon/code and h2h_other/code have it; check residual). Express them
   in our physics world by one rigid transform per episode (camera chain / static scene), and if the compose objects rest at a
   different table height or tilt than our world's table, rebuild the table plane from the compose resting poses (document). The
   compose camera calibration differs from ours (objects 2-5 % farther from the camera): adopt their poses as given.
   Exceptions by design: ep19/ep30 sweeper face and ep36 spoon roll follow the compose track even though our marker evidence
   disagrees (this set is a hedge). ep25: submit a SLOTTED version of our own board mesh (from our collision cut, e.g.
   ep25_v15/collision/wooden_piece_*_cut3mm, as the visual mesh) so registration onto their slotted board is deterministic; no
   scanned/scan-derived meshes (their board mesh is scan-derived — do not use it).
2. Build the episode world with that reference (corrected-source copy + lineage record: source track sha256, transforms, per-frame
   shift vs the v21 reference), port the v21 teacher/actor, timeline, re-plan (pickup/retention/release/ILC/margins as before;
   margins only via streamK_ep35/code/search_margins_k.py), distill on node01.
3. Stage into a NEW base named staging_C* in your stream dir; verify `sbatch common_v2_candidates/run_one.sbatch EP COMPONENT C_epXX_<name>`
   (exact), then score the verified rollout against compose_target with score_vs_compose.py. Report both numbers.

Rules: our meshes only (no gt_best_per_object, no scan-derived meshes); poses only from policy rollouts under build v2; never edit
submitted poses by hand; partition debug; distill/stage/eval --nodelist=node01; jobs <= 50 min; never touch ur-tl-* jobs; never
delete/overwrite; never upload; STATUS.md lines ("- DATE C epXX: ..."); README section in the stream dir.
Streams: T (eps 3, 5, 6; streamT_C356), U (ep8; streamU_C8), V (ep19; streamV_C19), W (ep36; streamW_C36), M (ep14 rack),
Q (ep25), I (ep30), R (ep32), P (ep33), K (ep35) — the last six resume in their existing stream dirs.

## Completion (2026-10-24, Codex recovery)

v21C is assembled and scored: **20/20 pass** against the compose reconstruction at the kit's required 3 cm / 30 deg thresholds. The tighter phase target of 2.4 cm / 24 deg is met for 19 episodes; episode 33's spoon is 2.56 cm / 23.9 deg. These are reconstruction proxy scores, not hidden-ground-truth leaderboard results.

Episode 33's recovered power grasp plus teacher ILC and learned finger margins is `C_ep33_pg3072_m83` (cup 1.95 cm / 15.3 deg, spoon 2.56 / 23.9). All twenty selected components have exact build-v2 replay; the eight reused episodes have identical packed poses and meshes to v21. All twelve retargeted-episode videos show the selected recorded rollouts.

- Full reproducible package: `/data1/SRL_V2D/challenge_submissions/track3_v21C_20261024/all_20_episodes`.
- SRL copy: `/data1/SRL_V2D/challenge_submissions_SRL/track3_v21C_20261024/all_20_episodes`.
- Scores, threshold details and v21 comparison: `compose_scores.json`, `compose_validation.json`, `v21_comparison.json` inside both packages.
- Diagnostic knockouts: `/data1/SRL_V2D/challenge_submissions_SRL/ablation_knockout_v21C_20261024` (20 singles + 3 groups).
- Final pipeline: `package_code/finish_v21c.py`; Slurm 54939 completed successfully. No ep19 relabel or ep15 mesh replacement.

The official packer, local uploader validation, offline preflight, manifests, submission-copy hash and all knockout hashes passed. Nothing was uploaded. The evaluator commit URL still returns HTTP 404; publish the already committed local branch before submission: `git -C /data1/SRL_V2D/.local/worktree_evaluator_build_v2 push kyoungin evaluator-build-v2`.
