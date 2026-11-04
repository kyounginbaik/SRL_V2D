# Phase 7 (2026-10-28): v22 — DexAgent retargeting onto our own reconstruction `selected20_final_v5`

User: "fix the eval36 spoon 295-306 too and after that do the retargeting we did last time using dexagent".
"Last time" = phase 6 (campaign_v16_20261023/PHASE6.md, v21C): take a new reconstruction as the reference, re-plan the
floating-Sharpa MuJoCo (DexAgent port) teacher, distill, stage, verify under evaluator build v2, assemble a 20-episode package.
This time the reference is OUR reconstruction-recovery bundle (validated on public GT: 10/10 public + 10/10 held-out public):
`/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v5/episode_XXXXXX`
(object_poses.npz: names, world_from_object (T,4,4), world_from_rgb, camera_from_object; meshes/<obj>.obj; camera.npz; intrinsics.json).

## Target and scorer
- Target in the compose layout: `campaign_v22_20261028/target_final_v5/episode_XXXXXX/<obj>_trajectory.npz` (T_world_obj; world =
  our canonical recon world, NOT the compose world) + `<obj>_mesh_metric.obj` (our recon mesh). Anchor frame-0 alignment makes the
  world gauge irrelevant for scoring.
- Score any build-v2 component output dir: `campaign_v22_20261028/code/score_any.sh EP TAG_DIR LABEL` -> `scores/cand/LABEL/score.json`
  (make_sub_any.py: achieved poses of env 0 + the component's own meshes; reproduces the package scores exactly).
- Pass criterion: every object <= 2.4 cm mean position and <= 24 deg mean orientation vs target_final_v5 (margin under 3 cm / 30 deg),
  MP-SR 1, build-v2 re-execution exact (`sbatch campaign_v16_20261023/common_v2_candidates/run_one.sbatch EP COMPONENT TAG`, completed.json
  exact_previous_physical_states_actions true). New tags: `X_epNN_<name>` with X = v22 stream letter, output stays in common_v2_candidates.
- Already-scored existing components: `scores/candidates_vs_final_v5.json` (all 102 build-v2 candidates/baselines, sorted, worst ratio
  = max(pos/3, deg/30)); v21 / v21C packages: `scores/v21_vs_final_v5.json`, `scores/v21C_vs_final_v5.json`.

## Recipe (per episode) = PHASE6.md recipe with compose_target -> target_final_v5
1. Reference: target poses converted into OUR submitted mesh body frames by the scorer's own registration (MP-SR._register_object_body,
   same seed/meshes as score_vs_compose.py; e.g. campaign_v16_20261023/streamT_C356/code/tlib.py, streamW_C36/code/build_compose_world.py).
   Express them in the episode's physics world by one rigid transform per episode G = Gt @ G0 (G0: target anchor (slot 0) frame-0 pose ->
   our physics anchor frame-0 pose; Gt: table-plane correction so the target RESTING contacts lie on the physics tabletop). Record
   transforms, source sha256, per-frame shift vs the v21 reference.
2. Build the episode world (corrected-source copy + lineage), port the best existing teacher/actor (start from the best candidate in
   candidates_vs_final_v5.json), re-plan (port modes ref/ach, pickup / retention CEM / release / ILC / margins as in the stream READMEs;
   margins only via streamK_ep35/code/search_margins_k.py), distill on node01.
3. Stage into a NEW staging dir in your stream dir, verify with run_one.sbatch (exact), score with score_any.sh. Report both.

Reusable stream code (read the README of each first): campaign_v16_20261023/streamT_C356 (pot+spoon 3/5/6), streamR_ep32, streamP_ep33,
streamJ_ep3233, streamK_ep35, streamW_C36 / streamG_ep36_30, streamH_ep1, streamQ_ep25, streamI_ep30, streamV_C19,
dexagent_remaining7_20261019 (ep20 lineage), dexagent_final_recon_20261017/code (shared tools, venv `.venv/bin/python`).

## Rules
- Our meshes only (the meshes v21 ships; no gt_best_per_object, no scan-derived meshes). Poses only from policy rollouts under build v2;
  never edit submitted poses by hand. Public GT is never read.
- Slurm partition debug; distill/stage/eval `--nodelist=node01`; MuJoCo/OpenGL renders `--nodelist=node100`; jobs <= 50 min.
- NEVER touch other jobs (ur-*, ego-retarget, anything not yours from this phase). Never delete or overwrite existing outputs of earlier
  phases; write only under campaign_v22_20261028/<your stream dir> and new tags in common_v2_candidates. Never upload. No git commits.
- Log one line per milestone to campaign_v22_20261028/STATUS.md ("- 2026-10-28 HH:MM <stream> epNN: ..."); README.md in your stream dir
  with results table (tag, component path, scores vs target_final_v5, exactness), what was tried, caveats.
