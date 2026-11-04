# retarget33 (2026-11-03): ep33 retargeting of our reconstruction, set up like the team's stream K33

Copy of `/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/code` (read only, untouched), paths re-pointed to this folder.
Everything here writes only under `/data1/hansol/retarget33`. Compute runs only in Slurm (node100, debug).

## What the method is (from the team doc EP32_EP33_RECON_RETARGET_20261103.md)
- Floating Sharpa hands in MuJoCo (`frozen_mujoco_rbf`), DexAgent `tool_library` hand rig (third_party/DexAgent e29bdad) via `mresH.py`.
- Start actor: v29 ep33 actor `streamH13a/runs/save_ep33_b1/actor` (H13_ep33_b1), plus a residual table (wrist pos/rot, finger flex, thumb squeeze per hand).
- `wbeamH.py` window-beam search: 0.5 s windows, beam 4, 16 candidates, every candidate a closed-loop rollout from cached physics state, scored kit-faithfully (mlibH -> public_kit_score) against `$E31_TGT`.
- `k33cwchain.sh` chains window groups as dependent Slurm jobs; `mresH.py save` bakes the table into the actor (official evaluate_policy_component).
- Not the native LLM DexAgent pipeline (run_dexagent.py stages 1-4).

## Changes vs the team copy
- All `streamK33_ep33` paths -> this folder (code, logs, tmp, targets, worlds, runs, cw, staging).
- Interpreter: `/home/hansol/v2d/.venvs/v2d-host/bin/python` (3.11) + the team venv site-packages on PYTHONPATH (the team venv python links into another user's home). Also used for the window-group helper in `k33cwchain.sh` (system python3 has no numpy).
- Not adapted yet: `vs.sh` / `score_any.sh` (write into the team's `scores/`), `svst.sbatch` build-v2 step (`run_one.sbatch` writes into the team's `common_v2_candidates`). Do not run those as is.

## Steps done
1. `targets/R33_HGA7b` = `cwtgt33.py` on `variants_20261030/recon_E14_best_B2A_S1_HGA/episode_000033` (11/03 15:21 zip, option 7b). Spoon f0 is 10.0 deg from v14 (start roll curve).
2. Check: team actor HGA_c9 replayed here vs the team target K33_cwHGAfix -> cup 0.92/3.2, spoon 0.65/6.5 (same as the team's score). Setup reproduces.
3. HGA_c9 vs R33_HGA7b: cup 1.12/9.5, spoon 3.33/16.3 (fails: f0 roll change rotates the aligned scene; the spoon start pose is fixed by the actor scene, same for every team run).
4. `worlds/R33_HGA7b` (mkref) and `runs/save_R33_HGA7b_c9w` (save in that world): no change in score (expected, scene unchanged).
5. Chain `R1_c1` (runs/cw/R1_c1): actor H13 save_ep33_b1 + table HGA_c9/g2/final_0, windows from 2.25 s, tol 0.5,6, hinge 8,1.0, 2 groups x 19 CPUs. Jobs 68771, 68772.
6. R1_c1 result (full rollout): cup 1.07/12.8, spoon 1.41/24.9. Spoon slipped in the hand at 10-11.5 s (42-67 deg), 7-8.5 s 35-40, 14-17 s 28-37.
7. Second pass (`code/r33hg.sh`, team hgburst/hgtail settings on our target, right hand only, table = R1_c1/g2/final_0):
   R2_A 6.5-9.0 s burst (0.25 s, dhinge 16,15), R2_B 9.0-12.0 s burst, R2_C 12.0-17.3 s tail (0.5 s, dhinge 8,20). Jobs 68775-68777.
   After A alone the full rollout lost the cup later (re-chained by B/C). Final R2_C/final_0: cup 1.40/7.7, spoon 1.44/12.1, MPPE 1.59 (within 1.5 cm / 15).
   Spoon by 40-frame segment (mean deg): 15, 16, 8, 7, 10, 17, 11, 12, 13; 12 frames over 30 deg. Spoon position 2.8-3.2 cm at 14.5-17 s (end placement).
8. `runs/save_R33_HGA7b_R2`: table baked into the actor, official evaluator, physics_verified, same score. Video `videos/R2/ep33_R2_vs_HGA_c9.mp4`.
Next: packaging (stage + build-v2 + splice into a full parquet) still points into team folders; adapt before use.
9. Packaging (11/03, all writes under this folder):
   - `code/stage_a_r33.sh` = team stage_a.sh with our interpreter; three lineage files under /home/exaFLOPs (actor_res.py, release_cem.py, multi_cem.py) are
     not readable for us and were dropped from the provenance list only (no effect on the rollout).
   - `code/r33run_one.sh` = team run_one.sbatch body (build-v2 common_evaluator, backend legacy), output -> `candidates/<TAG>`.
   - `code/r33svst.sbatch NAME SAVE_DIR` = stage + g18swap + build-v2 (save done before with mresH save). R2: job 68799, exact_previous_physical_states_actions True, 696 rows.
   - `code/r33pkg.py CAND_TAG LABEL` = FINAL_20261104 (0fca055b) with only ep33 rows from the candidate (k33splice/v32build rules), verify, kit OFFLINE validation
     (validate_submission + CD-O derivation). The kit's --dry-run is NOT used: it logs in to Kaggle (preflight) with the user's credentials.
   - Package `packages/track3_FINAL_ep33R33R2_20261103/all_20_episodes`: parquet sha a2c6131a..., only e000033 rows, columns pose + checkpoint_sha256
     (mesh cell kept, mesh byte-identical to K33_H5r16_c3). Kit score of its ep33 rows vs our 7b target: cup 1.40/7.7, spoon 1.43/12.1, MPPE 1.593.
   - Not uploaded. Upload is the user's.
