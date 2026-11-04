# Stream C7 (v22, PHASE7.md): episodes 35 and 36, wooden_spoon, legacy floating-Sharpa MuJoCo (DexAgent port), 2026-10-28

Goal: spoon <= 2.4 cm / 24 deg vs `campaign_v22_20261028/target_final_v5`, MP-SR 1, exact build-v2 replay.
Starting points vs target_final_v5 (score_any.sh): ep35 best K_ep35_ilc_plain 4.63 / 36.8; ep36 v21 G_ep36_margins 3.36 / 25.1
(baseline_ep36 3.34 / 30.3).

## Results (all scored with `campaign_v22_20261028/code/score_any.sh` on the build-v2 output; scores/cand/<tag>/score.json)
| ep | tag (common_v2_candidates/) | component | vs target_final_v5 cm / deg | MP-SR | build-v2 exact | checkpoint sha256 |
|---|---|---|---|---|---|---|
| 35 | **C7_ep35_follow1** (recommended) | staging_C7_35a/components/episode_000035 | **1.71 / 18.5** (max 6.8 cm / 74 deg) | 1 | true | 0e1cf39b0d3021f2... |
| 36 | C7_ep36_follow1 | staging_C7_36a/components/episode_000036 | 1.97 / 20.3 | 1 | true | 1645374e7e34c1d8... |
| 36 | C7_ep36_follow2 | staging_C7_36b/components/episode_000036 | 1.83 / 18.3 | 1 | true | 6e2aea13ea31eaa2... |
| 36 | **C7_ep36_follow3** (recommended) | staging_C7_36c/components/episode_000036 | **1.70 / 15.8** (max 17.9 cm / 82 deg) | 1 | true | 7d48bfdd1b1d4195... |
Portable archives: staging_C7_35a/portable_archives/episode_000035_portable.tar.gz (sha256 57464065cdb76bf3...),
staging_C7_36c/portable_archives/episode_000036_portable.tar.gz (29e3ac3115230f35...). Each staging ran the frozen-build flow
(fresh re-evaluation exact -> shared component -> portable replay exact -> common dispatcher exact -> archive), then
`run_one.sbatch` under evaluator build v2 (completed.json exact_previous_physical_states_actions = true). Mesh: our
wooden_spoon c9ab88e33e5c (byte-identical to the target mesh). No videos rendered.

## 1. Reference world (code/build_world.py, code/tlibC.py)
T_new(f) = G0 @ T_target(f) @ X. X = the scorer's own registration of our submitted (budget-decoded) mesh onto the target mesh
(seed 1_000_003*ep) = identity to 1e-14 (same mesh). G0 = frame-0 anchor onto the physics start (source frame 0 = keyframe = achieved
frame 0), which is exactly the scorer's single-object frame-0 alignment, so the internal cost equals the score_any number (checked:
identity plans reproduce 4.626 / 36.77 and 3.358 / 25.12). Gt = identity (any extra rigid transform would break that identity);
rest check (target_retarget.json): ep35 end rest is 1.6-1.8 cm BELOW our tabletop, ep36 0.2-0.7 cm above.
Written: source/episode_0000XX (copy of the actor's source with object_poses.npz / wooden_spoon_trajectory.npz rewritten,
target_retarget.json with sha256s, X, G0, per-frame shift vs the previous reference: ep35 4.68 cm / 34.0 deg, ep36 2.77 / 25.7 mean),
episode/episode_0000XX (copy of the actor's adapter episode dir with the spoon columns replaced; manifest -> new source).
Target vs our previous references: ep35 target rolls the spoon about its long axis ~ -45 deg at 3.4-4.3 s, ~+150 deg at 6.8-8.4 s
(during the fast swing) and ~-75 deg at 13.8-16.8 s, and holds it 4-5 cm off in-hand; ep36 target rolls back from the half turn at
15.6-16.6 s instead of 14.6-15.0 s and has a 20 cm outlier at f300-306 (15.0-15.3 s).

## 2. Re-plan (code/planC.py + code/cemC.py = copies of streamW_C36 plan36/cem36, generalised)
Base actors kept (weights except phase weights unchanged): ep35 streamK margins/C13_true_s42 (PinkyMarginPolicy, v21 actor),
ep36 streamG margins/seat17.0_rel18.5_..._warmdonor (v21 actor). The right-wrist command is re-planned to follow the target
(rotation/translation about the executed spoon, smoothed, windowed) and folded into the actor's phase weights; every CEM sample is a
full-horizon rollout of the evaluator controller (ComponentPolicy), cost scorer-identical. Additions vs plan36:
- roll term with sign (re-time the base actor's own back-roll) and up to 3 smooth roll pulses about the spoon long axis;
- **bug fix**: the ep35 wrist hinges sit under a fixed chart rotation A (body_quat of robot_right_right_base_roll; virtual wrist
  chart), R_palm = A R_XYZ(q). plan36 composed corrections as if A = I, so on ep35 every rotation correction turned the palm about the
  wrong axes (diag: commanded 75 deg roll about the spoon axis gave [0,-49,57] deg). Now the wrist angles use A^-1 C A while positions
  use C. With the fix the pinch survives 60-75 deg wrist rolls (the phase-1/6 note "any wrist rotation drops the ep35 spoon" was
  probably this chart issue; the shared ilc_axis.py may carry the same assumption, not checked). ep36 has A = I (unaffected).
| ep | round (search/) | result (search = evaluator) |
|---|---|---|
| 36 | grid1: back-roll re-time alone 3.26 / 21.3; position follow 1.93 / 27.0; full rotation following drops | |
| 36 | c36_cem1_s11 (re-time + gr .46 gp .93 sig .36) | 1.97 / 20.3 -> C7_ep36_follow1 |
| 36 | c36_cem2_s32 (on cem1_s11) | 1.83 / 18.3 -> follow2 |
| 36 | c36_cem3_s52 (on cem2_s32) | 1.70 / 15.75 -> follow3 (stopped: 0.13 cm / 2.5 deg per round) |
| 35 | position follow only, gp 1.0 | 1.56 / 32.8 (rotation floor from the three roll excursions) |
| 35 | + pulse -75 deg 13.7-16.8 s (after the chart fix) | 1.64 / 25.9; + pulse +45 deg 6.3-8.7 s: 1.99 / 22.6; 90-150 deg at 6.3-8.7 s drop the spoon (fast swing) |
| 35 | c35_cem2_s42 (gr .32 gp .94 sig .47, pulses +62 / -64 deg) | **1.71 / 18.45** -> C7_ep35_follow1 (s41 1.80 / 20.1, s43 2.08 / 21.0) |
| 35 | round 3 on s42 (wrist re-plan / extra pulse) | no gain (1.52 / 19.6, 1.65 / 24.6) |

## Caveats
- ep35 per-window orientation error (40 frames): 6.7, 25, 23, 25, 23, 11, 19, 12, 15, 22, 26 deg. Of the +150 deg roll at 6.8-8.4 s only
  ~60 deg is followed (larger rolls during the 42 cm/s swing fling the spoon); the end rest of the target lies below our table
  (2.5-3.2 cm position in the last 2 s). Max 6.8 cm / 74 deg.
- ep36 max position error 17.9 cm comes from the target's 20 cm outlier at f300-306; f40-120 (pickup) stays at 23-27 deg.
- The hold is chaotic (same plan family spans pass to drop); every number above is a deterministic build-v2 replay.
- Margins (search_margins_k.py) were not re-fitted: the base actors keep their v21 margins; only phase weights changed.
- Phase-6 scripts are untouched; all code here is a copy (planC/cemC from streamW, stage_c7.py from streamW stage script).

## Code
tlibC.py (scorer-identical registration/errors), build_world.py, planC.py, cemC.py (+cem.sbatch, absolute --output paths),
diag_roll.py / diag_frames.py / winerr.py (diagnostics), showgrid.py, stage_c7.py (+stage.sbatch, node01).
