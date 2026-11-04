# Stream H12 (phase 9): ep32 re-retargeted onto target_final_v10_g12, 2026-10-29

Target: `campaign_v22_20261028/target_final_v10_g12` (bundle `reconstruction_recovery_20261026/selected20_final_v10_g12`, stream G12).
- Compared with v8_g10b only `episode_000032/wooden_spoon_trajectory.npz` differs (cup track and both meshes byte-identical).
- The G12 consensus roll (relative to final_v6, about the spoon long axis): 35-43 deg already at f24-44, ramp to 137-233 deg over f60-80,
  fast swing back to 78 deg by f92, 95-160 deg f96-125, up to 246 deg f140-146, back to 74 deg by f160, -22/-38 deg at f180-184, about 15-38 deg to f224.

Rules followed: our meshes only (cup 3c3c5328, spoon c9ab88e3, byte-identical to B9_ep32_s6f0 / H9 / H10); poses only from build-v2 policy rollouts;
CPU on node01 (debug), every job <= 50 min; nothing overwritten, uploaded or committed.

## Results (score_any vs target_final_v10_g12; all build-v2 completed.json exact_previous_physical_states_actions = true)

| tag | component (streamH12_ep32/...) | cup cm/deg | spoon cm/deg | MPPE | AUC | MP / SP |
|---|---|---|---|---|---|---|
| **H12_ep32_e1f0** (recommended) | staging/ep32_e1f0/components/episode_000032 | 0.64/2.9 | **0.93/19.8** | 1.135 | 0.955 | 1 / 1 |
| H12_ep32_g3f0 | staging/ep32_g3f0/components/episode_000032 | 0.73/4.0 | 1.03/19.1 | 1.197 | 0.940 | 1 / 1 |
| H12_ep32_c1f0 | staging/ep32_c1f0/components/episode_000032 | 0.49/3.1 | 0.99/21.0 | 1.127 | 0.950 | 1 / 1 |
| H12_ep32_b3f0 | staging/ep32_b3f0/components/episode_000032 | 0.70/3.7 | 1.03/22.8 | 1.286 | 0.941 | 1 / 1 |
| H10_ep32_H2f0 (v26, before) | streamH10_v8/staging/ep32_H2f0 | 0.61/3.0 | 0.97/36.7 | 1.56 | 0.942 | 0 / 1 |
| H9_ep32_h19f0 (v25, before) | streamH9_ep32v7/staging/ep32_h19f0 | 0.55/2.4 | 0.73/38.9 | 1.51 | 0.951 | 0 / 1 |

- Build-v2 outputs: `campaign_v16_20261023/common_v2_candidates/H12_ep32_<x>/episode_000032`; scores `campaign_v22_20261028/scores/cand/H12_ep32_<x>/score.json`.
- e1f0: checkpoint sha256 de7f3a41253c..., portable archive 94cc24d9b4c7...; g3f0: checkpoint 21740335a961..., archive e08802a991b9....
- Why e1f0 over g3f0: g3f0 is 0.7 deg better on the spoon but worse on everything else (spoon 1.03 cm > 1.0, cup 4.0 deg, MPPE 1.20 vs 1.135, AUC 0.940 vs 0.955); 0.7 deg is well inside the hidden-noise band.
- Portable archives: `staging/ep32_<x>/portable_archives/episode_000032_portable.tar.gz` (assembler layout = legacy_component.parents[1]/portable_archives).
- Hedge for e1f0 (if v10 is wrong): vs target_final_v8_g10b spoon 0.93/31.6 MP 0 SP 1 (MPPE 1.44); vs target_final_v6 spoon 0.93/60.0 MP 0 SP 0.

Goal status (e1f0): MP pass vs v10 (19.8 < 30 deg); cup 0.64 cm / 2.9 deg met; spoon position 0.93 cm met; spoon never lost.
**Not met: spoon orientation 19.8 deg (goal <= 10, worst-case 15) and MPPE 1.135 (goal 1.1).**
G12's margin estimate: a rollout within ~15-20 deg of the consensus gives an expected hidden spoon error of about 22-27 deg, i.e. a thin MP margin.

Rounds 3, 5 and 6 gave no clear gain (several branches lost the spoon or knocked the cup), so the search was stopped at the plateau (about 19-20 deg).

## Where the remaining error is (code32/diag.py: twist about the spoon axis vs swing, per frame; runs/diag_e1.txt)
| frames | target roll | e1f0 mean deg | share of the 19.8 mean |
|---|---|---|---|
| f0-20 | 0 | 2.5 | 0.2 |
| f20-40 | 3 -> 35-43 in 4 frames (spoon still near the table) | 21.4 | 1.7 |
| f40-60 | 41 -> 96 | 29.4 | 2.3 |
| **f60-100** | 137 -> 233 -> 78 (about 150 deg back in 0.6 s) | **50.1** | **7.7** |
| f100-125 | 96-160 | 3.2 | 0.3 |
| f125-165 | 160 -> 246 -> 74 | 32.4 | 5.0 |
| f165-190 | 74-89 -> -38 -> 38 (110 deg in 4 frames) | 15.1 | 1.5 |
| f190-259 | 38 -> 0 | 4.4 | 1.2 |

The error is almost all twist about the long axis in the fast swings: the held spoon spins / slips (torsional slip, as H9/H10 found), the wrist torque saturates at 5 Nm,
and the spoon briefly slides 4-7 cm at f88, f136-139 and f153-156. Slow segments are tracked within 1-5 deg.

## Method
- **World W12** (`code32/build_world_b.py` = H10 copy, target v10_g12, `code32/w12.sh`): scene `W12/scene/scene_physics.xml` sha 36779927... byte-identical to W8/W7/W6;
  X, tilt 0.469 deg, lift 0.117 cm, cup yaw fix 21.14 deg unchanged (frame 0 unchanged). B9_ep32_s6f0 ported as `runs/port_s6f0` (vs v10: cup 0.52/2.9, spoon 0.85/67.1 = G12's v24 value).
- **Scan** (`code32/scan.py`, runs/scan1.json): all 113 H9/H10 beam finals re-scored vs v10 on the port; best H10 H2 36.7, F1 37.1, F2 37.4, h19 38.9.
- **Search**: `code32/wbeam4.py` (H10 copy: window beam from exact cached MuJoCo states, ILC / roll-only twist ILC / squeeze / time-advance / left-cup ILC candidates,
  per-frame cost pos/0.8 + deg/8 + 0.5 cup + grasp-loss hinge `--hinge 4,2.0`). Windows from 1.0 s (the target already rolls at f24).
- Chain (full-horizon result vs v10, cup / spoon):

| round | runs (init) | key settings | best |
|---|---|---|---|
| 1 | b1 (mix6), b2 (mix6), b3 (H10 H2), b4 (H9 h19) | b1/b3/b4 0.25 s to 9.5 s + 0.5 s tail; b2 0.5 s; look 1.0-1.5 | **b3 0.70/3.7, 1.03/22.8**; b4 1.42/28.9; b2 1.02/30.6; b1 lost spoon |
| 2 | c1-c4 (b3 beam) | c1 0.25 s look 1.0 roll alpha <= 2.0; c2 0.2 s; c3 tol deg 6 look 1.4; c4 rot-std 0.15 | **c1 0.49/3.1, 0.99/21.0**; c4 22.2; c2 22.5; c3 1.55/20.3 (cup 5.4 deg) |
| 3 | d1-d4 (c1, c1+c3+c4) | 0.15-0.2 s windows only on the transients, look 0.6-1.0 | no gain: d2 21.4; d1 cup knocked; d3/d4 lost the spoon (short look) |
| 4 | e1-e4 (c1, c1+d2) | e1 = c1 + time-advance 1,2,4,6 frames (g 0.5/1.0) + squeeze 0.4 | **e1 0.64/2.9, 0.93/19.8**; e4 (0.5 s offset) 25.8; e2/e3 lost the spoon |
| 5 | f1-f3 (e1) | e1 settings, more advance / roll / squeeze | no gain: f1 22.0; f2/f3 lost the spoon |
| 6 | g1-g4 (e1, e1+c1, c1) | e1 settings, new seeds, g2 hinge 5,1.5 | g3 0.73/4.0, 1.03/19.1; g2 cup knocked (24 deg); g1/g4 lost the spoon |

- Save / stage / build-v2 / score: `code32/svst.sh TABLE NAME` (mres.py save on the port -> `stage_b.py` (H10 copy; world W12, previous component B9_ep32_s6f0,
  lineage `episode_target_final_v10_g12_lineage.json`) -> `run_one.sbatch` body -> `code/score_any.sh 32 ... target_final_v10_g12`).

## Files
- code32/: build_world_b.py, blib.py, mlib.py, mres.py, wbeam4.py (H10 copies retargeted to v10 / W12), scan.py, diag.py, port.py, save_chain.sh, stage_b.py, stage_b.sbatch, st_v2.sh, svst.sh, res.sbatch, w12.sh, env.sh.
- W12/ (world); runs/ (port_s6f0, b1-b4, c1-c4, d1-d4, e1-e4, f1-f3, g1-g4, save_*, init_*.npz, win_*.txt, scan*.json, diag_*.txt); staging/; logs/.
