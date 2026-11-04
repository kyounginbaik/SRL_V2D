# Stream H13b (phase 9): eps 19, 3, 8, 1, 14, 25, 30, 16, 10, 15 re-retargeted onto target_final_v12b_g15, 2026-10-29

Target: `campaign_v22_20261028/target_final_v12b_g15` (bundle `selected20_final_v12b_g15`, stream G15). It is v10_g12 with every pose re-timed
onto the stereo/GT clock (+0.875..0.925 frame) and a clock-consistent camera path. Meshes are byte-identical. The v27 components lag
the new target by about 0.9 frame (G15 `out/eval_v27_vs_targets_v12b.txt`). The sibling stream H13a handles the other 10 episodes.

Rules followed:
- Our meshes only. Every new component's submitted meshes are byte-identical to its v27 component's, both staged and in the build-v2 output (`code/meshcheck.py`).
  This includes the ep25/ep30 A9 marker meshes (ep30 A9 sweeper ccb942f5; ep25 kept v27).
- Poses come only from build-v2 policy rollouts. Nothing was hand-edited or resampled.
- CPU work ran on node01 (debug), each job at most 50 min. Nothing was deleted, overwritten, uploaded or committed.

## Recommended (score_any vs target_final_v12b_g15; cm / deg; every tag build-v2 `exact_previous_physical_states_actions` true)

| ep | recommended | component (streamH13b/...) | v27 component: objects / MPPE / AUC vs v12b | new: objects / MPPE / AUC vs v12b | ckpt / archive sha256 (12) |
|---|---|---|---|---|---|
| 19 | **H13_ep19_a4** | staging/ep19_a4/components/episode_000019 | D9_ep19_b5: mini_sweeper 1.13/9.2, mini_dust_pan 1.03/4.5 / 1.198 / 0.9418 | mini_sweeper 0.72/7.4, mini_dust_pan 0.75/3.6 / 0.849 / 0.9701 | 1116aa10b9f8 / 7a699ffacd79 |
| 3 | **H13_ep03_a3b** | staging/ep3_a3b/components/episode_000003 | H10_ep03_H2f0: white_pot 0.48/3.7, wooden_spoon 1.27/13.8 / 1.067 / 0.9473 | white_pot 0.35/3.2, wooden_spoon 0.66/9.8 / 0.685 / 0.9741 | 824fc0573e23 / 8e00497cbaba |
| 8 | **H13_ep08_a2** | staging/ep8_a2/components/episode_000008 | F9_ep08_b4: beige_cup 0.65/2.5, white_pot 0.52/1.9 / 0.607 / 0.9834 | beige_cup 0.39/1.7, white_pot 0.32/1.6 / 0.376 / 0.9974 | 36538b1160c1 / 1f8b97195743 |
| 1 | **H13_ep01_a1** | staging/ep1_a1/components/episode_000001 | E9_ep01_b3: white_pot 0.86/3.3, white_pot_lid 0.75/5.0 / 0.871 / 0.9693 | white_pot 0.49/2.3, white_pot_lid 0.55/5.1 / 0.600 / 0.9854 | cd6571e69619 / d6b312120787 |
| 14 | **H13_ep14_t1p15** | staging/ep14_t1.15/components/episode_000014 | E9_ep14_b4: blue_cup 1.14/3.3, plastic_dish_rack 0.74/2.7 / 0.962 / 0.9642 | blue_cup 1.01/2.5, plastic_dish_rack 0.77/2.7 / 0.910 / 0.9674 | d9ee4cd6e1f1 / 5819e928b698 |
| 25 | keep v27 **D9_ep25_b4** | (v27) | D9_ep25_b4: wooden_piece_1 1.19/12.6, wooden_piece_2 1.43/4.8 / 1.508 / 0.8978 | no shift improved | - |
| 30 | **H13_ep30_t1p35** | staging/ep30_t1.35/components/episode_000030 | F9_ep30_b4: mini_sweeper 0.65/3.2, plastic_dish_rack 0.30/3.7 / 0.547 / 0.9898 | mini_sweeper 0.48/3.2, plastic_dish_rack 0.34/4.0 / 0.488 / 0.9932 | 93d25d788706 / cf490bb18736 |
| 16 | **H13_ep16_t1p1** | staging/ep16_t1.1/components/episode_000016 | E9_ep16_b8: plastic_dish_rack 0.39/2.4, wooden_spoon 1.05/6.1 / 0.794 / 0.9707 | plastic_dish_rack 0.42/2.6, wooden_spoon 0.83/5.7 / 0.702 / 0.9805 | 890c1e63294c / 91e6e8d9f768 |
| 10 | **H13_ep10_t1p0** | staging/ep10_t1.0/components/episode_000010 | X8_ep10_Blr512: blue_cup 0.46/1.8, beige_cup 0.74/2.8 / 0.624 / 0.9825 | blue_cup 0.48/1.8, beige_cup 0.53/3.0 / 0.538 / 0.9913 | 75caf435dbef / 24c18b754115 |
| 15 | keep v27 **E9_ep15_b6** | (v27) | E9_ep15_b6: plastic_dish_rack 0.66/2.9, wooden_spoon 1.05/7.4 / 0.954 / 0.9584 | no shift improved | - |

- Every recommended tag beats its v27 component vs v12b on MPPE and AUC, with MP 1 / SP 1. Every object is at most 1.01 cm / 9.8 deg.
- A few single numbers are slightly worse than v27 (the trade the global shift makes):
  - the rack in ep14 (0.74 -> 0.77 cm), ep30 (0.30/3.7 -> 0.34/4.0) and ep16 (0.39/2.4 -> 0.42/2.6);
  - ep10: blue cup 0.46 -> 0.48 cm and beige orientation 2.8 -> 3.0 deg;
  - ep1: lid orientation 5.0 -> 5.1 deg.
- ep19, ep3 and ep8 improve on every object.

Scores: `campaign_v22_20261028/scores/cand/<tag>/score.json`. Build-v2: `campaign_v16_20261023/common_v2_candidates/<tag>/episode_XXXXXX`.
Components: `streamH13b/staging/<dir>/components/episode_XXXXXX`, with the portable archive in `staging/<dir>/portable_archives/episode_XXXXXX_portable.tar.gz` (assembler layout, the same as the source streams).

All tags (including the superseded ones):

| ep | tag | objects cm/deg | MPPE | AUC | MP SP |
|---|---|---|---|---|---|
| 19 | v27 (G15_v27_ep19_vs_v12bg15) | mini_sweeper 1.13/9.2, mini_dust_pan 1.03/4.5 | 1.198 | 0.9418 | 1 1 |
| 19 | H13_ep19_a1 | mini_sweeper 0.86/8.6, mini_dust_pan 0.86/4.2 | 0.985 | 0.9613 | 1 1 |
| 19 | H13_ep19_a2 | mini_sweeper 0.76/7.7, mini_dust_pan 0.79/4.1 | 0.896 | 0.9668 | 1 1 |
| 19 | H13_ep19_a3 | mini_sweeper 0.74/7.5, mini_dust_pan 0.75/4.0 | 0.860 | 0.9691 | 1 1 |
| 19 | H13_ep19_a4 | mini_sweeper 0.72/7.4, mini_dust_pan 0.75/3.6 | 0.849 | 0.9701 | 1 1 |
| 19 | H13_ep19_t1p2 | mini_sweeper 0.85/9.0, mini_dust_pan 0.94/5.8 | 1.045 | 0.9547 | 1 1 |
| 3 | v27 (G15_v27_ep3_vs_v12bg15) | white_pot 0.48/3.7, wooden_spoon 1.27/13.8 | 1.067 | 0.9473 | 1 1 |
| 3 | H13_ep03_a1b | white_pot 0.47/3.8, wooden_spoon 0.83/10.6 | 0.821 | 0.9665 | 1 1 |
| 3 | H13_ep03_a2b | white_pot 0.42/3.9, wooden_spoon 0.69/9.9 | 0.735 | 0.9711 | 1 1 |
| 3 | H13_ep03_a3b | white_pot 0.35/3.2, wooden_spoon 0.66/9.8 | 0.685 | 0.9741 | 1 1 |
| 8 | v27 (G15_v27_ep8_vs_v12bg15) | beige_cup 0.65/2.5, white_pot 0.52/1.9 | 0.607 | 0.9834 | 1 1 |
| 8 | H13_ep08_a1 | beige_cup 0.39/1.7, white_pot 0.34/1.6 | 0.388 | 0.9974 | 1 1 |
| 8 | H13_ep08_a2 | beige_cup 0.39/1.7, white_pot 0.32/1.6 | 0.376 | 0.9974 | 1 1 |
| 8 | H13_ep08_t1p25 | beige_cup 0.53/2.8, white_pot 0.40/2.1 | 0.498 | 0.9953 | 1 1 |
| 1 | v27 (G15_v27_ep1_vs_v12bg15) | white_pot 0.86/3.3, white_pot_lid 0.75/5.0 | 0.871 | 0.9693 | 1 1 |
| 1 | H13_ep01_a1 | white_pot 0.49/2.3, white_pot_lid 0.55/5.1 | 0.600 | 0.9854 | 1 1 |
| 1 | H13_ep01_a2 | white_pot 0.45/2.2, white_pot_lid 0.59/5.2 | 0.589 | 0.9827 | 1 1 |
| 1 | H13_ep01_t0p85 | white_pot 0.75/3.1, white_pot_lid 0.63/7.1 | 0.801 | 0.9688 | 1 1 |
| 14 | v27 (G15_v27_ep14_vs_v12bg15) | blue_cup 1.14/3.3, plastic_dish_rack 0.74/2.7 | 0.962 | 0.9642 | 1 1 |
| 14 | H13_ep14_t1p15 | blue_cup 1.01/2.5, plastic_dish_rack 0.77/2.7 | 0.910 | 0.9674 | 1 1 |
| 25 | v27 (G15_v27_ep25_vs_v12bg15) | wooden_piece_1 1.19/12.6, wooden_piece_2 1.43/4.8 | 1.508 | 0.8978 | 1 1 |
| 30 | v27 (G15_v27_ep30_vs_v12bg15) | mini_sweeper 0.65/3.2, plastic_dish_rack 0.30/3.7 | 0.547 | 0.9898 | 1 1 |
| 30 | H13_ep30_t1p35 | mini_sweeper 0.48/3.2, plastic_dish_rack 0.34/4.0 | 0.488 | 0.9932 | 1 1 |
| 16 | v27 (G15_v27_ep16_vs_v12bg15) | plastic_dish_rack 0.39/2.4, wooden_spoon 1.05/6.1 | 0.794 | 0.9707 | 1 1 |
| 16 | H13_ep16_t1p1 | plastic_dish_rack 0.42/2.6, wooden_spoon 0.83/5.7 | 0.702 | 0.9805 | 1 1 |
| 10 | v27 (G15_v27_ep10_vs_v12bg15) | blue_cup 0.46/1.8, beige_cup 0.74/2.8 | 0.624 | 0.9825 | 1 1 |
| 10 | H13_ep10_t1p0 | blue_cup 0.48/1.8, beige_cup 0.53/3.0 | 0.538 | 0.9913 | 1 1 |
| 15 | v27 (G15_v27_ep15_vs_v12bg15) | plastic_dish_rack 0.66/2.9, wooden_spoon 1.05/7.4 | 0.954 | 0.9584 | 1 1 |

## Method
**World.** The physics scenes, start states and stagers are the v27 components' own (D9 ep19/25, H10 code3 ep3, E9 eps 1/14/15/16, F9 eps 8/30, X8 ep10 thin collision, sim only). Only the scoring
target was re-pointed: `code/mlibH.py` is a copy of F9 mlibF9 with TGT = target_final_v12b_g15 and eps 1/14/10 added in scorer slot order.
The frame-0 change of the new target is at most 0.23 cm (the lid and the ep14 cup also have symmetric-axis spin of 4.8 / 5.7 deg, which does not matter). The H10 precedent therefore applies, and no world rebuild was needed.
The stagers still gate on proxy MP-SR vs the old world reference, and every new actor passed it (reference_proxy_mp_sr 1.0).
`code/mresH.py` (copy of H10 code3/mres9.py with mlibH) reproduces the G15 baseline exactly at shift 0 (e.g. ep19 1.13/9.2, 1.03/4.5, MPPE 1.198).

**1. Global time-advance (`code/tadv.py`).**
- The actor is phase-RBF plus state feedback, u = phi(t/dur; centers) @ W + fb(state). Shifting every center by -s/(fps*dur) gives exactly phi(t + s/fps), so every phase command happens s frames earlier.
- Time-keyed wrapper fields are shifted by the same amount (ep3 `release_start_s`). Feedback weights, scene and start state are unchanged.
- 28 shifts were tried per episode (0-3 frames; fine 0.6-1.65): `runs/epNN/tadv*/tadv.json`.
- The outcome is chaotic in s, because grasps re-roll. ep3 loses the spoon at every shift, ep25 loses the boards, ep15 has no gain.
- Best shifts: ep19 1.2, ep8 1.25, ep14 1.15, ep30 1.35, ep16 1.1, ep10 1.0, ep1 0.85. These were baked with `tadv.py --save` (mresH.save_actor + official evaluate_policy_component), staged and verified.

**2. Window beam (`code/wbeamH.py` = H10 code3 wbeam9 + mresH + `--adv`).**
- New: fractional time-advance candidates. In each window, the wrist phase command of one or both hands is replaced by g * (c(t+s) - c(t)), with s in {0.5, 0.9, 1.3} frames, g in {0.5, 1}, full or position-only. This generalises H12 wbeam4 (integer, right hand only).
- Otherwise the settings are the source streams':
  - ep19: D9 b5 windows (0.5 s from 0.8 s), beam 5 then 4, tol 0.8/8;
  - ep3: H10 H2 windows (0.25-0.5 s from 2.0 s), hinge 4 at 2 cm, roll ILC + squeeze, tol 0.7/8. Each pass ran in two halves (2.0-8.75 s, then 8.75-15.6 s via `--init` from the first half's beam);
  - ep8 and ep1: F9 / E9 windows, tol 0.6/6.
- Chains:
  - ep19: from the v27 actor, a1 → a2 → a3 → a4;
  - ep3: from v27, a1+a1b → a2+a2b → a3+a3b;
  - ep8: from t1.25, a1 → a2;
  - ep1: from t0.85, a1 → a2.
- Each pass's final_0 was saved (`code/save.sbatch` / `svst.sbatch`), staged (`code/stageH.sbatch`, the unchanged source stagers plus H13b lineage) and verified (`code/verify.sbatch` = common_v2_candidates/run_one.sbatch body + score_any vs v12b).

## Not changed (keep v27)
- **ep25 D9_ep25_b4**: every shift loses or knocks a board (best s=1.5, MPPE 1.89 vs 1.51). This is a cheap-only episode, so it keeps v27.
- **ep15 E9_ep15_b6**: no shift improves it (best non-zero s=1.45, MPPE 0.977 vs 0.954). It keeps v27.

## Caveats
- The scores are proxies against our own reconstruction (target_final_v12b_g15), not hidden GT. The gain is real only if the hidden GT is on the b/c clock, as G15 found on public.
- The holds are chaotic, and every rollout is one deterministic outcome. Each is exact and replayable under build v2.
- ep1 a2 has a lower MPPE (0.589) but a lower AUC (0.9827 vs 0.9854) and a slightly worse lid than a1, so a1 is recommended.
- ep1 t0.85 (MPPE 0.801, AUC 0.9688) is below v27's AUC 0.9693, so it is not adopted.

## Files
- code/: mlibH.py, mresH.py, wbeamH.py, tadv.py, mkbaseH.py (E9 copy, v12b label), env.sh (v27 actor and staging paths), tadv.sbatch, res.sbatch, save.sbatch, stageH.sbatch, verify.sbatch, svst.sbatch, meshcheck.py, table.py
- runs/epNN/: tadv, tadv_fine (shift scans), save_t* (time-advanced actors), a*/ (beam passes: settings, progress, beam_w*, final_*), save_a* (saved actors + official eval)
- staging/: ep{19,8,14,30,16,10,1}_t*, ep19_a{1..4}, ep3_a{1b,2b,3b}, ep8_a{1,2}, ep1_a{1,2}
- logs/: tadv-, res-, save-, stage-, v2-, svst-<job>.log
