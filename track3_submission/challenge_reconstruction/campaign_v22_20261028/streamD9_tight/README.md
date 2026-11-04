# Stream D9 (phase 9): tighten episodes 3, 6, 19 and 25 against target_final_v6, 2026-10-28

Goal: every object <= 0.8 cm and <= 8 deg (mean), MPPE <= 0.9 and AUC >= 0.95 per episode, never worse than the current component.
Scores come from `code/score_any.sh EP COMPONENT LABEL target_final_v6` (`../scores/cand/<tag>/score.json`).
Every tag is in campaign_v16_20261023/common_v2_candidates/ and is build-v2 exact (`completed.json` exact_previous_physical_states_actions true).

- Poses come only from build-v2 policy rollouts.
- Meshes: ours only, and byte-identical to the current component's (ep25: to A9_ep25_boards).
- Nothing was uploaded, committed, deleted or overwritten.

## Results (cm / deg; vs target_final_v6)

| ep | tag (recommended in bold) | component = staging/<dir>/components/episode_XXXXXX | object 1 | object 2 | MPPE | AUC | exact | previous component |
|---|---|---|---|---|---|---|---|---|
| 3 | **D9_ep03_b3** | staging/ep3_b3 | pot **0.43 / 3.8** | spoon **0.72 / 5.7** | **0.66** | **0.983** | true | Q8_ep03_g11: 0.85 / 6.3, 1.13 / 9.7, 1.13, 0.941 |
| 6 | **D9_ep06_b2** | staging/ep6_b2 | pot **0.60 / 2.7** | spoon **0.77 / 7.5** | **0.79** | **0.977** | true | L8_ep06_slate: 0.64 / 2.8, 1.24 / 11.9, 1.09, 0.949 |
| 19 | **D9_ep19_b5** | staging/ep19_b5 | sweeper 0.82 / 8.5 | dust pan **0.74 / 3.9** | 0.91 | **0.966** | true | T8_ep19_c2: 1.28 / 13.7, 1.12 / 9.1, 1.45, 0.924 |
| 25 | **D9_ep25_b4** | staging/ep25_b4 | piece_1 1.10 / 12.5 | piece_2 1.19 / 4.7 | 1.35 | 0.913 | true | A9_ep25_boards: 1.31 / 14.6, 1.42 / 5.7, 1.60, 0.884 |
| 25 | D9_ep25_b2 (alt) | staging/ep25_b2 | piece_1 1.17 / 13.0 | piece_2 1.12 / 4.8 | 1.36 | 0.906 | true | |

- Every recommended tag is better than its previous component on every object (cm and deg), on MPPE and on AUC. MP-SR 1, SP-SR 1.
- **ep3 and ep6 meet every target.** ep19 misses by 0.02 cm / 0.5 deg on the sweeper and 0.01 on MPPE (plateau, see below). ep25 is the honest plateau.
- ep25 b4 against b2: b4 is better on piece_1, MPPE and AUC, and 0.07 cm worse on piece_2 position.
- Checkpoint / archive sha256 (first 12): ep3 6161aabc2673 / b46c4ea2e51a; ep6 6afcb1ea4c21 / f6bb46db36ad; ep19 b329c2a0f359 / 8f378471c0d2; ep25_b4 047850ce22b5 / f19303281284.
- Submitted meshes (sha256, first 8; build-v2 completed.json):
  - ep3 and ep6: pot 8cbd3543, spoon c9ab88e3;
  - ep19: sweeper cf7bf340, dust pan 1d298c01;
  - ep25: piece_1 ecdfc14c, piece_2 a2d89be1. These equal A9_ep25_boards' files (cmp), and the portable assets and source are identical to A9's (sha256 of every file).

## Method: two-hand window beam search (code/wbeam9.py)
This generalises streamV8 wbeam2.py / streamU8 wbeam_u8.py to any episode and both hands.

**Shared code.** `mres9.py` / `mlib9.py` are U8 mres / M8 mlib with the target switched to target_final_v6 and eps 3/6/19/25 added.
- mres9.Env reproduces all four current components exactly: same objects, same cm/deg as score_any.
- Every saved actor's official evaluate_policy_component reproduces the search value exactly.

**Search.** The episode is cut into 0.5 s windows. For each window and each beam parent (a phase-weight residual table on the shipped actor):
- children are 40 random knots split over both hands, plus windowed rigid-hold ILC steps for EACH object on its own hand (full/pos/rot/rotanc; sigma 0.1/0.25 s; alpha 0.15-1);
- the hands are: ep3/6 spoon right, pot left; ep19 sweeper right, dust pan left; ep25 piece_1 left, piece_2 right;
- each child is rolled from the parent's exact cached MuJoCo state; leaks before the cache frame are skipped, and an unedited child must equal its parent bit for bit (checked);
- children are ranked by the cumulative per-frame cost sum_o (pos/0.8 + deg/8) + MPPE_frame/0.8 over [0, window end + look];
- beam 4 (2 per parent), or 5 with 60 children in the late ep19 passes.

**Chaining.** Long episodes were run in two halves; each half starts from the best final table of the previous one. Repeated full passes then started from the previous pass's best.
- ep3: b1 (1.8-9.8 s) → b2 (9.8-15.6) → b3 (full).
- ep6: b1 → b2.
- ep19: b1 → b2 → b3 → b4 → b5. Pass b6 got worse (cum 1102 vs 1090), which is where it plateaus.
- ep25:
  - b1L, 1.0-9.5 s with look **3 s**. The 1 s look of b1 removed the board-2 early push at 6.5-7 s and lost the later grasp, ending at 9.7 cm (J8 found the same dependency).
  - → b2 (9.5-17.75 s, look 2) → b3 (first half again) → b4 (second half again).
  - Side branch b3x: left-hand rotation-weighted, 2.2-7 s. Not better.

**Saving, staging and verification.**
- `save.sbatch` bakes the table and runs the official eval on node01.
- `stage.sbatch` uses the unchanged stage scripts of each episode's world plus D9 lineage: Q8/L8 stage_compose_component (eps 3/6), J8 stage_ep19J (ep19, world_a40, teacher cem/a40_kA_LR_2_5.5), stream-Q stage_seven_component_corrected_boards (ep25).
- For ep25, `mkbase25.py` builds the staging base from J8's b48m4 base. `d9_meshswap25.py` then copies A9_ep25_boards' swapped meshes byte-identically and asserts that the pre-swap meshes equal A9's source meshes (same body frames). It updates readiness.json the way A9 stage.py does.
- Build-v2: run_one.sbatch on node01. Scoring: score_any.sh vs target_final_v6.
- Searches ran on node01 and node200. Every saved or staged actor was re-evaluated on node01 and matched exactly.

## Remaining error (per 0.5 s window, eval of the saved actors)
- **ep19 sweeper:**
  - 4.5-5.0 s is 72 deg, the target sweeper face flip (f88-100, T8). That alone is about 3.1 deg of the 8.5 deg mean.
  - 11.0-11.75 s is 2-2.9 cm: the hand is already on the table and the target sweeper is 3.6 cm lower (T8).
  - The dust pan at rest (0-1 s) is 1.1 cm, the world_a40 rest floor.
- **ep25:**
  - Both boards sit about 0.9 cm off at rest (0-1 s), a static world/target floor.
  - Board 2 is 4-4.4 cm off at 7-8.5 s (the early pinky push that the later grasp depends on).
  - Board 1 lift timing at 2.5-3 s (4.8 cm).
  - Board 1 in-plane hold rotation of 20-30 deg at 4.5-6.5 s. Left rot/rotanc ILC and the rotation-weighted branch did not reduce it.
  - Two more passes moved the whole-episode cost by only 1.4 %.

## Caveats
- Scores are proxies against our own reconstruction (target_final_v6), not hidden GT.
- The holds are chaotic. Each final rollout is one deterministic outcome of a deterministic search. It is exact and replayable under build v2, but not robust to upstream edits.
- These are phase-weight residuals only. Scene, start state, feedback weights and wrappers are unchanged.

## Files
- **code/**:
  - searches: wbeam9.py, mres9.py, mlib9.py;
  - jobs: res.sbatch, base.sbatch, save.sbatch, stage.sbatch, savestage.sbatch;
  - ep25 staging: mkbase25.py, d9_meshswap25.py;
  - helpers: env.sh (actor and mesh paths), wins.py, sumlog.sh.
- **runs/ep{3,6,19,25}/**: base_eval, b*/ (progress.json, final_*.npz, final.json), save_*/actor, eval_*.
- **staging/**: ep3_b3, ep6_b2, ep19_b5, ep25_b2, ep25_b4. **logs/**: res-, save-, svst-, stage-<job>.log.
