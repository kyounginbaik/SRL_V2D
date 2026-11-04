# Stream V8 (phase 8, v23): episode 32 (blue_cup + wooden_spoon) vs target_final_v5

Goal: <= 1.3 cm and <= 15 deg per object. **Met.** All tags are in campaign_v16_20261023/common_v2_candidates/ and are build-v2 exact
(`exact_previous_physical_states_actions` true), MP-SR 1. Scores come from score_any.sh. Meshes are unchanged (cup 3c3c5328, spoon c9ab88e3).
Nothing was uploaded or committed.

| tag | component (staging/.../components/episode_000032) | cup cm/deg | spoon cm/deg | MPPE / AUC | checkpoint / archive sha256 |
|---|---|---|---|---|---|
| **V8_ep32_wb4r2** (recommended) | staging/ep32_wb4r2 | **0.62 / 5.0** | **1.02 / 7.2** | 0.95 / 0.955 | see staged.json |
| V8_ep32_wb5 | staging/ep32_w2f5 | 0.92 / 5.7 | 1.16 / 7.9 | 1.14 / 0.934 | 6319875fd1a48dea / 5aace282a32c6a9e |
| V8_ep32_wb4 | staging/ep32_w2f4 | 0.92 / 5.7 | 1.17 / 7.8 | 1.14 / 0.934 | dbf9bddb7f92d709 / 3e15bd2d3e15bb2f |
| M8_ep32_late2 (previous) | streamM8 staging/ep32_late2 | 0.89 / 7.5 | 1.92 / 16.5 | 1.63 / 0.885 | |

## Diagnosis (before the search)
- The spoon error of late2 was in 4.5-9.2 s: 3-4.4 cm, and at 6.5-8 s a 52-71 deg roll about the long axis. In that window the target rolls the spoon
  by +-50-100 deg in 0.25 s. The jitter floor (gaussian sigma 2 frames) is only 0.66 cm / 7.4 deg, so the target is followable in principle.
- The hold is a weak fingertip/palm grasp. It is index_DP + ring_DP + thumb_DP + palm at 3-4 s. After the bowl enters the cup at 3.75-4.5 s it degrades to
  palm + thumb_DP only (6.5-9 s). The target spoon bowl dips INTO the cup at 3.75-4.5 and 7-7.75 s, and the target meshes intersect by up to 16 mm,
  so spoon-cup contact is part of the task.
- Robustness test (`code/rob.py`, 12 modifications x 16 1-mm perturbations): cup-avoid lift 1-3 cm and finger/thumb squeeze -0.1..+0.35 rad all
  give a median spoon error of 6-12 cm, and 0% of runs stay under 3 cm. No firmer-grasp edit makes the hold robust.
- The actor's feedback only sees robot joints (qadr of actuated joints), never the object. DART-style noise distillation therefore cannot learn to
  stabilise the spoon (route c is weak by construction).

## What worked: window beam search with exact cached physics states (`code/wbeam.py`, `code/wbeam2.py`)
- The episode from 2.4 s is cut into 0.5 s windows. For each window and each of B beam parents (phase-residual tables), children are N random
  right-wrist/finger knots plus rigid-hold ILC steps toward the target (right hand, spoon; in v2 also left-hand axis-only cup ILC).
- Each child is rolled out from the parent's exact physics state at the cache frame (mj_getState/mj_setState mjSTATE_INTEGRATION). It runs only to
  window end + a 1-1.5 s look-ahead. RBF leak before the cache frame is checked to be exactly 0. An unedited child must equal its parent bit for bit (checked).
- Children are ranked by the CUMULATIVE scorer-faithful per-frame cost over [0, window end + look]. The best B (max 2-3 per parent) form the next beam.
  This is the same idea as Q8's wlot, with a beam and cumulative ranking instead of a greedy step.
- Chain for the recommended tag:
  1. wb2 (beam 10, 2.4-9.4 s) on late2;
  2. late left cup ILC 9.3-12.9 and late right spoon ILC 9.0-12.9 (M8 mres.py): 1.17/7.8, 0.92/5.7 = V8_ep32_wb4;
  3. wb4 = a second wbeam2 round over 2.4-12.9 s from wb2 final_4 (tol 1.1/18, cup weight 1.0, left cup ILC candidates): 1.02/7.2, 0.62/5.0.
- Rollouts are bit-identical on node200 (Intel) and node01 (checked on a chaotic modified rollout), so searches ran on node200. Saving, staging and
  build-v2 ran on node01.

## Caveats
- Proxy scores are against our reconstruction (target_final_v5), not hidden GT.
- The hold is still chaotic: the result is a specific trajectory, which is exact under build v2 but not robust to edits.
- V8_ep32_wb4/wb5 have cup 0.92 > late2's 0.89 (deg 5.7 < 7.5). wb4r2 is better than late2 on every object and metric.

## Files
code/: rob.py (robustness test), diag.py, contacts2.py, tgtpen.py, jitter.py, inspect_tb.py, wbeam.py, wbeam2.py, sumwb.py, res.sbatch, stage_b.sbatch.
runs/: rob1, wb1-wb4, w2f{4,5}_cupl/_spl (ILC), save_* (actors with official evaluation). logs/.
