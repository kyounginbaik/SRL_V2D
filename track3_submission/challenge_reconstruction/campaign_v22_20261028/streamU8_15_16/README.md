# Stream U8 (phase 8, v23): episodes 16 and 15 vs target_final_v5, 2026-10-28

Goal: at most 1.3 cm and 15 deg per object (aim for 14.5 deg on the ep16 spoon), build-v2 exact.

- Meshes are unchanged (the v22b component meshes).
- Every component is the shipped actor plus a phase-weight residual table. ep15 also adds a hold primitive (see below).
- The scene, start state and feedback weights are unchanged.
- Nothing was uploaded, deleted or committed.

## Results
Scores are from score_any.sh on the build-v2 output (`../scores/cand/<tag>/score.json`), in cm / deg. All tags are in campaign_v16_20261023/common_v2_candidates/.

| ep | tag | component | rack | spoon | MPPE / AUC | exact | before |
|---|---|---|---|---|---|---|---|
| 16 | **U8_ep16_b** (best spoon) | staging/ep16_b/components/episode_000016 | 0.68 / 3.6 | **1.07 / 6.2** | 0.95 / 0.959 | true | M8_ep16_m: 0.37/2.3, 1.296/15.06 (1.05 / 0.922) |
| 16 | U8_ep16_a | staging/ep16_a/components/episode_000016 | 0.50 / 2.5 | 1.27 / 14.3 | 1.08 / 0.923 | true | |
| 15 | **U8_ep15_c** (best spoon) | staging/ep15_c/components/episode_000015 | 0.68 / 2.8 | **0.94 / 7.1** | 0.90 / 0.962 | true | v21 S6_ep15_newtrack_rel50p2: 0.60/2.3, 1.66/12.6 |
| 15 | U8_ep15_b | staging/ep15_b/components/episode_000015 | 0.65 / 2.7 | 1.20 / 8.4 | 1.00 / 0.954 | true | |
| 15 | U8_ep15_a | staging/ep15_a/components/episode_000015 | 0.77 / 2.8 | 1.11 / 8.3 | 1.03 / 0.952 | true | |

Every listed tag is build-v2 exact, MP-SR 1, and meets 1.3 / 15. The rows above are all verified and scored with score_any.

**No U8 tag beats the previous component on every object.** In every tag the rack is worse:
- ep16 rack: 0.50-0.68 cm and 2.5-3.6 deg, against 0.37 / 2.3.
- ep15 rack: 0.65-0.77 cm and 2.7-2.8 deg, against 0.60 / 2.3.

The cause: the spoon is now steered while it sits in the rack, and the hand on the spoon pushes the rack.

Two rack-weighted beam rounds were saved as actors (official evaluation) but were not staged before the stop:
- ep16: runs/ep16/save_beam_r2rack0/actor, rack 0.45/2.67, spoon 1.05/6.8;
- ep15: runs/ep15/save_beam_r3rack0/actor, rack 0.545/2.54, spoon 0.99/7.1.

**Window beam search** (code/wbeam_u8.py, a port of streamV8 wbeam2.py to any episode, with the U8 hold primitive and rotanc ILC):
- 0.5-0.6 s windows from 1.4 s to the end, beam 4.
- Children are 40 random right-hand knots plus rigid-hold ILC steps (full / pos / rot / rotanc).
- Each child is rolled out from the parent's exact cached physics state, with prefix-exactness checks, and ranked by cumulative cost from frame 0.
- About 6 min per run.
- U8_ep16_b = U8_ep16_a's table + beam (tol 1.3 cm / 10 deg).
- U8_ep15_c = U8_ep15_a's table + beam.
- The rack-weighted rounds (cup-w 2) cut the rack back with almost no spoon loss. A further round (cup-w 4) was cancelled when main stopped the searches.

## ep16: the seat tilt
- **Diagnosis** (code/tilt.py, code/feas.py on M8_ep16_m):
  - From 6.5 to 8.6 s the spoon is 20-22 deg off. The spoon-frame error rotvec is about (3, -12, 16) deg, a systematic tilt.
  - The target seated pose penetrates our rack by 1.3-2.1 cm. Even so, a rack-feasible pose exists within about 0.9 cm / 6 deg (random search over non-penetrating poses). So the tilt is not forced by the geometry.
- **What worked**, in order:
  1. One rigid-hold ILC step that rotates the right wrist about the spoon's registered anchor (new mres mode `rotanc`; alpha 0.1, sigma 0.3 s, 6.5-8.6 s): spoon 1.30/15.06 → 1.34/14.55.
  2. Position ILC in the same window, with the rack cap relaxed to 0.50: 1.286/14.42.
  3. A right-hand knot CEM over 3.6-6.8 s with the goal hinge: 1.2685/14.31. This is U8_ep16_a.
- **What did not help:**
  - Plain rot/rotanc ILC with the v22b rack cap 0.372: every rotation pushes the rack, so nothing was accepted.
  - Further CEMs (late with rack weight, early): at most 1.25/14.24.
  - Q8-style greedy windows (code/wwin.py): they drive the spoon position down to 0.93 cm but the angle up to 17 deg. Deg-weighted windows gave 1.20/14.16 with the rack at 0.69, or 1.22/14.33 with the rack at 0.57. None was clearly better than U8_ep16_a.
- **The left hand** never touches anything in ep16. A left-hand nudge was not needed.

## ep15: the floor, and how it was beaten
1. **Frame-window error budget** (code/tilt.py, feas.py, settle.py on S6):
   - **0-1.6 s pre-grasp** (about 1.6 cm): the target spoon lies about 1.2 cm below our table plane relative to the rack-anchored frame, and 1 cm off in x. The non-penetrating floor (table + rack) is about 1.0-1.5 cm. Re-building the world would only remove the x part (about 0.06 cm on the mean), so the world was **not** re-built.
   - **1.6-5.5 s carry/seat** (1.2-2.3 cm): the floor is about 0.1 cm. This is in-hand tracking error.
   - **5.5-8.3 s released rest** (1.6-1.9 cm, mostly +y): the target rest penetrates the rack by 1.5-2.1 cm. A released spoon cannot rest there:
     - In 300 seeded drops near the target (settle.py, k = 115), the settled error was at least 1.56 cm (median 2.5).
     - In M8's 1,500-draw lottery, the rest-phase error was at least 1.58.
     - **This proves a floor of about 1.56 cm over 30 % of the frames for any release design.**
2. **Place-offset grid and release primitive:**
   - Constant wrist offsets of ±2.5 cm over 2.6-5.6 s (gridoff.py, 151 rollouts): none better; the spoon funnels into the same slot.
   - The rcem release primitive: no gain.
3. **What worked: a hold primitive plus greedy window ILC.**
   - **Hold primitive** (mres `--freeze right,4.0,0.3,4.6`): the right fingers' phase weights after 4.0 s, and the wrist's after 4.6 s, are frozen to their own commanded value at that time. The hand keeps the spoon seated instead of releasing it. Alone this gives 1.80/12.2, with the rack 0.56.
   - **Greedy window search** (code/wwin.py, a port of Q8 wlot `--ilc` onto mres.Env):
     - Windows run from 1.4 to 8.25 s, 0.6-0.8 s each.
     - Each window tries 60 random knots plus 72 rigid-hold ILC steps (pos/full/rotanc; sigma 0.1-0.4 s; lead 0-1; alpha 0.15-1).
     - Candidates are scored on [w0, w1 + 0.5 s] only. The best is kept and the prefix is locked.
   - Result: 1.11/8.3 (U8_ep15_a). The same search without the hold reached 1.37/9.1, so the hold is what removes the rest floor.
4. **Also tried, with no or partial gain:**
   - Whole-episode ILC (chaotic; alpha collapses).
   - Knot CEMs with a relaxed rack cap: 1.53-1.58. M8's ep15 searches had used a rack cap (0.600/2.30) below the baseline itself (0.601/2.303), so every candidate was penalised.
   - Early holds (fingers and wrist both frozen at 4.0-4.4 s): the spoon is left 2.5-3 cm high and ILC cannot lower it without hitting the rack.

## Code (code/)
- **mres.py**: M8's copy plus three additions:
  - ILC mode `rotanc` (rotation about the object anchor);
  - `--ilc-modes`;
  - `--freeze side,t_h[,ramp[,t_wrist]]` (hold primitive baked into the phase weights).
- **wwin.py**: the greedy windowed search. **gridoff.py**: the place-offset grid. **settle.py**: the released-rest floor test.
- **feas.py**: the nearest non-penetrating pose to the target. **tilt.py**: error decomposition plus target-pose penetration.
- **chain16.sh**: the rotation → position ILC chain. **rcem.py**: M8's copy plus `--P/--D/--goal`.
- **Staging**: stage15/16.sbatch, the unchanged stagers.
- Runs are in runs/ep15 and runs/ep16. Logs are in logs/res-<job>.log.

## Caveats
- Scores are proxies against our reconstruction (target_final_v5), not hidden GT.
- The ep15 hold primitive changes the task: the right hand keeps holding the spoon in the rack until the end, as ep16 already does, instead of letting go. The policy and evaluator are unchanged; only the phase weights differ.
- The final ep15 rollout is one deterministic outcome of a chaotic search, as with Q8. It is exact and replayable under build-v2.
- The rack is worse than the previous components in both episodes (see above).
