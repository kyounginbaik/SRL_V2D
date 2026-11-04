# Stream H13a (phase 9): eps 20, 5, 35, 37, 32, 38, 28, 36, 33, 6 re-retargeted onto target_final_v12b_g15, 2026-10-29

Target: `campaign_v22_20261028/target_final_v12b_g15`. This is bundle `selected20_final_v12b_g15` from stream G15. Its meshes are byte-identical to
v10_g12. Its poses are v10_g12 retimed onto the stereo/GT clock (+0.875-0.925 frame; eval20 steps 0.925 -> 1.85 frames at f53), plus the
clock-consistent camera path. The sibling stream H13b handles the other 10 episodes.

Rules followed:
- Our meshes only. The submitted meshes are byte-identical (cmp) to each v27 component's meshes.
- Poses come only from build-v2 policy rollouts.
- CPU jobs ran on node01 (debug), each <= 50 min.
- Nothing was overwritten, deleted, uploaded or committed.

## Results (score_any vs target_final_v12b_g15; cm / deg; all build-v2 `completed.json` exact_previous_physical_states_actions = true)

| ep | recommended tag | component (streamH13a/...) | objects | MPPE | AUC | v27 component vs v12b (before) |
|---|---|---|---|---|---|---|
| 20 | **H13_ep20_b2** | staging/ep20_b2/components/episode_000020 | cup 0.36/1.3, pitcher 0.51/1.8 | **0.448** | 0.9944 | F9_ep20_b4: 0.82/1.6, 1.22/3.3; 1.036 / 0.9599 |
| 5 | **H13_ep05_b2** | staging/ep5_b2/components/episode_000005 | pot 0.54/4.5, spoon 0.80/4.8 | **0.757** | 0.9761 | C9_ep05_beam6: 1.14/4.9, 1.50/7.2; 1.403 / 0.9151 |
| 35 | **H13_ep35_b2** | staging/ep35_b2/components/episode_000035 | spoon 0.65/6.9 | **0.826** | 0.9624 | C9_ep35_beam6: 1.27/7.9; 1.390 / 0.9195 |
| 37 | **H13_ep37_b1** | staging/ep37_b1/components/episode_000037 | spoon 0.33/1.7 | **0.352** | 1.0000 | K8_ep37_ilc1: 0.69/3.4; 0.733 / 0.9776 |
| 32 | **keep v27 H12_ep32_e1f0** | streamH12_ep32/staging/ep32_e1f0 | cup 1.05/3.9, spoon 1.31/22.4 | 1.548 | 0.9228 | (unchanged) |
| 38 | **H13_ep38_b2** | staging/ep38_b2/components/episode_000038 | dust pan 0.38/3.2 | **0.457** | 0.9974 | F9_ep38_b3: 0.76/4.2; 0.841 / 0.9767 |
| 28 | **H13_ep28_b1** | staging/ep28_b1/components/episode_000028 | rack 0.63/1.7, pot 0.59/5.5 | **0.675** | 0.9732 | F9_ep28_b1: 0.78/1.6, 1.18/6.0; 1.021 / 0.9524 |
| 36 | **H13_ep36_b1** | staging/ep36_b1/components/episode_000036 | spoon 0.79/4.9 | **0.878** | 0.9679 | C9_ep36_beam5: 1.02/7.3; 1.166 / 0.9524 |
| 33 | **H13_ep33_b1** | staging/ep33_b1/components/episode_000033 | cup 0.82/4.5, spoon 0.91/7.5 | **0.974** | 0.9587 | C9_ep33_beam4: 0.92/4.7, 1.15/8.7; 1.141 / 0.9478 |
| 6 | **H13_ep06_b1** | staging/ep6_b1/components/episode_000006 | pot 0.49/2.4, spoon 0.63/6.1 | **0.647** | 0.9869 | D9_ep06_b2: 0.75/3.1, 1.27/8.6; 1.097 / 0.9502 |

- Every recommended tag has MP-SR 1 and SP-SR 1, and every object is well inside the MP limits (max 0.91 cm, 7.5 deg).
- Every recommended tag is better than its v27 component on MPPE, on AUC, and on every object's cm and deg, with one exception: ep28.
  - The ep28 rack orientation is 1.66 deg against 1.6. The rack position goes 0.78 -> 0.63 cm and MPPE goes 1.021 -> 0.675.
- Sum over the 9 replaced episodes: MPPE 9.83 -> 6.01. Over all 10 (with ep32 unchanged): mean MPPE 1.138 -> 0.756.
- Tags (build-v2 outputs) are in `campaign_v16_20261023/common_v2_candidates/H13_epNN_<x>/episode_XXXXXX`.
- Scores are in `campaign_v22_20261028/scores/cand/H13_epNN_<x>/score.json`.
- Portable archives use the assembler layout: `staging/epN_<x>/portable_archives/episode_XXXXXX_portable.tar.gz` (= legacy_component.parents[1]).
- Checkpoint / archive sha256 (first 12):
  - ep20 b2: cef1baa57cd2 / ae18c30e23e4;
  - ep5 b2: f5be3688c5de / 8b1f85bdbaef;
  - ep35 b2: 524e23ad28ab / 46f3183b2665;
  - ep37 b1: 3876392d35ff / 3e187ff328aa;
  - ep38 b2: 31e6174f2c4f / e16e6e3c47e3;
  - ep28 b1: 94565f73137b / 619102154e83;
  - ep36 b1: 3c17ca3998a2 / c234a42b6c92;
  - ep33 b1: 8401389ca37c / 1663f7c1a438;
  - ep6 b1: b91d8fbb731b / fdc38c34a2e9.
- Submitted meshes (sha256, first 8): cup ff3784a8, pitcher 08e6ab82 (ep20); pot 8cbd3543, spoon c9ab88e3 (eps 5, 6); spoon c9ab88e3 (eps 33, 35, 36, 37);
  blue cup 57bb269a (ep33); dust pan 1d298c01 (ep38); rack 00c51376, pot 69677910 (ep28).

All staged tags (fallbacks, all exact true; vs v12b):

| tag | objects | MPPE | AUC |
|---|---|---|---|
| H13_ep20_a1 (advance only) | 0.63/3.1, 0.83/3.6 | 0.758 | 0.9867 |
| H13_ep20_b1 | 0.41/1.9, 0.53/1.8 | 0.493 | 0.9940 |
| H13_ep05_a1 (advance only) | 0.68/5.0, 1.03/10.0 | 1.018 | 0.9614 |
| H13_ep05_b1 | 0.57/4.5, 0.82/4.9 | 0.776 | 0.9759 |
| H13_ep35_b1 | 0.73/7.1 | 0.910 | 0.9578 |
| H13_ep37_a1 (advance only) | 0.49/2.6 | 0.530 | 0.9938 |
| H13_ep38_a1 (advance only) | 0.59/4.9 | 0.693 | 0.9876 |
| H13_ep38_b1 | 0.40/3.3 | 0.478 | 0.9971 |
| H13_ep06_a1 (advance only) | 0.59/2.7, 0.86/7.0 | 0.808 | 0.9757 |

## Method
**Re-pointing (no world rebuild).**
- `code/h13lib.py` is F9 mlibF9 with TGT = target_final_v12b_g15 and NAMES for all ten episodes.
- Every v27 actor (`code/env.sh`, sha-checked against each component's staged.json) reproduces the G15 baseline score vs v12b exactly at advance 0.
- The worlds were not rebuilt. The target frame-0 change is only the retime resampling of the frame-0 sample: v12_g15 (retime only) and v12b are identical at f0-3.
  - The scorer aligns the scene at frame 0, so a global shift of the target does not matter.
  - The ep32 2 cm and ep20 1.4 cm relative f0 changes are noisy f0 samples (ep32 v10 f0 = 5.33 vs 3.05 cm at f1).
  - The staged internal reference (proxy MP-SR) stays the stream's own world reference. Only the scoring target changed.

**Step 1: global time-advance** (`code/h13adv.py`, `code/h13scan.py`, runs/scan/*.json).
- The actor's phase part phi(t)W is advanced by delta frames: the weight rows are interpolated along the centre times, which matches the true shift within 1 %.
- Registered release times (`*_start_s`, ep33) are moved earlier by the same delta. Feedback weights, margins and scene are untouched.
- Variants tried: constant 0.25-2 frames, wrist-only (`w` prefix), and piecewise for ep20 (0.925 -> 1.85 frames at 2.2-2.9 s).
- Result: the chaotic holds mostly break.
  - ep28 pot, ep32 spoon, ep33 cup, ep35 spoon and ep36 spoon are lost or degraded at almost every delta.
  - ep20 works only with the piecewise step at 2.2-2.4 s.
- Advances that were adopted as beam starts:

| ep | advance | scan result (MPPE) |
|---|---|---|
| 20 | piecewise 0.925@0;1.85@2.4 | 1.036 -> 0.759 |
| 37 | 1.0 | 0.733 -> 0.530 |
| 6 | 0.9 | 1.097 -> 0.808 |
| 38 | 0.7 | 0.841 -> 0.693 |
| 5 | 0.8 | 1.403 -> 1.018 (spoon deg 7.2 -> 10.0) |
| 28 | 0.25 | 1.021 -> 1.001 (marginal) |

- Eps 35, 36, 33 and 32 start from advance 0.

**Step 2: window beam** from the best start, using the source streams' recipes retargeted to v12b:
- `wbeamH13.py` (F9 wbeamF9) for eps 20, 38, 28 and 6. 0.5 s windows, beam 4, 40 random children, ILC per object hand, tol 0.6/6 (ep6 0.8/8).
- `c9beamH13.py` (C9 c9beam, chart-aware ILC via `h13res.py` = c9res + advance) for eps 5, 35, 36, 33 and 37.
  - Windows 0.4-0.6 s, beam 5, 40-48 children. ep5 has pot other-tol 1.2/6 plus left ILC. ep35/36 use gap 2.0.
- `wbeam4H13.py` + `mres32H13.py` (H12 wbeam4 + roll-continuous mres; hinge 4,2.0; adv/roll/squeeze candidates; e1 settings) for ep32.
- Pass chains:
  - ep20: b1, then b2 (offset windows, tol 0.5/5).
  - ep5: b1, then b2 (0.5 s windows).
  - ep35: b1, then b2 (offset 0.5 s windows, tol 0.9/10).
  - ep38: b1, then b2.
  - Eps 37, 28, 36, 33 and 6: b1 only (the b2 passes for 28 and 36 were cancelled for time).
- Every search value equals the official evaluate_policy_component re-evaluation and the build-v2 score_any value (to the printed digits).

**Save / stage / verify:** `code/svst.sbatch EP NAME ADV TABLE` runs four steps:
1. h13res save (bakes the advance and the table into the v27 actor's phase weights, then runs the official eval);
2. `code/stage.sbatch` (the unchanged stagers and teachers that staged each v27 component, plus H13a lineage files);
3. the `run_one.sbatch` body (build-v2);
4. `code/vs.sh`: exactness, archive, mesh cmp against the v27 component, and score_any vs target_final_v12b_g15.

## ep32: keep v27
- Every global advance (0.5-2 frames, wrist-only too) loses the spoon (11-10000 cm).
- One H12-recipe beam pass from the v27 actor vs v12b (runs/ep32/b1) lost the spoon in the f60-100 fast roll swing and finished at 2.63 cm / 57 deg.
- As in H12, the hold needs several chained rounds. There was no time for them within the budget.
- H12_ep32_e1f0 stays MP/SP 1 vs v12b (spoon 1.31/22.4).

## Caveats
- All scores are proxies against our own reconstruction, not hidden GT.
- The gain assumes the hidden GT is on the b/c clock, as G15 found on public.
- The rollouts are deterministic outcomes of chaotic holds. They are exact and replayable under build v2, but not robust to upstream edits.

## Files
- code/: h13lib.py, h13adv.py, h13res.py, mres32H13.py, h13scan.py, wbeamH13.py, c9beamH13.py, wbeam4H13.py, env.sh, job.sbatch, svst.sbatch, stage.sbatch, vs.sh, st.sh.
- runs/: scan/ (advance scans), epNN/bK (beam passes: settings, progress, beam_w*, final_*), save_epNN_x/actor (saved actors with official evaluation).
- staging/epN_x/: staged components and portable archives. logs/.
