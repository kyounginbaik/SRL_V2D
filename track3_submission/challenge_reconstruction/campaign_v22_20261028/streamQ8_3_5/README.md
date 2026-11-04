# Stream Q8 (phase 8, v23): episodes 3 and 5 (white_pot + wooden_spoon) vs target_final_v5, 2026-10-28

Goal: every object at or below 1.3 cm and 15 deg (mean, vs target_final_v5), build-v2 exact. **Both episodes now meet it.**
Starting points were L8_ep03_potlate (spoon 1.78/16.3) and L8_ep05_potl (spoon 1.59/11.5).

- Meshes are unchanged (the v22b ones).
- Every component is the L8 actor plus a finite phase-weight residual table. The scene, start state and feedback weights are unchanged.
- Nothing was uploaded, deleted or committed.

## Results (score_any.sh on the build-v2 output; cm / deg; ../scores/cand/<tag>/score.json)
All tags are in campaign_v16_20261023/common_v2_candidates/.

| ep | tag | component | pot | spoon | MPPE | AUC | exact | checkpoint / archive sha256 |
|---|---|---|---|---|---|---|---|---|
| 3 | **Q8_ep03_g11** (recommended) | ep3/staging_Q2/components/episode_000003 | **0.85 / 6.3** | **1.13 / 9.7** | 1.13 | 0.941 | true | 8aace13d8882.. / 53028ebfc7d1.. |
| 3 | Q8_ep03_wl2 | ep3/staging_Q1/components/episode_000003 | 0.96 / 7.4 | 1.56 / 15.3 | 1.48 | 0.908 | true | 3bb31568533f.. / 0543a378033b.. |
| 3 | (L8_ep03_potlate, start) | | 0.89 / 6.6 | 1.78 / 16.3 | 1.56 | 0.898 | true | |
| 5 | **Q8_ep05_g1** (recommended) | ep5/staging_Q3/components/episode_000005 | **1.22 / 4.5** | **1.22 / 9.2** | 1.34 | 0.923 | true | b795e707336c.. / fb207b28d837.. |
| 5 | Q8_ep05_wk1 | ep5/staging_Q2/components/episode_000005 | 1.22 / 4.3 | 1.33 / 10.4 | 1.40 | 0.913 | true | 3abdd313dc91.. / 180525b5cbc4.. |
| 5 | Q8_ep05_adv1 | ep5/staging_Q1/components/episode_000005 | 1.21 / 4.4 | 1.42 / 12.7 | 1.47 | 0.908 | true | ba17e59a62fa.. / ec1d49f4e032.. |
| 5 | (L8_ep05_potl, start) | | 1.18 / 4.3 | 1.59 / 11.5 | 1.52 | 0.901 | true | |

Every staging reports portable and common replays exact. Every build-v2 `completed.json` has exact_previous_physical_states_actions true.

Trade-off: the ep5 pot is 0.04 cm worse than L8 (1.22 vs 1.18), but still under 1.3. The ep3 pot is better than L8.

## ep5 floor (code/floor5.py, lag5.py, band.py, floor_scen.py, pen.py, slow5.py; on L8_ep05_potl): is 1.3 reachable?
- **Target jitter.** Following a Gaussian-smoothed target exactly gives 0.44 / 1.02 / 1.49 / 1.80 cm for sigma = 1 / 2 / 3 / 4 frames.
  - The 2-11 s motion is a real 2.1 Hz stir with about 3 cm rms in the high-pass band. It is not frame noise: frame-level jitter is 0.44 cm.
- **Rollout.** The spoon stirs at gain 0.48 and lags by 1 frame.
  - The cause is the wrist PD (kp 1000, kd 80, not saturated): it passes only 50 % of the commanded 2 Hz amplitude, with a 2-frame lag.
  - The actor's command itself carries gain 0.39.
  - The error spectrum is 67 % below 2.2 Hz, but mostly in the stirring band (1.7-2.8 Hz).
- **Ideal edits.**
  - Removing all slow error (sigma 4) gives 1.23 cm.
  - Removing the lag plus the slow error gives 1.04 cm.
  - Perfect slow tracking plus stirring gain 0.5 with zero lag gives 0.90 (G4) to 1.29 (G8) cm.
- **Geometric floor.** The target spoon penetrates the pot by up to 2.2 cm in 20 of 288 frames. Its lower bound is only 0.09 cm, but it explains part of the +0.4-0.7 cm z bias seen while stirring.
- **Conclusion:** 1.3 is reachable in principle, but only by fixing the lag (phase), not by the slow component alone. The run confirmed it: 1.22.

## What worked
1. **Stirring-band time advance**, code/boost.py --advance (ep5).
   - The residual is delta = g * HP_s(phase command)(t + lead) - HP_s(phase command)(t), applied on the right-wrist channels over 2-11.2 s.
   - Grid result: lead 1 frame, g_pos 1.12, s 2 frames → 1.42/12.7 (Q8_ep05_adv1). The spoon lag is gone (0 frames) and the gain rises to 0.56.
   - A gain-only boost reached 1.52. In a 2000-sample per-axis lottery of this family, the best was 1.377/12.6: a floor of about 1.38, with 27 samples below 1.4.
2. **Windowed proposals with full-episode acceptance**, code/wlot.py --topk (ep5).
   - Per 1 s window: about 100 random knot residuals plus low-pass rigid-hold ILC steps (sigma 0.1-0.4 s, lead 0-1, alpha 0.15-1, pos or full).
   - Candidates are ranked by the error on [ws, we + 0.8 s] (truncated rollout from a leak-checked prefix cache).
   - The top 8 are re-run over the full episode, and one is accepted only if the full cost drops (monotone).
   - Results: adv1 → 1.33/10.4 (Q8_ep05_wk1); advR3b → 1.3001/10.0.
3. **Greedy windowed rigid-hold ILC**, code/wlot.py --ilc without --topk (ep3 and ep5).
   - The same proposals, but each window keeps its best window-scored candidate, the prefix is locked, and the search moves on (13 windows over 2-15.6 s).
   - In ep3, every window picked a full-pose ILC step with alpha 0.6-1.0, so the wrist follows the target-implied rigid-hold pose one window at a time.
   - Intermediate full-episode scores are chaotic (the spoon is even transiently lost at 4-5 and 7-8 s). The final rollout lands in a much better mode:
     - ep3 from L8's actor: 1.13/9.7, pot 0.85/6.3 (Q8_ep03_g11);
     - ep5 from Q8_ep05_wk1: 1.22/9.2 (Q8_ep05_g1).
   - The late windows (10-15.6 s) are benign and gain a lot: ep3 1.56 → 1.13.
   - The greedy picks were deterministic ILC candidates, so seeds 11 and 12 gave identical results.
4. ep3 Q8_ep03_wl2 (interim): the greedy window lottery with random knots only (2-10 s), then late ILC windows → 1.56/15.3.

## What did not work (numbers)
- **ep5 low-pass filtered-residual ILC** (q8ilc.py; sigma 0.25-0.6 s, lead 0-2, 12 iterations, full-episode acceptance): 1.59 → 1.58.
  - Alpha collapsed to 0.002. Small edits flip the spoon orientation by 11 → 18 deg.
- **ep5 piecewise slow wrist offsets** (offgrid.py, 240 samples over 3 segments of ±6 mm / -8..+3 mm z): 0 better than adv1, median 2.66 cm.
- **ep5 greedy window lottery without ILC candidates** (wl1/wl2): 1.51-2.59. The window choices wreck the release/rest after 11 s.
- **ep3 fcem target-following teacher** (fcem3.py = streamB fcem.py ported to pot/spoon; t_lock 3.4, t_off 9.8, 2 seeds): loses the spoon, 5.2-7.2 cm / 44-70 deg against the teacher's 2.14/21.1, so it was stopped and not distilled.
  - The rigid target-following that worked was the per-window closed-loop ILC on the actor (point 3 above), not a teacher re-plan.
- **ep3 other searches:**
  - pickup roll-pulse lottery (rollpulse.py, wrist roll about the spoon axis, 800 samples): no gain;
  - stirring boost and advance on 6.4-9.9 s: no gain;
  - top-K-accepted windows 4-8 s: 0 accepted;
  - greedy shifted windows from wl2: spoon flips (8 cm).

## Caveats
- **Lottery-like outcomes.** The ep3/ep5 holds are chaotic, and the final rollouts are single lucky outcomes of deterministic searches. They are exact and replayable under build-v2, but small changes upstream would not reproduce them.
- **Hand-object forces.** Intermediate rollouts of the greedy search lost the spoon transiently. Only the final rollout counts, and it is physics_verified with max object penetration as reported by the staging.
- **ep5 margin.** ep5 g1 passes by 0.08 cm on both objects (spoon 1.22, pot 1.22).
- **Proxy scores.** All scores are against our own reconstruction, not hidden GT.

## Code (code/)
- **Searches:**
  - boost.py: stirring-band gain / advance, grid or per-axis random;
  - wlot.py: windowed lottery, ILC candidates, --topk full acceptance;
  - q8ilc.py: filtered-residual ILC with lead;
  - offgrid.py, rollpulse.py;
  - fcem3.py: streamB fcem port.
- **Diagnostics:** floor5.py, lag5.py, band.py, floor_scen.py, slow5.py, pen.py, qtcheck.py.
- **Job scripts:** res.sbatch, stage.sbatch (the unchanged streamA stage_compose_component.py with the v22b teacher, plus L8 and Q8 lineage files), env.sh.
- **Shared code:** l8res.Env / plib from streamL8_3_5_6/code, imported unchanged.
- **Outputs:** runs/ep{3,5}/<run>/ hold progress, tables and actors.
