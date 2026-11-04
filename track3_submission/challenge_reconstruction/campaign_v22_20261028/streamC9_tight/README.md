# Stream C9 (phase 9): tighten the spoon episodes 35, 36, 33 and 5 vs target_final_v6, 2026-10-28

Goal per object: <= 0.8 cm and <= 8 deg mean. Goal per episode: MPPE <= 0.9 and AUC >= 0.95. Every accepted tag must be no worse than the
v23b component on every object and on MPPE/AUC. For these four episodes target_final_v6 is byte-identical to target_final_v5 (checked file by file).

**Every goal is met in eps 35, 36 and 33.** ep5 meets every goal except the spoon position: it is 0.81 cm against 0.8 (plateau: the last round gained 0.004 cm).

- Every tag is build-v2 exact: `run_one.sbatch`, completed.json `exact_previous_physical_states_actions` true.
- Every tag is scored with `code/score_any.sh EP TAG_DIR TAG target_final_v6` (`../scores/cand/<tag>/score.json`).
- Tags are in campaign_v16_20261023/common_v2_candidates/.
- Submitted meshes are byte-identical to the v23b component's (cmp of the build-v2 output meshes):
  - pot 8cbd3543;
  - spoon c9ab88e3;
  - blue cup 57bb269a.
- Each component is the v23b component's own actor with a phase-weight residual table baked in. Scene, start state, feedback weights,
  margins and wrappers are unchanged.
- Nothing was uploaded, committed, deleted or overwritten.

## Results (recommended in bold; cm / deg vs target_final_v6)

| ep | tag | component (streamC9_tight/staging/...) | object 1 | object 2 | MPPE | AUC | exact | checkpoint / archive sha256 |
|---|---|---|---|---|---|---|---|---|
| 35 | **C9_ep35_beam6** | ep35_g0/components/episode_000035 | spoon **0.71 / 7.0** | | **0.889** | **0.953** | true | 5ae092ba42b6bbab / 562605432d4b5069 |
| 35 | C9_ep35_beam5 | ep35_f0 | spoon 0.76 / 6.9 | | 0.922 | 0.952 | true | |
| 35 | C9_ep35_beam4 | ep35_d0 | spoon 0.74 / 7.0 | | 0.913 | 0.952 | true | e88f528ebbcc895f / 209c7db60dabe950 |
| 35 | C9_ep35_beam3 / beam2 / beam1 | ep35_c0 / b0 / a0 | 0.84/7.4, 0.87/7.8, 1.05/9.0 | | 1.025, 1.063, 1.259 | 0.938, 0.933, 0.910 | true | |
| 35 | v23b K8_ep35_ilc2 | | spoon 1.27 / 13.5 | | 1.63 | 0.890 | | |
| 36 | **C9_ep36_beam5** | ep36_f0/components/episode_000036 | spoon **0.73 / 4.6** | | **0.817** | **0.970** | true | 2f68db15e8acf6eb / 64cf5c8c56fb8b6a |
| 36 | C9_ep36_beam4 | ep36_d0 | spoon 0.75 / 4.7 | | 0.836 | 0.970 | true | ae7ad85f030a70ac / d2ba6cdf1090c20d |
| 36 | C9_ep36_beam3 / beam2 / beam1 | ep36_c0 / b0 / a0 | 0.77/4.8, 0.82/5.2, 0.92/5.8 | | 0.854, 0.913, 1.015 | 0.969, 0.966, 0.959 | true | |
| 36 | v23b K8_ep36_ecem_ilc | | spoon 1.18 / 11.3 | | 1.46 | 0.930 | | |
| 33 | **C9_ep33_beam4** | ep33_d0/components/episode_000033 | cup **0.76 / 3.8** | spoon **0.75 / 5.2** | **0.820** | **0.968** | true | 111570adfe6d9ab4 / 0658b18224875d30 |
| 33 | C9_ep33_beam3 | ep33_c0 | cup 0.76 / 5.0 | spoon 0.76 / 5.5 | 0.846 | 0.966 | true | 63fe064096bdc105 / 04acc5b53cd723aa |
| 33 | C9_ep33_beam2 / beam1 | ep33_b0 / a0 | 0.88/4.9, 0.92/5.5 | 0.79/6.6, 1.03/7.6 | 0.926, 1.072 | 0.962, 0.946 | true | |
| 33 | v23b M8_ep33_r5 | | cup 1.22 / 9.4 | spoon 1.16 / 9.1 | 1.38 | 0.914 | | |
| 5 | **C9_ep05_beam6** | ep5_g0/components/episode_000005 | pot **0.53 / 4.4** (4.36) | spoon **0.81 / 5.0** | **0.755** | **0.977** | true | 96f4c2b7e142fe3c / 2ead3f6576c3ff5d |
| 5 | C9_ep05_beam5 | ep5_f0 | pot 0.55 / 4.3 | spoon 0.83 / 5.0 | 0.772 | 0.975 | true | 268cb2b95f262e2d / 364f0556222168cd |
| 5 | C9_ep05_beam4 / beam3 / beam2 / beam1 | ep5_e0 / c0 / b0 / a0 | 0.56/4.4, 0.64/4.4, 0.78/4.2, 0.94/4.2 | 0.86/5.1, 0.94/5.6, 0.96/5.9, 1.01/6.7 | 0.792, 0.876, 0.954, 1.063 | 0.973, 0.966, 0.960, 0.950 | true | |
| 5 | v23b Q8_ep05_g1 | | pot 1.22 / 4.5 (4.49) | spoon 1.22 / 9.2 | 1.34 | 0.923 | | |

- Every intermediate tag is also better than v23b on every object and metric. The pot orientation is the one tight case: v23b is 4.490 deg.
  - C9_ep05_beam6 4.356 and beam5 4.286 are below it.
  - Search round d (pot 4.508) was NOT staged because it is worse.
- The search value of every staged table equals the official actor evaluation (save_actor, node01) to all printed digits.
- Each staging reports portable and common replays exact, and build-v2 reproduces it.

## Method: window beam search with exact cached physics states, repeated in rounds (code/c9beam.py)
- `c9beam.py` is a copy of U8 `wbeam_u8.py` (itself V8 `wbeam2.py`). It takes the v23b component's actor (zero residual table at the start).
- The episode is cut into 0.4-0.6 s windows from about 0.5-1.4 s to the end.
- For each window and each of 5 beam parents, the children are:
  - 40-64 random right-hand wrist/finger knots;
  - rigid-hold ILC steps toward the target (full / pos / rot / rotanc, sigma 0.1 / 0.25 s, alpha 0.15-1);
  - for ep5/33, left-hand ILC steps on the pot or cup.
- Each child is rolled out from its parent's exact MuJoCo state at the cache frame and run to the window end + 1 s.
- Children are ranked by the cumulative scorer-faithful per-frame cost from frame 0: pos/P + deg/D per object.
- The best 5 children are kept, at most 2 per parent.
- The best final member is the table that gets saved. Its official actor evaluation runs on node01, then it is staged and run through build-v2.
- Each round starts from the previous round's table. Varying the window offset, window length and seed between rounds helped:
  every round found new improvements until the per-round gain dropped to about 0.02 cm.
- C9 changes to the U8 code (c9lib.py, c9res.py, c9beam.py):
  - c9lib: target_final_v6, plus NAMES for eps 3, 5, 35 and 36 in the scorer slot order (pot, spoon / cup, spoon).
  - c9res: **chart-aware wrist ILC** (R_palm = Ac R_XYZ(q), as in K8 ilcK). The ep35 right wrist hinges sit under a non-identity chart
    (body_quat 0.69/-0.32/-0.38/-0.52), so the plain mres correction would rotate about the wrong axes. For the other eps Ac = I.
  - c9beam:
    - single-object episodes (`--other none`);
    - `kc = 0` (full rollout from the keyframe) for early windows. For ep35/36 the actors' RBF width is 0.047 s, so with
      a 1.6 s gap every edit leaked into the cache prefix and was skipped. ep35/36 use `--gap 2.0`, which gives a leak of exactly 0.
    - `--other-tol` gives the second object its own per-frame tolerances. For ep5 the pot deg weight is doubled (1.2 cm, 6 deg) so the
      pot orientation stays under the v23b 4.49 deg.
    - `--left-frac/--left-scale` adds random left-hand children (see below).
- Search rollouts ran on node100 (ep35/36/5) and node01 (ep33). The official evaluation on node01 reproduced every node100 value exactly.

### Round chains (runs/epNN/beam_*, settings.json in each)
- ep35:
  - beam_a 0.5 s windows: 1.05/9.0;
  - beam_b offset 0.25: 0.87/7.8;
  - beam_c 0.4 s: 0.84/7.4;
  - beam_d tol 0.9/12: 0.745/7.03;
  - beam_g from d, 0.6 s windows, n 64: 0.713/7.00 (beam6);
  - beam_f (from c) gave 0.76/6.9, with MPPE 0.922, not better;
  - beam_h (from g) gave 0.715/6.89, a plateau. Stopped.
- ep36:
  - a 0.92/5.8;
  - b 0.815/5.18;
  - c 0.767/4.81;
  - d 0.749/4.67;
  - f 0.733/4.55 (beam5). Stopped: all goals met and the gain was below 0.02 cm per round.
- ep33:
  - a cup 0.92/5.5, spoon 1.03/7.6;
  - b 0.88/4.9, 0.79/6.6;
  - c 0.755/4.98, 0.763/5.49;
  - d 0.762/3.80, 0.745/5.21 (beam4). Stopped: all goals met.
- ep5:
  - a pot 0.94/4.2, spoon 1.01/6.7;
  - b 0.78/4.2, 0.96/5.9;
  - c 0.64/4.4, 0.94/5.6;
  - d 0.59/4.51 (pot deg above v23b, not staged);
  - e (pot tol 1.2/6) 0.558/4.43, 0.856/5.13;
  - f 0.547/4.29, 0.831/4.99;
  - g 0.530/4.36, 0.811/4.96 (beam6);
  - h (from g, 0.5 s, beam 6, n 64) 0.526/4.42, 0.8065/4.97: a plateau, and the pot deg got worse, so it was not staged. Stopped.

## What did not help
- **ep35 left-hand children** (beam_eL: 50 % of the random children on the left wrist, scale 3x, windows 8.5-13 s, look 1.5 s).
  - Motivation: the remaining 10.5-12 s error was 2.8-3.3 cm. P8 found the left hand presses the spoon at 9.2-11.8 s.
  - Result: the beam took left edits in every window, but the full-episode result was at best 0.823/7.41 against 0.836/7.38.
  - The left-hand edits change the later hold chaotically. Not staged.
  - The later right-hand rounds (d, g) removed most of that window anyway.
- **Search round ep5 d**: pot deg 4.508 vs v23b 4.490, so it was not staged. This led to `--other-tol`.

## Remaining error / floors
- ep35 end rest (19.5-21 s): 13-17 deg. The target's two rests disagree by about 12 deg (K8 analysis), so this is unreachable.
  It is about 0.8 deg of the mean.
- ep36 14.5-15.3 s: the target has a 20 cm outlier (K8). The spoon cannot follow it.

## Caveats
- All scores are proxies against our own reconstruction (target_final_v6), not hidden GT. The reconstruction itself is about 1.1 cm / 5 deg
  off public GT on average, so the hidden-GT error is larger.
- As with V8/U8/Q8, the final rollouts are specific deterministic outcomes of the search on chaotic holds. They are exact and replayable
  under build v2, but they are not robust to upstream edits.

## Files
- code/:
  - c9lib.py, c9res.py, c9beam.py (search);
  - wins.py (window lists), sumb.py (log summary);
  - res.sbatch (search/save/eval), stage.sbatch (the unchanged stagers: K8 stageK.py for 35/36, streamB stage_b.py for 33, streamA stage_compose_component.py for 5, with the v23b teachers);
  - vs.sh (exactness + score_any vs v6);
  - eps.sh (starting actors), tilt33.py (diagnostic).
- runs/epNN/: eval0 (reproduction of the v23b actor), beam_* (search rounds), save_* (actors with official evaluation).
- staging/epNN_XX/: staged components. logs/: job logs.
