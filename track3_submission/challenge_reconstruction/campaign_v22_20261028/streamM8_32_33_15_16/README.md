# Stream M8 (phase 8, v23): episodes 32, 33, 15, 16 vs target_final_v5

Goal (PHASE8.md, later relaxed by main to <= 1.3 cm / 15 deg per object): every object as close to target_final_v5 as possible, never worse
than the v22b component, lower per-frame peaks. All tags are in campaign_v16_20261023/common_v2_candidates/. Scores come from
campaign_v22_20261028/code/score_any.sh on the build-v2 output (`scores/cand/<TAG>/score.json`). All selected tags are build-v2 exact
(`exact_previous_physical_states_actions` true), MP-SR 1. Meshes are unchanged (the v22b component meshes). Nothing was uploaded or committed.

## Results (recommended tag first; cm / deg vs target_final_v5)

| ep | tag | staging (component = .../components/episode_X) | obj 1 | obj 2 | MPPE / AUC | v22b component (obj 1, obj 2, MPPE/AUC) |
|---|---|---|---|---|---|---|
| 33 | **M8_ep33_r5** | staging/ep33_r5 | cup 1.22 / 9.4 | spoon **1.16 / 9.1** | 1.38 / 0.914 | I_ep33_rel1: 2.06/13.1, 1.82/17.1 (2.21 / 0.809) |
| 33 | M8_ep33_cl (alt) | staging/ep33_cl | cup 1.20 / 9.5 | spoon 1.26 / 9.6 | 1.43 / 0.911 | |
| 33 | M8_ep33_spoonmid (superseded) | staging/ep33_spoonmid | 1.56 / 9.7 | 1.31 / 10.8 | 1.66 / 0.889 | |
| 16 | **M8_ep16_m** | staging/ep16_m | rack 0.37 / 2.33 | spoon 1.30 / 15.1 (1.296 / 15.06) | 1.05 / 0.922 | F_ep16_tgtret203: 0.37/2.36, 1.89/17.8 (1.32 / 0.900) |
| 16 | M8_ep16_j (verified) | staging/ep16_j | rack 0.37 / 2.3 | spoon 1.30 / 15.6 | 1.06 / 0.921 | |
| 16 | M8_ep16_cemd (superseded) | staging/ep16_cemd | 0.37 / 2.3 | 1.53 / 17.2 | 1.20 / 0.904 | |
| 32 | **M8_ep32_late2** | staging/ep32_late2 | cup **0.89 / 7.5** | spoon 1.92 / 16.5 (1.91798/16.4604, i.e. unchanged) | 1.63 / 0.885 | H_ep32_ilclate: 1.47/10.2, 1.918/16.461 (1.93 / 0.858) |
| 32 | M8_ep32_cuplate | staging/ep32_cuplate | 0.89 / 7.5 | 1.92 / 16.5 (bit-identical to v22b) | 1.63 / 0.885 | |
| 15 | none (keep v21 S6_ep15_newtrack_rel50p2) | | rack 0.60 / 2.3 | spoon 1.66 / 12.6 | | no candidate beat it |

Checkpoint / portable archive sha256 (prefix): ep16_m 45caf6dc0af68cb8 / 3a656bd050bebde2; ep33_r5 5b74e7d6a6d8e476 / b891074403bf5a09; ep33_cl e51a73cc28669515 / 8691eba8cd1d9c82;
ep16_j 3a94677f5cbc13e8 / 2b0d22d92a5b5fca; ep32_late2 2674348e524f5234 / 59cdc077f3996f29 (others: staged.json in each staging dir).

## Method: physical fine-tuning of the shipped actor's phase weights (`code/mres.py`)
- This generalizes streamH actor_res.py to any episode, both hands and every registered wrapper. LearnedPolicy, PinkyMarginPolicy, ReleasePolicy
  and MultiReleasePolicy share one data dict.
- A per-frame residual on wrist (6) and finger-squeeze (2) channels per hand is baked into the RBF phase weights. Feedback weights, margins,
  release-primitive parameters, scene and start state are unchanged. Architecture and evaluator are unchanged.
- Every candidate is a full closed-loop rollout with the evaluator's ComponentPolicy and substep schedule. It is scored by `code/mlib.py`
  (scorer-faithful MP-SR registration + frame-0 anchor alignment). That score reproduces score_any exactly on all four v22b components.
  The search value equals the official actor evaluation and the build-v2 score.
- Cost: max r + 0.5 sum r + 0.1 * sum(p95 pos / P) (per-frame peaks) + a large penalty for any object above its v22b value + an optional goal
  hinge (`--goal 1.3,15`). Here r = pos/P + deg/D.
- Modes:
  - `ilc`: rigid-hold correction toward the target expressed in the physics world (mlib.target_phys). `--axis-only` is for the axially symmetric cup.
  - `cem`: knot CEM on one hand.
  - `lot`: lottery with per-sample prefix-window errors.
  - `eval`, `save`.
- `code/rcem.py` is stream I's release-primitive CEM generalized to any episode. `code/evalmany.py` scores existing actor pools.
  `code/contacts.py` lists the spoon's contacts. `code/robust.py` is the P8-style decoupling test.
- Staging uses the unchanged stagers: streamB stage_b.py (ep32/33), the streamF ep16 retargeted stager, and the streamS newtrack stager
  (`code/stage_b.sbatch`, `stage16.sbatch`, `stage15.sbatch`; `code/mkbase.py` writes the BASE). Verification is campaign_v16
  run_one.sbatch with `--time=00:50:00`. Scoring is `code/score.sh`.
- Search runs are in `runs/` (symlink to tmp/runs). Logs are in `logs/res-<job>.log`.

## Per episode: what worked / what did not
- **ep33** (start: I_ep33_lr4 = c51 + MultiReleasePolicy):
  - Not chaotic like ep32.
  - Right-wrist spoon ILC over 3.6-14.6 s: spoon 1.82 -> 1.31.
  - Then alternating knot CEMs on the right hand (3.6-14.6 s) and the left hand (cup, 1.0-12.0 s): cup 1.56 -> 1.20, spoon -> 1.22.
  - A final spoon-weighted ILC brought the spoon to 1.16.
  - The release primitives (after 12.04 / 14.67 s) are untouched.
  - The 2.5-3.5 s lift window was tried with a knot CEM. It did not help beyond the chain (the chain is cem_r -> cem_l -> cem_r2 -> cem_r3 -> cem_r4 -> ilc_r5).
- **ep16**:
  - The spoon is held in the rack until the end, and the rack contact makes ILC noisy.
  - A knot CEM chain on the right hand (cem_a .. cem_m; later with P=1.3, D=15 and the goal hinge, plus rotation-focused windows
    1.2-4.2 / 2.8-6.0 / 5.5-8.6 s) moved the spoon from 1.89/17.8 to 1.30/15.06 with the rack <= v22b.
  - Orientation plateaus at about 15.0-15.2 deg. It is a systematic about 13 deg tilt of the seated spoon about its body y/z axes vs the target.
  - The 0-1.5 s pre-grasp floor is about 1.7 cm / 22 deg at f0, from the start layout.
- **ep32**:
  - Cup: left-wrist axis-only ILC over 9.3-12.9 s gives cup 1.47 -> 0.89, with the spoon bit-identical. The left hand after 9.3 s is decoupled from the spoon.
  - Spoon: no real gain. Right-wrist ILC over 9.0-12.9 s only reached 1.91798 / 16.4604.
  - Any change before 9.2 s re-rolls the chaotic power-grasp hold:
    - Cup ILC from 0.3 s drops the spoon (2.2-10 cm).
    - 2,880 lottery draws (`lot`, 1 mm / 0.008 rad, 3.8-9.2 s): none beats the base prefix-window spoon error (2.54/21.3). Median 7-10 cm.
  - P8 other-hand check (`contacts.py`): the left hand never touches the spoon in ep32/15/16. The ep32 spoon touches the left-held cup at 3.5-4.5 and 6.5-7 s.
  - Shifting the left wrist/cup 3-5 cm away over 3.0-9.4 s (`robust.py`) does not stabilize the hold: 0/29 1-mm perturbations stay < 3 cm at every shift, and the shift itself knocks the cup.
- **ep15** (v21 S6_ep15_newtrack_rel50p2, never re-planned): no improvement found. The seat/release into the rack is chaotic, and the baseline is a lucky draw:
  - ILC 1.6-5.2 s;
  - knot CEMs (1.3-5.6 s, carry 1.3-4.3 s, place 3.6-6.6 s);
  - a release-primitive CEM (rcem, 3 runs);
  - a 1,500-draw 0.3 mm lottery (0 strictly better; median spoon 1.81);
  - the 50-actor distilled_nt pool (best = the shipped actor).
  - Error structure: about 1.2-1.6 cm pre-grasp floor (target spoon about 1.4 cm lower than our table-resting spoon); the rest pose after 5.5 s is
    offset about 1.6 cm in y, and the target roll jitters by ±15 deg frame to frame.

## Caveats
- Proxy scores are vs our reconstruction (target_final_v5), not hidden GT.
- ep32's spoon and ep15 are unchanged in practice. Neither meets "strictly better on every object" in a meaningful sense: ep32's spoon is better
  only at the 1e-5 level, and ep15 has no new tag.
- ep16's spoon orientation is about 15.1 deg, just above the 15 deg relaxed goal.
- Searches ran on AMD nodes (node01/node100) only, and staging and verification on node01. Residual tables are in `runs/*/best_table.npz`.
- `mres.py` was extended during the stream (lottery mode, goal hinge). Each actor's training_report records the sha256 that produced it.
