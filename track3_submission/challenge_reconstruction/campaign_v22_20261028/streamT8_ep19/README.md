# Stream T8 (phase 8, v23): episode 19 (mini_sweeper + mini_dust_pan) vs target_final_v5, 2026-10-28

Goal: <= 1.3 cm and <= 15 deg per object. Start: J8_ep19_a40m2e (sweeper 1.40 / 13.3, dust pan 1.54 / 9.1). Goal met.
Meshes unchanged (stage script asserts sweeper cf7bf340.., dust pan 1d298c01..). Every pose comes from a build-v2 policy rollout. Nothing was edited, uploaded, deleted or committed.
World, teacher and lineage are the same as J8 (world_a40, teacher cem/a40_kA_LR_2_5.5). The actor is the J8 a40m2e plain LearnedPolicy. Only its wrist phase weights carry extra residuals.

## Results (score_any on the build-v2 re-execution; completed.json exact_previous_physical_states_actions true for all three)
| tag (campaign_v16_20261023/common_v2_candidates/) | staging (component = staging/components/episode_000019) | sweeper cm/deg | dust pan cm/deg | MPPE | AUC |
|---|---|---|---|---|---|
| **T8_ep19_c2 (recommended)** | staging_T3 (actor res/save_c2/actor) | **1.28 / 13.7** | **1.12 / 9.1** | 1.45 | 0.924 |
| T8_ep19_c1 | staging_T2 (res/save_c1/actor) | 1.28 / 13.7 | 1.21 / 9.1 | 1.49 | 0.920 |
| T8_ep19_dp1sw | staging_T1 (res/save_dp1_swL1/actor) | 1.37 / 13.3 | 1.16 / 9.3 | 1.51 | 0.917 |
| J8_ep19_a40m2e (start) | streamJ8_19_25_28_38/ep19/staging_J5 | 1.40 / 13.3 | 1.54 / 9.1 | 1.68 | 0.898 |

## Error budget (code/budget.py; per 0.5 s window, contribution to the mean = sum(cm) / 235)
Start (a40m2e):
- Dust pan: 7.0-8.0 s = 0.44 cm of 1.54. Everything above 1.3 came from this one window.
  It is not a tipping. The left wrist shakes the held pan by +-8 cm in physics y/z at about 3 Hz (f142-157; pan speeds up to 2 m/s).
  The target pan lifts about 4 cm, then is near-static and linearly interpolated (6.6 cm/s, f147-154).
- Sweeper: 4.5-5.5 s 0.29, 11.0-11.5 s 0.15, 3.0-3.5 s 0.12, 2.0-3.0 s 0.17.
  At 11.0-11.5 s the right wrist is already pressed on the table (z about 2 cm, ctrl below the table, about 120 N on the sweeper). The target sweeper anchor is 3.6-3.9 cm lower, and a rigid-hold position ILC cannot lower it.
After c2: dust pan 7.0-8.0 s is down to 0.05 (from 0.44). The sweeper still has 0.26 at 4.5-5.5 s (the target face flip at f94-97 stays near 160 deg) and 0.12 at 11.0-11.5 s. Both now sit under the 1.3 cm goal on the mean.

## What worked
1. **Rigid-hold ILC on the actor's own rollout** (code/ilc19t.py). It reuses J8 actor_res19.Env (the residual is baked into the phase weights; rollouts are identical to evaluate_policy_component).
   Each iteration adds gain x (target - achieved) to the wrist position channels. The object-anchor error is taken in the physics world through the scorer's frame-0 sweeper alignment, with a lead lag and a cosine-tapered window. A gain x lag grid runs in parallel.
   Left wrist, 6.8-8.0 s: dust pan 1.54 -> 1.33 -> 1.24 -> 1.18 -> 1.16 in 4 iterations (res/ilc_dp_L6.8_8.0). The sweeper is unaffected.
   A right-wrist sweeper ILC at 10.6-11.7 s gave 1.398 -> 1.374 (res/ilc_sw_R10.6_11.7). The two residuals were superposed with J8 merge_actor.py -> T1.
2. **Sweeper actor CEM** (J8 actor_res19.py cem, P 1.3 / D 15) on T1, run in separate windows, then **superposition** of the found residual tables (code/combo.py evaluates every subset).
   Right wrist 4.5-5.9 s gave 1.325 alone. Right wrist 10.3-11.8 s gave 1.334 alone. Together (M5+L2) they gave sweeper 1.281, dust pan 1.207 -> c1.
3. Dust-pan ILC re-run on c1 (left wrist, 6.8-8.0 s): dust pan 1.207 -> 1.115, sweeper unchanged -> c2.

## What did not work
- Sweeper ILC in the mid windows (1.4-2.6, 2.4-3.8, 2.9-3.4, 4.85-5.6 s), alone or jointly with the dust-pan term: the sweeper sweeps against the dust pan, so the dust pan diverges (2-17 cm).
- Sweeper CEM at 0.6-2.2 s or left 0.8-2.0 s: no gain. Right 2.3-3.7 s CEM: sweeper 1.32 alone and 1.27 on c1, but the dust pan gets worse (1.22) and the sweeper angle rises to 14-15 deg. Combined with M5 it breaks the dust pan (1.41). Not used.

## Caveats
- Same world caveat as J8: the sweeper start in world_a40 is 40 % toward the dust-pan-consistent target pose and settles about 3 deg before the grasp.
- The sweeper margin is thin (1.28 vs 1.3). The residual sweeper error is in the target face-flip frames (f88-100) and in the 11-11.5 s placement, where the hand is already on the table.
- Searches ran on node200. Every saved actor was re-evaluated on node01 with the official evaluate_policy_component, which reproduced the search numbers exactly. Staged on node01 (exact, portable exact) and re-executed with run_one.sbatch on node01 (exact).

code/: budget.py, frames.py (per-frame diagnosis), ilc19t.py, combo.py, ilc.sbatch / cem.sbatch (searches, node200), save.sbatch (node01 bake + official eval), stage.sbatch (J8 stage script, T8 lineage).
