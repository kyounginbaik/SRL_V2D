# Stream J8 (phase 8, v23): episodes 19, 25, 28, 38 vs target_final_v5, 2026-10-28

The final goal (after the coordinator relaxed it) is <= 1.3 cm and <= 15 deg per object, and strictly better than v22b on every object.
All tags are in campaign_v16_20261023/common_v2_candidates. Every recommended tag replays exactly under build v2
(completed.json exact_previous_physical_states_actions true). Scores come from code/score_any.sh -> ../scores/cand/<tag>/score.json (cm / deg).

| ep | recommended tag | component (under this dir) | obj 1 | obj 2 | MPPE / AUC | v22b (before) | goal 1.3/15 |
|---|---|---|---|---|---|---|---|
| 19 | J8_ep19_a40m2e | ep19/staging_J5/components/episode_000019 | sweeper 1.40 / 13.3 | dust pan 1.54 / 9.1 | 1.68 / 0.898 | F_ep19_twin3 1.95/14.8, 2.38/9.9 (2.36 / 0.818) | position not met |
| 25 | J8_ep25_b48m4 | ep25/staging_b48m4/components/episode_000025 | piece_1 1.27 / 14.5 | piece_2 1.14 / 5.6 | 1.44 / 0.898 | E_ep25_R103s62 2.03/16.0, 2.04/10.7 (2.27 / 0.795) | met |
| 28 | J8_ep28_w2k3c2048 | ep28/staging_B3/components/episode_000028 | rack 0.74 / 1.9 | pot 0.77 / 8.0 | 0.84 / 0.966 | O6_ep28_C710 0.79/3.2, 2.19/15.3 (1.65 / 0.846) | met |
| 38 | J8_ep38_ilc36c2048 | ep38/staging_J1/components/episode_000038 | dust pan 0.64 / 5.5 | - | 0.76 / 0.980 | v21 baseline 1.40/7.3 (1.50 / 0.915) | met |

Alternates, method notes and caveats are in each epNN/README.md. What worked:
- ep28: a target world with a settled-rack start (world2), then an object-relative wrist port and knot CEM with the scorer-faithful cost, then distillation.
- ep38: a target world, then the v21 carry ILC re-run against the target, then distillation.
- ep25: a two-hand ILC and windowed CEM run directly on the actor's own closed-loop rollout (res25.py).
- ep19: a partial-twin start world (a40), then knot CEM, then actor-residual ILC/CEM and merged windows.
Caveats: the ep28 and ep19 worlds change the initial scene (a settled rack, and a sweeper start 40 % toward the dust-pan-consistent pose).
No pose was edited, but frame 0 differs from the v22b worlds.
