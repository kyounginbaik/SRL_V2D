
## v21C (2026-10-24)

All 20 episodes pass against the coworker compose reconstruction using the kit scorers (3 cm / 30 deg), with exact build-v2 replay for every component. This is a reconstruction proxy result; hidden ground truth scores are unavailable. The 2.4 cm / 24 deg refinement target is reported separately in compose_validation.json. Episode 19 is packed without the sweeper relabel. All submitted meshes are our reconstructions. Episode 25 uses slotted versions of our board meshes.

| Episode | Object errors (cm / deg) |
| --- | --- |
| 1 | white_pot: 0.97 / 4.5; white_pot_lid: 1.68 / 15.9 |
| 3 | white_pot: 0.69 / 5.9; wooden_spoon: 2.14 / 20.5 |
| 5 | white_pot: 1.09 / 8.7; wooden_spoon: 2.27 / 22.1 |
| 6 | white_pot: 1.48 / 14.2; wooden_spoon: 1.59 / 19.1 |
| 8 | beige_cup: 1.14 / 3.8; white_pot: 1.69 / 6.1 |
| 10 | blue_cup: 1.43 / 6.1; beige_cup: 1.87 / 11.1 |
| 14 | blue_cup: 1.18 / 4.9; plastic_dish_rack: 0.50 / 2.3 |
| 15 | plastic_dish_rack: 0.32 / 2.3; wooden_spoon: 1.07 / 5.3 |
| 16 | plastic_dish_rack: 0.27 / 1.9; wooden_spoon: 2.18 / 12.3 |
| 19 | mini_sweeper: 1.76 / 10.0; mini_dust_pan: 1.70 / 12.0 |
| 20 | blue_cup: 0.32 / 2.2; water_pitcher: 2.08 / 6.2 |
| 25 | wooden_piece_1: 1.98 / 23.6; wooden_piece_2: 2.37 / 10.5 |
| 28 | plastic_dish_rack: 0.27 / 2.9; white_pot: 0.98 / 5.2 |
| 30 | mini_sweeper: 1.05 / 11.0; plastic_dish_rack: 0.65 / 3.1 |
| 32 | blue_cup: 1.49 / 5.7; wooden_spoon: 1.97 / 17.7 |
| 33 | blue_cup: 1.95 / 15.3; wooden_spoon: 2.56 / 23.9 |
| 35 | wooden_spoon: 2.03 / 19.5 |
| 36 | wooden_spoon: 1.37 / 14.5 |
| 37 | wooden_spoon: 2.17 / 22.1 |
| 38 | mini_dust_pan: 1.21 / 6.4 |

Videos for the 12 retargeted episodes compare v21 with the actual v21C rollout. Nothing uploaded.

Before submission, publish the existing evaluator build-v2 commit. Its recorded GitHub URL returned HTTP 404 and the branch was absent from the remote on 2026-10-24. The frozen runtime is already committed locally. Command: `git -C /data1/SRL_V2D/.local/worktree_evaluator_build_v2 push kyoungin evaluator-build-v2`.
