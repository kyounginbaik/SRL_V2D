
## track3_v31b_HGB_20261101 (2026-11-01)

20/20 episodes pass against the reconstruction recon_E14_best_B2A_S1_HGB (target_E14_HGB) with the kit scorers (3 cm / 30 deg); 18/20 against selected20_final_v14_g17. Every component replays exactly under evaluator build v2. Reconstruction proxy only; hidden-GT scores are unavailable. Meshes: v29 component meshes (G18 spoon mesh in eps 3, 5, 6, 15, 16, 32, 33, 35, 36, 37). v31 = v29 with E31 re-retargets of the episodes that fail MP vs the E14 target (3, 32, 33).

| Episode | Tag | Object errors vs target_E14_HGB (cm / deg) | vs v14_g17 |
| --- | --- | --- | --- |
| 1 | H13_ep01_a1 | white_pot: 0.50 / 2.1; white_pot_lid: 0.54 / 5.4 | white_pot: 0.49 / 2.3; white_pot_lid: 0.55 / 5.1 |
| 3 | E31A_ep03_a2 | white_pot: 0.47 / 3.5; wooden_spoon: 0.80 / 15.4 | white_pot: 0.44 / 3.8; wooden_spoon: 0.86 / 31.7 |
| 5 | G18_ep05_H13b2 | white_pot: 0.58 / 4.4; wooden_spoon: 0.94 / 7.2 | white_pot: 0.54 / 4.5; wooden_spoon: 1.00 / 7.1 |
| 6 | G18_ep06_H13b1 | white_pot: 0.48 / 2.0; wooden_spoon: 0.71 / 6.4 | white_pot: 0.49 / 2.4; wooden_spoon: 0.76 / 6.9 |
| 8 | H13_ep08_a2 | beige_cup: 0.46 / 1.8; white_pot: 0.31 / 1.8 | beige_cup: 0.39 / 1.7; white_pot: 0.32 / 1.6 |
| 10 | H13_ep10_t1p0 | blue_cup: 0.47 / 1.7; beige_cup: 0.57 / 2.9 | blue_cup: 0.48 / 1.8; beige_cup: 0.53 / 3.0 |
| 14 | H13_ep14_t1p15 | blue_cup: 1.04 / 2.6; plastic_dish_rack: 0.74 / 2.4 | blue_cup: 1.01 / 2.5; plastic_dish_rack: 0.77 / 2.7 |
| 15 | G18_ep15_E9b6 | plastic_dish_rack: 0.66 / 2.9; wooden_spoon: 1.12 / 8.4 | plastic_dish_rack: 0.66 / 2.9; wooden_spoon: 1.14 / 8.6 |
| 16 | G18_ep16_H13t1p1 | plastic_dish_rack: 0.40 / 2.4; wooden_spoon: 0.98 / 5.4 | plastic_dish_rack: 0.42 / 2.6; wooden_spoon: 1.01 / 5.7 |
| 19 | H13_ep19_a4 | mini_sweeper: 0.74 / 7.4; mini_dust_pan: 0.80 / 4.4 | mini_sweeper: 0.72 / 7.4; mini_dust_pan: 0.75 / 3.6 |
| 20 | H13_ep20_b2 | blue_cup: 0.36 / 1.3; water_pitcher: 0.56 / 1.5 | blue_cup: 0.36 / 1.3; water_pitcher: 0.51 / 1.8 |
| 25 | D9_ep25_b4 | wooden_piece_1: 1.17 / 12.5; wooden_piece_2: 1.43 / 4.7 | wooden_piece_1: 1.19 / 12.6; wooden_piece_2: 1.43 / 4.8 |
| 28 | H13_ep28_b1 | plastic_dish_rack: 0.64 / 1.7; white_pot: 0.60 / 5.7 | plastic_dish_rack: 0.63 / 1.7; white_pot: 0.59 / 5.5 |
| 30 | H13_ep30_t1p35 | mini_sweeper: 0.48 / 3.1; plastic_dish_rack: 0.27 / 3.0 | mini_sweeper: 0.48 / 3.2; plastic_dish_rack: 0.34 / 4.0 |
| 32 | E31A_ep32_w3 | blue_cup: 1.03 / 4.8; wooden_spoon: 1.68 / 20.3 | blue_cup: 1.05 / 5.0; wooden_spoon: 1.74 / 29.2 |
| 33 | E31A_ep33_a1t | blue_cup: 1.07 / 7.3; wooden_spoon: 1.05 / 15.2 | blue_cup: 1.09 / 7.4; wooden_spoon: 1.14 / 36.4 |
| 35 | G18_ep35_H13b2 | wooden_spoon: 0.88 / 7.1 | wooden_spoon: 0.90 / 7.4 |
| 36 | G18_ep36_H13b1 | wooden_spoon: 1.45 / 6.5 | wooden_spoon: 1.47 / 6.6 |
| 37 | G18_ep37_H13b1 | wooden_spoon: 0.54 / 2.2 | wooden_spoon: 0.54 / 1.7 |
| 38 | H13_ep38_b2 | mini_dust_pan: 0.44 / 4.6 | mini_dust_pan: 0.38 / 3.2 |

Before submission, publish the evaluator build-v2 commit: `git -C /data1/SRL_V2D/.local/worktree_evaluator_build_v2 push kyoungin evaluator-build-v2`. Nothing uploaded.
