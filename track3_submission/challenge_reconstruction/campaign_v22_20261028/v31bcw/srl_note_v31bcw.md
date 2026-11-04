
## track3_v31b_HGB_cwmesh_20261101 (2026-11-01)

20/20 episodes pass against the reconstruction recon_E14_best_B2A_S1_HGB (target_E14_HGB) with the kit scorers (3 cm / 30 deg); 18/20 against selected20_final_v14_g17. Every component replays exactly under evaluator build v2. Reconstruction proxy only; hidden-GT scores are unavailable. Meshes: the coworker bundle meshes (cwmesh_v5: byte-exact or rigidly re-expressed in the component body frame, per-object table in streamE31_cowork2/README.md); robot rollouts identical to the v29-mesh package. v31 = v29 with E31 re-retargets of the episodes that fail MP vs the E14 target (3, 32, 33).

| Episode | Tag | Object errors vs target_E14_HGB (cm / deg) | vs v14_g17 |
| --- | --- | --- | --- |
| 1 | CW_H13_ep01_a1 | white_pot: 0.50 / 2.1; white_pot_lid: 0.54 / 5.4 | white_pot: 0.49 / 2.3; white_pot_lid: 0.55 / 5.1 |
| 3 | CW_E31A_ep03_a2 | white_pot: 0.46 / 3.5; wooden_spoon: 0.91 / 15.0 | white_pot: 0.45 / 3.8; wooden_spoon: 0.95 / 30.9 |
| 5 | CW_G18_ep05_H13b2 | white_pot: 0.58 / 4.4; wooden_spoon: 0.80 / 4.8 | white_pot: 0.56 / 4.5; wooden_spoon: 0.85 / 4.9 |
| 6 | CW_G18_ep06_H13b1 | white_pot: 0.49 / 2.1; wooden_spoon: 0.63 / 5.5 | white_pot: 0.51 / 2.4; wooden_spoon: 0.70 / 6.0 |
| 8 | CW_H13_ep08_a2 | beige_cup: 0.53 / 2.6; white_pot: 0.34 / 1.9 | beige_cup: 0.46 / 2.6; white_pot: 0.32 / 1.6 |
| 10 | CW_H13_ep10_t1p0 | blue_cup: 0.47 / 1.7; beige_cup: 0.57 / 2.9 | blue_cup: 0.48 / 1.8; beige_cup: 0.53 / 3.0 |
| 14 | CW_H13_ep14_t1p15 | blue_cup: 1.01 / 2.5; plastic_dish_rack: 0.80 / 2.4 | blue_cup: 0.99 / 2.4; plastic_dish_rack: 0.82 / 2.7 |
| 15 | CW_G18_ep15_E9b6 | plastic_dish_rack: 0.66 / 2.9; wooden_spoon: 1.32 / 7.5 | plastic_dish_rack: 0.65 / 2.9; wooden_spoon: 1.33 / 7.8 |
| 16 | CW_G18_ep16_H13t1p1 | plastic_dish_rack: 0.40 / 2.4; wooden_spoon: 0.81 / 5.4 | plastic_dish_rack: 0.41 / 2.6; wooden_spoon: 0.84 / 5.9 |
| 19 | CW_H13_ep19_a4 | mini_sweeper: 0.74 / 7.5; mini_dust_pan: 0.77 / 4.5 | mini_sweeper: 0.72 / 7.4; mini_dust_pan: 0.70 / 3.5 |
| 20 | CW_H13_ep20_b2 | blue_cup: 0.36 / 1.8; water_pitcher: 0.54 / 1.5 | blue_cup: 0.36 / 1.6; water_pitcher: 0.49 / 1.8 |
| 25 | CW_D9_ep25_b4 | wooden_piece_1: 1.18 / 12.5; wooden_piece_2: 1.45 / 4.7 | wooden_piece_1: 1.20 / 12.6; wooden_piece_2: 1.46 / 4.8 |
| 28 | CW_H13_ep28_b1 | plastic_dish_rack: 0.64 / 1.7; white_pot: 0.70 / 5.8 | plastic_dish_rack: 0.63 / 1.7; white_pot: 0.65 / 5.6 |
| 30 | CW_H13_ep30_t1p35 | mini_sweeper: 0.48 / 3.1; plastic_dish_rack: 0.31 / 3.1 | mini_sweeper: 0.48 / 3.2; plastic_dish_rack: 0.37 / 4.2 |
| 32 | CW_E31A_ep32_w3 | blue_cup: 1.02 / 4.3; wooden_spoon: 1.93 / 20.0 | blue_cup: 1.03 / 4.5; wooden_spoon: 1.96 / 29.0 |
| 33 | CW_E31A_ep33_a1t | blue_cup: 1.08 / 6.9; wooden_spoon: 1.13 / 15.1 | blue_cup: 1.10 / 6.9; wooden_spoon: 1.18 / 36.3 |
| 35 | CW_G18_ep35_H13b2 | wooden_spoon: 0.63 / 6.7 | wooden_spoon: 0.65 / 6.9 |
| 36 | CW_G18_ep36_H13b1 | wooden_spoon: 0.74 / 4.5 | wooden_spoon: 0.79 / 4.9 |
| 37 | CW_G18_ep37_H13b1 | wooden_spoon: 0.35 / 2.3 | wooden_spoon: 0.33 / 1.7 |
| 38 | CW_H13_ep38_b2 | mini_dust_pan: 0.44 / 4.6 | mini_dust_pan: 0.38 / 3.2 |

Before submission, publish the evaluator build-v2 commit: `git -C /data1/SRL_V2D/.local/worktree_evaluator_build_v2 push kyoungin evaluator-build-v2`. Nothing uploaded.
