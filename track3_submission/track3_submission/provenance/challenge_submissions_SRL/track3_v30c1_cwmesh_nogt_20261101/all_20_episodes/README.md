# SRL Track 3 submission: all twenty episodes

`track3_submission.parquet` holds all 9,120 public pose rows and the reconstructed
object meshes for the twenty scored Track 3 episodes. It was produced by the
unchanged official packer and passed the official uploader's local validation and
offline preflight. No Kaggle API was called from this package.

Every episode is a closed-loop physics rollout (MuJoCo, 500 Hz) of a per-episode
learned controller driving two floating Sharpa hands; objects move only through
contact. The
achieved object poses were exported on the original 20 Hz timestamp grid.

- `results/`: one achieved-pose Parquet per episode (the inputs to the packer).
- `meshes/`: body-frame visual meshes per episode, as packed into the submission.
- `robot_actions/`: recorded controller states and actuator commands per episode.
- `videos/`: review renders (original video, kinematic reference, baseline physics,
  refined physics) where available.
- `submission_kit/`: the official validation and upload tools.
- `manifest.json`, `SHA256SUMS.json`: inventory and checksums.

Local checks compared against the reconstructed object trajectories as a proxy for
hidden ground truth (every object below 3 cm mean position and 30 deg mean
orientation error). These are not official leaderboard scores. Episodes that do
not pass that local proxy, if any, are listed under `best_effort_episodes` in
`manifest.json`.

To upload, from `submission_kit/`:

    python tools/preflight.py --track track_3 --results ../results
    python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m "SRL Track 3 full twenty episodes" --wait

## v30c1 (2026-11-01)

20/20 episodes pass against our reconstruction selected20_final_v14_g17 with the kit scorers (3 cm / 30 deg); every component replays exactly under evaluator build v2. Reconstruction proxy only; hidden-GT scores are unavailable. Our meshes only (the v29 meshes, incl. the G18 spoon mesh); see campaign_v22_20261028/streamC30_compose/README.md.

| Episode | Tag | Object errors (cm / deg) |
| --- | --- | --- |
| 1 | H13_ep01_a1 | white_pot: 0.49 / 2.3; white_pot_lid: 0.55 / 5.1 |
| 3 | G18_ep03_H13a3b | white_pot: 0.35 / 3.2; wooden_spoon: 0.99 / 19.1 |
| 5 | G18_ep05_H13b2 | white_pot: 0.54 / 4.5; wooden_spoon: 1.00 / 7.1 |
| 6 | G18_ep06_H13b1 | white_pot: 0.49 / 2.4; wooden_spoon: 0.76 / 6.9 |
| 8 | H13_ep08_a2 | beige_cup: 0.39 / 1.7; white_pot: 0.32 / 1.6 |
| 10 | H13_ep10_t1p0 | blue_cup: 0.48 / 1.8; beige_cup: 0.53 / 3.0 |
| 14 | H13_ep14_t1p15 | blue_cup: 1.01 / 2.5; plastic_dish_rack: 0.77 / 2.7 |
| 15 | G18_ep15_E9b6 | plastic_dish_rack: 0.66 / 2.9; wooden_spoon: 1.14 / 8.6 |
| 16 | G18_ep16_H13t1p1 | plastic_dish_rack: 0.42 / 2.6; wooden_spoon: 1.01 / 5.7 |
| 19 | H13_ep19_a4 | mini_sweeper: 0.72 / 7.4; mini_dust_pan: 0.75 / 3.6 |
| 20 | H13_ep20_b2 | blue_cup: 0.36 / 1.3; water_pitcher: 0.51 / 1.8 |
| 25 | D9_ep25_b4 | wooden_piece_1: 1.19 / 12.6; wooden_piece_2: 1.43 / 4.8 |
| 28 | H13_ep28_b1 | plastic_dish_rack: 0.63 / 1.7; white_pot: 0.59 / 5.7 |
| 30 | H13_ep30_t1p35 | mini_sweeper: 0.48 / 3.3; plastic_dish_rack: 0.34 / 4.0 |
| 32 | G18_ep32_H12e1f0 | blue_cup: 1.13 / 5.4; wooden_spoon: 1.98 / 23.4 |
| 33 | G18_ep33_H13b1 | blue_cup: 0.82 / 4.8; wooden_spoon: 1.25 / 7.5 |
| 35 | G18_ep35_H13b2 | wooden_spoon: 0.90 / 7.6 |
| 36 | G18_ep36_H13b1 | wooden_spoon: 1.47 / 6.6 |
| 37 | G18_ep37_H13b1 | wooden_spoon: 0.54 / 1.7 |
| 38 | H13_ep38_b2 | mini_dust_pan: 0.38 / 3.2 |

Before submission, publish the evaluator build-v2 commit: `git -C /data1/SRL_V2D/.local/worktree_evaluator_build_v2 push kyoungin evaluator-build-v2`. Nothing uploaded.

## Mesh variant: COWORKER MESHES for data-only objects only; OUR meshes for plastic_dish_rack, white_pot_lid, water_pitcher

Mesh-only swap of track3_v30c1_20261101 (poses, results, robot actions unchanged; only the step-0 object_mesh_b64 cells of 17 objects differ). Kept ours: ep1 white_pot_lid (nogt variant: guided mesh excluded), ep14 plastic_dish_rack (nogt variant: guided mesh excluded), ep15 plastic_dish_rack (nogt variant: guided mesh excluded), ep16 plastic_dish_rack (nogt variant: guided mesh excluded), ep20 water_pitcher (nogt variant: guided mesh excluded), ep25 wooden_piece_1 (kit registration twin check failed (180 deg flips across seeds)), ep25 wooden_piece_2 (kit registration twin check failed (180 deg flips across seeds)), ep28 plastic_dish_rack (nogt variant: guided mesh excluded), ep30 plastic_dish_rack (nogt variant: guided mesh excluded). Spoon = G18 (ours).

| ep | object | provenance (coworker README) | kit twin check max rot (deg) |
|---|---|---|---|
| 1 | white_pot | data-only per coworker README (final_episode_meshes_20261029) | 2.7 |
| 3 | white_pot | data-only per coworker README (final_episode_meshes_20261029) | 0.0 |
| 5 | white_pot | data-only per coworker README (final_episode_meshes_20261029) | 0.0 |
| 6 | white_pot | data-only per coworker README (final_episode_meshes_20261029) | 0.0 |
| 8 | beige_cup | data-only per coworker README (final_episode_meshes_20261029) | 0.2 |
| 8 | white_pot | data-only per coworker README (final_episode_meshes_20261029) | 0.0 |
| 10 | beige_cup | data-only per coworker README (final_episode_meshes_20261029) | 0.2 |
| 10 | blue_cup | data-only per coworker README (final_episode_meshes_20261029) | 0.1 |
| 14 | blue_cup | data-only per coworker README (final_episode_meshes_20261029) | 175.9 (about symmetry axis) |
| 19 | mini_dust_pan | data-only per coworker README (final_episode_meshes_20261029) | 0.5 |
| 19 | mini_sweeper | data-only per coworker README (final_episode_meshes_20261029) | 1.1 |
| 20 | blue_cup | data-only per coworker README (final_episode_meshes_20261029) | 165.7 (about symmetry axis) |
| 28 | white_pot | data-only per coworker README (final_episode_meshes_20261029) | 0.8 |
| 30 | mini_sweeper | data-only per coworker README (final_episode_meshes_20261029) | 1.5 |
| 32 | blue_cup | data-only per coworker README (final_episode_meshes_20261029) | 145.0 (about symmetry axis) |
| 33 | blue_cup | data-only per coworker README (final_episode_meshes_20261029) | 0.1 |
| 38 | mini_dust_pan | data-only per coworker README (final_episode_meshes_20261029) | 0.4 |
