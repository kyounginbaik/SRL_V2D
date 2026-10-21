# GCRL in-hand transition (episode 36 spoon)

Stage 3 of the UniRetarget pipeline (stable grasp -> **in-hand transition to the functional grasp** -> task). A goal-conditioned policy
moves the hand from a stable start grasp A to a goal grasp B while the spoon is held still in the air (it must not drop or move).
Each of the 2048 training worlds gets its own (A, B) pair: hold A 0.5 s -> transition 1.5 s -> hold B. The goal reaches the policy through
the commanded hand/object targets in its observation. 6 stable start grasps (large diameter, power sphere, fixed hook, light tool,
extensor type, small diameter) x 31 goals (21 Dexonomy types, the video's own grasp, 8 human clusters, 1 learned grasp) = 186 pairs.

| arm | algorithm | goal reward |
|---|---|---|
| `flashsac` | FlashSAC (off-policy SAC, distributional critic) | dense: hand + object tracking |
| `ppo` | PPO | SimToolReal-style: progress against a running best of hand-goal closeness + bonus while the goal grasp is held |
| `tldr` | PPO (same as `ppo`) | TLDR-style: decrease of a learned temporal distance ||phi(x)-phi(x_goal)|| + same bonus |

All arms: object-tracking reward, episode ends if the spoon moves > 5 cm / 0.5 rad, no assist force on the object.

## Run
Requirements: NVIDIA GPU (CUDA 12.x driver; RTX 3090/4090/A100/L40S), Docker + NVIDIA Container Toolkit, `git lfs pull` in the repo
(hand meshes), the flash_chord image built once (same as experiments/grasp_stability_test):

    docker build -t flash_chord:ur-pinned-22d733 -f episode_000036/flash_chord/docker/Dockerfile episode_000036/flash_chord
    ./run_gcrl.sh ppo 0          # train arm B on GPU 0 (30M steps), then evaluate + render
    ./run_gcrl.sh tldr 1         # arm C on GPU 1 (in parallel)
    ./run_gcrl.sh flashsac 2     # arm A
    ./run_gcrl.sh eval-ppo 0     # re-evaluate the latest checkpoint only
    ./run_gcrl.sh ppo 0 2000000  # quick smoke run (2M steps)

Outputs: `episode_000036/demo/gcrl_<arm>_transitions.mp4` (light blue = GOAL hand pose; border green = spoon kept and goal reached,
orange = kept but goal missed, red = dropped) and `gcrl_<arm>_transitions.json` (per pair: kept, goal_reached, dropped, max object
motion, final finger error; overall success / kept / reached / dropped rates). Checkpoints and metrics in `episode_000036/runs/`.

Contents: `flash_chord/` (simulator + trainers, with the contact-force observation), `frozen/` (60-frame task: spoon held still at its
mid-hold pose in episode 36, + spoon mesh), `priors/dexfit_types.npz` (all fitted grasps on the spoon), `code/` (the three training
scripts, evaluation/video, run config).
