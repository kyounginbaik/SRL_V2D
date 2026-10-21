# Grasp-stability test (episode 37 spoon)

Isolated, Dexonomy-style stability test of all 21 Dexonomy grasp types fitted on the episode-37 wooden spoon, plus the
UniRetarget-refined lateral tripod with its trained lower policy (bottom row of every page).

Each panel: (1) the hand is placed at the fitted grasp around the spoon with the fingers open (no table, gravity off) and the
fingers close; (2) the grasp settles for 0.5 s; (3) a force on the spoon ramps from 0 to 3 m*g over 3 s along one axis per column
(+x -x +y -y +z -z), drawn as a red arrow. Green border = spoon still in hand (< 2 cm and < 15 deg); red = lost, with the breaking
force. 4 pages (6, 6, 6, 3 grasps). Breaking forces also in `episode_000037/demo/iso37_p<k>.json`.

## Run
Requirements: NVIDIA GPU with a CUDA 12.x driver (RTX 3090 / 4090 / A100 / L40S; < 8 GB VRAM), Docker + NVIDIA Container Toolkit,
`git lfs pull` in the repo (hand meshes), and the flash_chord image built once:

    docker build -t flash_chord:ur-pinned-22d733 -f episode_000037/flash_chord/docker/Dockerfile episode_000037/flash_chord
    ./run_grasp_test.sh                # all 4 pages on GPU 0, ~15 min per page
    ./run_grasp_test.sh "1" 2          # page 1 only, on GPU 2

Output: `episode_000037/demo/iso37_p1.mp4 ... iso37_p4.mp4`.

Contents: `episode_000037/flash_chord` (the exact simulator code version used), `frozen/` (episode scene + spoon mesh),
`priors/screen/` (21 fitted grasp references + fits), `policy/` (refined reference v10 + lower policy checkpoint),
`camera_motion.npz` (ego camera, for the close-up view), `code/` (renderer + run config).
