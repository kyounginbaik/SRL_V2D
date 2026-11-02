# UniRetarget on YAM + Sharpa

Self-contained bilevel test: FlashSAC tracks the current reference (lower level), and `flash_chord/src/flash_chord/retargeting/yam_bilevel.py` updates the reference from simulated outcomes (upper level). This folder holds everything episode 37 needs, and nothing outside it is read at run time.

## Setup

Needs Docker with the NVIDIA Container Toolkit. `flash_chord/docker/Dockerfile` states host driver >= 580.

```bash
git clone -b uniretarget/yam-bilevel --single-branch <repo-url> SRL_V2D
cd SRL_V2D/uniretarget_yam
DOCKER_BUILDKIT=1 docker build --target dev -t flash_chord:yam flash_chord
```

Clone with `GIT_LFS_SKIP_SMUDGE=1` to skip the LFS files elsewhere in the repo; this folder does not use them.

## Run

```bash
GPU=0 ./run.sh U    # bilevel: human reference, upper level refines it
GPU=1 ./run.sh A    # FlashSAC on the fixed human reference
GPU=2 ./run.sh C    # FlashSAC on the fixed refined reference
```

Each arm needs one GPU and writes to `out/episode_000037/arm_<ARM>-seed42/`. Hydra overrides can follow the arm, for example `./run.sh U scene.world_count=1024` if GPU memory runs out.

Upper-level rounds start at 30M steps and repeat every 20M, of 250M. Follow them in `upper_updates.jsonl` in the output folder of arm U; refined references go in its `references/` folder.

## Not yet verified

- FlashSAC has never completed training on YAM, so training speed and GPU memory at 2048 worlds are unmeasured.
- The image built here comes from the same Dockerfile as `flash_chord:ur-pinned-22d733` on the original server, but is not byte-identical to it.

## Layout

- `flash_chord/`: trainer and upper level
- `code/`: YAM search and tools. Scripts other than the ones `run.sh` uses may still contain original-server paths.
- `episodes/episode_000037/`: references, URDF, compiled MuJoCo model
- `objects/`: spoon mesh and collision parts
- `robot/`: YAM and Sharpa meshes with their licenses
