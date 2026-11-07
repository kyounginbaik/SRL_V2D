# UniRetarget, Newton practice version (OakInk-v2 whiteboard writing)

Snapshot of the working code as of 2026-11-07. It is research code, not a finished release. Experiment history, numbers and negative
results are in `write/RESEARCH_LOG.md`.

## Algorithm (current)

Two levels. Everything runs in the training simulator (Newton/MJWarp), with no IK and no MuJoCo.

1. **Upper level: reference search by open-loop practice** (`write/scratch.py`, `write/scratch_tr.py`)
   - Input: only the object path (pen pose over time). No demo hand is used.
   - A candidate reference is a rigid grip, 13 numbers:
     - pen pose in the hand frame (offset from the hand's thumb/index/middle tripod point, from Newton forward kinematics, plus a rotation);
     - grip point along the pen;
     - 5 finger closures;
     - thumb opposition.
   - The hand reference = object path composed with the inverse grip, fingers constant.
   - Scoring is open-loop: the policy residual is zero, so the PD hand just follows the candidate, with no assistance force.
     - K=128 candidates are tiled in time in one batch (env `PIB_SEG`/`PIB_PROBE`).
     - 2048 worlds with per-world friction randomization (env `FRIC_DR`), about 16 trials per candidate.
     - Fitness = writing frames with the tip within 3 cm (normalized) + 0.5 x fraction of frames held. Held = pen within 3 cm of its place in the hand.
   - Search: random grips in generation 0, then 8 parents x 16 children, with parents re-measured each generation.
   - `scratch_tr.py` (with `PIB_TR=1`) scores each candidate after a training-style reset (fingers partly open, 20-step assisted settle).
     Round 7/8 evidence suggests this is the condition that matters for RL.
2. **Lower level: RL with no assistance** (FlashSAC, `oak_root/train.sbatch` arm `track`, `CURR=write_100m_novoc`)
   - Tracks the exported reference snapshot (`task.robot_reference_path` / `robot_reference_sha256`), with no virtual object assistance (VOC) at any point.

## Key findings so far (details in RESEARCH_LOG.md)
- The VOC assistance curriculum was the main obstacle. With no assistance, a reference that holds open-loop (planner grip g0) passes in 4/4 runs.
- Open-loop hold from the exact reference pose is not sufficient: a demo grasp edited to hold 92% open-loop still failed RL (0/2).
  The hold after a training-style reset is the current candidate criterion (only n=3 references so far).
- MuJoCo checks do not transfer to Newton for this thin object.

## Layout
- `flash_chord/`: FlashCHORD copy with the patches used here:
  - `envs/base.py`: `FRIC_DR` friction randomization, applied once per model;
  - `lifecycle/reset.py`: `PIB_SEG`, `PIB_PROBE`, `PIB_TR` tiled-slot resets;
  - `lifecycle/termination.py`;
  - `retargeting/assist_refine.py`, `retargeting/practice_batch.py`: earlier reference-update variants, now dropped.

  Large assets (meshes, policies) are not included; fetch them as in the original FlashCHORD repo.
- `write/`: task scripts.
  - Upper-level search: `scratch*.py` / `oloop.py` with their `.sbatch` files.
  - Open-loop hold test: `holdany.py`.
  - Newton headless rendering and finger-pen distance: `nrender.py` + `mk_mp4.py`.
  - Scoring: `score_*.py`, `wmetric.py`.
  - Clip tiling: `tile_clip.py`.
  - Pipeline chaining: `chain_scratch.sh`.
- `oak_root/`: eval scripts (`eval.py`, `eval_ref.py` for zero-residual rollouts, `eval_robust.py`), `train.sbatch`, curricula (`conf/curriculum`).

Paths in the sbatch files point at the yonsei-ai cluster (`/scratch2/exaFLOPs/srl_yam_run/...`, container `flash_chord_yam.sif`).
