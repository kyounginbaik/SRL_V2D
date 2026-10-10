# UniRetarget: Track 3 experiment

UniRetarget alternates lower-level FlashSAC policy learning with upper-level FlashSAC contextual-bandit reference edits. The upper learner evaluates proposals with physics rollouts; it does not differentiate through the simulator.

## Current configuration

- Floating Sharpa hands, pot and lid, 20 Hz reference.
- HaWoR hand reconstruction and tuned Mink initialization, with episode-specific left-hand grasp alignment performed before training.
- 1-second reference chunks; bounded wrist translation, SO(3) rotation and finger corrections.
- 2,048 environments per seed; seeds 42, 123 and 456; 251,658,240 environment steps per seed.
- Upper warmup and interval: 5,033,984 steps; four trials per update.
- Performance-gated VOC: probe the next lower assistance level every 1,048,576 steps. Both objects and hand tracking must pass for three consecutive probes. Failure holds the current level. Final evaluation uses zero assistance.

## Code entry points

- `robotic_grounding/flash_chord/scripts/train_flash_sac.py`: training entry point.
- `src/flash_chord/retargeting/manipulation_bilevel.py`: alternating optimization lifecycle.
- `src/flash_chord/retargeting/ttsa_editor.py`: proposal evaluation and acceptance.
- `src/flash_chord/retargeting/floating_chunk.py`: bounded floating-hand reference edits.
- `src/flash_chord/lifecycle/performance_voc.py`: multi-object performance gate.

The accompanying `run_episode.py` is the actual seed-42 experiment wrapper. Set `V2D_TRANSFER_RUN` to a directory containing `episode_000001/reference.parquet` and adjust the submission-kit path and `/workspace` checkout for your installation. Other seeds require changing both `reset.seed` and `training.seed`. The default budget is 251,658,240; `V2D_TRAIN_STEPS` overrides it. Invoke inside the configured FlashCHORD environment with `python run_episode.py episode_000001`. Existing partial runs are rejected rather than silently restarted. Reference datasets, robot assets, checkpoints and credentials are not included.

## Interpretation and limitations

The manually corrected grasp is an initialization correction, not a newly added contact loss or a guarantee against penetration. Mesh CD-O and policy success are distinct metrics. At the recorded status check around 62M steps, neither active seed had full-reference completions in training metrics, and VOC remained 1.0. Do not interpret a valid reference file or an attractive kinematic preview as physics success.
