# Stream H10 (phase 9): ep32 and ep3 re-retargeted onto target_final_v8_g10b, 2026-10-29

Target: `campaign_v22_20261028/target_final_v8_g10b` (bundle `reconstruction_recovery_20261026/selected20_final_v8_g10b`, stream G10).
- Compared with final_v6, only the wooden_spoon track changes, in two episodes. Frame 0 is unchanged in both.
  - ep32: a roll ramp f47-65, then the spoon is held flipped by 135-165 deg through f65-151, then a 3-frame unroll at f151-153.
  - ep3: a 105 deg spike at f48-58, a flip of 60 -> 180 -> 135 deg over f100-161, and a 45 deg plateau at f211-227.
- Meshes are unchanged.

Rules followed:
- Our meshes only. Submitted meshes are byte-identical to the previous components: ep32 cup 3c3c5328 / spoon c9ab88e3; ep3 pot 8cbd3543 / spoon c9ab88e3.
- Poses come only from build-v2 policy rollouts. Nothing was hand-edited.
- CPU work ran on node01, and every job was at most 50 min. Nothing was overwritten, uploaded or committed.

## Results (score_any vs target_final_v8_g10b; every tag build-v2 exact_previous_physical_states_actions = true)

| ep | tag | component (streamH10_v8/...) | obj 1 cm/deg | spoon cm/deg | MPPE | AUC | MP |
|---|---|---|---|---|---|---|---|
| 32 | **H10_ep32_H2f0** (recommended) | staging/ep32_H2f0/components/episode_000032 | cup 0.61/3.0 | **0.97/19.2** | 1.106 | 0.948 | 1 |
| 32 | H10_ep32_G1f0 | staging/ep32_G1f0/... | 0.61/3.2 | 0.96/20.9 | 1.170 | 0.949 | 1 |
| 32 | H10_ep32_F2f0 | staging/ep32_F2f0/... | 0.60/3.3 | 0.88/27.2 | 1.261 | 0.947 | 1 |
| 32 | H9_ep32_h19f0 (before) | streamH9_ep32v7/staging/ep32_h19f0 | 0.55/2.4 | 0.73/31.9 | 1.29 | 0.953 | **0** |
| 3 | **H10_ep03_H2f0** (recommended) | staging/ep3_H2f0/components/episode_000003 | pot 0.41/3.4 | **0.75/11.9** | **0.784** | **0.968** | 1 |
| 3 | H10_ep03_G2f0w | staging/ep3_G2f0w/... | 0.46/4.0 | 0.89/12.6 | 0.891 | 0.962 | 1 |
| 3 | H10_ep03_F2f0 (staged in the old streamA world) | staging/ep3_F2f0/... | 0.47/4.5 | 0.87/16.0 | 0.959 | 0.959 | 1 |
| 3 | D9_ep03_b3 (before) | streamD9_tight/staging/ep3_b3 | 0.43/3.8 | 0.72/28.8 | 1.15 | 0.973 | 1 (1.2 deg margin) |

- Build-v2 outputs are in `campaign_v16_20261023/common_v2_candidates/<tag>/episode_XXXXXX`. Scores are in `campaign_v22_20261028/scores/cand/<tag>/score.json`.
- Checkpoint sha256 (first 12): H10_ep32_H2f0 1a25a5e1b3d3; H10_ep03_H2f0 7de9c322d549.
- Portable archives (`staging/<dir>/portable_archives/episode_XXXXXX_portable.tar.gz`): ep32_H2f0 5616a2f9989e; ep3_H2f0 1c582ee8d668.

Goal status:
- **ep3 H2f0** meets every goal: each object <= 1.0 cm / 10 deg except spoon orientation at 11.9 deg, which is inside the 12 deg aim; MPPE 0.78; AUC 0.968.
- **ep32 H2f0** misses the goal, though it is well inside the hard limits.
  - Met: spoon position 0.97 cm and cup 0.61 cm / 3.0 deg. It passes MP-SR (19.2 deg vs the 30 deg limit), where H9_ep32_h19f0 failed at 31.9 deg.
  - Not met: spoon orientation 19.2 deg (goal 10), MPPE 1.11 (goal 1.0), AUC 0.948 (goal 0.95).

Hedge (if final_v6 turns out to be the truth):
- H10_ep03_H2f0 vs target_final_v6: spoon 0.75/31.2, MP 0.
- H10_ep32_H2f0 vs target_final_v6: spoon 0.97/50.3, MP 0, SP 1.
- Choosing these components is therefore a bet on the G10 reconstruction, just as v25 was a bet on v7_g9rot.

## Method
- **ep32 world W8** (`code32/build_world_b.py`): the unchanged H9/B9 recipe with the target set to v8_g10b.
  - The scene is byte-identical to W7/W6 (36779927...), and X, tilt and lift are the same, because frame 0 did not change.
  - B9_ep32_s6f0 was ported into W8 as `runs/ep32/port_s6f0` (vs v8: spoon 56.6 deg). H9's h19 beam tables apply unchanged on this actor.
- **ep3 world W8** (`code3/mkworld3.py`, `ep3/world/`):
  - The streamA ep3 world (target_final_v5 reference) is re-referenced to v8_g10b using the same G and the same scorer registration X. The v5 and v8 target meshes are byte-identical.
  - The physics scene is copied byte-identically.
  - Q8 g11 was ported with `code3/port3.py`. Its rollout is identical (`same_rollout: true`).
  - The stager is a copy, `code3/stage3.py`; only the world-path check points at the H10 world. This was needed because the stager gates on the evaluator proxy against the world reference.
- **Search:**
  - ep32 uses `code32/wbeam4.py` (the H9 copy). ep3 uses `code3/wbeam9.py`, a D9 copy extended with H9's roll-only twist ILC, ILC+squeeze candidates and `--init`. `code3/mres9.py` gains H9's time-continuous ILC.
  - **New: a per-frame grasp-loss hinge** `--hinge w,cm`, which adds w * max(0, pos - cm) / P per frame and object.
  - Without the hinge, chasing the roll slipped or dropped the spoon (round r1: within-window deg 12-22, but 2-7 cm off and the tails were lost). Half passes also left diverged tails.
  - Every useful round was therefore a full-horizon pass: 0.5 s windows (0.25 s on the roll ramps), look 1.2-1.5 s, hinge 4-5 at 1.5-2 cm.
- **Chains:**
  - ep32: h19 -> F1/F2 (25.4 / 27.2) -> G1 (20.9) -> H2 (19.2). The later rounds I1/I2/J1/J2 collapsed: the cup got knocked or the spoon was lost (greedy beam), so they were not used.
  - ep3: D9 b3 -> F2 (16.0) -> G1/G2 (12.8 / 12.6) -> H2 (11.9). H1 (14.3) was not used.

## Remaining error (ep32)
- The roll is held only partially. The hand cannot sustain a 135-165 deg in-hand roll of the thin handle without torsional slip (as found in H9).
- The 3-frame unroll at f151-153 (about 800 deg/s) and the ramp at f47-65 are physically out of reach.
- Gains were 5.7 -> 1.7 deg per round before the beam started collapsing.

## Files
- `code32/`: build_world_b.py, blib.py, mlib.py, mres.py, wbeam4.py (+hinge), port.py, stage_b.py, save_chain.sh, st_v2.sh, eval32.py, w8.sh.
- `code3/`: mlib9.py, mres9.py, wbeam9.py (+roll, hinge, init), mkworld3.py, port3.py, stage3.py, save/stage.sbatch, svst_v2.sh, eval3.py, env.sh.
- `W8/` (ep32 world); `ep3/world/` (ep3 world).
- `runs/ep32/{port_s6f0, r1a-d, F1, F2, G1, G2, H1, H2, I1, I2, J1, J2, save_*}`; `runs/ep3/{port_g11, r1a, r1c, r1e, F1, F2, G1, G2, H1, H2, save_*}`.
- `staging/`; `logs/`.
