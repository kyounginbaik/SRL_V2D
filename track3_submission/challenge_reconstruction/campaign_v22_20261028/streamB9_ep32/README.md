# Stream B9 (phase 9): episode 32 re-planned against target_final_v6

The reconstruction changed in one place (stream Z9): the ep32 wooden_spoon frame 0, which is the scorer's anchor, became its
static-window consensus (a 5.89 deg roll about the long axis, 4.5 mm). The scored scene therefore shifts.
V8_ep32_wb4r2 (v23b) scores cup 0.88 / 9.9 and spoon 2.44 / 9.9 against target_final_v6.
Goal: each object <= 1.3 cm / 15 deg against target_final_v6, aiming for <= 0.8 cm / 8 deg.

Rules followed:
- Our meshes only. The submitted meshes are byte-identical to V8_ep32_wb4r2's: cup 3c3c5328..., spoon c9ab88e3....
- Poses come only from build-v2 policy rollouts.
- Searches ran on node01, node100 and node200; distill, save and stage ran on node01; jobs were <= 50 min.
- Nothing was uploaded, committed or overwritten. Public GT was never read.

## Results (score_any.sh vs target_final_v6, build-v2 re-execution)

| tag | component | cup cm / deg | spoon cm / deg | MP-SR | MPPE / AUC | build-v2 exact | checkpoint / portable archive sha256 |
|---|---|---|---|---|---|---|---|
| **B9_ep32_s6f0** (recommended) | streamB9_ep32/staging/ep32_s6f0/components/episode_000032 | **0.53 / 2.9** | **0.85 / 5.2** | 1 | 0.75 / 0.967 | true | 4f4ed7d130a4c923.. / 6fda40fdd652d175.. |
| B9_ep32_s2f0 | streamB9_ep32/staging/ep32_s2f0b/components/episode_000032 | 0.53 / 3.1 | 0.94 / 5.5 | 1 | 0.80 / 0.965 | true | 50b1a210064e4912.. / b9d8416a2fc0abf4.. |
| V8_ep32_wb4r2 (v23b, before) | streamV8_ep32/staging/ep32_wb4r2/components/episode_000032 | 0.88 / 9.9 | 2.44 / 9.9 | 1 | - | true | 2342dbad.. |

Submitted meshes of both B9 components and both build-v2 outputs: blue_cup 3c3c5328180b5af8b28833e9b82f4a187a3b370ba7a12a63f2a054d133903276,
wooden_spoon c9ab88e33e5c78bfbd5e492d12565effafa8cd2c26d14fe6600e7fb0c1c5d904, byte-identical to V8_ep32_wb4r2's.
Both tags meet 1.3 cm / 15 deg with margin. The cup meets the 0.8 / 8 aim. The spoon is at 0.85 / 5.2: the orientation aim is met, the position aim is 0.05 cm short.
Full archive sha256: s6f0 6fda40fdd652d1757b6daa6dcb12a972824be8d0c48a31513d3575cd7da7055a; s2f0 b9d8416a2fc0abf4f922efe43e32a7e91b68b7ee2e5662e0d070716122dcca87.

Build-v2 outputs are in `campaign_v16_20261023/common_v2_candidates/<TAG>/episode_000032` (completed.json has
`exact_previous_physical_states_actions` true). Scores are in `campaign_v22_20261028/scores/cand/<TAG>/score.json`.

## 1. World W6 (`code/build_world_b.py`, `code/blib.py`, `code/cmpworld.py`)
- This is the unchanged streamB recipe. The only edit is the copy of `build_world_b.py`/`blib.py`, whose target dir is now target_final_v6.
  Same base scene (streamR regrasp/pick_full/power0.6), base episode dir, shipped meshes and `--short 0.0178`.
  - **Reproduction check:** the original streamB script on target_final_v5 rebuilds W1 byte-identically
    (scene 4c031e3a..., object_poses eb9e270d...; `tmp/W5check`).
- Steps:
  1. target_final_v6 tracks are converted into our body frames by the scorer's MP-SR registration.
  2. They are mapped into the physics world by ONE rigid transform G = Gt @ G0:
     - G0 = frame-0 Kabsch of both objects onto our previous layout;
     - Gt = minimal tilt about the frame-0 cup base, plus a lift that puts the shipped cup bottom and the lowest spoon point on the table;
     - cup yaw about its symmetry axis is fixed (scoring-invariant).
  3. The physics scene starts the objects at the new frame 0, so the spoon's initial pose is the v6 consensus pose.
- Record: `W6/source_corrected/episode_000032/target_retarget.json` (sha256 56884486779a5466...).
  - World transform X, tilt 0.469 deg, lift 0.117 cm, cup yaw fix 21.14 deg.
  - Frame-0 contacts: cup|table 0.91 mm; robot self-contacts unchanged.
- Hashes:
  - scene `W6/scene/scene_physics.xml` 36779927ac0c28c1...; corrected object_poses a4e203e1ebcda397...;
  - target tracks: cup 99d1601e... (identical to v5), spoon 61385ca3... (v5: 4ed482e6...).
- W6 vs W1 (`cmpworld.py`):
  - X6 X1^-1 = 0.53 deg / 0.53 cm.
  - Reference shift: cup 0.18 cm / 0.52 deg mean; spoon 0.22 cm / 0.55 deg mean.
  - Spoon initial pose: rolled 5.37 deg, 0.19 cm.
  - So the target motion barely moves in the physics world. Only the spoon's starting roll changes.

## 2. Port (`code/port.py`)
- V8_ep32_wb4r2's actor (checkpoint 2342dbad...) was evaluated in W6. As in streamP port_actor.py, only the scene binding changes:
  checkpoint scene_sha256, and the training report's scene / scene_sha256. The weights are unchanged.
- Port result: cup 8.47 / 45.7, spoon 8.67 / 45.8. The chaotic in-hand hold re-rolls from the different spoon start; the spoon is lost after 2.5 s.
- The ported actor (`runs/port_wb4r2b/actor`) is the base of every search (residual table = 0).

## 3. Re-plan: window beam search with exact cached states (`code/wbeam2.py`, `code/wbeam3.py`; copies of V8 wbeam2.py)
- Changes from the V8 copy:
  - scorer target = target_final_v6 (B9 `mlib.py` copy);
  - the cache frame is clamped at 0, so windows can start at 0.4 s;
  - `--init` resumes a beam from saved tables;
  - wbeam3 adds time-advance candidates for the right-wrist command. They were never selected.
- Per 0.5 s window, every beam parent spawns:
  - random right-wrist/finger knots;
  - right-hand rigid-hold spoon ILC;
  - left-hand axis-only cup ILC.
- Each child rolls out from the parent's exact MuJoCo state. Children are ranked by the cumulative scorer-faithful per-frame cost.
  Exactness checks: an unedited child equals its parent bit for bit, and the RBF leak before the cache frame is 0.
- Chain:

| run | init | windows | beam / n / look | tol (spoon P,D) / cup weight | full-rollout result cup / spoon |
|---|---|---|---|---|---|
| s1 | port | 0.4-12.9 | 8 / 48 / 1.2 | 1.3,15 / 1.0 | 0.47/4.1, 0.95/6.6 |
| s2 | port | 0.4-12.9 | 6 / 64 / 1.5 | 1.1,18 / 1.0 | 0.53/3.1, 0.94/5.5 -> B9_ep32_s2f0 |
| s3 | s2 beam | 1.4-12.9 | 6 / 48 / 1.5 | 1.0,12 / 0.7 | 0.58/3.0, 0.91/5.9 |
| s4 | s1 beam | 1.4-12.9 | 8 / 48 / 1.5 | 0.9,10 / 0.5 | 0.51/4.0, 0.89/5.7 |
| s5 (wbeam3) | s4 beam | 1.9-12.9 | 8 / 32 / 1.5 | 0.9,10 / 0.5 | 0.47/3.1, 0.86/5.6 |
| s6 (wbeam3) | s3 beam | 1.9-12.9 | 6 / 32 / 1.5 | 0.8,10 / 0.5 | 0.53/2.9, 0.85/5.2 -> B9_ep32_s6f0 |
| s7 / s8 | s6 / s5 beam | 1.9-9.4 only | | 0.8,10 / 0.4 | tail after 10.9 s diverged: best 0.58/3.6, 0.86/5.3 and 0.75/3.6, 0.84/6.8 (rejected) |
| s9 (wbeam3) | s6 beam | 1.9-12.9 | 8 / 48 / 1.5 | 0.85,10 / 0.6 | 0.59/2.9, 0.84/5.5 (not better overall; not staged) |
| s10 (wbeam3) | s6 beam | 1.9-12.9 | 6 / 40 / 2.0 | 1.0,12 / 0.7 | 0.60/3.6, 0.99/8.4 |

The spoon plateaus at about 0.84-0.86 cm. Each further round gains <= 0.01 cm or trades cup for spoon.

Rollouts were saved as actors with `mres.py save` (official evaluator; its score equals score_any). They were staged with the `code/stage_b.py`
copy (streamB stager: world dir W6, previous component V8_ep32_wb4r2, lineage names v6). Then `run_one.sbatch` was run (node01, 50 min).

## Error structure (B9_ep32_s2f0, `code/diag_spike.py`)
- The spoon spike at 2.45-2.75 s (up to 6.7 cm) is lift lag: the target lifts the spoon 1.8 -> 9 cm in 0.35 s, and the rollout lifts about 0.2 s later.
  Time-advance candidates for the wrist lift did not survive the cumulative ranking.
- 6.85-7.5 s: a steady 2.2 cm in-hand offset.
- 7.75-8.1 s: the target spoon jumps 2-4 cm between consecutive frames. This is a reconstruction-jitter floor.
- The rest of the episode sits at 0.2-0.6 cm.

## Caveats
- Proxy scores are against our reconstruction (target_final_v6), not hidden GT.
- The hold is chaotic. A result is one specific trajectory: it is exact under build v2 but not robust to edits.
  Windows that stop before the end (s7/s8) let the tail diverge, so every adopted run searched to 12.9 s.
- `staging/ep32_s2f0` is an aborted first staging attempt. The port's training report still named the W1 teacher scene, which the stager rejects.
  It was redone as ep32_s2f0b after the port also rebound the report's scene fields (streamP convention).

## Files
- code/: build_world_b.py, blib.py, cmpworld.py, port.py, mres.py, mlib.py (v6 copies of M8), wbeam.py, wbeam2.py, wbeam3.py, diag_spike.py,
  stage_b.py, stage_b.sbatch, save_chain.sh, res.sbatch.
- W6/: world (source_corrected, episode_corrected, scene). runs/: port_*, s1-s10, save_*, eval_*. staging/: ep32_s2f0b, ep32_s6f0 (+ later). logs/.
