# Stream K35 (2026-11-02): episode 35 (spoon only) - why it fails official MP-SR, GT-free fix, re-retarget, diagnostic packages

Rules followed: no public scan / GT mesh / GT pose anywhere (there is no public spoon); no constant fit to any GT or leaderboard number;
poses only from build-v2 policy rollouts (window-beam phase residuals on the v29 actor, official save + unchanged gates); nothing
overwritten or deleted (a failed mkref output was renamed `worlds/_failed_mkref_67731_H1rest`); Slurm debug on node100 (node01 hung),
<= 48 CPUs, every job < 50 min; no upload, no commit.

## Official facts used
- v29 ep35 FAILS (mesh-knockout on v30c1nogt stayed 0.85). Swaps of our older ep35 components into v29 also fail: v26 (5ae092ba),
  v22b (0e1cf39b), v21C (436016cc; polled from its receipt, 0.85).

## 1. Diagnosis (GT-free)
- **Mechanism (kit):** ep35 has one object, so `_align_registered_scene` aligns our whole track to the truth by the spoon's FRAME-0 pose.
  A constant roll error at frame 0 rotates the entire trajectory about the spoon's frame-0 long axis. Lever measured with the kit
  functions on our own target (`code/sens.py`, `out/lever_sens_v14.json`): ep35 0.23 cm/deg roll (10 deg -> 2.3 cm, 15 deg -> 3.4 cm);
  ep32/33 0.35 cm/deg; ep36 0.20; ep37 0.10.
- **Frame-0 roll of v14 is physically impossible** (`code/rest2.py`): in the gravity frame of our stereo support plane, our rigid spoon mesh
  at the tracked f0-25 poses is NOT statically stable; dropped onto the plane it would settle at roll -14 deg (per-frame -7..-20, median -14,
  COM 1.3 mm lower). At the END rest (f390-420, same face down, same table) the tracked pose IS stable (rest roll -2..+3).
  The start and end rest poses differ by a 13 deg tilt about the long axis although both lie flat on the same table (`f0->f420` rotation axis
  13 deg off vertical).
- **The physics sim agrees:** v29's rollout lies flat on the sim table at the end; vs v14 its end orientation error is 12.8-13.6 deg
  (f400-420); vs the corrected start (H1 below) it is 1.8-2.2 deg (`code/perframe.py`).
- **Official consistency:** all four officially failing ep35 variants fail vs H1 (v29 3.38 cm, v26 4.0, v22b 4.07, v21C 3.33), while v29 vs
  v14 is 0.90 cm / 7.4 deg (a pass) - i.e. v14 cannot be the truth, H1 can.
- Image evidence on f0-24 (`code/imgev.py`, our SAM2 mask / stereo depth / 2D balls): silhouette IoU is flat (0.648-0.660), stereo depth
  residual improves monotonically toward negative roll (5.20 mm at 0 -> 4.80 at -12), the 2D balls prefer -4 (4.33 px; 4.90 px at -12).
  The weak image cues do not contradict the physics; inverse-variance joint estimate (physics -14+-4, end-pose match -11+-3, balls -4+-6) = -11.
- Camera path is not the cause (support-plane normal drift p50 0.75 / p95 1.6 deg); the kit spoon-mesh registration is not seed-flipped
  (main's check). Same pattern elsewhere (not acted on here): ep33 start rest-roll -7..-16 at 0.35 cm/deg lever; ep36 start 0 (passes).

## 2. Hypotheses (bundles/<H>/episode_000035, targets/<H>; `code/build_h.py`)
v14 (selected20_final_v14_g17) + a body roll of the spoon about its long PCA axis through the COM on the static window f0-25, smoothstep
ramp back to the tracked track by f40 (lift-off is at f26-27); every other frame byte-identical.
- **K35_H1rest:** c = median physical rest roll on f0-25 = -14 deg.
- **K35_H2joint:** c = -11 deg (joint image + physics + end-pose estimate).

## 3. Re-retarget (build-v2; `code/c9beamH13.py` E31/H13a copy, TGT via E31_TGT; worlds via `code/mkref.py`; staging `code/stageK_k35.py`)
- From the v29 ep35 actor (H13_ep35_b2, streamH13a/runs/save_ep35_b2/actor), 0.5 s windows 1.25-21 s, beam 5, ILC alphas up to 1.3.
- Worlds `worlds/<H>/ep35/episode_corrected/episode_000035`: copy of the actor's world with the object reference (tracking_reference
  qpos + source object_poses) replaced by the hypothesis target in the physics world (needed because the stager/evaluator gate the
  proxy MP-SR against the world reference). `stageK_k35.py` = stageK.py with one change: the episode-dir check also accepts K35 worlds.
- Results (kit-faithful score_any, G18 mesh, build-v2 exact):

| tag | vs own target | vs other hyp | vs v14 |
|---|---|---|---|
| K35_K35_H1rest_b1 (from v29 actor) | H1: **0.92 cm / 7.2 deg**, MPPE 1.06, AUC 0.951 | H2: 1.07 / 7.6 | 3.11 / 14.7 (MP 0) |
| K35_K35_H2joint_b1 (from v29 actor + H1 b1 table) | H2: **0.63 cm / 6.2 deg**, MPPE 0.77, AUC 0.974 | H1: 1.01 / 6.5 | 2.62 / 13.3 (MP 1, thin) |

## 4. Diagnostic packages (`code/k35splice.py`: v29 parquet with only ep35 rows rebuilt from the candidate; `--check` reproduces v29's
own ep35 rows byte-exactly from its v29 candidate; non-ep35 rows identical to v29; uniform provenance kept)
- `challenge_submissions_SRL/ep35_fix_20261102/K35_H2joint_b1/all_20_episodes` (parquet sha 7e469bd0..., dry-run OK) - recommended first:
  within 1.5 cm / 15 deg of BOTH hypotheses.
- `challenge_submissions_SRL/ep35_fix_20261102/K35_H1rest_b1/all_20_episodes` (parquet sha 9eccda17..., dry-run OK).
- Reading: MP-SR 0.90 -> ep35 fixed (the frame-0 rest-roll diagnosis is right); 0.85 -> ep35 still fails.
