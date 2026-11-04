# Episodes 32 and 33: how the reconstruction and retargeting were done

Written 2026-11-03 from our logs and build records (`challenge_reconstruction/campaign_v22_20261028/STATUS.md`,
`reconstruction_recovery_20261026/README.md`, the stream READMEs, `V32_BUILD.json`).
Current best package: `challenge_submissions_SRL/FINAL_20261104` (sha 0fca055b).

**Status: both episodes still fail MP-SR.** Everything below describes the pipeline that produced the failing result, plus what we now know is wrong with it.

Both episodes have a `blue_cup` (slot 0) and a `wooden_spoon` (slot 1). The person lifts the cup with the left hand and repeatedly scoops the spoon into it with the right hand.

| | ep32 | ep33 |
|---|---|---|
| frames at 20 fps | 259 (12.95 s) | 348 (17.4 s) |
| rows used in the best package | `K32_R10_R10_c1` (checkpoint dd904268) | `K33_H5r16_c3` (checkpoint 381aa365) |
| target | v14 + spoon roll -10 deg on f0-20, ramp out by f35 | v14 + spoon roll -16 deg on f0-25, ramp out by f40 |

## Part 1: Reconstruction

No public or ground-truth mesh or pose is used. Public GT is used only by the scorer on public clips, to validate rules.

### Step 1. Prepare the inputs (`tools/reconstruction_audit/canonical.py`)
- Undistort ego_cam_a RGB with the supplied calibration.
- Project FoundationStereo depth from the ego_cam_b/c pair into the RGB camera.
- Object masks are from SAM (`objects/<object>/masks`). Output: `canonical/evaluation/episode_0000NN/`.

### Step 2. Camera path (`camera.py`)
Metric motion from static-background SIFT with object and hands masked, stereo lifting, and PnP-RANSAC refinement.

### Step 3. Several independent object trackers
Candidate tracks per object: raw stereo FoundationPose, CoTracker point map, multimap, bundle adjustment, motion odometry, and the half-turn "twins" of the raw track.

### Step 4. Fuse the candidates (`fuse.py`, version `fused_v7`)
Per-frame Viterbi choice among the candidates, scored by mask IoU and stereo depth with rotation continuity. Version v7 removed all hand-derived inputs, because a wrist prior had dragged a spoon off track.

Sources chosen for the ep33 spoon: half-turn twin 159 frames, raw 60, odometry 55, multimap 46, bundle adjustment 23, point map 5. So the spoon track is stitched from many sources, and the twin is the largest.

### Step 5. Clean up
- `gap_bridge.py` fills unobserved frames.
- `despike.py` removes out-and-back rotation excursions of up to 3 frames.
- `rescue.py` re-selects from raw trackers when IoU collapses. It fired on ep32 for 2 frames and ep33 for 3.
- Output: `fused_v7_gap_ds_rs`.

### Step 6. Spoon roll from mocap balls (`spoon_roll`, `joint_refine.py`)
The spoon is nearly symmetric in silhouette, so roll comes from the 2D positions of its 3 gray mocap balls, whose layout is chiral. ep32 additionally got 6-DoF joint refinement, because more than half its frames were tracked half-turned. Result: `selected20_final_v5`.

### Step 7. ep32-specific roll changes, driven by official scores
These are the reason ep32 has `z9_anchor.json`, `g10_spoon_roll.json` and `g12_spoon_roll.json` in its bundle and ep33 does not.
- final_v6 (Z9): frame-0 spoon consensus.
- final_v7_g9rot and final_v8_g10b: spoon roll changes.
- final_v10_g12: a consensus roll chosen to agree with official v24 to v26 and the balls. The spoon is flipped about 105 to 165 deg on f45-178.

**Caution:** this target was partly chosen to be consistent with official results. It is not purely image evidence.

### Step 8. Clock fix (G15, `final_v12b_g15`)
RGB lags the stereo pair by about 0.9 frame, measured from optical-flow cross-correlation between cameras. All poses were resampled onto the stereo clock and the camera path was made clock-consistent. On public clips, tuning MPPE went from 1.226 to 1.093 and held-out from 1.117 to 1.035.

### Step 9. Base reconstruction "v14" (`selected20_final_v14_g17`)
v12b + ep3 spoon roll fix. G16 found a constant -15 deg roll bias on ep32, but it was **not** adopted, because official v24 to v26 favoured v12b. ep32 and ep33 are unchanged from v12b at this step.

### Step 10. Spoon mesh (G18)
Our spoon mesh had an inverted inner shell (about 26 to 33% of its area) and no balls. G18 removes the shell and adds 3 balls from our own ball model. Poses are unchanged. CD-O improved. A later bend test (bowl bend 0, 16 and 24 deg versus our 8.6) showed the current bend is the best of those on CD-O.

### Step 11. Frame-0 roll, chosen by leaderboard sweep
The scorer aligns the whole episode from the spoon's frame-0 rotation and the cup's frame-0 position, so frame-0 roll error moves everything by about 0.3 cm per degree.
- **ep33:** sweep of +9, -10, -13, -16, -18, -21, -25. Best is -16 tight (AUC 0.90015). Target `K33_H5r16`.
- **ep32:** -10 helped (AUC 0.90237). -13 and -16 runs lost the spoon, so a larger roll is untested in physics. Target `K32_R10`.

**Caution:** these values were selected using official scores, not image evidence.

## Part 2: Retargeting (object track to robot rollout)

Submitted rows are the object poses achieved in a physics rollout, never hand-edited. The robot is floating Sharpa hands. The evaluation backend recorded in the build is `frozen_mujoco_rbf` (MuJoCo, CPU), with evaluator hashes in every row.

### Step 1. Starting robot policy ("actor")
Each episode has an actor from earlier campaigns: a phase-indexed policy for both hands.
- **ep33:** `H13_ep33_b1` (the v29 actor).
- **ep32:** `H12_ep32_e1f0` (the v29 actor). It was tuned to the g12 target and scored spoon 0.93 cm / 19.8 deg against it. Its re-tune onto the retimed target lost the spoon, so v29 kept this actor.

### Step 2. Target and world
Target directory: `<object>_trajectory.npz` (T_world_obj per frame) and `<object>_mesh_metric.obj`. For ep33, `mkref.py` rewrites the object reference in the physics world to the new target. ep32 R10 reused the existing world.

### Step 3. Window-beam search (`wbeamH.py`, chained by `k33cwchain.sh` / `k32rchain.sh`)
Edits a residual table over time: wrist position and rotation, finger flex and thumb squeeze per hand. Every candidate is a full closed-loop rollout from saved physics states.
- 0.5 s windows, beam 4, 16 candidates per window, split into 2 sequential job groups.
- Scored by the same alignment and error as the kit, against the target.

Recipes actually used:

| | ep33 `H5r16_c3` | ep32 `R10_c1` |
|---|---|---|
| lineage | H13_ep33_b1 actor, H4settle_b3 table, `H5r16_c2`, then `c3` | H12 e1f0 actor, no prior table |
| hands edited | right only | right only |
| tolerance (cm, deg) | 0.5, 6 | 0.6, 8 |
| hinge | 8 @ 1.0 | 6 @ 1.2 |
| windows start | 2.25 s | 1.00 s |
| CPUs | 2 x 19 | 2 x 20 |

### Step 4. Save, stage, build (`svst.sbatch`)
Bake the table into the actor, run the unchanged stager gates, swap in the G18 mesh, and re-execute with build-v2. `exact_previous_physical_states_actions` must be true. Both runs here are exact.

### Step 5. Package (`k33splice.py` / `k32splice.py`, `v32build.py`)
Replace only that episode's pose and checkpoint columns in the base parquet (v30c1, coworker meshes aligned to our body frames), verify only those rows differ, and dry-run `v2d_submit.py`.

## Part 3: Results

Rollout versus its own target (cm / deg, kit-style, mean over time):

| | cup | spoon | MPPE | AUC |
|---|---|---|---|---|
| ep33 `H5r16_c3` vs `K33_H5r16` | 1.40 / 7.7 | 1.68 / 7.6 | 1.64 | n/a here |
| ep32 `R10_c1` vs `K32_R10` | 1.42 / 7.3 | **2.16 / 31.0** | 2.24 | 0.856 |

The MP-SR limit is 3 cm and 30 deg, so **ep32's rollout already misses against its own target.**

Official per-object AUC (mesh knockouts on 0fca055b):

| | cup | spoon |
|---|---|---|
| ep32 | 0.83 | 0.45 |
| ep33 | 0.85 | 0.59 |

## Part 4: What is wrong, as of today

### ep32
- **Robot:** spoon rotates up to 101.5 deg/frame (target peak 61.3), with fast spins at 3.6-3.8 s, 4.35-4.65 s and 6.75-6.9 s, and up to 180 deg off its target. In the 3 to 4 s and 6 to 8 s windows the bowl doesn't clearly enter the cup (checked by eye on sampled frames).
- **Robot, end of episode:** mug about 2.5 cm off its target, while the reconstructed mug lines up with the video.
- **Reconstruction:** not cleared. Coworker notes say our cup tilt is too small around f130-162. This has not been measured here.
- **Search:** this actor loses the spoon on most target changes: pitch/yaw 3 of 4, and all time-shift attempts.

### ep33
- **Reconstruction, f100-140:** user review of the video found the spoon wrong at 5-6 s and 6-8 s. Overlays confirm the bowl is too edge-on around f116-136.
- **Mirrored roll:** at f116-140 the mesh balls sit on the opposite side from the real balls. A full-circle ball and silhouette solve gives a roll change of +65 to +86 deg at f120-136, stable across settings, with control ep37 staying at mean 5 deg, max 14. A similar mismatch appears at f168-196.
- **Robot:** follows its target well in most of these windows, so the target is the main suspect there.

### Correction to earlier conclusions
Our earlier claim that the tracks match images and stereo to about 1 cm came from median-based checks that could not see these errors. The leaderboard sweeps also closed several ideas using rollouts built on these flawed tracks, so those rejections are less firm than they were reported.

| idea tested | official result | how firm |
|---|---|---|
| whole-track roll -16, ep33 | flat | tested a constant offset only, not per-window roll |
| frame-0 mid/short +-10, ep33 | all worse | moderately firm, rollouts fit looser |
| spoon bowl bend | current is best on CD-O | firm for mesh shape |
| time shift +1 / +2 frames, re-indexed rows | ep33 MPPE 1.529 / 1.533 (base 1.561); ep32 flat | diagnostic only, not yet confirmed in physics |
| Jisu earlier ep32 tracks | worse overall | confounded by loose robot fit |
| Hansol ep33 flip, f100-260 | worse | flip window was much wider than the one found now |

## Part 5: In progress

- **ep33 `K33_RR1`:** occlusion-aware silhouette roll refinement. Physics running.
- **ep33 `K33_RR2`:** RR1 plus ball-driven full-circle roll on the mismatch window. Physics running.
- **ep32 `K32_J10`:** coworker v10 reference on a different actor lineage. Result pending.
- **Portable kit for the coworker:** `ep32_retarget_kit_20261103.tar.gz`.
- **ep33 time-shift packages (ts1 845f1b9a, ts2 12d8e3ca):** built, not uploaded, on hold.

Videos and overlays: `challenge_submissions_SRL/videos_current_ep32_33_20261103/`.

## Part 6: Was DexAgent used, and which agents or models did the work

Checked 2026-11-03 from the code imports and the session logs.

### DexAgent: partly used, as a library and physics setup

DexAgent is at `third_party/DexAgent`, commit e29bdad. The final ep32 and ep33 runs were **not** produced by its native pipeline.

| part of DexAgent | used for the ep32/33 rollouts? | evidence |
|---|---|---|
| `tool_library` (hand rig, episode tools) | **Yes** | `mresH.py` imports `tool_library.common.hand`, and `evaluate_policy_component.py` imports `tool_library.common.episode` |
| Episode setup folders (`setup`, `stage2`) | **Layout yes**, DexAgent's role in generating them not traced | `dexagent_final_recon_20261017/episodes/episode_000032` |
| Floating Sharpa port of its runtime and MPC | Port exists, written by us | `code/floating_runtime.py`, `code/floating_tracking.py` |
| `sampling_mpc_refinement` (its MPC) | **Not in the current retarget chain.** Whether earlier actors passed through it was not traced | Only `floating_tracking.py` imports it. The README describes it as a pilot on episodes 1, 3, 5, 8 and 10 |
| Native pipeline (`run_dexagent.py`, stages 1 to 4, LLM-driven) | **No evidence it was used.** The workspace README says the work is not the complete native pipeline | README of `dexagent_final_recon_20261017` |

What actually produced the rows is our own code running on top of those parts: the window-beam search (`wbeamH.py`), the residual table (`mresH.py`), and the frozen MuJoCo evaluator.

Part 2 therefore describes custom retargeting that uses DexAgent components. It should not be described as "retargeted with DexAgent" without that qualification.

### Models recorded in the session logs

Model names are the ones written in the logs. "First seen" is the earliest logged tool call naming that item, so it shows who was working on it, not who wrote every line.

| stage | first seen (UTC) | model |
|---|---|---|
| Recovery campaign start | 2026-10-26 | Codex, per project notes (session logs show `gpt-6.1-sol`) |
| Fusion `fused_v7` | 10-27 15:22 | Claude Opus 5.5 |
| Spoon ball roll v8 | 10-28 00:16 | Claude Opus 5.5 |
| ep32 consensus roll (G12) | 10-29 09:39 | Claude Opus 5.5, subagent |
| ep32 actor `H12_ep32_e1f0` | 10-29 12:10 | Claude Opus 5.5, subagent |
| Clock fix (G15) | 10-29 14:09 | Claude Opus 5.5, subagent |
| ep33 actor `H13_ep33_b1` | 10-29 16:37 | Claude Opus 5.5, subagent |
| Roll audit (G16) | 10-30 01:09 | Claude Opus 5.5 |
| Spoon mesh (G18) | 10-30 03:03 | Claude Opus 5.5, subagent |
| ep33 target `K33_H5r16` | 11-02 06:12 | Claude Opus 5.5, subagent |
| ep33 run `H5r16_c3` | 11-02 19:25 | Claude Opus 5.5, subagent |
| ep32 target `K32_R10` | 11-02 20:53 | Claude Opus 5.5 |
| ep32 run `R10_c1` | 11-02 20:55 | Claude Opus 5.5, subagent |

So the rows in the best package for ep32 and ep33 were run under Claude Opus 5.5.

Session of 2026-11-02 to 11-03 (this analysis):
- **Claude Fable 5.1**, until the user switched models on 11-03: knockouts, spoon mesh study, whole-track roll, pitch/yaw sweeps, time-shift tests, portable kit.
- **Claude Opus 5.5**, after the switch: the ep33 roll refinement (`K33_RR1`, `K33_RR2`).

Other people:
- Jisu built the ep32 kinematic v10 reference and the MuJoCo proxy on another server. Her README names DexAgent grasp synthesis for the proxy. The tools and models she used otherwise are not in our logs.
- Hansol built the HGA and HGB reconstructions. Tools are not recorded here.

### Not traced
- Who first built the older actor lineages that `H12_ep32_e1f0` and `H13_ep33_b1` descend from. Project notes record both Codex and Claude working on earlier actors.
- Codex sessions were searched only for the reconstruction terms and the current run names. No Codex session mentions those run names.
