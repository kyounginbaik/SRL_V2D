# Stream D (phase 7, v22): episodes 1 (white_pot + white_pot_lid) and 20 (blue_cup + water_pitcher), 2026-10-28

Goal (PHASE7.md): every object <= 2.4 cm / 24 deg vs `campaign_v22_20261028/target_final_v5`, MP-SR 1, exact build-v2 replay.

## Results (score_any.sh on the build-v2 re-execution, cm / deg)

| ep | tag (campaign_v16_20261023/common_v2_candidates/) | component | objects vs target_final_v5 | before (v21) | exact |
|---|---|---|---|---|---|
| 1 | **D_ep01_achs15c1024** | streamD_ep1_20/ep1/staging_D1/components/episode_000001 | pot 1.18 / 6.4, lid 1.28 / 12.5, MP-SR 1 | pot 1.01 / 6.0, lid 1.66 / 30.2 (fail) | true |
| 20 | **D_ep20_tach2c1024** | streamD_ep1_20/ep20/staging_D1/components/episode_000020 | cup 0.70 / 12.8, pitcher 0.69 / 3.7, MP-SR 1 | cup 2.74 / 6.1, pitcher 4.09 / 6.8 | true |

Portable archives: `ep{1,20}/staging_D1/portable_archives/episode_0000XX_portable.tar.gz`. Checkpoints: ep1 fc10f75c42d8, ep20 5512a95efe2f. Scores: scores/cand/D_ep01_achs15c1024, scores/cand/D_ep20_tach2c1024. No videos rendered.

## Diagnosis (code/timeline_v5.py, code/axis_decomp.py, code/diag_ref.py, code/lowz.py)
- ep1: the lid error is a CONSTANT spin of ~30 deg about the lid's symmetry (knob) axis (tilt 1-4 deg, position 1.1-1.5 cm), present from
  frame 0 (lid on pot) through the on-table phase. Pot fine. The target lid sits in the same places, just rotated about its axis.
  Note: the physics scene's frame-0 pot is 13.4 deg off the Stream H corrected tracking reference (fixedseat scene); the anchor G0 must be
  taken from the physics initial state (teacher execution initial_qpos), not the tracking reference (otherwise the target lid floats 4 cm).
- ep20: the scorer takes the alignment ROTATION from the pitcher (only non-symmetric object) but the TRANSLATION from slot 0 = blue_cup.
  The target's cup-pitcher relative placement differs from ours by ~2-4 cm, so the whole pitcher track carried a ~3.5 cm constant offset;
  plus the target pour path is 2-3 cm higher. Not a table-height problem (Gt 1.3 deg / -0.8 cm).

## World (code/build_world_d.py -> epE/world/{source_target,episode_target}, target_reference.json)
Same method as streamT build_world.py: target poses -> our body frames by the scorer registration (code/dlib.py), G = Gt @ G0,
G0 = anchor frame-0 onto the physics initial pose (ep1 white_pot, ep20 water_pitcher), Gt = table-plane LS fit so resting contacts
match the teacher's physical rest heights (ep1 0.90 deg / -0.27 cm, ep20 1.28 deg / -0.83 cm).
Axisymmetric blue_cup (scorer order 360): target spin is unobservable and differs from ours by ~150 deg; we keep the v21 reference spin
(minimal swing old axis -> target axis) and place the body so the scorer's anchor point is exactly the target's (recorded in
target_reference.json `axisymmetric_spin_kept_from_v21_reference`). The lid (order 2) keeps the full target rotation (the spin counts).

## Re-plan (code/port_d.py; generalised streamT port.py)
wrist_new = D @ wrist_old with D = T_new_obj @ inv(T_teacher_ACHIEVED_obj) (left<-pot/cup, right<-lid/pitcher), Gaussian sigma frames;
`--home-robot left` moves the left wrist home keyframe rigidly with the held cup (ep20 starts with the cup in hand).
Wrist chart check (stream C finding): robot_*_base_roll body_quat = identity in both scenes, so the XYZ-euler wrist composition is valid.
- ep20 teacher = v21 lineage final_pitcher_release_75_neutral/trial1. Port sigma 1.5 (ep20/port/test_ach2): cup 1.25 / 17.0,
  pitcher 0.91 / 5.9 (cup tips after release at the very end); sigma 4: 0.89 / 13.9, 1.17 / 5.2; left-D freeze after set-down 7.2-9 s:
  ~0.95 / 15, 1.41 / 9.4. Distill of test_ach2 (actor_grid_c.py, node01): 6/8 pass; c1024_p0.001 0.70 / 12.8, 0.69 / 3.7 (staged);
  c1024_p0.003 0.79 / 13.6, 0.61 / 3.6. (ep20/distilled/ach_s4 grid also launched.)
- ep1 teacher = refinement_20261021_ep1/lid/shift_0.02_0_0_12.8 (teacher of the shipped v21 actor). Port sigma 1.5: pot 1.13 / 5.7,
  lid 1.26 / 12.1; sigma 0 / 1 / 2 / 2.5 / 4 drop the lid (6-10 cm; the lid hold is chaotic). Distill: 5/8 pass;
  c1024_p0.01 pot 1.18 / 6.4, lid 1.28 / 12.5 (staged); c2048_p0.001 1.14 / 5.8, 1.37 / 16.5; c1024_p0.001 1.15 / 7.4, 1.52 / 15.0.
  Remaining lid error: on-table spin 10-24 deg (f100-240) and 2-5 cm after the lid goes back on the pot (f260-334).

## Code / tools
code/: dlib.py, build_world_d.py, port_d.py, stage_d.py (from streamA stage_compose_component.py), timeline_v5.py (per-frame aligned
error), axis_decomp.py, diag_ref.py, lowz.py, contacts_d.py, score_run.py/score_run.sh (score a raw run dir vs target_final_v5 with the
v21 package meshes), score_grid.sh, distill.sbatch, stage.sbatch, one.sbatch, env.sh. Raw scores: streamD_ep1_20/scores/<label>.

## Caveats
- Holds are chaotic (ep1 lid ports with sigma != 1.5 drop the lid; 3 of 8 actors drop it); selection is among deterministic build-v2 replays.
- ep20 cup spin follows our old reconstruction, not the target (irrelevant to the scorer: order-360 symmetry).
- Against the v21 reference these components are worse by construction (scored against target_final_v5).
- Nothing uploaded, deleted or committed; jobs on debug, distill/stage/eval on node01.
