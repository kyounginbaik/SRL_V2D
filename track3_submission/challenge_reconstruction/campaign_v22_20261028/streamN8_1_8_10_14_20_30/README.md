# Stream N8 (phase 8, v23): episodes 1, 8, 10, 14, 20, 30 vs target_final_v5

Goal (PHASE8.md): each object <= 1.2 cm / 12 deg vs `campaign_v22_20261028/target_final_v5` where possible and strictly better than the
v22b component; every finished tag build-v2 exact (`common_v2_candidates/run_one.sbatch`) and scored with `code/score_any.sh`.
Meshes: the v22b shipped meshes only (checked by the stage script). Nothing uploaded, deleted or committed.

## Results (score_any on the build-v2 re-execution, cm / deg)
All tags in `/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/<tag>`; every completed.json has exact_previous_physical_states_actions = true; scores in
`campaign_v22_20261028/scores/cand/<tag>/score.json`. Recommended tag per episode in **bold**.

| ep | tag | component | object 1 | object 2 | MP-SR | AUC | v22b component (obj1, obj2) |
|---|---|---|---|---|---|---|---|
| 1 | **N8_ep01_k2c2048** | ep1/staging_N2/components/episode_000001 | pot 0.76 / 4.9 | lid 0.84 / 9.4 | 1 | 0.949 | D_ep01_achs15c1024 1.18/6.4, 1.28/12.5 |
| 1 | N8_ep01_k2c512 | ep1/staging_N3 | pot 0.89 / 5.5 | lid 0.74 / 9.0 | 1 | 0.947 | |
| 1 | N8_ep01_kr2048 | ep1/staging_N1 | pot 1.16 / 5.5 | lid 0.90 / 10.1 | 1 | 0.925 | |
| 8 | **N8_ep08_klr512** | ep8/staging_N3/components/episode_000008 | beige_cup 0.71 / 2.9 | white_pot 0.83 / 7.1 | 1 | 0.961 | B_ep8_level 2.09/10.7, 1.75/7.7 |
| 8 | N8_ep08_sr15c1024 | ep8/staging_N1 | 0.82 / 2.4 | 1.03 / 7.6 | 1 | 0.955 | |
| 8 | N8_ep08_sr1c512 | ep8/staging_N2 | 0.92 / 2.3 | 1.00 / 8.0 | 1 | 0.946 | |
| 10 | **N8_ep10_up3c512** | ep10/staging_N1b/components/episode_000010 | blue_cup 0.51 / 1.3 | beige_cup 1.48 / 2.7 | 1 | 0.942 | G_ep10_klr6c512 1.32/7.1, 1.75/8.6 |
| 10 | N8_ep10_uk2c1024 | ep10/staging_N2b | 0.35 / 1.6 | 1.55 / 2.5 | 1 | 0.940 | |
| 10 | N8_ep10_G2 | /home/exaFLOPs/v22_overflow_20261028/streamG_ep10/staging_G2 (stream G, verified here) | 1.30 / 6.5 | 1.58 / 8.6 | 1 | 0.897 | |
| 14 | **N8_ep14_kl512** | ep14/staging_N2/components/episode_000014 | blue_cup 1.11 / 4.1 | rack 0.67 / 2.8 | 1 | 0.966 | v21C (C_ep14_r24_c1024p01) 1.56/10.0, 1.21/2.6 |
| 14 | N8_ep14_sl45c1024 | ep14/staging_N1 | 1.30 / 4.5 | 0.62 / 3.4 | 1 | 0.947 | |
| 20 | **N8_ep20_k3c1024** | ep20/staging_N1/components/episode_000020 | blue_cup 0.61 / 11.8 | pitcher 0.57 / 3.4 | 1 | 0.960 | D_ep20_tach2c1024 0.70/12.8, 0.69/3.7 |
| 20 | N8_ep20_k3c512s45 | ep20/staging_N2 | 0.60 / 11.9 | 0.59 / 3.0 | 1 | 0.960 | |
| 30 | **N8_ep30_k2c2048** | ep30/staging_N1/components/episode_000030 | sweeper 0.86 / 6.2 | rack 0.57 / 3.8 | 1 | 0.966 | E_ep30_stdB45 0.97/6.4, 0.97/5.6 |

Only regression vs v22b: ep14 rack orientation 2.8 vs 2.6 deg (see ep14). ep10 beige stays above 1.3 cm (physical floor, see ep10).

## Method
1. World (PHASE7 recipe step 1), `code/build_world_n8.py` (copy of stream D build_world_d.py, eps 8/14 added): target_final_v5 poses ->
   our shipped-mesh body frames by the scorer registration (`code/nlib.py`, meshes from the v22b package parquet), G = Gt @ G0 with
   G0 = the non-symmetric anchor (ep8 white_pot, ep14 plastic_dish_rack) frame 0 onto the physics initial pose of the v22b teacher,
   Gt = table-plane LS fit of the resting contacts (ep8 0.53 deg / -0.41 cm, ep14 0.49 deg / +0.31 cm). Axisymmetric cups keep the
   reference spin (minimal swing). Records: `ep8/world/target_reference.json`, `ep14/world/target_reference.json`.
   Frame-0 layout change: ep8 beige cup 1.37 cm / 10.2 deg vs the old physics start, ep14 blue cup 0.59 cm / 4.0 deg.
   ep1 / ep20 use the stream D worlds unchanged (`streamD_ep1_20/ep*/world`).
2. Port (`code/port_n8.py` = stream D port_d.py + `--sym`): wrist_new = D @ wrist_old with D from the teacher's ACHIEVED object track.
   For axisymmetric cups D must be swing-only about the scorer anchor point (port_g.py mode): with the full D the cup spin, which is
   arbitrary, drags the wrist ~11 cm around the cup and the cup is dropped (ep8 r15: 7.8 cm). Swing-only ports:
   ep8 (teacher streamB_tilt38 ilc cup_right_2.0_10.8_s63_gr0.5_gp0.8/iter1, right hand = cup, sigma 1): cup 0.83 / 3.5, pot 1.03 / 7.4;
   ep14 (teacher streamM_ep14 port_C/r24, left hand = cup, sigma 3): cup 1.34 / 3.9, rack 0.61 / 3.1.
3. Knot CEM (`code/kcem_n8.py` = stream G kcem.py with per-object --P cm / --D deg scales, `--Dobj`, and `--scene-home` for ep20):
   full-physics CEM over time-varying wrist offsets + finger squeeze, cost max/sum of pos/P and deg/D per object.
4. Distill (`code/distill.sbatch`, streamC actor_grid_c.py, node01), stage (`code/stage_n8.py`, stream D stage_d.py + eps 8/14 meshes,
   previous = v22b component), build-v2 verify (`code/verify.sh` = completed.json exactness + score_any.sh).

## Per episode
- ep8: re-planned to target_final_v5 (world ep8/world). Swing-only port sr1 (cup 0.83/3.5, pot 1.03/7.4), then knot CEM on both wrists 2-10.5 s
  (ep8/kcem/k_lr_20_105: 0.70/2.8, 0.80/9.6), plain distill c512_p0.001.
- ep14: re-planned to target_final_v5 (world ep14/world; the rack offset of v22b came from the cup-rack frame-0 layout of the coworker
  target). Swing-only port sl3/sl45, knot CEM left wrist 1-6.5 s (k_l_10_65), distill. The rack rests 2.45 deg / 0.34 cm off its
  target frame-0 pose (its collision hulls settle it in the first 5 frames, independent of a 2-3.5 mm lift: worlds world_dz0.002 /
  world_dz0.0035, ports identical tilt). A world with the rack home at its settled pose (world_settled) puts the target cup 7.8 mm into
  the table, so it was not used; rack orientation stays 2.7-2.9 deg in every actor (CEM with a 2.3 deg rack scale k3_l_rackdeg: 2.7).
- ep1 (stream D world): knot CEM right wrist 4.5-16.7 s (k_r_45_167: lid 0.86/10.6), second pass both wrists (k2_lr_r45: 0.87/5.1,
  0.74/9.4), distill.
- ep10: the v22b/G2 start error (both cups ~1 cm, 6-8 deg in f0-30) is the cups rocking upright: target_final_v5 has both cups 6-8 deg off
  the physics table normal at rest. `code/build_world_g10up.py` (stream G build_world_g.py + up-correction) rotates the target by 7.6 deg
  so the resting cup axes are vertical (height fit only after it) -> ep10/world_up. Swing-only port of the G1/G2 teachers, knot CEM both
  wrists 1-7.8 s (u_p3_LR 0.44/1.4, 1.49/2.6), distill, staged with the unchanged stream G wrapper (code/stage10.sbatch; the actor dir
  is copied to ep10/prepared/ with the teacher recorded, as stream G did). Floor: in the nested phase f100-160 the physical beige-blue
  anchor separation along the blue axis is 6.55 cm vs 2.9-3.9 cm in the target (target interpenetrates the blue rim), i.e. a 3.3 cm axial
  error on 61 of 161 frames = 1.25 cm of the beige mean before any other frame; 1.3 cm is not reachable with our meshes. A third CEM
  pass (u2_p3_LR) and early-window CEMs in the old world (k3_*, k4_*) did not move.
- ep20 (stream D world): the scorer treats blue_cup as axisymmetric (public_kit_score SYMMETRY order 360; spin ignored); the remaining
  ~12 deg is axis TILT of the cup held in the left hand. Knot CEM with `--scene-home` (the port moved the left-wrist home with the held
  cup) on the left wrist, then both wrists twice (k3_lr_init: 0.62/11.9, 0.59/2.9), distill.
- ep30 (stream E world_T1): knot CEM both wrists 1.5-13 s, two passes (k2_lr_15_130 0.84/6.2, 0.60/4.2), distill, staged with stream E's
  stage path (code/stage30.sbatch = streamE code30/stage.sbatch, base ep30/staging_N1 with actor_report.json + world input_manifest).

## Caveats
- Proxy scores are against our own reconstruction, not hidden GT.
- Holds are chaotic: the same teacher reference re-executed on another node can differ (ep1 port ach_s1.5 lid 1.26/12.1 in stream D vs
  7.2/19.3 here on node100; ep20 test_ach2 1.30/16.9 on node01); selection is always among deterministic build-v2 replays.
