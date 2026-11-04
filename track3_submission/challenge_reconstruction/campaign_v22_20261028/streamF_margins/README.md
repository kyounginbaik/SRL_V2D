# Stream F (v22, PHASE7.md): margin improvement for eps 16, 19, 6, 2026-10-28

Goal: every object <= 2.4 cm / 24 deg vs `campaign_v22_20261028/target_final_v5`, MP-SR 1, exact build-v2 replay; report a new
component only if exact AND better on the worst-object ratio max(pos/2.4, deg/24). Meshes: ours only (unchanged). Nothing uploaded,
deleted or committed. All scores: `code/score_any.sh EP <common_v2_candidates/TAG/episode_X> TAG` -> `../scores/cand/TAG/score.json`.

## Results
| ep | tag (campaign_v16_20261023/common_v2_candidates/) | staging (component = staging/components/episode_X) | obj 1 cm/deg | obj 2 cm/deg | worst ratio | exact | before (best exact) |
|---|---|---|---|---|---|---|---|
| 16 | **F_ep16_tgtret203** | ep16/staging_F1 | rack 0.37 / 2.4 | spoon 1.89 / 17.8 | 0.79 | true | N6_ep16_newtrack_z5rack 0.43/2.5, 2.79/17.2 (1.16) |
| 19 | **F_ep19_twin3** | ep19/staging_F3 | sweeper 1.95 / 14.8 | dust pan 2.38 / 9.9 | 0.99 | true | C_ep19_ilc 5.77/50.2, 40.1/177.2 (MP-SR 0 exact; v21's 2.62/14.6 needs the package relabel) |
| 19 | F_ep19_twin2 (backup) | ep19/staging_F2 | 2.02 / 15.3 | 2.51 / 11.4 | 1.05 | true | |
| 6 | **F_ep06_obj05a** | ep6/staging_F1 | pot 1.17 / 5.0 | spoon 1.55 / 16.4 | 0.68 | true | C_ep06_ronly_lilc 1.77/13.2, 2.47/20.0 (1.03) |
| 6 | F_ep06_obj05b (alt) | ep6/staging_F2 | 1.15 / 4.7 | 1.68 / 15.8 | 0.70 | true | |

Portable archives (staging/portable_archives/episode_X_portable.tar.gz) sha256: ep16 bd755ba2126779b1..., ep19 F3 895c24a006df4578...,
ep6 F1 bea5426ecb696240... (F2 e6b54c9b60e102a0...). Checkpoints: ep16 3984143ed74c0289..., ep19 5dcbb487a12c908e..., ep6 9d2d0d585aa21d3a...

## ep6 (ep6/code = copy of streamA_ep35/code, paths -> this stream)
- World: `build_world.py 6 dexagent_final_recon_20261017/episodes/episode_000006` (G = Gt @ G0, pot frame-0 anchor, Gt 2.61 deg / -0.62 cm;
  record ep6/world/compose_reference.json). Shift vs v21 reference: pot 2.89 cm / 6.6 deg, spoon 2.12 / 16.0.
- Teacher ported: streamT_C356/ep6/ilc/Ronly_L_gp0.3/iter1 (the C_ep06_ronly_lilc teacher), old world = streamT ep6 compose world.
  plain / ref-mode ports all drop the spoon (10-18 cm; new spoon rest pose differs); stream A's `obj` mode works (port_v2.txt):
  best O_s4_st0.5_a0.5 (pot 1.19/7.4, spoon 2.01/15.8) and Or05_s4_st1.0_a0.5 (1.20/5.7, 1.68/16.6).
- Distilled (node01, plain actor grid seed 44): O_s4_st0.5_a0.5 6/10 actors at 1.55-1.68 cm / 15-18 deg -> c512_p0.001 (F1), c512_p0.003 (F2).
- Retention CEM on Or05 (ret/Or05_r2.0_s61) reached 1.42 / 14.5 internally; not needed, not distilled.

## ep16 (ep16/code; streamN_ep16 code copied, unchanged logic)
- code/tlib16.py = scorer-identical errors (rack slot 0 is the anchor), matches score_any to ~0.02 cm on existing outputs.
- world_T (build_world_f16.py from world_6prong_newtrack): rack + spoon references = G0 @ T_target @ X (X scorer registration, G0 target
  rack f0 -> our physics rack f0); frame 0 keeps our physics start (search tool requires scene start == reference f0). Record
  world_T/source/episode_000016/target_retarget.json. Ported teachers reproduce score_any (z5rack 2.79/17.4 vs 2.79/17.2).
- 6 rack-weighted retention CEMs (streamN retention_rack.py) -> 1.89-1.99 cm / 17.8-18.6; z5p2_k203 (1.6-8.6 s, s0.5) distilled:
  8/10 actors 1.89/17.8 -> actor c1024_p0.003 staged. Pass-2 CEMs reached 1.74-1.76 internally (not distilled).

## ep19 (ep19/code)
- Diagnosis (ep19/verify): target_final_v5 shows the half-turn twin face of our physical sweeper (twin about its vertical long axis);
  the offline relabel of exact C_ep19_ilc gives 2.62/14.6 — a pose edit, not used.
- The exact twin of our sweeper home topples in physics (code/settle.py); twin_rest.py found a stable standing twin 11.7 deg from the
  exact twin (9.1 deg from target f0). world_st (build_world19.py from streamV build_corrected): sweeper starts at the settled twin,
  target anchored on it (scorer frame-0 alignment; internal cost == score_any).
- Teacher: C_ep19_ilc teacher (streamC_rigid_ilc/ep19/ilc/gr0.8_gp0.5/iter0) ported with port19.py (streamV port_relative + --twin:
  sweeper grasp relabelled through the half turn, dust-pan pre-blend 1.1 s, blend back 9.3 s, left squeeze +0.1 rad); ~200 variants;
  teacher g3_b_s6_g0.8_p_9.3_0.5_0.0_sq sweeper 1.99/15.2, dust pan 2.35/11.1. 13 actor grids (seeds 44-52); best
  st_g3best_s47/actor_c2048_p0.03 staged (F3). run_one.sbatch submitted with --time=00:50:00 (default 4 h stayed pending).

## Caveats
- ep19 dust-pan margin is thin (2.38 cm): with our stable twin 9 deg off the target f0, the aligned target dust pan rests ~2 cm below our
  table (~65 % of frames) -> ~2.2 cm vertical floor; shifted/yawed start worlds (world_shift*, world_yw, world_tw) broke grasps.
  The sweeper start is a settled twin pose ~12 deg from the exact twin (initial scene choice, no reported pose edited).
  Do NOT apply the package ep19 sweeper relabel (postprocess_v16) to F_ep19_twin3 — its face is already physical.
- ep16: hold-until-end (right hand stays on the placed spoon, as in N6); target spoon f0 is 2.2 cm / 22 deg from our start relative to
  the rack (pre-grasp error floor). world_T manifest new_input_provenance.hawor_source points at the world_T source copy (hand report
  byte-identical).
- ep6: approach uses half the rest-rotation change (restrot 0.5); hold is chaotic (4/10 actors of the teacher grid fail), the selected
  one is a deterministic build-v2 replay.
- Proxy scores vs our own reconstruction (target_final_v5), not hidden ground truth.
