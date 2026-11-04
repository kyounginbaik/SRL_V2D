# Stream A (phase 7, v22): episodes 3 and 5 (white_pot + wooden_spoon, legacy floating-Sharpa MuJoCo, DexAgent port), 2026-10-28

Goal (../PHASE7.md): per episode a build-v2-exact component whose rollout is <= 2.4 cm / 24 deg per object vs `target_final_v5`.
Meshes: ours only (pot 8cbd35431829..., spoon c9ab88e33e5c..., same as v21/v21C). Nothing uploaded, nothing deleted, no commits.

## Results (score_any.sh on the build-v2 re-execution vs target_final_v5, cm / deg; scores in ../scores/cand/<tag>/score.json)
| ep | tag (campaign_v16_20261023/common_v2_candidates/) | component | pot | spoon | MP-SR | exact | before (best v21C cand.) |
|---|---|---|---|---|---|---|---|
| 3 | **A_ep03_roll21** (recommended) | streamA_ep35/ep3/staging_A1/components/episode_000003 | 1.24 / 9.6 | 2.18 / 22.0 | 1 | true | C_ep03_ach10ilc 1.07/7.0, 3.25/55.8 |
| 3 | A_ep03_roll21b (alternative) | streamA_ep35/ep3/staging_A2/components/episode_000003 | 1.15 / 9.0 | 2.30 / 19.6 | 1 | true | |
| 5 | **A_ep05_obj05** (recommended) | streamA_ep35/ep5/staging_A1/components/episode_000005 | 1.59 / 8.2 | 2.10 / 21.0 | 1 | true | C_ep05_grasp92 1.87/11.9, 3.06/25.8 |
Checkpoints: ep3 A1 338da1abab85, A2 a9232c657ed0; ep5 5bd9b7f00307. Portable archives: epE/staging_A1/portable_archives/episode_00000E_portable.tar.gz
(ep5 40cf77746654..., ep3 577e912bf34a...). Videos (node100, target kinematic | v21C | v22): videos/episode_00000E/episode_0E_v21C_vs_v22.mp4.

## World (code/build_world.py = streamT build_world with CMP = target_final_v5; record epE/world/compose_reference.json)
Target poses converted into our body frames with the scorer registration (code/tlib.py), G = Gt @ G0 (G0 = target pot frame-0 -> our pot
frame-0; Gt = table-plane fit of the resting contacts). ep3: Gt 1.80 deg / -0.55 cm; ep5: 0.60 deg / -0.80 cm. Base episode dirs = the v21
ones (ep3 streamB_tilt38 calibrated_inputs, ep5 refinement_20261018 calibrated_scene_candidates). Corrected object_poses sha256: ep3
ba69790c1d47..., ep5 0fff4c4fad5b.... Shift vs v21 reference: ep3 pot 1.05 cm / 5.0 deg, spoon 1.95 cm / 48.9 deg; ep5 pot 1.87 / 7.6,
spoon 2.92 / 22.4.
Main change: in ep3 the target spoon rolls a full turn (360 deg) about its long axis between f102 and f196 (v21 / v21C hold within +-60 deg);
in ep5 the spoon roll differs by ~25 deg incl. the frame-0 rest.

## Re-plan
- Teachers ported: phase-6 C teachers (ep3 streamT ep3/ilc/A_s10_b1.0_gp1.0/iter0, ep5 streamT ep5/ret/g2_Aa_s6_a0.3_s92) with the
  phase-6 compose world as the "old" world. Plain / `ref` / `ach` ports (~60 variants) all drop the spoon at the pickup: `ach` blends
  D-space transforms, which rotates the hand by up to 80 deg while the fingers close (the target spoon orientation during lift-off differs).
- code/port.py additions: (1) rotvec unwrap in smooth_D (the phase-6 smoothing averaged across the pi wrap; needed for the full turn);
  (2) new mode `obj`: interpolate in OBJECT pose space - before first contact the hand approaches the settled new rest pose exactly as in
  the teacher; after contact + --obj-start s the target object pose blends (smoothstep over --blend s, scale --alpha) from the old achieved
  motion rigidly shifted onto the rest pose to the smoothed new reference (sign-continuous quaternion smoothing, --sigma frames), rotation
  along the unwrapped relative rotvec; D = T_target @ inv(T_old_achieved) carries the wrist rigidly with the object (so a full-turn roll
  is followed by the hand); frozen after last contact. --obj-restrot (fraction of the rest-rotation applied; 0 = translate only) and
  --obj-dz (approach height offset).
- ep5: obj, sigma 4, start 1.0 s, blend 1 s, alpha right 0.5 (port/O_s4_st1.0_a0.5: pot 1.57/8.0, spoon 2.21/19.9) -> actor grid
  (distill.sbatch, node01): 10/10 actors pass 3/30; best c2048_p0.01 (2.10/21.0) staged. Retention CEM teachers (ret/O_a0.5_r2.5_s11
  1.54/8.1, 1.93/17.9; r1.0_s12) distil less reliably (r11 0/10 pass; r12 best 2.16/19.7) - not used.
- ep3: pickup needs translate-only rest (restrot 0) + 6 mm higher approach (port/Or0_z0.006_s4_st1.0); the 2-3-contact pinch then
  slips during the roll (spoon lost ~8 s). Right-hand retention CEM (refine_versioned_retention_full.py, 96x12, start 2.5 s, stop 13 s,
  seed 21) -> ret/z6s4_r2.5_s21 teacher pot 1.25/11.3, spoon 2.14/21.1 -> actor grid seed 44: c2048_p0.001 (A1) and c1024_p0.03 (A2)
  pass (2/10; seed 45: 1/10). Other retention runs: s25 (start 2.0) 1.08/9.9, 2.15/21.9 (not distilled); s22/s24/s26/s31-s34 2.6-3.2 cm.
- ILC and margin search were not needed.

## Code (code/; copies of streamT_C356/code with paths -> this stream; phase-6 files untouched)
build_world.py, tlib.py (CMP=target_final_v5), port.py (+unwrap, +obj mode), port.sbatch (16 variants in parallel), ret.sbatch / ret48.sbatch,
ret_finish.py (finish a time-limited retention run from progress.json; not needed in the end), distill.sbatch, stage.sbatch,
stage_compose_component.py (ST -> this stream, v22 rationale), persec.py / rel.py / roll_diag.py / summ.py (diagnostics), videos/render.sbatch.

## Caveats
- Margins: ep3 spoon 2.18 cm / 22.0 deg, ep5 spoon 2.10 / 21.0 (0.2-0.3 cm under 2.4). The holds are chaotic (actors of the same teacher
  range 2.1-8 cm), selection is among deterministic build-v2 replays.
- ep3 follows the target's full-turn spoon roll by rolling the floating wrist ~360 deg about the handle (kinematically legal: wrist
  rot range +-4 pi), not a human-like in-hand roll.
- ep3 pre-grasp approach is 6 mm higher than the teacher's and keeps the teacher's orientation (rest rotation not applied).
- Proxy scores vs our own reconstruction, not hidden ground truth.
