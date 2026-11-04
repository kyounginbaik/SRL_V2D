# Stream B (v22, phase 7): episodes 32 and 33 (blue_cup + wooden_spoon, legacy floating-Sharpa MuJoCo) vs target_final_v5

Goal (PHASE7.md): every object <= 2.4 cm / 24 deg vs `target_final_v5`, MP-SR 1, exact build-v2 replay.
Starting point (score_any.sh): ep32 R_ep32_regrasp_power06 cup 1.64/12.2, spoon 1.80/39.7 (MP-SR 0); ep33 P_ep33_regrasp_cem3 cup 1.52/15.3,
spoon 4.52/63.4 (MP-SR 0).

**Outcome: neither episode clears the 2.4 cm / 24 deg margin.** Both now pass MP-SR under 3 cm / 30 deg against target_final_v5, with exact build-v2
replay. Before this stream, both episodes failed MP-SR against target_final_v5.

| ep | tag (recommended first) | component | cup cm/deg | spoon cm/deg | build-v2 exact | checkpoint / archive sha256 |
|---|---|---|---|---|---|---|
| 32 | **B_ep32_m4444** | staging_ep32_m4444/components/episode_000032 | 1.47 / 10.2 | 2.45 / 22.4 | yes | b3c1e8e31da66f89.. / 703d446d0926b2e9.. |
| 32 | B_ep32_fb_m512 (superseded) | staging_ep32_fb_m512/components/episode_000032 | 2.96 / 5.6 | 2.77 / 29.3 | yes | c5cd0a6b1f706c74.. / b9f310ea36363998.. |
| 33 | **B_ep33_c51** | staging_ep33_c51/components/episode_000033 | 2.06 / 13.1 | 2.43 / 20.2 | yes (output in overflow, see below) | 1107e1fe1033f2c9.. / 2bd6363bd8f712d0.. |
| 33 | B_ep33_b47c1536 | staging_ep33_b47c1536/components/episode_000033 | 1.83 / 5.9 | 2.65 / 19.1 | yes | d1e8de18832fb99c.. / c2fd32a3a7b388b1.. |
| 33 | B_ep33_i2c1024 (superseded) | staging_ep33_i2c1024/components/episode_000033 | 2.76 / 14.4 | 2.76 / 20.5 | yes | ed03e7e10a53a097.. / 25d9cbbc091f9fce.. |

Scores are from score_any.sh (or, for B_ep33_c51, an identical copy that writes to /home), using the verified build-v2 outputs. Archives are under
staging_*/portable_archives/episode_0000XX_portable.tar.gz. Verification outputs are in common_v2_candidates/<TAG>, except B_ep33_c51:
/data1 was full (0 bytes free, so even symlinks failed), and its run_one output is in /home/exaFLOPs/v22_overflow_20261028/common_v2_candidates/B_ep33_c51.
It was produced by an identical run_one copy (/home/exaFLOPs/v22_overflow_20261028/streamB_ep3233/code/run_one_overflow.sbatch; only B, the log path and
TMPDIR differ). It still needs a symlink into campaign_v16_20261023/common_v2_candidates/ once space allows. Work after about 05:40 is under
/home/exaFLOPs/v22_overflow_20261028/streamB_ep3233 (distill i2f/f1s3f/rel_*, margins, ilcrel).
Meshes: ep32 cup 3c3c5328 (the shortened recon cup v21 ships) and spoon c9ab88e3; ep33 cup 57bb269a and spoon c9ab88e3. No videos were rendered.

## 1. Reference and world (code/blib.py, code/build_world_b.py, code/check_ref.py)
- Scorer facts (MP-SR.py `_align_registered_scene`): the frame-0 alignment ROTATION comes from the first object without a symmetry axis
  (= the spoon, slot 1), the translation from slot 0 (the cup) frame-0 position. So the relative cup-spoon frame-0 layout matters.
- `blib.py` reproduces `public_kit_score.aligned_errors` per frame for arbitrary pose arrays (MP-SR registration of the SHIPPED meshes
  onto the target meshes, budget 4096, same seed). `diag.py` / `score_traj.py` reproduce score_any.sh exactly (R_ep32_regrasp_power06
  1.64/12.2, 1.80/39.7).
- `build_world_b.py EP OUT BASE_SCENE --base-epd --meshes [--short]` = streamR compose_retarget32_v3.py generalized: target_final_v5 ->
  our shipped body frames (scorer registration) -> G = Gt @ G0 (G0 frame-0 Kabsch of both objects onto the previous physics layout,
  Gt minimal tilt about the frame-0 cup base + lift putting the shipped cup bottom and the lowest spoon point on the table; cup yaw fixed
  to ours, scoring-invariant). Corrected source copy + `target_retarget.json` (sha256s, G, X, checks, per-frame shift vs the v21 reference),
  corrected episode dir, scene = base scene with only the two object initial poses/keyframe changed.
  - ep32 (`ep32/W1`, base scene = streamR regrasp/pick_full/power0.6, shipped cup 3c3c = 57bb shortened 1.78 cm): tilt 0.57 deg, lift 0.14 cm;
    shift vs the v21 reference cup 2.03 cm / spoon 1.24 cm, 47.6 deg mean (spoon roll differs by up to 180 deg 