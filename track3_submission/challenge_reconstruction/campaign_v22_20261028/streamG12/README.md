# Stream G12 (phase 9): ep32 truth from v24, v25 and v26 together; per-episode official calibration; GT-free mesh-scale audit

Rules followed: our meshes only, and no mesh was changed. Public scans and GT were used only inside scorers. The `scorer_only_scan_target_v9` folder
is v9 trajectories with G9's scorer-only scan meshes and is for validation only. Compose tracks were used only as hypotheses (ep32 spoon twist prior,
calibration labels). No submitted pose was edited, nothing was overwritten, nothing was uploaded and nothing was committed. Slurm debug on node01; every job took under 50 min.

## Bottom line
1. **ep32 new target: `target_final_v10_g12`** (bundle `reconstruction_recovery_20261026/selected20_final_v10_g12`). It is v8_g10b with only the ep32 spoon changed
   (`G12_CHANGES.json`; the G11 ep20 pitcher shift was **not** kept, see below). The spoon is final_v6 rolled about its long axis by a **consensus roll track**
   (`out/a32_consensus.npz`), which is the per-frame circular mean of the official-consistent, ball-near-optimal family. In that track the spoon is held flipped
   (about 105-165 deg) through f45-178, with a partial unroll around f90-108, and unrolls over f178-195. v8_g10 returned to ours at f152.
   - The official scorer path (`code/score_any.sh`) reproduces all three official ep32 outcomes:
     - v24 (B9_ep32_s6f0): spoon 67.1 deg, SP 0, MP 0.
     - v25 (H9_ep32_h19f0): 38.9 deg, SP 1, MP 0.
     - v26 (H10_ep32_H2f0): 36.7 deg, SP 1, MP 0.
   - Hidden-noise likelihood P(all three official outcomes) is **0.97 at 5 deg / 0.5 cm and 0.73 at 8 deg / 0.8 cm**. This is the best of every track tried.
     v8_g10 (G10) is 0 because it predicts that v26 passes.
   - 2D balls: median px on informative frames 7.05, against 6.49 for the unconstrained ball Viterbi and 9.9 for v6. The official constraints cost almost no ball evidence,
     because the extra flip lies on ball-poor frames (f151-178 has 5 informative frames).
2. **No existing ep32 component passes MP against it.** The best is 36.7 deg, so ep32 must be **re-retargeted** onto target_final_v10_g12 (only ep32).
3. **B (all 8 scored packages):** P(fail MP in v26):
   - ep32: 1.00.
   - **ep20: 0.76-0.89.**
   - **ep33: 0.48-0.87** (fragile, see below).
   - ep3: 0.31-0.60 counts-only; about 0.25 once G11's MPPE-delta evidence is added.
   - ep1: 0.30-0.43.
   - ep19, ep35, ep36: 0.11-0.32.
   - ep8: at most 0.12.
   - All others: about 0. This means no candidate truth (ours, c21 or c27) makes v26 fail there; it does not mean they pass.
4. **C:** no scale correction is supported. The GT-free estimator scatters by 13 % between clips that use the byte-identical pot mesh, so it cannot resolve 6 %. For every shared mesh,
   the eval clips fall inside the public range. One exception: the **eval20 pitcher sits 2.3 % (about 1.1 cm) deeper than stereo, while both public pitcher clips sit at 0.0 %**.
   That is weak GT-free support for the ep20 suspicion.

## A. ep32 (code/a32.py, a32b.py (E matrices), a32c*.py (frontiers), a32_consensus.py, a32d/a32e.py (exact + noise MC), build_v10.py)
Method:
1. E_k[f, roll] is the exact kit per-frame spoon orientation error of official component k against v6 with the spoon rolled.
2. A Viterbi minimises 2D ball px (rollscan) + lam|droll| (+ an optional c27-twist prior on ball-poor frames) - mu_k * min(E_k, cap).
3. It is subject to: v24 spoon > 54.4 deg (SP fail), 30 < v25, v26 < about 54.5 (MP fail, SP pass).
4. Frontier members were re-scored exactly and with noise MC.

| track | v24 / v25 / v26 spoon deg (exact) | balls px | P(official) 5 deg / 8 deg | dMPPE ep32 25-24 / 26-25 |
|---|---|---|---|---|
| v6 | 5.2 / 40.7 / 50.3 | 9.9 | 0 / 0 | |
| v8_g10 (G10, = v9) | 56.6 / 31.9 / 19.2 | 6.5 | 0 / 0 (v26 would pass) | |
| compose27 twist (own frame) | 74.8 / 45.0 / 53.7 | 9.6 (balls reject f179-199) | v26 SP fail | |
| s0 | 63.0 / 38.4 / 31.2 | 6.55 | 1.00 / 0.60 | -0.46 / -0.06 |
| s4 | 69.8 / 39.3 / 34.1 | 6.62 | 1.00 / 0.65 | -0.60 / -0.03 |
| s6 | 69.4 / 42.8 / 36.0 | 6.68 | 0.95 / 0.30 | -0.50 / -0.06 |
| s8 | 72.8 / 46.0 / 39.3 | 6.75 | 0.45 / 0.00 | -0.50 / -0.06 |
| s10 | 71.6 / 44.4 / 41.6 | 6.85 | 0.80 / 0.10 | -0.53 / +0.03 |
| **consensus (target_final_v10_g12)** | **67.1 / 38.9 / 36.7** | 7.05 | **0.97 / 0.73** | -0.56 / +0.04 |

The official ep32 deltas are MPPE -0.35 (25-24) and about +0.21 (26-25, after removing ep3's -0.37 under v8b). Hidden noise shrinks the predicted deltas.
The 26-25 residual (+0.04 predicted against +0.21) is the one constraint not matched. Tracks that match it make v26 worse than v25 (d-family `out/a32c_delta.json`) and lie within 13-20 deg of the consensus.
Large-margin tracks (s8, s10) fail under noise because the noise pushes v25 into SP fail.

**Predicted pass margin for a new ep32 rollout.** The consistent family members lie 9-20 deg (mean |roll difference|) from the consensus.
- Rollout tracking the consensus within 15 deg: expected hidden spoon error is about 18-25 deg (15 deg ⊕ member offset ⊕ about 4 deg noise), so MP passes with a 5-12 deg margin. I put P(MP pass) at about 0.6-0.8.
- Within 10 deg: P is about 0.8-0.9.
- SP is safe in every case (the cup is about 3 deg).
- H10 found that in-hand roll chasing slips the spoon. Use the grasp-loss position hinge (streamH10_v8 wbeam4 copies).

**Fallback if ep32 is not re-retargeted:** every existing component fails MP. H9_ep32_h19f0 (v25) has the lowest MPPE vs v10 (1.51 vs 1.56 for H10_H2f0) and the highest AUC (0.951), and it was officially SP-pass.
It is about -0.002 official MPPE vs v26.

**ep20 / G11 change: dropped.** The constant pitcher shift is learned from public GT body-frame bias, which is outside "GT only in the scorer". It moves the pitcher about 0.3 cm, so it cannot change MP.
Keeping it would also require an ep20 re-retarget for an MPPE-only hedge. The v10 ep20 equals v8_g10b, so F9_ep20_b4 stays exact (pitcher 0.39/1.7).

## B. Official calibration (code/calib.py; out/pkg/<pkg>_<target>/ep*.json; out/calib_*.json)
- All 8 scored packages were scored per episode against scan-in-v9 (= ours, official-like registration), c21, c27 and v9:
  - first: SP 6 / MP 13
  - pf6: 6 / 13
  - v13: 4 / 11
  - v21C: 4 / 9
  - v22b: 3 / 7
  - v24: 1 / 4
  - v25: 0 / 4
  - v26: 0 / 4
- Labels per episode: o, c21, c27, and i21/i27 (between: MP fails if either fails, SP only if both fail).
  - ep32 is restricted to the known v24-26 pattern.
  - eps 19/25/30 get a twin wildcard for the old packages, because their half-turn registration is a coin flip (Y9).
- An exact DP over episodes weights every labelling by prior x exp(-PEN x |count error|).

| ours-as-truth predicts | first | pf6 | v13 | v21C | v22b | v24 | v25 | v26 |
|---|---|---|---|---|---|---|---|---|
| SP / MP fails | 7/17 | 7/15 | 6/13 | 6/9 | 3/4 | 1/1 | 0/1 | 0/0 (ep32 only via v8) |
| official | 6/13 | 6/13 | 4/11 | 4/9 | 3/7 | 1/4 | 0/4 | 0/4 |

v22b SP {25, 30, 32} matches exactly under ours. The 3 missing MP failures in v22b-v26, together with the excess failures predicted for old packages, are what locate them.

| ep | P(v26 MP fail) over 4 settings (PEN 1/3, prior_o 0.4-0.8) | driver |
|---|---|---|
| 32 | 1.00 | official pattern (A) |
| **20** | **0.76-0.89** | Old packages are 4.1 cm vs ours but 2.1 vs both composes, and they officially had to pass MP to meet the old counts. Two independent older recons agree with compose, about 3.8-4.0 cm from ours. Also C: eval20 pitcher r = 1.023 vs 1.00 on public |
| **33** | **0.48-0.87** | Truth between ours and c21. Mostly driven by old-package SP counts (v13/v21C over-predicted under ours), so fragile. Balls favour ours vs c21 (G10: 74:20) |
| 3 | 0.31-0.60 (about 0.25 with G11 MPPE delta) | v24 29.4 deg vs ours (marginal). G11's v25 to v26 MPPE sign says it passes |
| 1 | 0.30-0.43 | c21 lid spin (v22b+ 44 deg); G11 GT-free spin estimator says ours is right (-1 deg) |
| 19 / 35 / 36 | 0.11-0.32 | intermediate labels only |
| 8 | ≤ 0.12 | |
| 5, 6, 10, 14, 15, 16, 25, 28, 30, 37, 38 | about 0 | not locatable by this method (no candidate truth fails v26 there) |

Confidence: ep20 moderate (two independent signals); ep33 low-moderate; the rest low. The 3 slots sum to about 3 over {20, 33, 3, 1, 19/35/36}.

## C. GT-free mesh scale audit (code/scale_est.py; out/scale/*.json)
On unoccluded, fully visible frames: s_hat = sqrt(rendered area / mask area) x median(z_render / z_stereo). This is invariant to where the tracker put the depth.
The depth ratio r is about 1 everywhere (the fused tracker fits stereo), so s_hat is dominated by mask-area bias.

| mesh (shared eval/public) | eval s_hat (r) | public / held-out s_hat (r) | kit scale ours/scan (scorer-side) |
|---|---|---|---|
| white_pot 734e4be7 | 1.166-1.176 (1.010-1.015) | p41/44 1.165/1.170; h27/42/43/45 1.03-1.05 | 1.00: **estimator scatter 13 % with the same mesh, so it cannot resolve 6 %** |
| water_pitcher 88458e8f | e20 0.845 (**1.023**, +10.7 mm) | p21 0.825 (1.000), h22 0.849 (0.996) | 1.064 (G11) |
| white_pot_lid 3590f16b | e1 0.942 (1.002) | h0 0.940, h2 0.955 (1.020) | 0.97 |
| blue_cup 415ade56 | 0.84-0.91, e14 1.10 | 0.86-0.89, h13 1.11 (both cup-in-rack scenes) | 1.01 |
| beige_cup b30ae10b | e8 0.93, e10 0.87 | p11 0.90 | 1.01 |
| rack f3938548 | 1.017-1.023 | 1.021-1.032 | |
| dust pan bd32cdf2 | 1.016-1.020 | 1.019-1.037 | |
| boards (eval meshes differ from public) | 0.961 / 0.968 | 0.977-0.993 | |
| spoon (no public) | 0.99-1.03 | - | |

- Conclusion: no correction is supported and no mesh was built. Eval is consistent with public for every shared mesh, and public (same files) passes 20/20.
- The only eval-specific anomaly is the ep20 pitcher depth (about 1.1 cm behind stereo; public 0.0). This would be worth a stereo-depth check of the ep20 pitcher track before Sunday, but it is too small alone (3 cm is needed).

## D. Recommendation for v27
- Bundle `selected20_final_v10_g12`, target `target_final_v10_g12`. diff vs v8_g10b: only `episode_000032/object_poses.npz`, plus G12_CHANGES.json and g12_spoon_roll.json.
  G11_CHANGES.json is a stale copy from v9; ep20 is v8's.
- Re-retarget **ep32 only**. Current v26 component vs the new target (score_any; `scores/cand/G12_*_vs_v10`):

| ep32 component | cup cm/deg | spoon cm/deg | MP / SP | MPPE | AUC |
|---|---|---|---|---|---|
| B9_ep32_s6f0 (v24) | 0.53/2.9 | 0.85/67.1 | 0/0 | 2.07 | 0.944 |
| H9_ep32_h19f0 (v25) | 0.55/2.4 | 0.73/38.9 | 0/1 | 1.51 | 0.951 |
| H10_ep32_H2f0 (v26) | 0.61/3.0 | 0.97/36.7 | 0/1 | 1.56 | 0.942 |
| H10_ep32_G1f0 | 0.61/3.2 | 0.96/37.6 | 0/1 | 1.59 | 0.944 |
| H10_ep32_F2f0 | 0.60/3.3 | 0.88/37.4 | 0/1 | 1.55 | 0.945 |
| H9_ep32_h15f0 | 0.57/3.1 | 0.83/38.6 | 0/1 | 1.56 | 0.946 |

- All other 19 episodes: target byte-identical to v8_g10b, so keep the v26 components (ep3 H10_ep03_H2f0 per G11).
- No GT-free fix exists for the located suspects (20, 33). Expected v27 gain is about +0.6 x 0.05 MP from ep32 if a rollout gets within 10-15 deg.

## Files
- code/: a32.py, a32b.py, a32c.py, a32c_smooth.py, a32c_delta.py, a32_consensus.py, a32d.py, a32e.py, build_v10.py, calib.py, scale_est.py, pkg.sbatch, cmd.sbatch, jobs_*.txt.
- out/: a32b_E.npz (per-frame per-roll errors of the 3 official components), a32c*_rolls.npz / json, a32_consensus.*, a32e_*.json, pkg/ (8 packages x 4 targets x 20 eps), calib_*.json, scale/*.json.
- scorer_only_scan_target_v9/: validation only (contains scan meshes); never for submissions or reconstruction.
