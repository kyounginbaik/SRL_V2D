# Stream G14 (2026-10-29): hedged (trust-region minimax) reference + v27 knockout plan

Rules followed:
- Our meshes only. No public scan or GT mesh was used in any target or bundle.
- Public GT entered only the unchanged public scorer (`tools/reconstruction_audit/score_bundle.py` -> bscore).
- Compose (c21) was used only for spoon and rack. Those are team meshes; no scan template was used for them.
- Housekeeping: no submitted pose was edited, nothing was overwritten, nothing was uploaded, and nothing was committed. Slurm debug on node01; every job took under 50 min.

## Bottom line
1. **The hedge cannot make MP-SR 1.0 reachable by itself.** On eval, every *admissible* GT-free hypothesis is already within
   **2.5 cm / 17 deg (time-mean)** of v10 in every episode (worst ratio max(pos/3, ori/30) = 0.83, ep19 dust pan vs fro).
   - All disagreements that could flip MP are flip- or gross-type: spoon roll in 3/5/6/15/16/32/33/35/36, and the ep14/ep28/ep32 tracker losses.
     The rule must not hedge these, so the base keeps its ball, image or official evidence.
   - The ep20 pitcher vs fro gap (3.6 cm, mostly frames more than 3 cm apart) is also outside the admitted set.
2. **Rule chosen (one fixed rule for all clips, GT-free): "H4"**, the time-level trust-region minimax:
   - Gate: a hypothesis frame counts only if it is within 3 cm / 30 deg of the base.
   - Admission: a hypothesis is admitted for an object only if at least 80 % of its frames pass the gate.
   - Margin: 1.0 cm / 10 deg.
   - Public / held-out results:
     - Our trackers only: 20/20, MPPE 1.1719, AUC 0.9312 (r8_base 1.1714 / 0.9312). Neutral: only pub21 changes, pitcher +0.02 cm.
     - With the fro-family source (the old pipeline's stage-B metric + polish, which exists for pub 11/39/41/44): 20/20, **MPPE 1.1642, AUC 0.9318**.
       It is better because pub44 pot goes from 1.24 to 1.08 cm. Evidence for the gain is a single clip.
   - Looser gates (6 cm / 60 deg, H1/H2) and every per-frame variant fail pub24 (19/20). They hedge toward a wrong tracker.
3. **Built (new, nothing overwritten):**
   - `reconstruction_recovery_20261026/selected20_final_v11_g14`: v10_g12 plus the H4 hedge. 12 episodes are touched, 5 of them materially (table C).
   - `campaign_v22_20261028/target_final_v11_g14`: the meshes are byte-identical to v10.
4. **Official consistency:** v22b, v24, v25 and v26 have identical SP/MP outcomes against v10 and v11 on all changed episodes, so the 4 v26 failures stay exactly as (un)explained as before.
   The old packages *first* and *pf6* lose predicted MP failures under v11:
   - first ep8 pot: 3.33 -> 2.74 cm.
   - first ep16 spoon: 3.47 -> 2.88 cm.
   - pf6 ep16 spoon: the same move as first.
   Their ours-as-truth over-prediction (17 vs 13 and 15 vs 13 official) shrinks to 15 and 14. This is weak evidence in the hedge's favour.
5. **v27 against the hedged target:**
   - Every v27 component still passes MP.
   - Worst deltas: ep8 pot 0.35 -> 0.93 cm, ep19 dust pan 0.74 -> 1.39, ep16 spoon 0.84 -> 1.14, ep1 lid 0.53 -> 0.80.
   - No re-retarget is needed for MP. Adopting v11 is an MPPE hedge only (expected effect ±0.02 official MPPE).
6. **Knockout plan (part 4)** is in `challenge_submissions_SRL/ablation_knockout_v27_plan_20261029/README.md`:
   - One upload: `SRL_one_3_20_32_33` gives 1.71 of 2 possible bits about the 3 unlocated failures. A returned 0.80 means exactly {3,20,33}.
   - Two uploads (adaptive second): 2.96 bits.

Confidence:
- High that the hedge is public-safe (no-op or better) and does not change MP on eval.
- Moderate that it improves MPPE: there is one public clip of evidence.
- High that the 3 failures are not in the hedgeable (<2x threshold) disagreements of our GT-free sources.

## A. Rule (`code/hedge.py`)
The rule works in the kit's own representation:
- Per object: the registered body anchor (the surface centroid of OUR mesh) plus the body quaternion.
- An alternative whose mesh differs from ours is put into our body frame by the kit's `_register_object_body`. If the registration rotation is larger than 20 deg it is a twin and is not trusted.
- Every alternative is aligned onto the base by the kit's `_align_registered_scene`, which uses frame 0: the anchor-slot rotation and the slot-0 position.
  So the hedge sees exactly the disagreement the scorer would see, and frame 0 is never moved.
- Symmetric objects (cups, lid) use the symmetry-equivalent of each alternative that is nearest to the base.
- Position (cm) and orientation (base-tangent rotation vector, deg) are hedged separately.

Per object, "time" mode (H4):
1. **Gate.** Frame t of hypothesis k counts only if it is within 3 cm AND 30 deg of the base. Flip- and gross-type frames never count.
2. **Admission.** Hypothesis k is admitted only if at least 80 % of its frames pass the gate.
3. **No-change test.** If every admitted hypothesis has a time-mean distance (over its gated frames) of at most margin M = 1.0 cm / 10 deg, nothing changes.
4. **Otherwise:**
   - rho = max(M, minimax value), where the minimax value is taken over {base} ∪ admitted and found by bisection.
   - The correction is the smallest per-frame track, in the L2 sense, that keeps every admitted hypothesis within rho in time-mean.
   - It is solved by dual ascent, with a per-frame fixed-point solve of min ||c||² + Σ w_k ||c - h_k||.
5. Frame 0 is never moved.

Sources:
- Our trackers: raw FoundationPose, multimap, pointmap_cotracker, and bundle_adjusted where it exists.
- fro: final_recon for 13 episodes, final_recon_ours for 14/15/16/20/25/28/30.
- c21: spoon and rack only.

A source object is admitted only if its mesh is our team mesh: byte-identical, resampled with sorted extents within 0.3 cm, or with the same vertex count and extents within 1.0 cm.
This excludes:
- fro blue cup, ep28 pot, sweeper and pieces in eps 14-30 (compose templates);
- every c21 pot, cup, pitcher, lid and dust pan.

Earlier solver: `hedge_v1_wgm.py`, used for variants T* and G*, was a weighted-geometric-median version. It is all-or-nothing per frame, so those variants are superseded and not reported.

## B. Public / held-out validation (GT only in the scorer; `code/make_public.py`, `code/compare14.py`, `out/public_compare.json`)
- Base: r8_base. Exports: `canonical/<split>/episode_X/exports/g14_<tag>`.
- Clips that are bit-identical to r8_base take r8_base's score, because the scorer is deterministic. Every changed clip was scored.
- `frolike` = public_gt_check_20261020 stage-B metric FoundationPose + polish (`pp_<obj>_metric.npz`). This is the same family as fro, on pub 11/39/41/44.

| variant | mode | gate | margin | sources | MP | pos cm | ori deg | MPPE | AUC |
|---|---|---|---|---|---|---|---|---|---|
| r8_base | - | - | - | - | 20/20 | 1.114 | 5.23 | 1.1714 | 0.9312 |
| A | per-frame | 6/60 | 1.0/10 | ours | 19/20 | 1.212 | 6.47 | 1.2819 | 0.9170 |
| B | per-frame | 6/60 | 0 (pure minimax) | ours | 19/20 | 1.202 | 6.62 | 1.2762 | 0.9171 |
| C / D / F | per-frame | 6/60 | 1.5/15, 0.5/5, no smoothing | ours | 19/20 | 1.21-1.23 | 6.2-6.6 | 1.27-1.31 | 0.914-0.918 |
| E | per-frame | 3/30 | 1.0/10 | ours | 20/20 | 1.124 | 5.47 | 1.1830 | 0.9299 |
| H1 / H2 | time, admit 0.8 | 6/60 | 1.0/10, 1.5/15 | ours | 19/20 (pub24 piece_1 1.61 -> 3.19) | 1.16 | 5.23-5.28 | 1.216 / 1.211 | 0.926 / 0.927 |
| H3 | time, admit 0.8 | 4.5/45 | 1.0/10 | ours | 20/20 | 1.115 | 5.23 | 1.1725 | 0.9311 |
| **H4** | time, admit 0.8 | **3/30** | **1.0/10** | ours | 20/20 | 1.115 | 5.23 | 1.1719 | 0.9312 |
| FH1 / FH2 | time | 6/60 | | ours + frolike | 19/20 | 1.16 | 5.23-5.27 | 1.200 / 1.211 | 0.928 / 0.927 |
| FH3 | time | 4.5/45 | 1.0/10 | ours + frolike | 20/20 | 1.105 | 5.23 | **1.1563** | **0.9330** |
| **FH4 (= rule H4)** | time | 3/30 | 1.0/10 | ours + frolike | 20/20 | 1.110 | 5.23 | **1.1642** | **0.9318** |

Reading the table:
- Our trackers rarely disagree moderately with the fused base. They either agree or are grossly off (twins, losses), and the fused base already picked the best of them per frame with image evidence.
- Ensembling them cannot help, and admitting moderate outliers hurts: pub24 piece_1 is hedged toward a wrong tracker and fails.
- The only gain comes from an *independent* pipeline (frolike): pub44 pot 1.24 -> 0.90 (FH3) or 1.08 (FH4).
- I chose H4 over H3 because H3 admits fro/c21 spoon hypotheses up to about 29 deg in eps 3/5/6. Those are roll-flip territory, where the ball/official evidence must win.
- Held-out: the 10 clips are unchanged under H4/FH4 (no admitted disagreement exceeds the margin).

## C. Eval: worst-case predicted time-mean error vs admitted hypotheses (`code/make_eval.py`, `code/summarize_eval.py`, `out/eval_H4/summary.{txt,json}`)
- "Worst" means the maximum over admitted hypotheses of the time-mean pos / ori between the reference and that hypothesis (kit-aligned, all frames).
- "Gross" lists the hypotheses outside the hedge. These are flip- or gross-type and are never hedged.
- An asterisk (*) marks a material change (> 0.5 cm or > 5 deg time-mean move vs v10).

| ep | object: worst before (v10) -> after (v11), by hypothesis | worst ratio max(p/3, o/30) before -> after | gross (not hedged) |
|---|---|---|---|
| 1 | pot 1.78/11.1 -> 1.44/10.5 (fro); **lid 1.66/11.6 -> 1.01/11.0 (multimap) \*** | 0.59 -> 0.48 | lid fro 3.9/35 |
| 3 | pot 0.65/4.3 -> same; spoon none admitted | 0.22 -> 0.22 | spoon raw/MM/BA 98-153 deg, PM 9.4 cm, c21 2.3/33 |
| 5 | pot 1.63/7.5 -> 1.41/7.5 (fro) | 0.54 -> 0.47 | spoon raw 82 deg, PM 6.3/41, fro 3.1/22 |
| 6 | pot 1.62 -> 1.20 (fro); spoon 1.66/16.1 -> 1.17/13.1 (c21) | 0.55 -> 0.44 | spoon MM 32 deg, PM 7.1/37 |
| 8 | beige 1.67/8.2 (raw); **pot 1.89/4.3 -> 1.04/4.3 (fro) \*** | 0.63 -> 0.56 | beige BA 4.9/60 |
| 10 | blue 1.78 -> 1.41 (fro); **beige 1.57 -> 1.59 (fro) \*** (move 0.55) | 0.59 -> 0.53 | blue raw 3.4/64 |
| 14 | blue 0.03; rack 0.92/1.5 | 0.31 | blue MM/PM 16-18 cm |
| 15 | rack 0.22; spoon 0.78/12.9 -> 0.78/10.3 (c21) | 0.43 -> 0.34 | spoon MM 36 deg, PM 15/68, fro 3.3/14 |
| 16 | rack 0.29; **spoon 2.03/15.1 -> 1.21/11.2 (c21) \*** | 0.68 -> 0.40 | spoon MM 39 deg, PM 13/75, fro 4.2/15 |
| 19 | **dust pan 2.48/6.6 -> 1.88/6.6 (fro) \***; sweeper none admitted | 0.83 -> 0.63 | - |
| 20 | blue 1.25 -> 1.01 (MM); pitcher 0.13 (raw) | 0.42 -> 0.34 | pitcher MM 4.4/32, PM 10.5/31, **fro 3.6/5** |
| 25 / 30 | pieces and sweeper: no admitted alternative (canonical/fro meshes differ); rack 0.30 | 0 / 0.10 | - |
| 28 | rack 0.42; pot 0.05 | 0.14 | pot MM/PM 11 cm |
| 32 | blue 1.65 -> 1.30 (fro); spoon none admitted | 0.55 -> 0.43 | cup raw/MM/PM 9-10 cm/155 deg; spoon trackers 41-42 cm, fro 3.9/69, c21 1.9/41 |
| 33 | blue 1.41/14.5 -> 1.26/12.8 (BA); spoon none admitted | 0.48 -> 0.43 | **spoon: every source 45-66 deg** (raw 4.3/45, MM/BA 3.7/63, PM 5.1/64, fro 3.8/60, c21 10.7/66) |
| 35 | spoon none admitted | 0 | spoon raw/MM/PM 25 cm, fro 4.2/34, c21 7.4/35 |
| 36 | spoon none admitted | 0 | **spoon: every source 42-72 deg** (raw 0.5/54, BA 1.4/51, fro 2.8/42, c21 4.9/52) |
| 37 | spoon 0.42/5.1 | 0.17 | - |
| 38 | dust pan 1.13/9.5 -> 1.07/9.5 (fro) | 0.38 -> 0.36 | - |

Abbreviations: MM = multimap, PM = pointmap_cotracker, BA = bundle_adjusted.

What the gross column says about the unlocated failures:
- **ep33 and ep36:** every independent source (our trackers, fro, c21) puts the spoon 42-72 deg (time-mean) from v10. v10 is the ball-roll-corrected track (balls 74:20 vs c21 in G10).
- **Single-object episodes (35, 36, 37, 38):** an orientation-only failure is impossible, because v26 SP-SR = 1.00 and SP is the object-mean ori ≤ 28.6 deg. So a v26 MP failure there must come from position (≥ 3 cm). The only source with a 3-4 cm spoon offset is fro (ep35 4.2 cm, ep36 2.8 cm).
- **ep20:** the fro pitcher is 3.6 cm away, in time-mean with all frames.
- **ep33:** the c21 spoon is 10.7 cm away.

These remain the candidates (consistent with G12/G13). None can be hedged under the 2x-threshold rule.

## D. Material changes (> 0.5 cm or > 5 deg) and v27 components (`code/pkg_compare.py`, `out/pkg_compare.txt`)
v27 components are scored with the official-path scorer (`score_vs_compose.py`, the same as `code/score_any.sh`). The v27 package was scored against each target.

| ep | changed object (move cm/deg) | v27 component | vs v10: objects / MPPE / AUC | vs v11_g14: objects / MPPE / AUC | MP |
|---|---|---|---|---|---|
| 1 | lid 0.66/1.3, pot 0.33/0.6 | E9_ep01_b3 | pot 0.53/3.0, lid 0.53/4.9 / 0.612 / 0.984 | 0.67/3.1, 0.80/5.0 / 0.788 / 0.977 | 1 -> 1 |
| 8 | pot 0.85/0.0 | F9_ep08_b4 | cup 0.29/1.4, pot 0.35/1.7 / 0.339 / 0.998 | 0.29/1.4, 0.93/1.7 / 0.623 / 0.988 | 1 -> 1 |
| 10 | beige 0.55, blue 0.37 | X8_ep10_Blr512 | 0.43/1.9, 0.57/2.7 / 0.528 / 0.991 | 0.51/1.9, 0.79/2.7 / 0.674 / 0.987 | 1 -> 1 |
| 16 | spoon 0.82/3.9 | E9_ep16_b8 | rack 0.39/2.6, spoon 0.84/5.3 / 0.687 / 0.979 | 0.39/2.6, 1.14/5.6 / 0.815 / 0.970 | 1 -> 1 |
| 19 | dust pan 0.97/0.0 | D9_ep19_b5 | sweeper 0.82/8.5, pan 0.74/3.9 / 0.910 / 0.966 | 0.82/8.5, 1.39/3.9 / 1.224 / 0.942 | 1 -> 1 |

Below 0.5 cm and 5 deg (all MP 1 -> 1): eps 5, 6, 15, 20, 32, 33 and 38 (MPPE change ≤ +0.05). Eps 3, 14, 25, 28, 30, 35, 36 and 37 are byte-identical to v10.
Recommendation:
- Do not re-retarget for v11. Every v27 component passes against it, with at least 1.6 cm and 10 deg of margin (the tightest is the ep32 spoon at 19.8 deg).
- Adopt v11 as the target only if the team accepts the one-clip public MPPE evidence.

## E. Official-consistency (`out/pkg/<pkg>_{v10,v11}/ep*.json`, `out/pkg_compare.txt`)
The 9 packages were scored on the 12 changed episodes:

| package | SP/MP fails (12 eps), truth = v10 | truth = v11 | official (20 eps) |
|---|---|---|---|
| first | 3 / 11 | 3 / 9 (ep8 pot 3.33 -> 2.74, ep16 spoon 3.47 -> 2.88) | 6 / 13 |
| pf6 | 3 / 9 | 3 / 8 (ep16) | 6 / 13 |
| v13 | 3 / 8 | 3 / 8 | 4 / 11 |
| v21C | 2 / 6 | 2 / 6 | 4 / 9 |
| v22b | 1 / 1 | 1 / 1 | 3 / 7 |
| v24 | 1 / 1 | 1 / 1 | 1 / 4 |
| v25 / v26 | 0 / 1 | 0 / 1 | 0 / 4 |
| v27 | 0 / 0 | 0 / 0 | not uploaded |

- No outcome changes for v13 through v26, so G12/G13's calibration posteriors are unchanged.
- The only changes reduce the old packages' over-prediction. G12: ours predicts first 17 / pf6 15 MP failures against an official 13 / 13. Under v11 these become 15 / 14.

## F. Knockout plan (part 4)
- Code: `code/calib14_joint.py` (G13's calib DP keeping the v26 failure SET in the state, giving a joint posterior; `out/joint_*.json`, 8 settings), `code/ko_plan.py` and `code/ko_decode.py` (`out/ko_plan_p0.7.txt`, `out/ko_decode.json`).
- Parquets (made with `ablation_knockout_20261020/make_knockouts.py` on v27 sha 98a52df6, MP-SR-only upload commands) and decode tables: `challenge_submissions_SRL/ablation_knockout_v27_plan_20261029/`.
- Mixture prior P(v26 MP fail): ep20 0.67, ep33 0.66, ep3 0.54, ep1 0.37, ep19 0.24, ep35 0.25, ep36 0.25. ep35/36 are exchangeable; prior entropy is 4.39 bits.

One upload: **S = {3, 20, 32, 33}**, which carries 1.71 bits about the 3 failures (worst setting 1.35; 2.0 is the maximum).
- 0.80: the 3 failures are {3, 20, 33}.
- 0.75: 2 of them plus 1 of {1, 19, 35, 36}.
- 0.70: 1 of them plus 2 others.
- 0.65: none of them; the failures are {1, 19, 35/36}.

Two uploads:
- Adaptive second upload, chosen from the first result: 0.75 -> {1,20,32,33,35}; 0.70 -> {3,32,35,36}; 0.65 -> {1,19,32,35}. Total 2.96 bits.
- Non-adaptive pair: {1,19,32,35,36} + {3,32,33,35,36}, 3.00 bits.
- Plain v27 alone gives 0.88 bits, all about ep32.

## Files
- code/:
  - hedge.py (rule) and hedge_v1_wgm.py (superseded solver);
  - make_public.py, make_eval.py, summarize_eval.py, compare14.py, build_v11.py, pkg_compare.py;
  - calib14_joint.py, ko_plan.py, ko_decode.py;
  - *.sbatch and job lists.
- out/:
  - eval_<tag>/ep*.{npz,json} (hedged poses and per-hypothesis diagnostics; H4 = adopted);
  - public_compare.json, pkg/, pkg_compare.{txt,json};
  - joint_*.json, ko_plan_p0.7.txt, ko_decode.json.
- Bundle `reconstruction_recovery_20261026/selected20_final_v11_g14`: G14_CHANGES.json, plus per-episode g14_hedge.json.
- Target `campaign_v22_20261028/target_final_v11_g14`.
- Public exports `canonical/{public,public_heldout}/episode_X/exports/g14_*`. The T*/G* exports are from the superseded solver.
