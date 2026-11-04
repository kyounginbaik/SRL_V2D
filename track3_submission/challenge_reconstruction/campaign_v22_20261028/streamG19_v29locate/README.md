# Stream G19 (2026-11-01): locate the 3 remaining v29 MP-SR failures

Rules followed:
- No upload, no commit, nothing overwritten. Everything new lives here and in `challenge_submissions_SRL/ablation_knockout_v29_plan_20261101/`.
- Slurm debug on node01; every job took under 50 min.
- Public scans and GT appear only in scorer-side targets (`scorer_only_scan_target_v{14,13,12b,10}`, built with G9's `build_scan_target.py`). They are validation only and never go into a submission or reconstruction.
- Every package pose is a build-v2 rollout.

## 1. Official-consistency calibration with v29 (code/tab19.py, calib19.py, calib19s.py, calib_sum.py)
- **Packages:** first, pf6, v13, v21C, v22b, v24, v25, v26 and **v29**. Official counts: v29 has 0 SP / 3 MP fails.
- **Kit-faithful outcome table:** `out/tab/<label>/ep*.json`, built with the G13 kit wrapper. It matches G12's `out/pkg` on all 480 overlapping (package, truth, episode) outcomes.
- **Labels:**
  - o = scan-in-v14 (new) or scan-in-v9 (G12);
  - c21, c27, and the intermediates ic21/ic27;
  - fro for ep20/33;
  - ep32: o (= v12b consensus), o32_v10, o32_v13. v13 is removed by the hard v24-v26 ep32 pattern; ic27 survives;
  - ep3: o (= v13 roll), o3_v12b, o3_v10;
  - 19/25/30: o_tw.
- **DP:** the exact DP keeps the (F26, F29) failure sets. There are 24 "g12" settings: PEN 1/3, prior img / dedup0.4-0.8 / none, o = v14 or v9.
- **Sensitivity "ext":** fro and ifro labels in every episode. Only 3 exact and 5 FFBS ext settings finished. Ext moves mass to 5/14/16 and lowers ep20 (one ext setting puts ep20 at 0.07). Ext lets the old packages' counts be absorbed by fro-type failures elsewhere. I report it as a sensitivity, not the main result.

**P(fail MP in v29), mixture of 24 g12 settings (exact DP):**

| ep | all g12 | o = scan-in-v14 | o = scan-in-v9 | range over settings |
|---|---|---|---|---|
| 20 | **0.83** (all incl. 3 ext: 0.75) | 0.88 | 0.63 | 0.29-0.99 (low only with img prior + scan-in-v9) |
| 33 | **0.58** | 0.55 | 0.52 | 0.19-0.89 |
| 3 | **0.48** | 0.47 | 0.48 | 0.17-0.81 |
| 1 | 0.35 | 0.34 | 0.32 | 0.10-0.54 |
| 19 | 0.26 | 0.19 | 0.32 | 0.03-0.51 |
| 35 / 36 | 0.21 / 0.20 | 0.19 | 0.23 | 0.08-0.52 |
| 32 | 0.11-0.14 | 0.12 | 0.16 | 0.01-0.37 |
| 16, 5, 30, 14 | at most 0.07 | | | |

**Most likely failing 3-sets (g12 mixture):**
1. {3, 20, 33}: 0.21-0.25.
2. {1, 20, 33}: 0.10-0.14.
3. {1, 19, 20}: 0.06.
4. {3, 20, 35} and {3, 20, 36}: 0.04 each.
5. {19, 20, 33}: 0.03.
6. {1, 20, 35} and {1, 20, 36}: 0.03 each.

**The +1 pass (fail in v26, pass in v29):**
- ep32: 0.86 (scan-in-v14 0.89, scan-in-v9 0.84).
- ep8: 0.03.
- ep15: 0.02.
- New v29 failures: P ≤ 0.03 (ep16, via the c27 spoon at 30.2 deg).

**What v29 adds:** almost nothing beyond confirming ep32. On every label of every other episode, v29's pass/fail equals v26's (`out/tab/*`), except ep6/ep16 under c27. The 3 failures are the same 3 as in v26 minus ep32, and the G12-G14 ranking (20 > 33 > 3 > 1 > 19 > 35/36) stands.

Files: `out/calib/c_*.{json,txt}` (exact DP per setting), `out/samp/s_*.json` (FFBS labelings, 20k per setting), `out/calib_summary.json`.

## 2. Knockout / swap plan (code/ko19.py, finish_ko19.py, cand_view.py, meshcheck.py)
Candidate pool:
- 341 + 6 build-v2 candidates (`common_v2_candidates`, overflow), scored kit-faithfully vs scan-in-v14/v9/v12b/v10/v13, c21, c27 and fro (`out/tab/K_*`).
- All mesh md5s are team meshes; none match a public GT scan (`out/meshcheck_*.json`).

Component roles:
- **Flip components** behave opposite to v29 on the decisive question:
  - **ep20 H8s_ep20_hLRc2048:** pitcher 3.9 cm vs ours, 0.6 cm vs c21/c27, 2.3 cm vs fro.
  - ep19 H8s_ep19_r2Rs46.
  - ep3 H8s_ep03_cem3s3.
- **Certain knockouts:** ep3 B_ep3_level (3.5 cm / 40 deg vs ours), ep33 H8s_ep33_s51c2048 (8 cm / 73 deg), ep35 C_ep35_ret2_plain, ep36 C_ep36_follow3.
- **Hedge:** ep1 H8s_ep01_rk12c512 (lid 26.5 / 19.2 / 10.1 deg vs ours / c21 / c27).

Planner:
- It maximises I(returned count; F29) over designs, with soft outcomes for swapped components (0.5 cm / 5 deg; also 0.8 / 8) and adaptive second uploads.
- **Best single upload: U1 = {3 KO, 20 flip, 33 KO}**, with 1.80 bits (1.62-1.89 across variants) of about 5.1 bits of F29 entropy.
  - Returning 0.90 (P 0.23) means F29 = {3,20,33} (0.96), and ep20 is already fixed by the flip.
- With an adaptive second upload the total is 3.2 bits.
- **Score-first alternative: FIX_flip20_hedge1** (E[MP] 0.894, P(≥0.90) 0.74, 1.5 bits).

Packages, sha256, dry-run results, decode tables and one-line commands: `challenge_submissions_SRL/ablation_knockout_v29_plan_20261101/README.md`. Evaluation: `out/plan/eval_{g12,all,g12_n8}.json`.

## 3. GT-free checks of our reconstruction and the v29 rollouts (code/sens19.py, mkhyp19.py, mkpert.py, evsum.py; G13 ev.py)
**(a) Kit-lever tolerance (`out/sens_summary.json`).** Truth = target_final_v14_g17 with one object's whole track given a constant body rotation about a principal axis. Smallest rotation that makes v29 fail MP:

| episode | object rotated | rotation that fails | effect |
|---|---|---|---|
| ep20 | pitcher (anchor) | **5 deg** (pc0) or 7.5 deg (pc2) | pitcher -> 3.4-4.4 cm, because the scene rotates about the slot-0 cup |
| ep32 | spoon | 5 deg | |
| ep33 | spoon | 7.5 deg (pc0, or roll pc2-) | cup/spoon lever |
| ep3, ep5 | pot (anchor) | 7.5 deg | spoon fails by position |
| ep6, ep8 | pot | 10 deg | |
| ep1 | pot | 12.5 deg | |
| ep19 | sweeper | 7.5 deg | dust pan fails |
| ep36 | spoon | 10 deg | |
| ep35 | spoon | 12.5 deg | |

Constant translations need 1.5-3 cm.

These levers match the compose-vs-ours pattern: c21 vs v29 in ep20 is pitcher 3.9 cm with only 6 deg orientation.

**(b) Are those levers visible in our images?** (`out/pert_summary.json`; IoU and stereo depth over the whole clip, RGB clock.)
- **ep20 pitcher, 5-7.5 deg: clearly rejected.** IoU 0.85 -> 0.73-0.80, with ours better on 198-202 of 202 frames. A pose-only error of our pitcher track cannot produce the ep20 failure. If ep20 fails, the cause is outside the image-visible pose: the kit registration of our pitcher mesh (6.4 % larger than the scan, G11) onto the hidden scan, or the GT itself. The ep20 flip upload is the only way to settle it.
- **ep33 spoon roll -7.5 deg: NOT detectable** (IoU 0.699 vs 0.706, frames 86:125). Thickness-axis rotation is rejected (316:19).
- **ep32 spoon roll -5 deg: slightly preferred** (96:61 frames, IoU +0.003), consistent with G16's -15 deg bias.
- **ep3 pot.** Spin about the vertical (pc0) by 7.5 deg is nearly invisible (IoU -0.003, 67:5 frames). Tilt (pc2) is rejected by depth (6.8 -> 8.9 mm).

**(c) Alternative tracks vs images** (`out/ev_summary.json`; c21 spoon only, because the other c21 meshes are scan templates).
- ep3: on its gross windows (>30 deg from ours), the fro spoon ties ours (58:49 frames, depth better). c21 loses (45:92).
- **ep33: c21 spoon ties or wins on silhouettes** (90:82, IoU 0.728 vs 0.706) but loses on depth (10.3 vs 6.9 mm). G10's balls favoured ours 74:20.
- **ep36: c21 spoon wins silhouettes 119:28** (0.75 vs 0.72) but loses depth (11.1 vs 6.6 mm).
- ep35: c21 wins 51:26.
- ep20 fro pitcher, ep1 fro lid and ep19 fro sweeper all lose clearly.

**(d) v29 rollouts vs our target** (camera frame). These rollouts are the v29 *execution*.
- ep3 spoon: 18.1 deg mean from v14, with 28 frames over 30 deg (IoU 0.36 vs 0.74). This is the main ep3 risk: the margin vs our own target is only 11 deg (v29 19.1 deg vs v14).
- ep35: 6.7 deg. ep36: 4.7 deg.
- ep20, ep33 pitcher/spoon: about 1 cm.

**Fix candidates (not built):**
1. **ep20:** if U1 or FIX returns ep20 as compose-type, adopt H8s_ep20_hLRc2048 in the final package. It is a build-v2 rollout with the same meshes. No GT-free reconstruction fix exists, because the images reject the needed pitcher rotation.
2. **ep3:** re-retarget for a tighter spoon (target ≤ 10 deg vs v14; H13 reached 9.8 vs v12b). Separately, add a GT-free pot-yaw estimate from the handles (G11-style handle depth scan): 7.5 deg of pot yaw is invisible to IoU and alone fails ep3/ep5.
3. **ep33:** ball-based spoon roll estimate (G16 tools) for a constant -5..-10 deg roll. Silhouettes cannot tell. If ep33 is confirmed failing, retarget onto the rolled track.
4. **ep35/36:** if located, the c21-like spoon windows that win silhouettes but lose depth deserve a ball check.

## Not done
- Ext label set incomplete: exact DP 3/24 and FFBS 5/24 settings finished; the rest timed out at the 50 min limit.
- No new retargets.
- No ball-based roll checks.
- Pot-yaw estimator not written.
- The c21 non-spoon tracks were not image-checked, because their meshes are scan templates.
- No videos rendered for swapped episodes (diagnostic packages have no review videos for those episodes).
