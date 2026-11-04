# Stream G15 (phase 9): error anatomy and pose accuracy of the reconstruction, applied to eval 20

Rules followed: our meshes only. No mesh was changed in the deliverable. Public GT and scans were read only inside scorer-side diagnostics
(`kitx.py`, `anatomy.py`, `tshift*.py`, `oracle_split.py`, `smooth_scan.py`, `depthshift_scan.py`, `meshdiag.py`, `cmp.py`). Every adopted fix is
GT-free. It was validated on the 10 tuning clips and confirmed on the 10 held-out clips. Nothing was overwritten. New outputs are the
`exports/g15_*` folders, the new bundles and targets, and this directory. No upload, no commit, and no submitted pose was edited. CPU jobs ran on node01 (debug), each under 50 min.

## Result

**Deliverable:** `reconstruction_recovery_20261026/selected20_final_v12b_g15` + `campaign_v22_20261028/target_final_v12b_g15`.
- It is v10_g12 plus two GT-free fixes: (1) a camera path with clock-consistent stereo lifting, and (2) every pose re-timed onto the stereo/GT clock.
- Meshes are byte-identical to v10_g12. The ep32 and ep3 spoon roll decisions and the ep25 and ep30 marker meshes are kept.
- `selected20_final_v12_g15` / `target_final_v12_g15` is an intermediate (retime only). It is superseded by v12b.

| public (kit scorer, GT only in scoring) | MP-SR | mean pos cm (max) | mean ori deg (max) | MPPE | AUC |
|---|---|---|---|---|---|
| tuning 10, r8_base (before) | 10/10 | 1.219 (2.20) | 4.00 (9.3) | 1.226 | 0.9335 |
| tuning 10, **G15 final (g15_camrt)** | 10/10 | **1.072** (2.09) | **3.79** (9.4) | **1.093** | **0.9496** |
| held-out 10, r8_base (before) | 10/10 | 1.002 (1.64) | 6.54 (28.7) | 1.117 | 0.9288 |
| held-out 10, **G15 final** | 10/10 | **0.921** (1.43) | **6.18** (28.1) | **1.035** | **0.9379** |
| all 20, before -> after | 20/20 | 1.114 -> 0.999 | 5.23 -> 4.95 | 1.171 -> 1.064 | 0.9312 -> 0.9437 |

**The "almost perfect" target (every object <= 0.5 cm / 3 deg, MPPE well under 1, AUC >= 0.96) was NOT reached, and the evidence says it is not reachable by post-hoc pose fixes.**
- After the fixes, even an oracle all-frame scene alignment (GT-fitted, diagnostic only) leaves 0.66 cm mean.
- The rest is per-frame tracking error while objects move: 1.38 cm moving vs 0.88 cm static before the fixes.
- Getting below that needs re-tracking with a better tracker or better depth (see "Next" below).

Per-object public before -> after (cm/deg): `out/public_r8base_vs_final.txt`. Largest gains:
- tuning: pub21 pitcher 1.49->1.11; pub12 cup 1.45->1.18; pub31 rack 1.31->0.88; pub24 pieces 1.61/1.58->1.29/1.30; pub18 dust pan 2.03->1.81.
- held-out: pub22 cup 1.14->0.80 and pitcher 1.64->1.43; pub42 pot 0.72->0.56; pub13 cup 1.10->0.90.

Small losses: pub27 rack 0.29->0.36, pub26 rack 0.38->0.43, pub23 piece_2 1.08->1.10, pub18 dust pan ori 6.7->7.2.

## 1. Error anatomy (r8_base, 33 objects, kit-faithful; `code/anatomy.py`, `out/anat_*.json`, per-frame `out/pf_*.npz`)

`kitx.py` reimplements the kit path and exposes the aligned tensors. It reproduces public_score.json exactly (e.g. pub11 0.318/1.41, 0.930/3.00). The steps are: register our budgeted mesh onto the scan, convert the body frame, align the scene to the anchor at frame 0, then apply symmetry. Errors are decomposed in the GT body frame, the aligned camera frame (along-ray vs lateral), and the world frame. The pieces:
- **(T) Clock offset between the RGB camera and the GT: the largest single fixable term (~0.1 cm of 1.11).** Shifting our tracks in time (`tshift.py`) lowers the error for every moving object, with the best shift at +0.75 to +1.5 frames. This holds for raw FoundationPose, CoTracker pointmap and multimap alike, so it is not tracker lag. It is constant within a clip (`tshift_win.py`). GT-free cause (`camsync.py`, optical-flow cross-correlation of the three ego cameras): **ego_cam_a (RGB) lags the stereo pair ego_cam_b/c by 0.875-0.925 frames in 37/40 clips.** pub21 lags by 1.88. pub22 and eval20 have a one-frame step: a frame drop at f49 and f53 respectively (`camsync_fine.py`, `out/sync_delta_t_summary.json`). The per-clip GT-optimal shifts track these numbers, including about 1.9 for pub21. The stereo depth (b/c) is also ahead of the RGB poses: `depthclock.py` gives a GT-free depth-residual minimum at +0.45 to +0.9 on 11 moving objects.
  **The public GT is on the b/c clock. Our poses are on the RGB clock, and our stereo lifting mixes the two.**
- **(d) Frame-0 scene alignment: the largest remaining term, not fixable post hoc.** Mean pos 1.03 cm (after T) -> 0.66 cm with an oracle all-frame alignment (`oracle_split.py`). The optimal scene rotation is 4-6 deg in pub12, 18, 31, 44, 0 and 43. The error is a relative placement bias:
  - pub12: the cup at f0-30 is 2 cm off relative to the rack and to its own later frames.
  - pub18: the sweeper f0 orientation rotates the dust pan.
  - These are biases, not noise. R8 already showed that f0 window averaging is neutral.
- **(a) Mesh geometry/scale.** Scorer-side (`meshdiag.py`, diagnostic only), relative to our meshes the scans are:
  - larger: rack 7 % (long axis 44.4 vs 50.9 cm, scaled-ICP chamfer 0.7 cm), dust pan 8 %, pub18 sweeper 8 %, pieces 5 %;
  - smaller: pitcher 6 %;
  - about the same: cups and pot.

  The GT-free silhouette refinement of our meshes (`meshfit.py`) gives tuning 1.097->1.059 but held-out 0.950->0.983, so it is **rejected**. Our SAM2 masks are systematically ~4 % tight (cups shrink 4-5 % although their scale is right). The fits do not recover the scan dimensions (rack long axis only +2 %) and are unstable on thin axes and on pots 41/44.
- **(b) Tracking depth bias.** Single-object constant camera-frame biases are invisible after f0 alignment. They matter only as relative placement between objects, which is (d).
- **(c) Camera path.** The camera path lifts SIFT keypoints with depth from the other clock (see T). Fixing that gives a small gain on both splits (fix 2).
- **(e) Lid spin: not addressed.**
  - The constant body-frame spin is 16 deg (pub0) and 28.6 deg (pub2).
  - The scorer's registration aligns our lid handle with the scan handle to within 2 deg, so the spin error is in tracking, not in the mesh frame.
  - Our lid mesh disc is elliptical (eigen-ratio 0.94).
  - G11's handle-depth estimator needs a GT-calibrated offset, so it was not adopted.
  - It affects eval ep1 only, which G11 rated likely fine.
- **Jitter.** Gaussian smoothing (`smooth_scan.py`) is neutral (at most -0.1 deg), so it was not adopted.

## 2. Fixes (GT-free; tuning -> held-out)
| step | tool | tuning pos / MPPE / AUC | held-out pos / MPPE / AUC | verdict |
|---|---|---|---|---|
| r8_base | | 1.219 / 1.226 / 0.9335 | 1.002 / 1.117 / 0.9288 | before |
| + retime, constant delta per clip | `retime.py` | 1.097 / 1.118 / 0.9485 | 0.950 / 1.057 / 0.9357 | adopted |
| + retime, piecewise delta (frame-drop steps) | `sync_profile.py`, `retime.py KEY` | 1.097 / 1.117 / 0.9487 | 0.952 / 1.059 / 0.9355 | adopted (neutral on public; needed for eval20's step at f53) |
| + clock-consistent camera path (keypoints lifted at their flow-predicted t+delta position) | `camera_sync.py` | **1.072 / 1.093 / 0.9496** | **0.921 / 1.035 / 0.9379** | **adopted (final)** |
| silhouette mesh refinement, scale+t / full | `meshfit.py` | 1.059 / 1.081 / 0.9532 ; full 8/10 MP | 0.983 / 1.078 / 0.9344 ; 9/10 MP | rejected |
| temporal smoothing sigma 0.7-3 | `smooth_scan.py` | +-0.01 cm, -0.1 deg | same | neutral, not adopted |
| along-ray component shifted less (alpha 0-0.75) | `depthshift_scan.py` | best alpha 0.75 | best alpha 0 | inconsistent, not adopted |

## 3. Eval bundle `selected20_final_v12b_g15` (GT-free QC)

What changed:
- **Retime.** Every episode has delta 0.875-0.925 frames. eval20 is piecewise: 0.925 for f0-52, then 1.85.
- **Camera path.** The path changed by a mean of 0.07-0.6 cm per episode; ep28 changed by 1.13 cm. Valid fraction is unchanged (1.0, ep19 0.996, ep33 0.991->0.994). Reprojection medians are unchanged (0.7-1.0 px). The ep14 path is jerkier (max acceleration 4.4 -> 8.2 cm/frame²); eps 6, 21 and 24 are smoother (`out/camera_eval_diff.txt`).
- **Unchanged.** camera_from_object, all spoon roll, rescue and lid decisions, and the meshes. The old camera is kept as `camera_pre_g15.npz`.

QC caveat:
- `qc.py` IoU is measured against RGB-clock masks. Retimed poses therefore lose IoU by construction: mean IoU p10 0.672 -> 0.609; the spoons are hit hardest (`out/` qc via `code/qccmp.py`, folders `qc/`).
- The underlying track is unchanged and is only resampled 0.9 frame later.
- The clock-consistent check is the stereo depth, which is on the b/c clock. `depthclock.py` on public shows the depth residual prefers the shifted poses (e.g. pub39 dust pan 12.2 mm at -0.9 vs 4.5 at 0; pitcher 26.3 at 0 -> 20.5 at +0.9).
- Ball pixel QC is likewise RGB-clock and was not recomputed.

## 4. Eval: current v27 components vs the new target (`code/score_any.sh`, `out/eval_v27_vs_targets_v12b.txt`)
All 20 stay MP 1 / SP 1 against target_final_v12b_g15. MPPE goes 0.710 -> 1.025 and AUC 0.9747 -> 0.9540. This is the cost of the rollouts lagging the b/c clock by about 0.9 frame, plus the camera change.

| ep | v27 comp | MPPE vs v10_g12 -> vs v12b_g15 | AUC | worst object vs v12b |
|---|---|---|---|---|
| 20 | F9_ep20_b4 | 0.347 -> 1.036 | 0.9958 -> 0.9599 | pitcher 1.22/3.3 (step to 1.85 frames) |
| 5 | C9_ep05_beam6 | 0.755 -> 1.403 | 0.9765 -> 0.9151 | spoon 1.50/7.2 |
| 35 | C9_ep35_beam6 | 0.889 -> 1.390 | 0.9534 -> 0.9195 | spoon 1.27/7.9 |
| 37 | K8_ep37_ilc1 | 0.282 -> 0.733 | 1.000 -> 0.9776 | spoon 0.69/3.4 |
| 32 | H12_ep32_e1f0 | 1.135 -> 1.548 | 0.9545 -> 0.9228 | spoon 1.31/22.4 |
| 38 | F9_ep38_b3 | 0.438 -> 0.841 | 0.9979 -> 0.9767 | dust pan 0.76/4.2 |
| 28 | F9_ep28_b1 | 0.655 -> 1.021 | 0.9713 -> 0.9524 | pot 1.18/6.0 |
| 36 | C9_ep36_beam5 | 0.817 -> 1.166 | 0.9704 -> 0.9524 | spoon 1.02/7.3 |
| 33 | C9_ep33_beam4 | 0.820 -> 1.141 | 0.9679 -> 0.9478 | spoon 1.15/8.7 |
| 6 | D9_ep06_b2 | 0.788 -> 1.097 | 0.9769 -> 0.9502 | spoon 1.27/8.6 |
| 19 | D9_ep19_b5 | 0.910 -> 1.198 | 0.9656 -> 0.9418 | sweeper 1.13/9.2 |
| 3 | H10_ep03_H2f0 | 0.784 -> 1.067 | 0.9678 -> 0.9473 | spoon 1.27/13.8 |
| 8 | F9_ep08_b4 | 0.339 -> 0.607 | 0.9981 -> 0.9834 | cup 0.65/2.5 |
| 1 | E9_ep01_b3 | 0.612 -> 0.871 | 0.9836 -> 0.9693 | pot 0.86/3.3 |
| 14 | E9_ep14_b4 | 0.782 -> 0.962 | 0.9717 -> 0.9642 | cup 1.14/3.3 |
| 25 | D9_ep25_b4 | 1.353 -> 1.508 | 0.9127 -> 0.8978 | piece_1 1.19/12.6 |
| 30 | F9_ep30_b4 | 0.402 -> 0.547 | 0.9967 -> 0.9898 | sweeper 0.65/3.2 |
| 16 | E9_ep16_b8 | 0.687 -> 0.794 | 0.9789 -> 0.9707 | spoon 1.05/6.1 |
| 10 | X8_ep10_Blr512 | 0.528 -> 0.624 | 0.9909 -> 0.9825 | cup 0.74/2.8 |
| 15 | E9_ep15_b6 | 0.886 -> 0.954 | 0.9633 -> 0.9584 | spoon 1.05/7.4 |

**Re-retargeting:**
- All 20 episodes would need re-retargeting onto target_final_v12b_g15 to collect the gain; every pose is shifted about 0.9 frame.
- Priority by MPPE increase: 20, 5, 35, 37, 32, 38, 28, 36, 33, 6, 19, 3, 8, 1; then 14, 25, 30, 16, 10, 15 (< 0.2).
- No mesh changed, so no mesh-swapped components are needed.
- Expected official effect, from the public analogue (reconstruction vs GT): MPPE about -0.1 cm and AUC about +0.012 per episode. This holds only if the hidden GT is on the b/c clock as in public. It is the same capture rig, and the eval videos show the same 0.9-frame a-vs-b/c lag.

## 5. Next (not done)
1. **Re-track with time-consistent inputs.** FoundationPose sees RGB at t and depth at t+0.9. It could be fed depth[t-1] (0.1-frame mismatch), or the poses resampled at the b/c clock with flow-warped RGB. Moving-frame error is 1.4 cm vs 0.9 cm static. This needs the whole fuse/rescue/spoon chain re-run on eval.
2. **Relative placement at frame 0 (d).** Joint multi-object f0 refinement with both objects' silhouettes and depth on a consistent clock.
3. **Lid.** Circularise the disc of our lid mesh, then do a handle-only spin search.

## Files
- code/: kitx.py, anatomy.py, tshift.py, tshift_win.py, tshift_lib.py, camsync.py, camsync_win.py, camsync_fine.py, sync_profile.py, retime.py, camera_sync.py, mk_cam_export.py, build_bundle.py, meshfit.py, mk_mf_export.py, meshdiag.py, liddiag.py, oracle_split.py, smooth_scan.py, depthshift_scan.py, depthclock.py, cmp.py, qccmp.py, gtframe.py, cmd.sbatch, py.sh
- out/: sync_delta.json, sync_delta_t.json (+_summary), camsync*/syncfine* json, anat_*/pf_*, tshift*, mf_*/ (rejected meshes), public_r8base_vs_final.txt, eval_v27_vs_targets*.txt, camera_eval_diff.txt
- cam/: the G15 camera paths for public and eval. Public exports: g15_rt, g15_rt2, g15_cam, **g15_camrt (final)**, g15_rt_mfs, g15_rt_mff (rejected)
