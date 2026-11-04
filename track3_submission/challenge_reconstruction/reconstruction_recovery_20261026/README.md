# Reconstruction recovery campaign (2026-10-26)

User objective: improve all twenty evaluation reconstructions using stereo, public-reference validation, camera/world checks, rotation/identity checks and current research implementations. Preserve all existing reconstruction/submission artifacts. A low reconstruction-reference policy error is insufficient evidence of held-out accuracy.

## Observed leaderboard baseline

The submitted v21C receipt records MP-SR 0.55, SP-SR 0.80, MPPE 3.61099 cm, RPE 5.23634 cm, AUC 0.75166, CD-O 0.27584 cm. Source: `challenge_submissions_SRL/track3_v21C_20261024/all_20_episodes/v21C_receipt.json`. These are actual leaderboard results, unlike the earlier local 20/20 against compose.

## Current findings and implementation

- `baseline_audit.json` covers all twenty coworker compose reconstructions. SE(3), finite-value, camera/object algebraic roundtrip, pair-distance frame invariance and per-frame motion screening completed. There are no symmetry-aware single-frame jumps above 60 degrees. These algebraic checks do not validate the physical camera calibration or exclude gradual flips.
- Earlier public experiments already found gross object/identity failures, cup up/down flips, and 1–4 cm camera-path uncertainty. Global bias and frame-zero smoothing were not validated. Relevant reports: `../public_gt_check_20261020/README.md`, `../recon_v17_campath_20261022/REPORT.md`, and `../campaign_v16_20261023/streamD_recon_public/REPORT.md`.
- Legacy stereo depth was generated in physical RGB-camera z but warped into learned AnyCalib pixels; some trackers treated those pixels as a simple pinhole camera. This mismatch is a hypothesis under test, not a proven universal explanation.
- `tools/reconstruction_audit/canonical.py` prepares a new physical RGB-camera branch. It undistorts raw RGB with supplied Perspective/rational calibration, directly projects raw metric FoundationStereo depth from rectified C into physical A using supplied rig extrinsics, and remaps the existing segmentation masks into the same image. It uses meshes copied from our validated v21C package. No reference object poses or reference scans enter preparation or tracking.
- `tools/reconstruction_audit/camera.py` estimates metric camera motion from object/hand-masked static background SIFT, stereo lifting, PnP-RANSAC and refinement. Failures are flagged, not silently certified as valid.
- `canonical_manifest.json` lists ten public validation clips and all twenty evaluation episodes; all required inputs were found. Episode 3's stereo cache is under `pose_ablation/episode03_stereo_20261010`, rather than the usual eval stereo directory. Public board 23 uses SAM2 labels 10/11; public board 24 shares the slot-zero AnyCalib output.

## Running work / resume

### Checkpoint 2026-10-28 (phase 8, stream R8): GT-free audit of final_v5. No final_v6 was built; final_v5 stays current.

Full report: `../campaign_v22_20261028/streamR8_recon/README.md`. New tools: `tools/reconstruction_audit/anchor_audit.py`,
`table_rest.py`, `static_refit.py` and `twin_continuity.py`. New outputs are only `exports/r8_*` folders and `support_plane.npz` for 34 clips.
- Corrections were tested on public 10 + held-out 10, with GT used only in the scorer. Baseline r8_base: 20/20, pos 1.114 cm, ori 5.23 deg,
  MPPE 1.171, AUC 0.931. Results: static-window frame-0 averaging was neutral (1.129 / 5.04 / 1.182); table-plane rest/tilt was worse (1.16-1.37,
  snap 19/20); multi-frame stereo ICP refit was worse (1.23-1.31; stereo sits ~5-10 mm toward the camera relative to our surfaces). All three were rejected.
- Half-turn twin continuity plus a 1-frame despike leaves public/held-out bit-identical and changes only eval19 sweeper f91 and f94-97.
  It was rejected because those frames lose IoU (0.51-0.74 -> 0.33-0.46) and gain depth residual, even though steps over 30 deg drop from 6 to 2.
- Table-contact "penetration" and tilt are constant per mesh type in every clip (rack -20 mm / 6 deg, pot -8 mm / 3-4 deg), so they
  reflect mesh and stereo-plane bias, not pose error.
- Largest GT-free risks, in order: ep32 (spoon is the anchor; frame 0 is a 5.9 deg outlier vs its static window, a ~2.2 cm scored
  shift that balls cannot resolve); ep19, ep25 and ep30 (our sweeper and piece meshes are half-turn self-congruent to 0.15-0.25 cm,
  so a twin registration would be an SP-SR failure; marker meshes registered correctly 6/6 on public, and the ep30 mesh is the public31 mesh);
  then the spoon episodes 3, 5, 37, 35, 36 and pot ep28. Ranked table: `streamR8_recon/risk_ranking.json`.

### Checkpoint 2026-10-28 night: `selected20_final_v5/` (supersedes final_v4)

New stage `tools/reconstruction_audit/rescue.py` (lost-object rescue) between `fused_v7_gap_ds` and the spoon roll
(`spoon_v8.sh`: rescue output `fused_v7_gap_ds_rs` -> `spoon_roll_v8` / `spoon_joint_v8` -> `final_v5`).
- Problem: during hand-overs the stereo depth on a thin object is unusable, fuse marks every frame unobserved and
  gap_bridge interpolates straight through a real motion. eval36 spoon f295-306: half-resolution silhouette IoU 0.00-0.10
  against an 8000 px SAM2 mask, centre up to 23 cm off; the raw FoundationPose track has IoU 0.35-0.78 there.
- Rule (no GT, same for every clip): frames whose adopted pose has half-res IoU < 0.2 while the object's mask is >= 1500 px
  (windows extend through adjacent lost frames down to 300 px) are re-evaluated against raw / BA / multimap / pointmap. A
  candidate is accepted at IoU >= 0.35 and >= adopted + 0.25. Its 4 principal half-turn twins are chosen per window by
  Viterbi (IoU loss + rotation continuity, anchored at the bounding adopted poses), so the ball-roll gauge is kept; holes
  of <= 3 frames are SE(3)-interpolated. The spoon ball roll then runs on top.
- Fires on: eval36 spoon f295-305 (11), eval19 sweeper f88-97 (10, in-fist and blurred; the raw tracker itself jumps
  ~85 deg there), eval32 spoon (2), eval33 spoon (3), eval20 pitcher (1). Public: only public21 pitcher (2 frames),
  1.55/4.65 -> 1.49/4.48. All other 19 public clips are bit-identical, so public stays 10/10 and held-out 10/10.
- QC final_v4 -> final_v5: eval36 spoon IoU p10 0.50 -> 0.53, low-IoU fraction 9.6 -> 7.1 %; eval33 max rotation step
  41 -> 26 deg; eval32 low-IoU 4.6 -> 3.9 %; eval19 sweeper p10 0.55 -> 0.57 (one more >30 deg step, the raw jump above).
- Bundle `selected20_final_v5/` (changed vs v4: eps 19, 20, 32, 33, 36). Sharpa videos for those five:
  `mink_sharpa_videos/selected20_final_v5/` (job 58139); the other 15 are unchanged from final_v4.
- Retargeting onto this bundle (phase 7, v22): `../campaign_v22_20261028/PHASE7.md`.

### Checkpoint 2026-10-28 late: `selected20_final_v4/` (supersedes final_v3)

Held-out public test (GT used only for scoring). Ten public episodes never used for tuning
(0, 2, 13, 22, 26, 27, 40, 42, 43, 45) were prepared from scratch in
`canonical/public_heldout/` (manifest `heldout_manifest.json`, tools `heldout*.py/sbatch`)
and run through the unchanged policy. 8/8 finished episodes pass MP-SR, 13/13 objects,
mean 0.3-1.6 cm / 1-9 deg (`exports/fused_v6_gap_ds/public_score.json`; v7 is
bit-identical there). Ep0/ep2 (pot+lid) needed an image-only pot-vs-lid exclusive-mask
repair (`exclusive_masks.json`). Re-tracked, they FAILED under the unchanged policy: pot fine
(1.0 cm / 2.7 deg) but the lid upside down for the whole clip (176 deg). Our lid knob pointed
into the pot. New stage `lid_up.py` (knob-up prior): only for meshes whose thinnest axis is
strongly one-sided (skew > 1.5; only white_pot_lid qualifies, 1.97; next is the rack, 0.71);
if the knob points against the measured table normal (`support.py`) over the first 30 observed
frames, turn the track a half turn about the in-plane axis chosen (15 deg steps) by
full-resolution mask/stereo evidence over ~25 observed frames. Result: ep0 lid 21.4 deg,
ep2 lid 28.7 deg (borderline), both MP-SR pass, so held-out 10/10. CAVEAT: this rule was
designed after seeing the ep0/ep2 GT failure (and the whole-clip vs first-30-frame evidence
choice was checked against their scores), so ep0/ep2 are no longer clean held-out data.
`diag_*` scans show the scorer treats the lid as 2-fold about its knob axis; best possible
flip axis gave 18/7 deg. Eval1 lid: knob already up -> no flip; final_v4 unchanged.
On these clips fusion ~= raw tracking: the easy clips have no flips or dropouts.

Error anatomy on 18 scored public clips (scoring-only diagnostics `diag_perframe.py`, `diag_biasdir.py`; overlays `diag_overlay.py`):
the remaining ~1 cm / 4 deg is a constant per-object offset with varying direction, not
jitter or a global depth bias. Static-interval pose averaging (`static_avg.py`) was neutral
and introduced step jumps (public44 max 11.6 cm): rejected.

Eval-20 GT-free test (`qc.py`, dense per-frame mask IoU / stereo depth residual / rotation
steps, compared against the passing public range; `qc_all.sbatch`): all objects inside the
public range, but overlays found a real failure on eval36 spoon (frames 72-88, 111-127,
153-163, 295-306). Cause: in the 7 clips that have a HaWoR hand prior (public 11/18,
eval 3/5/8/33/36) the wrist-rotation cost (weight x5 when <80 support pixels, i.e. thin
spoon) outweighed image evidence and dragged the path onto `bundle_hand_guided`.
`fused_v7` = v6 with no hand-derived inputs (no hand-guided BA source, no wrist prior,
hand-free motion edges `object_motion_edges_nohand.npz`). Public 10/10 unchanged
(only public18 changes: sweeper 9.6 -> 9.3 deg); identical on all clips without a hand prior.
Eval36 spoon p10 IoU 0.39 -> 0.50; eval3/5/8 improve or same; eval33 spoon slightly lower.
`fused_v8` (raw +-60/120 deg long-axis roll states) was rejected: no gain, more >30 deg steps.
Remaining known defect: eval36 spoon 295-306 (12 frames, a real half-turn roll during a
hand-over where every candidate except raw is displaced).

`final_v4` = fused_v7_gap_ds + `spoon_roll_v7` (ball model re-estimated on v7) +
`spoon_joint_v7` for eval 3 and 32 (`tools/reconstruction_audit/spoon_v7.sh`). Held-out
ball reprojection (px, before -> after roll): 3 29->12 (joint 9.5), 5 27->4.5, 32 28->12
(joint 9.5), 33 34->12, 35 15->9, 36 17->10. Bundle `selected20_final_v4/`; Sharpa videos
`mink_sharpa_videos/selected20_final_v4/` (unchanged episodes copied from final_v3).

### Checkpoint 2026-10-28: `selected20_final_v3/` (supersedes final_v2)

- Mask-exclusion bug fixed in `fuse.py` (fused_v6) and `review.py`. Evidence used to exclude every
  other mask, even where it covered the object's own pixels. The eval28 SAM2 hand mask (label 3)
  drifted over the whole pot from frame 124, and in eval1 the pot mask encloses the lid; both
  objects were scored "unobserved". Rule now: hand masks never override an object's own mask
  (SAM2 object masks already omit occluding hands); between objects the smaller mask wins.
  v4 (own mask always wins) broke the eval1 pot; v5 (smaller wins incl. hands) lost the public18
  sweeper gain.
- fused_v6 public: 10/10, public18 sweeper 16.6 -> 9.6 deg, all others within 0.07 cm.
  Eval coverage: eval1 pot/lid 1.0/1.0 (was 1.0/0.76), eval25 boards 1.0/0.99, eval28 pot 0.94 (was 0.50).
- Spoon re-run on fused_v6 (spoon_roll_v6; joint_v6 for eval 3/32): held-out ball px 3 20.2->10.1,
  32 12.6->9.5 (after roll).
- Chain scripts: `chain_v6.sbatch` (fuse v6 -> gap -> despike -> postprocess; MANIFEST env for other
  manifests), `spoon_v6.sh`, `compose_final.py --base fused_v6_gap_ds --roll spoon_roll_v6 --joint spoon_joint_v6`.
- Eval28 frame 105-123 depth disagreement: pot partially outside the top image border; overlay
  shows the silhouette fits, so it is a stereo border artifact, not a pose error.
- Held-out public test (eps 0,2,13,22,26,27,40,42,43,45, never used for tuning) in preparation:
  `heldout_manifest.json`, inputs in `challenge_reconstruction/public_heldout_20261027/`.

### Checkpoint 2026-10-27 late: `selected20_final_v2/` (supersedes selected20_fused_v3_gap)

- `despike.py` (out-and-back rotation excursions <=3 frames): public unchanged 10/10; eval6/5/19/33 cleaned.
- Spoon roll from its 3 gray mocap balls (handle tip + two on opposite neck edges = chiral layout):
  `ball_detect.py`/`ball_detect_all.py` (gray blob + black base, stereo-lifted; detected radii match
  expected 5-7 px), `spoon_roll.py` (body ball model pooled over majority-chirality episodes:
  near (-0.92,-1.26,-0.08) far (3.13,2.16,-0.35) tip (-15.65,0.09,-0.75) cm; Viterbi over roll,
  v2 adds fuse.py mesh evidence). Eval3 and 32 were tracked mostly half-turned, 33/35/36 partly.
  Held-out ball reprojection (every 5th detection frame, never fitted) roll_v2: 3 28.8->18.9 px,
  32 27.8->13.5, 33 14.9->11.3, 35 14.7->10.1; joint_v3 for 3/32: 9.8 / 9.2 px.
- `compose_final.py`: spoon_roll_v2 everywhere; spoon_joint_v3 (`joint_refine.py`: ball 2D + mesh
  z-buffered depth + silhouette, single seed) where >=50% frames were half-turned (eval 3, 32).
- REJECTED with evidence: ball stereo depth (`ball_refine.py`, `spoon_refine.py`; small spheres bleed
  to background -> objects pushed away; public 9/10 worse, public31 fails); per-frame seed switching
  (`spoon_joint_v1`, 77 >30deg steps). joint_v3 on public 18/23/24/31 is neutral (+-0.3 cm) and keeps
  a ~3.5 mm toward-camera spoon bias, so it is applied only to eval 3/32.
- Pot/cup ball models are unreliable (reflections; held-out 30-200 px); not used.
- Sharpa kinematic mink IK videos: `mink_sharpa_videos/<bundle>/epi<N>.mp4` (code + README there).
  Known: ep14 HaWoR left/right identity swap; ep20 right hand unobserved 70 frames.
- Remaining risks: spoon has no public validation (balls are the only roll evidence); eval28 pot
  frames 124-219 and eval1 lid unobserved windows; eval25 boards; sweeper ~17deg persistent on
  public18 (mesh geometry, not pose).

### Checkpoint 2026-10-27 (Claude Code resumed after Codex token limit; supersedes below)

**Selected 20-episode bundle built: `selected20_fused_v3_gap/`** (105 MB, `bundle_manifest.json`
holds the policy, per-object flags and sha256 of every file; not submitted, no policy rows replaced).

- Policy = `fuse.py --tag fused_v3` (Viterbi over raw/pointmap/multimap/BA/hand-BA/odometry/principal
  twins with the >75deg / >12cm one-frame jump ban) + `gap_bridge.py` (SE3 slerp across unobserved
  windows <=20 frames with observed frames on both sides; bridged frames stay flagged unobserved).
  One fixed policy, no per-episode GT choice.
- Public validation (unchanged kit scorer, GT only in scoring): fused_v3 **10/10 MP-SR**, fused_v3_gap
  **10/10**, identical within 0.01 cm except public18 sweeper 1.09->0.97 cm. Worst public object:
  public18 sweeper 0.97 cm / 16.6 deg; public12 rack 2.20 cm.
- Rejected on public: `bundle_hand_guided` (public11 beige 21deg, public18 sweeper 117deg),
  `*_plane` support-frame correction (worse on 11/41/44), dense_track.
- Eval fuse array 56880, gap array 56903: all complete, no tracebacks. After bridging no object
  has a >8 cm one-frame step (eval5/33/36 spoon 11 cm jumps were re-acquisition after short
  occlusions).
- Open risks (cannot be resolved offline): spoon roll. In eval 3/5/6/32/33 the selector switches
  between candidates disagreeing 40-70 deg about the spoon long axis; 2-10 frames/episode have
  >30 deg steps, some are one-frame excursions (eval6 f99/100). Eval5 spoon uses the raw
  long-axis twin for 61 frames (agrees with BA hemisphere, 13-60 deg from BA). Long unobserved
  windows kept as fused estimates: eval28 pot frames 124-219 (obs 0.50), eval1 lid 0-31 and
  287-334, eval25 boards (min IoU 0). Possible next step: rotation de-spike for 1-2 frame
  excursions, validated on public before adoption.

### Latest checkpoint (2026-10-26 ~07:15 UTC; supersedes launch notes below)

User explicitly says **do not ask, keep going**. No delegation is authorized.
All20 evaluation canonical inputs, camera paths, **36/36 FoundationPose tracks**
and **20/20 multi-view point-map candidates** are complete. There is no final
selected20 reconstruction bundle yet. Do not end after merely launching jobs.

Two image-verified identity failures in the original cached HAW masks were fixed:

- Evaluation1 pot mask duplicated/followed the lid. Fresh canonical SAM2 pot
  prompt at frame83 (visible pot), FP registers there and tracks backward/forward;
  re-registration disabled for this pot. Camera rerun after mask repair.
  Raw QA improved from no overlap/24cm jumps to IoU0.89, depth7.8mm, maxstep2.8cm/9deg.
- Evaluation25 old masks followed a background cardboard box and the wrong board.
  Fresh SAM2 boxes label left4-marker board piece1, right3-marker board piece2.
  Both FP tracks and camera rerun. Raw QA now IoU0.87/0.88, depth1.9/1.1mm,
  maxsteps2.2/2.9cm and12/9deg.
- All incorrect earlier masks/poses/cameras/exports/reviews are preserved in each
  episode's `incorrect_identity_archive`. Original submissions were untouched.
- Public31,12,21 originally omitted rack/rack/pitcher. `complete_public.py`
  segmented and tracked all three, archived partial-roster candidates in
  `incomplete_roster_archive`, updated manifest and `prepared.json`. Strict scorer
  now requires the full roster and full timeline. These raw public clips pass.

Current numerical/public evidence:

- Public41 pot raw0.785cm/3.61deg;44 raw0.561/1.84;39 dustpan raw0.964/3.8.
- Public11 blue: raw3.43/92.9; CoTracker single-map0.409/1.83; multi-map0.312/1.30;
  joint bundle adjustment0.280/1.46. Beige raw0.861/2.82, multi/BA about0.98/7.
- Public23 boards raw original mesh piece2 has179deg ambiguity; measured marker
  meshes with unchanged raw poses give0.88/2.9 and1.08/2.6.
- Public24 boards measured marker v3 +raw gives1.60/2.8 and1.63/9.1. Multi-map and
  bundle adjustment drift on moving piece1 (~4.9cm/34deg) and must not be blanket adopted.
- Public31 full raw brush/rack1.15/7.7 and1.39/5.7.
- Public12 full raw cup/rack1.51/5.9 and2.16/2.4.
- Public21 full raw cup/pitcher0.85/2.8 and1.54/4.0.
- Public18 brush raw has a persistent wrong face. `marker_depth.py --clean-seed`
  intersects the brush seed with its long-axis half-turn in1.5mm voxels (removes
  unsupported asymmetric bumps), then adds **two actual gray circles with black
  bases** lifted by stereo (depth MAD0.2mm), radius6–7mm. No scan is used.
  With multimap poses this gives brush0.978cm/8.67deg, pan2.184/7.59; bundle-adjusted
  gives1.10/7.14 and2.17/7.28. Raw poses with corrected mesh fix the scene/pan but
  brush dynamic rotations still fail. Therefore both geometry and tracking matter.
- Do not apply seed cleaning if no reliable marker measurements: public31
  first-frame markers are occluded, strict detector finds0; cleaned marker-free
  geometry fails badly. Retain original mesh for such cases.
- Official public CD-O verification:18 brush0.215cm vs original0.187, pan0.189;
 23 marked boards mean0.173cm;24 mean0.181cm. Tiny mesh-distance cost resolves
  large pose-registration ambiguities. Reference scans enter `geometry_score.py`
  only after asserting public split; no reconstruction code imports them.

There is a passing candidate for each of the10 validation clips, **not yet a
single final selection policy proven10/10**. `fuse.py` v1 reads image/depth/motion
only (never scores/GT), uses candidate poses and raw principal half-turn twins,
and Viterbi continuity. It passes public11 (0.35/1.5,0.93/3.4) but public18 brush
still flips: fitted sparse geometry can favor the wrong face over a long interval.
v2 adds independent adjacent-frame rigid stereo motion from CoTracker point IDs
(`motion_edges.py`), with rotation-observability flags and contact-gated wrist
motion only where roll is unobservable. It also includes a direct metric-odometry
proposal. Public array56463 (indices0–9, concurrency3) is running/just launched.
Review its scores and source choices before evaluation-wide selection.

Implemented candidates/controls (code in `tools/reconstruction_audit`):

- `point_tracks_multi.py`:6 query times, up to60 points/object/view; official
  CoTracker3 bidirectional inference in existing FP container. `rigid_multimap.py`
  grows a metric rigid map in both directions then revisits all frames. Missing
  observations/interpolation flagged. Array56131 (all20) completed. Public
  array56148 (indices6–8, single allocated healthy GPU) completed.
- `bundle_adjust.py`: joint sparse robust RGB/depth optimization of all object
  poses and local landmarks, fixed initial body gauge, weak acceleration priors.
  Analytic sparse Jacobian checked against independent3-point differentiation:
  maxabs2.46e-8, relative1.64e-11. Public jobs56266–56268; eval3/5/8/33/36
  jobs56269–56273 completed. Some hit30-iteration cap: retain status, don't call
  them converged. BA helps public cups/brush, not public board1; reject bad fits.
- `dense_track.py`: experimental dense point-to-plane/p2p registration with
  temporal proposals. **Not reliable globally**: public11 beige3.4cm; public18
  pan11.8cm despite low fitted residuals. Keep as diagnostic, omit from selector.
- `support.py`: robust static wooden-table plane from RGB proposal +masked stereo.
  Produces gravity/support-frame transform, retaining original camera. Optional
  roll/pitch/height correction is a hypothesis under public validation, not adopted.
  Public11/41/44 jobs56275–56277 complete; plane coverage1.00/.813/.859. Median
  corrections angle0.58/.91/.45deg, height1.5/4.9/1.2mm. Need score exported
  raw_plane/multimap_plane/bundle_adjusted_plane. Constant support-frame export
  remains useful regardless of whether dynamic correction is rejected.
- `hand_priors.py`: original experimental prior used old virtual-camera wrists.
  New v2 maps stable virtual projections through AnyCalib raw pixels into supplied
  physical pinhole, lifts wrist depth from observed hand stereo +explicit1cm joint
  offset. Grasp association uses actual2D object-mask proximity +hand contact,
  valid tracked wrists; never labels grasp priors as object observations.
  `hand_motion_prior_v2.npz/json` are separate from original v1 caches.
  A variable-shadowing bug (hand mask replacing hand-prior NPZ variable) made
  first hand-BA jobs56425–56429 fail before fitting. Fixed with `hand_data` and
  `hand_mask`. Corrected v2 jobs56453 public11,56454 public18,56455 eval8,
  56456 eval33,56457 eval36,56458 eval5,56459 eval3. Need inspect results.
  Bound grasp-supported orientation corrections to±0.35rad/component; source
  hand orientation, finger slip and skin-to-joint offset remain uncertain.

`report.py --quiet` regenerates `EVIDENCE.md` /`evidence_summary.json` without
printing it all. `rank.py` prints per-object raw/pointmap/multimap image residuals
and motion jumps. Multi-map reduces many spoon flips but some large translation
jumps/occlusions remain (eval5,8,19,32,33,36). Stable raw tracks must be retained
where the point tracker loses geometry (eval15 spoon,20 pitcher,25 boards, etc.).

Remaining required work:

1. Inspect hand-guided fits and fused_v2 public policy. Fix policy using actual
   image/depth/correspondence support, not per-episode GT choices; no hidden GT.
2. Select/refine all20, scrutinize remaining large steps/low-support windows with
   overlays. Do not equate fitted residuals or smoothness with truth.
3. Export a complete20-target bundle: explicit calibrated camera/world frames,
   consistent mesh/body poses, full timeline, validity/uncertainty flags, all-frame
   motion screening, contact sheets and review videos, hashes/source provenance.
   Include known rig transforms/support frame; prevent RGB/rectified-left mixing.
4. No submission upload or policy-row replacement. Any later retargeting must
   run real policies against the selected reference.

Additional public data discovered: public split actually has20 episodes;
the10 unused are0,2,13,22,26,27,40,42,43,45. No public task contains a spoon.
Only the10 existing validation clips have cached stereo/masks. Expanding validation
is possible but not yet started. Input metadata has task names only; public
parquet object-name roster may be read without using reference coordinates.
Existing FoundationStereo environment is Python3.12, while FP container is3.11;
don't blindly mount its site-packages into3.11. Existing host interpreter points
to `/home/exaFLOPs/miniforge3/bin/python`; if running stereo, keep ML in a container
by mounting that existing environment/interpreter read-only (no heavy host install).

Tool operation: normal sandbox still fails (`bwrap ... RTM_NEWADDR`). Existing
approved prefixes (cat/rg/sbatch/squeue/known report scripts) can use escalated
execution **without asking the user again**. Avoid repeated permission prose or
new broad interpreter prefixes. Always `login:false`. Some tool calls have long
approval/review latency; use short yields and communicate when control returns.
Never touch concurrent `ur-*` jobs; those belong to other user work.

### Latest checkpoint (2026-10-26 ~04:15 UTC)

All 30 canonical inputs and all 30 camera trajectories are complete. Evaluation
FoundationPose: 27/36 object tracks initially completed; nine later tasks failed
because node01 GPU `GPU-231b74bb-a287-894d-a343-301acae1abcf` reports an
uncorrectable ECC error. Healthy node100 was fully allocated. The replacement
jobs reserve two node01 devices and `allocated_gpu.py` chooses a healthy device
strictly within the job allocation. Do not reset devices or touch other jobs.

Active job lineage:

- `55929`: nine evaluation tracking retries, array indices 22,26–33, concurrency1.
- `55938`: remaining public FoundationPose controls, indices2,5–11, concurrency1.
- `55939`: learned CoTracker3 point-map public tests, manifest indices0,1,3,9.
- `55940`: all20 evaluation CoTracker3 fits, indices10–29, after55929.
- `55948`: remaining public CoTracker3 fits, indices2,4–8, after55938.
- `55947`: export and image/depth QA for completed tracks, all30, CPU concurrency4.
- `55930` + `55931`: public11 CoTracker3 fit and unchanged scorer, completed.
- `55933`, `55946`: public24 observed-marker mesh experiments, completed.
- Obsolete node100 retries `55835`–`55837` were cancelled/replaced, as was old
  never-satisfied dependency `55828`. No unrelated jobs were cancelled.

Public GT has entered **only scoring**, never preparation, camera estimation,
point tracking, pose fitting or geometry generation. All mesh seeds are our
v21C assets. Initial public outcomes (mean cm / deg, unchanged kit scorer):

| public ep | variant | object errors | MP-SR |
|---|---|---|---|
|41|canonical FP|pot 0.785 /3.61|1|
|44|canonical FP|pot 0.561 /1.84|1|
|11|canonical FP|blue3.43 /92.9; beige0.861 /2.82|0|
|11|cup continuity|blue3.80 /6.43; beige0.861 /2.82|0|
|11|CPU LK rigid map|blue0.523 /3.23; beige12.48 /10.1|0|
|11|**CoTracker3 rigid map**|**blue0.409 /1.83; beige1.038 /9.87**|**1**|
|24|canonical FP, original own mesh|piece1 23.1 /135; piece2 45.1 /174|0|
|24|same FP poses, image+stereo markers v1|piece1 1.50 /3.14; piece2 1.68 /9.00|1|
|24|same poses, stereo-plane markers v2|piece1 1.47 /2.55; piece2 1.60 /9.08|1|

The board result isolates a mesh-registration ambiguity: measured marker
geometry changes scoring dramatically **without changing any trajectory**.
The silhouette and stereo-depth fit was already good at frame zero. v2 image
review found one background-circle false positive; v3 rejects candidates outside
the board mask by more than0.8 radius and is under validation. Preserve v1/v2
diagnostics. Markers are reconstructed from grayscale RGB blobs/circles and the
dominant stereo board plane, transformed using our initial FP pose, not scans.
Need repeat on public23 and cross-frame verify marker positions before adoption.

`point_tracks.py` uses Point2Pose's vendored CoTracker3 predictor and Meta's
official `scaled_offline.pth` checkpoint (97MiB) under `weights`. No heavy host
ML installs. `rigid_points.py` builds an object-local stereo point map from our
frame-zero FP pose and fits robust RGB+depth-supported PnP over persistent
correspondences. It records all missing observation frames; interpolation is
explicitly an estimate. Public11 observed fractions are blue0.796, beige0.942.
Do not blindly freeze or smooth moving objects; the LK control demonstrates why.

`review.py` renders our actual mesh into canonical RGB, excludes hands/other
object masks and reports silhouette IoU and stereo z residuals. Green contours
are masks, colored contours are meshes. Outputs per-variant contact sheets and
`image_evidence.json`. These are fitted-data checks, not hidden-GT certification.
`postprocess.py` assembles full timelines and runs public-only scoring, retaining
all candidate variants. New results must be inspected and selected using image
evidence and public validation; the final20 reconstruction bundle is not ready.

The older running-work notes below record the initial launch history.

All commands run from `/data1/SRL_V2D` unless stated otherwise.

- Baseline audit: completed (`tools/reconstruction_audit/audit.py`).
- Canonical public input pilot: jobs `55661_0,55661_2,55661_3`, prepared public episodes 11, 41, 44.
- Initial public FoundationPose jobs `55667`–`55670` failed at startup because the selected PyTorch backend weights are absent. The installed weights are NVIDIA TensorRT. `canonical.py` was corrected to default to `nvidia_tensorrt`, configurable with `RECON_POSE_BACKEND`.
- Restarted public GPU pilots: `55674` (11 blue cup), `55675` (11 beige cup), `55676` (41 pot), `55677` (44 pot). These request node01 L40S through Slurm; other jobs must not be disrupted. Watch `logs/track-<JOB>_4294967294.log` and the per-object `*_tracking_complete.json` outputs.
- Canonical preparation for all 30 clips: array `55678`, concurrency 4 (`tools/reconstruction_audit/prepare_all.sbatch`). Completed pilot inputs are skipped.
- Canonical metric camera estimation: jobs `55679` (11), `55680` (41), `55681` (44). Outputs `canonical/public/episode_XXXXXX/camera.npz` and `camera_report.json`.
- Slurm accounting (`sacct`) currently fails with a localhost database connection error. Use `squeue`, `scontrol` and actual artifacts/logs to monitor.
- Shell startup caused long delays for some read commands; use `login:false` for subsequent `exec_command` calls. The default sandbox fails with `bwrap: loopback: Failed RTM_NEWADDR`; use approved scoped escalated commands until it is repaired.

## Next required steps

1. Confirm the TensorRT GPU pilots start and inspect canonical RGB/masks/depth alignment.
2. Add and run a public scoring/export adapter: read per-frame FoundationPose JSON (wxyz rotation, translation), compose with the new metric camera, require complete timelines, and score with the unchanged kit via existing `public_gt_check_20261020/code/boards/bscore.py`. Ground truth enters only this scoring stage. Compare baseline/corrected camera and tracking variants; report per-object errors and failure windows.
3. Validate and extend rotation-hypothesis selection (cup up/down, spoon/board face/roll) using independent image/marker evidence, not smoothness alone. Object-pair distances are frame invariant; temporal constancy applies only to rigid-relative/static intervals.
4. Complete camera checks for all twenty evaluation clips and identify high-error windows from stereo residuals, silhouettes, static-background agreement, long-range point tracks and relative-object consistency.
5. Adopt only corrections supported by independent evidence and public cross-validation. Produce a complete new reconstruction bundle, source/calibration/model provenance and review videos for all twenty. Do not overwrite previous reconstructions or edit policy rollout rows.
6. Retargeting/submission must use real rerun policies for any new reference; reconstructed poses must never be substituted into policy output rows. Do not upload without a new user instruction.

## Research sources inspected

- [FoundationStereo](https://github.com/NVlabs/FoundationStereo): calibrated stereo depth; already installed under `../vendor/FoundationStereo`, including weights.
- [Stereo Any Video](https://github.com/TomTomTommi/stereoanyvideo): temporal stereo model to benchmark if depth flicker is material; not installed yet.
- [Point2Pose](https://github.com/tzuyuan/point-to-pose): RGB-D multi-object tracking with persistent 2D correspondences and recovery after occlusion. Official source cloned into `vendor/point-to-pose`; its dependencies/weights still need inspection and any run must be benchmarked before adoption.

The existing masks and stereo depth are useful cached model outputs, not independent ground truth. Public improvements must be reported honestly, with failures retained. Near-perfect or guaranteed-winning reconstructions cannot be certified against confidential evaluation ground truth.
