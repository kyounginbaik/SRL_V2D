# recon_v14 (2026-10-22): metric-mesh tracking applied to the 20 evaluation episodes

Recipe (validated on public GT, see public_gt_check_20261020/postprocess_results.md): stereo FoundationPose with the METRIC submission
mesh -> metric stereo/silhouette polish -> light smoothing.

Which eval objects were affected: every object whose final track was initialised by stereo FoundationPose used a mesh 1.26-1.73x too large
(all spoons in 3/5/6/32/33/35/36/37, the sweeper in 19, blue cups in 32/33). ComPose-initialised objects (pots, lid, dust pans, cups in
8/10, and the seven Compose episodes 14/15/16/20/25/28/30) already used metric meshes (ratio 0.86-1.05) and were not rerun.

Rerun: fp_metric/episode_X/poses_<obj> (Slurm 50281-50291), tracks built by code/build_v14.py (candidates kept as
episode_X/object_poses_v14_candidates.npz, per-object report.json), image-evidence comparison vs the current finals in
episode_X/evidence_compare.json (code/evidence_compare.py). Selection: replace only when v14 wins on >= 65% of frames.

| ep | object | v14 wins frames | current IoU / depth | v14 IoU / depth | decision |
|---|---|---|---|---|---|
| 3 | wooden_spoon | 30% | 0.84 / 0.23 cm | 0.82 / 0.25 cm | keep current |
| 5 | wooden_spoon | 20% | 0.84 / 0.24 | 0.76 / 0.30 | keep |
| 6 | wooden_spoon | 24% | 0.85 / 0.23 | 0.83 / 0.25 | keep |
| 19 | mini_sweeper | 54% | 0.83 / 0.22 | 0.83 / 0.24 | keep (tie) |
| 32 | blue_cup | 33% | 0.92 / 0.23 | 0.92 / 0.29 | keep |
| 32 | wooden_spoon | 29% | 0.79 / 0.25 | 0.77 / 0.28 | keep |
| 33 | blue_cup | 77% | 0.90 / 0.24 | 0.92 / 0.26 | **v14** |
| 33 | wooden_spoon | 30% | 0.80 / 0.25 | 0.57 / 0.34 | keep |
| 35 | wooden_spoon | 50% | 0.82 / 0.24 | 0.82 / 0.25 | keep (tie) |
| 36 | wooden_spoon | 19% | 0.81 / 0.27 | 0.53 / 0.41 | keep (v14 is 178 deg off the marker-verified corrected roll in 12.6-14.65 s) |
| 37 | wooden_spoon | 80% | 0.82 / 0.24 | 0.84 / 0.26 | **v14** |

Why most spoons keep the current track: those finals were already refined against this same evidence over several stages (metric ICP,
full-frame polish, roll/face fixes, ep33 stereo roll check, ep36 marker check), so a fresh metric FP + polish rarely beats them; no
public spoon episode exists to validate further.

episode_X/object_poses.npz in this tree = current best track for all 20 episodes (same format/frames/names), with the manual corrections
kept: ep30 sweeper half-turn corrected track (rackscan_20261021/ep30_v8), ep36 spoon roll correction (refinement_20261021_ep36/
source_corrected); ep1 lid-on pot seating was a scoring-reference correction only and the recon track is unchanged. selection.json lists
the base source per episode. The recipe does not reproduce the ep30/ep36 corrections (those were gradual half-turn twins that per-frame
methods cannot see); the evidence selector correctly rejects the ep36 v14 candidate.

Material changes (> 1 cm or > 10 deg mean on an object):
| ep | object | centre change mean / max | axis-tilt change mean (cup) or rotation change mean | re-retarget? |
|---|---|---|---|---|
| 37 | wooden_spoon | 1.03 / 5.43 cm | 6.5 deg | yes (marginal, 1.03 cm) |
| 33 | blue_cup | 0.78 / 4.12 cm | 2.8 deg tilt (axial spin unscored) | optional (below threshold) |
No other eval episode changes.
