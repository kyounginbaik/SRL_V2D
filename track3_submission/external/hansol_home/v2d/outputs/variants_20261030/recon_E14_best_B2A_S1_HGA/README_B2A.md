# recon_E14_best_B2A

Combined bundle for retargeting = **B14 B2** (family consensus, `B14_family_consensus`) **+ A14 mild** (guarded Whittaker smoother, g8 settings,
applied on top of the B2 tracks) in all eval episodes except 32, **+ L3b ep32** (`recon_E14_ep32_annot_roll/episode_000032`, copied whole, not smoothed).
Format = `selected20_final_v14_g17` (G15 clock, v14 meshes; G18 spoon mesh is applied at retarget as in v29). Not submitted; no GT read for eval.

## What changed vs v14_g17
- `object_poses.npz` of all 19 eval episodes except 32: world_from_object and camera_from_object (same body-frame change on both); every other npz key unchanged.
  In the 14 episodes B2 changed (1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 28, 30, 38) the input is B2's track; in 25, 33, 35, 36, 37 it is v14's
  (identical to A14_smooth_mild there, checked byte-for-byte).
- Frame 0 equals the B2 input. B2 itself moves frame 0 in its 14 episodes, so frame 0 there differs from v14 (it equals v14 in 25/33/35/36/37).
- `episode_000032/`: the L3b directory (object_poses.npz sha256 9e0b5470..., plus its `annot_roll.json`); see that bundle's README.
- Added: `manifest.json`, this `README.md`, B14's own `README_base_B2.md` / `manifest_base_B2.json`. 208 of 231 files are byte-identical to v14_g17;
  meshes, cameras, intrinsics, prepared/selection json are untouched.

## Public 20 (10 public + 10 held-out, unchanged `score_bundle.py --metrics`, same rule applied on top of B2's public export `v14_B2`)

| export | MP-SR | SP-SR | pos cm | ori deg | MPPE | AUC | public pos / MPPE | held-out pos / MPPE |
|---|---|---|---|---|---|---|---|---|
| g15_camrt (v14) | 20/20 | 20/20 | 0.999 | 4.95 | 1.064 | 0.9437 | 1.072 / 1.093 | 0.921 / 1.035 |
| A14 mild | 20/20 | 20/20 | 0.984 | 4.87 | 1.048 | 0.9448 | 1.054 / 1.074 | 0.909 / 1.021 |
| v14_B2 | 20/20 | 20/20 | 0.989 | 4.70 | 1.041 | 0.9452 | 1.061 / 1.061 | 0.912 / 1.022 |
| **B2A (this)** | 20/20 | 20/20 | **0.976** | **4.65** | **1.027** | **0.9459** | 1.045 / 1.045 | 0.902 / 1.010 |

Better than each reference on every aggregate and on both splits. Objects worse by > 0.05 cm: vs v14_B2 none (max +0.03 cm / +0.08 deg);
vs g15_camrt and vs A14 mild one, public ep18 mini_dust_pan 1.81 -> 2.09 cm (ori 7.2 -> 6.7 deg), which comes from B2 (v14_B2 has the same).
Re-scoring v14_B2 ep11 in this mirror reproduced B's stored score exactly.

## Eval QC (GT-free, clock-consistent: IoU / ball on the RGB clock = track shifted by -delta, |depth| on the stereo clock, 34 objects, all frames)

| | IoU | abs depth mm | ball px (mean) | frames IoU drop > 0.1 | steps > 15 deg | > 30 deg | centre > 3 cm | ang. acc > 20 |
|---|---|---|---|---|---|---|---|---|
| v14_g17 | 0.7588 | 7.91 | 24.66 | - | 125 | 8 | 84 | 41 |
| B2 | 0.7596 | 7.99 | 24.66 | 1 vs v14 | 120 | 8 | 83 | 36 |
| A14 mild | 0.760 | 7.67 | 24.42 | 0 vs v14 | 69 | 3 | 41 | 8 |
| **B2A** | 0.7601 | 7.76 | 24.42 | 0 vs B2, 2 vs v14 | 69 | 3 | 43 | 8 |

- The 2 frames vs v14 are ep19 mini_sweeper f52 (0.735 -> 0.608) and f79 (0.685 -> 0.578); f79 is B2's own drop (B2 0.535), and at f52 B2 is already >= 0.077 below v14 (the guard allows <= 0.05 vs its B2 input).
- Depth vs v14 is worse in some B2 objects (ep10 beige_cup 4.32 -> 5.04 mm, ep14/28/30 dish rack +0.2-0.4 mm) - B2's own change; the smoother lowers depth error vs B2 on most objects.
- Remaining > 30 deg steps are kept persistent jumps (ep19 sweeper, ep36 spoon hand-over). Per-object numbers: `manifest.json` -> eval_qc_clock_consistent.

## Limits
- No robot rollout for this bundle; ep32 needs the re-retarget noted in the L3b README.
- No public spoon clip, so spoon changes are checked GT-free only.
- `bundle_manifest.json` is v14's (byte-identical); its step fields describe v14 poses.

Code: `/home/hansol/v2d/outputs/variants_20261030/A_work/code/` (guard_clip14.py `--src-export v14_B2`, gen_array.sbatch, score_array.sbatch, compare14.py, clockqc_b2a.py, build_b2a.py).
