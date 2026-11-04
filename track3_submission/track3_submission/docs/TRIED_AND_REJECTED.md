# Spoon refits after FINAL (spoon3537): used and rejected

All code is in `external/hansol/spoon3537/` (server `/data1/hansol/spoon3537`).
- `code/`: reconstruction refits and renders.
- `rt/code/`: the team A20 retarget chain, copied: `s3run.sh`, `job.sbatch`, `s3svst.sbatch`, `stage_s3.sh`, `stage_b_s3.sh`, `g18swap_s3.py`, `s3run_one.sh`, `s3score.sh`, `s3pkg.py`, `s3win.py`.
- `rt/jobs.txt`: job ids.
- `rt/packages/*/S3_PACKAGE.json`: one record per package.

## Used in the final file 9bc322af: ep36 s4

- Reconstruction, 2–10 s (`tgt_s1`):
  - `code/fit_spoon.py` and `fit_spoon.sbatch` fit the spoon pose per frame to the SAM2 spoon mask in the RGB camera (`world_from_rgb` + anycalib intrinsics), with hand pixels excluded.
  - Priors keep the fit near the old rotation and depth. Smoothing σ = 3, 0.25 s ramps, f0 unchanged.
  - Source: `target_final_v14_g17` ep36.
- Reconstruction, 12.5–17.5 s (`tgt_s4`, on top of `tgt_s1`):
  - `code/s4pos.py` and `s4fin.py`: depth-locked position fit. `s5.py` handles the f297–308 jump with a spline and slerp.
  - Roll refit: `code/spoon_annot_roll3.py` (free end), using the **agent** front/back marks `am36/agent_marks36.json` with ball weight 3 and an IoU term.
  - Check script: `code/s4check.py`.
- Retarget:
  - Starting point: the FINAL ep36 actor (`streamH13a/runs/save_ep36_b1/actor`) and table (`streamALL20K/runs/ep36/b2/final_0.npz`).
  - Beam s1a (1.75–10.5 s, `tgt_s1`) → s1d (10.5–20.25 s) → s4 beam (12.5–20.25 s, `tgt_s4`, from `s1d/beam_w03.npz`).
  - Save `rt/runs/save_tgt_s4_ep36_s4` → candidate `rt/candidates/S3_ep36_s4` (checkpoint `5d5738b7`) → `rt/code/s3pkg.py` → `rt/packages/track3_FINALc_ep36s4`.
- Leaderboard compared with 8b6fa2b9:
  - MPPE 1.40441 → **1.38958**;
  - AUC 0.92086 → **0.92269**;
  - RPE unchanged (single-object episode);
  - MP-SR expected 1.0.

## Rejected

| attempt | what | where | why not in the final |
|---|---|---|---|
| **B = s4 + ep37 refit** (`452d4a7a`) | ep37 `fit_spoon` refit + retarget s1a, combined with s4 (`code/mk_s4e37.py`, `mkB.sbatch`) | `rt/packages/track3_FINALc_B_ep36s4_ep37s1a`, `rt/candidates/S3_ep37_s1a` | Leaderboard MPPE 1.39543, worse than s4's 1.38958; AUC 0.92323; MP-SR 1.0 |
| ep37 alone | `fit_spoon` refit and retarget s1a | `rt/packages/track3_FINALc_ep37s1a_tgt_s1`, `work/ep37/report.json` | Only used inside B |
| ep35 | `fit_spoon` refit | `work/ep35/report.json` | Not packaged; FINAL ep35 (K35_H2joint_b1) kept |
| ep3 fix | retarget against `tgt_v32` in 5.75–8.75 s | `rt/candidates/S3_ep3_fix`, `rt/packages/track3_FINALc_ep3fix` | Not adopted; FINAL ep3 (A20_ep3_b2) kept |
| ep3 e3s | reconstruction refit (`ep3fit.py`: 6-DoF about the COM, accepted only if both the ball error and the stereo IoU improve) and retarget 5.25–11.25 s | `rt/candidates/S3_ep3_e3s`, `rt/packages/track3_FINALc_ep3recon_e3s`, `work/e3s/report.json` | Not adopted |
| ep36 s1d / s1w17 alone | `tgt_s1` only (2–10 s) | `rt/packages/track3_FINALc_ep36s1d_tgt_s1`, `..._ep36s1w17_tgt_s1` | Superseded by s4, which extends to 12.5–17.5 s |
| ep36 s2 | other late-window target `tgt_s2` (13.25–17.25 s) | `rt/packages/track3_FINALc_ep36s2_tgt_s2`, `code/fit_spoon_s2.sbatch` | Superseded by s4 |
| ep36 s3 | `tgt_s3` (agent-mark roll refit, `code/mk_s3.py`), beam 10.5–20.25 s | `rt/jobs.txt` (70095–70097) | Superseded by s4 (depth-locked position added) |
| ep36 s5 | `tgt_s5` (s4 + the f297–308 jump fix only, beam 14.5–16.5 s) | `rt/candidates/S3_ep36_s5`, `rt/packages/track3_FINALc_ep36s5` | Not adopted |
| ep36 s6 / s6b / s6c | s5 jump fix + roll refit with the agent marks, **except f298–312 set to edge-on [0, 1] from the user's visual judgement (a human hint)** | `code/s6*.py`, `work/s6*/marks36_s6_PROVENANCE.json` | Rejected; **not in the final** |
| A = ep36 s1d + ep37 s1a | combined package | `rt/packages/track3_FINALc_A_ep36s1d_ep37s1a` | Superseded by s4 and B |

## Camera finding: RGB vs stereo

- `s4check.py` and `ep3check.py` cross-check the per-frame fits in the canonical (stereo) camera.
- In the late ep36 window, the SAM2 + `world_from_rgb` overlay with the undistorted K **disagrees with the stereo evidence**: marker-ball distance and stereo IoU.
  - A fit that looks better on the RGB overlay can look worse in the stereo camera.
- Even so, on the leaderboard the RGB-based ep36 fix (s4) **improved** MPPE (1.40441 → 1.38958) and AUC (0.92086 → 0.92269).
  - So for this episode the RGB-camera fit was closer to the hidden truth than our stereo-side checks suggested.
  - The ep37 analogue (B) did not improve MPPE.
- Treat the stereo checks as a guide, not a veto. TODO: find which camera path the scorer's frame matches.
