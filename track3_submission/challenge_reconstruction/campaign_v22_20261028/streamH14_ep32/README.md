# Stream H14 (phase 9): ep32 spoon margin vs target_final_v12b_g15, 2026-10-30 (IN PROGRESS)

Start: H12_ep32_e1f0 (v27/v28 ep32 component): cup 1.05/3.9, spoon 1.31/22.4 vs v12b (MP 1, thin margin to 30 deg).
Rules: our meshes only (byte-identical to e1f0), poses only from build-v2 rollouts, node01 CPU <= 50 min, nothing overwritten/uploaded/committed.

## Diagnosis (code/diag.py, logs/H14-diag-65613.log)
Error is almost all twist about the spoon long axis: f20-40 24 deg, f40-60 32, f60-80 51, f80-100 60, f125-145 37, f145-165 34, f165-190 19, rest 3-8.

## Method
- code/wbeamH14.py = H13a wbeam4H13 + MONOTONE FULL-HORIZON ranking: window children are continued from the cached state to the episode end and the beam is chosen by
  whole-episode per-frame cost (unedited child = parent, so the best never gets worse). `--full-m` (0 = all), `--slack` (prune members > elite*(1+s)), `--wdeg`, `--win-from/--win-to`, roll alpha up to 3.
- code/beam.sh launcher (e1f0 actor; `--init beam_wXX.npz` to resume / start round 2), code/svst.sbatch (save -> H12 stage_b -> build-v2 TAG H14_ep32_<x> -> vs.sh), code/vs.sh (exact, archive, mesh cmp, score_any vs v12b and v13_g16 if present).

## Findings so far
- Every edit in windows 1.0-3.5 s (b1/b2/b3, ~107 children each, all rolled to the end) loses the spoon later: full cost 11k-21k vs base 1757. f20-60 is not editable this way.
- Re-acquisition CEM r1 (right wrist + squeeze, 0.4-1.2 s, full-horizon cost): no improvement after 2 iterations (median 12903 -> 34.7 vs base 5.52).
- a1-a3 (pure full-horizon beam without pruning) cancelled: beam filled with lost-spoon branches.

## Runs
b1/b2 cancelled by main and relaunched as b1L (65655) / b2L (65656) with --win-from 10 (3.5 s on); b3 (65652) and r1 (65653) continue. No candidate staged yet.
