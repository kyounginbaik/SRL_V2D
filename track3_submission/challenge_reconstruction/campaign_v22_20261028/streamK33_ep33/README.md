# Stream K33 (2026-11-02): episode 33 (blue_cup + wooden_spoon) - K35 start-roll method applied; DIAGNOSTIC packages

Rules: no public scan/GT mesh/GT pose (no public spoon exists); no constant fit to GT/leaderboard; poses only from build-v2 rollouts
(window-beam phase residuals on the v29 ep33 actor H13_ep33_b1, official save + unchanged gates); nothing overwritten/deleted; Slurm debug
node100, <=48 CPUs, jobs < 50 min; no upload by this stream, no commit. Code = E31 copies (paths re-pointed, svst/vs/mkref K33) + K35
build_h/imgev/k35splice adapted (build_h33.py, imgev33.py, k33splice.py) + rollrel.py, flipscan33.py, cupcheck.py, prog.py.

## Kit mechanism in ep33
`_align_registered_scene`: rotation from the first object without a symmetry axis = wooden_spoon (frame 0); translation offset from
SLOT 0 = blue_cup frame-0 position. A spoon frame-0 roll error rotates the scene about the cup -> lever 0.35 cm/deg on the spoon.

## Diagnosis (out/)
- rest2_v14.txt: spoon static f0-~52 (lift ~f53); per-frame physical rest roll on our stereo plane f0-25: -6..-16, median -10;
  f30-50 converges to ~0 (tracker roll drifts while static). End window f300-347 also unstable (-1..-19) but spoon not clearly at rest.
- rollrel_vs_v14.txt: v29/v26/Hansol sim rollouts (start = v14 f0 pose) settle by ~-11.5 deg roll (rest2 convention) on f0-25.
- imgev_v14_f0-24.txt: IoU, depth and 2D balls all prefer negative roll (-4..-8).
- cup_rest_v14.txt / cupcheck_v14.txt: cup f0 tilt ~3 deg, f0 position well constrained by mask + stereo.
- flipscan_v14.txt: held phase 0 vs 180 roll - no sustained flip evidence (f100-200 ball-poor).
- official_failing_vs_hyp.txt, large_roll_consistency.txt: which hypotheses make every officially failing variant fail.

## Hypotheses (targets/<H>/episode_000033; spoon roll about long PCA axis through COM on f0-25, smoothstep ramp to f40; cup = v14)
K33_H1rest -10 (physics median = joint estimate), K33_H2joint (= H1, duplicate, unused), K33_H3img -6, K33_H4settle -13,
K33_H5r16/r18/r21 (-16/-18/-21, consistency tests only).

## Runs (build-v2 exact, all gates pass; kit-faithful score_any, G18 meshes)
| tag | vs own | vs H1 | vs H4 | vs v14 |
|---|---|---|---|---|
| K33_K33_H1rest_b1 | H1 cup 1.30/6.9 spoon 1.45/9.7 | - | 1.51/7.6, 2.19/11.0 | 1.02/9.2, 2.68/10.8 |
| K33_K33_H1rest_b2 (b1 table) | H1 1.14/5.9, 1.06/8.2 | - | 1.33/7.7, 1.82/9.3 | 1.09/7.0, 2.90/11.5 |
| K33_K33_H4settle_b3 (b2 table) | H4 1.27/7.0, 1.40/7.5 | 1.13/5.3, 1.36/7.7 | - | 1.28/7.5, 3.47/13.0 |

## Packages (base = ep35_fix_20261102/K35_H2joint_b1 = v29 + ep35 fix, official 0.90; only ep33 rows replaced; k33splice --check exact)
- ep33_fix_20261102/K33_H1rest_b2 sha c52a166f - OFFICIAL 0.90 (uploaded by main): ep33 still fails at -10.
- ep33_fix_20261102/K33_H4settle_b3 sha 94259f82 - dry-run OK, not uploaded. Low odds (see large_roll_consistency.txt).
