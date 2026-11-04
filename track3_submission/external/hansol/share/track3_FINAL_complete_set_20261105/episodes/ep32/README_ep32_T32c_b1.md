# ep32: how it passed MP-SR (0.90 -> 0.95) on 2026-11-04 — handoff

Official: `package/track3_submission.parquet` (sha256 bd5b3321a8017359...) = FINAL_20261104 (0fca055b) with only the 518 ep32 pose rows
replaced (mesh cells untouched). Kaggle MP-SR submission 57979782 = **0.95** (ep32 passes on the hidden evaluator; FINAL was 0.90).
AUC/MPPE/RPE of this package not uploaded yet. No GT used anywhere; public GT never read.

## 1. Reconstruction (recon/, target/)
Problem found by video review: the spoon pose in the old track (K32_R10 = v14_g17 + leaderboard start roll) swung ~35 deg for a few frames
at 4.5 s and 7.6 s, and the bowl roll was reversed vs the human's spoon in the scoops; the cup wobbled and drifted at 6.5-8.2 s.
Fix = GT-free per-frame re-solve against the video itself (recon/tools/):
- `refine.py` / `refine_cup.py`: per-frame pose fit to the object masks (hand-occlusion aware), stereo depth, and the spike-ball detections
  (grey marker balls on the objects), starting from the old track and Hansol's track.
- `refine2.py`: temporal DP over candidates + red-black sweeps with a second-difference prior.
- `merge.py`: joint spike-model + pose solve (candidates: old / Hansol / ours / Jisu + roll variants), cost = masks + stereo + spikes +
  the user's bowl-direction marks (`marks_ep32.json`, 58 marks; weight 4 = `MW=4`) + temporal prior.
- `merge_cup2.py`: cup re-solve adding CoTracker point tracks (LAM 0.4), then SG smoothing (rot 9 f, pos 7 f).
- `mk_target2.py`: assembles bundle + target `K32_RP2` (spoon f0-20 and cup f0-5 kept from K32_R10; the scorer aligns on frame 0).
Held-out checks (detections/points the solver never saw): spoon spikes 4.4 px (old 8.2, Hansol 6.4), cup spikes 3.9 px (old 9.9),
user marks 57/58 (old 29, Hansol 42-46), mask IoU spoon 0.83 cup 0.93 (old 0.69 / 0.86). Video: `videos/ep32_recon_old_vs_final_RP2_depth.mp4`.

## 2. Retarget (teacher/, actor/, beam_table/, candidate/)
The old actor (H12 save_e1f0) spins the spoon 300 deg at 3.5-5 s and 7.5-8 s: a wrist residual learned against the old reversed-roll recon.
No window-beam / hold-seed / unwrap could remove it (all dropped the spoon). New actor pipeline (streamT32_teacher, code/):
1. World `W_RP2` = `build_world_b.py 32 OUT <streamR power0.6 scene> --base-epd <streamR compose/C4 epd> --meshes <B9 meshes 3c3c/c9ab> --short 0.0178`
   with blib.TGT -> targets/K32_RP2. (Later patch: a no-penetration safety lift at frame 0.)
2. Teacher port: `delta_port.py <streamR A3_t10_g0.3/it8 reference> ... --blend 0.5 --right-gain 0`.
3. `fcem2.py` = team fcem with BOTH wrists following their object targets from t_lock (left: cup, right: spoon), per-hand knots and squeeze,
   cost ratios vs 1.2 cm / 15 deg, optional `--t-on`, `--scoop-w`. Best teacher `teacher/fcem2_best_s11b` (lk 3.25, 90 it):
   cup 1.00/13.7, spoon 1.56/14.2; the `--t-off 12.0` variant reached cup 1.06/12.4 spoon 1.25/12.7. Single-hand fcem failed
   (the spoon knocks the cup out of the left hand at 4.5 s). `fcem3.py` adds 3 squeeze groups per hand (thumb / index+middle / ring+pinky).
4. Plain distill (`distill.sbatch` -> actor_grid_c): every actor loses the spoon EXCEPT `actor/distilled_actor_c1024_p0.01`
   (proxy MP-SR 1: cup 1.79/21.3, spoon 1.93/17.0).
5. Window beam (team wbeamH via `k32rchain_t32.sh`: both hands, ow cup 2, hinge 6@1.0, the actor's OWN meshes 3c3c/c9ab) on that actor:
   `beam_table/T32c_b1_g2_final_0.npz` -> cup 0.73/8.6, spoon 0.76/9.3 (full-horizon re-roll).
6. `t32_finish.sh`: `mresH.py save` (bake, E31_EPD = W_RP2 world) -> streamB stager `stage_b.py` (lineage root patched to streamT32)
   -> `run_one.sbatch` (canonical build-v2, FINAL evaluator hashes cbe866f9/fdddb348, exact replay) -> vs.sh -> k32splice -> `splice32.py`
   -> dry-run. `candidate/T32_T32c_b1_episode_000032` = the build-v2 candidate (results parquet, meshes, completed.json, backend evaluation).
   Score vs K32_RP2: cup 0.73/8.6, spoon 0.74/9.3, MPPE 0.961, AUC 0.957. Video: `videos/ep32_T32c_b1.mp4` (4 panels).

## 3. Pitfalls we hit
- Physics is CPU-dependent (chaotic contact): node200 (Intel) vs node01/node100 (AMD) give different rollouts for the same plan
  (spoon 2.0 vs 4.8 cm). Everything that feeds a package ran on node01/node100.
- CEM resume into an existing out dir fails at export (FileExistsError); export the best with `--init progress.json --iterations 0` into a fresh dir.
- Hold-seed (holdgrid) and both-hands beam on the OLD actor broke the grasp; the both-hands beam only works on the new-lineage actors.
- Old vs new recon differ ~30 deg in spoon roll on average; any actor built on the old recon is far off the new one (RP2_c1: 2.09/28.1).

## 4. Where things live
streamK32_ep32/recon_perfect (recon), streamT32_teacher (teacher/actor), streamK32_ep32/runs/t32 (beams, saves), common_v2_candidates/T32_T32c_b1,
challenge_submissions_SRL/track3_final_ep32_T32c_b1_20261104 (package). `notes/RECIPE.md` = staging recipe; `notes/jobs_log.txt` = job log.
