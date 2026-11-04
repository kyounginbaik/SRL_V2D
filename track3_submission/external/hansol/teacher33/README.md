# teacher33 (2026-11-04 KST): new ep33 teacher with a left-side cup grasp and a firmer spoon grasp, target R33_HGAc18

Everything here writes only under `/data1/hansol/teacher33`. Compute only in Slurm (debug, `--exclude=node200`, node01/node100).
GT not used (target = our c18 recon `retarget33/targets/R33_HGAc18`). Nothing uploaded.

## Goal
1. LEFT hand grasps the blue cup from the cup's LEFT side, as the human does (today: right/front side, in the path of the spoon / right hand).
2. RIGHT hand holds the spoon with more fingers (today 1-3 fingers, drops at 13.5-14 s and under 1 mm wrist perturbations).
Reference to beat: R12 vs c18 cup 1.21 cm / 8.6 deg, spoon 2.09 / 17.8 (spoon dropped at 13.5-14 s).

## Where the current grasp comes from (read only)
- Current actor `streamH13a/runs/save_ep33_b1/actor` <- distilled from `streamB_ep3233/ep33/distill/i2c_sd51/actor_c1536_p0.0035_v0.0001`,
  teacher = `streamB_ep3233/ep33/ilc2/gr0.3_gp0.4_w10.5_all/iter1` (open-loop servo reference: reference.npz = per-frame qpos targets,
  executed by `floating_runtime.execute_reference` in `W1`).
- Teacher lineage: ilc2 iter1 <- ilc2 iter0 (teacher_ilc_b.py, spoon ILC 10.5-15 s) <- fcem4/off15.2_lk3.6 (fcem.py: CEM right-wrist plan
  3.6-15.2 s following target_final_v5) <- port/pg_rg0_b0.5 (delta_port.py) <- streamP_ep33 regrasp/tilc7/.../iter2 (v16 power-grasp pickup
  teacher in the compose world) <- v19 / refinement_20261021 teacher.
- LEFT cup grasp side: never planned explicitly. delta_port.py ports the left wrist by the cup-axis tilt + anchor shift only ("cup yaw
  ignored"), so the left hand keeps the old teacher's azimuth about the cup. Finger poses are those of the v16/v19 teachers (spoon: thumb-ring-
  pinky pinch, power-grasp pickup).

## Tools (code/)
- `job.sbatch` (retarget33 copy, paths here), `mlibH.py`, `mresH.py`, `mkref.py`, `snapf0_32.py`, `fcem32.py` (retarget32 copies).
- `azim33.py`: azimuth of the left/right palm about the cup axis seen from the camera (phi 0 = cup front, -90 = cup image-LEFT side) for the
  human (W1 source hand_targets, same frame as its object poses) and any rollout. -> `out/azim_base.txt`.
- `t33lib.py` / `t33table.py`: per-0.5 s table: cup/spoon error vs `$E31_TGT` (mlibH, shipped G18 meshes), number of distinct right-hand FINGERS
  (thumb/index/middle/ring/pinky/palm) touching the spoon, left-hand fingers touching the cup, palm azimuths, spoon-cup penetration.
- `lside33.py`: teacher edit. Left-wrist commands rotated about the executed cup symmetry axis by Delta(t) (const, or `follow` = keep the left
  palm at a goal azimuth), ramped in before the left hand touches the cup; right-hand finger squeeze. `grid` (servo rollouts) / `make`
  (execute_reference -> teacher dir).
- `fcem33.py`: fcem32 for ep33 (G18 ep33 meshes, scorer mlibH vs `$E31_TGT`).

## Steps
1. Azimuth (`out/azim_base.txt`, job srun 69318): human left palm phi -84..-114 deg (cup's image-left side), 6.6 cm from the cup centre.
   Team teacher left palm phi -69 at 0 s, then +2..+116 deg (front / right side) during the whole hold, 12-15 cm off the cup axis (fingertip grip).
2. `worlds/R33c18_W1` = mkref.py (actor i2c_sd51, its own W1 world, target c18; job 69319). i2c_sd51 there: cup 2.05/11.4, spoon 2.45/48.7.
3. Tables vs c18 (`out/table_base_teacher.txt`, `out/table_R12.txt`): scorer reproduces R12 (cup 1.21/8.6, spoon 2.08/17.8).
   - Base teacher ilc2 iter1: cup 2.47/15.5, spoon 2.51/47.8 (spoon roll follows target_final_v5, 130-150 deg off c18 at 6-7, 7.5-8, 8.5-9,
     12.5-13 s); right fingers on spoon 2.3-4.0 (thumb, ring, pinky, palm; index from 10 s); left cup grip thumb + middle (+pinky).
   - R12: right fingers on spoon 2.0-3.0 (thumb, ring, pinky), 0.4 at 13.5-14 s and 0 after (drop); left thumb + middle, phi +70..+125.
4. Grid 1 (`runs/lside_g1`, job 69331, 16 servo rollouts in 26 s): const D -100/-130, follow -100/-85, right squeeze 0.25 (+thumb 0.15),
   left squeeze. Every variant loses the cup at some point. follow -100: left grasp at phi -80..-120 from 0.5 s, holds to 8.0 s, then lost
   when the spoon enters the cup at 7-8 s (spoon-cup penetration 0.8-0.9 cm pushes the cup out of the thumb+middle fingertip pinch, 13 cm off
   the cup axis). Right squeeze changes the finger count little (2.6-3.5); thumb+index/middle squeeze does not bring index/middle onto the spoon.
5. `lside33.py` + left ILC (lilc, teacher_ilc_b.py left part vs the c18 cup in physics). Grid 2 (`runs/lside_g2`, job 69332): only
   D -100 + lilc 2 keeps the cup over the clip: cup 1.77/18.4, left fingers 2.4, but phi only -47 (front-left); tilt error 39-51 deg from 11.5 s.
   follow variants + ILC still lose the cup at 7-8 s. Executed: `teacher/L_D100i2` (job 69335; execute_reference == servo grid).
6. `worlds/R33c18_W1s` = snapf0_32.py of R33c18_W1 (frame-0 gap cup 3.1 mm, spoon 9.2 mm), for distillation + margin search.
7. Parallel CEMs (node100):
   - `teacher/F_D100i2` = fcem33.py on L_D100i2 (right-wrist spoon plan vs c18, t_lock 3.6, t_off 15.2, ramp 0.15, 10 knots, 64x20, seed 3301,
     grasp term wg 0.3 / wd 1.0), job 69336.
   - `teacher/LC_f100` = lcem33.py on the team teacher, follow -100 (left cup-frame wrist knots 8 over 1.2-16.2 s, phi offset, left squeeze;
     cost + azimuth penalty wa 1 + left <2 fingers wl 1.5), 64x15, seed 3302, job 69337. INIT cost 20.8 (cup lost).
   - All three (69336, 69337, follow-up 69339) CANCELLED at ~16:45 on the main session's request: spoon grasp first, left-side cup deferred.
     Last states: F_D100i2 it0 cup 1.63/13.3, spoon 2.68/24.6 (MP-SR pass); LC_f100 it8 cup 3.19/17.6, spoon 3.04/51.7, phi -95, left <2 fingers 9%.

## Spoon grasp (priority from 16:45)
8. Where the pinch comes from (`code/fdist33.py`, `out/fdist_base.txt`, `out/fdist_R12.txt`):
   - Flexion headroom (upper limit - qpos, summed over the finger's non-AA joints): index 2.9-3.2 rad in the team teacher (almost fully open),
     1.7-2.7 in R12; middle 1.5-1.7 / 0.2-0.9; ring 0.9-1.1 / 0.15.
   - Geometry (spoon frame, cm): all five fingertips within +-3 cm of one point at x -1..-6 near the spoon origin; palm at x -13..-15. The handle
     runs ALONG the fingers and is pinched at the tips (not across the palm).
   - Origin: the v16/v19 teacher hold. `prepare_pickup_b.py` transplants that wrist pose relative to the spoon unchanged; its power variant only
     closes middle/ring/pinky (not index). Team ep33 pick1 (physics_pickup_search_v2 from that anchor): held spoon 6.08/41.3, power0.6 12.05/102.6
     (no grasp; its search log tracks 13 cm, both-contact 2%).
9. Grid 3 (`runs/rflex_g3`, job 69352): per-finger flexion offsets (index +0.4..1.0 rad per joint, middle +0.4..0.8, AA +-0.2, ramp 1.6-2.4 s).
   None adds fingers on the spoon (<= base 3.1); several throw it (nominal spoon 17-7400 cm).
   (The g3/g4 "spoon_drop_rate" fields used a first drop test that also counted tracking error > 8 cm vs c18; superseded below.)
   Drop test since 17:04 (lside33.drop_metrics, target-independent): spoon drop = no right finger on it >= 0.5 s or spoon-palm > 20 cm for
   >= 0.25 s within 2.6-14 s; cup lost = no left finger >= 0.5 s or cup-palm > 22 cm >= 0.25 s within 1.5-16 s.
   Team teacher under the team fcem_r noise (`code/pert33.py`, `out/pert_base.json`, job 69369; 4 draws per level):
   0.5 / 1 / 2 mm: spoon drop 0/4, 0/4, 0/4; cup lost 2/4, 0/4, 3/4; pass vs c18 0/12; median spoon 8.7 / 13.7 / 4.9 cm (nominal 2.5) ->
   the spoon slips/turns in the pinch (not dropped), the cup grip is fragile.
10. Human grip (`code/ikgrip33.py`, `out/ikgrip_base.txt`): W1 source Mink IK `sharpa_ik.npz` (quaternion wxyz: FK fingertips 2.6 cm from MANO
    tips; xyzw 17 cm). Placed on the teacher's held spoon at 5 s it touches with thumb, index, middle, ring, pinky and palm (fingertips spread
    along the handle x -4..-8; 5-17 mm penetration, IK has no contact). Grid 4 (`runs/ikg_g4`, job 69361): blending wrist (slerp relative to the
    executed spoon) + fingers to that grip from 1.7 / 2.6 / 3.0 s, ramp 0.5 / 1.0 s, relax 0.9 / 1.0, alpha 0.5-1.0: EVERY variant loses the
    spoon (right fingers ~0; before contact the wrap cannot form around a handle lying on the table, after the lift the switch releases it).
    => a firmer grasp needs a new pickup (not reachable today; the team pickup search failed twice on this spoon).
11. Teacher CEMs on the team teacher (no left change), target c18, t_lock 3.6, t_off 15.2, ramp 0.15, 10 knots:
    - `teacher/FR_base` = fcem33r.py (cost = mean of nominal + 3 noisy rollouts at 0.5 mm + 0.5 max), 48x12, seed 3311, job 69362.
    - `teacher/FN_base` = fcem33.py (nominal + finger-count term), 64x20, seed 3312, job 69363.
12. RESULT `teacher/FN_base` (executed, `out/table_FN_base` = logs/T33-tabFN-69390.log): vs c18 cup 1.42/6.1, spoon 2.22/21.4 (official proxy
    MP-SR 1.0, proxy spoon 2.05/21.4). Right fingers on the spoon 3.0-4.8 over 3-14.5 s, never < 2; index + middle join for most of 6-11.5 s;
    hold until 15.0 s (team teacher: 2.3-4.0 fingers; R12 2.0-3.0 and 0 from 13.5 s). Spoon deg 24-50 at 13-15.5 s (put-down roll).
    Perturbation (pert33, `out/pert_FN_base.json`, job 69391): 0.5 / 1 / 2 mm: spoon drop 0/4, 0/4, 0/4; cup lost 0/4, 1/4, 0/4; pass 0/4, 0/4, 1/4;
    median spoon 39.0 / 3.8 / 3.1 cm. Still a slipping grip, but no drop and fewer cup losses than the team teacher (5/12).
    FR_base (robust cost): stuck from iteration 1 at nominal cup 1.73/19.6, spoon 2.5/30.8 (robust cost 2.98); left to finish.
13. Distillation DN (`code/distill33.sbatch`, team actor_grid_c.py, snapped W1s world, job 69373, ~1 min): no grid point passes (as on ep32).
    c1536_p0.002 cup 1.78/4.3, spoon 3.06/21.0 (closest); c1024_p0.001 1.30/9.2, 3.36/28.4; c3072_p0.003 0.90/4.0, 2.84/53.8;
    c2048_p0.003 0.95/7.4, 2.37/69.6; c1536_p0.0035 (team point) 2.09/6.2, 34.9/56.7 (lost).
14. Closed-loop perturbation (`code/perturb33.py` = retarget32 perturb32.py for ep33, team V8 rob.py draws, 16 x 8-knot right-wrist 1 mm / 8 mrad /
    8 mrad squeeze over 3-14 s): R12 (H13 actor + R12_D table) nominal cup 1.21/8.6, spoon 2.08/17.8; perturbed pass 0/16, spoon lost (> 10 cm)
    12/16, median spoon 663 cm (`out/perturb_R12/perturb33.json`, job 69389).
15. Margins stage 1 (`code/margins33.sbatch`, team search_margins_k.py, right hand, 64x12, ~18 min each):
    - `runs/distill/DNm1_c1536_p0.002` (job 69393): PASSED, official proxy cup 1.88/7.7, spoon 2.85/25.4 (mlibH vs c18: 1.79/7.7, 3.00/25.4).
    - `runs/distill/DNm1_c3072_p0.003` (job 69394): mlibH 1.22/7.9, 2.60/33.5 (fails on spoon deg).
    - Unmargined DN c1536_p0.002 under perturb33 (job 69395): nominal 1.63/4.2, 3.19/21.0; pass 0/16, spoon lost 9/16, median spoon 27.3 cm.
    - FR_base (robust teacher CEM, job 69362) final: proxy MP-SR 0, mlibH cup 1.73/19.6, spoon 2.50/30.8 -> not distilled (DR 69374 cancelled;
      the nodes were 244/256 CPUs busy).
16. Margins stage 2 from DNm1 (96x12, seed 3333, job 69412, 26 min): no accepted gain (team cost kept DNm1; same checkpoint 6245250c).
    DNm1 (`out/` logs T33-tabM1-69414, T33-pertM1-69413): vs c18 cup 1.79/7.8, spoon 2.99/25.4; right fingers thumb+ring(+pinky)+palm 2.5-3.5,
    middle only to 8.5 s, released at 14.5 s. perturb33: spoon lost 1/16, pass 1/16, median spoon 3.23 cm (IQR 3.09-4.07)  [R12: 12/16, 0/16, 663 cm].
17. Window-beam on DNm1 (`code/t33hg.sh` = r33hg.sh + ACT= / TABLE none / c18 / teacher33 outputs; team burst/tail settings, 16 workers):
    M1_A burst 2.5-5.0 s, 0.25 s windows (job 69417, 21 min): final_0 cup 1.53/4.1, spoon 2.27/28.0 (full rollout).
    M1_B burst 5.0-8.0 s (69430, from M1_A/beam_w09) -> M1_C tail 8.0-17.35 s, 0.5 s windows (69431, from M1_B/beam_w11).
18. Fallback actor `runs/save_M1_A_f0/actor` (`code/save33.sbatch` = mresH save in W1s, job 69451): checkpoint 47830eb8..., physics_verified,
    official proxy MP-SR 1.0 (kit proxy cup 1.77/4.2, spoon 2.16/28.0; mlibH vs c18 1.53/4.1, 2.27/28.0; vs r6 2.67/16.5, 6.06/26.5).
    perturb33 (job 69452): spoon lost 2/16, pass 2/16, median spoon 2.72 cm (IQR 2.64-3.70), 34.9 deg.
19. Packaging chain (teacher33 copies, all output here): `code/t33svst.sbatch NAME SAVE_DIR` = stage_a_t33.sh (env T33=<teacher dir>, T33NOTE=;
    stager stage_b_t33.py also accepts teacher33/worlds) + g18swap_t33.py + t33run_one.sh (build-v2) -> `candidates/R33_HGAc18_<NAME>`;
    `code/t33pkg.sbatch CAND_TAG LABEL` = r33pkg.py copy -> `packages/<LABEL>/all_20_episodes`.
    M1_A_f0 (jobs 69455, 69461): build-v2 exact True, 696 rows; package `packages/track3_FINAL_ep33T33_M1Af0_c18_20261104/all_20_episodes`
    sha256 2be2d8437209b28a16bbeba2a5eb83faf53817c9686dd3ff00a0463568105cf3 (CD-O a03230e5d6ecd598), only e000033 rows differ (696), kit
    validate_submission OK (9120 rows, 20 episodes). NOT uploaded.
    Package score (score_vs_compose.py, `out/pkgscore/`): vs k33x HGAc18 cup 1.52/4.1, spoon 2.28/28.0; vs k33x HGAZr6_HGA (= R33_HGA_r6 source)
    cup 2.67/16.5, spoon 6.05/26.5. R12 package vs HGAZr6_HGA: cup 2.02/13.3, spoon 5.31/26.1 -> neither passes vs r6 (frame-0 spoon roll; needs
    an r6 world + retarget, not done).
20. FINAL (20:28 KST). Beam M1_B 5-8 s (job 69430): final_0 cup 1.44/6.1, spoon 1.72/24.9. Tail M1_C 8-17.35 s (69431): final_0 cup 1.27/7.4,
    spoon 1.00/9.7. Baked `runs/save_M1_C_f0/actor` (job 69548): checkpoint 0d63ea988b915c39..., physics_verified, official proxy MP-SR 1.0
    (kit proxy cup 1.49/7.5, spoon 1.05/9.7; mlibH vs c18 1.27/7.35, 1.00/9.66). Table `logs/T33-tabC-69550.log`: right fingers on the spoon
    2.5-3.8 over 3-15 s (thumb+ring+pinky+palm, middle 3.5-7.5 s, index at 6.5-8 / 12.5-15 s), hold to 15.0 s (R12: 0 from 13.5 s); spoon
    <= 1.4 cm in every 0.5 s window from 3.0 to 15.0 s; put-down 15-15.5 s 3.35 cm / 32.6 deg.
    perturb33 (69549, 16 x 1 mm / 8 mrad right-wrist draws): pass 10/16, spoon lost 2/16, median spoon 1.81 cm / 19.8 deg (IQR 1.49-3.04)
    [R12: pass 0/16, lost 12/16, median 663 cm].
    Package (69551 build-v2 exact True; 69552) `packages/track3_FINAL_ep33T33_M1Cf0_c18_20261104/all_20_episodes/track3_submission.parquet`
    sha256 282ca71521dad556da3cc2c369cd8e02ed27344ebbcb4f0918fe541ec3efa5a8, only the 696 e000033 rows differ, kit validate_submission OK.
    score_vs_compose: vs HGAc18 cup 1.27/7.3, spoon 1.01/9.7; vs HGAZr6_HGA (r6) cup 2.45/14.2, spoon 5.81/19.8 (R12 2.02/13.3, 5.31/26.1).
    NOT uploaded. Left-side cup grasp: not done (deferred at 16:45; tools lside33.py / lcem33.py and notes in steps 4-7).

## Same recipe on FINAL's own ep33 target K33_H5r16 (main session request, 20:33 KST)
21. Target `/data1/hansol/retarget33/targets/R33_H5r16` = byte copy of streamK33_ep33/targets/K33_H5r16 (SHA256SUMS identical, CW_TARGET.json).
    World `worlds/R33H5r16_W1` (mkref, i2c_sd51 actor, job 69563; i2c_sd51 there cup 2.79/26.6, spoon 5.99/29.0) and `worlds/R33H5r16_W1s`
    (snapf0; frame-0 gap cup 2.8 mm, spoon 51.1 mm: H5r16 starts with the spoon rolled -16 deg vs every scene's start pose).
    Kit-faithful (mlibH) vs H5r16: FINAL c3 (streamK33 save_K33_H5r16_ep33_c3) cup 1.40/7.7, spoon 1.68/7.6 (`out/table_FINALc3_H5r16.txt`);
    team teacher 3.66/28.3, 5.76/33.7; R12 2.04/13.3, 5.29/50.8. All three have spoon 4.46 cm at 0-0.5 s (shared rest-phase offset).
    Scripts now take T33TGT / T33EPD (distill33.sbatch, save33.sbatch; c18 defaults unchanged, backups *.bak_c18); new margins33auto.sbatch
    (picks the distilled actor with the lowest max(cm/3, deg/30), then margins stage 1).
22. Chain (all TG=R33_H5r16, W1s H5r16): fcem33 FN_H5r16 (69565, seed 3321) -> table 69582 -> distill DH5 (69571) -> margins33auto DH5m1 (69572)
    -> beam H5_A 2.5-5.0 (69573) -> H5_B 5.0-8.0 (69574) -> H5_C 8.0-17.35 (69575) -> save_H5_C_f0 (69576) -> perturb33 (69577), table (69578),
    t33svst (69579) -> t33pkg track3_FINAL_ep33T33_H5Cf0_H5r16_20261104 (69580).

## Same recipe on R33_HGA_r6 (main session, 21:10 KST: user says the flipped spoon direction (c18/r6 family) is correct)
23. World worlds/R33r6_W1 (mkref, i2c_sd51 actor, job 69637; i2c_sd51 there cup 2.76/26.6, spoon 6.00/49.4) -> worlds/R33r6_W1s (snap, 69638;
    frame-0 gap cup 2.8 mm, spoon 51.1 mm, same -16 deg start-roll gap as H5r16). Snapping changes only the world reference frame 0, not the kit
    scorer anchor (rollout frame 0 = scene start pose), so the ~4.5 cm rest-phase spoon offset at 0-2.5 s stays.
    Chain: fcem33 FN_r6 (69639, seed 3371) -> table 69640 -> distill DR6 (69641) -> margins33auto DR6m1 (69642) -> beam R6_A/B/C (69643-69645)
    -> save_R6_C_f0 (69646) -> perturb33 (69647), table (69648), t33svst (69649) -> t33pkg track3_FINAL_ep33T33_R6Cf0_r6_20261104 (69650).
    -> H5r16 chain cancelled 21:20 (69572-69580; user: flipped direction is right). Unmoved r6 chain cancelled 21:19 (69639-69650).

## Moved-start r6 world (main session 21:15: move the physics scene start to the r6 frame-0 poses)
24. code/build_world_t33.py = streamB build_world_b.py with reference $E31_TGT (blib.TGT override) and --src (previous layout W0 + source copy).
    Run: E31_TGT=R33_HGA_r6, base scene = team ilc2 iter1 execution scene, --base-epd W1 episode_corrected, --src W1 source_corrected, G18 meshes,
    --short 0 (as the ep33 W1 build) -> worlds/R33r6M/ep33/{scene,source_corrected,episode_corrected}. Start moved vs W1: cup 0.35 cm / 1.9 deg,
    spoon 0.47 cm / 15.7 deg; tilt -0.60 deg, lift 0.08 cm; spoon bottom +1 mm; no hand-object penetration at frame 0.
25. code/delta_port_t33.py = streamB delta_port.py + --const (frame-0 deltas only). Ports of the team teacher (jobs 69653-69660):
    full_b0.5: pickup OK, right fingers 1.8-4.3, cup 1.59/17.4, spoon 4.67/105.7; full_b1.0 cup 3.00/16.3, spoon 3.54/61.4;
    const_b0.5 cup lost later (9.9 cm at 12.5 s); const_b1.0 cup thrown. Spoon at 0-0.5 s vs r6: 1.06 cm / 6.6 deg in all (was 4.46 / 11.7).
26. Chain on P_r6M_full_b0.5 (world R33r6M; frame 0 = keyframe by construction): fcem33 FN_r6M (69661, seed 3401) -> table 69662 ->
    distill DR6M (69663) -> margins33auto DR6Mm1 (69664) -> beam R6M_A/B/C (69665-69667) -> save_R6M_C_f0 (69668) -> perturb33 (69669),
    table (69670), t33svst (69671) -> t33pkg track3_FINAL_ep33T33_R6MCf0_r6moved_20261104 (69672).
27. R6M results so far: teacher FN_r6M (69661) vs r6 cup 0.85/7.2, spoon 1.43/13.7 (proxy MP-SR 1); distill DR6M picked c512_p0.003 cup 0.80/6.9,
    spoon 1.38/15.2 (passes); margins stage 1 (69664) kept it unchanged -> DR6Mm1. Beam R6M_A..C running (right hand).
28. R6M2 (both hands, main session 21:55): t33hg.sh OW= override (backup t33hg.sh.bak_2215); HANDS=wooden_spoon:right,blue_cup:left RS=right,left
    ADVS=right,both HG=6,1.0 OW spoon 3 / cup 2 (as ep32 T32c_b1): beam R6M2_A/B/C 69712-69714 -> save 69715 -> perturb 69716, table 69717,
    svst 69718 -> pkg track3_FINAL_ep33T33_R6M2Cf0_r6moved_20261104 (69719).
29. RPM: targets/R33_RP (retarget33) = byte copy of streamK33 targets/K33_RP; world worlds/R33RPM (build_world_t33, job 69720; frame-0 start moved
    as R33r6M); delta_port_t33 full/0.5 (69721) -> fcem33 FN_RPM (69723) -> distill DRPM (69725) -> margins33auto (69726) -> beam RPM_A/B/C
    (69727-69729) -> save (69730) -> perturb (69731), table (69732), svst (69733) -> pkg track3_FINAL_ep33T33_RPMCf0_RPmoved_20261104 (69734).

## 30. R6M package (23:42 KST Nov 4)
packages/track3_FINAL_ep33T33_R6MCf0_r6moved_20261104 — sha256 ad3487002b5c465696b665b40e9461ea4467ec3112aaa331225d3a9d798feaf6 (CD-O a03230e5d6ecd598).
Official eval vs R33_HGA_r6: cup 0.67/3.5, spoon 0.84/7.7, MP-SR 1.0, physics_verified. Beams A 1.15/48.5 → B 1.00/11.4 → C 0.84/7.7 (spoon).
perturb33 16 draws: pass 10/16, spoon lost 0/16, median spoon 1.41 cm / 20 deg. build-v2 exact, kit validate OK.
0.5 s table: spoon ≤1.4 cm at 3-14 s; pickup peak 2.2/24.9 at 2.0-3.0 s; spoon released at 14.0 s (end-of-clip fix target).

## 31. R6M2 package (23:57 KST Nov 4) + R6M end-of-clip fix chains (00:0x KST Nov 5)
R6M2 (both hands): packages/track3_FINAL_ep33T33_R6M2Cf0_r6moved_20261104 — sha256 c96244f0eac93d68014ec0f247d4ac56ff897264319397cd6b3548e357571988. vs r6 cup 0.645/3.44, spoon 0.832/7.79, MP-SR 1.0; perturb 12/16 pass, 0/16 lost; build-v2 exact, kit validate OK. Same early spoon release at 14 s as R6M.
R6M_E1/E2 (main session request; code/r6me.sh): burst beam on runs/cw/R6M_C/final_0.npz, windows 13.75-16.00, wbeamHGg SH 4,4,5,280,322 NC 60 WK 24, right fingers held from 13.9 s.
  E1 FZ right,13.9,0.2,15.0: beam 69881 save 69882 pert 69883 table 69884 svst 69885 pkg 69886 -> packages/track3_FINAL_ep33T33_R6ME1_r6moved_20261104
  E2 FZ right,13.9,0.2,17.25: beam 69887 save 69888 pert 69889 table 69890 svst 69891 pkg 69892 -> packages/track3_FINAL_ep33T33_R6ME2_r6moved_20261104
