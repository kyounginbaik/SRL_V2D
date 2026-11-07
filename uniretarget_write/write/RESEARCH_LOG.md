# UniRetarget auto-research log (pen, write_on_whiteboard)
Pass rule: write/wmetric.py (fixed before runs) + overlap <= 0.3 cm. All runs: 23 g pen, board, friction DR 0.6-1.4, 100M steps, eval at VOC 0 from frame 0.

## Baselines so far (start = raw human IK unless noted): 0/16 pass
- plain RL (base, no DR) 0/3; sequential practice loop 0/3; plain RL + DR 0/3; practice-in-batch v1 (assisted fitness) killed; v2 (probe fitness) 0/3
- contact-projected reference: plain 0/2 (+1 pending), practice-in-batch 0/2 (+1 pending)
- checked grip g0 (planner-found): plain 1 run 8/32 worlds; sequential practice 1/3 runs (32/32)
- grasp-first smoke: pen lost after ~3 frames unassisted for every strategy

## Round 1: assistance-weighted reference update (engine assist, retargeting/assist_refine.py)
Arms: track (hand tracking reward 0.5/0.5, no upper), assist (s_max .75), assistall (s_max 1.0); seeds 42,43. Jobs: research_r1_jobs.txt

## Diagnostics between rounds
- Static frame-0 hold (MuJoCo, official fingers): human IK holds (0.3 cm drift), contact-projected drops (3.4 cm) unless squeezed, checked grip holds (0.1 cm).
- Rigid human grips, no jitter (rigid_ik.json): the frame-0 human grip with squeeze 0.2 + servo 0.1 PASSES the full pen rule in MuJoCo (tip 0.61 cm, on-board 1.0, overlap 0.11). Grips from later IK frames lose the pen.
  => The human grasp is fine; the human finger MOTION (which penetrates the pen, 1.06 cm median) is what breaks it.

## Round 2: rigid frame-0 human grasp reference (clip write_g6 = demo's own initial grasp, settled, held rigid on the object path)
Arms: base (plain RL), assist (hand tracking + assistance-weighted update); seeds 42,43, A100 queue. Jobs: research_r2_jobs.txt

## Round 1 result: 0/6 pass (all from raw human IK, DR)
- track s42 tip 13.1 cm held .19 (nominal); passes nearly under friction 1.3 (tip 1.2, held .97) and mass 2 -> fragile, closest of all; track s43 37 cm
- assist s42/s43 37 cm, assistall 37/34 cm. Upper level drifted monotonically: every update hit the 5 mm trust region, wrist drift 7 cm by 95M, obj err in counted samples 2.3 cm (no improvement).
- Diagnosis: the update moves the reference toward the executed state, which lags the reference in every sample (tracking error), so the reference ratchets after the hand instead of toward success.
- Seed-44 contact-projected: plain 40 cm, PIB 22 cm. Totals now 0/22 from demo-derived refs.

## Round 3 (parallel with round 2): advantage-form update (baseline=1)
Delta = (success-weighted mean - plain mean) of executed states per frame: the shared lag cancels, only what separates good samples from bad moves the reference.
Arms: g6 assistb s42,s43; g6 base s44 (3rd seed). Jobs: research_r3_jobs.txt

## Round 2 result: 0/3 pass (g6 = demo frame-0 grasp held rigid; plain s42/43/44 tip 38/36/34 cm; assist s42/43 37/38 cm)
Old assist update again saturated the trust region every time (drift 5.8 cm).

## Key diagnostic: zero-residual rollout IN NEWTON (eval_ref.py / evalzero.sbatch: policy output forced to 0 = pure reference tracking, VOC 0)
| reference | pen held | tip cm | pen drops at |
| g0 (planner grip) | 100% | 1.3 (on-board 0: tip hovers) | never |
| g6 (demo grasp, MuJoCo-checked PASS) | 0% | 39 | ~1 s |
| wb raw IK | 0% | 28 | ~1-2.5 s |
| wbp contact-projected | 0% | 39 | <0.13 s |
=> (1) MuJoCo validation does not transfer to Newton: the feasibility check must run in the training simulator.
=> (2) RL from g0 got only 8/32 although the reference alone holds 100%: training under VOC assistance + exploration breaks a feasible grip.
   RL from g6 drops the pen faster (0.25 s) than the open-loop reference (1 s).

## Round 4: g0 with NO assistance at all (curriculum write_100m_novoc) - plain vs hand-tracking reward, s42/43. Jobs: research_r4_jobs.txt
Hypothesis: when the reference already holds the object open-loop in the sim, assistance is harmful; practice = RL from a physically valid reference with VOC 0.

## Round 3 result: 0/2 (advantage-form update from g6: tip 42/38 cm). Assist-update family dropped.

## Round 5: open-loop practice in the training simulator (write/oloop.py, oloop.sbatch)
Upper level = CEM over 11 grip numbers (6 constant grip offset in the object frame + 5 per-finger squeezes) on top of a parent hand motion.
Each generation: 16 candidates tiled in time, 2048 Newton worlds with friction DR (~128 trials per candidate), zero policy residual, no VOC.
Held = pen has not slipped 3 cm in the hand (goal-distance criterion was wrong: the PD wrist lags the fast early motion by 3-4 cm).
Sanity (parent g0): mean candidate held 69% of DR worlds, best candidate 100% held, tip within 3 cm on 97.5% of writing frames.
Searches: parent g6 (demo grasp rigid), parent wb (raw human IK); 30 generations each.

## Round 4 result: 4/4 PASS (g0, no assistance at all)
| arm | s42 | s43 |
| plain RL, VOC 0 | 32/32, tip 0.26 | 32/32, tip 0.21 |
| hand tracking, VOC 0 | 32/32, tip 0.22 (passes from 25M on) | 32/32, tip 0.20 (passes from 25M on) |
Previously from g0 with the VOC curriculum: plain 8/32 (1 run), practice loop 1/3 runs.
Robust: pass at friction x0.7/x1.3, kp x0.5/x2, mass x2 (base-s42 kp2 19/32, track-s42 fr.7 23/32); fail at friction x0.5 (outside DR).
Finger-mesh overlap p5 0.12-0.15 cm (rule <= 0.3): full pass. Videos videos/r1/noAssist_*_PASS.mp4.
=> Assistance (VOC) was the main obstacle. Once the reference holds open-loop in the training sim, RL with NO assistance refines it.

## Round 6: same no-assistance recipe (hand tracking, VOC 0) on demo-derived refs: g6 and raw IK wb, s42/43. Jobs: research_r6_jobs.txt

## Round 6 result: 0/4 at 100M (no assistance, hand tracking)
- g6 s42: 4/32 @25M, 29/32 @50M (tip .36), 14/32 @75M, 0/32 @100M (collapsed). g6 s43: 0 at all checkpoints (tip .78 @50M but off-board), 0 @100M.
- wb raw IK s42/s43: 0 at all checkpoints.
=> Without assistance the demo grasp CAN be practiced to success, but it is unstable: the reference itself does not hold open-loop in Newton (drops ~1 s), so the policy fights it.

## Search fix: CEM averaging diverged (mean fitness fell every generation after gen 3). Re-scoring confirmed the scorer is reproducible (x=0 gen0 0.0192 vs 0.0193).
Near the gen-3 best, a candidate holds open-loop in 92% of DR worlds with the tip within 3 cm on 96% of writing frames
(edit: 1.6 cm offset, 21 deg roll, thumb/index squeeze +.11/+.17) -> saved as oloop/g6_open92/best.npz. Search switched to elitist (1+15).

## Round 7: no-assistance RL (hand tracking) from g6_open92 (demo grasp + Newton open-loop grip edit), s42/43. Jobs: research_r7_jobs.txt
Elitist searches from scratch (es_g6, es_wb) in parallel: does the search find a holding edit without luck?

## 2026-11-07 10:20 — bug: friction DR compounded across generations
- `envs/base.py` FRIC_DR multiplied `shape_material_mu` in place on every RLEnv build; oloop builds one env per generation on the same model, so gen g had friction fac^g per world. Explains the monotone decay of the elite's fitness in ol_es_g6 (0.88 at gen 5 -> 0.15 at gen 25) and ol_es_wb, and taints gens>0 of cem_g6/cem_wb.
- g6_open92 (92% held) came from a single-generation re-evaluation, i.e. one env build -> valid.
- Fix: cache the original mu on the model (`_fric_dr_mu0`) and apply the factor to that copy. Training builds one env, so RL runs were unaffected.
- Relaunched: ol_fx_g6 (2704916), ol_fx_wb (2704917). r7 training jobs had sat in the A100 queue for ~7 h; moved back to base_qos.

## Round 7 partial: g6_open92 (demo grasp + Newton grip edit) no-VOC track s42 -> 0/32 at all checkpoints (tip 7-22 cm). s43 still training.

## Round 8 (2026-11-07 11:30): everything in Newton, no demo hand (user: "do all in newton", "doesn't need to start from the demo")
- Upper level `write/scratch.py` (+ scratch.sbatch): candidate = rigid grip (pen pose in hand frame relative to the hand's tripod point from
  Newton FK, grip point along pen, 5 finger closures, thumb opposition) = 13 numbers; hand ref = object path o inv(grip). Only the object path is input.
  K=128 slots tiled (clip write_wb_pib128), 2048 worlds (16 DR trials per candidate), zero-residual rollouts, VOC 0, no terminations.
  Gen 0 uniform random orientations; then 8 parents x 16 children, parents re-measured. Seeds 0/1 (sc_s0 2705266, sc_s1 2705267).
  Gen 0 already: best random grip holds in 67% / 93% of DR worlds.
- Lower level: chain_scratch.sh N -> no-VOC track RL s42/s43 on write_wb with the best snapshot (runs/write_wb_scN), eval/zero/robust. Jobs -> research_r8_jobs.txt.
- Rendering in Newton: `write/nrender.py` (headless ViewerGL, PYGLET_HEADLESS=true) + nrender.sbatch; finger-pen distance from Newton's collision pipeline.

## Round 7 result: 0/2 (g6_open92 = demo grasp + Newton edit, holds open-loop 92%; no-VOC track RL s42/s43). Best ckpt s42@25M tip 8 cm held 78%; 100M: s42 tip 7.5, s43 tip 31. Robust s43 all 0.
- Training did load the snapshot (hydra config robot_reference_path = oloop/g6_open92/best.npz).
- => "holds open-loop from frame 0" is NOT sufficient (g0 passes, g6_open92 fails). Hypothesis: training resets at random clip frames; the oloop score only tests frame-0 starts.
  g0 (rigid planner grip) may hold from any frame, the edited demo grip not. Next check: zero-residual hold from random start frames for g0 vs g6_open92 vs scratch best.
- Fixed (non-compounding) ol_fx_g6 reached held 99-100% by gen 5 before cancel: the search itself works on the demo grasp.
- Reset mechanics (lifecycle/reset.py): training resets pick a random frame (frame 0 w.p. 0.1 when curriculum VOC < 0.1), open the fingers by
  a random factor in [0, 0.7], and run a 20-step settle in which applied VOC decays 1 -> curriculum value (so even "no-VOC" training has a short
  assisted grasp-formation phase). oloop probes used exact pose + no settle, i.e. not what training sees.
- `write/holdany.py` (+holdany.sbatch): zero-residual hold for 3 s after a training-style reset, 2048 worlds, friction DR.
  | ref | RL | held3s from frame 0 | held3s from random frames |
  | g0 | PASS 4/4 | 0.33 | 0.19 |
  | g6_open92 | FAIL 0/2 | 0.00 | 0.37 |
  | g6 | FAIL | 0.00 | 0.31 |
  Only the frame-0 training-style hold separates pass/fail (n=3 refs). To check on round-8 scratch grips.
- Round 9 launched: scratch_tr.py (search scored after training-style resets, PIB_TR=1), seeds 0/1 -> /oak/scratch/tr_s0, tr_s1
