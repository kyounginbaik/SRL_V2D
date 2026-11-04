# Campaign v16, phase 2 (2026-10-23 evening): shrink the robot-side error on the borderline episodes

v16 is assembled (challenge_submissions/track3_v16_20261023/all_20_episodes_post, SRL copy built, NOT uploaded).
Honest expectation: v16 will not pass all 20 on hidden GT. Official MPPE is 5.1 cm while the robot-vs-our-reference error is
below 2 cm everywhere, so the failing episodes lose mostly on reconstruction-vs-GT error, which we cannot measure for the spoon.
What we CAN still control is the robot-vs-reference term. Where that term is near 2-3 cm it adds to an unknown recon error and
almost surely fails; getting it to ~1.2 cm / 12 deg is the only lever left that does not need leaderboard feedback.

Phase-2 targets (v16 local error cm / deg vs our reference; component dirs in common_v2_baseline/components.txt):

| stream | episode | objects (v16) | target | starting point |
|---|---|---|---|---|
| E | 15 | rack 0.87/4.1, spoon 2.93/26.5 | spoon <= 1.8 / 15 | dexagent_remaining7_20261019/components/episode_000015 (teacher + actor), rackscan_20261021/ep15_v8 (rack refit); earlier 3 placements x 8 retention re-plans failed; spoon slips in the fast in-hand turn |
| F | 25 | piece_1 2.49/25.5, piece_2 2.44/14.8 | both <= 1.8 / 15 | dexagent_remaining7_20261019/ep25_v15 (edge push + corner grasps, slot collision cut); Stream C ILC did not help |
| G | 36 | spoon 2.13/16.5 | <= 1.6 / 12 | dexagent_final_recon_20261017/refinement_20261021_ep36 (staging_v13); v10 roll-flip track correction 12.55-14.65 s; v13 finger margins |
| G | 30 | sweeper 1.89/17.9, rack 0.66/3.3 | sweeper <= 1.4 / 12 | rackscan_20261021/ep30_v9; Stream C gentle ILC never beat the teacher (best 2.03/16.6) -> re-plan the teacher (retention/release), not ILC |

Tooling that worked in phase 1 (read the stream READMEs before starting):
- streamA_spoon/README.md, streamA2_crotch/README.md: spoon pickup CEM searches, retention searches, release re-plans, gentle wrist ILC
  (iter1 only; iter2+ diverges on pinches), distill grid (feedback gain 0.01/0.03/0.1, centers 512/1024/2048), wrapped vs plain actors.
- streamC_rigid_ilc/ep19, ep28: ILC on rigid grasps, distill.sbatch / stage.sbatch patterns (stage with
  dexagent_final_recon_20261017/code/stage_pot_component.py for final_recon episodes, or
  dexagent_remaining7_20261019/package_code/stage_seven_component.py with an actor_report.json for remaining7 episodes).
- streamC_rigid_ilc/ep19: the shipped actor is a plain learned_feedback_policy.LearnedPolicy; re-applying old pinky-margin wrapper
  keys dropped the dust pan in all 16 actors. Prefer plain actors; if margins are needed, fit them fresh (refinement_20261021_ep33).
- After staging: verify under evaluator build v2 with
  `sbatch /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/run_one.sbatch EP COMPONENT_DIR TAG`
  (writes common_v2_candidates/TAG/episode_XXXXXX with completed.json; must be exact).
- Physics replays are not bitwise reproducible across nodes: pin every distill / stage / evaluation job to node01
  (`#SBATCH --nodelist=node01`); searches may run anywhere on partition debug.

Deliverable per stream: a staged component (components/episode_XXXXXX with staged.json, portable archive) that beats the v16 numbers,
verified under build v2, plus a README in the stream dir and STATUS.md lines. If the target is not reached, report the best
achieved numbers and why, and still stage the best actor if it improves on v16.

Hard rules (unchanged, user): never upload; no DexAgent mention inside challenge_submissions_SRL; no scanned meshes
(gt_best_per_object) in new work; poses only from policy rollouts under the frozen evaluator; never edit poses by hand; partition
debug; video renders only with --nodelist=node100; never cancel other people's jobs (ur-tl-* are the user's); never delete or
overwrite existing outputs; write only new directories under this campaign folder (streamE_ep15, streamF_ep25, streamG_ep36_30).
