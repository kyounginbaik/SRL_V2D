# Phase 8 (2026-10-28): v23 — tighten every episode against our reconstruction, and harden the reconstruction

User (after the official v22b result): "improve it and make it all perfect for sr 1.0 auc almost 1.0 and rpe or mppe lower."

Official v22b (hidden GT): MP-SR 0.65, SP-SR 0.85, AUC 0.751, RPE 3.52 cm, MPPE 3.11 cm, CD-O 0.276.
Local proxy of the same package vs target_final_v5 (our reconstruction): MP-SR 1.0, AUC 0.867, MPPE 1.77 cm.
So the hidden-GT error ~= rollout-vs-reconstruction error (+) reconstruction-vs-GT error. Our reconstruction is ~1.1 cm / 5 deg
off public GT on average (public 10 + held-out 10). Rollouts that sit 2-2.4 cm off the reconstruction leave no room for that.

## Goal of the rollout streams
Every object as close to target_final_v5 as possible: aim <= 1.2 cm and <= 12 deg mean, per object; never worse than the v22b
component. Lower error everywhere also lowers MPPE / RPE and raises AUC (AUC integrates the per-frame error curve).
Same recipe, scorer, rules and verification as PHASE7.md (read it). v22b components are the starting point:
selection = /home/exaFLOPs/v22_overflow_20261028/v22b/selection_v22b.json; their per-object scores =
/data1/SRL_V2D/challenge_submissions_SRL/track3_v22b_20261028/all_20_episodes/target_final_v5_scores.json.
Also try to keep per-frame error low in every frame, not only the mean (RPE and AUC see the whole curve).

Techniques that worked in phase 7 (read the stream READMEs; streamH/I/G live in /home/exaFLOPs/v22_overflow_20261028):
- late-window ILC on the actor's own right-wrist phase weights, scored on its closed-loop rollout (H ep32, actor_res.py);
- registered ReleasePolicy / MultiReleasePolicy primitives fitted by closed-loop CEM for place/release (I ep33);
- knot CEM over time-varying wrist offsets with a scorer-faithful cost, then distill (G ep10, kcem.py);
- target-following fcem + finger margins (B), obj-mode port (A), achieved-plan mode (D).
Chaotic in-hand holds (spoon power grasps) resist perturbation; prefer corrections in benign windows and primitives over re-distills.

## Rules (unchanged)
Our meshes only (the meshes v22b ships; no gt_best_per_object, no scan-derived meshes). Poses only from policy rollouts under
evaluator build v2; never edit poses by hand. Public GT is never read by rollout streams. Partition debug; distill/stage/eval
--nodelist=node01 (AMD); renders node100; jobs <= 50 min. Never touch other jobs (ur-*, ego-retarget, other streams). Never delete
or overwrite earlier outputs. No uploads, no git commits. Tags: <stream letter>8_epNN_<name> in campaign_v16_20261023/common_v2_candidates
(write there; /data1 has space again). Stream dirs: campaign_v22_20261028/stream<X>8_<eps>/. Status lines to
campaign_v22_20261028/STATUS.md ("- 2026-10-28 HH:MM <X>8 epNN: ..."), README.md per stream with a results table.

## Update (user, 2026-10-28): position goal relaxed to <= 1.3 cm per object (orientation goal stays <= 12 deg).
## Update 2 (user, 2026-10-28): orientation goal relaxed to <= 15 deg (goal now <= 1.3 cm and <= 15 deg per object).
