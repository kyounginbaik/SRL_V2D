# Five-episode refinement review

Updated 2026-10-17T14:55:53.682122+00:00

**Current block-polish result: 1/5 reconstructed-reference MP-SR passes (episode 8).** Pilot baseline and pilot MPC each pass 0/5; CEM passes 1/5. These are proxy scores for five requested episodes, not official hidden-ground-truth results. The outputs are not submission-ready learned policies.

Each cell below is full-horizon mean position error in cm / symmetry-aware orientation error in degrees. Every object must be strictly below 3 cm / 30° for the episode to pass.

| Episode | Object | Pilot baseline | CEM | Block polish | Block object pass |
|---|---|---:|---:|---:|:---:|
| 000001 | white_pot | 6.74 / 12.15 | 10.59 / 60.44 | 7.94 / 54.85 | no |
| 000001 | white_pot_lid | 12.79 / 33.09 | 7.06 / 73.39 | 6.81 / 68.08 | no |
| 000003 | white_pot | 4.57 / 28.31 | 1.49 / 10.28 | 0.98 / 8.52 | yes |
| 000003 | wooden_spoon | 13.40 / 38.99 | 12.93 / 26.27 | 12.92 / 26.36 | no |
| 000005 | white_pot | 8.47 / 33.13 | 4.12 / 39.70 | 3.61 / 35.78 | no |
| 000005 | wooden_spoon | 16.05 / 42.98 | 4.62 / 49.76 | 2.95 / 42.99 | no |
| 000008 | beige_cup | 10.19 / 32.20 | 2.67 / 9.97 | 2.34 / 8.79 | yes |
| 000008 | white_pot | 3.74 / 12.22 | 1.91 / 10.88 | 1.86 / 11.09 | yes |
| 000010 | blue_cup | 657.36 / 39.16 | 11.17 / 72.61 | 8.38 / 66.36 | no |
| 000010 | beige_cup | 641.18 / 41.28 | 5.06 / 21.22 | 8.69 / 27.79 | no |

Actual grasp observations from aligned videos and recorded contact audits:

- **Episode 1:** the pot handle is missed by the left hand; the right hand tips the pot while attempting the lid. Neither intended grasp is retained. A better scalar search cost does not solve this episode.
- **Episode 3:** the pot lift is retained from 3.3–11.4 s, with contact in all sampled frames and only 1.76 mm palm-relative translation drift. The spoon never lifts, so the episode still fails.
- **Episode 5:** both objects are picked up and now return to the table. During a 3.7–5.75 s unsupported interval, pot contact is continuous but palm-relative translation changes 4.49 cm; spoon contact is about 90%, with 2.68 cm / 58.38° relative motion. Pot tilt and spoon rotation remain wrong.
- **Episode 8:** the cup is held during a 3.75–8.85 s unsupported interval with continuous sampled contact; its maximum rise is 18.88 cm. The pot rises only 1.77 cm versus a 7.32 cm reference maximum. The mean-error metric passes even though peak motion is not fully reproduced.
- **Episode 10:** the right hand lifts the beige cup temporarily, but the blue cup tips and nesting fails. The separate original-left/CEM-right combination also fails and finishes with toppled cups.

Contact percentages above are sampled geometry, not measured force closure. Relative object-to-palm motion includes slip, rolling, regrasp, and finger articulation.

| Episode | CEM cost: initial → best | CEM minutes | Block cost: initial → best | Block minutes | Replay/audit | Video |
|---|---:|---:|---:|---:|---|---|
| 000001 | 5.041 → 4.277 | 28.4 | 4.277 → 3.261 | 15.8 | passes | [block](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/block_polish/episode_000001/videos/reference_baseline_refined.mp4), [CEM](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_v2/episode_000001/videos/reference_baseline_refined.mp4) |
| 000003 | 5.090 → 4.910 | 15.7 | 4.910 → 4.885 | 10.0 | passes | [block](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/block_polish/episode_000003/videos/reference_baseline_refined.mp4), [CEM](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_v2/episode_000003/videos/reference_baseline_refined.mp4) |
| 000005 | 6.345 → 2.038 | 16.6 | 2.038 → 1.762 | 12.4 | passes | [block](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/block_polish/episode_000005/videos/reference_baseline_refined.mp4), [CEM](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_v2/episode_000005/videos/reference_baseline_refined.mp4) |
| 000008 | 3.419 → 1.080 | 7.6 | 1.079 → 0.955 | 1.8 | passes | [block](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/block_polish/episode_000008/videos/reference_baseline_refined.mp4), [CEM](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_v2/episode_000008/videos/reference_baseline_refined.mp4) |
| 000010 | 8.951 → 4.401 | 10.6 | 4.401 → 3.609 | 4.9 | passes | [block](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/block_polish/episode_000010/videos/reference_baseline_refined.mp4), [CEM](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_v2/episode_000010/videos/reference_baseline_refined.mp4) |

Times are per-run wall times; parallel jobs overlap. Pilot-to-CEM also changes the controller and starting commands, so it is not a controlled search-only ablation.

Original-reference collision diagnostics use the unchanged object poses and the exact experiment colliders. Values are maximum / all-frame mean of the maximum penetration at each original timestamp, in mm.

| Episode | Object–object | Object–table |
|---|---:|---:|
| 000001 | 32.08 / 7.39 | 1.85 / 0.02 |
| 000003 | 9.50 / 0.09 | 27.95 / 2.86 |
| 000005 | 12.07 / 0.31 | 11.30 / 0.53 |
| 000008 | 0.00 / 0.00 | 12.28 / 2.52 |
| 000010 | 21.20 / 4.56 | 24.56 / 4.91 |

These overlaps diagnose incompatibility with the reconstructed collision pieces and inferred table, not proven impossibility of passing a mean-error threshold. No reference object poses or evaluation targets were changed.

Episode 5 gap-fill counterfactual: changing only the left contact envelope improves the pot to 2.62 cm / 29.49° but causes the spoon to fall (14.64 m mean position error). Right thumb-plus-finger contact ends at 7.182 s and all right contact at 7.472 s; continuous pot–spoon contact starts at 7.474 s. Earlier coupled contacts increase, but the evidence does not prove a single collision caused release.

Episode 10 cross-sections show hollow collision interiors, with typical inward intrusion of a few millimeters. The visual meshes themselves also intersect at the 2.8 s insertion and final 8.0 s nesting, so hull repair alone cannot make the exact reference collision-free. [Cross-sections](reference_collision_audit/episode10_cavities/reference_nesting_sections.png) and [cavity diagnostic](reference_collision_audit/episode10_cavities/individual_cup_sections.png).

Completed later trials are listed separately below. Each family/episode row is its lowest normalized error score among complete recorded rollouts; a reported metric without independent audit is explicitly marked. These trials do not overwrite the frozen comparison above.

Episode 5 continuation seed46 lowers cost but leaves the spoon off to the right at 14 s instead of returning it to the table. It has more spoon contact but worse spoon tracking and palm-relative drift, so the original block result remains the conservative handoff. Seeds47/48 retain the block result exactly. Episode 3 feedback improves pot pose while retaining it; its 3.77 mm relative translation drift is slightly larger than the block result's 1.76 mm. Cup acquisition and lid-retention trials add no new proxy passes.

The episode 3 C2 transport blends (0.4/0.8/1.2 s) preserve identical acquisition prefixes and remove the abrupt velocity transition, but spoon errors remain 12.92–12.95 cm. They do not establish a successful retained grasp and are not selected.

Final trials also add no passes. Extended cup acquisition deadlines fail. Observed-pickup retention also fails despite zero measured qpos discrepancy over its original pickup prefix. The final atomic lid controller detects 4.52 N finger contact at 1.492 s and switches to thumb closing in that same update, but still reports failed acquisition at 2.302 s.

The final geometry-qualified spoon neck/bowl poses also fail acquisition: spoon error is 13.21 cm / 28.21° and 13.56 cm / 37.43°, while the unchanged left-pot commands preserve approximately 1.49 cm / 10.28°. Prepositioning the lid thumb closer does not establish capture either: the two trials have lid errors 12.37 cm / 41.67° and 28.62 cm / 46.26°. All four complete the original horizon and pass exact recorded-control replay; none is promoted to the selected handoff.

Correction to the finger diagnosis: at the captured spoon posture, the contact Jacobian shows that decreasing the index DIP target closes the contact. The negative target-minus-measured angle is not an opening command. Resetting the target to the measured angle removes contact preload. The 0/1/2 mm measured-finger preload tests nevertheless all fail, with spoon position error about 13.01–13.02 cm; this is not established as the sole cause of failure.

| Episode | Later trial | Per-object cm / degrees | Proxy pass | Independent audit | Video |
|---|---|---|:---:|---|---|
| 000001 | grasp_transfer/anchor4.0_blend0.6 | white_pot: 11.70 / 74.58; white_pot_lid: 9.99 / 72.07 | no | reported; pending audit | pending |
| 000001 | lid_collision_feedback/candidate_03_open6 | white_pot: 7.05 / 18.40; white_pot_lid: 12.68 / 40.44 | no | reported; pending audit | pending |
| 000001 | lid_contact_seeking_atomic/candidate_00_open10 | white_pot: 7.20 / 18.45; white_pot_lid: 12.74 / 40.25 | no | reported; pending audit | pending |
| 000001 | lid_retention/contact_only_gate | white_pot: 6.19 / 19.98; white_pot_lid: 12.41 / 41.76 | no | reported; pending audit | pending |
| 000001 | lid_thumb_preposition/thumb_open2 | white_pot: 7.25 / 19.79; white_pot_lid: 12.37 / 41.67 | no | reported; pending audit | pending |
| 000003 | contact_retention_fingers/preload_02 | white_pot: 1.49 / 10.28; wooden_spoon: 13.01 / 26.79 | no | reported; pending audit | pending |
| 000003 | contact_retention_measured/geometry | white_pot: 1.49 / 10.28; wooden_spoon: 12.93 / 29.26 | no | reported; pending audit | pending |
| 000003 | contact_retention_smooth/blend_0800 | white_pot: 1.49 / 10.28; wooden_spoon: 12.92 / 29.46 | no | reported; pending audit | pending |
| 000003 | feedback_polish/both_1.0 | white_pot: 0.54 / 3.72; wooden_spoon: 12.93 / 26.27 | no | recomputed | [video](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/feedback_polish/episode_000003/both_1.0/videos/reference_baseline_refined.mp4) |
| 000003 | grasp_transfer/anchor4.3_blend0.3 | white_pot: 0.98 / 8.53; wooden_spoon: 12.97 / 25.33 | no | reported; pending audit | pending |
| 000003 | spoon_collision_feedback/pose_00 | white_pot: 1.49 / 10.28; wooden_spoon: 13.21 / 28.21 | no | reported; pending audit | pending |
| 000005 | block_continue_s46/episode_000005 | white_pot: 2.92 / 34.88; wooden_spoon: 3.79 / 41.46 | no | recomputed | [video](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/block_continue_s46/episode_000005/videos/reference_baseline_refined.mp4) |
| 000005 | block_continue_s47/episode_000005 | white_pot: 3.61 / 35.78; wooden_spoon: 2.95 / 42.99 | no | reported; pending audit | pending |
| 000005 | block_continue_s48/episode_000005 | white_pot: 3.61 / 35.78; wooden_spoon: 2.95 / 42.99 | no | reported; pending audit | pending |
| 000005 | feedback_polish/both_1.0 | white_pot: 2.74 / 32.06; wooden_spoon: 10.41 / 66.94 | no | reported; pending audit | pending |
| 000005 | grasp_transfer/anchor6.0_blend0.6 | white_pot: 51.99 / 26.13; wooden_spoon: 4.79 / 61.58 | no | reported; pending audit | pending |
| 000008 | feedback_polish/both_0.5 | beige_cup: 1.92 / 7.95; white_pot: 2.22 / 11.92 | yes | recomputed | [video](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/feedback_polish/episode_000008/both_0.5/videos/reference_baseline_refined.mp4) |
| 000008 | grasp_transfer/anchor6.0_blend0.3 | beige_cup: 3.52 / 11.80; white_pot: 1.33 / 6.92 | no | reported; pending audit | pending |
| 000010 | cup_feedback/a2.5_clear0.000_pre0.04 | blue_cup: 9.00 / 35.52; beige_cup: 18.55 / 47.52 | no | reported; pending audit | pending |
| 000010 | cup_observed_retention/a2.2_clear0.015 | blue_cup: 6.12 / 35.07; beige_cup: 6.66 / 48.58 | no | reported; pending audit | pending |
| 000010 | feedback_polish/both_1.0 | blue_cup: 11.90 / 72.75; beige_cup: 114.18 / 41.17 | no | reported; pending audit | pending |
| 000010 | grasp_transfer/anchor2.5_blend0.3 | blue_cup: 14.49 / 73.00; beige_cup: 15.26 / 37.08 | no | reported; pending audit | pending |
| 000010 | hand_combinations/original_left_cem_right | blue_cup: 11.96 / 37.11; beige_cup: 15.48 / 57.68 | no | recomputed | [video](/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/hand_combinations/episode_000010/original_left_cem_right/videos/reference_baseline_refined.mp4) |

Run `code/summarize_refinement_audits.py` with the workspace virtualenv to refresh the JSON and tables. Detailed per-object contact windows, immutable input hashes, replay checks and video paths are recorded in `metrics_review.json` and the linked audit files.
