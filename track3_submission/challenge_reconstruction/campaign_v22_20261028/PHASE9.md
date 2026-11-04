# Phase 9 (2026-10-28): reach the top leaderboard level by fixing the reconstruction, then retargeting

Top score (richchentw): AUC 0.9327, MP-SR 1.0, SP-SR 1.0, RPE 1.29 cm, MPPE 1.20 cm. Ours (v22b official): 0.751 / 0.65 / 0.85 / 3.52 / 3.11.
Our final_v5 reconstruction on 20 public clips, scored kinematically with the kit: MPPE 1.17, AUC 0.931, 20/20 (streamR8_recon r8_base).
So the official gap comes from gross failures that occur only in eval episodes. Candidates:
- spoon (eps 3, 5, 32, 33, 35, 36, 37): no public scan;
- half-turn twin registration of the sweeper / wooden pieces (eps 19, 25, 30);
- ep32 anchor frame 0 (5.9 deg outlier);
- mesh size bias (rack 43-46 vs about 51 cm; ep32 blue cup 15.6 vs about 17.4 cm).
Official localisation: the user uploads v23b plus the group knockouts in challenge_submissions_SRL/ablation_knockout_v23b_groups_20261028.

## Rules (user, 2026-10-28: "we shouldn't use public mesh")
- NO public scans or GT meshes in any submitted mesh or in reconstruction preparation (track_3/public/mesh, gt_best_per_object, scan-derived). Public GT and scans may be used ONLY inside the scorer, for validation.
- Every fix must be validated on public 10 + held-out 10 with the kit scorer (as in streamR8_recon) and be no worse there.
- Never overwrite selected20_final_v5 or other earlier outputs; write new exports / bundles. No uploads, no git commits.
- Slurm partition debug; cpu/eval node01; render node100; jobs <= 50 min. Never touch other users' jobs or the user's other run (B2-*, B8-*, E8s-*, H8s19-*, S8s-*, v23D-*, ur-*, ego-retarget).
- Status lines go to STATUS.md ("- 2026-10-28 HH:MM <stream> ..."); README.md per stream dir.
