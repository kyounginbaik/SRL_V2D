# Track 3 final candidate: complete set (2026-11-05, Hansol)

Submission (`submission/`): FINAL_20261104 + ep32 T32c_b1 + ep33 R6M2. sha256 8b6fa2b9a0513b8b2f14176f51b92ebf02ecde1eccef932b641b4b6dc9960907 (CD-O a03230e5d6ecd598).
Kit offline validation OK; provenance uniform over 20 episodes; the ep32-only and ep33-only packages each scored MP-SR 0.95.

Per episode (`episodes/epXX/`):
- `recon/`: the reconstruction the robot was retargeted to (object poses, meshes, target trajectories).
- `retarget/`: for the 18 FINAL episodes, the team component (portable actor, lineage code, original/validation rollouts, results).
  ep32: teacher, actor, beam table, build-v2 candidate, code and notes from the teammate handoff.
  ep33: moved-start world, new teacher, distilled+margins actor, beam tables A/B/C, saved actor, build-v2 candidate, perturbation, 0.5 s table.
- `review/` (18 episodes): 4-panel H.264 video (RGB | recon + SAM2 outline | robot objects + recon outline | MuJoCo rollout) and per-frame / 0.5 s diagnostics. `videos/` for ep32 and ep33.
- `docs/`: episode review report (worst windows, top-5 fixes), ep33 agent-vs-human spoon marks, submission issues (KR/EN), FINAL lineage table.

| ep | final source | note |
|---|---|---|
| 1 | A20 donor: A20_ep1_b1 (streamALL20K) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 3 | A20 donor: A20_ep3_b2 (streamALL20K) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 5 | A20 donor: A20_ep5_b2 (streamALL20K) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 6 | base v29/v30c1: G18_ep06_H13b1 (streamG18v29) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 8 | base v29/v30c1: H13_ep08_a2 (streamH13b) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 10 | A20 donor: A20_ep10_b1 (streamALL20K) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 14 | base v29/v30c1: H13_ep14_t1p15 (streamH13b) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 15 | A20 donor: A20_ep15_b1 (streamALL20K) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 16 | anchor donor: A20_ep16_anc1 (streamANCHOR (staged by streamALL20K)) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 19 | A20 donor: A20_ep19_b2 (streamALL20K) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 20 | base v29/v30c1: H13_ep20_b2 (streamH13a) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 25 | base v29/v30c1: D9_ep25_b4 (streamD9_tight) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 28 | A20 donor: A20_ep28_b2 (streamALL20K) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 30 | A20 donor: A20_ep30_b1 (streamALL20K) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 32 | T32c_b1 (team; replaces FINAL K32_R10) | MP-SR passes alone (0.95); recon uses human bowl marks (marks_ep32.json) |
| 33 | R6M2 (ours; replaces FINAL K33_H5r16_c3) | MP-SR passes alone (0.95); recon uses human bowl marks (docs/EP33_AGENT_SPOON_MARKS.md); right hand releases the spoon about 0.5 s early |
| 35 | K35 donor: K35_K35_H2joint_b1 (streamK35_ep35) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 36 | A20 donor: A20_ep36_b2 (streamALL20K) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 37 | A20 donor: A20_ep37_b1 (streamALL20K) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |
| 38 | A20 donor: A20_ep38_b1 (streamALL20K) | unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md |

Rules: no GT used for reconstruction or retargeting; all compute in Slurm on node01/node100; nothing uploaded from this set.
