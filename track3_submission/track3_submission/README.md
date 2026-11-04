# SRL — V2D Challenge Track 3 code submission (draft, 2026-11-05)

This folder is an **overlay for the team repository** `kyounginbaik/SRL_V2D`.
Nothing here has been committed, pushed or uploaded.

Path mapping:

| overlay path | server path |
|---|---|
| top level (`challenge_reconstruction/`, `third_party/`, `experiments/`) | `/data1/SRL_V2D/<same path>` |
| `external/hansol/<path>` | `/data1/hansol/<path>` |
| `external/hansol_home/<path>` | `/home/hansol/<path>` (node01) |

Scripts use these absolute paths; the container links them (see `container/`).

## The submitted file

`/data1/hansol/spoon3537/rt/packages/track3_FINALc_ep36s4/all_20_episodes/track3_submission.parquet`
- sha256 `9bc322af8242e7ee1ec71369d9e725ac737e4cce49920f6270e3a7aa113d16a1`
- CD-O file `a03230e5…`
- AUC submission 57997980 (0.92269), MPPE submission 57997979 (1.38958).

It was built in two steps:
1. `8b6fa2b9` = FINAL_20261104 (`0fca055b`) with:
   - ep32 rows from package `bd5b3321` (T32c_b1);
   - ep33 rows from package `c96244f0` (R6M2).
   - Combined by `external/hansol/teacher33/code/mk_ep32ep33.py`. It asserts that only e000032 and e000033 differ and that each equals its donor. Officially MP-SR 1.0, AUC 0.92086, MPPE 1.40441, RPE 1.50325.
2. `9bc322af` = 8b6fa2b9 with **only ep36** rebuilt from candidate `S3_ep36_s4` (checkpoint `5d5738b7`), by `external/hansol/spoon3537/rt/code/s3pkg.py`.
   - It verifies that only ep36's pose and checkpoint columns differ, then runs the kit offline validation and the CD-O derivation.

Per-episode lineage: `provenance/FINAL_9bc322af_episodes.tsv`. It gives the candidate, the work dir, `checkpoint_sha256`, and the absolute component and candidate dirs. The older tables `FINAL_8b6fa2b9_episodes.tsv` and `FINAL_0fca055b_episodes.tsv` are kept for reference.

## What is included

| Path | What |
|---|---|
| `third_party/DexAgent/` | DexAgent upstream `e29bdad` (MIT), **unchanged** (its `examples/` and `.git` left out) |
| `challenge_reconstruction/dexagent_final_recon_20261017/`, `dexagent_remaining7_20261019/` | **SRL modification of DexAgent**: floating Sharpa-hand port of its MPC runtime (`floating_runtime.py`, `floating_tracking.py`) and our grasp, contact, ILC, feedback and distillation code (`docs/DEXAGENT_MODIFICATIONS.md`) |
| `challenge_reconstruction/evaluator_build_v2_20261023/` | Common evaluator build v2 (eval_script `cbe866f9`, eval_code `fdddb348`, team convention) |
| `challenge_reconstruction/campaign_v16_20261023/` | Build-v2 candidate runner and the `completed.json` of every final candidate, including `T32_T32c_b1` |
| `challenge_reconstruction/campaign_v22_20261028/` | Retargeting and refinement campaigns (`stream*/code`). Includes ep32 `streamK32_ep32/recon_perfect` (K32_RP2 recon) and `streamT32_teacher` |
| `challenge_reconstruction/reconstruction_recovery_20261026/`, `recon_v14…v17/` | GT-free reconstruction streams behind the v14 target |
| `external/hansol/share/track3_FINAL_complete_set_20261105/` | README, INDEX and docs of the frozen complete set. ep32: `README_ep32_T32c_b1.md`, `recon/tools` (refine, merge, mk_target2), `retarget/code` (build_world_b, delta_port, fcem2/3, splice32, …), `retarget/notes`, `recon/marks_ep32.json` |
| `external/hansol/ep33_20261102/` | ep33 reconstruction: `code/` (spoon_annot_roll2, p33g/p33h builders, …), `orient/` (refit scripts, human mark files, `fit4/mk.sbatch` → r6), `agentmarks*/` (independent agent annotation: prompt, scripts, outputs) |
| `external/hansol_home/v2d/outputs/variants_20261030/` | ep33 base bundle records (`recon_E14_best_B2A_S1_HGA`: manifests, ep33 json including `ANNOT_MARKS_20261103.json`), smoothing and selection code (`A_work`, `B_work`, `B14_work`, `CD_work`) |
| `external/hansol/retarget33/`, `external/hansol/teacher33/` | ep33 target (`R33_HGA_r6`) and teacher chain: build_world_t33, delta_port_t33, fcem33, distill33, margins33, t33hg.sh, save33fz, t33svst, t33pkg, stage_a/b_t33, g18swap_t33, mk_ep32ep33. Steps and job ids: `teacher33/README.md`, `out/jobs.txt` |
| `external/hansol/spoon3537/` | **ep36 s4, used in the final file.** `code/` holds the spoon refits (fit_spoon, s4pos, s4fin, s5, spoon_annot_roll3, crops36, renders); `am36/` holds the agent marks; `rt/code/` holds the retarget chain and s3pkg; `rt/jobs.txt` lists the jobs. The rejected variants (ep37/B, ep35, ep3, ep36 s2/s3/s5/s6) are listed in `docs/TRIED_AND_REJECTED.md`. |
| `experiments/dexagent_egotools_10/` | Agentic DexAgent run on non-challenge clips; **no submitted row comes from it** |
| `track3_submission/` | `provenance/`, `container/` (submission.json, run.sh, Dockerfile, collect_weights.sh), `docs/`, `LICENSE` (Apache-2.0), `WRAPUP_KO.md` |

**Not included:**
- weights (20 components, about 1.7 GB; `container/collect_weights.sh`);
- npz, meshes, rollouts and videos;
- `.venv`, kit copies, and run outputs.

## Disclosure

- ep32 and ep33 use human per-frame spoon-bowl labels on the evaluation video as a cost term.
- ep36 uses agent labels, after the user flagged the window to fix from renders.
- Some choices were leaderboard-informed. See `docs/HUMAN_IN_THE_LOOP.md`.

## License

Our code here is released under Apache-2.0 (`track3_submission/LICENSE`). Third-party code keeps its license (`docs/NOTICE.md`).
