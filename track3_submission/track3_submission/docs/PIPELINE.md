# Pipeline, mapped onto the official Track 3 starter toolkit

Reference: NVIDIA `video_to_data` repository.
- `upstream/main` = `9cad120d`: `docs/e2e_workflow.md` and `run_e2e.sh`, the reconstruction-to-policy E2E workflow.
- The V2D submission kit (same commit): `tracks/track_3`, `tools/pack_submission.py`, `submit/v2d_submit.py`, `CONTAINER_SUBMISSION.md`.

| Starter toolkit stage (`docs/e2e_workflow.md`) | SRL implementation | Where |
|---|---|---|
| Reconstruct a raw video (Route A) / import a prepared reconstruction (Route B) | **Route B.** Our own GT-free reconstruction streams. Fused trackers, despike, lost-object rescue, spoon roll from marker balls (G10/G16), frame-0 anchor consensus (Z9), placement (G17) and spoon mesh (G18). Coworker meshes for CD-O. | `challenge_reconstruction/reconstruction_recovery_20261026/stream*/code`, `recon_v14…v17_*/`, `campaign_v22_20261028/streamE31_cowork2`, `streamA9_meshswap` |
| Retarget the task to an embodiment | Floating Sharpa hands (allowed embodiment, FAQ). DexAgent MPC runtime ported to floating hands. | `challenge_reconstruction/dexagent_final_recon_20261017/code/floating_runtime.py`, `floating_tracking.py`, `wrist_reference.py` |
| Train and validate the source expert | Per-episode learned actor (RBF phase policy) with ILC / feedback / distillation. Refined against our own reconstruction target only. | `dexagent_final_recon_20261017/code/{distill_teacher,ilc_carry,ilc_reference,learned_feedback_policy,...}.py`, `campaign_v22_20261028/streamT32_teacher` |
| (refinement) | Chained window-beam search on phase residuals (`wbeamH.py`), rolled spoon start (`--roll`), gate `a20select.py`. | `campaign_v22_20261028/streamALL20K/code`, `streamK32_ep32`, `streamK33_ep33`, `streamK35_ep35`, `streamANCHOR` |
| Evaluate in closed loop | Common evaluator build v2. One fresh physics rollout per checkpoint; no reuse of earlier rollouts. | `challenge_reconstruction/evaluator_build_v2_20261023/package_code/common_evaluator.py`, `campaign_v16_20261023/common_v2_candidates/run_one.sbatch` |
| Pack and submit | Donor packages per episode → `v32build.py` → kit packer/uploader (hashes unchanged from the kit: packer `49b13052`, uploader `d31d4a7c`) | `campaign_v22_20261028/streamALL20K/code/{a20splice,v32build}.py`, `challenge_submissions_SRL/FINAL_20261104/` |

## FINAL 0fca055b lineage

- Base: `track3_v30c1_cwmesh_nogt_20261101`, which is v29 plus coworker GT-free meshes.
- 15 episodes are replaced from donors:
  - ALL20-K round 1: episodes 1, 3, 5, 10, 15, 19, 28, 30, 36, 37, 38;
  - ep16 rack de-tilt (anchor);
  - ep32 start roll −10 (K32_R10_c1);
  - ep33 start roll −16 (K33_H5r16_c3);
  - ep35 start roll −11 (K35_H2joint_b1).
- Episodes 6, 8, 14, 20 and 25 come from the base.

The per-episode candidate, stream, checkpoint sha256 and component directory are in `../provenance/FINAL_0fca055b_episodes.tsv`.

## Final file 8b6fa2b9 = FINAL + ep32 T32c_b1 + ep33 R6M2

**ep32 (T32c_b1, checkpoint `a5a181f8`)**
- Recon: `streamK32_ep32/recon_perfect` and `external/hansol/share/.../ep32/recon/tools`.
  - `refine.py` / `refine_cup.py` → `refine2.py` → `merge.py` (with the human marks) → `merge_cup2.py` → `mk_target2.py` → target `K32_RP2`.
- Retarget: `streamT32_teacher` and `.../ep32/retarget/code`.
  - World: `build_world_b.py` → `W_RP2`.
  - Teacher: `delta_port.py` → `fcem2.py` (both wrists) → distill (`actor_c1024_p0.01`).
  - Beam: `k32rchain_t32.sh` (team wbeamH).
  - Finish: `t32_finish.sh` (`mresH save` → `stage_b.py` → `run_one.sbatch` → `splice32.py`).
- Candidate `campaign_v16_20261023/common_v2_candidates/T32_T32c_b1`.

**ep33 (R6M2, checkpoint `c4996715`)**
- Recon:
  - Base bundle `/home/hansol/v2d/outputs/variants_20261030/recon_E14_best_B2A_S1_HGA`, which carries the 5–7 s refit p33g with the human marks (`spoon_annot_roll2.py`, `orient/refit*.sbatch`).
  - `orient/fit4/mk.sbatch`: start roll −6° (`ep33_rollcurve_target.py`, p33h) → `recon_E14_best_B2A_S1_HGAZr6` → `retarget33/code/cwtgt33.py` → target `R33_HGA_r6`.
- Teacher chain (`teacher33/README.md` §25–31):
  1. Moved-start world `worlds/R33r6M` (`build_world_t33.py`).
  2. `delta_port_t33.py` → teacher `fcem33.py FN_r6M`.
  3. `distill33.sbatch` DR6M → `margins33auto.sbatch` DR6Mm1.
  4. Two-hand beams `t33hg.sh` (OW spoon 3 / cup 2) R6M2_A/B/C (jobs 69712–69714).
  5. `save33fz.sbatch` → `runs/save_R6M2_C_f0` → `t33svst.sbatch` (`stage_a_t33.sh`, `stage_b_t33.py`, `g18swap_t33.py`, `t33run_one.sh`) → candidate `R33_HGA_r6_R6M2_C_f0` → `t33pkg.sbatch` → package `c96244f0`.
- vs r6: cup 0.645 cm / 3.44°, spoon 0.832 cm / 7.79°; build-v2 exact.

**Combine:** `teacher33/code/mk_ep32ep33.py` → `8b6fa2b9`.

## Final file 9bc322af = 8b6fa2b9 + ep36 s4

**ep36 (s4, checkpoint `5d5738b7`)**, all under `external/hansol/spoon3537`:
- Reconstruction:
  - `code/fit_spoon.py` (2–10 s, RGB-camera SAM2 fit) → `tgt_s1`;
  - `code/s4pos.py`, `s4fin.py` and `s5.py` (12.5–17.5 s, depth-locked position, f297–308 jump fix);
  - `code/spoon_annot_roll3.py` with the agent marks `am36/agent_marks36.json` → `tgt_s4`.
- Retarget (`rt/code`, team A20 chain):
  - FINAL ep36 actor and table;
  - beam s1a 1.75–10.5 s → s1d 10.5–20.25 s → s4 12.5–20.25 s from `s1d/beam_w03.npz`;
  - `s3svst.sbatch` (`stage_s3.sh` / `stage_b_s3.sh`, `g18swap_s3.py`, `s3run_one.sh` build-v2) → candidate `rt/candidates/S3_ep36_s4`.
- Package: `rt/code/s3pkg.py 36 S3_ep36_s4 track3_FINALc_ep36s4` → `9bc322af`. Job ids are in `rt/jobs.txt` (s4: 70188–70190).

Reproduce one episode on the cluster (inside Slurm; this writes under `common_v2_candidates/<TAG>`):

```bash
sbatch challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/run_one.sbatch EP COMPONENT_DIR TAG
```
