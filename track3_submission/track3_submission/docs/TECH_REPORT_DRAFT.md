# SRL — V2D Challenge Track 3 (egocentric, end to end): technical report (draft)

Format, length and deadline have not been published by the organizers.
Sections 6–8 follow the requested items: tools and roles, resource usage, automation and human involvement, and keys to effectiveness.
Sources:
- `campaign_v22_20261028/STATUS.md` and `challenge_submissions_SRL/TRACK3_END_TO_END_SUMMARY_20261102.md`;
- `external/hansol/teacher33/README.md` and `external/hansol/share/.../episodes/ep32/README_ep32_T32c_b1.md`;
- `docs/HUMAN_IN_THE_LOOP.md`.

## 1. Overview
- Input: 20 egocentric stereo videos.
- Output: per-frame object trajectories, achieved by a learned policy in MuJoCo physics, plus reconstructed meshes.
- Embodiment: floating Sharpa hands.
- Final file `9bc322af` = FINAL `0fca055b` with ep32 (T32c_b1), ep33 (R6M2) and ep36 (s4) replaced.

| file | MP-SR | AUC | MPPE (cm) | RPE (cm) |
|---|---|---|---|---|
| FINAL 0fca055b | 0.90 | 0.90306 | 1.561 | 1.619 |
| 8b6fa2b9 (+ ep32, ep33) | **1.0** | 0.92086 | 1.40441 | 1.50325 |
| **9bc322af (+ ep36 s4)** | expected 1.0 (TODO) | **0.92269** | **1.38958** | unchanged (ep36 is single-object) |

TODO: SP-SR and CD-O.

## 2. GT-free reconstruction
- Fused object trackers with despike and lost-object rescue.
- Spoon roll from marker balls (G10, G16).
- Frame-0 anchor consensus (Z9), placement (G17), spoon mesh (G18).
- Coworker GT-free meshes for CD-O.
- The fixed policy gives 10/10 on public and 10/10 on held-out public.
- **ep32 (K32_RP2):** per-frame re-solve against masks, stereo, marker balls and point tracks, plus the human bowl labels (§7).
- **ep33 (r6):** roll refit with the human bowl labels (§7), then a start roll of −6°.
- **ep36 (s4):** spoon refit in two parts.
  - 2–10 s: per-frame spoon fit to the SAM2 mask in the RGB camera.
  - 12.5–17.5 s: depth-locked position plus a roll refit with **agent** bowl labels.
  - On the leaderboard this improved MPPE and AUC, even though our stereo-side checks disagreed with the RGB-camera fit (`TRIED_AND_REJECTED.md`).

## 3. Retargeting and policy learning (DexAgent port)
- DexAgent predictive-sampling MPC is ported to floating hands (`floating_runtime.py`).
- Per-episode learned RBF phase actors are trained with ILC, feedback and teacher → student distillation.
- Chained window-beam refinement on phase residuals runs against our own reconstruction target only.
- **ep32:** new teacher (`fcem2`, both wrists follow their objects) → distillation → window beam → T32c_b1.
- **ep33:** world with a moved start (r6 frame-0 poses) → teacher `FN_r6M` (fcem33) → distillation and margins (DR6Mm1) → two-hand window beams R6M2 A/B/C → saved actor → candidate. vs r6: cup 0.645 cm / 3.44°, spoon 0.832 cm / 7.79°; perturbation test 12/16 pass.

## 4. Evaluation and packaging
- Evaluator build v2 runs a fresh physics rollout per checkpoint, with frozen backends and the provenance columns.
- `v32build.py` builds FINAL. `mk_ep32ep33.py` then replaces the ep32/ep33 rows from their packages.
- Kit packer, `validate_submission` and uploader complete the packaging.

## 5. Results and ablations
- History table and the what-worked / did-not-work lists from the summary.
- Rejected attempts are in `docs/TRIED_AND_REJECTED.md`:
  - B (s4 + ep37): MPPE 1.39543, worse than s4;
  - ep35, ep3 fix and e3s;
  - ep36 s1d, s2, s3, s5, the s6 human-hint variants, and A.
- The same file holds the RGB-vs-stereo camera finding.

## 6. Tools and roles
| who / what | role |
|---|---|
| Claude Code (agent sessions, several in parallel) | wrote and ran the reconstruction streams, the DexAgent port and actors, window beams, packaging, audits and videos; kept logs (`STATUS.md`, READMEs) |
| Claude subagents | independent per-frame spoon-bowl annotation: for ep32/ep33 as a comparison only; for **ep36 the labels used in s4** |
| DexAgent (MIT, e29bdad) | MPC runtime and reference interface, ported; its agentic stages were evaluated separately (`experiments/dexagent_egotools_10`) |
| FoundationPose-based tracking, SAM2 masks, stereo depth, CoTracker | perception for reconstruction |
| Team members | direction and decisions, video review, ep32/ep33 spoon labels, Kaggle uploads |

## 7. Automation and human involvement
- Automated: reconstruction, retargeting, distillation, beam search, evaluation, packaging and validation.
- Human:
  - per-frame spoon-bowl labels on the evaluation video in ep32 (58 frames) and ep33 (78–92 frames), used as a cost term (`docs/HUMAN_IN_THE_LOOP.md`);
  - choices between candidates;
  - leaderboard-informed choices: start rolls, single-episode MP-SR swaps, and s4 vs B;
  - ep36: the user reviewed renders and flagged the window to fix. The label values came from the agent; the human-hint variant s6 was rejected.
- An independent agent annotation reproduces the decisive ep33 flip and prefers the same ep32 reconstruction. It abstains where the bowl is hidden.

## 8. Resource usage
- CPU MuJoCo jobs in the Slurm debug partition (node01 and node100). Typical sizes:
  - window beam: 16–40 CPUs, about 20 min per window set;
  - margins: about 18–26 min;
  - build-v2 evaluation: 2 CPUs per episode.
- TODO: total CPU-hours, from the job ids in `out/jobs.txt` and the `STATUS.md` logs.
- Agent tokens for the bowl annotation:
  - ep33: about 282k;
  - ep32: about 257k;
  - ep36: TODO, not recorded on the server.
- TODO: total session tokens.

## 9. Keys to effectiveness
- The frame-0 anchor pose dominates scoring: the start roll of the spoon or rack moves every frame.
- The robot fit must be tight on **both** objects. A loose second object costs RPE.
- Validate reconstruction with checks the solver never saw: marker balls, stereo IoU, held-out public clips.
- RGB-camera and stereo evidence can disagree.
  - The RGB-based ep36 fix still improved the leaderboard; the ep37 analogue did not.
  - Treat stereo checks as a guide, and decide by held-out evidence.
- Use exact, hash-recorded build-v2 evaluation for every candidate, and gate it before packaging.

## 10. Code and reproduction
- Repository commit; the overlay README.
- Container: `run.sh verify` / `reproduce`.
- Weights list: `provenance/FINAL_9bc322af_episodes.tsv`.
- Disclosure: eval_code hash convention.
- CPU-dependent physics: Intel vs AMD differences were observed on ep32.
