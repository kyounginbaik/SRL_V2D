# V2D Track 3 final submission: open issues (2026-11-04, evening, Hansol)

Sources: `V2D_FINAL_SUBMISSION_REQUIREMENTS.md`, built from the 11-02 kit (`CONTAINER_SUBMISSION.md`, `tools/verify_container.py`), the Kaggle Rules, and the challenge website and FAQ. Kaggle submissions close **Thu Nov 5, 07:00 KST** (Nov 4, 5:00 p.m. EST).

## A. Must be done before the deadline (07:00 KST)

1. **Select the same file as Final Submission on all six boards** (Kaggle UI, up to 2 per board).
   - Without a selection, each board counts its best-scoring submission.
   - Today the best file differs by board: RPE = aa8fde7c; MP-SR, SP-SR and CD-O = v33a; AUC and MPPE = 0fca055b; and so on.
   - In that case the post-deadline container must reproduce **every** one of those files, each with its receipt, checkpoints and meshes.
2. **Check that `Nvidia-V2D-Challenge` has been invited as a collaborator on `kyounginbaik/SRL_V2D`.**
   - `Nvidia-V2D-Challenge` is a GitHub user account, not an org.
   - The repo is private (anonymous access returns 404).
   - Every row's `code_commit_url` is `kyounginbaik/SRL_V2D/commit/5cc5ca51…`.
3. **Check that the team registration form (Google Form) was submitted.** This is separate from accepting the Kaggle rules (Rules 2.8.d).

## B. Possibly fatal (ask the organizers in advance)

4. **Evaluator hash and backend differ from the official ones.**
   - FINAL's `eval_code_sha256 = fdddb348` uses the team build-v2 convention (a hash over the concatenated source bytes).
   - The official `verify_container.py` recomputes it over FlashCHORD `flash_chord/evaluation/{__init__,harness,metrics,rollout}.py` (official value `b1b3ebe3`). That check will therefore **fail by construction**.
   - The rollouts also come from the team MuJoCo build-v2 backend, not the official FlashCHORD runtime. Rules 2.9.d and the container guide assume the "organiser-provided evaluator".
   - Kaggle accepted the files because the `accepted_*_sha256` allow-lists are empty. Leaderboard ranks stay provisional until the container check (Rules 2.3.c).
   - Proposal: ask v2d_challenge@nvidia.com how a submission produced with a custom evaluator wrapper will be verified, and whether it is valid.
5. **Does the single commit (5cc5ca51, Oct 23) reflect all of FINAL's code?**
   - That commit only changes 3 files under `evaluator_build_v2_20261023/package_code/`.
   - The code that produced the Nov 2–4 checkpoints and meshes (K32/K33 window-beam, teacher ILC/CEM, distillation, reconstruction, splice) may not be in it. If so, Rules 2.9.a could be breached ("that commit must reflect the code used to generate that Submission").
   - The commit URL of an already-uploaded file cannot be changed.
   - For new uploads (e.g. the ep33 fix), consider a commit that includes that code. The provenance columns must still be identical across all 20 episodes.

5a. **Human labels on the test video (Foundational Rules 3.4.b).**
   - Rule 3.4.b: no "hand labeling or human prediction" on test data.
   - The successful ep32 reconstruction (0.95) uses `marks_ep32.json`, human bowl-direction marks on 58 frames of the evaluation video, as a cost term.
   - Our ep33 r6/c18 lineage (including R6M and R6M2) fits the spoon roll to human marks on 92 frames.
   - The organisers' Astra guidelines ask us to document "human involvement", and their example lists an "iterative process with human intervention". Proposal: do not hide it; **document it accurately** as human-in-the-loop.
   - For ep33 we compared the marks with an independent agent annotation: 43 agree, 29 agent abstained, 20 disagree. The agent independently reproduces the 6–10 s flip. Report wording is in `EP33_AGENT_SPOON_MARKS.md`.
   - In the report, give no order between the two annotations (human 11/03, agent 11/04), and do not quote an agent "success rate".
   - The same comparison is done for ep32 (58 human-marked frames): 17 agree, 13 agent abstained, 26 oblique views that the agent called front and the human labelled edge-on, 2 disagree. The agent annotation alone also prefers the successful K32_RP2 over the earlier v14 (86/88 vs 77/88). Wording is in the same file.
   - The team should also decide whether to document values chosen from leaderboard feedback (the K32_R10 start roll, per-episode ep33 uploads).

## C. After the deadline (no deadline published)

6. **Docker container** (`/submission`):
   - `code/`: the repo at the commit, with `.git`, clean.
   - `kit/` and `verifier_kit/`.
   - `submitted/`: the uploaded files plus their receipts.
   - `weights/`: the 20 checkpoints as original bytes.
   - `meshes/`.
   - `run.sh verify|reproduce`: a single command that rolls out all 20 episodes and packs them, with no network access.
   - Ship it as `docker save | zstd` plus its sha256 and image digest, behind a link that can be downloaded without an access request, together with the technical report.
   - Still to check: whether we can build Docker images on our cluster, the missing Track 3 `submission.json` template, and how the CD-O file should be referenced.
7. **Technical report.** Format, length, deadline and how to submit are not stated anywhere public.
   - The report should explain how to run the container.
   - It should also give our run-to-run variation, since episodes near the MP-SR thresholds can flip on a re-run.
8. **License.** The Kaggle rules and FAQ say Apache-2.0, while the website says "research license".
   - Third-party components must meet the OSI and commercial-use terms of Foundational Rules 3.6.c. MANO (non-commercial), DexAgent, SAM3D and FoundationPose all need checking.

## D. ep33 (for reference)

- The ep33 fixes are built with the same provenance as FINAL: evaluator hashes cbe866f9 / fdddb348, packer and uploader hashes, and commit 5cc5ca51. Whatever is decided for items 4 and 5 applies to them as well.
- An ep33 built with the official FlashCHORD evaluator (160d24f5 / b1b3ebe3) mixes hashes with FINAL and fails strict validation (our earlier FC33 attempt).
