# Human involvement and agent use: disclosure

Scope: the final file `9bc322af`, which is 8b6fa2b9 with ep36 replaced by s4.
- Human labels on the evaluation video were used in **two episodes, ep32 and ep33**, and only for the spoon.
- **ep36** uses **agent** labels, plus human review of the renders. See the ep36 section below.
- The other 17 episodes come unchanged from FINAL `0fca055b`. We found no code in their lineage that reads human labels.
- Open check: `finish_v31.py` describes v14 as "annotation-fitted spoon roll eps 3/32/33". The v14 bundle documents its spoon roll as the GT-free marker-ball estimate (G10/G16). Confirm with the v14 author.

No ground truth (public or hidden) was used for reconstruction or retargeting.

## What the human labels are

These are per-frame spoon-bowl orientation labels on the evaluation RGB video. Each label is one of:
- `"out"`: front, the concave side faces the camera;
- `"in"`: back;
- `[dx, dy]`: edge-on, with the image direction of the bowl as a unit vector.

| ep | file (in this overlay) | frames | span | label mix | how it enters |
|---|---|---|---|---|---|
| 32 | `external/hansol/share/track3_FINAL_complete_set_20261105/episodes/ep32/recon/marks_ep32.json` | 58 (every 3rd) | f40–209 (2.0–10.5 s) | edge 40, out 11, in 7 | `recon/tools/merge.py` joint solve: masks + stereo + marker balls + marks (weight `MW=4`) + temporal prior → bundle/target `K32_RP2` |
| 33 | `external/hansol/ep33_20261102/orient/ep33_marks_merged_20261103*.json` (versions a–d), `external/hansol_home/v2d/outputs/variants_20261030/recon_E14_best_B2A_S1_HGA/episode_000033/ANNOT_MARKS_20261103.json` | 78–92 (every 2–4) | f60–280 (3.0–14.0 s) | out 36, edge 29, in 13 (version a) | `ep33_20261102/code/spoon_annot_roll2.py` roll fit (`orient/refit*.sbatch`) → p33g refit → p33h start roll → `orient/fit4/recon_E14_best_B2A_S1_HGAZr6` → target `retarget33/targets/R33_HGA_r6` |

How it happened:
1. A Claude-based agent produced the reconstruction on a FoundationPose-based pipeline.
2. Where the spoon bowl kept flipping, a team member gave guidance in the form of these labels.
3. The labels entered the spoon-roll fit as a cost term.

## Independent agent annotation (made after the human labels, for comparison)

Method:
- `ep33_annot_crops.py` (and `ep32_annot/ep32_annot_crops.py`) cut crops around the SAM2 spoon mask. The crops show no reconstruction and no marks.
- Three fresh Claude subagents labelled the frames. They had no conversation context and never saw the human labels. Prompt: `external/hansol/ep33_20261102/agentmarks/agentmarks_prompt.md`.
- Outputs: `agentmarks/agent_marks.json` (ep33) and `agentmarks32/agent_marks.json` (ep32).
- Cost:
  - ep33: about 282k subagent tokens, about 4 minutes wall-clock, 174 crops (every 2nd frame, f0–346);
  - ep32: about 257k tokens, 110 crops.

Results (from `share/.../docs/EP33_AGENT_SPOON_MARKS.md`):
- **ep33.** Of the 92 frames that carry a final human-reviewed label:
  - the agent agrees on 43 (40 front, 3 edge-on same direction);
  - the agent abstained on 29, where the bowl is hidden by the cup or blurred, and the human decided;
  - the agent and the human disagree on 20, and the human decision was kept.
  - Agent labels agree with the final reconstruction r6/c18 on 123/135 frames, and with FINAL's H5r16 on 110/135.
  - Without human input, the agent reproduces the decisive flip at 6–10 s.
- **ep32.** Of the 58 human-labelled frames:
  - 17 agree;
  - the agent abstained on 13;
  - 26 are oblique views (agent front, human edge-on);
  - 2 are opposite.
  - Agent labels agree with the final K32_RP2 reconstruction on 86/88 frames, and with v14_g17 on 77/88.

Report wording (EN):
> The front/back orientation of the spoon bowl in ep32 and ep33 was set human-in-the-loop on the evaluation video.
> - A Claude-based agent reconstructed the scene on a FoundationPose-based pipeline and generated per-frame crops and silhouette overlays.
> - Where the spoon kept flipping, a team member labelled frames as front, back, or edge-on with a direction: 58 frames in ep32 and 78–92 frames in ep33.
> - These labels enter the spoon-roll fit as a cost term.
> - An independent agent annotation, made without seeing the human labels (prompt attached), agrees on most visible frames. It abstains where the bowl is hidden. On its own it prefers the same final reconstructions.

## ep36 (s4): agent labels, human review of renders

**Labels.**
- `external/hansol/spoon3537/am36/agent_marks36.json` holds 76 frames from f200 to f350, every 2nd frame: out 44, in 27, edge-on 5.
- They come from one fresh Claude subagent that labelled the RGB crops from `code/crops36.py`. The raw output is `am36/agentmarks36_raw.json`, and the per-frame check against the base is `am36/fit.txt`.
- The labels enter the 12.5–17.5 s roll refit (`spoon_annot_roll3.py`) with ball weight 3 and an IoU term.
- **No human label value is used in s4.**
- TODO: record the subagent's token use. It is not stored on the server, so take it from the session that ran it.

**Human involvement.**
- The user reviewed the renders (`code/render_ff.py`: bowl inner and outer faces drawn on the video) and pointed out the late-window spoon roll and depth problems. That review triggered the refits.

**Human hint (rejected).**
- The variant s6, with s6b and s6c, set f298–312 to edge-on `[0, 1]` from the user's visual judgement. The provenance is in `work/s6*/marks36_s6_PROVENANCE.json`.
- s6 was rejected and is **not** in the final file.

**Report wording (EN).**
> In ep36 the late-window spoon roll was refit with front/back labels produced by an independent agent instance on RGB crops (76 frames). A team member reviewed rendered overlays and flagged the window to fix; no human label values were used. A variant using a human hint for 15 frames was tried and not submitted.

## Other human or leaderboard-informed choices (also disclose)
- **ep36 s4 vs B.** We chose between s4 and B (s4 + ep37 refit) by their leaderboard MPPE and AUC.
- **ep33 start roll −6° (p33h).** It was chosen with a kit-scored sweep against six ep33 robots that had officially failed MP-SR. The −16° total equals the start roll with the best ep33 AUC in the team's leaderboard sweep (`external/hansol/ep33_20261102/code/build_p33h.py`).
- **ep32.** The earlier FINAL ep32 (K32_R10) was described as "v14_g17 + leaderboard start roll".
- **Per-episode swaps.** Single-episode packages were uploaded to the MP-SR board (`--only maniptrans_sr`) to learn whether that episode passes. Examples: ep32 T32c_b1 (submission 57979782, 0.95), and the ep33 R2 failure (57947230).
- **User decisions recorded in the logs.** For example, "the flipped spoon direction (c18/r6 family) is correct" (teacher33 README §23–24, 21:10 KST Nov 4).

## Tools and roles
- **Agents:**
  - Claude Code sessions planned, wrote and ran the reconstruction, retargeting and packaging code;
  - Claude subagents did the independent bowl annotation;
  - the DexAgent agentic pipeline was evaluated separately (`experiments/dexagent_egotools_10`, not in the submission).
- **Humans:**
  - task direction and decisions, video review, the spoon labels above, and uploads.
- **Compute:**
  - Slurm debug partition on node01/node100, CPU MuJoCo. No GPU is needed for retargeting.
