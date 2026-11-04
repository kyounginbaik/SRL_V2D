# Stream E9 (phase 9): tighten episodes 1, 14, 15 and 16 against target_final_v6, 2026-10-28

Goal:
- per object, at most 0.6 cm and 6 deg (mean);
- per episode, MPPE at most 0.7 and AUC at least 0.97;
- never worse than the current component (selection_v24) on any object, on MPPE or on AUC.

Scores come from `code/score_any.sh EP COMPONENT LABEL target_final_v6` (`../scores/cand/<tag>/score.json`).
Every tag is in campaign_v16_20261023/common_v2_candidates/ and is build-v2 exact (`completed.json` exact_previous_physical_states_actions true).

- Poses come only from build-v2 policy rollouts.
- Submitted meshes are byte-identical to the current component's (checked with cmp and sha256).
- Nothing was uploaded, committed, deleted or overwritten.

## Results (cm / deg; vs target_final_v6)

| ep | tag (recommended in bold) | component = staging/<dir>/components/episode_XXXXXX | object 1 | object 2 | MPPE | AUC | exact | current component (selection_v24) |
|---|---|---|---|---|---|---|---|---|
| 1 | **E9_ep01_b3** (= E9_ep1_b3) | staging/ep1_b3 | pot **0.53 / 3.0** | lid **0.53 / 4.9** | **0.612** | **0.984** | true | N8_ep01_k2c2048: 0.76 / 4.9, 0.84 / 9.4, 0.963, 0.949 |
| 1 | E9_ep1_b2 (alt) | staging/ep1_b2 | pot 0.61 / 3.3 | lid 0.50 / 4.9 | 0.637 | 0.982 | true | |
| 14 | **E9_ep14_b4** | staging/ep14_b4 | blue_cup 0.93 / **2.4** | rack **0.59 / 2.8** | 0.782 | **0.972** | true | N8_ep14_kl512: 1.11 / 4.1, 0.67 / 2.8, 0.924, 0.966 |
| 14 | E9_ep14_b2 (alt) | staging/ep14_b2 | blue_cup 0.94 / 2.4 | rack 0.60 / 2.8 | 0.789 | 0.972 | true | |
| 15 | **E9_ep15_b6** | staging/ep15_b6 | rack 0.66 / **2.8** | spoon 0.93 / 7.1 | 0.886 | 0.963 | true | U8_ep15_c: 0.68 / 2.8, 0.94 / 7.1, 0.899, 0.962 |
| 16 | **E9_ep16_b8** | staging/ep16_b8 | rack **0.39 / 2.6** | spoon 0.84 / **5.3** | **0.687** | **0.979** | true | U8_ep16_b: 0.68 / 3.6, 1.07 / 6.2, 0.945, 0.959 |
| 16 | E9_ep16_b6 (alt) | staging/ep16_b6 | rack 0.41 / 2.7 | spoon 0.86 / 5.5 | 0.711 | 0.978 | true | |
| 16 | E9_ep16_b2, E9_ep16_b1 (alts) | staging/ep16_b2, ep16_b1 | 0.49 / 3.0; 0.52 / 2.9 | 0.97 / 6.1; 0.99 / 6.1 | 0.802; 0.827 | 0.971; 0.970 | true | |

Unrounded official evaluator values (mresE9 rescoring of each saved actor's official evaluate_policy_component, equal to the search value):
- ep1 b3: pot 0.5311 / 3.029, lid 0.5294 / 4.867.
- ep14 b4: cup 0.9331 / 2.355, rack 0.5896 / 2.758. Current: 1.1089 / 4.051, 0.6674 / 2.846.
- ep15 b6: rack 0.6625 / 2.828, spoon 0.9314 / 7.057. Current: 0.6832 / 2.846, 0.9450 / 7.104.
- ep16 b8: rack 0.3905 / 2.611, spoon 0.8396 / 5.304. Current: 0.6798 / 3.609, 1.0711 / 6.199.

**Every recommended tag is better than its current component on every object (cm and deg), on MPPE and on AUC.** MP-SR 1, SP-SR 1.

How each episode stands against the targets:
- **ep1 meets every target.**
- **ep16** meets the MPPE, AUC and deg targets. The spoon is 0.84 cm against the 0.6 aim; about 0.4 cm of that is a static floor (see below).
- **ep14** has a plateau on the cup, set by the cup rest pose.
- **ep15** is a rack-vs-spoon trade-off. Only small Pareto-safe gains were found.

Hashes (sha256, first 12):

| tag | checkpoint | portable archive |
|---|---|---|
| E9_ep01_b3 | b3e9ad53217e | 3a95cf1646bb |
| E9_ep14_b4 | 11f8e6f983a0 | 32f5d9f3fdfa |
| E9_ep15_b6 | d5723a82a9ab | 1b02af08ccc1 |
| E9_ep16_b8 | c173e03a43b8 | d2f21b589510 |

Each archive is at `legacy_component.parents[1]/portable_archives/episode_XXXXXX_portable.tar.gz`.

Meshes (unchanged, sha256 first 8):
- ep1: pot a5531704, lid bc1443c5;
- ep14: cup 78bd065c, rack 090b7d17;
- ep15: rack 1162524a, spoon 6cc4d2b2;
- ep16: rack 67050d1f, spoon 6cc4d2b2.

## Method: D9 two-hand window beam search, reused (code/wbeamE9.py)
- **Shared code.** `mresE9.py`, `mlibE9.py` and `wbeamE9.py` are copies of D9's mres9 / mlib9 / wbeam9 (target_final_v6), with these additions:
  - mlibE9 adds eps 1 (white_pot, white_pot_lid) and 14 (blue_cup, plastic_dish_rack).
  - wbeamE9 gives blue_cup (kit order 360) axis-only (swing) ILC rotations, so the arbitrary cup spin is not chased.
  - `--owd` weights the deg term of one object only.
  - `--pareto SLACK`: a child whose running per-object pos or deg sum over [0, window end + look] exceeds the current component's sum (plus slack) is penalised by 100 per tolerance unit. The beam then follows prefixes that are no worse on any object. The current component is the unedited actor, or `--pareto-table`.
- **Exactness check, before any search.** mresE9.Env reproduces all four current components exactly (runs/ep*/base_eval):
  - N8_ep01_k2c2048 0.760 / 4.94, 0.837 / 9.41;
  - N8_ep14_kl512 1.109 / 4.05, 0.667 / 2.85;
  - U8_ep15_c 0.683 / 2.85, 0.945 / 7.10;
  - U8_ep16_b 0.680 / 3.61, 1.071 / 6.20.
  - Every saved actor's official evaluate_policy_component reproduced its search value exactly.
- **Actors.** The search starts from the shipped actors:
  - ep1: N8 distilled k2_lr_r45 c2048;
  - ep14: N8 distilled k_l_10_65 c512;
  - ep15: U8 save_beam_r2b0, which has the U8 hold primitive baked in;
  - ep16: U8 save_beam_ab0.
  - Only phase-weight residual tables are added. Scene, start state, feedback weights and wrappers are unchanged.
- **Hands.** The hands were chosen from a contact analysis (`code/whotouch.py`):
  - ep1: pot left, lid right, random knots on both hands.
  - ep14: cup left; random knots on the left hand. The rack has no hand, and the right hand touches nothing.
  - ep15/16: spoon right; random knots on the right hand. The rack has no hand.
- **Settings.** tol 0.6 cm / 6 deg, MPPE weight 1, 0.5 s windows, beam 4 with 40 children (or beam 6 with 60 children and look 2).
- **Staging.** `code/stage.sbatch` uses the unchanged stagers of each world with E9 lineage added:
  - eps 1/14: N8 stage_n8.py with the N8 teachers k2_lr_r45 / k_l_10_65;
  - ep15: streamS stage_seven_component_newtrack.py;
  - ep16: streamF stage_seven_component_retargeted.py;
  - for 15/16, `code/mkbaseE9.py` derives the base from U8's ep15_c / ep16_b staging base.
- **Verification.** Build-v2 with run_one.sbatch on node01, then score_any.sh against target_final_v6.

Passes:
- **ep1:**
  - b1 covered 1.5-9 s and fixed the lid: 0.84/9.4 → 0.73/5.1.
  - b2 covered 9-16.7 s: pot 0.61/3.3, lid 0.50/4.9.
  - b3 (full pass): 0.53/3.0, 0.53/4.9, MPPE 0.61. All targets met; stopped.
- **ep14:**
  - b1 gave cup 0.95/2.3, but the rack rose to 2.90 deg, above 2.85.
  - b2 (rack x3) and b3 (rack x2) fixed it.
  - b4 searched the placement window 2.5-6 s with look 2.5 and beam 5.
  - Passes b2-b4 differ by less than 0.01 cm: plateau.
- **ep15:**
  - Every weighting traded the rack against the spoon:
    - b1 (no weights): rack deg 2.96;
    - b2 (rack x3) and b3/b3x (rack 1.7 and 2): spoon 1.01-1.05 cm;
    - b4 (rack deg x3): spoon 1.01.
  - The Pareto-constrained beam b5 (look 1) could not escape the 4 s seat event.
  - b6 (Pareto, look 2, beam 6) is the first strict improvement on every object.
  - Chained b7 and a fresh b8 (look 2.5, spoon x1.5) did not beat it: plateau.
- **ep16:**
  - Rack x1.5, chained b1 → b8. The last pass changed the cumulative cost by only 1.3 %: plateau.
  - Spoon by pass: 0.99 → 0.97 → 0.94 → 0.91 → 0.90 → 0.86 → 0.84 → 0.84.
  - Rack: 0.52 → 0.39.

## Remaining error (per 0.5 s window, eval of the final tables: runs/ep*/eval_*)
- **ep14 cup rest floor.** From 5 to 7.85 s the cup sits in the rack 1.7-2.3 cm off, which is about 0.7 cm of the 0.93 mean.
  - `code/restpen.py` puts the target_final_v6 cup at its target pose in the physics world. At frames 115-157 it penetrates our rack by 16-18 mm and the table by 1-5 mm.
  - So the target rest pose is not reachable with our meshes. The achieved rest is 1.6-2.0 cm away, with 4 mm contact.
  - Rack: 0-1 s about 0.5 cm, the settling floor N8 found, plus 4-4.5 s about 1.3 cm when the cup pushes it.
- **ep16 spoon.**
  - 0-2 s pre-grasp: 1.78 cm static. The target spoon is below our table plane (U8). That is about 0.41 cm of the 0.84 mean.
  - 6.5-7 s: the rack is 1.05 cm off, where the seated spoon pushes it.
  - 7-7.5 s: spoon 9 deg.
- **ep15.**
  - 0-1.5 s pre-grasp spoon: 1.55 cm static, the same target-below-table floor, about 0.28 cm of the mean.
  - 4.0-4.5 s seat event: spoon about 3 cm / 10-17 deg, and the rack 1.6 cm / 5 deg as the spoon is pushed into it.
  - Steering the seat moves the rack, which is why rack and spoon trade.
- **ep1.**
  - Pot 13.5-14.5 s: 8-10 deg.
  - Lid 5-6.5 s: 10 deg, resting on the table after release.
  - Lid 13-13.5 s: 1.45 cm.

## Caveats
- Scores are proxies against our own reconstruction (target_final_v6), not hidden GT.
- The holds are chaotic. Each final rollout is one deterministic outcome of a deterministic search. It is exact and replayable under build v2, but not robust to upstream edits.
- E9_ep1_b2 / E9_ep1_b3 were first verified under one-digit tag names. E9_ep01_b3 is the same component re-verified under the required naming: identical checkpoint, identical actions, exact.

## Files
- **code/**:
  - search: wbeamE9.py, mresE9.py, mlibE9.py;
  - jobs: res.sbatch, base.sbatch, save.sbatch, stage.sbatch, savestage.sbatch;
  - mkbaseE9.py;
  - diagnostics: whotouch.py, restpen.py;
  - helpers: env.sh (actor and mesh paths), wins.py, sumlog.sh.
- **runs/ep{1,14,15,16}/**: base_eval, b*/ (settings.json, progress.json, final_*.npz, final.json), save_*/actor, eval_*.
- **staging/**: ep1_b2, ep1_b3, ep14_b2, ep14_b4, ep15_b6, ep16_b1, ep16_b2, ep16_b6, ep16_b8.
- **logs/**: res-, save-, stage-, svst-<job>.log.
