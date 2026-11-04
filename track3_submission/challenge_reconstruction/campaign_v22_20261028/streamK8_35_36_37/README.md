# Stream K8 (v23, PHASE8.md): episodes 35, 36, 37 (single wooden_spoon), 2026-10-28

Goal (PHASE8): spoon <= 1.2 cm / 12 deg mean vs `campaign_v22_20261028/target_final_v5`, strictly better than the v22b component,
lower per-frame peaks; build-v2 exact; scored with `campaign_v22_20261028/code/score_any.sh`.
v22b components: ep35 C7_ep35_follow1 1.71/18.5, ep36 C7_ep36_follow3 1.70/15.8, ep37 v21 legacy actor 2.20/11.7 (never re-planned).
Single object => the scorer anchor is the spoon itself (frame-0 alignment on the spoon): every internal cost here is the
scorer-identical per-frame error (tlibK.errors = tlibC; identity plans reproduce the v22b numbers exactly: 1.711/18.45, 1.698/15.75, 2.203/11.71).

## Results

All are build-v2 exact (`run_one.sbatch`, completed.json exact_previous_physical_states_actions = true) and scored with score_any.sh
(`campaign_v22_20261028/scores/cand/<tag>/score.json`). Build-v2 outputs: `campaign_v16_20261023/common_v2_candidates/<tag>/episode_0000NN`.

| ep | tag (recommended in bold) | component | spoon cm / deg | p95 / max (cm, deg) | MPPE / AUC | v22b (cm/deg, p95, max) | checkpoint / archive sha256 |
|---|---|---|---|---|---|---|---|
| 35 | **K8_ep35_ilc2** | staging_K8_35b/components/episode_000035 | **1.27 / 13.5** | 3.63, 41.3 / 6.79, 72.3 | 1.63 / 0.890 | 1.71/18.5, 3.63/41.6, 6.85/73.6 (MPPE 2.16, AUC 0.812) | 52f9900e5fbfa609.. / c89e05e7f5cfeff8.. |
| 35 | K8_ep35_ilc1 | staging_K8_35a/components/episode_000035 | 1.40 / 14.5 | | 1.76 / 0.877 | | |
| 36 | **K8_ep36_ecem_ilc** | staging_K8_36d/components/episode_000036 | **1.18 / 11.3** | 2.84, 35.5 / 18.0, 72.1 | 1.46 / 0.930 | 1.70/15.8, 4.35/43.2, 17.9/81.7 (MPPE 2.08, AUC 0.863) | e9507adfde0a8deb.. / aa8cf07d11ebffef.. |
| 36 | K8_ep36_ilc1 | staging_K8_36a/components/episode_000036 | 1.44 / 15.0 | | 1.82 / 0.895 | | |
| 37 | **K8_ep37_ilc1** | staging_K8_37b/components/episode_000037 | **0.24 / 2.0** | 0.43, 3.6 / 0.50, 14.0 | 0.28 / 1.000 | 2.20/11.7, 6.73/33.2, 10.1/51.6 (MPPE 2.33, AUC 0.801) | b78bc81d28e29161.. / 9fcfa36d969740fa.. |

- Lineage: ep35 = C7_ep35_follow1 actor (c35_cem2_s42) + ILC rounds a, b (runs/i35_a, i35_b); ep36 = C7 c36_cem2_s32 + early-window
  planK CEM (runs/e36_s3) + 6-window ILC (runs/i36_d); ep37 = v21 legacy actor gain_0.03 + ILC (runs/i37_a, saved in runs/i37_a_save).
  Architecture unchanged (ComponentPolicy over the same checkpoint fields; only phase weights change; margins/feedback/scene/meshes unchanged).
  Mesh c9ab88e33e5c (unchanged, checked by stageK.py).
- staging_K8_37a is an aborted partial staging (2026-10-19 wrapper refused ep37; re-staged as 37b with the build-v2 wrapper). Do not use.
- Goal 1.2 cm / 12 deg: met for ep36 and ep37; ep35 is 1.27 / 13.5 (floor = chaotic 1-9 s hold + an unreachable tilted end rest).
- Peaks: ep37 max drops from 10 cm / 52 deg to 0.5 cm / 14 deg; ep36 p95 4.35 -> 2.84 cm, 43 -> 35.5 deg (max 18 cm = target outlier
  f290-306); ep35 peaks are in the unchanged chaotic 1-9 s window (p95/max unchanged).

## What worked
1. **ep37 re-plan to target_final_v5 (PHASE7 recipe step 1).** `code/build_world.py 37` (copy of the C7 script): T_new = G0 @ T_target @ X,
   X = scorer registration (identity, same mesh), G0 = frame-0 anchor onto our physics start; written to `source/` + `episode/episode_000037`.
   On this world the v21 legacy actor (refinement_20261018/expanded_batch/learned/episode_000037/gain_0.03) is 2.20/11.7 (= score_any).
   - planK (= C7 planC) target-following wrist re-plan: grid gr 1, gp 1, sig 0.05, window 1.6-5.0 s -> 0.68/4.0; CEM (cemK) 0.50-0.52/4.0-4.3.
   - **windowed ILC (`code/ilcK.py`) directly on the legacy actor**: 1.4-5.2 s then 1.4-7.5 s -> **0.24 / 2.0** (p95 0.43 cm / 3.6 deg).
     The ep37 pinch hold is benign (most ILC candidates improve).
2. **ilcK = stream H's actor_res.py ILC, adapted**: single spoon scored with tlibK; ILC target = frame-0-aligned target (A0 @ T_target X),
   separate position / rotation gains (ap, ar) x smoothing sigma grid (30 candidates per iteration, step halves when nothing improves),
   **wrist chart fix** (ep35 rot hinges sit under a fixed chart rotation Ac: a world correction D enters as Ac^-1 D Ac, as C7 found),
   cost = max(pm/1.2, dm/12) + 0.25 (pm/1.2 + dm/12) + 0.05 (p95 terms) + 10 x never-worse penalty (vs the v22b numbers),
   and a bit-exact prefix cache (MuJoCo integration state per frame; restart from the first frame whose phase output differs; checked
   `CACHE exact True` in every log, and every accepted step is re-run from the keyframe). Every saved actor is re-run by the unchanged
   evaluate_policy_component; the official actor evaluation equals the search value in every case.
   - ep35 (on C7 c35_cem2_s42): late windows benign: a (2.8-21 s, 4 windows) 1.40/14.5 -> b (9.2-21 s, 5 windows) **1.27/13.5**.
   - ep36 (on C7 c36_cem3_s52): a 1.44/15.0 -> b 1.28/12.5 -> c 1.23/11.9.
3. **ep36 early-window lottery + late ILC.** cemK with the cost restricted to frames 0-170 (`K8_FRAMES`) plus a keep-the-hold penalty on
   the late frames (`K8_LATE`, excluding the target's f290-330 outlier) over planK params (on C7 c36_cem2_s32, init = C7 cem3 params,
   + one roll pulse): e36_s3 early 1.11/15.2 (C7 1.30/18.0), full 1.52/15.7; then 6-window ILC -> **1.18 / 11.3** (i36_d).
   Without the late penalty (e36_s1/s2) the early optimum dropped the spoon at 15 s.

## What did not work
- ILC / knot CEM in the chaotic power-grasp windows (ep35 0.6-9.2 s, ep36 1.6-8.2 s): 0 improving candidates out of ~700 knot-CEM
  samples and every ILC grid (confirms streams B/H: the hold there is chaotic).
- ep35 early-window planK CEM (4 seeds, with/without late penalty): no better early trajectory that keeps the hold.
- Registered ReleasePolicy primitive (`code/relK.py`, copy of stream I release_cem.py, margins kept) for the end rest: best ep35
  1.68/17.5, ep36 1.70/15.7 from the C7 actors (holding the phase freezes the base's own late motion); the late ILC is better.
- ep35 end rest (19-21 s, ~12-15 deg): the target spoon rests pitched ~9-12 deg about its own y axis (long axis rising 0.15) while our
  spoon lies flat on our table: not reachable physically (cf. final_recon table tilt).
- ep36 f290-306 (14.5-15.3 s): the target has a 20 cm outlier there; ILC windows touching it get worse.

## Code (all new copies; phase-7 code untouched)
build_world.py (ep37 world), tlibK.py (= tlibC), planK.py / cemK.py (C7 planC/cemC + ep37 + K8 cost, K8_FRAMES / K8_LATE, --space),
ilcK.py (+ res.sbatch), relK.py (+ rel.sbatch), export_plan.py (+ export.sbatch), stageK.py (+ stage.sbatch: C7 stage_c7 generalised;
ep37 uses the build-v2 wrapper because the 2026-10-19 wrapper refuses legacy ep37), diag_cur.py (per-window errors of a build-v2 output).
Runs: runs/ (g37_grid1, c37_cem1_s*, i37_a(_save), i35_a..d, i36_a..e, e35_s*, e36_s*, k35_a/k36_a (knot CEM), r35_s*/r36_s* (release)).
Searches ran on node200 (physics rollouts checked bit-identical to the AMD evaluations: C7 actors reproduce exactly); staging and
build-v2 on node01.

## ep35 extension (coordinator request: reach 1.2 / 12) — not reached; K8_ep35_ilc2 (1.27 / 13.5) stays the best
(a) Distillation lottery (`code/lot.sbatch` = streamC actor_grid_c.py unchanged, 16 configs c512-c2048 x p 0.001-0.03, evaluated on the
C7 target_final_v5 world): teachers = i35_b rollout (seeds 44/45/46, 48 actors), e35_s2 and e35_s1 early-window CEM rollouts (16 each).
i35_b-distilled actors are all worse early (>= 20.9 deg on 0-9.5 s vs 20.6). e35_s2-distilled actors reach better early windows (best
3-6 s window 19.0 deg vs 27.0) but lose the late hold (late 2.6-3.8 cm / 36-48 deg); late ilcK (6 windows) on the three best of them
recovers only to 1.77-1.92 / 18.4-18.7.
(b) Finger margins via streamK_ep35/code/search_margins_k.py (unchanged; `code/margins.sbatch`, 96 x 12, bound 0.25, tol 2.0, left pinky):
on the C7 actor 1.89/18.3 and 1.77/19.4 (start 1.71/18.5); on i35_b 1.41/14.6 and 1.60/16.2 (start 1.27/13.5; margins change the whole
hold so the late ILC fit is lost). No early window better than i35_b's.
(c) Floor (`code/scan35.py`, per-frame scorer-identical errors of all 470 scanned ep35 rollouts vs target_final_v5), K8_ep35_ilc2 per window
(cm / deg; share of the 421-frame mean):
| frames | current | share of mean | best observed in any rollout |
|---|---|---|---|
| 0-22 (rest) | 0.27 / 3.4 | 0.01 / 0.18 | 0.26 / 3.3 |
| 22-60 (pickup) | 2.51 / 12.6 | 0.23 / 1.14 | 1.73 / 11.4 |
| 60-120 (3-6 s hold) | 1.51 / 27.0 | 0.22 / 3.85 | 1.39 / 19.0 |
| 120-190 (6-9.5 s swing) | 2.01 / 24.9 | 0.34 / 4.14 | 1.52 / 24.0 |
| 190-260 | 1.24 / 10.3 | 0.21 / 1.72 | 1.16 / 10.7 |
| 260-330 | 0.92 / 5.2 | 0.15 / 0.86 | same |
| 330-392 | 0.54 / 4.3 | 0.08 / 0.63 | same |
| 392-421 (end rest) | 0.60 / 14.7 | 0.04 / 1.01 | 0.45 / 15.3 |
- End rest: the target spoon is tilted 20-21 deg from the world vertical at f390-420 while its own start rest (f0-20) is tilted 8-9 deg
  (same table): the target's two rests disagree by ~12 deg (reconstruction inconsistency). Our spoon rests flat at 7-12 deg, so >= 12.2
  deg on 29 frames = 0.84 deg of the mean is unreachable.
- Even a cherry-picked composite (best observed window over all rollouts for 0-9.5 s + the current 9.5-21 s) gives 1.10 cm / 12.12 deg:
  above 12 deg, and no single physical rollout realises it (the early-better trajectories lose the late hold).
