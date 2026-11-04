# Stream F9 (phase 9): tighten episodes 8, 20, 28, 30 and 38 against target_final_v6, 2026-10-28

Aims: every object <= 0.6 cm and <= 6 deg (mean), MPPE <= 0.7 and AUC >= 0.97 per episode, never worse than the current (selection_v24) component.
Scores come from `code/score_any.sh EP COMPONENT LABEL target_final_v6` (`../scores/cand/<tag>/score.json`; label = tag).
Every tag is in campaign_v16_20261023/common_v2_candidates/ and is build-v2 exact (`completed.json` exact_previous_physical_states_actions true).

- Poses come only from build-v2 policy rollouts. The policies are phase-weight residuals on the shipped actors.
- Meshes: ours only. They are byte-identical (cmp) to the current component's build-v2 meshes. For ep30 they equal A9_ep30_sweeper's.
- Nothing was uploaded, committed, deleted or overwritten.

## Results (cm / deg; vs target_final_v6)

| ep | tag (recommended in bold) | component = staging/<dir>/components/episode_XXXXXX | object 1 | object 2 | MPPE | AUC | exact | current component (selection_v24) |
|---|---|---|---|---|---|---|---|---|
| 8 | **F9_ep08_b4** | staging/ep8_b4 | beige_cup **0.29 / 1.4** | white_pot **0.35 / 1.7** | **0.339** | **0.998** | true | N8_ep08_klr512: 0.71 / 2.9, 0.83 / 7.1, 0.851, 0.961 |
| 8 | F9_ep08_b3 | staging/ep8_b3 | 0.30 / 1.5 | 0.36 / 1.8 | 0.351 | 0.998 | true | |
| 8 | F9_ep08_b2 | staging/ep8_b2 | 0.33 / 1.7 | 0.41 / 2.1 | 0.395 | 0.997 | true | |
| 8 | F9_ep08_b1 | staging/ep8_b1 | 0.42 / 2.2 | 0.54 / 3.4 | 0.517 | 0.992 | true | |
| 20 | **F9_ep20_b4** | staging/ep20_b4 | blue_cup **0.25 / 1.4** | water_pitcher **0.39 / 1.7** | **0.347** | **0.996** | true | N8_ep20_k3c1024: 0.61 / 11.8, 0.57 / 3.4, 0.812, 0.960 |
| 20 | F9_ep20_b3 | staging/ep20_b3 | 0.26 / 1.5 | 0.40 / 1.7 | 0.355 | 0.995 | true | |
| 20 | F9_ep20_b2 | staging/ep20_b2 | 0.28 / 1.7 | 0.41 / 2.0 | 0.375 | 0.994 | true | |
| 20 | F9_ep20_b1 | staging/ep20_b1 | 0.32 / 2.0 | 0.50 / 2.9 | 0.453 | 0.991 | true | |
| 28 | **F9_ep28_b1** | staging/ep28_b1 | plastic_dish_rack 0.72 / **1.8** | white_pot **0.45 / 5.0** | **0.655** | **0.971** | true | J8_ep28_w2k3c2048: 0.74 / 1.9, 0.77 / 8.0, 0.842, 0.966 |
| 30 | **F9_ep30_b4** | staging/ep30_b4 | mini_sweeper **0.39 / 2.2** | plastic_dish_rack **0.31 / 3.0** | **0.402** | **0.997** | true | A9_ep30_sweeper: 0.90 / 6.1, 0.54 / 4.4, 0.819, 0.961 |
| 30 | F9_ep30_b3 (alt) | staging/ep30_b3 | 0.39 / 2.4 | 0.33 / 2.9 | 0.415 | 0.996 | true | |
| 30 | F9_ep30_b2 | staging/ep30_b2 | 0.40 / 2.7 | 0.37 / 3.1 | 0.441 | 0.995 | true | |
| 30 | F9_ep30_b1 | staging/ep30_b1 | 0.50 / 3.2 | 0.42 / 3.4 | 0.522 | 0.988 | true | |
| 38 | **F9_ep38_b3** | staging/ep38_b3 | mini_dust_pan **0.36 / 3.4** | | **0.438** | **0.998** | true | J8_ep38_ilc36c2048: 0.64 / 5.5, 0.759, 0.980 |
| 38 | F9_ep38_b2 | staging/ep38_b2 | 0.37 / 3.5 | | 0.452 | 0.997 | true | |
| 38 | F9_ep38_b1 | staging/ep38_b1 | 0.43 / 3.7 | | 0.513 | 0.994 | true | |

- Every tag listed is better than its current component on every object (cm and deg), on MPPE and on AUC. MP-SR 1 and SP-SR 1 for all of them.
- **Eps 8, 20, 30 and 38 meet every aim**, in every pass from b1 on.
- **ep28 meets every aim except the rack position: 0.72 cm against the 0.6 cm aim.** This is a target floor (see below).
- ep30: b4 is better than b3 on the sweeper, rack position, MPPE and AUC, and 0.07 deg worse on rack orientation (2.97 vs 2.90). Both are far better than A9.
- Checkpoint / portable archive sha256 (first 12):
  - ep8 b4: 366c1cd8e6aa / b01e0727f931;
  - ep20 b4: 8f3b1a764d34 / 7d3222e5bf94;
  - ep28 b1: 8b1cc5ede464 / ae6e537e8318;
  - ep30 b4: 1ee5d4dc2d74 / 983a5cf8174e;
  - ep30 b3: 5b9b0109c866 / 51e276209056;
  - ep38 b3: 0be7383c55fd / f8efaea91f80.
  - Each archive is at `staging/<dir>/portable_archives/` (= legacy_component.parents[1]).
- Submitted meshes (sha256, first 8; build-v2 completed.json). `code/verify.sh` cmp-checked every file against the current tag's build-v2 meshes:
  - ep8: beige_cup 2c041e69, white_pot 8cbd3543;
  - ep20: blue_cup ff3784a8, water_pitcher 08e6ab82;
  - ep28: rack 00c51376, white_pot 69677910;
  - ep30: mini_sweeper ccb942f5 (A9's half-turn-correct final_v6 mesh) and rack 37f78ed0. Both equal A9_ep30_sweeper's files;
  - ep38: dust pan 1d298c01.

## Method: D9 two-hand window beam search (wbeam9), reused
**Reproduction first.** `code/mresF9.py` and `code/mlibF9.py` are D9 mres9 / mlib9 with eps 8/20/28/30/38 added. The NAMES order is the scorer slot order from score.json.
- `mresF9 eval` on each current actor + current component meshes reproduces the score_any cm/deg of all five current components.
- Its rollout qpos is **bit-identical** to the build-v2 re-execution trajectory (`code/cmpbase.py`, max |dq| = 0 for all five).
- The ep30 registration uses A9_ep30_sweeper's swapped meshes.

**Search.** `code/wbeamF9.py` = D9 wbeam9.py with one change: ILC candidates on axisymmetric objects (kit order 360: the ep8 beige cup and the ep20 blue cup) use `axis_only`, because the cup spin is not scored.
- 0.5 s windows, beam 4 (2 per parent), and 40 random knot children split over the hands.
- Windowed rigid-hold ILC candidates (full/pos/rot/rotanc, sigma 0.1/0.25 s, alpha 0.15-1).
- Exact cached MuJoCo states. Unedited children equal their parent bit for bit (checked).
- Cumulative per-frame cost sum_o (pos/0.6 + deg/6) + MPPE_frame/0.6 (tol = the aims).

Hands came from the contact frames of the base rollout (`code/handsF9.py`):
- ep8: beige cup right, pot left.
- ep20: blue cup left, pitcher right.
- ep28: pot left. The rack is untouched (left hand: 2 frames).
- ep30: sweeper right, rack left.
- ep38: dust pan right.

**Passes.** Each pass covers the full episode (ep8 1.0-10.9 s, ep20 0.5-10.05, ep28 0.5-10.95, ep30/38 1.0-13.1). Each pass starts from the previous pass's best final table (seeds 1..4). One pass takes 7-14 min on 44 cores.
- ep8: b1 → b4.
- ep20: b1 → b4.
- ep30: b1 → b4.
- ep38: b1 → b3. It stopped at a plateau (b2→b3 gained 3 %). The ep8 and ep20 b4 passes gained 2-4 %, so those plateau too.
- ep28:
  - b2 (rack weight 2): rack 0.722 / 1.87, pot 0.49.
  - b3 (late pot windows, beam 5, n 60): rack 0.723 / 1.93, pot 0.47.
  - b4 (beam 5, n 60, rack ILC on the left hand too): rack 0.704 / **2.03**, worse than J8's 1.9 deg, so rejected.
  - None of these beats b1 on every object, so ep28 stays at b1.

**Saving, staging and verification.**
- `save.sbatch` bakes the table into the actor's phase weights and runs the official evaluate_policy_component on node01. The FINAL line equals the search value.
- `stage.sbatch` uses the unchanged stage scripts of each episode's world plus F9 lineage files:
  - N8 stage_n8.py for ep8 (teacher kcem/k_lr_20_105) and ep20 (kcem/k3_lr_init);
  - J8 stage_d.py for ep28 (WORLD=world2, teacher cem/w2_k3_sd8) and ep38 (WORLD=world0, teacher ilc/g36/iter3);
  - for ep30, N8's stage30 path (stream E corrected-track wrapper, WORLD=streamE world_T1). The base is built by `mkbase30.py` (= D9 mkbase25 from N8's ep30 staging_N1 base).
- **ep30 mesh swap** (`code/f9_meshswap30.py`, = D9 d9_meshswap25.py for ep30):
  - It asserts that the staged sweeper is A9's source mesh 26c2fa2d (same body frame), then copies A9's swapped mesh byte-identically.
  - It updates readiness.json the way A9 stage.py does, and asserts that every submitted mesh equals A9_ep30_sweeper's (cmp).
  - Physics: every portable file except the policy, lineage, release evidence, bundle manifest and pycache is sha256-identical to N8's ep30 component (scene, collision parts, assets, source). This holds for b1, b3 and b4.
- Build-v2: `run_one.sbatch` (with --time 00:50:00) on node01. `code/verify.sh EP TAG` then checks exactness, scores with score_any vs target_final_v6, cmps the meshes against the current tag, checks the archive, and flags anything worse than the current scores.

## Remaining error / ep28 rack floor
- **ep28 rack (0.72 cm):**
  - The physics rack is static until 8 s (no hand touches it).
  - The target rack moves 0.5-1.9 cm at 1-5 s while the pot is lifted out, and its z goes 0.7-1.2 cm DOWN, into the tabletop.
  - Per-window rack error is 1.0-1.85 cm there and 0.4-0.6 cm elsewhere (`code/rackdiag.py`, `code/winerr.py`).
  - Pushing the rack around with the pot to chase that drift is not physical, and the beam passes that tried (rack weight, rack ILC) did not move it.
- **ep28 pot:** 7.5-9 s (put-down) still shows 8-20 deg per window.

## Caveats
- Scores are proxies against our own reconstruction (target_final_v6), not hidden GT.
- Each final rollout is one deterministic outcome of a deterministic search. It is exact and replayable under build v2, but chaotic holds are not robust to upstream edits.
- These are phase-weight residuals only. Scene, start state, feedback weights and wrappers are unchanged.

## Files
- **code/**:
  - searches: mresF9.py, mlibF9.py, wbeamF9.py;
  - checks: cmpbase.py, handsF9.py, winerr.py, rackdiag.py, verify.sh;
  - jobs: base.sbatch, res.sbatch, save.sbatch, stage.sbatch, savestage.sbatch;
  - ep30 staging: mkbase30.py, f9_meshswap30.py;
  - helpers: env.sh (actor and mesh paths), wins.py, sumlog.sh.
- **runs/ep{8,20,28,30,38}/**: base_eval, b*/ (progress.json, beam_w*.npz, final_*.npz, final.json), save_b*/actor.
- **staging/**: ep8_b1..b4, ep20_b1..b4, ep28_b1, ep30_b1..b4, ep38_b1..b3. **logs/**: base-, res-, svst- <job>.log.
