# Stream L8 (phase 8, v23): episodes 3, 5 and 6 (white_pot + wooden_spoon) vs target_final_v5, 2026-10-28

Goal (../PHASE8.md, later relaxed by main to <= 1.3 cm / 15 deg per object): per object as close to target_final_v5 as possible,
strictly better than the v22b component on every object, and lower per-frame peaks. Meshes are unchanged (the v22b ones: pot 8cbd3543..,
spoon c9ab88e3..). Nothing was uploaded, deleted or committed. Searches ran on AMD nodes only (node01 / node100). Staging and
build-v2 ran on node01.

## Results (score_any.sh on the build-v2 output, vs target_final_v5; cm / deg; scores in ../scores/cand/<tag>/score.json)
| ep | tag (campaign_v16_20261023/common_v2_candidates/) | component | pot | spoon | MPPE | AUC | exact | v22b component (pot, spoon, MPPE, AUC) |
|---|---|---|---|---|---|---|---|---|
| 3 | **L8_ep03_potlate** (recommended) | ep3/staging_L2/components/episode_000003 | **0.89 / 6.6** | **1.78 / 16.3** | 1.56 | 0.898 | true | A_ep03_roll21: 1.24/9.6, 2.18/22.0, 2.02, 0.846 |
| 3 | L8_ep03_rprel (alternative) | ep3/staging_L1/components/episode_000003 | 1.14 / 9.0 | 1.78 / 16.3 | 1.71 | 0.881 | true | |
| 5 | **L8_ep05_potl** (recommended) | ep5/staging_L2/components/episode_000005 | **1.18 / 4.3** | **1.59 / 11.5** | 1.52 | 0.901 | true | A_ep05_obj05: 1.59/8.2, 2.10/21.0, 2.11, 0.842 |
| 5 | L8_ep05_mmlate (alternative) | ep5/staging_L1/components/episode_000005 | 1.34 / 5.5 | 1.58 / 11.8 | 1.60 | 0.896 | true | |
| 6 | **L8_ep06_slate** (recommended) | ep6/staging_L2/components/episode_000006 | **0.64 / 2.8** | **1.24 / 11.9** | 1.09 | 0.949 | true | F_ep06_obj05a: 1.17/5.0, 1.55/16.4, 1.58, 0.895 |

Checkpoint / portable archive sha256: ep3 L2 9e1f02b876de.. / 92abbb648b7e..; ep3 L1 8ebf9e3684a6.. / dda368a86a90..;
ep5 L2 129c4f1c1852.. / bd9d14161b39..; ep5 L1 2ffc5b3ae673.. / e530b47a6f5d..; ep6 L2 faa23bc8cd2b.. / bffb991eaef8...
Every staging reports portable and common replays exact. Build-v2 `completed.json` shows exact_previous_physical_states_actions true.

Against the relaxed 1.3 cm / 15 deg goal: ep6 meets it on both objects. ep5 meets it on the pot and on spoon orientation (11.5 deg),
but the spoon position is 1.59 cm. ep3 meets it on the pot. The ep3 spoon is 1.78 cm / 16.3 deg. All three are strictly better than
v22b on every object.

## Diagnosis (code/diag.py, code/twist.py, code/perr.py, code/lcontact.py)
- Spoon orientation error is mostly roll (twist about the spoon's long axis) and comes from four sources:
  - in-hand roll lag or lead while the spoon is held;
  - a pickup roll spike (ep3 2.4-3.0 s: about 90 deg; ep6 1.5-2.5 s: about 35 deg);
  - post-release rest tilt (ep3 38 deg, ep6 19 deg);
  - in ep5 and ep6, the physics start rest (the spoon settles about 10-13 deg before contact; this is part of the scene, so no fix).
- The target spoon itself jitters. A Gaussian smoothing of 2 frames already differs from the target by 0.6-1.0 cm / 3.6 deg, and
  4 frames by 1.4-1.8 cm / 6-7 deg. The ep5 spoon position (stirring, 2-3 cm fluctuating error and no consistent bias) is close to
  this floor.
- Benign windows: the late hold, the release, and the whole left-hand pot hold in ep5 and ep6.
- Chaotic window: the ep3 full-turn roll at 3.4-9.6 s. In a random-edit lottery, the median spoon error was 4-7 cm and about 0.7 %
  of samples improved.

## What worked
1. **Phase-weight ILC on the shipped actor** (code/l8res.py = streamH actor_res.py generalised).
   - The residual is baked into the LearnedPolicy phase weights. Every search evaluation is a closed-loop rollout scored
     scorer-faithfully (plib.py = streamB blib with pot/spoon).
   - A prefix cache is used. The RBF leak before the cache frame is checked to be exactly 0, and the cached prefix is checked to
     equal the full rollout.
   - Cost: worst ratio vs (1.2 → 1.3 cm, 12 → 15 deg) + 0.25·sum + p95 peak term + a hard penalty if any object is worse than v22b.
   - Windows are chained (each run starts from the previous best residual table).
   - The ep6 late window (8.2-10.6 s) alone gave spoon 1.55/16.4 → 1.32/13.1.
2. **Left-hand pot ILC**, a new channel set: rigid-hold correction of the left wrist toward the target pot.
   - ep6: pot 1.17/5.0 → 0.66/2.9. ep5: 1.59/8.2 → 1.18/4.3. ep3 (late window 9.8-15.6 s only): 1.14/9.0 → 0.89/6.6.
   - In ep3, early pot windows perturb the chaotic spoon hold.
3. **Release primitive** (code/l8rel.py = streamI release_cem.py with generalised episode, meshes and cost; registered ReleasePolicy wrapper).
   - ep3: the rest tilt after release dropped, giving spoon 1.85/17.6 → 1.78/16.3.
   - Fixed a latent bug from the stream I copy: spawn workers kept the default start bounds 13.6-15.2 s, so the ep6 search was a no-op.
4. **ep3 pickup lottery**: 1024 random right-wrist residuals over 1.6-3.6 s, keeping the best exact rollout (1.97 → 1.85 cm after the late ILC).
5. **Non-holding-hand clearance (after P8)** (code/l8clear.py).
   - In ep5 the left thumb presses the spoon (3.0-3.5 s and 5.75-7.75 s).
   - Moving the left wrist 1 cm in xy away from the spoon over 5.4-8.2 s (bias channel, baked into the phase weights) made the hold
     robust to early edits: across 16 random right-wrist edits, the median spoon error went from 2.81/33.7 to 1.65/12.9, and the
     share under 3 cm from 56 % to 100 %.
   - Mid-window CEM and ILC on top of it, plus pot ILC, then gave L8_ep05_potl.
   - In ep3 the left hand touches the spoon only at 9.25-9.5 s, and clearance gave no gain there. The best ep3 actor is already
     robust (median 2.03/18.3).
   - In ep6 the left hand never touches the spoon.

## What did not work
- ep3 mid window (3.4-9.6 s): ILC, knot CEM at small std, and a rotation-only roll ILC/CEM over 4.2-6.6 s gave no strict improvement.
  The random-edit lottery won only when its result was replaced by later windows.
- Pickup ILC in ep5/ep6 (no gain on top of the late tables). Pot ILC over the early ep3 windows (it breaks the spoon).

## Caveats
- Every component is the v22b actor plus finite residuals or primitive parameters. The scene, start state, meshes and feedback
  weights are unchanged. The teacher recorded in the lineage is the v22b component's teacher (not shipped).
- The ep5 component carries the left-wrist clearance bias; the pot follows the left hand, and its error still improved.
- Search outputs live in runs/ (a symlink to tmp/runs, because the jobs ran with cwd=TMPDIR). Actor result.json files reference the
  tmp/runs paths.
- These are proxy scores against our own reconstruction, not hidden GT.

## Code (code/)
plib.py, l8res.py (ILC / CEM / rand / eval / save; right + left wrist + bias), l8rel.py (release CEM), l8clear.py (clearance and
robustness test), diag.py / twist.py / perr.py / lcontact.py (diagnostics), res.sbatch, stage.sbatch (calls the unchanged
stage_compose_component.py of streamA (eps 3/5) or streamF (ep6)), env.sh, prog.sh. base/ep*_v22b.json are the strictly-better references.
