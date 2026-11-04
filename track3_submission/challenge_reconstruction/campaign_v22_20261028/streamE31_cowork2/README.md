# Stream E31: cowork2 E14 bundles (HGA / HGB) -> Track 3 packages v31a / v31b (+ cwmesh variants), 2026-11-01

Inputs (untrusted data, loaded only with np.load(allow_pickle=False) / json / trimesh OBJ parse; no code from them was run):
`challenge_submissions_SRL/cowork2_recon_gdrive_20261101/extracted/recon_E14_best_B2A_S1_{HGA,HGB}_sharpa`.

Rules followed:
- Nothing was uploaded or committed, and no other user's job was touched.
- Poses come only from build-v2 policy rollouts.
- No public scan or GT mesh was used. The scorer targets are the coworker bundles.
- Compute: **node01's /data1 mount hung on 2026-11-01** (my 3-min write test there produced no output). With main's approval all E31 CPU jobs ran on **node100** (partition debug, no GPU, each job <= 50 min, E31 <= 48 CPUs; one moment reached about 50).
  - The stuck node01 jobs 67212/67213/67232 were cancelled. They stay in COMPLETING until node01 recovers.
- Nothing earlier was overwritten. Superseded E31-internal outputs were kept:
  - `cwmesh`, `cwmesh_v2`, `cwmesh_v3`, `cwmesh_v4`;
  - `worlds/HGA`, `worlds/HGA_s`;
  - `runs/save_HGA_ep33_a1`, `runs/save_HGA_ep33_a1s`;
  - `selection_v31{a,b}_draft_ep32w2.json`.
- Exceptions:
  - I deleted my own empty `runs/HGA/ep32/a1` dir from a job cancelled seconds after launch.
  - `v31a/videos/episode_000032` was re-rendered for w3 over the unpackaged w2 render.

## 1. Format check (`code/verify_format.py`, `out/verify_format.json`)
- Both bundles have the same file set as `selected20_final_v14_g17`.
- 208 files are byte-identical, including every mesh, camera.npz, intrinsics and meshes.json.
  - My array_equal camera flag reads False only because of NaNs; the sha256 values are equal.
- Only `object_poses.npz` differs, in all 20 episodes. It has the same keys, dtypes and shapes, the same names, and the same frame counts (335 ... 263). Only `world_from_object` and `camera_from_object` changed.
- Extra files are reports and annotations only.
- HGA vs HGB: the object_poses differ only in eps 3, 5, 6, 32, 33 (the other 15 are byte-identical).
- Target change vs v14 on the spoon of eps 3/32/33 is a pure roll about the long axis (axis alignment 0.98-1.0):
  - ep3 mean 24 deg;
  - ep32 21 deg;
  - ep33 33 deg, with 7 windows at 110-160 deg.

## 2. Targets
- `campaign_v22_20261028/target_E14_HGA` and `target_E14_HGB` were built with `code/build_recon_target.py`, the same builder as target_final_v14_g17.
- Their meshes are cmp-identical to target_final_v14_g17 (old spoon c9ab88e3).
- Deduplication: one target per variant. 15 episodes are identical between the two.

## 3. v29 vs the new targets (kit-faithful score_vs_compose of the v29 package, G18 spoon mesh; `out/v29_vs_E14_*_n100`, `out/v29_scores_3targets.json`)
- v29 vs HGA and vs HGB: MP 0.85, MPPE 0.924, AUC 0.966.
- Fails (spoon orientation):

| ep | HGA | HGB |
|---|---|---|
| 3 | 1.03 cm / 30.9 deg | 30.9 deg |
| 32 | 1.94 cm / 31.0 deg | 31.0 deg |
| 33 | 1.18 cm / 36.4 deg | 36.2 deg |

- The other 17 episodes pass with margin (worst ep36 1.45 cm, ep25 12.5 deg). Their changes vs v14 are small (<= 0.6 cm / 4 deg mean). **Decision: keep v29 for the 17**, including eps 5 and 6 (HGA 0.63/4.3, 0.91/7.0 and 0.49/2.1, 0.70/6.4; HGB equivalent).

## 4. Re-retarget of eps 3, 32, 33 (vs target_E14_HGA, from the v29 actors; every adopted run is build-v2 exact true)
**Code.** `code/` holds H13a/H13b/H14 copies with TGT = `$E31_TGT`. In-loop scoring uses the v29 G18 meshes. The in-loop objects reproduce score_any exactly.

**Beams.**
- ep3 and ep33: wbeamH (H13b), 0.5 s windows, beam 4, roll-ILC alphas up to 2, hinge 4 @ 2 cm.
- ep32: the H14 monotone full-horizon beam (from 3.5 s) found **0 improving edits in 8 windows**, because every edit loses the spoon later. I switched to the windowed wbeam4H13 (H12 recipe, from 3.5 s, roll alphas up to 3).

| ep | tag (adopted bold) | vs HGA | vs HGB | vs v14_g17 |
|---|---|---|---|---|
| 3 | E31A_ep03_a1 | 1.02/22.1, MPPE 1.135 | 1.01/22.2 | 1.05/32.8 (MP0) |
| 3 | **E31A_ep03_a2** | pot 0.43/3.5, spoon **0.80/15.3**, MPPE 0.899, AUC 0.969 | 0.80/15.4 | 0.86/31.7 (MP0) |
| 32 | E31A_ep32_w1 | spoon 1.70/24.7 | 1.68/24.7 | 1.73/27.7 |
| 32 | E31A_ep32_w2 | spoon 1.66/22.2 | 1.66/22.2 | 1.73/30.5 (MP0) |
| 32 | **E31A_ep32_w3** | cup 1.04/4.7, spoon **1.68/20.3**, MPPE 1.679, AUC 0.903 | 1.68/20.3 | cup 1.05/5.0, spoon 1.74/29.2 (MP1, thin) |
| 33 | **E31A_ep33_a1t** | cup 1.07/7.4, spoon **1.01/15.2**, MPPE 1.306, AUC 0.924 | 1.05/15.2 | 1.14/36.4 (MP0) |

- The ep32 w3 v14 scores are taken from the package's target_final_v14_g17_scores.json.
- Each run that was tuned on HGA also passes HGB with the same margin. **v31a and v31b therefore use one selection**, and their parquets are byte-identical.
- The new target conflicts with v14: eps 3 and 33 now fail vs v14_g17. That is expected from the roll re-fit.

**ep33 staging.** The Stream B stager gates on the proxy MP-SR against the world's internal reference, which still carries the old roll (35.7 deg there).
- `code/mkref.py` copies the W1 episode dir and source dir to `worlds/HGA_r`. In the copy, every frame of the object reference is replaced by the E14 target in the physics frame (mlibH.target_phys, which is what the ILC uses), with the scoring-invariant cup spin matched at f0.
- Scene, robot reference and camera are unchanged.
- `code/stage_b_e31.py` is stage_b.py with one change: its directory check also accepts `streamE31_cowork2/worlds`. Every gate is unchanged.
- Result: proxy 1.0; the build-v2 common_evaluator gate also passed.
- eps 3 and 32 passed their original stagers unchanged.

**Pipeline.** `code/svst.sbatch` runs: save (official eval), the unchanged source stager (+ E31 lineage), the G18 mesh-only swap (`g18swap.py`, meshes asserted == v29), build-v2, then `vs.sh`.

## 5. Packages (finish chain = `code/finish_v31.py`, a copy of v29/finish_v29.py with paths, selection and targets re-pointed)
- Changes vs the v29 chain:
  - new renders only for eps 3, 32, 33 (`v31a/render.sbatch`, node100); the others reuse the v29 videos after a trajectory identity check;
  - scored vs its own E14 target and vs target_final_v14_g17;
  - no knockouts.
- Selections: `campaign_v22_20261028/selection_v31{a,b}.json` (identical) and `selection_v31{a,b}cw.json` (CW_ tags).

| package (challenge_submissions_SRL/...) | meshes | parquet sha256 | vs own E14 target | vs v14_g17 |
|---|---|---|---|---|
| track3_v31a_HGA_20261101 | v29 component meshes (G18 spoon) | 2f9f3fecaffd931efe210619b7b2d0d77c9fe8821d52e0ff09da0cb07c6c5244 | MP 20/20, SP 1, MPPE 0.865, AUC 0.9665, worst ep32 spoon 20.3 deg | 18/20 (3, 33 fail), MPPE 0.923 |
| track3_v31b_HGB_20261101 | same (byte-identical parquet to v31a) | 2f9f3fec... (same) | MP 20/20, MPPE 0.867, AUC 0.9666 | 18/20 |
| track3_v31a_HGA_cwmesh_20261101 | coworker bundle meshes (cwmesh_v5) | 2d8dcfbeb3910114153da7a8babbce91fc8151f2c7582a93f6b59896a9959487 | MP 20/20, MPPE 0.818, AUC 0.9700, worst 19.9 deg | 18/20, MPPE 0.869 |
| track3_v31b_HGB_cwmesh_20261101 | same (byte-identical parquet to v31a_cwmesh) | 2d8dcfbe... (same) | MP 20/20, MPPE 0.819, AUC 0.9699 | 18/20 |

- The v29-mesh vs cwmesh pairs differ only in mesh payload rows: 26 rows (`out/verify_v31{a,b}_vs_v31{a,b}_cwmesh.json`), with 0 non-mesh columns differing.
- v29 vs v31a: only the rows of eps 3/32/33 differ (pose columns + checkpoint_sha256), and no mesh rows (`out/verify_v29_vs_v31a.json`).
- The local cwmesh scores look better partly because the target meshes are the bundle meshes.
- Kit dry-run: OK for all 4 (9,120 rows / 20 episodes, valid, quota 19 left; `logs/dryrun_all.txt`).
- Zips (python zipfile, deflate, no __pycache__, testzip OK):

| zip | sha256 |
|---|---|
| track3_v31a_HGA_20261101.zip | d89e56b3... |
| track3_v31b_HGB_20261101.zip | 93e6f74e... |
| track3_v31a_HGA_cwmesh_20261101.zip | b598d839... |
| track3_v31b_HGB_cwmesh_20261101.zip | 26394b2b... |

- Each SRL package also has a challenge_submissions/<pkg> twin, as in v29.

## 6. cwmesh: how the coworker meshes enter (per object; `cwmesh_v5/cwmesh_plan.json`)
- The robot poses are in each component's body frame, so a bundle mesh can be shipped byte-exact only where that frame equals the bundle object frame.
- Derived meshes (identical faces) are measured exactly by vertex correspondence. All others use the best of 10 kit-registration seeds (`MP-SR._register_object_body`, bundle -> component mesh, lowest surface residual), checked against the scorer's own registration.
- Rule (main): byte-exact if the frame differs by <= 0.5 deg and <= 1 mm; otherwise the bundle geometry is rigidly re-expressed in the component frame (coworker size kept, no rescale).
- Every cwmesh component: build-v2 exact true, rollout arrays identical to its v29-mesh twin, MP pass vs HGA and HGB (`logs/cwv2-*`).

**Byte-exact (bundle file bytes):**
- ep1 pot and lid;
- spoon in eps 3, 5, 6, 32, 33, 35, 36, 37 (= v28 spoon c9ab88e3; G18 shares its frame);
- ep8 and ep10 beige cup; blue cup in eps 10 and 33;
- rack in eps 14, 15, 16, 28, 30 (the components are the bundle rack scaled about the frame origin, rotation 0);
- dust pan in eps 19 and 38; ep20 pitcher;
- ep32 blue cup: the component cup 3c3c5328 is the bundle cup with z scaled 0.900 about its bottom (x/y unchanged, exact affine). That is a shape edit in the same frame. A similarity fit's 9 mm "offset" is an artefact of that edit, so this is byte-exact.

**Re-expressed (bundle geometry, rigid transform into the component frame):**

| object | transform | method / notes |
|---|---|---|
| white_pot eps 3/5/6/8 | 2.28 deg | vertex correspondence; the component is a re-scaled + rotated pot |
| boards ep25 | 0.11 deg / 2.8 mm and 0.53 deg / 5.5 mm | exact rigid, rms 0 |
| sweeper ep30 | 0.82 deg / 1.0 mm | exact rigid |
| blue cup eps 14/20 | axes permuted (~90 deg) | kit; all 10 seeds on one twin, axis gaps <= 4 deg; scorer agrees within 1.3 / 1.0 deg |
| white_pot ep28 | 173.5 deg | kit 10/10; scorer agrees within 0.56 deg |
| sweeper ep19 | 3.1 deg | kit 8/10 seeds on the best twin; 2/10 on the 180-deg twin with 0.08 mm worse fit. The scorer's registration (validated by v29's ep19 7.4 deg) agrees with the chosen twin within 0.22 deg |
| spoon eps 15/16 | 0.62 deg / 3.75 mm | kit 10/10, vs the e6c98973-frame spoon; scorer agrees within 3 deg |

- Deviation from a literal "kit registration everywhere": for derived meshes I used vertex correspondence, which is exact. The kit registration centres on surface moments, so it would have shifted the ep32 cup by 9.6 mm and the racks by 1.4-2.7 mm wrongly.

## 7. Submit commands (not run)
```
cd /data1/SRL_V2D/challenge_submissions_SRL/track3_v31a_HGA_20261101/all_20_episodes/submission_kit && /data1/SRL_V2D/challenge_submissions_SRL/.venv/bin/python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m "track3_v31a_HGA_20261101" --wait
cd /data1/SRL_V2D/challenge_submissions_SRL/track3_v31a_HGA_cwmesh_20261101/all_20_episodes/submission_kit && /data1/SRL_V2D/challenge_submissions_SRL/.venv/bin/python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m "track3_v31a_HGA_cwmesh_20261101" --wait
```
- v31b and v31b_cwmesh have byte-identical parquets to the two above, so submitting them gives the same score. Their commands are the same with the b paths and names.
- Before submission, publish the evaluator build-v2 commit (as noted in every package README).

## Not done / caveats
- No HGB-specific beams were run: the HGA-tuned runs score the same on HGB.
- ep32 has the thinnest margin (20.3 deg, 1.68 cm).
- All scores are proxies against the coworker reconstruction, not hidden GT.
- The new target conflicts with v14_g17 on eps 3/33. Which is closer to the hidden GT is unknown. The README of the coworker bundle reports a conflict with the team's G11 inference for ep3.
- Holds are chaotic. Each rollout is one exact, replayable outcome, not a robust policy.
