# Stream C30: coworker compose_v1..v8 reconstructions -> Track 3 packages v30c1..v30c8 (2026-11-01)

Rules followed:
- Nothing was uploaded or committed.
- Submitted poses come only from build-v2 policy rollouts. Every adopted component has `exact_previous_physical_states_actions = true`.
- Downloaded coworker files were read as data only (`np.load(allow_pickle=False)`, json).
- Public GT was used only inside scorers. One exception, a scorer-side provenance diagnostic, is described in section 2.
- No other users' jobs were touched.
- **Jobs ran on node100, not node01**, because node01's /data1 mount hung at 13:20. The node01 jobs 67211 and 67214 were cancelled. C30 stayed at 48 CPUs or fewer, apart from a ~2 CPU overshoot for a few minutes.

**Incident (13:05-13:22).** The first `c30build.py` wrote object_poses.npz *into* `reconstruction_recovery_20261026/selected20_final_v14_g17` for 17 episodes.
- `target_final_v14_g17` was never touched.
- Main restored the bundle with user approval. The damaged copies are in `reconstruction_recovery_20261026/quarantine_v14_g17_c30bug_20261101`.
- The script is fixed. It now builds v14 from its sources (v12b_g15 + v13_g16 ep3) and checks that the result equals target_final_v14_g17. It refuses any output path inside a source bundle, and every write is guarded to stay under OUT.
- Every C30 output was rebuilt or checked against the fixed build:
  - `target_compose_v8_on_v14` and `selected20_compose_v8_on_v14` poses are identical to the fixed rebuild.
  - The v7/v6/v2/v1 bundles that aborted in the bug window are marked `INVALID_DO_NOT_USE.txt`. Their valid builds are `*_fix`.
  - `out/changes_v8.json` (its vs-v14 statistics are wrong) is superseded by `out/changes_v8_fix.json`.

## 1. Transplant of compose_vK onto selected20_final_v14_g17 (`code/c30build.py`)
The coworker bundles are `recon_v26_selected20_final_v8_g10b` plus their changes. Our current reconstruction is v14_g17, which is v8_g10b plus three steps:
- **G12:** ep32 spoon per-frame roll (v10_g12).
- **G15:** clock-consistent camera, then a retime onto the stereo clock (+0.875 to 0.925 frame; ep20 piecewise 0.925 -> 1.85). This gives v12b_g15.
- **G16:** ep3 spoon constant -13.5 deg roll.

Per episode and object, in the RGB camera frame:
1. **Changed frames.** These are frames where C_vK != C_v8g10b (more than 1e-6 m or 1e-4 deg). Whole-track replacements come from the coworker's own manifests (`compose_windows_v6.json`, `compose_whole_v7/v8.json`).
2. **Moving into v10 space.**
   - Window and smoothing frames: C_vK @ B, where B = inv(C_v8) C_v10 is the G12 body correction (identity except the ep32 spoon).
   - Whole tracks: theirs on every frame, with no G12 roll.
3. **G15 step.** Exactly as in G15 build_bundle: world := G15 camera @ C, then `retime_npz` with the per-episode delta.
4. **G16 step.** The ep3 spoon is right-multiplied by the -13.5 deg roll. In compose this spoon has windows only. Their ep3 window poses are small corrections of our track (median roll 4.9 deg), not an independent roll, so the roll was applied.
5. **Untouched objects.** The chain reproduces v14 to within 1e-8 (asserted), and the v14 arrays are copied bit-exactly. Every untouched target trajectory and mesh is bit-identical to target_final_v14_g17.

Other checks on every version:
- world_from_rgb equals v8_g10b.
- Meshes are md5-identical to v8_g10b.
- object_observed and camera_valid are identical.
- World/camera consistency is within 3e-16.

Outputs:

| | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 |
|---|---|---|---|---|---|---|---|---|
| bundle (reconstruction_recovery_20261026/) | selected20_compose_v1_on_v14_fix | ..._v2_on_v14_fix | selected20_compose_v3_on_v14 | ..._v4_on_v14 | ..._v5_on_v14 | ..._v6_on_v14_fix | ..._v7_on_v14_fix | selected20_compose_v8_on_v14 |
| target (campaign_v22_20261028/) | target_compose_v1_on_v14 | ... | ... | ... | ... | ... | ... | target_compose_v8_on_v14 |

Change lists:
- Exact (ep, object, frames) lists with max/mean deltas vs v14: `out/changes_vK_fix.json`.
- Markdown tables: `out/change_tables_v8_v7.md` and `out/change_tables_v6_v1.md`.
- Target hashes per episode: `out/target_hashes.json`. These were used to dedupe runs. For example, the ep3 target is identical in v7 and v8.

Note: v1-v4 are not strictly nested. For example, v2 does not contain v1's ep10/14/28/30 windows. v5 onward is cumulative.

Whole tracks (theirs) vs v14 (time-mean / max):
- v6+: ep32 spoon 1.96 cm / 46 deg (max 171 deg).
- v7+: ep3 pot 0.75 / 5.2; ep35 spoon 1.06 / 27 (f0 roll differs by 84 deg); ep36 spoon 1.74 / 40.
- v8: ep8 pot 0.68 / 4.0; ep16 spoon 0.73 / 30.5; ep28 pot 0.93 / 12.3.

The v5 smoothing changes 11 objects by up to 2 cm / 21 deg per frame. Cups dominate, and cups are symmetric.

## 2. Pot provenance (ep3/8/28 whole tracks from compose_eval_20261021)
`code/potprov.py`, `out/pot_provenance.json`. This is scorer-side only: the public scan is read only to classify a mesh, and nothing is fitted or copied from it.
- **Method.** The compose_eval pot meshes (121k vertices, a placed similarity copy of one mesh per episode) were compared with the public white_pot scan. The comparison is ICP from PCA starts at metric scale and at scale-normalised scale.
- **Pots.**
  - Chamfer 3.13-3.15 mm, p90 5.0 mm.
  - Extents: short axis 9.8-10.1 cm vs the scan's 13.2 cm.
  - For comparison, our own pot gives 3.66 mm.
- **Control.** The c21 ep20 pitcher, which is documented as the public scan, gives 0.00 mm with identical extents.
- **Verdict:** the pot track mesh is **not** a public-scan copy. No pot variant without it was needed.

The compose README also documents the following:
- The ep20 pitcher was public-scan-tracked on 10-21 and 10-26. Only the ep25 boards and the ep1 lid were remeshed public meshes on 10-26. None of these are in the compose_v1-v8 whole tracks.
- The sweeper team mesh was fused on GT-ICP poses (10-23 note). The ep30 sweeper whole track was excluded by the coworker.

## 3. Retargeting
1. **v29 rollouts scored vs every distinct target.** Labels `scores/cand/C30_v29epNN_vs_<c8|hash>`, summarised by `code/summ.py`.
   - v1-v5: no episode needed a run. Every changed episode stays within ~0.06 MPPE of v29@v14, except ep20 with c5.
   - The ep20 c5/c6/c7/c8 targets give 0.448 -> 0.688 MPPE (blue cup 0.36 -> 0.61 cm). It still passes with a large margin, so v29 is kept.
   - v29 fails MP against the v6/v7/v8 whole-track spoons:
     - ep32: c6-c8 1.25/8.0 and 2.20/49.4 deg.
     - ep35: c7/c8 17.2 cm / 83.9 deg. Their frame-0 spoon roll is 84 deg from ours. The spoon is the only object, so the kit's frame-0 anchor alignment rotates everything.
     - ep36: c7/c8 3.19 / 42.2.
   - v29 also fails ep16 against c8 (spoon 30.9 deg).
2. **H13-style re-retargets** (`h13a/`, `h13b/` = H13 code with TGT from env `C30_TGT`; `code/beam.py` reruns the source pass recipe):
   - Each run starts from v29's actor and uses v29's meshes (the G18 spoon for spoon episodes).
   - Every INIT reproduced the v29-vs-c8 score exactly.
   - Pipeline per run: save, then the unchanged source stagers, then a G18 mesh-only swap for spoon episodes (`code/g18swap.py`), then build-v2, then score (`code/svst.sbatch`).

| ep | target | tag | vs own target (MPPE; objects cm/deg) | v29 on the same target | vs target_final_v14_g17 | used in |
|---|---|---|---|---|---|---|
| 3 | c7 = c8 (d4ca1b6d) | C30_ep03_c8c1b (H13b a3+a3b recipe, 2 halves) | 1.111; pot 0.66/5.4, spoon 1.13/13.9 | 1.566; 0.89/5.2, 1.81/18.3 | 1.287 (v29 1.011) | v30c7, v30c8 |
| 8 | c8 (737caf6d) | C30_ep08_c8a1 (H13b a2 recipe) | 0.564; cup 0.58/3.5, pot 0.47/1.7 | 0.668 | 0.685 (0.376) | v30c8 |
| 16 | c8 (d2e55921) | C30_ep16_c8c1 (wbeamH roll-ILC, E9 b8 windows) | 0.801; spoon 1.05/8.1, **MP 0 -> 1** | 1.409; spoon 1.14/30.9, MP 0 | 1.352 (0.778), spoon 29.5 deg | v30c8 |
| 28 | c8 (254aa5c9) | C30_ep28_c8b1 (H13a b1 recipe) | 0.767; pot 0.79/2.9 | 1.008; pot 1.06/12.4 | 1.053 (0.675) | v30c8 |

All four tags are build-v2 exact true, and their meshes are sha-identical to the v29 components.

**Not done:**
- **ep32, ep35 and ep36 against the v6-v8 whole-track spoons.**
  - Window beams cannot reach them: they need roll flips of 84-171 deg, at frame 0 (ep35) or in hand.
  - Reaching them needs a world, teacher and start-state rebuild on the coworker track.
  - So v30c6/c7/c8 knowingly fail these episodes against their own target. v30c6 fails ep32 only.
- **Second passes.** No second beam passes were run.
- **ep20 against c5+.** Not re-run (v29 passes with margin).

## 4. Packages
- **Builder:** `code/finish_v30c.py K {render-args|assemble}`. It is the v29 finish script with these differences:
  - the selection is `campaign_v22_20261028/selection_v30cK.json`;
  - videos are reused when the rollout is identical, otherwise rendered once per tag on node100 into `streamC30_compose/videos/<tag>`;
  - it scores against its own target and against target_final_v14_g17;
  - no knockouts.
- **Zip:** `code/mkzip.py`.
- **Coworker-mesh variants** (`code/cwalign.py`, `code/cwpkg.py`, per the user instruction of 2026-11-01 night):
  - **Alignment.** Each coworker final_episode_meshes_20261029 mesh is rigidly expressed in our body frame by multi-start ICP onto our v29 mesh. There is no scaling, and only the rigid transform is applied.
  - **Check.** The new mesh is registered onto ours with the kit `_register_object_body` at 10 seeds (the kit seed + 1..9). Results:
    - Pots, lid, rack, pitcher, sweeper, dust pan and beige cups: all seeds under 5 deg.
    - Blue cups (ep14/20/32): rotations up to 176 deg, but all about the symmetry axis (within 4 deg), so accepted.
    - **wooden_piece_1/2: 180 deg twin flips across seeds, so rejected. Our boards are kept.**
  - **What changes.** Only the step-0 `object_mesh_b64` cells and `meshes/` differ from the pose-equivalent package (asserted). Poses, results and robot_actions are identical. The spoon stays G18.
  - **`_cwmesh`:** 24 objects swapped.
  - **`_cwmesh_nogt`:** our meshes for plastic_dish_rack, white_pot_lid and water_pitcher; 17 swapped.
  - **What the coworker README says** about the 10-29/10-30 meshes:
    - rack and lid: "proxy-guided", deformed toward a hand-specified target, "no GT geometry";
    - pitcher: data-only;
    - the earlier rack/lid/pitcher versions were GT-guided.
    The user instruction treats them as GT-guided. Both variants were built, and each package README labels this.

Packages are in `challenge_submissions_SRL/track3_v30c{K}[_cwmesh|_cwmesh_nogt]_20261101/`, each with a `.zip`. The parquet sha256 values and scores are in the table below.

**v30c1..v30c6 have the same rollouts and parquet as v29** (7f44ef5e). No re-retarget was needed for them. Their cwmesh and nogt variants are therefore also identical across K: a2523f03 and 37038573. Uploading more than one of them per mesh variant gains nothing.

Submit (do not run without the user):
`cd <pkg>/all_20_episodes/submission_kit && /data1/SRL_V2D/challenge_submissions_SRL/.venv/bin/python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m "v30c<k>" --wait`

## 5. Package scores (local proxy, kit scorer; GT-free targets)
See `out/package_summary.md` (generated at the end; includes dry-run status, zip sha256 and the cwmesh rescoring `out/pkgscores/`).

## Files
- `code/`: c30build.py (fixed), c30build_BUGGY_wrote_into_v14.py.txt (the faulty version, kept for the record), potprov.py, mkjobs.py, summ.py, chgtable.py, beam.py, beam.sbatch, svst.sbatch, g18swap.py, score_arr.sbatch, cmd.sbatch, finish_v30c.py, assemble.sbatch, render.sbatch, mkzip.py, cwalign.py, cwpkg.py, pkgscore.py/.sbatch.
- `h13a/`, `h13b/`: C30 copies of the H13 code (target from env); `runs/` beam passes and saved actors; `staging/` components; `videos/` new renders.
- `out/`: changes_vK_fix.json, change_tables_*.md, target_hashes.json, v29_components.json, pot_provenance.json, settings_ep16_recipe.json, pkgscores/, package_summary.md.
