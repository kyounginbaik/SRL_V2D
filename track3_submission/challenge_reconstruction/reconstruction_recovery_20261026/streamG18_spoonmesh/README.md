# Stream G18: does the kit's spoon-mesh registration land in a wrong body frame? (GT-free study + GT-free mesh fix)

Rules followed:
- No public scan, GT mesh, or scan-derived number is in any mesh or pose here.
- Public scans were read in exactly one place, `code/diag_scan_balls.py`. That script is a scorer-side yes/no check (do the scans contain mocap balls?) and keeps only counts and radii.
- Every truth variant below is built from OUR spoon mesh and OUR ball detections.
- Nothing was overwritten. No package was built, nothing was uploaded, and nothing was committed.
- CPU jobs ran on node01 (debug), each under 50 min: 65811 (study), 65839 (scoring sim), 65840 (study of the fix mesh).

## Bottom line
- **The hypothesis (180-deg roll flip or >10-deg roll offset in the kit registration of our spoon) is NOT supported.**
  - We used the kit's exact `_register_object_body` (25 PCA starts, trimmed ICP, kit seeds of all 10 spoon episodes x 2, random scan frame).
  - Our submitted spoon mesh was registered onto 35 realistic truth variants (700 registrations). Each variant is our spoon with mocap balls, deformed: thickness 0.6-1.2, width +-10 %, length -5 %, scale 0.92-1.08, smoothing, +-1.5 mm offset, 1 mm noise, missing underside/top/tip/bowl-underside/side, balls at r 5-8 mm / mirrored / swapped / up / down / front-surface positions, and combinations.
  - Result: **0 flips and 0 cases above 10 deg.** Body-frame error was median 1.2, p90 3.9, max 7.0 deg; roll |p90| 3.8 deg. Over all 56 variants, including ones without balls, the max is 8.1 deg.
  - The spoon bowl fixes the roll. The balls are only about 6 % of the surface area and do not decide the registration.
- **Real but small mesh defects were found, and a GT-free fix was built.**
  1. Our mesh (5fcd334c, used in 8 of 10 spoon episodes) has **no mocap balls**. The scorer-side diagnostic shows the public GT scans DO contain the balls (3-4 spheres, r ~7 mm, on 9 of 10 public objects), so the hidden spoon scan almost certainly has them.
  2. It carries a **fully internal, inverted inner shell**: component 2, 630 v / 1256 f, every vertex 0.5-5.8 mm inside the outer surface, volume -20 cm3. That is **26 % of the surface area**, and the kit samples registration points by area, so a quarter of our registration points lie on a surface no scan can have.
  - Fix mesh: `mesh/wooden_spoon_g18.obj` (md5 47d9f9bf). It is the outer surface plus 3 icospheres (sub 2) at OUR pooled ball centres with OUR measured radius of 6.06 mm. It has 1908 v / 3800 f (within 4096, so the kit budget does not decimate it) and is in the same body frame, so no pose changes. ep15/16 version: `mesh/wooden_spoon_g18_sub15_16.obj` (ee5aee4b).
  - With the fix, over the same 700 cases: median 0.35, p90 1.0, max 5.2 deg, 0 flips; anchor error median 0.7 mm vs 1.9 mm, p90 3.3 vs 5.5 mm.
- **Expected effect on official MP-SR: none from orientation, plus a small position/robustness gain (largest on ep32/33/36).**
  - Scoring sim of the v28 components against target_final_v14_g17 (truth = variant in a random scan frame): MP-SR stays 1 for every episode under every variant, with one exception. ep32 with the current mesh fails in 2 of 68 draws (truth axz0.6+bfront_7, spoon position >= 3 cm via the cup-pivot lever). With the fix mesh it passes all 68 (p90 spoon pos 2.54 -> 1.89 cm).
  - The mesh is therefore **not** the explanation of the 3 unlocated MP failures. At most it adds a few tenths of a cm on the lever episodes.

## 1. Our spoon mesh vs the ball model (`code/ballcenters.py`, `out/ballcenters.json`)
- 5fcd334c: 2052 v / 4096 f, 30.8 x 6.3 x 2.75 cm, 2 components. The outer surface is 1422 v / 2840 f (232 cm2). The inner shell is 630 v / 1256 f (81 cm2), fully inside the handle/neck (x -0.16..0.05), inverted.
- **No balls in the mesh.** The handle cross-section is a smooth 1.9 x 1.7 cm, with no bumps at the ball positions.
- GT-free ball centres. Method: stereo ball points (front surface) + r along the viewing ray, mapped to body coordinates with the RGB-clock v10_g12 poses, mode near the spoon_roll_v8 seeds, pooled over 10 episodes, n=3754.

| ball | centre (m, our body frame) | signed dist to our surface (+ out) |
|---|---|---|
| near (neck, y-) | (-0.0098, -0.0090, +0.0045) | +3.9 mm |
| far (neck, y+) | (+0.0312, +0.0244, +0.0016) | +12.1 mm |
| tip (handle end) | (-0.1590, +0.0041, -0.0022) | -4.5 mm (our handle end is fat) |

- Detection radius: median 6.06 mm (IQR 5.3-6.5).
- The old spoon_roll_v8 model is the front-surface median; it sits 1-5 mm from these centres.
- The layout is chiral in x-y: the two neck balls sit at different x on opposite edges, so they would break a pure 180 roll. That turned out to be unnecessary because the bowl already does it.

## 2. Registration ambiguity study (`code/g18lib.py`, `code/regstudy.py`, `out/reg_*.json`, `out/regG_*.json`, `out/reg_summary.txt`)
- Candidates:
  - A = submitted 5fcd334c
  - O = outer surface only
  - B6/B7 = O + balls r 6/7 mm
  - B6f = O + balls at the front-surface model
  - G = the built fix mesh, through the kit packer budget
- For each, 20 registrations per truth (10 kit seeds x 2), each with a random scan frame.
- Error measures: E = Rt^T rot (deg); twist about our long axis; anchor error (mm, through the inverse deformation).

Aggregate over the 35 realistic truths (balls present, no inner shell):

| cand | ang med / p90 / max deg | abs roll p90 / max | >5 deg | >10 / flips | anchor med / p90 / max mm |
|---|---|---|---|---|---|
| A (submitted) | 1.21 / 3.94 / 7.04 | 3.84 / 6.85 | 4.4 % | 0 / 0 | 1.9 / 5.5 / 12.9 |
| O | 0.37 / 0.93 / 5.29 | 0.81 / 4.86 | 0.1 % | 0 / 0 | 1.5 / 5.2 / 15.6 |
| B6 | 0.36 / 0.99 / 5.53 | 0.88 / 5.39 | 0.1 % | 0 / 0 | 0.8 / 3.5 / 13.3 |
| B7 | 0.37 / 1.20 / 4.94 | 1.11 / 4.78 | 0 | 0 / 0 | 0.6 / 2.8 / 10.3 |
| **G (fix)** | **0.35 / 1.04 / 5.18** | 0.95 / 4.79 | 0.1 % | 0 / 0 | **0.7 / 3.3 / 11.1** |

- Worst cases for A: a thinner handle with balls (axz0.6+bfront_7: median 5.4, roll -5.2 deg; axz0.6+bdown_7: 3.0 / 6.9 max) and axz0.8+balls (3.3 / 7.0).
- Without balls, A is fine (<= 3 deg) except when the truth lacks the bowl underside (cutbowl), which is ~3 deg for every candidate.
- Removing the inner shell is the main effect. The balls mainly tighten the anchor/translation.
- Ball radius (5-8 mm) and ball side (mirror, swapped, up/down) do not matter: <= 1.7 deg median for A and <= 0.4 for the fix.
- So a ball mismatch cannot produce a flip or a large roll.

## 3. Do the GT scans contain mocap balls? (scorer-side, `code/diag_scan_balls.py`, `out/diag_scan_balls.json`)
**Yes.** Protruding spheres with >55 % coverage, r 4-9 mm, were found as follows:
- beige_cup 3, blue_cup 4, sweeper 3, pitcher 4, pot 4, lid 3, piece_1 4, piece_2 3 (radius ~6.6-7.7 mm)
- rack: 13 hits (wires/balls mixed)
- dust pan: 0 detected

No position or radius from this check was used. The fix uses our own 6.06 mm.

## 4. Fix mesh and simulated effect on v28 (`code/build_mesh.py`, `code/subframe.py`, `code/scoresim.py`, `out/ss_ep*.json`, `out/scoresim_summary.json`, `out/scoresim_p90.txt`)
- Submitted spoon meshes in v28:
  - 5fcd334c (= ours) in eps 3, 5, 6, 32, 33, 35, 36, 37.
  - **e6c98973** (older single-shell 20000-face mesh, packer-decimated) in eps 15, 16. Its frame is ours plus 1.0 deg / ~1 mm (ICP, `out/subframe.json`). The fix for 15/16 is expressed in that frame.
- Sim setup:
  - Truths: 17 realistic variants x 4 random scan frames per episode.
  - Non-spoon objects are registered exactly onto our target meshes.
  - Poses: v28 package parquet poses vs target_final_v14_g17 trajectories.

| ep | exact-reg proxy spoon cm/deg | A: MP, spoon pos p50/p90/max | O | **G (fix)** | other obj p90 A -> G |
|---|---|---|---|---|---|
| 3 | 0.71 / 18.1 | 1.00, 0.79/1.00/1.44 | 0.78/1.03/1.87 | 1.00, 0.75/0.97/1.38 | pot same |
| 5 | 0.81 / 5.2 | 1.00, 0.86/0.99/1.26 | 0.86/1.10/1.71 | 1.00, 0.84/0.94/1.41 | same |
| 6 | 0.60 / 6.3 | 1.00, 0.64/0.97/1.52 | 0.63/0.86/1.75 | 1.00, 0.61/0.73/1.21 | same |
| 15 | 1.14 / 7.4 | 1.00, 1.38/1.54/1.75 | 1.24/1.36/1.59 | 1.00, 1.25/1.34/1.55 | same |
| 16 | 0.88 / 5.6 | 1.00, 0.88/1.03/1.19 | 0.76/0.89/1.58 | 1.00, 0.76/0.89/1.31 | same |
| **32** | 1.48 / 22.4 | **0.97**, 1.57/2.54/3.08 | 0.985, 1.58/2.02/3.72 | **1.00**, 1.57/1.89/2.47 | cup 1.19/5.9 -> 1.13/4.1 |
| 33 | 0.91 / 7.5 | 1.00, 1.22/1.98/2.23 | 0.95/1.20/2.19 | 1.00, 0.95/1.07/1.80 | cup 1.02/6.2 -> 0.84/4.6 |
| 35 | 0.65 / 6.9 | 1.00, 0.86/1.32/1.57 | 0.75/0.94/1.37 | 1.00, 0.72/0.85/1.51 | - |
| 36 | 0.79 / 4.9 | 1.00, 0.95/1.79/2.22 | 0.83/1.08/2.04 | 1.00, 0.81/1.03/2.08 | - |
| 37 | 0.33 / 1.7 | 1.00, 0.41/0.69/0.90 | 0.34/0.38/0.61 | 1.00, 0.34/0.40/0.76 | - |

Spoon orientation changes by <= 0.8 deg median in every episode; the fix is 0-0.8 deg better.

### What a package would need (not built)
`campaign_v22_20261028/v28/finish_v28.py` takes each component from `campaign_v16_20261023/common_v2_candidates/<tag>/<episode>` (`selection_v28.json`). `assemble_twenty_v2.py` then:
- copies `<component>/meshes/<episode>/wooden_spoon.obj` into the package;
- checks it against `completed.json['meshes'][obj]{path, sha256}`;
- `pack_submission.py --meshes` budget-packs it into `object_mesh_b64`.

Steps to adopt the fix:
1. For each spoon episode, make a mesh-only swap copy of the v28 component (A9 `streamA9_meshswap/code/stage.py` pattern). Copy the legacy component (never overwrite) and replace only `meshes/<episode>/wooden_spoon.obj` as follows:
   - `mesh/wooden_spoon_g18.obj` for eps 3, 5, 6, 32, 33, 35, 36, 37 (identity frame);
   - `mesh/wooden_spoon_g18_sub15_16.obj` for eps 15, 16.
   Update the readiness.json mesh sha. Poses, policy, actions and the portable bundle stay untouched; the sim still uses its own source mesh, so the physics does not change.
2. Re-run build-v2 to produce `common_v2_candidates/G18_ep<NN>_<srcTag>/<episode>`, with a new `completed.json` whose `meshes.wooden_spoon.{path,sha256}` point at the new file.
3. Write `selection_v29.json` = v28 with those 10 tags, and add a `finish_v29.py` copy of finish_v28 (V/OUT/SRL -> v29 paths; videos reused, since the rollouts are identical).

## 5. Other eval objects
- Every other eval object (cups, pot, lid, rack, pitcher, sweeper, dust pan, pieces) exists in the public set and was audited scorer-side by A9 (8/8 kit seeds land in the same twin; only the sweeper and boards were twin-prone, and they were fixed in v24).
- The spoon is the only eval object without a public scan, so no further study was needed.

## Files
- code/:
  - study: g18lib.py, regstudy.py, scoresim.py
  - inputs and fix: ballcenters.py, subframe.py, build_mesh.py
  - scorer-side only: diag_scan_balls.py
  - jobs: py.sh, arr.sbatch, jobs_reg.txt, jobs_regG.txt, jobs_ss.txt
- out/: ballcenters.json, subframe.json, diag_scan_balls.json, reg_*.json / regG_*.json (with per-seed rows), reg_summary.txt, ss_ep*_{0,1}.json, scoresim_summary.json, scoresim_p90.txt
- mesh/: wooden_spoon_g18.obj, wooden_spoon_g18_sub15_16.obj
