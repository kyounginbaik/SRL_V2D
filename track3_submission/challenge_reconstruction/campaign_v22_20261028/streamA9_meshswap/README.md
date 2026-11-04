# Stream A9 (phase 9): half-turn / marker-side audit of every v23b object + mesh-only swaps, 2026-10-28

Rules (PHASE9.md) followed: no public scan or GT mesh is in any staged mesh. Every replacement mesh is our own final_v6 mesh
(`reconstruction_recovery_20261026/selected20_final_v6/<ep>/meshes`) under a rigid transform. Scans appear only inside the kit
registration, in `code/regcheck.py` and Y9's `twin_margin.py` (scorer-side). Poses, policies, actions and the portable bundles
(scene, collision parts, source meshes) are untouched. Earlier outputs were copied, never overwritten. No upload, no commit. ep32 was audited only.

## Result
**Two v23b objects are on the wrong half-turn twin. Both are fixed by a mesh-only swap. Build-v2 is exact for every new tag.**

| v24 swap | episode | replaces | build-v2 exact | score_any vs target_final_v6 (cm / deg) | twin (kit, scan-side) |
|---|---|---|---|---|---|
| **A9_ep30_sweeper** | 30 | v23b N8_ep30_k2c2048 | true | **sweeper 0.90 / 6.1, rack 0.54 / 4.4, MP-SR 1, MPPE 0.82, AUC 0.961** (v23b: 18.51 / 29.7, 7.85 / 177.0, MP-SR 0, MPPE 14.45, AUC 0.226) | old: 180 deg from v6 on 8/8 seeds. New = v6 convention; margin 0.010 cm at the kit seed, 1.00 of 17 seeds |
| **A9_ep25_piece2** (recommended) | 25 | v23b J8_ep25_b48m4 | true | piece_1 1.27 / 14.5, piece_2 1.19 / 5.7, MP-SR 1, MPPE 1.47, AUC 0.898 (v23b: 1.27 / 14.5, 1.14 / 5.6, MPPE 1.44, AUC 0.898) | piece_2 old: 180.0 deg from v6 at the kit seed, margin 0.002, 0.35 of seeds. New: margin 0.015, 1.00 |
| A9_ep25_boards (alternative) | 25 | v23b J8_ep25_b48m4 | true | piece_1 1.31 / 14.6, piece_2 1.42 / 5.7, MP-SR 1, MPPE 1.60, AUC 0.884 | additionally piece_1: margin 0.007 -> 0.019 at the kit seed, 0.88 -> 0.94 of seeds |

- ep30: the swap cures the official-style failure. The local score now matches N8's own proxy (0.85 / 6.2, 0.57 / 3.8).
- ep25: the local target is our own mesh, so the local score cannot see the twin. Old and new both register correctly there.
  - A9_ep25_piece2 is locally the same: +0.05 cm on piece_2, everything else equal. It removes a kit-seed flip, which would cost 180 deg on piece_2 against the scan.
  - A9_ep25_boards also swaps piece_1. piece_1's v23b template mesh is consistent at the kit seed but has a thin margin. Y9 found that the template boards fail public 24 (23 cm / 135 deg).
    It costs 0.28 cm locally on piece_2, because piece_1 is the anchor and the frame-0 alignment changes. Main decides. The A9 recommendation is piece2 (meets "same or better locally").
- No other object needs a swap. ep32's cup and spoon are consistent, so there is nothing to report to B9 / main as a fault.

## Audit method (task 1)
`code/audit.py` (GT-free) handles each (ep, object). The submitted mesh is the component's `meshes/<ep>/<obj>.obj`, which is byte-identical to the v23b package mesh.
1. `T_cam`: v6 body -> submitted (sim/source) body, the robust mean over frames of `C_src(f)^-1 C_v6(f)`.
   - `C_src` = the component's `portable/source/<ep>/object_poses.npz` (world_from_rgb^-1 world_from_object). These poses equal the build-v2 reference poses to 1e-7.
   - `C_v6` = final_v6 camera_from_object. Both use the same camera.
   - So `T_cam` is the pose-convention mapping.
2. ICP of the v6 mesh onto the submitted mesh, started at `T_cam` -> `T_icp`. The twin basin is `T_twin` = ICP started at `H T_icp`, where H is the submitted mesh's own best >90 deg self-ICP.
   - Shape margin = mean symmetric surface distance in the twin basin minus that in the pose basin. It is positive when the submitted mesh agrees with v6 under the clip's pose convention.
3. Scorer-side, `code/regcheck.py`: kit registration (exact kit rule; kit seed + 7 seeds) of the submitted mesh and of v6@T_icp against the scan.
   The relative rotation is about 0 for the same twin and about 180 for the opposite twin.
4. Ball evidence, `code/balls_v6map.py`: the clip's stereo ball detections are lifted with OUR final_v6 poses and mapped by T_icp into the submitted body frame (Y9 mode clustering).
   A mode votes 'mesh' or 'twin' if it lies within 3 mm of only the mesh or only its self-twin. The test runs for the twin-prone objects.
5. v6's own convention was established by Y9 for 19/25/30: ball-built clip meshes and public 100 % vs 0 % seed tests. It is re-checked here: ep30 v6 mesh 38 mesh / 0 twin / 150 both; ep19 138 / 0.
6. Y9 `twin_margin.py` (eval mode: margin relative to the kit's own choice) on the old and new ep25/ep30 meshes: `out/margins/m1_item*.json`.

Only the sweeper (core self-twin 0.17-0.19 cm) and the boards (0.12-0.13 cm, twin about the face normal) are twin-prone.
- Pot, rack, dust pan, pitcher and spoon have their twin basin 0.19-1.23 cm worse than the pose basin, so the registration cannot flip them.
  The spoon's roll twin is broken by the bowl. The spoon has no scan, so it has no kit check.
- Cups are axisymmetric. The kit uses order 360, so the orientation about the axis is not scored.
- The lid is scored with order 2.

### Audit table (all 35 v23b objects)
"Kit seeds same twin as v6": 8 seeds (the kit seed + 7). Hashes are md5 / sha256 (first 8 hex digits) of the submitted mesh.
| ep | object | submitted md5 / sha256 | = final_v6 file | sim body vs v6 (T_cam deg) | shape margin cm (twin - pose basin) | core self-twin cm | kit seeds same twin as v6 | consistent | evidence |
|---|---|---|---|---|---|---|---|---|---|
| 1 | white_pot | 734e4be7 / a5531704 | yes | 4.2 | +0.859 | 0.910 | 8/8 (kit seed 0.0 deg) | yes | shape-asymmetric (twin basin +0.86 cm); kit same twin |
| 1 | white_pot_lid | 3590f16b / bc1443c5 | yes | 4.1 | +0.030 | 0.214 | 8/8 (kit seed 0.3 deg) | yes | same file as v6, no half turn vs v6 poses; kit scores the lid with order-2 symmetry, so its half-turn twin is not penalised |
| 3 | white_pot | b5ebc9d6 / 8cbd3543 | no (v6 734e4be7) | 1.9 | +0.723 | 0.966 | 8/8 (kit seed 1.3 deg) | yes | shape-asymmetric (twin basin +0.72 cm); kit same twin |
| 3 | wooden_spoon | 5fcd334c / c9ab88e3 | yes | 0.7 | +0.254 | 0.338 | no public scan | yes | no scan; bowl breaks the roll twin (see shape margin); T_cam 0.7 deg (no half turn) |
| 5 | white_pot | b5ebc9d6 / 8cbd3543 | no (v6 734e4be7) | 5.8 | +0.712 | 0.966 | 8/8 (kit seed 1.5 deg) | yes | shape-asymmetric (twin basin +0.71 cm); kit same twin |
| 5 | wooden_spoon | 5fcd334c / c9ab88e3 | yes | 5.4 | +0.204 | 0.338 | no public scan | yes | no scan; bowl breaks the roll twin (see shape margin); T_cam 5.4 deg (no half turn) |
| 6 | white_pot | b5ebc9d6 / 8cbd3543 | no (v6 734e4be7) | 6.8 | +0.705 | 0.966 | 8/8 (kit seed 2.5 deg) | yes | shape-asymmetric (twin basin +0.70 cm); kit same twin |
| 6 | wooden_spoon | 5fcd334c / c9ab88e3 | yes | 6.4 | +0.240 | 0.338 | no public scan | yes | no scan; bowl breaks the roll twin (see shape margin); T_cam 6.4 deg (no half turn) |
| 8 | beige_cup | b30ae10b / 2c041e69 | yes | 58.7 | -0.010 | 0.117 | n/a (axisymmetric, kit order 360) | yes (n/a) | axisymmetric; orientation scored modulo axis |
| 8 | white_pot | b5ebc9d6 / 8cbd3543 | no (v6 734e4be7) | 4.8 | +0.709 | 0.966 | 8/8 (kit seed 2.4 deg) | yes | shape-asymmetric (twin basin +0.71 cm); kit same twin |
| 10 | blue_cup | 415ade56 / 57bb269a | yes | 80.5 | -0.002 | 0.120 | n/a (axisymmetric, kit order 360) | yes (n/a) | axisymmetric; orientation scored modulo axis |
| 10 | beige_cup | b30ae10b / 2c041e69 | yes | 85.6 | -0.024 | 0.117 | n/a (axisymmetric, kit order 360) | yes (n/a) | axisymmetric; orientation scored modulo axis |
| 14 | blue_cup | 19b79f8f / 78bd065c | no (v6 415ade56) | 136.3 | +0.002 | 0.169 | n/a (axisymmetric, kit order 360) | yes (n/a) | axisymmetric; orientation scored modulo axis |
| 14 | plastic_dish_rack | f3938548 / 090b7d17 | yes | 4.3 | +0.834 | 0.904 | 8/8 (kit seed 0.0 deg) | yes | shape-asymmetric (twin basin +0.83 cm); kit same twin |
| 15 | plastic_dish_rack | 77c6ed93 / 1162524a | no (v6 f3938548) | 4.1 | +0.600 | 0.870 | 8/8 (kit seed 0.5 deg) | yes | shape-asymmetric (twin basin +0.60 cm); kit same twin |
| 15 | wooden_spoon | e6c98973 / 6cc4d2b2 | no (v6 5fcd334c) | 3.5 | +0.308 | 0.420 | no public scan | yes | no scan; bowl breaks the roll twin (see shape margin); T_cam 3.5 deg (no half turn) |
| 16 | plastic_dish_rack | ec778422 / 67050d1f | no (v6 f3938548) | 4.1 | +0.822 | 0.904 | 8/8 (kit seed 0.9 deg) | yes | shape-asymmetric (twin basin +0.82 cm); kit same twin |
| 16 | wooden_spoon | e6c98973 / 6cc4d2b2 | no (v6 5fcd334c) | 2.7 | +0.296 | 0.420 | no public scan | yes | no scan; bowl breaks the roll twin (see shape margin); T_cam 2.7 deg (no half turn) |
| 19 | mini_sweeper | bbd4d71c / cf7bf340 | no (v6 c172b576) | 12.1 | +0.003 | 0.192 | 8/8 (kit seed 2.5 deg) | yes | kit 8/8 same twin (2.5 deg); v6-pose ball modes 138 mesh / 16 twin; Y9 4 deg (pair +x, ep31 convention file) |
| 19 | mini_dust_pan | bd32cdf2 / 1d298c01 | yes | 8.2 | +0.489 | 0.630 | 8/8 (kit seed 0.9 deg) | yes | shape-asymmetric (twin basin +0.49 cm); kit same twin |
| 20 | blue_cup | 277d45fc / ff3784a8 | no (v6 415ade56) | 164.3 | +0.000 | 0.169 | n/a (axisymmetric, kit order 360) | yes (n/a) | axisymmetric; orientation scored modulo axis |
| 20 | water_pitcher | 88458e8f / 08e6ab82 | yes | 3.3 | +1.226 | 1.278 | 8/8 (kit seed 0.0 deg) | yes | shape-asymmetric (twin basin +1.23 cm); kit same twin |
| 25 | wooden_piece_1 | e3b5ab14 / 3caafb99 | no (v6 40ef8729) | 1.9 | -0.004 | 0.124 | 8/8 (kit seed 0.3 deg) | yes, but **ambiguous** (template balls) | kit 8/8 same twin; base template with symmetric template balls (self-twin about the face normal 0.12 cm, shape margin -0.004 cm); no detection mode within 3 mm of mesh or twin; Y9: template fails public 24 (23 cm / 135 deg) |
| 25 | wooden_piece_2 | 8bd9f632 / 20f7a845 | no (v6 2d57d07b) | 1.4 | -0.024 | 0.127 | 5/8 (kit seed 180.0 deg) | **NO (flipped at kit seed)** | kit seed 180 deg vs v6 (5/8 seeds); template mesh, 165 detections vote twin, 0 mesh; Y9 180.0 deg |
| 28 | plastic_dish_rack | 305330a1 / 00c51376 | no (v6 f3938548) | 5.6 | +0.577 | 0.864 | 8/8 (kit seed 0.4 deg) | yes | shape-asymmetric (twin basin +0.58 cm); kit same twin |
| 28 | white_pot | a3600c35 / 69677910 | no (v6 734e4be7) | 172.3 | +0.741 | 1.021 | 8/8 (kit seed 2.0 deg) | yes | shape-asymmetric (twin basin +0.74 cm); kit same twin |
| 30 | mini_sweeper | be217ca0 / 26c2fa2d | no (v6 0bde870a) | 4.9 | -0.058 | 0.168 | 0/8 (kit seed 180.0 deg) | **NO (flipped)** | kit 180 deg vs v6 on 8/8 seeds; v6-pose ball modes: 171 detections vote twin, 0 mesh; Y9 179.3 deg |
| 30 | plastic_dish_rack | bb3ce1dd / 37f78ed0 | no (v6 f3938548) | 4.9 | +0.677 | 0.926 | 8/8 (kit seed 0.7 deg) | yes | shape-asymmetric (twin basin +0.68 cm); kit same twin |
| 32 | blue_cup | 223662de / 3c3c5328 | no (v6 415ade56) | 21.3 | +0.000 | 0.116 | n/a (axisymmetric, kit order 360) | yes (n/a) | axisymmetric; orientation scored modulo axis |
| 32 | wooden_spoon | 5fcd334c / c9ab88e3 | yes | 1.4 | +0.258 | 0.338 | no public scan | yes | no scan; bowl breaks the roll twin (see shape margin); T_cam 1.4 deg (no half turn) |
| 33 | blue_cup | 415ade56 / 57bb269a | yes | 119.8 | +0.000 | 0.120 | n/a (axisymmetric, kit order 360) | yes (n/a) | axisymmetric; orientation scored modulo axis |
| 33 | wooden_spoon | 5fcd334c / c9ab88e3 | yes | 0.7 | +0.248 | 0.338 | no public scan | yes | no scan; bowl breaks the roll twin (see shape margin); T_cam 0.7 deg (no half turn) |
| 35 | wooden_spoon | 5fcd334c / c9ab88e3 | yes | 10.5 | +0.239 | 0.338 | no public scan | yes | no scan; bowl breaks the roll twin (see shape margin); T_cam 10.5 deg (no half turn) |
| 36 | wooden_spoon | 5fcd334c / c9ab88e3 | yes | 3.4 | +0.252 | 0.338 | no public scan | yes | no scan; bowl breaks the roll twin (see shape margin); T_cam 3.4 deg (no half turn) |
| 37 | wooden_spoon | 5fcd334c / c9ab88e3 | yes | 6.3 | +0.193 | 0.338 | no public scan | yes | no scan; bowl breaks the roll twin (see shape margin); T_cam 6.3 deg (no half turn) |
| 38 | mini_dust_pan | bd32cdf2 / 1d298c01 | yes | 3.9 | +0.573 | 0.630 | 8/8 (kit seed 0.1 deg) | yes | shape-asymmetric (twin basin +0.57 cm); kit same twin |

Twin-prone ball votes (detections in modes; `out/balls_v6map.json`):
| ep / object | submitted mesh: mesh / twin / both / none | final_v6 mesh: mesh / twin / both / none |
|---|---|---|
| 30 sweeper | **0 / 171** / 17 / 0 | 38 / 0 / 150 / 0 |
| 19 sweeper | 138 / 16 / 163 / 12 | 138 / 0 / 26 / 165 |
| 25 piece_2 | **0 / 165** / 0 / 563 | 162 / 56 / 0 / 510 |
| 25 piece_1 | 0 / 0 / 0 / 450 (template balls match no mode within 3 mm) | 46 / 177 / 0 / 227 (4-ball layout nearly symmetric about the normal) |

Twin margins (Y9 twin_margin, eval mode, cm; relative to the kit's choice, which for the OLD ep30 / piece_2 meshes is the wrong twin):
| mesh | kit seed | mean | min | seeds agreeing | CD |
|---|---|---|---|---|---|
| ep30 sweeper old be217ca0 | 0.0099 | 0.0132 | 0.0080 | 1.00 (all on the flipped twin) | 0.195 |
| ep30 sweeper new (v6 0bde870a @ T) | 0.0102 | 0.0100 | 0.0066 | 1.00 | 0.187 |
| ep25 piece_2 old 8bd9f632 | 0.0022 | 0.0000 | -0.0044 | 0.35 | 0.188 |
| ep25 piece_2 new (v6 2d57d07b @ T) | 0.0154 | 0.0131 | 0.0061 | 1.00 | 0.188 |
| ep25 piece_1 old e3b5ab14 | 0.0066 | 0.0040 | -0.0011 | 0.88 | 0.157 |
| ep25 piece_1 new (v6 40ef8729 @ T) | 0.0186 | 0.0087 | -0.0055 | 0.94 | 0.167 |

### v23b components vs target_final_v6 (score_any.sh, labels scores/cand/A9v6_<tag>)
| ep | v23b tag | MP-SR | MPPE | AUC | objects cm / deg |
|---|---|---|---|---|---|
| 1 | N8_ep01_k2c2048 | 1 | 0.96 | 0.949 | white_pot 0.76 / 4.9; white_pot_lid 0.84 / 9.4 |
| 3 | Q8_ep03_g11 | 1 | 1.13 | 0.941 | white_pot 0.85 / 6.3; wooden_spoon 1.13 / 9.7 |
| 5 | Q8_ep05_g1 | 1 | 1.34 | 0.923 | white_pot 1.22 / 4.5; wooden_spoon 1.22 / 9.2 |
| 6 | L8_ep06_slate | 1 | 1.09 | 0.949 | white_pot 0.64 / 2.8; wooden_spoon 1.24 / 11.9 |
| 8 | N8_ep08_klr512 | 1 | 0.85 | 0.961 | beige_cup 0.71 / 2.9; white_pot 0.83 / 7.1 |
| 10 | X8_ep10_Blr512 | 1 | 0.53 | 0.991 | blue_cup 0.43 / 1.9; beige_cup 0.57 / 2.7 |
| 14 | N8_ep14_kl512 | 1 | 0.92 | 0.966 | blue_cup 1.11 / 4.1; plastic_dish_rack 0.67 / 2.8 |
| 15 | U8_ep15_c | 1 | 0.90 | 0.962 | plastic_dish_rack 0.68 / 2.8; wooden_spoon 0.94 / 7.1 |
| 16 | U8_ep16_b | 1 | 0.95 | 0.959 | plastic_dish_rack 0.68 / 3.6; wooden_spoon 1.07 / 6.2 |
| 19 | T8_ep19_c2 | 1 | 1.45 | 0.924 | mini_sweeper 1.28 / 13.7; mini_dust_pan 1.12 / 9.1 |
| 20 | N8_ep20_k3c1024 | 1 | 0.81 | 0.960 | blue_cup 0.61 / 11.8; water_pitcher 0.57 / 3.4 |
| 25 | J8_ep25_b48m4 | 1 | 1.44 | 0.898 | wooden_piece_1 1.27 / 14.5; wooden_piece_2 1.14 / 5.6 |
| 28 | J8_ep28_w2k3c2048 | 1 | 0.84 | 0.966 | plastic_dish_rack 0.74 / 1.9; white_pot 0.77 / 8.0 |
| 30 | N8_ep30_k2c2048 | 0 | 14.45 | 0.226 | mini_sweeper 18.51 / 29.7; plastic_dish_rack 7.85 / 177.0 |
| 32 | V8_ep32_wb4r2 | 1 | 1.82 | 0.871 | blue_cup 0.88 / 9.9; wooden_spoon 2.44 / 9.9 |
| 33 | M8_ep33_r5 | 1 | 1.38 | 0.914 | blue_cup 1.22 / 9.4; wooden_spoon 1.16 / 9.1 |
| 35 | K8_ep35_ilc2 | 1 | 1.63 | 0.890 | wooden_spoon 1.27 / 13.5 |
| 36 | K8_ep36_ecem_ilc | 1 | 1.46 | 0.930 | wooden_spoon 1.18 / 11.3 |
| 37 | K8_ep37_ilc1 | 1 | 0.28 | 1.000 | wooden_spoon 0.24 / 2.0 |
| 38 | J8_ep38_ilc36c2048 | 1 | 0.76 | 0.980 | mini_dust_pan 0.64 / 5.5 |

## Mesh-only swaps (task 2)
`code/stage.py TAG EP OBJ...` does the following:
- It copies the v23b legacy component to `staging/<TAG>/components/<episode>` (cp of the whole tree).
- It rewrites ONLY `meshes/<episode>/<obj>.obj`. The v6 OBJ is copied line by line, and only the `v`/`vn` lines get `x_sub = T x_v6`.
- It updates that object's `readiness.json` mesh sha256 entry and writes `a9_meshswap.json` (old and new hashes, T, T_cam, distances).

The portable bundle is untouched. That covers the frozen scene, the collision parts, `portable/source/<ep>/assets` and the policy, so the physics is byte-identical, as in X8 in reverse.
- Build-v2 re-executed every tag with `exact_previous_physical_states_actions` true.
- All 32 trajectory.npz numeric arrays are identical to N8's for ep30.
- The results parquet data are identical. Only the metadata paths differ.

T = T_icp (v6 body -> component body). It is about identity for all three objects, because the component bodies share the canonical frame.
The twin is therefore set by the v6 mesh's own marker side, which follows the clip convention (Y9). T is applied to the full-resolution v6 mesh.

| tag | object | old md5 / sha256 | new sha256 | v6 source sha256 | T rotvec deg / t mm | mean sym. surface dist to old (pose basin / twin basin) cm | T_icp vs T_cam |
|---|---|---|---|---|---|---|---|
| A9_ep30_sweeper | mini_sweeper | be217ca0 / 26c2fa2d50799bc4.. | ccb942f574b628b70b6ca6c847b1046c2d7506076dca031b8d8500b03db0ce2a | 267cb2b239b20eb9.. (0bde870a) | (0.61, 0.29, 0.46) / (-0.49, 0.58, 0.66) | 0.156 / 0.097 (the old mesh sits on v6's twin) | 5.7 deg / 0.63 cm |
| A9_ep25_piece2, A9_ep25_boards | wooden_piece_2 | 8bd9f632 / 20f7a8458adb6e90.. | a2d89be1189345adb731ac2f90918aead44dd352706b78a0de2bc5ddad656df1 | 07228eb8f638998d.. (2d57d07b) | (0.01, 0.54, -0.01) / (-0.53, 0.00, -5.49) | 0.115 / 0.091 (old on the twin) | 1.2 deg / 1.13 cm |
| A9_ep25_boards | wooden_piece_1 | e3b5ab14 / 3caafb99e661fe7d.. | ecdfc14c68be8f57cefad31fbcee68cab999f41f8feb6b31043895984918fb51 | 535151e6e30b7f41.. (40ef8729) | (0.02, -0.11, 0.01) / (2.84, 0.00, 0.06) | 0.085 / 0.081 (template: twin-ambiguous) | 1.9 deg / 1.50 cm |

Full 4x4 transforms: `staging/<tag>/components/<episode>/a9_meshswap.json` and `<obj>.a9_T_sub_from_v6.npy`.
Unchanged meshes in the new components: ep30 rack 37f78ed0.. (bb3ce1dd); A9_ep25_piece2 piece_1 3caafb99.. (e3b5ab14).

Build-v2 outputs are in `campaign_v16_20261023/common_v2_candidates/{A9_ep30_sweeper,A9_ep25_piece2,A9_ep25_boards}/<episode>`.
completed.json mesh sha256: ep30 sweeper ccb942f5 + rack 37f78ed0; piece2 3caafb99 + a2d89be1; boards ecdfc14c + a2d89be1.

## Not swapped (consistent), notes
- **Legacy but consistent:**
  - pot 3/5/6/8 (b5ebc9d6) and 28 (a3600c35; its body frame is about 172 deg from v6, and the poses agree);
  - rack 15/16/28/30;
  - spoon 15/16 (e6c98973);
  - cups 14/20/32;
  - sweeper 19 (bbd4d71c, the pub31 file, pair at +x as in ep19).
  - These do not need a swap for the twin. The legacy cups 14/20/32 fit the scan worse than v6 in the kit registration (0.32-0.37 vs 0.26 cm), which is a shape issue, not a twin one. The legacy pots fit better (0.35 vs 0.41).
- **ep19:** the T8 poses differ from v6 by 12 deg / 3.3 cm (T8 world_a40 target). T_icp (shape) is 2 deg, so the twin is unambiguous.
  Swapping to v6 c172b576 is possible but not needed. A 4-deg registration difference between bbd4d71c and final_v5 was already noted by Y9.
- **ep32 (B9 owns the poses):** cup 223662de is axisymmetric. Spoon 5fcd334c is the v6 file, with T_cam 1.4 deg (no half turn). Nothing inconsistent.
- **ep1:** the submitted pot is the v6 file (734e4be7). The sim source asset is b5ebc9d6. They share a body frame (shape distance 0 at T_icp = I; T_cam 4.2 deg).

## Files
- code/: `a9lib.py`, `audit.py` (+ `audit.sbatch`), `regcheck.py` (+ `reg.sbatch`), `balls_v6map.py`, `stage.py`, `summ.py`, `score_array.sbatch`, `margin_array.sbatch` (runs Y9 twin_margin.py).
- out/:
  - `audit_ep*.json` (T_cam, T_icp, T_twin, distances, ball modes), `reg_ep*.json`, `balls_v6map.json`;
  - `margins/m1_item*.json`, `audit_table.md`, `v23b_vs_v6_table.md`;
  - `meshes_audit/` (v6 meshes mapped into the submitted frames, for audit only).
- staging/: A9_ep30_sweeper, A9_ep25_piece2, A9_ep25_boards.
- Scores: `scores/cand/A9v6_*` (all 20 v23b tags and the 3 new tags, vs target_final_v6).
