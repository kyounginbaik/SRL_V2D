# Stream X8 (phase 8, v23): episode 10, thinner SIM-ONLY cup collision proxies, 2026-10-28

User-approved change: "try thinner collision for ep10". Only the collision geometry of blue_cup / beige_cup in the MuJoCo scene
changes. The visual meshes, which are also the submitted meshes, are our reconstruction meshes and are byte-identical to
N8_ep10_up3c512's (beige 2c041e69..., blue 57bb269a...). No GT was used. Every pose comes from a build-v2 policy rollout. Nothing was
uploaded, committed or deleted.

## Results (score_any vs target_final_v5, build-v2 re-execution, cm / deg)

| tag | component | world | blue_cup | beige_cup | MP-SR | MPPE | AUC | exact |
|---|---|---|---|---|---|---|---|---|
| **X8_ep10_Blr512** (recommended) | streamX8_ep10/staging_B1/components/episode_000010 (actor B_LR c512_p0.001, checkpoint 36ab15df.., actions 377c886d.., archive staging_B1/portable_archives/episode_000010_portable.tar.gz sha256 0d5aadcd82db3784057002ff497ffa307a851e66837053b4ac9d1dcae6f6a428) | B world_t15_d07 | **0.43 / 1.9** | **0.57 / 2.7** | 1 | 0.53 | 0.991 | true |
| X8_ep10_Arc512 (used by main for v23b) | streamX8_ep10/staging_A1/components/episode_000010 (A replay c512_p0.003, checkpoint ecf6702f.., archive sha256 6a03c0b28a6e8b07a482a53d7dc6a6b4dcad2fc2ae5cf0f040aa26b93964bfa4) | A world_t08_d0 | 0.64 / 1.8 | 0.80 / 2.8 | 1 | 0.74 | 0.976 | true |
| before: N8_ep10_up3c512 | streamN8.../ep10/staging_N1b | N8 world_up (stream-L proxies) | 0.51 / 1.3 | 1.48 / 2.7 | 1 | - | 0.942 | true |
| (not staged) B2_LR c512_p0.001 actor eval | distilled/B2_LR | B | 0.42 / 1.9 | 0.52 / 2.7 | 1 | | | - |

Build-v2 outputs: campaign_v16_20261023/common_v2_candidates/X8_ep10_{Blr512,Arc512}/episode_000010 (completed.json
exact_previous_physical_states_actions true); scores via score_any.sh (scores/cand/X8_ep10_*). Submitted meshes in both components and
both build-v2 outputs: beige 2c041e69fbd3be79fa953609276ffc4a54a513ec2ca7e7f31020a78ab6871431, blue
57bb269a06a44f9f56ea59c9eea8b0d704b20d5a8cfd8bbc89f73f5e2f2b1d8f, byte-identical to N8_ep10_up3c512's. The frozen
portable scene references the X8 collision parts (2 x 241 paths). portable/assets holds all 482 X8 part hashes and none of the
stream-L-only parts (checked for B1).

## Why (streams W8 / N8 / G / L)
Target_final_v5 nests the beige cup with rim-to-rim h of about 2.5 cm (median of f100-160 = 2.51 cm, p10 2.29). W8 showed the image supports this.
Physically, the stream-L thin-wall proxies stop the beige about 3-4 cm higher, so in N8 the beige error stayed at a 1.48 cm floor.
`code/nesttest.py` drops the beige proxy coaxially into the welded upright blue proxy in MuJoCo, with the episode solver options, 7-9 spins / tilts / offsets:

| blue wall t (mm) / beige shift d (mm) | analytic coaxial h (aligned / worst sector) | MuJoCo settled h median (min-max) |
|---|---|---|
| stream L (t=1.5 + 2.8 mm lip, d=0) | 5.93 / 6.39 | 5.92 (5.52-6.36) |
| 1.5 (lip removed only), 0 | 3.75 / 4.21 | 3.43 (3.17-3.85) |
| 1.5, 0.5 | 2.89 / 3.55 | 2.77 (2.36-5.45) |
| 1.5, 0.6 | 2.71 / 3.39 | 2.63 (2.15-4.20) |
| **1.5, 0.7 (world B, adopted)** | (interp.) | (~2.45; between 0.6 and 0.75) |
| 1.5, 0.75 | 2.43 / 3.11 | 2.34 (1.71-3.07) |
| 1.5, 1.0 | 1.95 / 2.63 | 1.99 (1.26-2.53) |
| 1.2, 0 | 3.29 / 3.83 | 3.02 (2.77-3.75) |
| 1.0, 0 | 2.89 / 3.57 | 2.77 (2.36-5.19) |
| 1.0, 0.25 | 2.43 / 3.11 | 2.20 (1.71-4.33) |
| **0.8, 0 (world A, alternative)** | 2.51 / 3.21 | 2.36 (1.93-4.88) |
| 0.8, 0.5 | 1.61 / 2.25 | 1.54 (0.79-3.68) |
Raw rows: logs/nest_*.log, out/nest_*.json. The high maxima are tilted drops (2 deg) that wedge with friction.

## The collision modification (code/thinlib.py, code/build_world_x8.py; record world_*/collision_x8.json)
Everything is derived from the radial profiles of our visual meshes. `thinlib.profile` reproduces the stream-L build_recon_world.py profile
exactly; the regenerated stream-L parts are byte-identical (241/241 per cup). Same 10 bands x 24 sectors, 1/cos(pi/24) push, base disk.
Only the band-edge radii change:

**World B = `world_t15_d07` (adopted): blue rim lip removed, beige wall -0.7 mm**
- blue_cup: inner radius R_in = R_out - 1.5 mm at every band edge. The outer radius is unchanged. Bands 1-8 are unchanged, because they were already R_out - 1.5.
  The lip bands get a larger inner radius: 15.73 cm +1.83 mm (4.121 -> 4.304 cm), rim 17.31 cm +1.28 mm (4.319 -> 4.448). The floor
  corner at 1.46 cm gets +1.10 mm (3.266 -> 3.376; it is above the base disk, and the beige never reaches it).
- beige_cup: the whole wall moves inward by 0.70 mm at every band edge (R_out 3.429..4.528 -> 3.359..4.458 cm, R_in likewise),
  so the thickness is unchanged.
- mm per band edge (height above the cup bottom):

| blue h (cm) | 1.46 | 3.04 | 4.63 | 6.21 | 7.80 | 9.38 | 10.97 | 12.55 | 14.14 | 15.73 | 17.31 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| blue d inner (mm) | +1.10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +1.83 | +1.28 |
| blue d outer (mm) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

| beige h (cm) | 0.50 | 2.17 | 3.84 | 5.51 | 7.18 | 8.85 | 10.52 | 12.19 | 13.86 | 15.53 | 17.21 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| beige d outer = d inner (mm) | -0.70 | -0.70 | -0.70 | -0.70 | -0.70 | -0.70 | -0.70 | -0.70 | -0.70 | -0.70 | -0.70 |

- Mass stays 0.1 kg. COM and inertia are recomputed from the new parts (uniform density, the stream-L convention). Blue COM moves 8.2 mm
  toward the base (61.9 -> 53.7 mm above its bottom) because the lip volume is gone. Beige COM is unchanged (81.3 -> 81.4 mm).
- sha256 (concatenation of part_000..240 in order; per-part hashes are in collision_x8.json):
  blue `e5031586754f8e0a7058aed3eb0e8bd86a6b46936a15159c5eccaaa902cb184e` (stream L a3dd3578bfd7a11d..),
  beige `7baba0d43d04c5e422d000083c98d67a5b61b7183509c9cbc6ec536f086b2b4d` (stream L 3dd4993e589b809b..);
  urdf blue 507add7dd1bc9ef0.., beige d9fd1950b4c6d7ae...

**World A = `world_t08_d0` (alternative): blue wall 1.5 -> 0.8 mm (inner +0.7 mm on bands 1-8, lip +2.53 / +1.98 mm, floor corner
+1.80 mm), beige unchanged** (its parts are byte-identical to stream L, 3dd4993e..). Blue concat sha 9e05fa36393d791a960b92f580ca299d84c6f4f802d42b437038866ef95e4ccd.
Blue COM moves 21.9 mm toward the base (61.9 -> 40.0 mm). That larger dynamics change is why B is preferred.

Reference: the visual shells are about 0.4-0.7 mm thick (R_out - visual raw inner, bands 1-8). In B the blue collision inner wall stays 1 mm inside
the blue visual inner surface. The beige collision outer wall sits 0.7 mm inside the beige visual outer surface. At contact the two visual
surfaces overlap by at most about 0.2 mm, which matches W8's finding that the real cups are 1-3 mm thinner than our meshes at the rim and wall.

## World and re-plan
1. `code/build_world_x8.py OUT T_BLUE_MM D_BEIGE_MM` writes new parts, urdfs and collision_x8.json, plus a copy of N8 world_up/episode_target (the 7.6 deg
   up-rotated target, unchanged) whose input_manifest collision entries point at the new parts. `code/swap_scene_x8.py` text-swaps the 2 x 241 collision
   mesh paths and the two cup `<inertial>` elements in a frozen scene. Everything else stays byte-identical, including the visual paths (stream L world_v1, sha 57bb../2c04..).
2. `code/replay_x8.py`: N8's teacher kcem/u_p3_LR (the teacher of up3c512) replayed in the X8 scene:
   A 0.65/1.8, 0.84/2.7; B 0.62/1.8, 0.86/2.9 (old world 0.44/1.4, 1.49/2.6).
3. N8 knot CEM (kcem_n8.py, unchanged; both wrists, 1.0-7.8 s, 10 knots, 48 x 12, P 1.0/1.0 cm, D 12):
   A_LR 0.51/2.2, 0.53/3.1; **B_LR 0.47/1.8, 0.47/2.8**; B2_LR (second pass on B_LR, pos-scale 0.004) 0.43/1.9, 0.47/3.1.
4. Distill (stream C actor_grid_c.py, node01, seed 44, 8 configs; all 8 pass in every grid):
   A replay: best c512_p0.003 0.64/1.8, 0.80/2.8. A_LR: c2048_p0.003 0.52/2.1, 0.59/2.9.
   **B_LR: c512_p0.001 0.43/1.9, 0.57/2.7** (c2048_p0.001 0.40/1.9, 0.58/3.0). B replay: c1024_p0.003 0.57/1.7, 0.72/2.8.
5. Stage: `code/stage10.sbatch` (N8 stage10 = unchanged stream G stage_G_component.py; lineage also carries collision_x8.json, the X8 urdfs
   and thinlib/build_world_x8/swap_scene_x8/replay_x8/nesttest), then `common_v2_candidates/run_one.sbatch 10 <component> <tag>` (node01,
   --time 50 min), then score_any.sh via N8 verify.sh.

## Caveats
- The proxies are thinner than the stream-L ones by design, which is why the beige can sit as deep as the reconstruction's nest. With tilt,
  friction can still wedge the beige higher (the nesttest maxima). The policy rollout settles it at the target depth.
- Blue orientation (1.8-1.9 deg) is slightly worse than N8 (1.3 deg); blue position and beige are better.
- These proxy scores are against our own reconstruction, not hidden GT.
