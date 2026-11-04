# v22 stream E: episodes 25 (wooden boards cross) and 30 (mini_sweeper + plastic_dish_rack) vs target_final_v5 (2026-10-28)

Reference: `campaign_v22_20261028/target_final_v5` (our reconstruction). Scores = `code/score_any.sh` (kit scorers, our mesh
registered onto the target mesh, anchor = slot 0 frame 0), cm / deg. Our meshes only (ep30: mini_sweeper 26c2fa2d.., rack 6-prong
37f78ed0..; ep25: streamQ meshes_slotted_C boards 3caafb99.. / 20f7a845..). No scan-derived or GT meshes. All code copied into this
dir (`code/`, `code25/`, `code30/`); nothing earlier was edited.

## Results

| ep | tag (common_v2_candidates) | component | object scores vs target_final_v5 | MP-SR | build-v2 exact |
|---|---|---|---|---|---|
| 30 | **E_ep30_stdB45** | `ep30/staging_T1/components/episode_000030` | sweeper **0.97 / 6.4**, rack **0.97 / 5.6** | 1 | true |
| 25 | **E_ep25_R103s62** | `ep25/staging_R103s62/components/episode_000025` | piece_1 **2.03 / 16.0**, piece_2 **2.04 / 10.7** | 1 | true |
| 25 | E_ep25_R103 (fallback) | `ep25/staging_R103/components/episode_000025` | piece_1 **2.14 / 14.4**, piece_2 **2.25 / 10.8** | 1 | true |

Before: ep30 best existing C_ep30_f48 sweeper 18.35 / 138.4, rack 7.20 / 174.7; ep25 C_ep25_slotC1 2.84 / 26.5, 4.15 / 10.5.
Portable archives: `ep30/staging_T1/portable_archives/episode_000030_portable.tar.gz` (sha256 07dc68db..),
`ep25/staging_R103/portable_archives/episode_000025_portable.tar.gz` (a05612ce..), `ep25/staging_R103s62/portable_archives/episode_000025_portable.tar.gz` (58fd4a7f..).
Local proxy (world reference) equals the scorer here because achieved frame 0 = keyframe = reference frame 0 and the reference is the
target expressed in our body frames with the scorer's own registration (R103: proxy 2.136/14.44, 2.258/10.81; scorer 2.14/14.4, 2.25/10.8).

## ep30
1. Target check (`code/diag30.py`, overlay `ep30/overlay_target_sweeper.jpg` via tools/reconstruction_audit/diag_overlay.py on
   canonical episode_000030 exports/final_v5): the target sweeper fits the masks at f0-250, and its face is CONSTANT (no f52 flip).
   The hook is unambiguous at f62-150 and matches the target. Relative to the target, our earlier physics worlds start the sweeper as
   the half-turn twin (178 deg about the vertical, self-chamfer 5.4 mm mean / 27 mm max, so not a true symmetry). v21 kept the early
   face for the whole episode and compose flipped at f52; the target uses the late face from frame 0. Rack-anchored, the target agrees with
   our world (rack 0.27 cm / 1.8 deg), so the target is right and the old anchor broke because of the sweeper face.
2. Reference (`code/conv30.py` -> `ep30/target_ourworld.npz` + `_lineage.json`): scorer registration (all seeds agree within 0.3/0.6 deg);
   G anchored on the STATIC RACK at f0 (target rack lands on the physics rack; Gt = I). This is the recipe's sweeper-anchored G0 is invalid
   here because our physics sweeper f0 is the twin of the target's. Sweeper per-frame shift vs the C world is 178 deg for f0-50 and 5-19 deg after.
3. World `ep30/world_T1` = streamI build_world.py `--pose-override target_ourworld.npz --collision-cut --cut-frames 0 20 40 238 245
   250 255 262` (14 of 128 rack parts sliced, residual 0.17 mm; frozen settle sweeper 0.31 cm / 1.1 deg).
4. Pickup donor anchor made twin-consistent (`code30/prep30.py`): A = same absolute hand, object -> twin (fails: 0/4 pass at pickup);
   **B = hand carried with the object** (body-relative grasp preserved; works). 12 pickup+chain runs (code30/pickup.sbatch, chain.sbatch).
   Best: std_B_s45 -> lift 0.06 -> retention 2.0-12.0 -> release 11.6: teacher 1.04 / 6.3, rack 0.95 / 5.3.
5. Plain actor grid (node01): 20 of 20 pass (0.93-1.17 / 5.7-6.6, rack 0.97-1.10). Staged actor_c1024_p0.03_v0.0005
   (checkpoint 5a213298..). The second retention pass (ret2/) was not needed.
Right-wrist chart rotation (coordinator note from stream C) is the identity in both E scenes, so the anchor-B wrist rotation is unaffected.

## ep25
1. Mesh: the scorer registration of the slotC1 boards onto the target boards is deterministic (20/20 seeds). The slotless boards flip on
   5-6/20 seeds, and the other slotted variants on 4/20 for piece_1 (`code/diag25.py`).
2. The target is internally inconsistent at rest: board 2's bottom face is 2.66 cm below board 1's plane and the face normals are 8.5 deg
   apart, so no flat-table world can match both. Anchoring only on board 1 (G0) would leave board 2 about 3 cm off for the first 9 s.
   `code/conv25.py`: G = Gt @ G0. G0 maps target board 1 f0 onto the worldC/c1 board 1. Gt rotates the mean resting-face normal onto
   +z, with the face normal in the body frame taken from the boards SETTLED in C_ep25_slotC1 (the worldC reference is itself 4.3 deg
   off flat). Gt is a 4.3 deg tilt plus a vertical shift that puts the lower board 0.2 mm above the table. Because the scorer anchors on
   the keyframe (= reference f0), this splits the inconsistency between the boards: rest error is about 1 cm on board 1 for about 2.5 s
   and 0.2 cm / 1 deg on board 2.
3. World `ep25/world_T1` (`code25/build_worldE.py` = streamQ build_worldC.py with the reference from conv25; lineage
   source_corrected/episode_000025/board_track_correction.json). Settle: board 1 drops 1.1 cm, board 2 0.2 cm.
4. Port of the C teachers (`code25/hand_follow.py`): the yaw variants knock board 1 off. With no wrist follow, board 2's right-hand
   grasp fails at 8-9 s (9.6 cm). The right-hand retention CEM 5.5-17.75 s on hf_s91_L0.8_R4.0_y0 gives teacher R103 1.83 / 18.8, 1.82 / 7.4.
   Left CEMs on top of it (L111-L114) trade board 2 for board 1 (best 1.63 / 16.7, 1.95 / 8.2) and distil worse. The L99 base plus right CEM gives
   2.40 / 14.3, 1.42 / 8.3.
5. Distillation is the bottleneck: about 1 of 24 actors pass on the default grid, and passing actors sit 0.2-0.4 cm above the teacher.
   Seed sweeps of the configurations that worked (`ep25/distilled/R103_c`, `L99R103_b`) gave s62 (2.03 / 16.0, 2.05 / 10.7).
Caveats: the ep25 position margin is thin (about 2.0-2.25 cm against the 2.4 target). Board 2 is still not fully seated. The rest-phase
split is a design choice forced by the target's internal inconsistency.
