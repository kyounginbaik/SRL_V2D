# Stream Y9 (phase 9): mesh twin asymmetry and mesh size bias

Rules followed: no public scan or GT mesh is used in any mesh or in its preparation. Scans and GT appear only in
`twin_margin.py`, `reg_compare.py`, `chamfer_cd.py` and `score_bundle.py`/bscore, all scorer-side. Nothing earlier was
overwritten. No upload, no commit.

## Result
**Adopted one change: the ep30 mini_sweeper submitted mesh.** Bundle: `reconstruction_recovery_20261026/selected20_final_v6_y9/`.
It is a copy of final_v5 in which only `episode_000030/meshes/mini_sweeper.obj` (0bde870a) and that episode's meshes.json
plus bundle_manifest.json differ. Every pose file is byte-identical.

**The v23b package has the same problem in two places.** `track3_v23b_20261028/all_20_episodes/meshes`:
- ep30 sweeper (be217ca0) registers 179.3 deg from the ep30 convention.
- ep25 wooden_piece_2 (8bd9f632, the base template mesh) registers 180.0 deg from the ep25 convention.
- ep19 (bbd4d71c) and ep25 piece_1 are consistent.

## Mechanism (task 1)
- Our sweeper and board cores are half-turn symmetric. The tracker locks one twin per clip, and that twin is stable over the
  whole clip (`ball_layout_clip.py`, which splits the detections into time thirds). The mocap balls are the only real asymmetry:
  - Sweeper: 2 balls on one plate face, apparently 1 on the other.
  - Boards: balls on one face only (piece_1 has 4, piece_2 has 3).
- Without markers the registration is a coin flip. Kit picks the correct twin on 6-65 % of seeds: board_nomk, board_rb, sw_core.
  The pooled 4-ball sweeper (two conventions mixed) gives 24-76 %.
- With the marker side in the clip's own convention it picks correctly on 100 % of seeds. With the wrong side it picks correctly on 0 %.
- In clips 19 and 31 the sweeper's ball pair is at body +x; in clips 18 and 30 it is at -x.
- The final_v5 / v22 ep30 mesh is bbd4d71c, the same file as public31, so it carries the ep31 convention. It therefore registers on the twin of the ep30 poses.
- Fix: `sw_seedtwin` = the same mesh rotated by its own refined self-twin half turn. The surface moves by a mean of 0.18 cm (p90 0.32 cm), so it is the same physical surface with the marker side swapped.

### Twin margin (best twin-basin score minus best correct-basin score, cm; kit 25 starts + 16-24 random starts; 17-25 sampling seeds)
| clip / object | mesh | margin kit seed | margin mean | min | kit correct (seeds) | kit-choice error vs GT |
|---|---|---|---|---|---|---|
| pub18 sweeper | r8_base (rgbd 2-ball, -x) | 0.011 | 0.010 | 0.005 | 1.00 | 1.01 cm / 9.3 deg |
| pub18 sweeper | sw_seedtwin | 0.013 | 0.010 | 0.004 | 1.00 | 0.93 / 9.3 |
| pub18 sweeper | sw2_c18 / sw3_c18 | 0.004 / 0.006 | 0.008 / 0.008 | | 1.00 / 1.00 | 1.03 / 1.00 |
| pub18 sweeper | WRONG side (sw3_c31) | -0.008 | -0.007 | -0.012 | **0.00** | 6.2 / 28.5 (dust pan 46 cm / 177 deg) |
| pub18 sweeper | sw_core / sw_mk4 (no or both sides) | | 0.001 / 0.001 | | 0.59 / 0.53 | |
| pub31 sweeper | r8_base (bbd4d71c) | 0.011 | 0.010 | 0.004 | 1.00 | 1.00 / 5.7 |
| pub31 sweeper | sw2_c31 / sw3_c31 | 0.010 / 0.010 | 0.008 / 0.007 | | 1.00 / 1.00 | 0.97 / 0.98 |
| pub31 sweeper | WRONG side (sw3_c30 / sw_seedtwin) | -0.012 | -0.010 | -0.014 | **0.00** | 15.5 / 30 (rack 22 cm / 179 deg) |
| pub31 sweeper | sw_core / sw_mk4 | | -0.001 / -0.001 | | 0.24 / 0.35 | |
| pub23 piece_1 / piece_2 | r8_base (v3 markers) | 0.010 / 0.020 | 0.007 / 0.016 | -0.007 / 0.010 | 0.92 / 1.00 | 0.88/2.9, 1.08/2.6 |
| pub24 piece_1 / piece_2 | r8_base (v3 markers) | 0.010 / 0.018 | 0.009 / 0.015 | | 0.92 / 0.96 | |
| pub23 / 24 boards | board_nomk | | -0.004..-0.001 | | 0.06-0.41 | |
| pub23 / 24 boards | board_rb (rebuilt, slot only) | | -0.004..0.000 | | 0.06-0.53 | |
| pub23 / 24 boards | board_mk | | 0.007-0.012 | | 0.88-1.00 | |
| pub23 / 24 boards | board_rb_mk | | 0.006-0.011 | | 0.94-1.00 | |

Eval clips have no GT. The margin is measured relative to the kit's choice, and `reg_compare.py` compares that choice with a mesh built from the clip's own ball detections:
| eval | final_v5 mesh registers | v23b mesh registers | margin (final_v5 / fix) |
|---|---|---|---|
| 19 sweeper | consistent (0.6 deg vs sw3_c19) | consistent (4 deg) | 0.010 / - |
| 30 sweeper | **flipped (179.6-179.8 deg vs sw2/sw3_c30)** | **flipped (179.3 deg)** | fix sw_seedtwin: 0.010 kit seed, 0.010 mean, 1.00 of seeds |
| 25 piece_1 | consistent (1.0 deg vs board_mk) | consistent (0.1 deg) | 0.010 |
| 25 piece_2 | consistent (1.2 deg) | **flipped (180.0 deg)** | 0.015 |

The base template board meshes (what v23b submits for ep25) fail on public: pub23 piece_2 1.25 cm / 179.3 deg (MP 0, MPPE 4.10);
pub24 piece_1 23 cm / 135 deg, piece_2 45 cm / 176 deg (MP 0, MPPE 34.9). Exports: `y9_NEGCTRL_basemesh`.

Honest limit: even the correct margins are about 0.01 cm (0.1 mm of mean surface distance). The scan's ball-driven asymmetry
is small. Adding a physically complete 3rd sweeper ball (sw3) or rebuilding the boards (board_rb_mk) did not enlarge it.

### Public validation of the adopted rule
- Rule: the submitted twin-object mesh's marker side must equal the clip's detected ball side. Public 18/31/23/24 already satisfy it,
  so public 20 is bit-identical to r8_base: 20/20, pos 1.114, ori 5.23, MPPE 1.171, AUC 0.931.
- Applying the ep30 fix mesh itself to public18 (same convention): sweeper 1.01/9.3 -> 0.93/9.3, dust pan 2.03/6.7 -> 2.00/5.0,
  MPPE 1.638 -> 1.600, AUC 0.892 -> 0.897.

### CD-O (kit _object_chamfer_cm vs scan, mean of 5 seeds, cm)
- Sweeper: final_v5 ep19 0.210, final_v5 ep30 0.187, **sw_seedtwin 0.187** (rotation of the same mesh), sw2_c30 0.211, sw3_c30 0.218.
- Boards: piece_1 v3 0.166 -> board_mk 0.158; piece_2 0.187 -> 0.170.

## Rejected
- **board_mk for piece_1.** Public is mixed: pub23 MPPE 0.996 -> 1.239, pub24 1.606 -> 1.377. The margin gain is small.
- **board_mk for piece_2.** Margin 0.016 -> 0.008.
- **board_rb, board_rb_mk.** Board rebuilt as an extruded outline with the slot recut to board thickness. No margin gain.
- **sw_core, sw_mk4, sw3_*.** No gain, or a coin flip.
- **Task 2, rack.**
  - Unmasked stereo over 10 rack clips (public, held-out and eval) puts the rack ends at -22.1..-22.6 / +22.6..+23.1 cm. That is a
    length of 45.0-45.5 cm against the mesh's 44.4 cm, and the 0.6-1.1 cm excess fits stereo edge fattening. Mesh overlays match both image ends.
  - Our evidence does not show a missing end or a 51 cm rack. Note that the kit rescales our mesh to the scan's surface radius, so only proportions matter.
  - Test variant rack_z: stretched along the long axis to the stereo 45.0 cm. CD 0.529 -> 0.517.
  - rack_z poses on pub12 / pub31 / held-out 13 / 26 / 27: MPPE 1.850 -> 1.799, 1.203 -> 1.271, 1.147 -> 1.428, 0.927 -> 0.947, 0.779 -> 0.758.
    Net worse, rejected.
- **Task 2, blue cup.** No change needed. The final_v5 / ep32 blue_cup mesh (415ade56) is already 17.5 cm tall; the 15.6 cm
  figure came from an older mesh. Masked stereo along the cup axis reaches 15.7-17.5 cm (the bottom is occluded). CD 0.177.

## Files
- Tools: `twin_margin.py`, `twin_axis_body.py`, `reg_compare.py`, `chamfer_cd.py` (scorer-side); `build_twin_meshes.py`,
  `build_sweeper_clip.py`, `ball_layout_clip.py`, `fuse_body_points.py` (GT-free).
- Inputs: items_*.json; `margins/*.json`, one per item.
- Meshes: `reconstruction_recovery_20261026/exports/y9_meshes/<variant>/`, with build reports and the sw_seedtwin `twin_transform.npy`.
- Per-clip exports: `canonical/*/episode_*/exports/y9_*`. The adopted one is `canonical/evaluation/episode_000030/exports/y9_swseedtwin`.
