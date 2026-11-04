# Stream R8 (phase 8): GT-free audit of the 20 eval reconstructions (selected20_final_v5)

Scope: the reconstruction itself, not rollouts. Public GT entered only the unchanged public scorer
(`score_bundle.py` -> `bscore`); every eval-side number here is GT-free. Our meshes only. Nothing earlier was
overwritten: new tools, new export folders `exports/r8_*` and this directory only.

**Result: no `selected20_final_v6` was built.** Each correction tried either made public/held-out worse, was neutral
there, or (the one that left public bit-identical) made eval image evidence worse. Rollout streams should stay on
`selected20_final_v5`.

## Tools (new, in `tools/reconstruction_audit/`)
- `anchor_audit.py`: per object, the initial static window, frame-0 deviation from the window's robust mean, table
  contact/tilt against the stereo support plane, the scorer lever arm, and spoon ball reprojection. Outputs are in `audit/v5/*.json` (eval + 20 public);
  `summarize.py v5` prints the table.
- `table_rest.py`: (a) `--correction none` replaces the initial static window by its robust mean pose and fades the
  change out; (b) `--correction tilt [--snap]` additionally rotates/shifts the object so it rests on the
  measured table plane in the nearest quasi-static stable pose of our mesh (window-only or constant body bias).
- `static_refit.py`: multi-frame stereo partial scan over the static window, then point-to-plane IRLS ICP of our
  mesh (rigid or translation only), applied to the window or as a constant body bias.
- `twin_continuity.py`: Viterbi over {pose, half-turn twins about principal axes} for meshes whose own twin
  residual is < 0.5 cm (sweeper, wooden pieces; spoon excluded), followed by a single-frame despike.
- In this directory: `target_delta.py` (scorer-faithful difference between two reconstructions, same mesh on both
  sides), `f0_sensitivity.py`, `flip_impact.py`, `risk.py`, `compare.py`, `make_*.py`, and the sbatch wrappers.
- `support.py` was run (no `--export`) on the 34 clips without `support_plane.npz` (job 60710). That was additive only.

## Public / held-out validation (20 clips, 33 objects; GT only in the scorer)
| variant | MP-SR | mean pos cm | mean ori deg | MPPE | AUC | verdict |
|---|---|---|---|---|---|---|
| r8_base (= final policy: fused_v7_gap_ds[_lid]_rs) | 20/20 | 1.114 | 5.23 | 1.171 | 0.9312 | baseline |
| f0avg_all (static-window robust mean, all objects) | 20/20 | 1.129 | 5.04 | 1.182 | 0.9294 | neutral, rejected (pub21 pitcher 1.49 -> 2.18) |
| f0avg_anchor | 20/20 | 1.130 | 5.05 | 1.183 | 0.9293 | rejected |
| tilt_win_all (table-plane rest) | 20/20 | 1.162 | 5.77 | 1.247 | 0.9224 | rejected |
| tilt_body_all | 20/20 | 1.163 | 5.51 | 1.247 | 0.9226 | rejected |
| tilt_body_anchor | 20/20 | 1.174 | 5.43 | 1.253 | 0.9227 | rejected |
| snap_body_all (rest + snap to plane) | 19/20 | 1.366 | 5.51 | 1.415 | 0.9069 | rejected |
| refitT_win_all (stereo ICP, translation) | 20/20 | 1.309 | 5.09 | 1.353 | 0.9143 | rejected |
| refitT_body_all | 19/20 | 1.291 | 5.09 | 1.339 | 0.9123 | rejected |
| refitR_body_all (stereo ICP, rigid) | 20/20 | 1.230 | 5.02 | 1.304 | 0.9201 | rejected |
| twin_continuity + 1-frame despike | 20/20 | bit-identical on all 20 | | | | public-neutral; eval QC worse (below), rejected |

What the failures say:
- **Table-plane evidence is biased, not pose error.** Lowest-vertex "penetration" is constant per mesh type in
  every clip, public and eval: rack about -20 mm, pitcher -23, cups -11, pot -8. The tilt to the nearest stable pose is
  also constant per mesh (rack about 6 deg, pot 3-4 deg). This comes from mesh-bottom geometry and the stereo
  table fit, so it cannot be used as a correction.
- **Stereo is about 5-10 mm toward the camera relative to our mesh surfaces.** The signed normal residual is +2 to +5 mm, and the
  refit asks for -z shifts in almost every object, and cups for 2-3 cm, because of interior/rim points. Public
  scoring rejects it, which agrees with the earlier rejection of ball stereo depth.
- **Frame 0 is about as good as the static-window mean.** The tracker registers at frame 0, and averaging moved errors
  both ways, so the frame-0 deviation is not a reliable error signal on its own.
- **Half-turn twins are not image-equivalent in eval19's in-fist window.** On the changed frames 91 and 95-97, IoU
  dropped from 0.51-0.74 to 0.33-0.46. Frame 94 went from 0.05 to 0.27. The depth residual rose on all five (e.g. f95 7 -> 32 mm). Steps over 30 deg went from 6 to 2 and the max step from 156 to 54 deg.
  This trades image evidence for smoothness, so it fails criterion (b).

## Anchor frame-0 audit (GT-free)
The kit rotates the whole scene by the anchor's frame-0 orientation (anchor = first non-symmetric slot) about slot
0's frame-0 position. Shift = scorer-faithful difference if the anchor's initial pose were the static-window
mean (`f0_sensitivity.json`), i.e. the size of the frame-0 uncertainty:

| ep | anchor | static frames | f0 dev deg / mm | window spread deg | largest scored shift |
|---|---|---|---|---|---|
| 1 | white_pot | 80 | 1.2 / 3.5 | 1.4 | white_pot_lid 0.51 cm / 1.2 deg |
| 3 | white_pot | 65 | 0.8 / 3.3 | 0.7 | white_pot 0.30 cm / 0.8 deg |
| 5 | white_pot | 28 | 0.5 / 1.3 | 0.5 | wooden_spoon 0.27 cm / 0.5 deg |
| 6 | white_pot | 37 | 0.4 / 3.8 | 1.5 | wooden_spoon 0.45 cm / 0.4 deg |
| 8 | white_pot | 80 | 0.7 / 1.4 | 0.5 | white_pot 0.28 cm / 0.8 deg |
| 10 | none (blue axis + blue->beige) | - | - | - | 0 |
| 14 | plastic_dish_rack | 80 | 0.5 / 0.6 | 0.9 | rack 0.22 cm / 0.9 deg |
| 15 | plastic_dish_rack | 80 | 0.5 / 2.9 | 0.4 | rack 0.32 cm / 0.7 deg |
| 16 | plastic_dish_rack | 80 | 0.3 / 1.3 | 0.7 | rack 0.20 cm / 0.6 deg |
| 19 | mini_sweeper | 32 | 0.9 / 1.0 | 1.9 | dust pan 0.24 cm / 0.9 deg |
| 20 | water_pitcher | 29 | 0.3 / 0.4 | 0.2 | pitcher 0.21 cm / 0.4 deg |
| 25 | wooden_piece_1 | 40 | 0.1 / 0.6 | 0.3 | piece_1 0.08 cm / 0.1 deg |
| 28 | plastic_dish_rack | 26 | 0.4 / 2.0 | 0.5 | white_pot 0.35 cm / 0.4 deg |
| 30 | mini_sweeper | 48 | 0.3 / 0.5 | 1.3 | sweeper 0.11 cm / 0.6 deg |
| **32** | **wooden_spoon** | 24 | **5.9 / 4.5** | 1.2 | **wooden_spoon 2.19 cm / 5.9 deg**, cup 0.48 / 5.8 |
| 33 | wooden_spoon | 49 | 1.2 / 0.6 | 3.7 | spoon 0.28 cm / 1.6 deg |
| 35 | wooden_spoon | 11 | 2.0 / 0.7 | 1.0 | spoon 0.47 cm / 2.0 deg |
| 36 | wooden_spoon | 34 | 2.2 / 1.1 | 1.4 | spoon 0.47 cm / 2.2 deg |
| 37 | wooden_spoon | 34 | 0.3 / 1.3 | 0.7 | spoon 0.14 cm / 0.5 deg |
| 38 | mini_dust_pan | 44 | 1.5 / 0.9 | 0.4 | dust pan 0.28 cm / 1.6 deg |

The spoon is the rotation anchor in five episodes (32, 33, 35, 36, 37). In ep32, frame 0 is a 5.9 deg outlier in an
otherwise tight window (per-frame deviation 5.9, 1.7, 2.7, 1.2, 3.2, ...). Ball reprojection cannot decide between
frame 0 and the window mean (5.82 vs 5.71 px), so this is an unresolved ~2 cm / 6 deg uncertainty on the ep32 spoon. The cup
(slot 0, which sets the translation) is held in hand from frame 0. Its overlay fits the rim, and its low IoU (0.54) comes
from hand occlusion.

## Spoon (no public validation)
Ball reprojection of final_v5, clip median / p90 px: 3: 9.1/94, 5: 4.8/62, 6: 2.6/28, 15: 7.0/44, 16: 6.3/53,
32: 8.8/139, 33: 9.9/149, 35: 8.3/54, 36: 7.8/109, 37: 3.3/37. The p90 values over 100 px (32, 33, 36) are in-hand
windows. Our spoon mesh's half-turn twin residual is 0.37-0.57 cm, against 0.15-0.25 for the sweeper and pieces,
so the scorer's registration of the spoon is probably unambiguous.

## Sweeper and wooden-piece twins (19, 25, 30)
Our own meshes are nearly self-congruent under a half turn: sweeper major axis 0.15 cm (ep19) and 0.25 cm (ep30),
pieces 0.23-0.26 cm about all three axes. If the scorer's ICP lands on the twin, `flip_impact.py` gives:
ep19 dust pan 11-40 cm / 180 deg, ep30 rack 7-10 cm / 180 deg, ep25 piece_2 14-42 cm / 180 deg (anchor flip) or
piece_2 alone 180 deg. Each of these would be an SP-SR failure. Mitigating evidence (GT used only by the public scorer):
the marker-informed meshes registered on the correct twin in all six public registrations (18, 31 sweeper;
23, 24 both pieces). The ep30 mesh is byte-identical to the public31 mesh, which passed. The ep19 and ep25 meshes are
episode-specific and have no public counterpart.

## Risk ranking (`risk.py`, `risk_ranking.json`)
Expected hidden error per object = v22b proxy vs final_v5 + public mean error of the final policy for that object
type + frame-0 shift above. The spoon prior (1.3 cm / 8 deg) is assumed, because there is no spoon in the public set. The sum is linear,
so it is conservative. Risk = max(pos/3 cm, ori/30 deg) of the worst object.

| rank | ep | risk | driver |
|---|---|---|---|
| 1 | 32 | 1.80 | spoon anchor frame 0 (5.9 deg outlier, 2.2 cm scored shift) + spoon proxy 1.92/16 + in-hand steps (6 steps > 30 deg) |
| 2 | 19 | 1.30 + twin | dust pan proxy 2.38; sweeper anchor twin coin flip; in-fist window f88-97 unresolved (IoU 0.05-0.55, 86-156 deg steps) |
| 3 | 1 | 1.29 | lid orientation (public lid prior 25 deg comes from held-out ep0/2 flips; eval1 knob is up, so likely lower) |
| 4 | 3 | 1.24 | spoon proxy 2.18/22, spoon 75 % half-turn corrected by balls |
| 5 | 5 | 1.22 | spoon proxy 2.10/21 |
| 6 | 37 | 1.21 | spoon proxy 2.20 cm |
| 7 | 28 | 1.19 | pot proxy 2.19/15 |
| 8 | 35 | 1.16 | spoon anchor, 11-frame static window, f0 shift 0.47 cm / 2 deg |
| 9 | 36 | 1.16 | spoon anchor, f0 shift 0.47 cm / 2.2 deg; hand-over roll window (rescued f295-305) |
| 10 | 25 | 1.15 + twins | both pieces are twin-ambiguous meshes (piece_1 is the anchor) |
| 11 | 33 | 1.13 | cup proxy 2.06, spoon in-hand p90 149 px |
| 12-20 | 16, 6, 15, 8, 38, 10, 14, 20, 30 | 1.10 to 0.70 | 30 = twin risk only (mesh identical to the passing public31 mesh) |

The official v22b results were SP-SR 0.85 (3 gross failures) and MPPE 3.11 vs 1.77 proxy. Three ~10 cm failures alone
would explain the MPPE gap, which fits twin flips (19, 25, 30) or a gross spoon error (32) better than a uniform
~1 cm reconstruction bias.
