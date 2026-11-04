# Stream W8 (phase 8, v23): episode 10 nest depth (beige_cup in blue_cup), 2026-10-28

Question: target_final_v5 puts the beige cup ~3.3 cm deeper into the blue cup than our meshes allow (rim-to-rim h ~2.5 cm vs
~6 cm non-penetrating). Should the reconstruction move? Evidence was our own SAM2 masks / stereo / meshes only. GT entered only
the unchanged public scorer (score_bundle.py).

**Result: the image evidence supports the deep nest, so the target does NOT move. No selected20_final_v6 / target_final_v6 was
built, and part 2 (robot re-score/retarget against v6) did not apply.** N8_ep10_up3c512 stays the ep10 tag.

## Tools (code/)
- `nestlib.py`: container axis, fcl mesh-mesh collision + min non-penetrating axial offset, embree ray-cast renderer,
  `render_nested` = joint render where a container wall is opaque (the inner object's outer surface below the outer rim, seen from
  outside, is hidden; the outer's inner wall below the rim is covered by an inner object within 15 mm behind it). This makes
  the image test independent of the ~mm wall interpenetration. The first plain z-buffer / depth-bias scans (`scan_ep10.py`,
  out/scan_*_eps*.json) were rendering artifacts (beige outer wall pierced the blue wall, or blue showed through the beige opening). Do not use them.
- `scan2_ep10.py`: per frame, a joint 2-D fit of the blue axial shift sb (-3..+3 cm) x coaxial beige rim-to-rim h (1..8 cm).
  Score: 3-class pixel agreement with the SAM2 masks, hands excluded. Output in out/scan2_*.json and logs/scan2_*.log.
- `radial_fit.py`, `radial_fit2.py`: silhouette radial scale of a cup mesh about its axis (2: jointly with depth along the ray).
- `lip_test.py`, `geo_v5.py`, `geo_any.py`, `census.py`: geometry/collision diagnostics. `panels.py`, `overlay.py`: images in out/.
- `nest_rule.py`: the general rule (below). `run_all.sh` runs it on all 40 clips -> `exports/w8_nest` (reports out/w8_nest_*.json).

## (a) Image evidence: deep nest (h ~2.5-3 cm), physical nest (h ~6.5 cm) rejected
Profile = max over the blue shift of agreement at a given h (logs/scan2_*.log):

| frame | v5 h cm | best (sb, h) | acc@h2.5 | acc@h4.5 | acc@h6.5 |
|---|---|---|---|---|---|
| 92 (held, still inserting) | 5.49 | (-1.0, 6.0) | 0.777 | 0.873 | 0.921 |
| 96 | 5.28 | (-1.0, 6.0) | 0.802 | 0.895 | 0.921 |
| 100 | 3.09 | (-1.0, 4.0) | 0.868 | 0.898 | 0.835 |
| 104 | 3.14 | (-0.5, 3.0) | 0.908 | 0.878 | 0.795 |
| 108 | 2.05 | (-0.5, 3.0) | 0.905 | 0.852 | 0.761 |
| 112 | 2.66 | (-0.5, 3.0) | 0.908 | 0.857 | 0.773 |
| 124 | 2.59 | (-0.5, 3.0) | 0.906 | 0.854 | 0.764 |
| 128 | 2.49 | (-0.5, 3.0) | 0.900 | 0.847 | 0.759 |
| 132 | 2.42 | (0.0, 2.5) | 0.907 | 0.841 | 0.755 |
| 136 | 2.59 | (-0.5, 3.0) | 0.910 | 0.850 | 0.763 |
| 140 | 2.19 | (-0.5, 3.0) | 0.908 | 0.847 | 0.757 |

The method does find a shallow nest when there is one: in f92-96, while the beige is still being lowered, it picks h = 6.0 cm, close to v5's 5.3-5.5.
In the settled nest (f104-160) it picks h = 2.5-3.0 cm, close to v5's 2.2-3.1, and moving the beige up to 6.5 cm costs about 0.15 pixel agreement in every frame.
Visually (out/p3.png, out/panels_*_nested15.png), v5's beige rim top and the beige/blue boundary line up with the masks, while the
h = 6.5 render puts the beige rim well above the mask. Stereo was not used as an absolute cue; R8 measured a 5-10 mm bias toward the camera.
The relative beige-minus-blue depth residual stays within about 4 mm at v5, so it gives no sign that the beige sits higher.

Public11 (same two meshes, md5 b30ae10b / 415ade56, same task) independently has the same deep nest in our policy (h 2.5-3.1 cm from
f95 on) and scores beige 0.93 cm / 3.0 deg. Forcing the non-penetrating nest there: see the rule table.

## (b) Are the meshes oversized?
- Outer silhouette: no. With depth along the ray free, the best radial scale is 0.995 (blue) and 1.000 (beige) on unoccluded frames 0-40, IoU 0.95.
  With depth fixed, the fit prefers 0.96, but that is the pose sitting about 1.5 cm too close to the camera, not mesh size: the silhouette cannot separate
  depth from size, and the fit only rescales radially.
- The conflict sits at the rim and wall, below what the silhouette can resolve. Both scans are thin shells (rin ~ rout within 0.2 mm), except
  for a 2.8 mm inward rim band on the blue cup (rin 4.09 vs rout 4.37 cm at z 15.2). At equal height the beige wall is also 0.5-0.8 mm wider than the blue
  (4.15 vs 4.07 cm at z 10.7), although these are the same stacking-cup design. Removing the blue's top 2.5 cm lets coaxial h = 4 cm become
  collision-free (lip_test.py). The remaining gap down to 2.5 cm is about 1-2 mm of radius along the taper (slope 0.062).
- Verdict: the meshes are plausibly 1-3 mm too large in inner rim/wall radius for these real nesting cups. The deep nest is the real one;
  non-penetration with our meshes is a mesh artefact, not pose evidence.

## General rule (nest_rule.py) and validation
Rule: for each object pair (outer = larger bounding volume), a frame is a nest frame if both meshes are elongated (PCA variance ratio
>= 1.5), the inner axis is within 20 deg of the outer axis, the inner centroid is within 1.5 cm of the outer axis and inside its height, and the meshes
intersect (fcl). Candidate = the inner made coaxial and lifted to the smallest non-intersecting offset along the outer axis. A window (>= 3 frames)
is adopted only if the median change in container-opaque 3-class mask agreement is >= -0.005, so the image does not reject it. Edges blend over 5 frames.

| clip | nest frames | lift median | agreement change | adopted |
|---|---|---|---|---|
| eval10 | 74 (f87-160) | 3.96 cm | -0.204 | no |
| public11 | 74 (f63-136) | 3.34 cm | -0.211 | no |
| all other 38 clips | 0 | - | - | - |

So the rule fires nowhere: public 10/10 and held-out 10/10 are bit-identical to r8_base (20/20 checked), all 20 eval clips are
bit-identical to final_v5, and the means are unchanged.
Public11 before/after: rule (gated) = before = blue 0.32 cm / 1.41 deg, beige 0.93 cm / 3.00 deg, MP-SR 1.
Forced (no gate, `exports/w8_nest_nogate`, i.e. the physically consistent nest adopted regardless of the image): blue 0.32 / 1.41 (same), **beige 0.93 -> 2.23 cm** (3.00 -> 1.63 deg), MPPE 0.66 -> 1.28, AUC 0.983 -> 0.915, RPE 0.80 -> 2.18. The public GT agrees with the image gate: the real cups nest deep, and moving to non-penetration with our meshes is wrong by about 3 cm on the nested frames.

## Robot (part 2)
Not run: part 1 was not adopted. Ep10 stays N8_ep10_up3c512, build-v2 exact true, score_any vs target_final_v5 blue 0.51/1.3,
beige 1.48/2.7 (stream N8). The 1.3 cm beige miss comes from our cup meshes being unable to reproduce the real deep nest in physics. It is
not a reconstruction error to fix in the target. Options that remain: a thinner-rim/narrower collision proxy for the cups in the
simulation only (the scored visual meshes stay the same), which is outside this stream's scope.

## Outputs
New only: `exports/w8_nest` (all 40 clips, identical copies), `exports/w8_nest_nogate` (eval10, public11). Nothing in final_v5/target_final_v5 was touched.
