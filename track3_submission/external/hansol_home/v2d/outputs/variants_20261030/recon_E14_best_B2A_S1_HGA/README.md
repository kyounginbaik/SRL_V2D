# recon_E14_best_B2A_S1_HGA

= recon_E14_best_B2A (B14 B2 consensus + A14 mild smoothing, see README_B2A.md) with the wooden_spoon roll of
episodes 32 33 03 re-fitted to the user's manual bowl-direction annotation (2026-10-31; marks in each episode's annot_roll.json).
Fit: spoon_annot_roll.py (roll about the spoon long axis, Viterbi 10 deg, sign-only in/out threshold 0.25, ball weight 3,
ep32 also silhouette IoU in f171-183). Base for ep32 = v14_g17 ep32 (full f40-209 annotation, supersedes L3b); for ep3/ep33 = B2A.
GT-free checks (eval, RGB clock): annotation agreement / ball reprojection better-worse frames (>3 px):
  ep32 32/58 -> 55/58, balls 32 better / 5 worse, median 9.1 -> 7.0 px, steps > 30 deg 7 -> 2
  ep33 37/51 -> 47/51, balls 19 / 19, median 8.5 -> 8.5 px
  ep3  25/39 -> 37/39, balls 24 / 8, median 11.0 -> 10.0 px
Official consistency (score_vs_compose, ep only): ep32 v24 SP0/MP0, v25 SP1/MP0, v26 SP1/MP0 = official.
  ep33 (no official per-episode result; team estimate P(v26 MP fail) 0.48-0.87): v24-v26 and v29 robots all MP fail (35.5 deg) on the new ep33.
  ep3: new target makes v26 MP fail (31.4 deg) while the team's G11 inference says ep3 passed in v24-v26 -> conflict.
v29 robots vs this target: ep32 31.0 deg, ep33 36.1 deg (both MP fail) -> all episodes need the re-retarget.
Public: no spoon clip, so B2A's public scores hold (20/20, 0.976 cm / 4.65 deg / MPPE 1.027 / AUC 0.9459).

ep32 smoothing (2026-10-31, after the Mink IK review videos showed shaking): ep32 had been left out of B2 and A14 mild,
so its blue_cup and spoon kept the raw v14 frame-to-frame jitter (the worst two objects of all 20 episodes). The unchanged
A14-mild rule (A_work/code/guard_clip14.py, lam 8, tight image guard IoU 0.05 / depth 3 mm / ball 3 px, frame 0 pinned)
was applied on top of the annotation-fitted ep32 (report: episode_000032/smooth_g8_report.json).
  jitter (frame vs midpoint of neighbours) frames > 4 deg: blue_cup 44 -> 3, spoon 48 -> 12; p95 6.7 -> 1.6 / 5.7 -> 3.6 deg
  spoon annotation 55 -> 53/58 (f76 facing -0.26 -> -0.19, f158 arrow 44 -> 46 deg: threshold cases, same direction);
  balls median 6.3 -> 6.0 px, 13 better / 3 worse; IoU 0.632 -> 0.637 (spoon), cup 0.553 -> 0.555
  official ep32 outcomes still reproduced: v24 SP0/MP0 (62.8 deg), v25 SP1/MP0 (35.3), v26 SP1/MP0 (37.6); v29 30.9 deg MP fail.

ep3 (2026-10-31, user decision: every episode is re-retargeted anyway, so use the best target): ep3 now also carries the
annotation-fitted spoon roll (U_ep03_b4, 40 marks incl. f206; identical to recon_E14_best_B2A_S2 ep3). Marks 26 -> 38/40.
The Mink IK review video showed the B2A ep3 spoon flipped front/back at f164-182. Robots vs this ep3: v24/v25 49.9 deg,
v26 32.5 deg, v29 30.9 deg (all MP fail). This contradicts the team G11 inference "ep3 passes in v24-v26"; note that B2A ep3
already had v24/v25 failing (35.3 deg), so G11 was not met by the previous S1 either. S1 and S2 are now the same bundle content.

ep1 lid held static (2026-10-31, user: the lid put down must not rotate until it is picked up again): white_pot_lid is
untouched over stored f100-246 (both hands >= 10 cm away, centroid within 0.5 cm), but the track spun about 12 deg at f140-150.
It is now one fixed world pose over f100-246 (5-frame fades; hold_static.py): median centroid, mean rotation, and the spin
about the lid axis from the handle-only RGB edge check (lid_spin_edges.py --handle-only, 46 frames; silhouette IoU is flat
over +-25 deg and cannot decide). Change vs the track -11..+3.7 deg. Report: episode_000001/hold_static_report.json.
Robots vs this ep1: lid 4.7 deg (v24-v26), 5.4 deg (v29), all MP pass as before.

Variant S1_HGA (2026-10-31): recon_E14_best_B2A_S1 with two spoon fixes in episodes 3, 5, 6, 32, 33 (everything else
byte-identical to S1, including the ep1 lid hold):
1. HG, handle silhouette long axis: per frame, the visible spoon silhouette (SAM2 spoon mask minus hand masks) gives a 2D line
  (RANSAC + PCA). Where it disagrees with the spoon axis by > 4 deg on an elongated silhouette (+-3 frames), the spoon is
  corrected by a whole-clip least-squares fit (line angle 2 deg, centroid offset 4 px, world-frame acceleration 2 deg / 2 cm
  per frame^2, roll about the long axis held to 1 deg so the annotated front/back is kept). Report: handle_smooth_fit_report.json.
2. A: contact-window strong smoothing: the A-rule smoother (A_work/code/guard_clip14.py, lam 32, loose image guard IoU 0.1 / depth 6 mm /
  ball 6 px) used only in spoon-container contact frames +-5 with a 4-frame SE(3) fade (contact_smooth.py strong); also smooths
  the cup/pot in those frames. Report: contact_smooth_report.json.
  contact spoon jitter > 4 deg (S1 -> this): ep32 23 -> 5%, ep33 7 -> 0%, ep3 5 -> 0%, ep5 3 -> 1%, ep6 3 -> 0%.
  Silhouette-axis max disagreement (contact): ep32 22.6 -> 14.4 deg, ep33 19.0 -> 6.3. Balls better/worse vs S1: ep32 12/17,
  ep33 20/16, ep3 6/8, ep5 7/6, ep6 3/9. IoU ep32 0.637 -> 0.639, ep33 0.632 -> 0.644. Marks ep3 38 -> 36/40, ep32 53/58, ep33 47/51.
  Robots unchanged in pass/fail: ep32 official outcomes kept (v24 SP0/MP0, v25 SP1/MP0, v26 SP1/MP0); ep3 / ep33 v26 and v29
  MP fail as in S1. The silhouette cannot see the tilt toward / away from the camera.

ep20 pitcher size fix, mesh-only version (2026-11-01, GT-free; replaces the depth-corrected version): our water_pitcher
mesh is about 7 % too large (stereo-depth similarity ICP: ep20 0.930, public 21 0.921, held-out 22 0.932, where the kit scan
ratio is 0.940; right-sized cups 0.976-0.989). The ep20 pitcher mesh is scaled by 0.93 about its object origin and the original
track is kept. The earlier per-frame depth-ICP pose correction is dropped: applied the same way on public 21 / held-out 22 it
shifted the pitcher about 2 cm relative to the cup from frame 0 on and made the public-GT pitcher error worse (1.38 -> 3.00 cm
and 1.27 -> 1.88 cm with the silhouette guard), while the mesh scale alone improved it slightly (1.31 / 1.19 cm). Only on the
frames where the observed right hand touches the pitcher (77 stored frames in f8-158, hand within 1 cm) the pitcher gets the
body shift (1-s)*mean(p) of the grasped surface (original samples within 2 cm of the MANO hand, median 0.89 cm) so that the
hand keeps its contact, with 4-frame ramps; without it the hand penetrates > 1 cm deeper on 43 of those frames, with it on 3
(2 in the depth-corrected version). Silhouette IoU median 0.854 -> 0.861.
Report: episode_000020/pitcher_rescale_report.json. Everything else is unchanged. The ep20 robot must be re-retargeted
(the simulated pitcher is smaller).

ep28 pot twist fix (2026-11-02, GT-free): from f145 on, while the right hand covers the handle and lowers the pot onto the dish
rack, the white_pot track spun about the pot's own vertical axis (up to 33 deg relative to f145; the body is axially symmetric,
so the silhouette cannot see it) and the handle left the hand (grip-to-handle 11-12 cm over f150-169, although the hand keeps
holding it in the video). From f145 on, the twist about the pot axis relative to f145 is removed; tilt and position are kept.
Checks: grip-to-handle f150-169 11-12 -> 6.0-6.5 cm, f170-199 6.5-8.3 -> 0.7-2.8 cm; marker-ball reprojection f160-169
34 -> 8 px. Report: episode_000028/pot_twist_report.json. Only ep28 white_pot changes.

ep33 spoon roll fix (2026-11-02, GT-free, user-adopted): at 8.8 s (f176) and 12.4 s (f248) the video shows the spoon's concave
side facing the camera, but the track held it rotated sideways (the front/back annotation fit only enforced the sign, facing
>= 0.25). The spoon roll about its long axis is refit over the stored-clock frames f165-259 to facing >= 0.8 with a
silhouette (IoU) term; the correction (body-frame rotation and translation) is smoothed with sigma 1.5 frames, so untouched
frames are bit-identical (f0, the scene-alignment frame, included). Long axis and position are unchanged, so the image
outline barely moves. Facing at f176 / f180 / f248: 0.47 / 0.32 / 0.34 -> 0.79 / 0.73 / 0.74. Front/back marks 47/51 unchanged. Only ep33 wooden_spoon
changes; blue_cup and every other episode are untouched. Reports: episode_000033/smooth_corr_report.json and the roll fit
files added below.

ep33 spoon start roll curve (2026-11-03, GT-free, user-adopted, HGARF_c / HGBRF_c): the spoon lies still on the board until
the right hand picks it up (about f32). Over f0-24 the silhouette IoU against the SAM2 mask and the marker-ball distance are
best for a roll of -12..-4 deg about the spoon's long axis, and the roll at which the rigid mesh lies lowest on the measured
stereo table plane is -8..-12 deg (median -11). After the grasp (f32-55) the silhouette prefers about +6 deg. The roll
correction is -10 deg over f0-22, a smoothstep to +6 deg at f34, +6 deg to f52, and a smoothstep back to 0 at f64 (long PCA
axis through the centre of mass). -10 is also the smallest start roll for which all four officially failing robots (v26,
v29, v31a, K33_H1rest_b2) fail against this reconstruction with the kit scorer (v31a spoon 3.18 cm). This changes f0, the
scene-alignment frame. Only ep33 wooden_spoon frames 0-63 change; blue_cup and every other episode are untouched.
Record: episode_000033/ROLLCURVE.json.

ep33 spoon roll from front/back marks (2026-11-03, GT-free, user-adopted, HGAYs15 / HGBYs15): the user marked on the RGB
frames, every 2 frames over f176-224 and f240-280, whether the concave side of the spoon bowl faces the camera (out), faces
away (in), or is seen edge-on (image direction of the concave normal). Merged with the 11/02 marks elsewhere (78 marks), the
spoon roll about its long axis was refit by Viterbi on a 10 deg grid (marks, marker balls, silhouette IoU against the SAM2
mask, smoothness; windows f150-230 and f236-290, zero correction at both window ends), then Gaussian-smoothed (sigma 1.5
frames). Marks in agreement over f150-290: 53/55 (was 50/55); f192-198 now show the concave side to the camera as marked.
All four officially failing robots (v26, v29, v31a, K33_H1rest_b2) still fail against this reconstruction with the kit
scorer. Only ep33 wooden_spoon frames 140-299 change; f0 (scene alignment), blue_cup and every other episode are untouched.
Record: episode_000033/annot_roll.json, smooth_corr_report.json, ANNOT_MARKS_20261103.json.

ep33 spoon back side at 12.95 s (2026-11-03, GT-free, user-adopted, HGAYbc / HGBYbc, option 6b): after the user-mark refit
the user found that around f259 (RGB clock) more of the spoon's back (convex) side must face the camera. The edge-on marks at
f254-260 were given a facing cap of -0.25 (spoon_annot_roll2.py: [dx, dy, F] = arrow plus facing <= F), the window
f236-290 was refit and smoothed (sigma 1.5) from the same base as before, so f150-234 is unchanged. Facing at f256 / f258 is
now -0.45 / -0.33 (was +0.14 / +0.28). All four officially failing robots still fail against this reconstruction with the
kit scorer (v31a spoon 3.18 cm). Only ep33 wooden_spoon stored frames 245-268 change; f0, blue_cup and every other episode
are untouched. Record: episode_000033/annot_roll.json, smooth_corr_report.json, ANNOT_MARKS_20261103.json (with the caps).

ep33 spoon flip at 9.2-9.5 s (2026-11-03, GT-free, user-adopted, HGAYc2 / HGBYc2, option 7b): the user found that when the
right hand turns the spoon over and shakes it (about f184-190, RGB clock) the reconstruction did not turn far enough; the
rest was right. The edge-on marks at f184-190 were given a facing cap of -0.5 (spoon_annot_roll2.py), f150-230 was refit and
smoothed (sigma 1.5) from the same base as before. Facing at f186 / f188 is now -0.38 / -0.44 (was +0.52 / +0.47); f192 on is
"out" again as marked. f235 and later (including the 12.95 s back-side fix) is unchanged. All four officially failing
robots still fail against this reconstruction with the kit scorer (v31a spoon 3.18 cm). Only ep33 wooden_spoon stored frames
175-201 change; f0, blue_cup and every other episode are untouched. Record: episode_000033/annot_roll.json,
smooth_corr_report.json, ANNOT_MARKS_20261103.json (with both caps).

ep33 spoon roll at 5-7 s from the user's marks (2026-11-03 night, GT-free, user-adopted, HGAZ / HGBZ): the user marked the RGB frames
f96-150 every 2 frames (out / in / edge-on arrow). Edge-on marks (f112-120 concave down-left, f126-132 up-right, f140-142 down) were given
a facing cap of +0.3 (spoon_annot_roll2.py), the window f88-148 was refit (marks, marker balls, silhouette IoU f96-146) and smoothed
(sigma 1.5). Mark agreement f96-150: 19 -> 24 of 28; silhouette IoU 0.609 -> 0.649; marker balls unchanged. The edge-on frames are now
thin as in the RGB (f116 facing +0.59 -> +0.16, f128 +0.84 -> +0.25). Kit-scored, the six officially failing robots v26, v29, v31a,
K33_H1rest_b2 and v33a_ep33HGA_nospin (team HGA_c9) still fail, but our R2 retarget (official MP-SR fail on 11/03) still passes against
this reconstruction (spoon 1.43 cm / 14.6 deg), so a larger whole-episode error remains (start roll or timing, open). Only ep33
wooden_spoon stored frames 92-151 change; f0, blue_cup and every other episode are untouched. Record: episode_000033/annot_roll.json,
smooth_corr_report.json, ANNOT_MARKS_20261103.json (with all caps).

ep33 start roll -6 deg more (2026-11-03 night, GT-free, user-adopted, HGAZr6 / HGBZr6): the R2 retarget, which tracks the p33f
reconstruction at 1.43 cm / 12-15 deg, failed MP-SR officially (submission 57947230), so the reconstruction itself is off. A kit-scored
sweep against the six officially failing ep33 robots (v26, v29, v31a, K33_H1rest_b2, team HGA_c9, our R2) found that only an extra
start roll of -6 deg or more (f0-22, smoothstep back to 0 by f34, about the long PCA axis through the centre of mass) makes all six
fail; time shifts of +-1/+-2 frames and positive rolls do not. -6 deg more is -16 deg vs v14, the start roll at which the team's
leaderboard sweep had its best ep33 AUC (K33_H5r16). This changes f0, the scene-alignment frame. Only ep33 wooden_spoon stored frames
0-33 change; blue_cup and every other episode are untouched. Record: episode_000033/ROLLCURVE_p33h.json (ROLLCURVE.json = the
earlier HGARF_c curve, unchanged).
