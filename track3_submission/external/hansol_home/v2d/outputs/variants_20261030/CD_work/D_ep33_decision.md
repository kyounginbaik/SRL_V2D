# Variant D (ep33 only): not built

Decision (2026-10-30): **no D bundle**. No GT-free image evidence supports changing ep33, and the one available general rule (the G15 clock fix) is not specific to ep33.
On ep33 alone it would most likely leave an upload flat, so the upload would tell us nothing.

## How the scorer anchors ep33
- slot 0 = blue_cup (translation pivot); the rotation comes from the wooden_spoon (first asymmetric object) at frame 0.
- Lever: the spoon averages 23.3 cm (max 30.5 cm) from the cup's f0 position over the clip.
  - A 1 deg spoon f0 orientation error moves the spoon 0.21-0.39 cm time-mean (worst axis: world y) and the cup 0.06-0.10 cm.
  - So a constant spoon f0 error of about 5 deg or more, on top of about 1 cm of tracking error, would be needed to fail MP by position.

## Checks (v8_g10b ep33, our data only)
| hypothesis | evidence | verdict |
|---|---|---|
| spoon f0 anchor outlier | f0 is 0.65 deg / 0.08 cm from the mean of its own static window f1-14 (Z9 found 1.2 deg, about 0.3 cm scored) | no |
| constant spoon roll bias (like ep3/ep32) | G16's six estimators give 0 / +5 / +15 / +10 / +4.0 / -4.2 deg: the signs disagree, and the whole-clip 2D roll is 0 [CI 0, 0] | no consistent bias |
| spoon roll flips / tip-handle flips | G16: no Viterbi departure > 60 deg; silhouette rejects end flips by 0.2-0.3 IoU | no |
| object depth/scale vs stereo | spoon r = 0.996 (dz -2.2 mm, f0-30 and f336-346), cup r = 0.995 (public cups 0.995-0.997); same estimator as G12 | no |
| cup silhouette / placement | cup IoU median 0.87, f0 0.88, as on public; G17: cup f31-42 raw IoU slightly better, about 0.2 cm time-mean | negligible |
| hand-over windows f158-202 | 3 frames with IoU 0.1-0.3 (Z9), ball px > 120 on f189-191 for both hypotheses; under 2 % of frames | negligible time-mean |
| clock offset | constant 0.875 frame, like 36/40 clips (no step). Kit-style v8 vs v12b difference: spoon 0.86 cm / 5.7 deg, cup 0.44 cm / 2.2 deg | general rule, already in team v28 |

## The most plausible remaining ep33 failure mechanism (not fixable within this task's rules)
- The kit registers our spoon mesh onto the hidden scan, and the anchor rotation inherits that registration error. That error is multiplied by the 23 cm lever.
- G18 found that our spoon mesh has no mocap balls and an inverted inner shell. Its realistic-truth simulation gives a registration error of median 1.2, p90 3.9, max 7.0 deg.
  At 0.39 cm/deg that is up to about 2.7 cm. G18's simulation still keeps ep33 MP-pass, with p90 spoon position improving under the fixed mesh.
- The fix is a mesh change: G18's `wooden_spoon_g18.obj`, already packaged by the team as v29. Changing meshes is outside the C/D rules (our meshes, unchanged).

## Why an ep33-only retime bundle was not built anyway
- Under G15's retimed truth, the current ep33 rollout (C9_ep33_beam4) is spoon 1.15 cm / 8.7 deg, a pass with margin.
- So if ep33 is failing officially, the clock is not the cause, and an ep33-only retime upload would most likely be flat.
- The rule is already in the team's v28/v29 for all episodes.
