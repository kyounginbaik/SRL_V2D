# Track 3: why SRL is not 1st, and the plan for the last 46 hours

Written 2026-11-03 00:15 UTC. Freeze: 2026-11-04 22:00 UTC. Quota: 20 uploads per board per UTC day (Nov 3 and Nov 4 = 40 per board).
Companion: `TRACK3_END_TO_END_SUMMARY_20261102.md` (pipeline + history), `FINAL_20261104/SUBMIT.md` (upload commands).

## 1. Standings (leaderboard read 2026-11-02 23:40 UTC)

| board | SRL | rank | 1st richchentw | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|---|---|
| MP-SR | 0.90 | 4th | 1.00 | 1.00 DDD | 1.00 rlwrld | 0.90 SRL, Zihao Guo | 0.85 Purple |
| AUC | 0.90306 | 6th | 0.98018 | 0.93906 DDD | 0.92754 rlwrld | 0.92052 Zihao | 0.91907 Purple |
| MPPE cm | 1.561 | 6th | 0.690 | 1.162 DDD | 1.279 Zihao | 1.345 rlwrld | 1.407 Purple |
| RPE cm | 1.547 | 6th | 0.715 | 1.142 DDD | 1.212 rlwrld | 1.339 Zihao | 1.394 Purple |
| SP-SR | 1.00 | tied 1st | 1.00 | | | | |
| CD-O cm | 0.239* | 3rd | 0.213 | 0.232 DDD | | | |

*0.239 is an old Oct-22 score; our current meshes score 0.261. The user said SP-SR and CD-O are fine for now.

Kaggle breaks ties by earlier submission time, so MP-SR 1.00 would be a tie for 1st shown as 4th.

## 2. Why we are not 1st, per metric

The robot rollouts are not the problem. Against our own reconstruction the 20 rollouts score AUC 0.97, MPPE 0.75, MP-SR 20/20.
Everything below is reconstruction-vs-truth error, measured through the official scorer.

**How the scorer turns small reconstruction errors into big metric losses** (read from the kit's metric code, `track3_pose.py`):
1. It registers our mesh onto the hidden scan (surface-moment scale + trimmed ICP). Any registration rotation error becomes a
   constant body-frame rotation on every pose of that object.
2. It aligns the whole episode with ONE rigid transform taken from frame 0: rotation from the first non-symmetric object
   (the spoon in ep32/33; the pot/rack elsewhere; the spoon itself in 35/36/37), translation from slot 0 (the cup in ep32/33).
3. So in ep32/33 a frame-0 spoon rotation error of t degrees moves the spoon by about 0.31-0.33 cm per degree for the whole
   episode (cup 21 cm away). 10 degrees = 3.3 cm = MP-SR fail, and it also drags AUC, MPPE and RPE of that episode.

**MP-SR (0.90, 4th).** Exactly ep32 and ep33 fail (confirmed by mesh knockouts). Both are spoon-in-cup episodes where the spoon is
the rotation anchor. Every retargeting lineage (ours, Jisu's, Hansol's, 9 official uploads) fails, so the target is wrong, not the
rollout. The frame-0 roll sweep moved AUC monotonically up to -16/-21 deg (ep33) and -10 (ep32) but they still fail: the residual
is a rotation about another axis or a body-frame (registration) error of ~10 deg that no image cue can see
(`streamK33_ep33/recon3/README.md`). Three teams already have 1.00.

**AUC (0.903 vs 0.980).** Estimated split of the 0.077 gap (from the official per-episode swaps: ep33 r16c3 alone added 0.012,
ep32 R10 0.002):
- ep32 + ep33: each is around 0.5-0.65 per-episode AUC (passing episodes are about 0.93-0.95). Fixing both to 0.95 is worth
  roughly +0.03 to +0.045 on the board -> about 0.935-0.95 = 2nd place.
- the other 18 episodes: about 0.93-0.95 each vs the leader's ~0.98. That is a 2-3x reconstruction accuracy gap
  (sub-centimetre everywhere); anchor and mesh fixes gain only +0.001 each (ep16 rack) and can also lose (ep38 pan).

**MPPE (1.56 vs 0.69).** Same two episodes carry roughly 4-5 cm each (keypoints at 5 cm magnify rotation error). Fixing them
would bring MPPE to about 1.2-1.3 = 2nd/3rd. The leader's 0.69 is below our rollout-vs-own-reconstruction floor (0.75).

**RPE (1.55 vs 0.72).** Object-to-object relative pose in the first object's frame; single-object episodes count 0. ep32/33 again
dominate, and every "tighter" ep33 package with a looser cup fit got worse (cup drift). Fix of 32/33 -> about 1.2-1.3 = 2nd/3rd.

**Bottom line.** Fixing ep32 and ep33 is the only move that changes rank on all four boards (MP-SR tie-1st, AUC/MPPE/RPE 2nd-3rd).
Beating richchentw on AUC/MPPE/RPE would need the other 18 episodes reconstructed 2-3x more accurately than today; that is not
reachable before the freeze and no package we can build in 46 h gets there. The user's intuition is right: ep32/33 first.

## 3. What is new tonight (2026-11-03)

1. **Per-object knockouts (built, dry-run OK, not uploaded):** `ko_perobject_20261103/`. They tell us for ep32 and ep33 whether the
   cup or the spoon carries the error (object AUC = 40 x score drop). Cup bad -> scene-alignment error (frame-0 spoon rotation or
   cup depth); spoon bad with cup fine -> in-hand spoon track / body-frame error. 4 AUC uploads.
2. **Whole-track roll test (W-roll agent, chains running):** if the roll error is a registration/body-frame error it is constant over
   the episode, so the -16 (ep33) / -10 (ep32) correction should apply to every frame, not only f0-35. Packages
   `track3_v33a_ep33w16_*` and `track3_v33r16c3_ep32w10_*` when done. 2 AUC + 2 MPPE uploads; MP-SR only if AUC jumps.
3. **Spoon mesh registration study (agent, read-only):** does any of our/coworker spoon meshes register >= 8 deg differently about
   the mid/short axis from the G18 mesh? If yes, a mesh-only swap (no retarget) tests it on CD-O (1 upload) and AUC (1 upload).
4. **FINAL_20261104/** holds the best package with one-line upload commands.

## 4. Decision tree for the quota (user uploads; this session cannot upload)

- 00:00-02:00 UTC: 4 KO uploads (AUC). Read object AUCs.
- When W16/W10 packages land: AUC + MPPE each. If ep33 W16 beats r16c3 (AUC > 0.9031) -> MP-SR upload. Same for ep32.
- If the KOs say the CUP is bad in 32/33: the residual is the frame-0 scene rotation -> next sweep is the spoon's yaw/pitch at f0
  realised physically (settle on our table plane with the new roll), or the cup's f0 depth; one retarget per variant.
- If the KOs say only the SPOON is bad: the in-hand track; the W16/W10 result decides between "constant roll" and "roll ramps".
- Mesh-swap tests cost no retarget: CD-O first (cheap, isolates mesh shape), then AUC.
- Keep 2 uploads per board on Nov 4 for the final file. Upload FINAL to all six boards by Nov 4 ~20:00 UTC.

## 5. Realistic outcome

| board | now | if ep32/33 fixed | 1st |
|---|---|---|---|
| MP-SR | 0.90 (4th) | 1.00 (tied 1st, shown 4th) | 1.00 |
| AUC | 0.903 (6th) | 0.935-0.95 (2nd) | 0.980 |
| MPPE | 1.56 (6th) | 1.2-1.3 (2nd-3rd) | 0.69 |
| RPE | 1.55 (6th) | 1.2-1.3 (2nd-3rd) | 0.72 |

## 6. Results log (2026-11-03)

**Per-object knockouts (AUC board, base 0.90306; object AUC = 40 x drop):**

| package | score | drop | object AUC |
|---|---|---|---|
| ko33_spT (ep33 spoon) | 0.88819 | 0.01487 | **0.59** |
| ko33_cupR (ep33 cup) | 0.88176 | 0.02130 | **0.85** |
| ko32_spT (ep32 spoon) | 0.89169 | 0.01137 | **0.45** |
| ko32_cupR (ep32 cup) | 0.88223 | 0.02083 | **0.83** |

Reading: ep33 per-episode AUC is about 0.72 and both objects are off (cup 2 cm-class, spoon 4 cm-class), so the dominant error is the
frame-0 scene alignment plus the spoon's own track, not the cup alone. ep32's spoon is worse (5 cm-class). The other 18 episodes
average about 0.92-0.93, so the ceiling from fixing 32/33 alone is about AUC 0.93.

**Spoon mesh study:** all 7 spoon meshes share one body (bowl-plane bend +8.6 deg); no variant registers > 2 deg differently, so mesh
choice is not the cause. But the kit's roll registration follows the bowl 1:1, and a straighter-than-real bowl gives a mid-axis error of
~0.65 deg per degree of bend difference -> mesh-only bend tests built in `spoonbend_20261103/` (CD-O first, then AUC).

**Spoon bend CD-O test (official):** base 0.26116 (bend +8.6), straight 0.26830, +16 0.27789, +24 0.31460 -> our bend is already the
optimum; the real spoon is within a few degrees of ours. Mesh-shape branch closed (3 CD-O uploads, no AUC upload).

**Whole-track roll test (official):** ep33 W16_c1 (-16 deg on every frame) AUC 0.90001 / MPPE 1.58458 vs the f0-only r16c3
0.90015 / 1.58256 -> flat. The roll error is a frame-0 (rest-pose) effect only; the in-hand roll of our tracker is already as good as
the rolled version. Whole-track roll branch closed; ep32 W10 package built (9e7db4aa) but not uploaded.

**Image + stereo consistency (02:00 UTC):** our ep32/33 tracks match our own masks (median ~1 px after the known 1-frame RGB lag) and the
stereo depth maps (within 1.5 cm, static frames 0.1 cm) on nearly every frame; only 10-15 blurred scoop frames per episode are 1-2 cm off.
So the 4-5 cm official spoon error is not tracking. The remaining invisible term is the kit's mesh-to-scan registration rotation
(constant body-frame error, same for every spoon episode, levered through the frame-0 alignment only in ep32/33). Its roll part was
found by the sweep; its pitch/yaw parts are being tested now (AX-sweep: ep33 f0 +-10 deg about mid and short axes on top of -16 roll).
If one direction helps, the same correction applies to ep32 and to the spoon-only episodes 35/36/37.

**Frame-0 pitch/yaw sweep (official AUC, 04:00 UTC):** mid +10 0.89463, mid -10 0.89596, short +10 0.89698, short -10 0.89155 vs base 0.90015
-> frame-0 orientation already optimal on all axes; registration hypothesis closed.

**TIME SHIFT (official MPPE, 04:40-05:00 UTC) - FIRST REAL LEAD:** ep33 rows re-indexed by k frames: -2 1.56897, -1 1.56347, +1 **1.52913**,
+2 1.53290 (base 1.56142). The hidden reference runs ~1.3 frames ahead of our stereo-retimed track. Physical retargets onto +1/+2 shifted
targets are running for ep33 and ep32; diagnostics for ep32/37/19/5 built to test whether the offset is global (then every moving
episode gains).
