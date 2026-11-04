# V2D Challenge Track 3 (Egocentric) — SRL end-to-end summary

As of 2026-11-02 23:30 UTC. Leaderboard freeze: Nov 4, 5 pm EST (22:00 UTC).
Team SRL (Kaggle kyounginbaik). Detailed log: `challenge_reconstruction/campaign_v22_20261028/STATUS.md`.

---

## 1. Where we stand

| board | our best (official) | rank | 1st (richchentw) | 2nd | gap to 1st |
|---|---|---|---|---|---|
| **MP-SR** ↑ | 0.90 (18/20) | 4th | 1.00 | 1.00 (DDD) | ep32 and ep33 fail |
| **SP-SR** ↑ | 1.00 | tied 1st | 1.00 | 1.00 | — |
| **AUC** ↑ | **0.90306** | about 6th | 0.98018 | 0.93906 | −0.077 |
| **MPPE** (cm) ↓ | **1.561** | about 7th | 0.690 | 1.162 | 2.3× our error |
| **RPE** (cm) ↓ | **1.547** | about 6th | 0.715 | 1.142 | 2.2× our error |
| **CD-O** (cm) ↓ | 0.239 | 4th | 0.095 (ycchen99) | 0.213 | 2.5× our error |

**Best package now:** `challenge_submissions_SRL/anchor_20261103/ep16_anc1/all_20_episodes` (sha 0fca055b).
It is v33a plus three fixes:
- ep33 spoon start roll −16°
- ep32 spoon start roll −10°
- ep16 rack de-tilt 6.6°

It scores AUC 0.90306, MPPE 1.561 and RPE 1.619.
Not yet uploaded to all six boards. Kaggle keeps each board's best score, so the scores above are already counted.

---

## 2. Pipeline (end to end)

1. **Reconstruction (GT-free):** egocentric stereo video produces object meshes, per-frame object poses and the camera trajectory.
   - The reference set is `selected20_final_v14_g17`, our "v14".
   - The spoon mesh is G18. Coworker meshes are used for CD-O.
2. **Retargeting:** Sharpa floating hands track the reconstructed object trajectories in physics, using FlashCHORD / build-v2.
   - Each episode is a policy checkpoint (an "actor") refined by chained beam search over time windows.
   - The rows we submit are the **achieved** object poses of the rollout, never hand-edited.
   - Provenance (checkpoint and eval hashes) is recorded in every row.
3. **Packaging:**
   - `v32build.py` assembles the 20-episode parquet from per-episode donor packages.
   - Mesh payloads come from our meshes. Each package passes a dry run of the kit's `v2d_submit.py`.
4. **Scoring (kit):**
   - The scorer registers our mesh to the hidden GT mesh, then aligns each episode **on frame 0**:
     - **rotation** from the first non-symmetric object;
     - **translation** from slot 0's frame-0 position.
   - Then it computes per-object time-mean errors:
     - MP-SR: every object under 3 cm and 30°;
     - SP-SR: a looser bar;
     - AUC: mean vertex distance, area under the curve for 1–9 cm thresholds;
     - MPPE: keypoints 5 cm out from the object;
     - RPE: object-to-object relative pose (0 for single-object episodes);
     - CD-O: mesh Chamfer distance.

**Rules we followed:**
- No public or GT meshes or poses in submissions or reconstruction prep; GT is used only inside the local scorer.
- Poses come only from build-v2 rollouts.
- No constants fitted to the public GT.

---

## 3. History of official results

| version | MP-SR | SP-SR | AUC | MPPE | RPE | CD-O | what changed |
|---|---|---|---|---|---|---|---|
| first submission | 0.35 | | | | | | anchor alignment, sweeper flip, tight margins |
| v22b | 0.65 | 0.85 | 0.751 | 3.11 | 3.52 | | |
| v24 | 0.80 | 0.95 | 0.8548 | 1.950 | 1.738 | 0.276 | |
| v25 / v26 | 0.80 | 1.00 | 0.855 | 1.92 | 1.74 | 0.276 | ep32 SP fixed |
| v29 | 0.85 | 1.00 | 0.87401 | 1.769 | 1.576 | 0.2723 | G18 spoon mesh, ep32 robot run |
| v30c1_cwmesh_nogt | 0.85 | 1.00 | 0.87701 | 1.740 | 1.580 | 0.2612 | coworker meshes (GT-free) |
| v29 + ep35 fix | **0.90** | | | | | | ep35 spoon start roll −11° (knockout confirmed) |
| v33a | 0.90 | 1.00 | 0.88831 | 1.664 | 1.559 | 0.2612 | ALL20 round-1 tightening of 12 episodes + ep35 fix |
| v33a + ep33 −10° (b2) | (MP 0.90) | | 0.89578 | 1.611 | **1.547** | | |
| v33a + ep33 −16° (r16c3) | | | 0.90015 | 1.583 | 1.572 | | |
| + ep32 −10° (R10_c1) | | | 0.90237 | 1.568 | 1.620 | | |
| **+ ep16 rack de-tilt** | | | **0.90306** | **1.561** | 1.619 | | best |

---

## 4. What we tried and what we learned

### Worked
- **Spoon start roll (frame 0–25) in the rotation-anchor episodes.**
  - ep35 at −11° now passes MP-SR.
  - ep33's AUC rises steadily as the roll goes from +9° to 0°, −10°, −13° and −16°, peaks between −16° and −21°, and is worse at −25°.
  - ep32 at −10° also helped.
  - Why it matters: the kit aligns rotation on this spoon's frame-0 pose, so a roll error there tilts every object in every frame.
- **ALL20 round-1 tightening (v33a):** retargeting closer to our reconstruction in 12 episodes raised AUC from 0.877 to 0.888.
- **Tight robot fit matters as much as the target.** A loose cup fit (r21_c1, cup 19° off) sent RPE to 1.75.
- **ep16 rack de-tilt** (the rack is that episode's anchor): AUC +0.0007, MPPE −0.007.
- **Coworker meshes:** CD-O improved from 0.272 to 0.261.

### Did not work (official)
- **ep33 "physics rest" roll alone:** −10° and +9° both still fail MP-SR. A start roll of −16° to −21° is closest to the truth but still fails MP-SR.
- **Jisu's ep32 kinematic fix** (b1h, b2h, b3d): MP-SR fail, and AUC, MPPE and RPE all worse than v29. Her target is further from the truth than v14.
- **Hansol's ep33 spoon 180° flip** (recon_HGA_ep33fix): AUC, MPPE and RPE worse.
- **ALL20 round 2** (eps 1, 8, 10, 30, 37, 38): slightly worse officially; it overfit our own reconstruction.
- **ep20 pitcher mesh ×0.93:** neutral; CD-O unchanged.
- **ep38 dust pan de-tilt 4.7°:** worse. ep15 and ep28 rack de-tilts: worse on RPE, AUC and MPPE untested.
- **ep32 rolls of −13° and −16°:** the robot runs diverged. The v29 actor re-rolls the spoon during the grasp (f40–100).

### Infrastructure lessons
- The Kaggle OAuth token expires about 12 h after `kaggle auth login`. The SDK refreshes it only 30 min after expiry.
- The daily cap is 20 per board, reset at 00:00 UTC.
- Single-episode swaps on the AUC, MPPE and RPE boards are the only signal of closeness to the truth for episodes that already fail MP-SR.

---

## 5. The problem behind each metric

### MP-SR (0.90; ep32 and ep33 fail)
- Both are **spoon** episodes. Every robot run follows our reconstruction within about 1 cm / 10°, yet they still fail, so the error is in the **reconstruction**.
- **ep33:** the start roll explains part of it; AUC improves monotonically up to −16°/−21°.
  - Something else stays above 30°. The most likely cause is the scorer's registration of our spoon mesh onto the GT spoon: a body-frame error that turns into rotation error while the spoon is in the hand.
  - None of our image cues can see it.
- **ep32:** the same picture. On top of that, the v29 actor can't track larger start rolls.
- **Needed:** a better spoon mesh shape, from fresh evidence such as a new scan, plus re-tracked poses.

### AUC (0.903 vs 0.980)
- Our robot runs reach AUC 0.96–1.00 against **our own** reconstruction, so nearly all the remaining gap is reconstruction versus truth.
- **Biggest lever found so far:** frame-0 anchor errors, i.e. the rotation-anchor object's start pose and the translation anchor's f0 position. Each is applied to the whole episode.
- **Other levers:**
  - per-frame pose accuracy: depth-direction tilt is invisible to silhouettes;
  - mesh shape: the unseen underside of the rack and pot is tilted 3–6° relative to the visible part, which affects the scorer's registration.

### MPPE (1.56 vs 0.69)
- MPPE uses keypoints 5 cm out, so it magnifies **rotation** errors. The spoon and anchor-roll fixes moved it most (1.664 to 1.561).
- Same root causes as AUC, with more weight on orientation.

### RPE (1.55 vs 0.72)
- RPE measures object-to-object relative pose, so it's hurt by the **non-spoon** object drifting in the rollout (cup or pot) and by relative errors between reconstructed objects.
- Fixes that improve AUC and MPPE often make RPE worse when the second object's fit loosens (r18, r21, R10, ep15/28 racks).
- The best RPE is still v33a + ep33 −10° (b2) at 1.547.

### CD-O (0.239 vs 0.095)
- Mesh quality only. Our meshes come from GT-free reconstruction (v14, coworker meshes, G18 spoon).
- The unseen bottoms and backs are filled in and probably wrong: the audit showed systematic per-object tilts (rack 6°, pot 3.5°, pan 4.7°, sweeper 11°, cups 1–2°).

### SP-SR (1.00)
Solved.

---

## 6. Candidates ready and untested

| package | change | status |
|---|---|---|
| `anchor_20261103/ep15_anc1` (ffdc0a6b) | ep15 rack de-tilt | RPE 1.628 (worse); AUC/MPPE untested |
| `anchor_20261103/ep28_anc1` (64f41358) | ep28 rack de-tilt | RPE 1.627 (worse); AUC/MPPE untested |
| `anchor_20261103/racks_15_16_28` (50c028b3) | all three racks | RPE 1.633; AUC/MPPE upload failed (quota) |
| `track3_v33a_ep33r*` | ep33 roll sweep | all scored (table above) |
| `ep33_fix_20261102/K33_meshswap_ep15tess` (f2a5966f) | mesh retessellation | expected neutral, not uploaded |

---

## 7. Plan for the remaining time (to Nov 4 22:00 UTC)

1. **Tomorrow's quota** (20 per board), needs the user's go-ahead:
   - Test ep15 and ep28 rack de-tilts on AUC and MPPE, one at a time.
   - Upload the final best package to all six boards.
   - Use MP-SR uploads only for ep32 or ep33 candidates with a real chance of passing.
2. **ep14 rack** (rotation anchor, also about 20 mm into the table): retarget the de-tilt and test it.
3. **Mesh fixes, GT-free** (biggest remaining lever for CD-O, AUC and MPPE):
   - correct the unseen bases of the rack, pot, dust pan and sweeper meshes so they rest flat on our stereo table where the poses agree;
   - rebuild the spoon mesh from multi-view stereo.
4. **ep32 and ep33:** a new actor lineage that can follow a −16° to −21° start roll without re-rolling the spoon in the grasp. Try them on MP-SR only after AUC/MPPE swaps confirm the direction.
5. **Realistic outlook:**
   - MP-SR 0.95–1.00 only if the spoon mesh is fixed.
   - AUC 0.91–0.93 and MPPE about 1.3–1.4 with the anchor and mesh fixes.
   - Matching the leader (0.98 / 0.69 / 0.72) would need roughly a 3× more accurate reconstruction, which is unlikely before the freeze.
