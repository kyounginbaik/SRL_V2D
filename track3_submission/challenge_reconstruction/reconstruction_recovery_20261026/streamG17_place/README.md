# Stream G17: non-spoon gross-error audit and joint frame-0 placement (2026-10-30)

Rules followed:
- Public GT was read only by the scorers (`score_bundle.py`) and by the scorer-side diagnostic `code/gtdiag.py`. Nothing was fitted to it.
- No scans or compose poses were used. Nothing was overwritten (new export tags `g17_*` and new folders only).
- No upload and no commit. Jobs ran on node01 (debug), each under 50 min.

## Deliverable
`reconstruction_recovery_20261026/selected20_final_v14_g17` + `campaign_v22_20261028/target_final_v14_g17`.

Provenance (per main's instruction, after G16 finished):
- The base is v12b_g15.
- **ep3**: object_poses.npz comes from selected20_final_v13_g16. This is the G16 constant spoon roll, -13.5 deg; the white_pot slot is identical to v12b.
- **ep32**: kept exactly as in v12b. The G16 -15 deg change is NOT inherited, because the official v24-v26 outcomes contradict it.
- **G17 changes: none adopted.** Meshes are byte-identical.

Deltas v12b -> v14_g17 (time-mean):
- ep3 wooden_spoon: 0.01 cm / 13.5 deg.
- Every other object in every episode: 0.00 / 0.0.
- Only the ep3 robot run needs redoing.

## Task 1: gross audit (code/audit.py, out/audit/epN.json, summary code/sum_audit.py)
Our v12b track was brought back to the RGB clock and compared with raw FoundationPose, multimap and CoTracker pointmap in the camera frame. Disagreement windows are > 2 cm / 15 deg and at least 3 frames long. The audit also checked half-turn twins on 16 frames and per-frame IoU / stereo-depth QC.

**No non-spoon object shows a GT-free-detectable error that could reach 3 cm / 30 deg time-mean.**
- **ep20.** Cup: IoU 0.86, depth 0.45 cm, raw agrees. Pitcher: IoU 0.77, raw agrees (0.2 cm); multimap and pointmap are 5-11 cm off and worse on image evidence. The only tracker-better window is pitcher f95-100 (multimap IoU 0.72 vs ours 0.61, depth bad for all; about 3 cm over 6 frames, time-mean about 0.1 cm). All twins are worse (0/16).
- **ep33.** Blue cup f31-42: raw is 5.3 cm / 31 deg away and has better IoU (0.74 -> 0.89) with equal depth. 12 of 348 frames, time-mean about 0.2 cm. Everything else favours ours.
- **ep3, ep8 and ep28 pot; ep14-16 rack.** Trackers agree (0.0-0.4 cm). The ep28 multimap and pointmap pot is 11 cm off, and the image favours ours.
- **ep1 pot / lid.** Trackers agree.
  - The pot shows a large depth residual (-6 cm median) only while the lid is on (f0-30, f288-334). The render does not include the lid occluding the pot interior. This is not a pose error.
  - The lid's low-IoU frames f54-76 are a hand phase, and no tracker does better there.
- **ep19 sweeper.** Twins are worse (14-15/16 frames favour ours). Multimap is better only at f60-62 (3 frames, and its depth is worse).
- **ep19 dust pan.** Clean.
- **ep30 sweeper.** The twin about the long axis has slightly better IoU on 13/16 frames (0.83 vs 0.78). This is not acted on: A9's ball votes were 171-0 for ours, and official SP passes since v24.
- **ep5 and ep6 pot, ep32 cup.** There are short 3-29 frame windows where raw has better IoU but its depth is worse. Mixed evidence.

## Task 2: joint multi-object frame-0 refinement (code/f0refine.py, apply_f0.py)
Method:
- Each object gets one world correction, fitted on its static start window.
- All objects are fitted jointly using silhouettes ("iou" or erosion-invariant "mom" moments), relative stereo depth on the stereo clock, and relative table-plane support.
- Corrections are applied either to the window with a blend (win) or to the whole track (all).
- 13 public multi-object clips were run.

| variant | tune MPPE / AUC (base 1.0931 / 0.9496) | held-out MPPE / AUC (base 1.0351 / 0.9379) |
|---|---|---|
| A mom+dep+sup | win 1.88/0.876 (9/10 MP); all 1.64/0.898 (9/10 MP) | 1.39/0.905 (9/10 MP); 1.18/0.925 |
| B iou+dep | 1.27/0.931 (9/10 MP); 1.24/0.933 | 1.16/0.928; 1.10/0.934 |
| C mom+sup | 1.88/0.870; 1.66/0.890 | 1.57/0.888; 1.41/0.900 (8/10 MP) |
| D iou+dep+sup | 1.32/0.929; 1.29/0.932 | 1.25/0.919; 1.12/0.932 |
| E/F (translation only) | 1.13-1.35, worse | 1.06-1.56, worse |

**Rejected: every variant is worse on both splits.**

Why: `code/gtdiag.py` (scorer-side only) maps the GT pose back onto OUR mesh at frames 0-40. In 12 of 13 clips, our own pose fits the silhouettes, depth and centroid better than the GT-implied pose does. Example, pub12 rack: IoU 0.74 vs 0.65, depth -0.3 vs -2.3 cm.

So the "f0 relative placement" error that G15 found is not a pose error visible in the images. It is a mesh-geometry and registration effect: the scans are 5-8 % larger for rack, dust pan, sweeper and pieces, and 6 % smaller for the pitcher. A smaller mesh fitted to the same image sits about 1-2 cm closer to the camera than the scan's centroid. Only GT-free correct mesh sizes would fix it. G12 and G15 found no reliable GT-free scale estimator; a public-fitted constant would be leakage.

Lid idea (circular disc, spin from the handle): not done.
- The public lid clips are both held-out, so a fix cannot be validated on both splits.
- G11's handle-only estimator has a constant -13 deg offset vs the scorer. Applying it raw on ep1 (+11.9 deg) would most likely move away from the truth.

## Extra: window-swap rule W (code/winswap.py)
The rule adopts a tracker on a disagreement window when its IoU is at least 0.10 higher and its depth is not worse by more than 0.3 cm.
- Public: tune MPPE 1.0931 -> 1.0902, AUC 0.9496 -> 0.9499.
- Held-out: MPPE 1.0351 -> 1.0315, but AUC 0.9379 -> 0.9362 and ori 6.18 -> 6.28 (pub43 pot ori 8.9 -> 11.1).
- It fires on only 3 clips. The result is mixed, so it was **not adopted**.
- On eval it would change only the ep33 cup at f31-42.

## Files
- code/: g17lib.py, audit.py, sum_audit.py, depcheck.py, depwin.py, f0refine.py, apply_f0.py, mk_pub.py, gtdiag.py (scorer-side), winswap.py, mk_w.sh, cmp17.py, cmd.sbatch, py.sh, jobs_*.txt
- out/: audit/epN.json, ref/<A-F>/public_N.json, gtdiag_g15_camrt_N.json
- Public exports: canonical/<split>/episode_N/exports/g17_{A..F}_{win,all}, g17_W10(_pre) and g17_W05(_pre). All are rejected.
