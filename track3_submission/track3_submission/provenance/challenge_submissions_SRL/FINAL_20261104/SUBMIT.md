# FINAL_20261104 - submission-ready folder (Track 3, team SRL)

`all_20_episodes/` is a verbatim copy of the current best package
`anchor_20261103/ep16_anc1/all_20_episodes` (parquet sha256 0fca055b...), dry-run OK on 2026-11-03 00:10 UTC.

Contents: v33a round-1 tightening (12 episodes) + ep35 start roll -11 + ep33 start roll -16 (K33_H5r16_c3)
+ ep32 start roll -10 (K32_R10_c1) + ep16 rack de-tilt 6.6 deg, coworker GT-free meshes (CD-O 0.261), G18 spoon.
Every row is a build-v2 physics rollout; no edited poses; no public/GT meshes.

Official scores of this exact file: AUC 0.90306, MPPE 1.56142, RPE 1.61939 (uploaded 2026-11-02 22:3x UTC).
Not yet uploaded to MP-SR, SP-SR and CD-O (expected MP-SR 0.90, SP-SR 1.00, CD-O 0.26116 - all already on the board from v33a).
Kaggle keeps the best score per board, so uploading it to those three boards cannot lower anything.

## Upload all six boards (one line, paste as is)

cd /data1/SRL_V2D/challenge_submissions_SRL/FINAL_20261104/all_20_episodes/submission_kit && /data1/SRL_V2D/challenge_submissions_SRL/.venv/bin/python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m "SRL final 20261104 (ep16_anc1 0fca055b)" --wait

## If a better candidate lands before the freeze (Nov 4 22:00 UTC)

Replace `all_20_episodes/` with the winning package (rsync, excluding *.kaggle-receipt.json), re-run the dry run below, then upload:

cd /data1/SRL_V2D/challenge_submissions_SRL/FINAL_20261104/all_20_episodes/submission_kit && /data1/SRL_V2D/challenge_submissions_SRL/.venv/bin/python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m dry --dry-run

## RPE note
The best RPE on the board (1.54679) came from v33a + ep33 b2 (start roll -10, `track3_v33c_ep33b3`-era package
`ablation_v33a_ep3233_20261102` b2, sha aa8fde7c). This file scores RPE 1.619. Kaggle keeps the best per board, so nothing to do.
