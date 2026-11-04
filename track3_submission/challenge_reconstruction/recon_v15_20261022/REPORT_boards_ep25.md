# recon_v15 — wooden boards (evaluation episode 25), 2026-10-22

`episode_000025/object_poses.npz` (same format as the Compose source: world_from_object (356,2,4,4), world_from_rgb, frames,
names [wooden_piece_1, wooden_piece_2]; Compose table-aligned world; body frames of the SUBMITTED Compose meshes
`dexagent_remaining7_20261019/source/final_recon_ours/episode_000025/assets/wooden_piece_X_visual.obj`).
Variant `object_poses_compose_camera_variant.npz` uses Compose's own camera trajectory instead of the PnP camera (0.8 cm / 0.5 deg apart).
`change_report.json` has the numbers below plus per-second changes. NOT retargeted.

Recipe (validated on public GT, see public_gt_check_20261020/boards_results.md: ep23 1.95/3.8 + 2.23/4.2, ep24 2.04/2.4 + 2.04/3.2, both MP-SR pass):
per-board SAM2 masks from table-constrained Grounding DINO boxes -> stereo FoundationPose per board with its own true-size mesh ->
identity by mocap-ball count (4 balls = wooden_piece_1; ep25: left board 4, right board 3) -> SIFT + stereo PnP camera ego-motion ->
half-turn continuity -> constant body twin matched to the current Compose track (for robot/scene continuity; absorbed by the scorer for the anchor).

Change vs the current Compose track (mean / max):
| object | position cm | rotation deg |
|---|---|---|
| wooden_piece_1 | 1.83 / 4.47 | 8.9 / 35.6 (5-7 s, while the board is flipped and stood up) |
| wooden_piece_2 | 2.00 / 4.15 | 5.7 / 17.9 |

Conclusion: the Compose ep25 board tracks already agree with the GT-validated recipe to ~2 cm / 6-9 deg (same identity, same camera
within 0.8 cm). The recon is not what fails ep25 locally (that failure is the robot not reproducing the assembly); re-retargeting ep25
on these tracks is optional and would change the reference by ~2 cm.
