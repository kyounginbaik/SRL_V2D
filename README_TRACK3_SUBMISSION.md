# SRL — V2D Challenge Track 3 (Egocentric) code submission, 2026-11-05

Team SRL (Kyoungin Baik, Hansol Ryu, Jisu Shin, Hae-Gon Jeon, Youngwoon Lee).

## What is in this commit
- `track3_submission/` — the exact pipeline used for the final Track 3 file: reconstruction campaign scripts,
  per-episode retargeting streams (window-beam / CEM / distillation), the build-v2 evaluator and packaging
  (`challenge_reconstruction/evaluator_build_v2_20261023/package_code`), candidate manifests, and the
  submission kit tools. Unpacked from `SRL_V2D_track3_code_v1_20261105.tar.gz`.
- `compose/` — ComPose (object pose reconstruction from RGB with hand-motion cues) source as used for the
  reconstructions; model weights (`*.pth`) are shipped in the verification container, not in git.
- The rest of the tree is the `uniretarget/yam-bilevel` branch: UniRetarget (bi-level kinematics + physics
  retargeting) implementation under `robotic_grounding/flash_chord` (see `UNIRETARGETING.md`).

## Final submission files
| file | parquet sha256 (16) | MP-SR | SP-SR | AUC | MPPE (cm) | RPE (cm) |
|---|---|---|---|---|---|---|
| track3_FINAL_ep32T32cb1_ep33R6M2_20261105 | 8b6fa2b9a0513b8b | 1.000 | 1.000 | 0.92086 | 1.40441 | 1.51105 |
| track3_FINAL_ep32T32cb1_ep33R6M2_ep36s4_20261105 | 9bc322af8242e7ee | see Kaggle (submitted 21:03 UTC 2026-11-04) | | | | |

Poses in every submitted row come from recorded build-v2 physics rollouts of per-episode actors; mesh cells are the
team's own reconstructions (no public scan or ground-truth meshes were used anywhere in the pipeline).
