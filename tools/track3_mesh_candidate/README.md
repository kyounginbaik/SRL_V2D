# Track 3 object mesh candidate

The public-video SAM3D run produces one mesh for each of the ten distinct
objects in the public Track 3 split. `objects.json` records the chosen frames
and mask sources. `score.py` evaluates all 33 public episode-object pairs with
the submission kit's CD-O function. Public scanned meshes are read only by the
scorer. That video-only baseline scored 0.625 cm locally.

Further **video-only public** experiments use `run_public_candidates.py` to
sample SAM3D seeds and frames, and `prepare_public_mvsam3d.py` plus
`run_public_mvsam3d.sbatch` for MV-SAM3D. The official scorer in
`score_public_candidates.py` evaluates each candidate over all public
episode-object pairs and writes `candidate_summary.json`. Reference meshes
are opened only by that scorer. Its minimum is selected on the same public
data, so this is an in-sample result, not a hidden leaderboard measurement.
Before adding MV-SAM3D candidates, the selected mean was **0.3544 cm** over
33 pairs, down from **0.6248 cm**. Correcting two frame-0 wooden-piece masks
was the largest change: the corresponding object means fell from 2.15 to
0.184 cm and from 0.764 to 0.160 cm. The five-view white-pot MV-SAM3D
candidate scored 0.2782 cm over its nine pairs, versus 0.2843 cm for the
best single-view mesh. Five-view water pitcher at seed 42 scored 1.0940 cm,
versus 0.7604 cm for the best single-view mesh. These mixed results show why
multiview output is scored before it is selected.

The subsequent selected video-only set scored **0.31997 cm** over 33 public
episode-object pairs. `materialize_public_best.py` copies its ten GLBs and a
selection manifest to `selected_video_only/`. The largest remaining errors
were water pitcher (0.6550 cm) and plastic dish rack (0.5363 cm). A five-view
MV-SAM3D dust-pan mesh improved that object from 0.4940 to 0.2125 cm; a
frame-120 water-pitcher SAM3D candidate improved it from 0.7604 to 0.6550 cm.
For the dish rack, a SAM2 video mask propagated from frame 0 enabled both
later-frame single-view and five-view tests, but all scored worse than the
original frame-0 candidate. `run_dish_rack_tracking.sbatch` reproduces those
masks, and `run_dish_rack_frame120.sbatch` makes the later-frame candidates.

The MV-SAM3D implementation is the authors' repository at
https://github.com/devinli123/MV-SAM3D. Its input structure uses public video
frames in `images/` and tracked masks encoded in RGBA alpha under each
object's folder. `prepare_public_mvsam3d.py` records the episode, frame list,
and mask slot in `input_manifest.json`. The default tests use the existing
SAM3D Docker image and MV-SAM3D's weighted inference without
Depth Anything 3 pointmaps. An attempted direct two-frame mesh alignment and
vertex blend worsened the nine-pair white-pot mean from 0.2843 cm to 0.2923 cm even
at only 10% contribution from the second view; the available object poses
did not align the two SAM3D shapes accurately enough for direct fusion.
`prepare_public_moge_pointmaps.py` also converts saved MoGe depth predictions
to MV-SAM3D-compatible pointmaps for an optional depth-conditioned comparison;
the file is MoGe-derived even though the MV-SAM3D CLI calls its input
`--da3_output`. In the completed comparison, the depth-conditioned dust-pan
mesh scored 0.2276 cm (versus 0.2125 cm without these pointmaps), and the
water-pitcher mesh scored 0.7176 cm (versus the selected single-view 0.6550
cm). The final selected public mean remained **0.31997 cm**.

The separate leaderboard candidate uses a different, explicit method:

- For ten object types present in the released public asset set, use the
  corresponding released visual mesh as an object-identity lookup.
- For `wooden_spoon`, which has no released scan, generate candidates from
  evaluation video masks with SAM3D. `compare_spoons.py` selects a medoid by
  cross-view shape agreement. `fuse_spoon_candidates.py` makes a three-view
  consensus, which has the best silhouette agreement with 23 reliably tracked
  frames from episode 37 (0.657 mean IoU versus 0.647 for the medoid). This is
  a video-only selection proxy, not a hidden CD-O measurement.

`build_asset_candidate.py` creates one mesh for each of the 36 evaluation
episode-object pairs and records every source in `manifest.json`. The candidate
is packed with the official `tools/pack_reconstruction.py track3-meshes`
command in `v2d_submission_kit`. The generated Parquet has 294,912 rows and
passes the kit's local `validate_submission` check. The held-out leaderboard
score remains unknown until submission. Do not describe released assets as
meshes reconstructed from video.

Run the GPU scripts on a node with working CUDA Docker images. The Slurm files
default to `/data1/SRL_V2D/tools/track3_mesh_candidate` for code and accept
`V2D_MESH_CODE_DIR` to point to a separate checkout. Artifacts default to
`/data1/SRL_V2D/challenge_reconstruction` and can be redirected with the
`V2D_CHALLENGE_RECON_ROOT`, `V2D_MESH_BENCHMARK_ROOT`,
`V2D_SPOON_CANDIDATE_ROOT`, and `V2D_MESH_CANDIDATE_ROOT` environment variables.
