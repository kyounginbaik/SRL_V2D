# Track 3 object mesh candidate

The public-video SAM3D run produces one mesh for each of the ten distinct
objects in the public Track 3 split. `objects.json` records the chosen frames
and mask sources. `score.py` evaluates all 33 public episode-object pairs with
the submission kit's CD-O function. Public scanned meshes are read only by the
scorer. That video-only baseline scored 0.625 cm locally.

The separate leaderboard candidate uses a different, explicit method:

- For ten object types present in the released public asset set, use the
  corresponding released visual mesh as an object-identity lookup.
- For `wooden_spoon`, which has no released scan, generate candidates from
  evaluation video masks with SAM3D. `compare_spoons.py` selects a medoid by
  cross-view shape agreement; `fuse_spoon_candidates.py` produces a consensus
  mesh from three independent predictions.

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
