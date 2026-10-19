# Finish the episode 37 replacement

The thirteen physical evaluations are complete under evaluator commit
`6f81ed0e77835f7da40b6a4f7ef34f8a33b339c7`. The twelve legacy episodes reproduce
their earlier state/action recordings exactly. Episode 37 uses the requested
`policy_2508800.safetensors` and `reference_0005.npz`, with world 28 selected before
evaluation. Its fresh rollout passes full-frame reference-proxy MP-SR.

The fresh episode 37 body states differ numerically from the historical video.
The original video is retained as historical evidence; a new rendering of the
actual exported recording is the matching submission preview. Rendering and
packaging are separate postprocessing and do not change the evaluator commit,
checkpoint, refined reference, physical rollout, or evaluator hashes.

1. Run `../render_tools/render_fresh.py` through its GPU rendering launcher. It
   reads recorded body states; it does not train or simulate a replacement rollout.
   Outputs are `../rendered_fresh/epi37_fresh.mp4` and its `.video_evidence.json`.
2. Use the existing CPU environment to finalize:

```bash
EP37_BASE=/data1/SRL_V2D/challenge_reconstruction/episode37_ilchold_submission_20261019
EP37_PYTHON=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
EP37_KIT=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/submission_readiness/latest_official/v2d_submission_kit
"$EP37_PYTHON" "$EP37_BASE/postprocess/finalize_fresh_release.py" --kit "$EP37_KIT"
```

Add `--postprocess-commit-url https://github.com/kyounginbaik/SRL_V2D/commit/<sha>`
to record the separate rendering/packaging revision. The script checks the sealed
common build, all thirteen completed recordings, unchanged source inventory,
video/action/body-state hashes, 151 rendered frames, and the public mesh/row
contracts before producing the new package. Existing releases are not overwritten.

Outputs include `portable_staging_v2/`, `episode_000037_portable_v2.tar.gz`, and
`/data1/SRL_V2D/challenge_submissions/track3_mixed_ilchold_20261019/available_13_episodes`
with its transport `.tar`. The portable README includes the pinned Docker image,
dependency and asset mounts, writable cache locations, and reproduction command.
The image itself must be transferred separately; a local image digest is not a
downloadable registry address. A second extracted-archive GPU replay on another
server has not been claimed.

Episode 37 actions are **150 actual 56-dimensional residual policy actions at
20 Hz**, paired causally with 151 states; internal physics runs at 400 Hz. Saved
processed targets accompany the raw actions. The other twelve episodes retain
their legacy 500 Hz position-target action convention. Do not treat the two
action arrays as interchangeable.

The exact older kit at `/data1/video_to_data_challenge/v2d_submission_kit` uses a
pose CSV and a separate CD-O mesh Parquet. The frozen latest kit used here packs
poses and meshes together into one Parquet. Both require the same twenty episodes
and identical 9,120 public pose rows. Their per-episode trajectory reader is
identical, but the new common validation requires the latest kit's combined
template and mesh helpers. Do not switch the sealed run to the older kit midway.

This package covers the user's thirteen requested episodes. The other seven are
still required for a complete official upload. Proxy scores use reconstructed
references, not hidden ground truth. No submission is uploaded by this workflow.
