# MV-SAM3D object meshes in the HaWoR/HaMeR ego pipeline

Run these commands from `reconstruction/`. The consolidated entrypoint uses
MV-SAM3D for generated object meshes by default with HaWoR or HaMeR hand
tracking. `--mesh_generator sam3d` keeps the original single-view path.
`--object_mesh /path/to/mesh.obj` bypasses both generators and preserves the
existing optional mesh-scale estimation behavior. The DynHaMR legacy branch
still uses its original SAM3D pipeline.

## Code and weights

This integration targets [devinli123/MV-SAM3D](https://github.com/devinli123/MV-SAM3D)
commit `abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd`, running
`run_inference_weighted.py` with its stage-1/stage-2 entropy-weighting defaults,
no Depth Anything 3, and `--model_tag hf`. It reuses the SAM 3D Objects
`facebook/sam-3d-objects` checkpoint downloaded by the existing helper. The
tested local checkpoint snapshot was
`2e73555018d2741ccd486e56c24fac41155a1dc6`; its `checkpoints/pipeline.yaml`
SHA-256 was `53c3d226b21df85c0bb3d16e6e4fa63abde0d6167525765eb929d02bfa9d358c`.
The run manifest records the actual checkout, checkpoint revision when present,
configuration hash, Docker image, seed, selected frames, mask sources, quality
checks, and fallback reason.
The [integration validation record](mv_sam3d_ego_mesh_validation.md) separates
CPU, GPU generation, and later-stage loader evidence.

```bash
./scripts/install_ego_reconstruction_packages.sh
python -m v2d.sam3d.docker.build
docker build -f modules/v2d_pipelines/Dockerfile.mvsam3d \
  -t v2d_mvsam3d:latest modules/v2d_pipelines
python -m v2d.sam3d.docker.run_download_weights --output_dir data/weights/sam3d
mkdir -p data/vendor
git clone https://github.com/devinli123/MV-SAM3D.git data/vendor/MV-SAM3D
git -C data/vendor/MV-SAM3D checkout abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd
```

SAM3D checkpoint access requires a Hugging Face token. The existing setup
guide covers the other pipeline packages, images, and weights. The MV checkout
and all model data stay below ignored `reconstruction/data/`; do not commit
checkpoints, videos, generated GLBs/OBJs, or run outputs. The default
`v2d_sam3d:latest` base image was validated on L40S. The
`v2d_mvsam3d:latest` layer adds the Open3D system libraries needed by MV-SAM3D.
A B200 image needs PyTorch,
PyTorch3D, and Kaolin rebuilt for that architecture; the image can be selected
with `--mvsam3d_docker_image` after those dependencies are verified.

## Run

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video data/clip.mp4 --object_prompt "a pot" \
  --output_dir data/outputs/clip --hand_tracking hawor \
  --mesh_generator mvsam3d --seed 42
```

The runner takes matching RGB, metric depth, and the SAM2 mask for the same
object-track ID. It checks image/mask/depth dimensions, track ID, prompt,
reference frame, and reference mask overlap with the Grounding DINO box,
object-mask connectedness, mask border, blur proxies, hand proximity, and
depth validity. At most 48 evenly spaced candidate frames are inspected.
The quality-passing frames are selected uniformly by temporal rank: the
reference frame is retained when it passes, with up to three helpers. Two or
three valid views are passed through as two or three views; exactly one valid
view falls back to the existing SAM3D generator. Zero valid views fail with
an explicit error. The MV input RGBA alpha is verified pixel-for-pixel against
the SAM2 object mask and the RGBA RGB is verified against its source image.
The repository `tools/e2e reconstruct` command also defaults to MV-SAM3D for
HaWoR/HaMeR and accepts `--mesh-generator sam3d` for the legacy choice.
The pot experiment's optional cross-view pose-support check cannot run before
FoundationPose has produced poses; this selector uses its image/mask/depth
quality gates and temporal-rank rule only.
The SAM2 track ID and reference detector overlap cannot prove that the track
never switches to a visually similar object later. Inspect selected view
masks in the manifest when the scene contains similar objects.

The raw MV-SAM3D `raw_mv_sam3d.glb` and unaligned OBJ copy are saved under
`<output_dir>/mesh_mv/`. The OBJ copy is passed into the existing
MoGe/FoundationPose scale estimator, then its scaled output goes to the
FoundationPose tracker and downstream pipeline. The public GT shape is never
used in selection, generation, or scale estimation. Evaluation-only GT-aligned
preview meshes must never be used as the pipeline mesh.

`<output_dir>/mesh_mv/mesh_generation_manifest.json` records the selection
and generation. The explicit SAM3D path records
`<output_dir>/mesh/mesh_generation_manifest.json`; a supplied mesh records
`<output_dir>/mesh_generation_manifest.json` with an override reason.
Changing generator or inputs should use a fresh output directory so cached
pose/refinement products cannot be mixed across mesh sources.

```bash
# Explicit legacy generator
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video data/clip.mp4 --object_prompt "a pot" \
  --output_dir data/outputs/clip_sam3d --mesh_generator sam3d

# Supplied mesh: generation skipped; add --skip_object_scale_estimation to trust its scale
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video data/clip.mp4 --object_prompt "a pot" \
  --output_dir data/outputs/clip_mesh --object_mesh data/assets/pot.obj
```
