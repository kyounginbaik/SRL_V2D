# MP-SR evaluation for the DexAgent trial

The primary outcome is `reference_proxy_mp_sr`: a **local estimate**, using the immutable
`final_recon` object poses in place of confidential ground truth. It is not a Kaggle score.
Per frame, measure each object's centre translation error and symmetry-aware orientation
error. Average each error over the **whole original episode**, then declare success only if
**every object** has mean translation error **strictly less than 0.03 m** and mean orientation
error **strictly less than 30 degrees**. MP-SR averages the episode successes, weighting each
episode equally. A large brief excursion can still pass this mean-error criterion; this is
why per-object maximum errors are also reported as diagnostics.

Verified primary source: [official submission kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip),
downloaded 2026-10-17, SHA256 `8d2bce64ecab6da4eee938fe2859bd60e767535af30068b386e2dc20a72cf43e`.
Definitions are in `tracks/track_3/README.md` and `v2dlb/track3_pose.py`, particularly
`_symmetry_orientation_errors`, `_align_registered_scene`, and `_registered_pose_metric`.
The provided Kaggle overview could not be fetched by the web tool, so the official linked
kit was checked directly. Unit tests compare random-data orientation and scene-alignment
results with those exact downloaded functions.

## Geometry and coordinate frames

Official declared equivalences, applied independently at each frame:

| Object | Allowed orientation equivalence |
|---|---|
| `white_pot_lid` | Identity or 180 degrees about the lid normal |
| `beige_cup`, `blue_cup` | Any spin about the central axis; tilt and upside-down errors still count |
| `white_pot`, `wooden_spoon`, other objects | Full orientation |

The confidential scan anchors and axes are unavailable. This trial uses the exact local
mesh's surface centroid as the position anchor and records its source path and SHA256.
The lid normal is the minimum-variance surface direction; the tall cup axis is the
maximum-variance surface direction, from 50,000 deterministic surface samples. Axes are
explicitly labelled **estimated proxy geometry**, never official scan geometry. Ambiguous
axes fail with an error and require a reviewed geometry JSON. A geometry JSON can be
generated once and frozen with `--prepare-geometry`; it must preserve the official
symmetry classes and provide a unit local axis where needed.

Rollout translations and quaternions must refer to the original mesh body frame, not the
inertial COM or a recentered mesh frame. The same body coordinates on both sides make mesh
registration identity. If the simulator recentres a body, its writer must restore the
source mesh frame before exporting. The adapter translates the whole scene to a setup
frame; `--manifest` reads the explicitly recorded `source_from_setup` rigid transform.

For official-style reporting, one further rigid transform from the initial scene is applied
to all objects and all times together, using exactly the official alignment rule. Its full
matrix is reported. `declared_frame_mp_sr` omits this fitted transform and exposes any initial
state error. There is no per-object trajectory fitting, similarity scaling, time shift,
dynamic time warping, or truncation to a successful prefix.

## Recording contract

Write a numeric/non-pickle NPZ containing:

| Key | Shape | Meaning |
|---|---|---|
| `timestamps_seconds` | `(T,)` | Absolute episode times, starting at zero |
| `object_names` | `(O,)` | Unicode object names, every source object exactly once |
| `object_positions` | `(T,O,3)` | Source mesh body-origin positions in metres in the adapter setup frame |
| `object_quaternions_wxyz` | `(T,O,4)` | Unit scalar-first mesh-body rotations |

An optional world axis `(T,W,O,...)` supports multiple rollouts; MP-SR then averages the
all-object pass indicator over worlds. No reset may occur mid-recording. Record initial
state at `t=0` and continue through `(reference_frames-1)/reference_fps`. A failed task must
still produce the whole physical rollout if it can continue; a crash or early termination
is an invalid/incomplete evaluation, never a successful partial episode.

The evaluator samples simulation poses at the unmodified source timestamps with linear
position interpolation and quaternion SLERP. Extrapolation is prohibited. Missing temporal
gaps larger than 1.5 source frame intervals and any missing object fail validation. Endpoint
tolerance is only 1 microsecond for floating-point representation. Non-finite values and
non-unit quaternions fail validation.

```bash
python code/evaluate_mp_sr.py \
  --reference-dir /data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000001 \
  --trajectory episodes/episode_000001/rollout.npz \
  --manifest episodes/episode_000001/input_manifest.json \
  --scene episodes/episode_000001/scene.xml \
  --output episodes/episode_000001/mp_sr.json
```

Python API: `evaluate_files(reference_dir, trajectory_path, manifest_path=None,
scene_path=None, geometry_path=None) -> dict`. `scene_path` records the scene digest as
provenance; it does not replace robot/physics validation. MP-SR alone cannot detect object
teleportation, actuator violations, kinematic replay, or invalid contacts. Those checks are
separate requirements of the DexAgent physics trial.

CPU verification: `python -m pytest -q code/tests/test_evaluate_mp_sr.py`.
