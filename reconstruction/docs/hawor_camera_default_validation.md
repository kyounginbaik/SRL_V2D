# HaWoR camera-default validation

This record separates checks of existing `episode_000000` (e0) and
`episode_000002` (e2) artifacts from checks of the new execution path. The
existing artifact checks did not rerun either full video.

## Existing e0/e2 artifacts (read only)

Inputs were the existing local benchmark folders under
`/home/hansol/v2d/outputs/hand_benchmark/` and the corresponding baseline
reconstruction folders. OpenCV reported 20 fps and 373 source frames for e0,
20 fps and 343 source frames for e2. HaWoR metadata reported 560 and 515
native 30 fps frames, respectively.

| Episode | Right raw/aligned JSONs | Left raw/aligned JSONs | Right missing | Left missing |
| --- | ---: | ---: | ---: | ---: |
| e0 | 344 / 344 | 323 / 323 | 29 | 50 |
| e2 | 313 / 313 | 338 / 338 | 30 | 5 |

For every existing raw and aligned JSON, the filename matched `frame_idx`, the
track ID and handedness matched its side, and `hawor_native_frame_idx` matched
`round(source_idx * (native_count - 1) / (source_count - 1))`. The raw and
aligned MANO parameters matched. Aligned camera translations were finite, had
positive Z, and contained the stabilized real `fx`, `fy`, `cx`, and `cy`.
Every compared right-hand record had its corresponding SAM2 mask. Depth
alignment recorded nonzero `dz` for 343 e0 and 313 e2 right-hand frames.
The right-hand camera Z medians were 0.427 m raw and 0.510 m aligned in e0,
and 0.450 m raw and 0.562 m aligned in e2. The common aligner decodes its
depth PNGs to metres and keeps the positive camera-Z convention.
These checks establish schema, timeline, coordinate direction, and alignment
inputs; they do not establish ground-truth hand-pose accuracy.

The existing downstream `_load_hand_track` loader read each aligned side with
exactly the JSON frame count marked valid and missing frames marked invalid.
The existing result bundles had 373 and 343 `camera_to_world_transform`
matrices. Their rotations had determinant approximately 1, and composing the
bundled right wrist camera transforms with these matrices reproduced the
bundled world transforms within `1.8e-7` (e0) and `2.4e-7` (e2) maximum
absolute element error. The new path uses this same bundle writer and common
camera-to-world convention.

## New execution path

- `tests/test_camera_default.py`: 5 CPU tests passed. They exercise the
  camera-only detector/motion call sequence, 30-to-20 fps export with an
  explicit detector gap, source time and track ID fields, metadata handling
  for a full detector miss, hand IDs from `hand_tracks.json`, and default versus
  explicit HaMeR CLI selection.
- Two focused `tools.e2e.tests.test_e2e.ConfigTests` tests passed: the
  `reconstruct` default resolves to HaWoR and the DynHaMR/refinement rejection
  remains intact.
- Modified Python files passed `compileall`; `git diff --check` passed.
- A 40-frame, 20 fps HaWoR camera-only GPU smoke job was submitted under
  Slurm with one allocated GPU and a script that resolves the allocated GPU
  index to its actual UUID before Docker. Job 42609 on node200 exited before
  Docker with code 6 and produced no inference output. The direct-Docker
  retry, job 42610 on node100, was **pending** at the time of this record.
  No GPU inference result is claimed here. The full videos were not rerun.

The e0/e2 records are reused artifacts. The CPU tests cover the changed code;
they do not substitute for a successful model execution on an allocated GPU.
