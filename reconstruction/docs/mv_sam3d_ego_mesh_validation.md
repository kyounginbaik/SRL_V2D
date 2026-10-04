# MV-SAM3D ego mesh integration validation

2026-10-04. The HaWoR branch and its previous experiment artifacts were not
rebased or overwritten. The public white-pot episode 000000 was used only for
this smoke; the selector and pipeline have no episode or frame-158 constant.
No GT mesh or GT pose was read during selection, generation, or downstream
loading.

## CPU

- Eight focused selector/generator tests passed: four-view temporal-rank
  selection, direct two/three-view support, one-view fallback and reason,
  track identity metadata, DINO-box rejection, RGB/mask coordinate rejection,
  explicit SAM3D manifest, and raw GLB-to-OBJ conversion/cache behavior.
- All 61 `tools.e2e.tests.test_e2e` tests passed after expanding the sparse
  checkout to include an unrelated retarget script used by one existing test.
- Modified Python files compiled, `git diff --check` passed, and both the
  consolidated and direct WiLoR entrypoints exposed the new mesh options.
- The default `v2d_mvsam3d:latest` Docker layer built successfully from the
  SAM3D base image and imported Kaolin, Open3D, and PyTorch3D on CPU.
- On existing public white-pot RGB/depth and the corrected SAM2 pot-body mask,
  24 of 48 inspected candidate frames passed quality gates. Seed 42 selected
  `[158, 63, 166, 261]`. The four paired RGB/RGBA inputs passed exact RGB and
  alpha-mask checks.

## Slurm GPU smoke

- Job 42703 used one node200 B200. The pinned official MV-SAM3D script loaded
  all four selected views, completed weighted shape/texture inference, and
  saved a 5.1 MB `result.glb` (266,144 triangle faces).
- The raw GLB was copied and converted to OBJ without GT alignment. GLB and
  OBJ bounds matched. The OBJ loaded through `v2d.mesh.lib.mesh.Mesh.load`,
  the exact reader used at the start of FoundationPose scale estimation, with
  133,072 vertices and 266,144 faces.
- The B200 image had a Kaolin binary compiled against an older PyTorch. This
  smoke used a temporary import shim for Kaolin visualization symbols and
  `check_tensor`; the pinned MV-SAM3D checkout was unchanged. The default
  L40S image was used in the prior four-view pot experiment. A second L40S
  smoke was queued, then canceled after this one succeeded.
- Scope ends at FoundationPose's mesh loader. The full scale grid search,
  FoundationPose tracking, and HaWoR reconstruction were not rerun in this
  smoke. This is the remaining runtime validation limit.
