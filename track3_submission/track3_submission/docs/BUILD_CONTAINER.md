# Building the submission container (draft)

The organizers' guide is `CONTAINER_SUBMISSION.md` in kit `9cad120d` (upstream PR #215, Oct 31). Steps below follow it.
All compute runs inside Slurm, and nothing is written under `/data1/SRL_V2D`.

1. **Commit.** Put this overlay on a checkout of `5cc5ca51`, commit and push it to `kyounginbaik/SRL_V2D`, and share the repository with the GitHub user `Nvidia-V2D-Challenge`.
   - The uploaded FINAL rows say `5cc5ca51`. The verifier checks that the row URL equals `HEAD` of `code/`.
   - So either re-pack and upload FINAL with the new commit URL before the freeze, or keep `code/` at `5cc5ca51` and ship this overlay as documented supplementary code. This is a team decision.
2. **Code.** `submission/code/` must be a `git clone` of that commit with `.git` and no modified tracked files.
3. **Kits.**
   - `submission/kit/` = `FINAL_20261104/all_20_episodes/submission_kit` (the copy that packed FINAL).
   - `submission/verifier_kit/` = kit `9cad120d` (has `tools/verify_container.py`).
   - Also copy `dexagent_final_recon_20261017/refinement_20261017/submission_readiness/latest_official/v2d_submission_kit`. `run_one.sbatch` passes it as `--kit`.
4. **Weights.** `sbatch … container/collect_weights.sh <build>/submission`. It copies 20 component dirs and the 20 build-v2 candidates and checks each `policy.npz` sha256 against `provenance/FINAL_9bc322af_episodes.tsv`.
5. **Meshes and outputs.**
   - `submission/meshes/final/`: the meshes packed into 9bc322af (the same mesh cells as 0fca055b; ep36 mesh byte-identical to the v29 selection). These are coworker GT-free meshes from v30c1 plus the G18 spoon; the paths are in `CWMESH_SWAP.json` and `V32_BUILD.json`.
   - `submission/outputs/final/rollouts/`: the build-v2 candidate outputs `campaign_v16_20261023/common_v2_candidates/<candidate>/episode_XXXXXX/`.
6. **Submitted files.** `submission/submitted/track3_submission.parquet` (9bc322af) plus its receipts for every board where it is Final (`receipt_s4.json` next to the file in `spoon3537/rt/packages/track3_FINALc_ep36s4/all_20_episodes` has AUC and MPPE).
7. **Manifest.** Fill the TODOs in `container/submission.json`: Kaggle submission IDs, the runtime `evaluation_package` and the commit. Remove the `_` notes.
8. **Build and check.**

```bash
docker build -t srl-v2d-track3:final --build-arg BASE_IMAGE=<image with the .venv's Python/MuJoCo> .
```

```bash
docker run --rm --network none srl-v2d-track3:final /submission/run.sh verify
```

```bash
docker run --rm --network none -v OUT:/output srl-v2d-track3:final /submission/run.sh reproduce --label final --output /output --episodes 1
```

9. **Ship.**

```bash
docker save srl-v2d-track3:final | zstd -o srl-v2d-track3-final.tar.zst
```

   Then share a Drive link, the tar sha256 and the image digest with the tech report. The user or team does this.

Known expected verifier result: `eval_code_sha256` FAILs by construction. Our value `fdddb348` follows the build-v2 convention (concatenated sources in `build_manifest.json`), not the kit's 4-module digest.
