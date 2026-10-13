# Episode delta search diagnostic

The upper editor previously used 17 one-second chunks and 12 wrist channels
(204 parameters). This diagnostic compares temporal granularity and finger
editing independently, using a frozen lower FlashSAC policy:

| Variant | Temporal coefficients | Spatial channels | Parameters |
|---|---:|---:|---:|
| episode_wrist | 1 | 12 wrist | 12 |
| episode_full | 1 | 12 wrist + 44 finger | 56 |
| chunk_wrist | 17 | 12 wrist | 204 |
| chunk_full | 17 | 12 wrist + 44 finger | 952 |

A single episode delta adds the same coefficient vector throughout the episode.
The existing 0.25-second endpoint taper, joint limits, correction speed limits,
and 0.25 publication blend remain active. Thus the *projected* offset can vary
near joint limits and episode boundaries. This is a ReActor-inspired temporal
parameterization, not a reproduction of the ReActor optimization algorithm.

`episode_*` uses a 60-second chunk, longer than this 16.75-second episode, so
it retains the same contact feedback and parallel evaluation implementation.
The script checks that the resulting basis has exactly one column and that
its interior equals one. It also checks dimensions, fixed endpoints, and
whether finger references change in the intended variants.

## Run

Inside the existing FlashCHORD environment, from this directory:

```bash
python scripts/probe_episode_delta.py \
  --checkpoint /absolute/path/to/policy_5033984.safetensors \
  --output /absolute/path/to/results/episode_full/seed_43 \
  --variant episode_full --seed 43 --worlds 2048
```

Repeat for the four variants and seeds 43, 44, 45. Use the **same checkpoint**
for all 12 runs. Inputs referenced by checkpoint metadata (parquet, meshes,
robot-reference snapshot, finger PCA basis) must remain accessible. These
private/generated assets and model weights are not included in Git.

The script evaluates 2,048 distinct candidates simultaneously with one world
per candidate, no object assistance, the same starting reference, and a frozen
lower actor. It performs one upper search batch per run. It does not resume or
modify the running training experiments. W&B runs use group
`episode-delta-frozen-policy-20261013`; use `--no-wandb` for local-only execution.

The current implementation includes the parallel reference wiring, wrist/full
hand edit scopes, correlated finger exploration and its stable QR covariance
factorization, stronger-controller configuration, and performance-gated VOC
needed to load the current checkpoints. The VOC gate uses object performance;
hand RMS is diagnostic, not a promotion criterion.

## Read results

Each output directory contains `config.json` with checkpoint/reference hashes,
`status.json`, `summary.json`, per-candidate `search_batches/step_1/scores.json`,
`proposals.npz`, selected reference, and published reference snapshots.

Compare gate-pass fraction, raw improvement fraction, best valid cost relative
to baseline, deterministic acceptance, per-object position/orientation and
contact errors, and published finger displacement. Acceptance alone is not
success: the gate requires at least 1 mm contact improvement and limits
per-object regressions. A selected proposal is rerun before publication.

If episode_full outperforms chunk_full across search seeds, that supports a
search difficulty caused by the larger temporal parameterization *at this
checkpoint and candidate budget*. Comparing episode_full to episode_wrist
separately tests whether finger freedom helps. A negative result does not prove
that dimension is irrelevant: a constant correction can be too restrictive,
the lower policy may not respond to edits, or the acceptance gate can reject
otherwise useful proposals. This one-batch diagnostic does not establish
long-term bilevel learning performance or physical feasibility for all grasps.

Validation:

```bash
JAX_PLATFORMS=cpu python -m pytest \
  tests/retargeting/test_episode_delta.py \
  tests/retargeting/test_contact_proximity.py \
  tests/retargeting/test_ttsa_editor.py \
  tests/lifecycle/test_performance_voc.py -q
```

The initial implementation was checked with 21 targeted tests passing (one
optional test skipped), 26 FlashSAC learner tests passing, and a complete GPU
search over 2,048 full-hand episode candidates. Scientific comparisons require
all matched variants; an accepted first batch is not evidence of superiority.
