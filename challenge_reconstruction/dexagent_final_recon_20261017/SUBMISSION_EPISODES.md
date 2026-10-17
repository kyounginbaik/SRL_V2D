# Verified learned-policy episode components

This workspace adapts the released DexAgent MuJoCo tools to floating Sharpa
hands. A successful physical reference is distilled into a learned
phase/proprioception feedback controller, then evaluated through fresh online
inference in the exact scene. This controller is radial-basis behavior cloning,
not FlashSAC and not a native end-to-end DexAgent LLM run.

## Episode 8

The selected learned checkpoint is
`refinement_20261018/learned_episode8/gain_0.01/training/policy.npz`.
Its independent full-horizon local reference-proxy MP-SR is **1**:

| Object | Mean position error | Mean symmetry-aware orientation error |
| --- | ---: | ---: |
| Beige cup | 1.9159 cm | 8.2593 degrees |
| White pot | 1.9932 cm | 11.3697 degrees |

The policy is evaluated online at 500 Hz from the exact scene home, with
56 bounded robot actuators, no object actuation/attachments, no object force
assistance, and the full original 219-frame, 10.9-second timeline. Every recorded
action reproduces from the checkpoint and its causal pre-step observation.
Independent control replay reproduces all 5,450 physical steps exactly.

The component is staged at:

```
refinement_20261018/policy_handoff/episode_000008/
  results/episode_000008.parquet
  meshes/episode_000008/{beige_cup,white_pot}.obj
  readiness.json
  validation/
  provenance/
  portable/
```

The official trajectory reader verifies all 438 episode/object rows. The
official mesh packer verifies the exact body-coordinate meshes and its budgets.
The source `final_recon` was not overwritten. Object poses in the submitted
component are achieved simulated poses, not the reconstructed reference.

### Reproduce on another server

Copy the whole `portable/` directory, then create a Python 3.11 environment:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r /path/to/portable/requirements.txt
.venv/bin/python /path/to/portable/code/reproduce_portable_policy.py replay \
  --bundle /path/to/portable --output /tmp/episode8-reproduction
```

The helper enforces single-thread BLAS/OpenMP before numerical imports. This is
material: different BLAS reduction order changed the very first control by
4.4e-16 and later contact dynamics diverged. With the tested settings, node01
and node100 reproduce the selected rollout bit for bit. Do not assume arbitrary
hardware/library versions are deterministic; inspect `portable_validation.json`.

The bundle preserves original checkpoint/XML/teacher data and includes all scene
assets. Relocation changes only paths and the copied checkpoint's XML-hash
metadata. All learned arrays and compiled physical arrays/options are checked
against the original before running. `--retrain` additionally fits a separate
policy from the preserved causal teacher data; the primary reproduction always
uses the original learned weights. Training uses eight BLAS threads in the
original experiment; single-thread refitting may differ slightly numerically.

To install the source checkout's CPU dependencies and pinned upstream DexAgent:

```sh
bash challenge_reconstruction/dexagent_final_recon_20261017/code/bootstrap_policy.sh
```

`learned_feedback_policy.py` trains the policy;
`evaluate_learned_feedback.py` executes it online;
`stage_learned_policy_episode.py` validates and writes an episode component;
`reproduce_portable_policy.py` builds/replays the self-contained artifact.
The physical teacher data is supplied with the artifact, not fabricated by the
learner. Synthetic recovery labels encode a local stabilizing-teacher assumption;
interleaved held-out loss measures interpolation, not task generalization.

## Submission scope and provenance

These are components for the current official combined Track 3 Parquet packer.
One episode is not a complete leaderboard submission. Do not edit the official
20-episode roster, fill missing episodes with reference trajectories, substitute
checkpoint hashes, or label this custom evaluator as the bundled FlashCHORD
runtime. The producing commit and actual evaluator/source hashes must accompany
the eventual complete pack. No upload is performed by these scripts.

MP-SR here uses immutable reconstructed object poses as a **proxy** for the
unavailable confidential ground truth. Passing it does not guarantee a held-out
score or perfect motion at every frame. Grasp/video audits remain separate:
episode 8's cup is held through its lift, while the pot under-lifts relative to
the demonstration. The measured diagnostics document slip/rolling and contacts.

Episode 8's meshes match the participant-supplied `final_submission.zip` sources:
source PLY bytes match, OBJ vertices/faces preserve their body frame. The archive
README describes data-only reconstruction without ground-truth priors; deeper
reconstruction history referenced by that README is not present locally. The
handoff records that distinction instead of inventing a reconstruction lineage.

Current requested input scope is every episode present in `final_recon`:
`1, 3, 5, 6, 8, 10, 19, 32, 33, 35, 36, 37, 38`.
The official roster additionally includes `14, 15, 16, 20, 25, 28, 30`, which are
not in that folder. Episode 3's promising metric result is still under grasp
posture review. Components are promoted individually only after the current
physical, metric, visual, and checkpoint checks pass.
