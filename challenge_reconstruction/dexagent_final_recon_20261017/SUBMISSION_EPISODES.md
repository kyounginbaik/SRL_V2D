# Incremental learned-policy episode components

This workspace adapts the released DexAgent MuJoCo tools to floating Sharpa
hands. A successful physical reference is distilled into a learned
phase/proprioception feedback controller, then evaluated through fresh online
inference in the exact scene. This controller is radial-basis behavior cloning,
not FlashSAC and not a native end-to-end DexAgent LLM run.

The live collection is `../../challenge_submissions/track3_dexagent_20261018`.
Its `status.json` distinguishes metric/format/replay passes from unresolved hand
quality checks. All 13 available episodes are in scope; no complete submission
archive or upload has been made.

## Episode 37

The selected learned checkpoint is
`refinement_20261018/expanded_batch/learned/episode_000037/gain_0.03/training/policy.npz`.
The wooden spoon has mean position error **1.462 cm** and mean orientation error
**9.603 degrees**, passing reference-proxy MP-SR. Fresh online policy inference,
causal action verification and independent physical replay pass. Maximum measured
finger soft-limit excursion is **0.919 degrees**. The separately measured physical
teacher has continuous hand contact during its elevated motion and an opposed
thumb/finger pinch; reference poses are never substituted for achieved motion.

Its component is `refinement_20261018/policy_handoff/episode_000037`, with all
151 expected pose rows, meshes, actual trained weights, and a portable replay.
The portable execution was repeated on another node with bitwise-identical
controls and states under the documented single-thread numerical settings.

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

An additional all-joint audit found actual pinky CMC soft-limit excursions of
5.650 degrees on the right (during initial settling) and 3.798 degrees on the
left. Commands remain within bounds, but that does not guarantee actual joint
positions do. The frozen passing component is preserved; joint-quality review
and separate corrective policy trials are pending. Do not describe it as having
passed every physical-quality check.

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
not in that folder. Episodes 3 and 5 now have fresh learned-policy metric passes,
but their inherited left-pinky soft-limit excursions are still under review.
Components are promoted individually only after the current
physical, metric, visual, and checkpoint checks pass.

## Versioned reconstruction and grasp work

For the eight added scenes, `inspect_table_support.py` fits the tabletop to
calibrated stereo depth while excluding hand/object masks. `prepare_episodes.py`
can apply that measured rigid frame to objects, both wrists and moving cameras;
the inverse transform is used for scoring and export. This corrects the support
and gravity setup without editing source object trajectories. Original adapters
and source `final_recon` directories remain available.

`physics_pickup_search_v2.py` tests staged approach, open pinch, closure and lift
commands in physics. `refine_versioned_retention.py` preserves an acquired pickup
prefix and refines subsequent robot commands against the full original object
timeline. Search trajectories are research teachers; a separate trained policy
must succeed online before its result is packaged. The scripts and SBATCH files
record exact scene/reference paths; these research paths are not portable data.
Use each completed component's `portable/` bundle for server-to-server replay.

Episode 1 has a separately versioned RGB/stereo-supported pot registration/mesh
repair under `refinement_20261018/episode_000001/reconstruction_repair`. It is
not promoted merely because its image fit improves; collision, settling, grasp,
full-horizon tracking and learned-policy checks remain required. Optional
reconstruction dependencies are in `code/requirements_reconstruction.txt`.
