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

## Episode 3

The selected actor is
`refinement_20261018/episode_000003/quant6_release_gentle_v5/policy.npz`.
It improves the earlier unused-finger posture and replaces the spoon's unstable
release with a physically fitted placement/opening primitive. Full-motion mean
errors are **2.683 cm / 16.918 degrees** for the pot and **2.569 cm / 21.149
degrees** for the spoon. Maximum measured finger soft-limit excursion is
**1.074 degrees**. The final spoon release has a brief **3.75 mm** settling rise,
then sustained tabletop support and no hand contact.

The controller includes the learned phase/proprioception actor and posture head,
with finite physically fitted grip and release parameters. It observes no
object target or future state at runtime. The posture-only teacher's task
rejection is preserved as lineage; final success is measured independently.
Six-decimal command rounding removes the cross-host rotation roundoff issue.

The component is `refinement_20261018/shared_policy_handoff/episode_000003`.
It contains all 626 official-schema pose rows. Fresh shared evaluation on node01
and node100, and a separate portable replay from `/tmp`, reproduce every state
and action exactly. Earlier frozen component bundles keep their original
evaluator manifests; final aggregation must re-evaluate under one common build.

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

## Episode 5

The selected checkpoint is
`refinement_20261018/pinky_margin_v2/episode_000005/policy.npz`.
Fresh online inference passes reference-proxy MP-SR: pot **2.5094 cm / 20.4090
degrees**, spoon **2.5524 cm / 25.8931 degrees**. Both hands' measured joints are
audited at every 500 Hz physics step; the maximum angular soft-limit excursion
is **0.919 degrees**. Object targets, timestamps, initial state, masses and
actuator limits are unchanged.

This actor is the trained phase/proprioception controller plus physically fitted
pinky command margins. Its calibrated table/gravity adapter uses one rigid
transform for hands, objects and cameras, with the inverse applied at export.
The teacher and every original learning/search report remain separately recorded.

Its component is `refinement_20261018/shared_policy_handoff/episode_000005`.
It contains all 576 official pose rows, exact reconstructed object meshes and a
portable checkpoint. A fresh replay on node200 from `/tmp` reproduced every
recorded physical state and action exactly. Use:

```sh
python /path/to/portable/code/reproduce_policy_component.py replay \
  --bundle /path/to/portable --output /tmp/episode5-reproduction
```

The newer `evaluate_policy_component.py` registry supports both the original
BC actors and their explicitly recorded posture/finger refinements.
`prepare_shared_component.py` stages these actors with a consistent evaluation
code manifest and an explicit adapter path. Existing episode8/37 bundles remain
unchanged. The final combined submission must use one common evaluator build;
older components are re-evaluated and compared before final aggregation.

## Episodes 10 and 38

Episode 10 now completes both cup pickups, nesting and a supported release.
Its learned controller has mean errors **1.046 cm / 5.283 degrees** for the blue
cup and **2.576 cm / 5.427 degrees** for the beige cup. Both hands disengage by
6 seconds; the cup stack stays supported through the complete 8-second episode.
The source reconstruction gives a shallower simulated nesting than the observed
video; that remaining error is included in the score. Maximum actual finger
soft-limit excursion is **1.012 degrees**, recorded separately from the metric.

Episode 38's learned controller carries and releases the dustpan with mean
errors **1.790 cm / 23.794 degrees**. Hand contact is present throughout the
measured airborne motion. Some rolling/slip remains and is documented in its
physical review; maximum actual finger soft-limit excursion is **0.981 degree**.

Their components are `refinement_20261018/shared_policy_handoff/episode_000010`
and `episode_000038`, with 322 and 263 official-schema pose rows, respectively.
Fresh portable policy execution on another node reproduced all physical states
and actions exactly. They use the same shared evaluator build as episode 5.
`finalize_component_provenance.py` verifies each evaluator file against both its
recorded SHA-256 and the actual Git commit before recording provenance. The
portable archive includes the complete episode component, learned weights and
source/teacher evidence. These remain individual components, not a complete
20-episode submission.

## Episodes 33 and 35

Episode 33 now carries both objects and places them with support before opening
the hands. Mean errors are **2.506 cm / 5.333 degrees** for the cup and
**2.554 cm / 20.517 degrees** for the spoon. Both retain continuous geometric
hand contact while elevated. The maximum measured finger soft-limit excursion
is **1.505 degrees**; spoon rolling and some extended unused fingers remain.
`MultiReleasePolicy` applies independently fitted release primitives to the two
hands while retaining the learned phase/proprioception feedback.

Episode 35 carries the spoon and completes a supported set-down, replacing the
previous late bounce. Mean errors are **2.461 cm / 22.279 degrees**, with
**1.202 degrees** maximum finger soft-limit excursion. The selected controller
keeps the right-hand grasp for final placement instead of reproducing the human
left-hand takeover. It therefore passes the object-motion criterion, not an
exact hand-role imitation criterion. Brief contact gaps, rolling/slip and a
short pinky/table contact are documented in its physical review.

Episode 35 uses a separately versioned wrist-coordinate frame to avoid the
original near-singular Euler trajectory. The hand's reference world poses are
verified unchanged by this coordinate conversion. The virtual robot axes and
their dynamics change, so this is explicitly recorded as an embodiment variant;
object geometry, target poses, timing, masses and force caps are unchanged.

Components are `refinement_20261018/shared_policy_handoff/episode_000033` and
`episode_000035`, with 696 and 421 official-schema rows. Fresh portable execution
from `/tmp` on another node reproduces every physical state and action exactly.
Their source and teacher lineage, full joint audits, support-force evidence and
visual limitations accompany the checkpoints. The archives contain the portable
component and evidence; comparison videos remain adjacent to the archive.

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
not in that folder. Episode 3's supported-release correction and explicit command
precision now pass cross-node and portable checks; its earlier rejected variants
remain preserved as research evidence.
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
