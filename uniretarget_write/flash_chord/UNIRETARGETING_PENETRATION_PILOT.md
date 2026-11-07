# UniRetargeting tissue-box penetration pilot (2026-10-07)

This is a feasibility check, **not** a demonstrated Track 3 improvement. The
object pose, mesh, and contact reference remain immutable. The upper editor
changes bounded Vega/Sharpa hand-joint references; Newton + a frozen FlashSAC
policy checks the resulting physical rollout before publication.

## What was implemented

- A visible-finger/box visual-overlap proxy samples finger mesh vertices inside
  the convex hull of the reconstructed visible object mesh. It is a **proxy**:
  the hull fills holes and concavities and only supports this single-rigid-box
  pilot. Newton collision-shape penetration is also available but did not
  faithfully quantify the visible finger/box overlap in this example.
- The TTSA upper step can search smooth finger-synergy actions against the
  static reference geometry. Its guided action occupies the first upper trial,
  and all upper trials use the same frozen lower-policy physical rollout.
- A candidate may publish only after a fresh physical validation and bounds on
  object tracking, reference overlap, and rollout overlap. The joint-reference
  trust region remains 0.12 rad and 0.5 rad/s added velocity.
- The outer score includes reference and rollout overlap, but this is still a
  simulator-scored two-time-scale editor, **not** an exact bilevel gradient.

## GPU tests

The online 32-world/4,096-transition smoke tests (`43593`, `43594`) completed
on L40S. The guided upper trial improved its training-seed score, but neither
arm published an edit. This short lower policy was unstable: the guided edit
failed the object-orientation gate. Its unassisted evaluation is too early to
compare policy quality.

The more informative three-seed test froze previously trained tissue policies
(`fixed_seed42`, `123`, `456`) and compared each edit against its own baseline
on the same Newton rollout seed, using two worlds per test. Mean differences
below are candidate minus baseline; negative penetration is better.

| TTSA blend | Mean reference overlap | Peak reference overlap | Mean physical-rollout overlap | Mean object-position error | Gates under current 0.1 mm rollout tolerance |
|---|---:|---:|---:|---:|---:|
| 0.25 | −0.393 mm | 0 mm | +0.071 mm | −0.280 mm | 1/3 |
| 0.67 | −0.722 mm | 0 mm | +0.159 mm | +0.103 mm | 2/3 |

The GPU reports were written before tightening the rollout gate from 1 mm to
0.1 mm, so their stored `accepted=true` fields mean **the former gate** passed
in all three seeds. The table's final column recomputes the new gate on those
same measured metrics, without claiming a new GPU run under that gate. A single
stronger direct-edit pilot also reduced physical overlap for seed 42, showing
that proposal choice matters, not only edit magnitude.

Reports are in `outputs/uniretargeting_tissue_20261007/online_guided_frozen_*`.
The targeted container test suite passed (4 tests). No UniRetargeting batch is
left running.

## Interpretation and next decision

This establishes that bounded physical-feedback-aware editing is runnable and
can improve the **kinematic** reference, but does **not** yet show reliable
improvement in actual physical overlap or end-to-end ADD-AUC. A larger blend
improves the static picture but does not solve the physical trade-off. The next
experiment should physically score several geometry-improving actions per
upper update and select among them on held-out resets; the gate must keep
rejecting physical overlap regressions. Only after that works should we spend
on matched multi-seed end-to-end training.

For diverse tasks, replace the box convex-hull check with a verifier interface:
rigid-object signed-distance/contact and grasp stability; articulated joint
limits and mechanism actuation; deformable interpenetration and material
strain. These should share the same immutable task-target, trust-region, and
held-out-rollout publication contract. The current one-object assumption,
fixed finger synergies, three smooth temporal modes, and hand-tuned thresholds
are pilot-specific.

[DexAgent](https://arxiv.org/abs/2609.35318) uses a self-evolving library of
task-property-specific skills and verifiers. A compatible *optional* extension
here would retain successful verifier configurations and upper-edit proposals
across videos to warm-start new tasks. That is different from the central
UniRetargeting claim: lower-policy physical feedback should improve a reusable
robot reference while preserving the demonstrated manipulation objective.
