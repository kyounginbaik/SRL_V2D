# DexAgent: what we use and what we changed

## Upstream

DexAgent is from https://github.com/dexagent1/DexAgent, commit `e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633` ("cleanup"), under the MIT license.
- `third_party/DexAgent/` is a copy of that checkout. On the cluster, `git status` there is clean.
- Its `examples/` folder (70 MB of demo media) and `.git` are left out.
- Restore the exact tree with:

```bash
git clone https://github.com/dexagent1/DexAgent third_party/DexAgent && git -C third_party/DexAgent checkout e29bdad
```

We did **not** edit any upstream file. All SRL changes sit beside it and import it through
`PYTHONPATH=challenge_reconstruction/dexagent_final_recon_20261017/code:third_party/DexAgent`.
The entry points that set this path are `campaign_v22_20261028/*/code/job.sbatch` and `svst.sbatch`. 32 files in `campaign_v22_20261028` import the port.

## SRL modification: `challenge_reconstruction/dexagent_final_recon_20261017/code/`

| Module | Change |
|---|---|
| `floating_runtime.py`, `floating_tracking.py` | Port DexAgent's released predictive-sampling MPC runtime and its reference interface from the YAM arm to **floating Sharpa hands** (56 actuators, 500 Hz control). Both load the frozen scene XML and reset to its exact saved keyframe. |
| `wrist_reference.py`, `rechart_floating_wrist.py` | Repair equivalent Euler-angle branches of the wrist reference without changing physical wrist poses. |
| `refine_grasp_cem.py`, `refine_*`, `temporal_refine.py` | CEM search over 56 bounded wrist/finger corrections, with unactuated objects and the original timing and initial state preserved. |
| `contact_acquisition_runtime.py`, `contact_retention_runtime.py` | Feedback during pickup and transport from measured contact forces and geometry (fixes pushes and same-side contacts). |
| `ilc_carry.py`, `ilc_reference.py`, `learned_feedback_policy.py`, `distill_teacher.py`, `component_policy.py` | Learned per-episode actors: ILC on the carry phase, feedback policy, teacher → student distillation. The saved actor is the episode's checkpoint (`portable/original/policy.npz`). |
| `audit_*.py`, `*.sbatch` | Audits (contacts, joint limits, release clearance, replay consistency) and Slurm launchers. |

`dexagent_remaining7_20261019/code/` extends the same port to the remaining objects:
- cups, pot hold and release, pitcher, sweeper and dust pan;
- wooden boards, plus the cross-host check for components.

Status against the declared commit `5cc5ca51`: 337 files are byte-identical, including both `floating_*.py`. 203 files are newer and not committed, and 1 file changed (`refine_episode1_collision_meshes.py`). The per-file list is `../provenance/dexagent_evaluator_vs_5cc5ca51.tsv`.

## Agentic DexAgent run (not in the submission lineage)

`experiments/dexagent_egotools_10/` drives the **unmodified** DexAgent agent pipeline through its external-model transport on 10 tool-use clips that are not challenge data (`configs/01_…10_*.yaml`):
- `control.py` and `continue_control.py`: a coding agent supplies the tool calls, and requests and responses stay on disk;
- TODO: check which DexAgent stages each run reached (`results.json`, `round*.json`) before the report describes them.

This was an evaluation of the agentic stages. **No Track 3 row comes from it.**
The rules allow pretrained models and external data. The evaluated-row lineage uses only the MPC port and learned actors above.
