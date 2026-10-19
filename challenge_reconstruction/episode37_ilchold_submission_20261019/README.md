# Episode 37 replacement: user-selected ilchold run

Selected video: `/data3/exaFLOPs/epi37.mp4`.

This preparation uses the underlying UniRetarget run, rather than replacing only
the preview video. The selected lower checkpoint is
`policy_2508800.safetensors` (SHA256
`46653141bf7c97a95e9dcc9a2edc8be0c997ca514ecc1fec1ff4aef9f2e9ab35`),
the upper reference is `reference_0005.npz`, and the video uses world 28 of a
32-world deterministic evaluation. The original episode 37 package remains
available in the unchanged `track3_dexagent_20261018/available_13_episodes` release.

The independent review in `review/source_audit.json` confirms that the video's
entire 151-frame object trajectory passes the reconstructed-reference MP-SR proxy:
2.137 cm mean position error and 10.767 degrees mean orientation error. These are
not hidden-ground-truth leaderboard results. The reference object poses remain
unchanged. The spoon visual mesh is exactly the existing source/submission mesh.

The saved video recording has body transforms but no robot actions. A fresh GPU
execution of the same checkpoint/reference pair is required to record the actual
actions and states and compare them to that recording. The new controller has
56D raw residual actions at 20 Hz and 400 Hz physics. These differ from the old
MuJoCo controller's absolute position-target actions at 500 Hz.

`source/episode_000037` preserves a byte-checked run snapshot; its inventory is
`source_manifest.json`. It includes the lower policy and training state, upper
references, frozen scene/assets, runtime source, and selected evaluation evidence.
The Docker image is pinned to
`sha256:22d733893d932297a65adfa788d663a42c0778842d96aa20ae2713c07b69971c`.
`code/export_ilchold_rollout.py` records causal state/action transitions without
object assistance. `package_code/common_evaluator.py` executes both controller
backends under one recorded build and validates the public episode grid.

The final replacement package is produced only after fresh episode 37 validation
and exact fresh replays of the other twelve controllers. Their original portable
archives, source lineage, and reviewed physical states/actions are preserved.
See `release_status.json` for actual execution state; this README is a description
of the workflow, not a claim that a pending GPU replay has finished.

The package still covers only the thirteen requested episodes. The official
uploader requires all twenty; no upload is performed by this workflow.
