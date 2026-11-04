# Common evaluator build v2 (2026-10-23)

Same frozen backend sources as episode37_ilchold_submission_20261019 (legacy MuJoCo RBF actors and the ILC-hold runtime);
only package_code/common_evaluator.py changed: the legacy backend may now execute episode 37 (user decision 2026-10-23,
legacy learned actor instead of the ILC-hold run). Producing commit: branch evaluator-build-v2 (5cc5ca51), to be pushed to
github.com/kyounginbaik/SRL_V2D by the user; build_manifest.json records that commit URL.
Every v16 component must be executed under this build (all rows share eval_script/eval_code hashes and the commit URL).
