# White-pot carry correction (episodes 3, 5, 6, 8) — 2026-10-20

Observation (user): in the first upload the pot tilts while carried, unlike the video.
Measured against the reconstructed reference (`code/summarize_pot_mass_grid.py`): mean tilt while
lifted 22 / 26 / 31 / 18 deg (episodes 3 / 5 / 6 / 8); pot sags 1.9 cm (ep3) and 4.2 cm (ep8).

## 1. Pot mass is not the cause (`mass_grid/`, `mass_grid_summary.json`)
`code/evaluate_pot_mass_variant.py` re-runs the submitted learned actor in the frozen scene with the pot's
inertial mass/inertia scaled (200 → 150 / 100 / 50 / 30 g). The 200 g control reproduces the submitted
rollouts bit-exactly. Tilt while lifted stays 20–35 deg at every mass; 150 g passes with marginal changes,
≤100 g breaks grasps (spoon flies, pot drops). The tilt is geometric: the pot pivots in the grasp until
it meets a stop, independent of weight.

## 2. Fitted carry wrist correction (`carry_tilt_grid*/`)
`code/carry_tilt_policy.py` wraps the existing component actor: during a window from lift-off (+delay)
to set-down/release, the holding hand's commanded pose is rotated by `rot_gain × (mean tilt)^-1` about a
pivot `pivot_fraction × (pot centre in hand frame)` and translated by `pos_gain × mean position error`,
with smooth ramps. Parameters are measured from the submitted rollout by `code/fit_carry_tilt.py`
(round 1), `fit_carry_tilt_round2.py`, `fit_carry_tilt_round3_ep6.py`. Evaluated by
`code/evaluate_carry_tilt_variant.py` (same loop as the frozen evaluator). Ranking: `code/summarize_variants.py`.

Best wrapper rollouts (pot pos / orient; other object; tilt while lifted):
- ep3 `rg1.00_pg0.0_pf1.0_setdown`: 0.96 cm / 10.0° ; spoon 2.65 / 24.1 ; tilt 22 → 10°
- ep5 `rg1.00_pg1.0_pf1.0_release`: 2.54 / 14.7 ; spoon 2.45 / 27.0 ; tilt 26 → 19°
- ep8 `rg1.00_pg0.0_pf1.0_setdown`: 2.46 / 8.7 ; cup 1.98 / 8.7 ; tilt 18 → 10° (pot still under-lifts 4.5 cm)
- ep6: no robust gain. Spoon/pot contact dynamics are chaotic; the only passes (rg0.5–0.6) push the spoon to
  2.5–2.95 cm and leave tilt at 26–29°. Episode 6 keeps its previous component.

## 3. Distillation back into the registered architecture (`distilled/`, `distilled_sweep/`)
`code/distill_carry_variant.py` trains the plain radial-basis phase/proprioception actor
(`learned_feedback_policy.train`) from the corrected rollout and evaluates it with the UNCHANGED
`evaluate_policy_component.py` in the unchanged scene. No new runtime module is shipped.
- ep3: pot 0.97 cm / 10.1°, spoon 2.51 / 22.6 (pass)
- ep5: pot 2.57 / 14.7, spoon 2.46 / 26.5 (pass)
- ep8 (`rg1.00_pg0.0_pf1.0_setdown_fg0.03_c512`): pot 2.36 / 8.6, cup 1.91 / 7.9 (pass). Other
  hyperparameters flip pass/fail (grasp sensitivity), see `distilled_sweep/`.
- ep6: distillation of the marginal variant failed (spoon 10 cm) — not shipped.

## 4. Components and packaging (`staging/`)
`code/stage_pot_component.py`: fresh re-evaluation (exactness) → shared component + portable bundle →
portable replay from /tmp → common frozen-build dispatcher → portable archive. Same evaluator build /
hash pair as the 2026-10-19 package, so the uniform-provenance rule of the official tools holds.
`dexagent_remaining7_20261019/package_code/assemble_twenty_swap.py` replaces episodes 3, 5, 8 in the
complete twenty-episode package, re-packs with the official packer and re-validates; the sanitised SRL
copy is made by `challenge_submissions_SRL/make_srl_folder.py`.

## 5. Videos (`videos/`, `videos_distilled/`)
`code/render_pot_videos.py` → `render_rollouts_titled.py` (node100): original video | kinematic reference |
submitted rollout (pot 200 g) | corrected rollout, 640×480, 20 fps.

## 6. Second pass after user review (ILC, `ilc/`, `ilc2/`, `distilled_sweep_v2/`, `staging_v2/`)
User review: ep5/ep6 pot still rotated by the spoon during stirring (6–10 s); ep8 pot should be lifted more (5–8 s).
`code/ilc_carry.py` + `ilc_driver.sbatch`: iterative per-frame correction tables (rotation about the pot centre and
lift), optional finger-closing bias on the holding hand. ep8: finger bias 0.3 rad, 5 iterations → pot 1.41 cm / 8.8°,
sag 4.2 → 2.0 cm (distilled `distilled_sweep_v2/.../ilc_fb0.3_iter2_fg0.01_c512`, staged in `staging_v2`,
packaged as `challenge_submissions/track3_pot_fix_v2_20261020`). ep5: ILC removes the tilt (5–8°) but the pot position
passes 3 cm and the spoon degrades → kept the round-1 variant. ep6: every chain loses the spoon → original kept.

## 7. Power-grasp re-plan for episodes 3, 5, 6 (`power_grasp/`)
`code/prepare_pot_power_pickups.py`: DexAgent grasp-pocket proposals on the handle at 11.5 / 14 cm from the pot centre
(palm down, two finger directions) → `physics_pickup_search.py` (CEM, 8 workers, ~6 min each; `power_grasp/search/`).
Tilt while lifted drops to 4–12 deg with every grasp; failures were the set-down (old plan blended back and knocked
the pot) and a yaw offset at closure. `code/ilc_reference.py`: per-frame left-wrist correction about the pot's reference
centre + press/open/retreat release held to the end (`ilc_ref/`, `ilc_ref2/`). Then right-hand spoon retention CEM on
the new pot plan, full length without the first-pass stop (`code/refine_versioned_retention_full.py`, `retention_full/`).
Teachers: ep3 pot 1.08/5.6 spoon 1.99/19.7; ep5 1.22/3.8 spoon 1.83/12.0; ep6 1.08/11.7 spoon 1.27/13.1.
`code/distill_teacher.py` sweeps (`distilled/`): ep5/ep6 distil robustly; ep3's full teacher never distils (spoon lost in
12 settings) → shipped the earlier ep3 distilled actor (pot 1.05/5.8, spoon 2.72/28.5). Staged in `staging_v3`
(ep5, ep6) and `staging_v3fb` (ep3); package `challenge_submissions/track3_pot_fix_v3_20261020`.

## 8. Episode 3 spoon release (v4, `power_grasp/release_ep3/`)
User review of v3: the spoon ends in a wrong pose. Trace: the full-teacher distillations track the teacher until 14 s and
then drop the spoon at the final release (plain LearnedPolicy has no release primitive; the first upload's ep3 actor had
one). `code/add_release_primitive.py` wraps the distilled actor (`mid_a_v2_full/fg0.01_c512`) with the registered
release_feedback_policy.ReleasePolicy using the old checkpoint's fitted release parameters; all 12 timing/placement
variants pass (spoon 1.8–2.0 cm / 20°). Shipped `start14.4_place1.0_bias`: pot 1.27/5.7, spoon 1.77/19.8 (22.8° at the
last frame). Staged in `staging_v4`; package `challenge_submissions/track3_pot_fix_v4_20261020`.

## 9. Episode 3 pot set-down (v5, `power_grasp/release_grid_ep3*/`, `release_ep3_v5/`)
The unwrapping fingers pushed the pot 3–4 cm sideways at release. `code/rewrite_release.py` grids the plan-level release:
best = open 50 % over 1.5 s, retreat 8 cm along the handle axis, then finish opening → pot after release 1.4 cm (was 4 cm),
episode mean 0.75 cm / 8.3°. Distilled (fg0.01_c512) + spoon release primitive → pot 0.84/8.9, spoon 1.91/19.8.
Staged in `staging_v5`; package `challenge_submissions/track3_pot_fix_v5_20261020`.

## 10. v6: balanced episode 3 release (`power_grasp/release_grid3_ep3/`, `release_ep3_v6/`, `staging_v6`)
The v5 release (open 50 % over 1.5 s, slide along the handle) left the pot 1.4 cm away but twisted it ~30 deg at the end.
Grid 3: open 60 % over 2.5 s, slide 8 cm along the handle over 0.6 s, finish opening → final 2.2 cm / 13 deg (v4: 4.1 cm / 11 deg).
Distilled fg0.01_c1024 + spoon release primitive → pot 0.79 / 5.5, spoon 1.92 / 23.2. Package `track3_pot_fix_v6_20261020`.
