# Campaign phase 3 (2026-10-23 night): episodes 1, 32, 33, 35, 30 on OUR reconstruction meshes

User instruction (2026-10-23): "keep refining 1, 32, 33, 35 and 30. also use the mesh we got from reconstruction, not the
GT scanned one" (the user confirms the scanned cup and white dish rack were used before; that stops now).

v17 is assembled (challenge_submissions/track3_v17_20261023/all_20_episodes_post, NOT uploaded). Phase 3 produces
candidate components for the five episodes; the integrator assembles v18 from v17 + whatever beats v17.

## Mesh audit (sha256 prefix / bytes of the staged visual mesh in common_v2_baseline, i.e. what v17 ships)

| ep | object | v17 mesh | origin |
|---|---|---|---|
| 1 | white_pot a553.. 137 KB, white_pot_lid bc14.. 159 KB | repaired reconstruction (refinement_20261018/episode_000001/reconstruction_repair/patch_height094) | OK, verify no gt_best lineage |
| 30 | plastic_dish_rack 3d8b.. 114 KB | SCAN-derived (rackscan_20261021/code/build_rack_assets.py from /data1/SRL_V2D/gt_best_per_object/plastic_dish_rack.ply) | MUST be replaced by our reconstruction rack fa3a.. 702 KB (dexagent_remaining7_20261019/components/episode_000030/meshes, world dexagent_remaining7_20261019/episodes/episode_000030) |
| 30 | mini_sweeper 26c2.. 411 KB | reconstruction | OK |
| 32 | blue_cup 3c3c.. 138 KB | SCAN-derived (refinement_20261021_ep32/code/build_cup_assets.py, cup_scan_provenance.json; scan visual 044c.. 141 KB) | MUST be replaced by our reconstruction cup 57bb.. 194 KB (/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032/assets/blue_cup.obj; pre-swap collision parts in refinement_20261021_ep32/source/episode_000032/assets/blue_cup_convex_previous_ours) |
| 33 | blue_cup 57bb.. 194 KB | visual is our reconstruction cup, but refinement_20261021_ep33 also ran build_cup_assets.py (cup_scan_provenance.json): the COLLISION geometry / placement may be scan-derived | verify every cup asset (visual + convex parts + urdf/xml) traces to final_recon/episode_000033/assets; rebuild if not |
| 32/33/35 | wooden_spoon c9ab.. 193 KB | reconstruction (no scan exists) | OK |

Rule for phase 3: every mesh in a new world (visual AND collision) must trace to /data1/SRL_V2D/challenge_reconstruction/final_recon/episode_XXXXXX/assets
(or the repaired reconstruction for ep1). Nothing from /data1/SRL_V2D/gt_best_per_object, nothing produced by build_rack_assets.py /
build_cup_assets.py / swap_rack_scene.py. Record the mesh sha256s you ship in your README. Reference object poses are NOT changed
(except ep1's lid-on corrected reference, see below, which is a reconstruction-side correction with lineage, like ep36's roll track).

## Streams (local error cm / deg vs our reconstruction; v17 numbers from common_v2_baseline)

| stream | ep | v17 | target | notes |
|---|---|---|---|---|
| H | 1 | pot 1.50/10.0, lid 2.90/12.8 (lid max 5.3 cm) | lid <= 1.8 / 12, pot <= 1.3 / 10 | recon tilts the pot ~12 deg while the lid is on (stereo says flat): lid-on corrected references in refinement_20261021_ep1/v13/ref_* (lid_on_correction.py); lid-shift / retention searches in refinement_20261021_ep1/lid (none shipped: find out why from logs); shipped actor lineage in refinement_20261021_ep1/staging/components/episode_000001/provenance |
| I | 30 | sweeper 1.89/17.9, rack 0.66/3.3 (scan rack world) | sweeper <= 1.6 / 14, rack <= 1.0 / 5 on the RECON rack world | recon-rack baseline component dexagent_remaining7_20261019/components/episode_000030: sweeper 2.80/21.0, rack 1.16 (validation/reference_proxy_mp_sr.json). Diagnose first (frame-0 rack penetration? brush slump at 0-2.3 s was 0.43 cm / 4.8 deg of the mean in the scan world), then early-steady pickup / retention / release re-plans (rackscan_20261021/code/retention.py, retention_upright.py, extra_lift.py, chain30.sbatch are world-agnostic). Compose staging: dexagent_remaining7_20261019/package_code/stage_seven_component.py with actor_report.json + input_manifest.json in the staging base (see streamF_ep25/code/stage.sbatch) |
| J | 32, 33 | 32: cup 1.27/9.4, spoon 1.94/16.0; 33: cup 1.49/12.3 (max 11.9!), spoon 1.99/15.7 | spoon <= 1.6 / 14, cup <= 1.2 / 10 | rebuild each episode dir with our cup (visual + collision), re-score the shipped actors (score_vs_track.py or evaluate_policy_component), then re-plan: retention_spooncup.py (refinement_20261021_ep32/code), teacher re-plan of the 7-10 s scoops for 33, fresh margins (search_finger_margins33.py), plain actors preferred. Staging: refinement_20261021_ep32/code/stage_3233_component.py (check it does not pull the scan cup) |
| K | 35 | spoon 1.98/20.1 (max 65 deg at the 19 s release) | spoon <= 1.6 / 14 | loose late grip + 42 deg spike at release. Previous: retention windows 9-18.4 s, ILC grids, wrapped margins (refinement_20261021_ep35). Phase-2 recipe that worked on ep36 (streamG_ep36_30/README.md): seated-grasp wrist re-plan after the slip + gentle wrist ILC (1 iter) + release re-plan + carry retention pass, then fresh physically searched margins (refinement_20261022_margins/code/search_finger_margins_gen.py) |

## Procedure (same as phase 2, PHASE2.md)

- Work only inside your stream dir under this campaign folder (streamH_ep1, streamI_ep30, streamJ_ep3233, streamK_ep35); never
  delete or overwrite existing outputs anywhere.
- Slurm: partition debug; distill / stage / evaluation pinned `--nodelist=node01` (replays are not bitwise reproducible across
  nodes); searches anywhere on debug; keep `--time` <= 50 min per job (longer jobs do not backfill behind the user's pending ur-tl
  GPU jobs; chain with dependencies instead); never touch ur-tl-* jobs; videos only `--nodelist=node100`.
- One stage job per staging base dir (two in the same base race and corrupt scene_physics.xml).
- After staging: `sbatch /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/run_one.sbatch EP COMPONENT_DIR TAG`
  (tags H_ep1_*, I_ep30_*, J_ep32_*, J_ep33_*, K_ep35_*); the re-execution must be exact (completed.json) and the numbers are read
  from common_v2_candidates/TAG/episode_XXXXXX/backend/evaluation/mp_sr.json.
- Append one-line milestones to logs/STATUS.md (`echo "- $(date +%F) H ep1: ..." >> logs/STATUS.md`); write streamX/README.md
  (diagnosis, what was tried, numbers, component path, mesh sha256s, how to reproduce).
- Hard rules (user): never upload; no DexAgent mention inside challenge_submissions_SRL; poses only from policy rollouts under the
  frozen evaluator build v2; never edit poses by hand; no scanned meshes.

## Addendum (user, later on 2026-10-23): "for cups, there is the GT mesh or GT resampled mesh we used; don't use that, use what we had from our reconstruction"

Cup audit of the v17 package (vertex counts: GT blue_cup.ply 2144, GT beige_cup.ply 1831; our recon blue_cup 2074, beige_cup 2057):

| ep | shipped cup(s) | verdict |
|---|---|---|
| 8 | beige_cup 2c04.. = final_recon | OK |
| 10 | beige_cup 1831 verts, blue_cup 2144 verts = GT scan cups (lineage 000_episode10_scan_cup_lineage.json) | REBUILD -> stream L (streamL_ep10) |
| 14 | blue_cup 78bd.. = final_recon | OK (rack is scan-derived, see below) |
| 20 | blue_cup ff37.. = final_recon | OK |
| 32 | blue_cup 3c3c.. 2074 verts, re-exported/rescaled copy (scale_scan_over_ours 1.063) | REBUILD -> stream J |
| 33 | blue_cup 57bb.. = final_recon visual; collision built by build_cup_assets.py | verify/rebuild -> stream J |

Rack audit: eps 14, 16, 28, 30 worlds use the scan-derived rack (rackscan_20261021); ep15 physics uses our rack but the
package swaps the scan rack in (postprocess_v16.py, use --skip-ep15-mesh for v18). Recon-rack pre-scan components exist
(dexagent_remaining7_20261019/components/episode_0000XX): ep14 cup 2.45/20.3, ep16 spoon 2.16/21.4, ep28 pot 2.10/8.6 (v17 scan
world: 1.50/1.7, 1.60/14.6, 1.32/7.6). Rebuilding 14/16/28 on the recon rack is NOT launched (user named only ep30); flagged.

## Correction to the cup audit (Stream J, 2026-10-24)

- ep32's shipped cup 3c3c.. (2074 verts) is OUR 57bb reconstruction cup shortened 10% along its axis (refinement_20261018/expanded_batch/
  cup32_height_candidate/scale_0.900, "source-supported dimension correction", mesh_repair_provenance.json, created 2026-10-18 before
  gt_best_per_object existed; affine check max error 5e-9 m). Its collision parts are the same shortening of the recon CoACD parts.
  Integrator decision: counts as our reconstruction (same class as ep1's repaired pot); user may overrule -> 57bb-world chain is the fallback.
- ep33's shipped world is already 100% reconstruction (57bb visual, collision parts byte-identical to final_recon/episode_000033/assets/blue_cup_convex).
- The scan cups (044c.., bac2.., 2144 verts) exist only in unshipped experiment dirs under refinement_20261021_ep32/33.
- ep10 remains the only shipped episode with GT cups (stream L).

## Phase 3b (user, 2026-10-24): "change the rack to our rack; just use the previous one we had"

Rebuild the remaining scan-rack episodes 14, 16, 28 on OUR previously reconstructed plastic_dish_rack (per-episode final_recon rack
visuals, 702 KB: ep14 ccc315ab3efa, ep16 d500aac0f145, ep28 d4bddc301cda; pre-scan worlds dexagent_remaining7_20261019/physics_collision_v2/
episodes/episode_0000XX with the 128-part repaired collision; pre-scan components dexagent_remaining7_20261019/components/episode_0000XX).
Template: streamI_ep30 (README sections 1-6, code/build_world.py --rack-visual, pickup/chain/ret/rel/distill/stage sbatch files).
Streams: M ep14 (streamM_ep14), N ep16 (streamN_ep16), O ep28 (streamO_ep28). Targets: rack <= 1.0 / 5; ep14 cup <= 1.6 / 14;
ep16 spoon <= 1.6 / 14; ep28 pot <= 1.5 / 10 (v17 scan-world numbers: 1.50/1.7+0.30/1.6, 1.60/14.6+1.33/5.2, 1.32/7.6+0.06/1.2;
pre-scan recon-world components: cup 2.45/20.3+0.63/2.5, spoon 2.16/21.4+0.80/6.3, pot 2.10/8.6+0.36/2.8). Same rules as above.
