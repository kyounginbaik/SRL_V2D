# T32: new ep32 teacher/actor on K32_RP2 -> distilled actor -> beam polish -> build-v2 component -> FINAL splice

Written 2026-11-04 (fork, investigation only; nothing below was run). Paths: C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028,
T=$C22/streamT32_teacher, V16=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023, R=/data1/SRL_V2D/challenge_submissions_SRL,
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python, EPD=$T/ep32/W_RP2/episode_corrected/episode_000032.

## 0. Inputs produced so far
- World: $T/ep32/W_RP2 (build_world_b.py on blib.TGT=targets/K32_RP2; scene/scene_physics.xml, episode_corrected, source_corrected/target_retarget.json).
- Teacher CEMs: $T/ep32/fcemW_rg{0,1}_s{2,3}/ (reference.npz, result.json with input_dir=EPD, execution/{scene_physics.xml,trajectory.npz}).
- Actor CEM: $T/ep32/acemW_rg0_s2/ -> teacher-style dir + `actor_c512_p0.003_v0.0001/` (training/policy.npz, evaluation/result.json; acem.py line 83,
  NOT `actor/`). Its evaluation already ran `evaluate_policy_component` against EPD, so it is directly stageable if it passes the gate in step 3.

## 1. Distil an fcem teacher (skip for the acem actor)
    sbatch --nodelist=node01 $T/code/distill.sbatch 32 $T/ep32/fcemW_rg0_s2 $T/ep32/distill/fcemW_rg0_s2 $EPD            # [GRID] [SEED=44]
- Verified usage (actor_grid_c.py): `--episode E --teacher DIR --output DIR --episode-dir DIR [--grid c,p,v ...] [--workers N] [--seed 44]`;
  default grid 512/1024/2048 centers x gains 0.001/0.003/... ; each `OUT/actor_c{c}_p{p}_v{v}/` has training/policy.npz + evaluation/result.json.
  distill.sbatch pins node01, 8 CPUs, 50 min, log $C22/streamB_ep3233/logs/distill-%j.log (edit the #SBATCH lines in the T32 copy first).
- Selection: for each actor read evaluation/result.json: `metrics.reference_proxy_mp_sr` must be 1.0, `execution.independent_control_replay.ok`
  true; rank by `metrics.official_style_reference_proxy.objects.{blue_cup,wooden_spoon}.mean_*` (3 cm/30 deg proxy vs EPD). Then score
  scorer-faithfully vs targets/K32_RP2 with `E31_TGT=$C22/streamK32_ep32/targets/K32_RP2 python $C22/streamK32_ep32/code/mresH.py eval ...`
  (or the beam's INIT line, step 2). Precedent: plain distills of the passing Oct-28 teacher all dropped the spoon (STATUS 02:32); acem fixed it.
- Optional finger margins: $T/code/margins.sbatch EP ACTOR OUT [hand] [std] (search_margins_k.py, 1 worker, slow).

## 2. Window-beam polish on K32_RP2 (streamB-lineage actor works with the K32 tools)
- mresH.Env(cfg) takes any actor dir with evaluation/result.json: scene = result.execution.scene, epd = result.input_dir (or $E31_EPD), target
  from $E31_TGT via h13lib (targets/<V>/episode_000032/{blue_cup,wooden_spoon}_{trajectory.npz,mesh_metric.obj}), registration meshes = cfg['meshes'].
  So wbeamH (K32 chain) applies; streamB9 wbeam.py/mres.py are older ep32-only copies bound to target_final_v5/v6 - do not use.
- Meshes for registration MUST be the actor's shipped meshes (3c3c5328 cup / c9ab88e3 spoon), e.g.
  M=$C22/streamB9_ep32/staging/ep32_s6f0/components/episode_000032/meshes/episode_000032 (k32rchain200.sh hardcodes the G18 dir M=...G18_ep32_H12e1f0...;
  make a copy k32rchainT.sh with M replaced, or export a variable).
    cd $C22/streamK32_ep32; export HANDS=wooden_spoon:right,blue_cup:none RS=right ADVS=right
    bash code/k32rchainT.sh K32_RP2 T1_c1 <ACTOR_DIR> none none 0.6,8 6,1.2 1.00 16 2 '{}'     # args: V NAME ACTOR TABLE|none INIT TOL HINGE W0 WK NG OW
  Output runs/roll/T1_c1/g2/final_0.npz (table) + FINAL line in logs/K32r-T1_c1-g2-*.log (cup/spoon cm/deg vs K32_RP2). The INIT line = unedited actor.
- Bake the table into the actor: `E31_TGT=...targets/K32_RP2 E31_EPD=$EPD python code/mresH.py save --ep 32 --actor ACTOR --meshes M --out SV --table TBL`
  (this is what svst.sbatch does before staging; mresH.save writes SV/actor/{training,evaluation}; evaluation uses E31_EPD -> proxy vs K32_RP2).
  NOTE the acem actor's own evaluation already used EPD, so with no table you can stage it directly.

## 3. Stage as a build-v2 component (streamB stager, NOT stage_b32_k32)
    sbatch --nodelist=node01 $T/code/stage.sbatch 32 <ACTOR_DIR> $T/staging/<NAME> <TEACHER_DIR> "T32: fcem/acem teacher on K32_RP2 world W_RP2"
- stage.sbatch -> stage_b.py --episode --distilled ACTOR --base BASE --teacher TEACHER --lineage-file ... (pins node01, 2 CPUs, 50 min).
- Gate (stage_b.py ~line 50): result.json must have learned_policy, reference_proxy_success, policy_checkpoint_sha256 == sha(training/policy.npz),
  metrics.reference_proxy_mp_sr == 1.0, execution.physics_verified, independent_control_replay.ok. A baked actor from mresH.save satisfies this only
  if its save-time evaluation (E31_EPD=EPD) passes 3 cm/30 deg.
- PATCH REQUIRED (T32 copy only): line 18 `H = ROOT / '.../streamB_ep3233'` -> `.../streamT32_teacher`; line 62 requires report.input_dir to start
  with $H/ep32 and parent 'episode_corrected' -> $T/ep32/W_RP2/episode_corrected/episode_000032 then passes. input_manifest.json['target_retarget']
  ['record'] (written by build_world_b.py) and manifest['source'] object_poses sha are checked - W_RP2 has them.
- EXPECTED_MESHES = cup 3c3c5328..., spoon c9ab88e3... (line 46); staging ships those. Do NOT run g18swap.py (asserts 'streamK32_ep32/staging' in path).
- stage_b.py itself runs: reevaluate_component (exactness), prepare_shared_component (portable bundle), portable replay validation, and
  common_evaluator.py evaluate --backend legacy (build-v2) into $BASE/common_components/episode_000032 (completed.json: exact_previous_physical_states_actions).
- Then the canonical candidate + scoring (as B9 did):
    sbatch $V16/common_v2_candidates/run_one.sbatch 32 $T/staging/<NAME>/components/episode_000032 T32_<NAME>     # pins node01, 4 h limit
    bash $C22/streamK32_ep32/code/vs.sh 32 T32_<NAME> K32_RP2        # exact flag, archive, mesh cmp vs v29 (spoon will differ: c9ab vs G18 02ecf22f - expected), score_any
  Output: $V16/common_v2_candidates/T32_<NAME>/episode_000032/{completed.json,results/episode_000032.parquet,meshes/...}.

## 4. Package = FINAL with only ep32 pose rows replaced
    cd /tmp && $PY -B $C22/streamK32_ep32/code/k32splice.py T32_<NAME> $R/ep32_fix_20261104/T32_<NAME>/all_20_episodes T32_<NAME>
    python3 -I $R/ep32_fix_20261104/splice32.py $R/FINAL_20261104/all_20_episodes/track3_submission.parquet \
        $R/ep32_fix_20261104/T32_<NAME>/all_20_episodes/track3_submission.parquet $R/track3_final_ep32T32_<NAME>/all_20_episodes/track3_submission.parquet
    cp -r $R/FINAL_20261104/all_20_episodes/submission_kit $R/track3_final_ep32T32_<NAME>/all_20_episodes/ && cd there/submission_kit && \
        $R/.venv/bin/python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m dry --dry-run
- splice32 replaces pos_*/quat_*/checkpoint_sha256 of the 518 ep32 rows only; mesh cells stay FINAL's.
- Mesh/body-frame consistency (verified today): FINAL's ep32 mesh cells are NOT the kit encoding of any candidate's meshes (neither 3c3c/02ecf22f
  nor 3c3c/c9ab) - they are the coworker GT-free meshes (CD-O 0.261) that meshswap_20261103 aligned INTO the K32 body frames (align.py/build18.py;
  memory: "a new mesh must sit in each episode's body frame because poses are unchanged"). The K32 and streamB lineages share those body frames:
  cup 3c3c5328 in both; G18 spoon 02ecf22f was "built in this body frame" (g18swap record: transform identity) = c9ab88e3's frame (same extents).
  So poses from a streamB-lineage rollout are consistent with FINAL's cells exactly as R10_c1/RP2_c1 were (the leaderboard-validated pairing).
  Sanity check before packaging: register FINAL's ep32 cup/spoon cells and the component meshes with the kit registration (step1_mesh-style) and
  confirm the body-frame transform is identity within ~1 mm/1 deg; or simply diff the f0 rows of the new donor vs FINAL (f0 is identical by construction).

## Pitfalls
- Node pinning: distill/stage/run_one pin node01 (AMD; scipy rotation math differs 1e-16 on node200 -> "search on AMD nodes" note, STATUS 02:32);
  acem.sbatch has no pin in the file (submit with --nodelist=node100/node01). Beam chains ran fine on node200 and were build-v2 exact afterwards.
- stage.sbatch/distill.sbatch log paths point into streamB_ep3233/logs - edit to $T/logs. Time limits 50 min (stage can exceed; run_one 4 h).
- stage_b.py PREVIOUS_BY_EP[32] = $V16/streamR_ep32/staging_roll_regrasp/components/episode_000032 (lineage only; must exist - it does for B9).
- Stager refuses a proxy MP-SR < 1 (hard 3 cm/30 deg vs EPD) - same gate that refused RP2_c1 before the mkref world; here EPD is already K32_RP2.
- vs.sh mesh comparison vs the v29 component will report the spoon differs (c9ab vs G18) - informational, not a packaging blocker.

## Unknowns (not verified)
- Whether `mresH.py save` works unchanged on a streamB-lineage actor (wrapper = LearnedPolicy; mresH supports LearnedPolicy per its docstring).
- acem actor dir name: `actor_c512_p0.003_v0.0001` by default args (centers 512, gain 0.003, vgain 0.0001) - confirm from the run.
- Whether run_one.sbatch is needed after stage_b.py's own common_evaluator step (B9 ran it; it gives the canonical common_v2_candidates/<TAG> path that k32splice.py expects).
- Exact registration identity between FINAL's coworker cells and the c9ab/G18 spoon frames was inferred from records, not re-measured today.
