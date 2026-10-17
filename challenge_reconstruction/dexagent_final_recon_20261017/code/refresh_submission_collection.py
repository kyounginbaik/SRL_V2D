#!/usr/bin/env python3
"""Refresh an incremental local collection without fabricating unready episodes."""
from pathlib import Path
import hashlib
import json
import os

B=Path(__file__).resolve().parents[1]
ROOT=B.parents[1]
DEST=ROOT/'challenge_submissions/track3_dexagent_20261018'
AVAILABLE=[1,3,5,6,8,10,19,32,33,35,36,37,38]
MISSING=[14,15,16,20,25,28,30]


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True,exist_ok=True);episodes=[];verified=[];pending_provenance=[]
    for ep in AVAILABLE:
        name=f'episode_{ep:06d}';source=ROOT/'challenge_reconstruction/final_recon'/name
        source_meta=read(source/'manifest.json')
        row=dict(episode=name,source=str(source),objects=source_meta['objects'],frames=source_meta['frames'],fps=source_meta['fps'],
            immutable_source_hashes={f:sha(source/f) for f in ['reference.parquet','object_poses.npz','manifest.json']},
            status='refinement or baseline evaluation pending',submission_component=None)
        handoff=B/'refinement_20261018/policy_handoff'/name
        # Explicitly registered unified components use the same evaluator build.
        # Keep original frozen components selected until a replacement is
        # actually staged, independently replayed and has real code provenance.
        shared=B/'refinement_20261018/shared_policy_handoff'/name
        if ep in (3,5,10,38) and (shared/'readiness.json').exists():
            sr=read(shared/'readiness.json');sp=shared/'portable_validation/portable_validation.json'
            if sr.get('episode_ready_for_packaging_review') and sp.exists():
                portable_report=read(sp)
                if portable_report.get('all_physical_states_actions_exact') and portable_report.get('source_object_targets_unchanged'):
                    handoff=shared
        if (handoff/'readiness.json').exists():
            r=read(handoff/'readiness.json');commit=handoff/'provenance/producing_code.json'
            portable=handoff/'portable_validation/portable_validation.json'
            if r.get('episode_ready_for_packaging_review') and portable.exists() and not commit.exists():
                ev=read(r['policy_evaluation_report']['path']);pv=read(portable)
                if ev['learned_policy'] and ev['reference_proxy_success'] and pv['all_physical_states_actions_exact']:
                    row['status']='Learned-policy MP-SR, public format and portable replay pass; real producing-code provenance pending'
                    row['staged_learned_policy_component']=dict(handoff=str(handoff),
                        checkpoint=r['learned_policy_checkpoint'],reference_proxy_mp_sr=ev['metrics']['reference_proxy_mp_sr'],
                        object_metrics=ev['metrics']['official_style_reference_proxy']['objects'],
                        portable_validation=str(portable),producing_code_commit=None)
                    pending_provenance.append(name)
            check=r.get('episode_ready_for_packaging_review') and commit.exists() and portable.exists()
            if check:
                cp=r['learned_policy_checkpoint'];ev=read(r['policy_evaluation_report']['path']);pv=read(portable)
                check=sha(cp['path'])==cp['sha256'] and ev['learned_policy'] and ev['reference_proxy_success']
                check=check and pv['all_physical_states_actions_exact'] and pv['source_object_targets_unchanged']
                if check:
                    validation=read(handoff/'validation/official_reader.json')
                    if validation['problems']:raise ValueError('Previously valid schema now reports problems')
                    if sha(r['policy_rollout']['path'])!=r['policy_rollout']['sha256']:raise ValueError('Staged trajectory changed')
                    row.update(status='verified local learned-policy episode component; organizer acceptance not certified',
                        submission_component=dict(handoff=str(handoff),readiness_sha256=sha(handoff/'readiness.json'),
                            checkpoint=cp,scene=dict(path=ev['execution']['scene'],sha256=sha(ev['execution']['scene'])),
                            adapter_input_sha256=r['source_manifest_sha256'],
                            reference_proxy_mp_sr=ev['metrics']['reference_proxy_mp_sr'],
                            object_metrics=ev['metrics']['official_style_reference_proxy']['objects'],
                            producing_code=read(commit),portable_validation=str(portable),
                            scope='Custom frozen-scene learned controller, validated public format; not bundled official harness'))
                    link=DEST/'components'/name;link.parent.mkdir(exist_ok=True)
                    if link.is_symlink():
                        if link.resolve()!=handoff.resolve():raise FileExistsError('Existing collection component points elsewhere')
                    elif link.exists():raise FileExistsError('Refusing to overwrite existing component directory')
                    else:link.symlink_to(handoff.resolve(),target_is_directory=True)
                    verified.append(name)
                    quality=DEST/'quality'/name/'actual_joint_limits.json'
                    if quality.exists():
                        joint=read(quality)
                        row['actual_joint_quality']=dict(audit=str(quality),
                            max_hinge_violation_degrees=joint['hinge_max_violation_degrees'],
                            quality_review_pending=joint['hinge_max_violation_degrees']>1.0,
                            threshold_interpretation='1 degree flags local review only; no official joint tolerance is claimed')
                        if row['actual_joint_quality']['quality_review_pending']:
                            row['status']='MP-SR, public format and replay verified; actual joint-limit quality review pending'
        baseline=B/'refinement_20261018/expanded_batch/baseline'/name/'result.json'
        if baseline.exists():
            result=read(baseline);row['raw_reference_baseline']=dict(path=str(baseline),
                reference_proxy_mp_sr=result['metrics']['reference_proxy_mp_sr'],
                object_metrics=result['metrics']['official_style_reference_proxy']['objects'],
                physics_verified=result['execution']['physics_verified'],learned_policy=False)
        native=B/'refinement_20261018/expanded_batch/native_mpc'/name/'result.json'
        if native.exists():
            result=read(native);row['native_mpc_baseline']=dict(path=str(native),
                reference_proxy_mp_sr=result['metrics']['reference_proxy_mp_sr'],
                object_metrics=result['metrics']['official_style_reference_proxy']['objects'],
                physics_verified=result['execution']['physics_verified'],learned_policy=False)
        candidates=[]
        expanded=B/'refinement_20261018/expanded_batch'
        for path in sorted(expanded.glob(f'*/{name}/**/result.json')):
            candidate=read(path)
            if 'metrics' not in candidate or 'execution' not in candidate:continue
            if not candidate['execution'].get('physics_verified'):continue
            candidates.append(dict(path=str(path),learned_policy=bool(candidate.get('learned_policy',False)),
                reference_proxy_mp_sr=candidate['metrics']['reference_proxy_mp_sr'],
                object_metrics=candidate['metrics']['official_style_reference_proxy']['objects'],
                quality_promotion_claim=False))
        if candidates:row['physically_replayed_experiment_results']=candidates
        episodes.append(row)
    status=dict(collection='Incremental local Track3 learned-policy episode components',available_source_episodes=AVAILABLE,
        source_folder_episode_count=13,official_required_episode_count=20,missing_official_source_episodes=MISSING,
        verified_local_components=verified,verified_count=len(verified),
        staged_components_pending_real_code_provenance=pending_provenance,episodes=episodes,
        source_pose_proxy_not_official_ground_truth=True,complete_submission_ready=False,kaggle_ready_archive_created=False,uploaded=False,
        next_step='Refine and physically verify remaining episodes, train actual learned controllers, independently rerun, then stage each component. Final20-episode packing requires all missing source episodes too.')
    tmp=DEST/f'.status.{os.getpid()}.json';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(DEST/'status.json')
    lines=['# Incremental Track 3 collection','',
        f'{len(verified)} locally validated learned-policy episode component(s) are linked under `components/`. Validation covers MP-SR, public format and replay; physical quality limitations are listed below. This is a partial working collection; no Kaggle-ready archive or upload has been made.','',
        'The source folder contains episodes 1, 3, 5, 6, 8, 10, 19, 32, 33, 35, 36, 37 and 38. The official 20-episode roster also requires missing source episodes 14, 15, 16, 20, 25, 28 and 30.','',
        'MP-SR below uses immutable reconstructed object poses as proxy ground truth. Each object must have mean position error below 3 cm and symmetry-aware mean orientation error below 30 degrees. These are not confidential leaderboard scores.','',
        '| Episode | Object | Mean position | Mean orientation | Proxy MP-SR |','|---|---|---:|---:|---:|']
    for row in episodes:
        component=row['submission_component'] or row.get('staged_learned_policy_component')
        if component:
            for obj,metrics in component['object_metrics'].items():
                lines.append(f"|{row['episode']}|{obj}|{metrics['mean_position_error_cm'][0]:.4f} cm|{metrics['mean_orientation_error_degrees'][0]:.4f} deg|{component['reference_proxy_mp_sr']:.0f}|")
    if pending_provenance:
        lines.extend(['','Learned-policy components awaiting real producing-code provenance: '+', '.join(pending_provenance)+'. Their local metric/schema/replay checks have passed, but they are not yet counted as completed collection components.'])
    for row in episodes:
        q=row.get('actual_joint_quality')
        if q and q['quality_review_pending']:
            lines.extend(['',f"**{row['episode']} quality review pending:** actual simulated soft joints exceed their bounds by up to {q['max_hinge_violation_degrees']:.3f} degrees despite bounded commands. The frozen component retains its valid proxy score and exact replay, but this does not certify acceptable grasp/joint quality. See `{q['audit']}`."])
    lines+=['','`status.json` records the actual checkpoint, scene, input hashes, producing commit, full-horizon physical verification and portable reproduction evidence for each component. Custom learned MuJoCo execution is explicit; public-format validity does not certify organizer acceptance.',
        '', 'Symlinks currently point to frozen experiment handoffs. When every required episode is ready, final packing should copy the actual trajectory and mesh bytes into a complete portable submission and run the unmodified official roster validators. Missing episodes are never padded or fabricated.','']
    (DEST/'README.md').write_text('\n'.join(lines));print(json.dumps({'collection':str(DEST),'verified_components':verified,'complete_submission_ready':False}))


if __name__=='__main__':main()
