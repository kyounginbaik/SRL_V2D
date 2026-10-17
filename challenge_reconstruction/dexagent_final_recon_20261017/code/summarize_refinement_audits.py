#!/usr/bin/env python3
"""Refresh the five-episode pilot -> CEM -> block-polish evidence report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1]
EPISODES = (1, 3, 5, 8, 10)


def read(path):
    return json.loads(Path(path).read_text())


def metric_summary(metric):
    return dict(mp_sr=metric['reference_proxy_mp_sr'],
        full_horizon=metric['full_horizon'],
        objects={name:dict(mean_position_cm=v['mean_position_error_cm'][0],
            mean_orientation_deg=v['mean_orientation_error_degrees'][0],
            passes=v['success_per_world'][0])
            for name,v in metric['official_style_reference_proxy']['objects'].items()})


def execution_summary(execution):
    return dict(physics_verified=execution.get('physics_verified'),
        full_horizon=execution.get('execution_completed'), duration_s=execution.get('duration_s'),
        independent_replay_passes=execution.get('independent_control_replay',{}).get('ok'),
        scene=execution['scene'], trajectory=execution['trajectory'])


def grasp_summary(grasp):
    objects={}
    for name,obj in grasp['objects'].items():
        hands={}
        for side,hand in obj['hands'].items():
            windows=hand['elevated_without_nonhand_contact_windows']
            longest=max(windows, key=lambda w:w['duration_s'], default=None)
            hands[side]=dict(contact_seconds=hand['any_contact']['duration_s'],
                thumb_and_finger_contact_seconds=hand['thumb_and_any_finger_contact']['duration_s'],
                longest_unsupported_elevation_window=longest,
                longest_contact_loss_while_elevated_s=hand['contact_loss_while_elevated']['longest_interval_s'])
        objects[name]=dict(max_com_rise_cm=obj['max_com_rise_m']*100, hands=hands)
    return dict(sample_source=grasp['sample_source'], sample_count=grasp['sample_count'],
        diagnostic_lift_threshold_m=grasp['settings']['lift_threshold_m'], objects=objects)


def normalized_error_score(metrics):
    ratios=[max(v['mean_position_cm']/3, v['mean_orientation_deg']/30)
            for v in metrics['objects'].values()]
    return max(ratios)+.25*sum(ratios)/len(ratios)


def summarize():
    root=B/'refinement_20261017'
    episodes={}
    for number in EPISODES:
        name=f'episode_{number:06d}'
        pilot_path=B/'runs/pilot_v2'/name/'seed_44/result.json'
        pilot=read(pilot_path)
        stages={}
        for key,prefix in [('pilot_baseline','baseline'),('pilot_refined','refined')]:
            stages[key]=dict(result=str(pilot_path),metrics=metric_summary(pilot[f'{prefix}_metrics']),
                execution=execution_summary(pilot[f'{prefix}_execution']))
        for tag in ('cem_v2','block_polish'):
            directory=root/tag/name
            result=read(directory/'result.json')
            independent=read(directory/'audit/mp_sr_independent.json')
            consistency=read(directory/'audit/search_replay_consistency.json')
            grasp=read(directory/'audit/grasp_stability_frames.json')
            video=directory/'videos/reference_baseline_refined.mp4'
            stages[tag]=dict(result=str(directory/'result.json'),metrics=metric_summary(independent),
                execution=execution_summary(result['execution']),
                initial_search_cost=result['initial']['cost'], final_search_cost=result['best']['cost'],
                search_wall_seconds=result['elapsed_seconds'], completed_generations=len(result['history']),
                search_replay_consistent=consistency['consistent'],
                independent_audit=str(directory/'audit/search_replay_consistency.json'),
                grasp_audit=str(directory/'audit/grasp_stability_frames.json'),
                grasp=grasp_summary(grasp), video=str(video), video_exists=video.is_file(),
                learned_policy=result['learned_policy'])
        collision=read(root/'reference_collision_audit'/f'{name}.json')
        collisions={kind:{k:collision[kind][k] for k in
                    ('max_penetration_mm','mean_frame_max_penetration_mm','time_of_max_s')}
                    for kind in ('object_object','object_table')}
        episodes[name]=dict(stages=stages,reference_collisions=collisions,
                            reference_collision_audit=str(root/'reference_collision_audit'/f'{name}.json'))
    counts={tag:sum(v['stages'][tag]['metrics']['mp_sr'] == 1 for v in episodes.values())
            for tag in ('pilot_baseline','pilot_refined','cem_v2','block_polish')}
    trials=[]
    families=('feedback_polish','grasp_transfer','hand_combinations','lid_collision_feedback',
              'lid_retention','contact_retention_measured','contact_retention_smooth',
              'contact_retention_fingers','cup_feedback','cup_observed_retention','lid_contact_seeking_atomic',
              'spoon_collision_feedback','lid_thumb_preposition')
    paths=[(family,path) for family in families for path in sorted((root/family).glob('episode_*/*/report.json'))]
    paths += [(path.parent.parent.name,path) for path in sorted(root.glob('block_continue_s*/episode_*/result.json'))]
    for family,path in paths:
        trial=read(path)
        episode=trial.get('episode',path.parent.parent.name)
        if episode not in episodes or 'metrics' not in trial or 'execution' not in trial:
            continue
        audit=path.parent/'audit/mp_sr_independent.json'
        grasp=path.parent/'audit/grasp_stability_frames.json'
        video=path.parent/'videos/reference_baseline_refined.mp4'
        metric=metric_summary(read(audit) if audit.exists() else trial['metrics'])
        trials.append(dict(family=family, episode=episode, variant=path.parent.name,
            episode_identity_source='report' if 'episode' in trial else 'episode_directory',
            report=str(path), metrics=metric, normalized_error_score=normalized_error_score(metric),
            execution=execution_summary(trial['execution']), independently_recomputed=audit.exists(),
            grasp=grasp_summary(read(grasp)) if grasp.exists() else None,
            video=str(video) if video.exists() else None, learned_policy=trial.get('learned_policy',False)))
    return dict(schema='dexagent.five_episode_refinement_review.v1',
        generated_utc=datetime.now(timezone.utc).isoformat(), episodes_requested=list(episodes),
        episode_count=5, full_challenge_episode_count=20, proxy_success_count=counts,
        official_leaderboard_score=None, submission_ready=False, learned_policy=False,
        scope='Five requested episodes only; floating-Sharpa MuJoCo research rollouts with reconstructed references as proxy GT.',
        interpretation=[
          'Pass requires every object mean position <3 cm AND symmetry-aware mean orientation <30 degrees over the original full horizon.',
          'Scores use the official-style shared initial registration with locally reconstructed anchors/symmetry axes; not confidential official GT.',
          'All objects and original times are retained. No per-object alignment, time warping, object actuation, or pose projection.',
          'Pilot-to-CEM changes include controller/seed changes, so that comparison does not isolate the search algorithm.',
          'Search cost is not MP-SR; lowering it can sacrifice one object or orientation without producing a stable grasp.',
          'Contact audits are geometric at 20 Hz, not force-closure certificates. Palm-relative drift may include slip, rolling, regrasp, or finger articulation.',
          'Corrected spoon diagnosis: at the captured posture, decreasing the index DIP target closes contact according to the contact Jacobian. Resetting the target to the measured angle removes preload; the negative target-minus-measured angle is not evidence of an opening command.',
          'MPC/reference optimization and deterministic replay do not constitute a learned-policy challenge artifact.'],
        episodes=episodes, additional_completed_trials=trials)


def markdown(report):
    lines=['# Five-episode refinement review','',
        f"Updated {report['generated_utc']}",'',
        '**Current block-polish result: 1/5 reconstructed-reference MP-SR passes (episode 8).** '
        'Pilot baseline and pilot MPC each pass 0/5; CEM passes 1/5. '
        'These are proxy scores for five requested episodes, not official hidden-ground-truth results. '
        'The outputs are not submission-ready learned policies.','',
        'Each cell below is full-horizon mean position error in cm / symmetry-aware orientation error in degrees. '
        'Every object must be strictly below 3 cm / 30° for the episode to pass.','',
        '| Episode | Object | Pilot baseline | CEM | Block polish | Block object pass |',
        '|---|---|---:|---:|---:|:---:|']
    for ep,entry in report['episodes'].items():
        stages=entry['stages']
        for name in stages['block_polish']['metrics']['objects']:
            values=[stages[tag]['metrics']['objects'][name] for tag in ('pilot_baseline','cem_v2','block_polish')]
            formatted=[f"{v['mean_position_cm']:.2f} / {v['mean_orientation_deg']:.2f}" for v in values]
            lines.append(f"| {ep[-6:]} | {name} | {' | '.join(formatted)} | {'yes' if values[-1]['passes'] else 'no'} |")
    lines += ['', 'Actual grasp observations from aligned videos and recorded contact audits:', '',
        '- **Episode 1:** the pot handle is missed by the left hand; the right hand tips the pot while attempting the lid. Neither intended grasp is retained. A better scalar search cost does not solve this episode.',
        '- **Episode 3:** the pot lift is retained from 3.3–11.4 s, with contact in all sampled frames and only 1.76 mm palm-relative translation drift. The spoon never lifts, so the episode still fails.',
        '- **Episode 5:** both objects are picked up and now return to the table. During a 3.7–5.75 s unsupported interval, pot contact is continuous but palm-relative translation changes 4.49 cm; spoon contact is about 90%, with 2.68 cm / 58.38° relative motion. Pot tilt and spoon rotation remain wrong.',
        '- **Episode 8:** the cup is held during a 3.75–8.85 s unsupported interval with continuous sampled contact; its maximum rise is 18.88 cm. The pot rises only 1.77 cm versus a 7.32 cm reference maximum. The mean-error metric passes even though peak motion is not fully reproduced.',
        '- **Episode 10:** the right hand lifts the beige cup temporarily, but the blue cup tips and nesting fails. The separate original-left/CEM-right combination also fails and finishes with toppled cups.',
        '', 'Contact percentages above are sampled geometry, not measured force closure. Relative object-to-palm motion includes slip, rolling, regrasp, and finger articulation.', '',
        '| Episode | CEM cost: initial → best | CEM minutes | Block cost: initial → best | Block minutes | Replay/audit | Video |',
        '|---|---:|---:|---:|---:|---|---|']
    for ep,entry in report['episodes'].items():
        c=entry['stages']['cem_v2']; p=entry['stages']['block_polish']
        okay=all(v['execution']['physics_verified'] and v['execution']['full_horizon'] and
                 v['execution']['independent_replay_passes'] and v['search_replay_consistent'] for v in (c,p))
        lines.append(f"| {ep[-6:]} | {c['initial_search_cost']:.3f} → {c['final_search_cost']:.3f} | {c['search_wall_seconds']/60:.1f} | {p['initial_search_cost']:.3f} → {p['final_search_cost']:.3f} | {p['search_wall_seconds']/60:.1f} | {'passes' if okay else 'CHECK'} | [block]({p['video']}), [CEM]({c['video']}) |")
    lines += ['', 'Times are per-run wall times; parallel jobs overlap. Pilot-to-CEM also changes the controller and starting commands, so it is not a controlled search-only ablation.', '',
        'Original-reference collision diagnostics use the unchanged object poses and the exact experiment colliders. Values are maximum / all-frame mean of the maximum penetration at each original timestamp, in mm.', '',
        '| Episode | Object–object | Object–table |', '|---|---:|---:|']
    for ep,entry in report['episodes'].items():
        parts=[]
        for kind in ('object_object','object_table'):
            c=entry['reference_collisions'][kind]
            parts.append(f"{c['max_penetration_mm']:.2f} / {c['mean_frame_max_penetration_mm']:.2f}")
        lines.append(f"| {ep[-6:]} | {' | '.join(parts)} |")
    lines += ['',
        'These overlaps diagnose incompatibility with the reconstructed collision pieces and inferred table, not proven impossibility of passing a mean-error threshold. No reference object poses or evaluation targets were changed.', '',
        'Episode 5 gap-fill counterfactual: changing only the left contact envelope improves the pot to 2.62 cm / 29.49° but causes the spoon to fall (14.64 m mean position error). Right thumb-plus-finger contact ends at 7.182 s and all right contact at 7.472 s; continuous pot–spoon contact starts at 7.474 s. Earlier coupled contacts increase, but the evidence does not prove a single collision caused release.', '',
        'Episode 10 cross-sections show hollow collision interiors, with typical inward intrusion of a few millimeters. The visual meshes themselves also intersect at the 2.8 s insertion and final 8.0 s nesting, so hull repair alone cannot make the exact reference collision-free. [Cross-sections](reference_collision_audit/episode10_cavities/reference_nesting_sections.png) and [cavity diagnostic](reference_collision_audit/episode10_cavities/individual_cup_sections.png).', '',
        'Completed later trials are listed separately below. Each family/episode row is its lowest normalized error score among complete recorded rollouts; a reported metric without independent audit is explicitly marked. These trials do not overwrite the frozen comparison above.', '',
        'Episode 5 continuation seed46 lowers cost but leaves the spoon off to the right at 14 s instead of returning it to the table. It has more spoon contact but worse spoon tracking and palm-relative drift, so the original block result remains the conservative handoff. Seeds47/48 retain the block result exactly. Episode 3 feedback improves pot pose while retaining it; its 3.77 mm relative translation drift is slightly larger than the block result\'s 1.76 mm. Cup acquisition and lid-retention trials add no new proxy passes.', '',
        'The episode 3 C2 transport blends (0.4/0.8/1.2 s) preserve identical acquisition prefixes and remove the abrupt velocity transition, but spoon errors remain 12.92–12.95 cm. They do not establish a successful retained grasp and are not selected.', '',
        'Final trials also add no passes. Extended cup acquisition deadlines fail. Observed-pickup retention also fails despite zero measured qpos discrepancy over its original pickup prefix. The final atomic lid controller detects 4.52 N finger contact at 1.492 s and switches to thumb closing in that same update, but still reports failed acquisition at 2.302 s.', '',
        'The final geometry-qualified spoon neck/bowl poses also fail acquisition: spoon error is 13.21 cm / 28.21° and 13.56 cm / 37.43°, while the unchanged left-pot commands preserve approximately 1.49 cm / 10.28°. Prepositioning the lid thumb closer does not establish capture either: the two trials have lid errors 12.37 cm / 41.67° and 28.62 cm / 46.26°. All four complete the original horizon and pass exact recorded-control replay; none is promoted to the selected handoff.', '',
        'Correction to the finger diagnosis: at the captured spoon posture, the contact Jacobian shows that decreasing the index DIP target closes the contact. The negative target-minus-measured angle is not an opening command. Resetting the target to the measured angle removes contact preload. The 0/1/2 mm measured-finger preload tests nevertheless all fail, with spoon position error about 13.01–13.02 cm; this is not established as the sole cause of failure.', '',
        '| Episode | Later trial | Per-object cm / degrees | Proxy pass | Independent audit | Video |',
        '|---|---|---|:---:|---|---|']
    best={}
    for trial in report['additional_completed_trials']:
        ex=trial['execution']
        if not (ex['physics_verified'] and ex['full_horizon'] and ex['independent_replay_passes']):
            continue
        key=(trial['episode'],trial['family'])
        if key not in best or trial['normalized_error_score']<best[key]['normalized_error_score']:
            best[key]=trial
    for key,t in sorted(best.items()):
        values='; '.join(f"{name}: {v['mean_position_cm']:.2f} / {v['mean_orientation_deg']:.2f}" for name,v in t['metrics']['objects'].items())
        video=f"[video]({t['video']})" if t['video'] else 'pending'
        lines.append(f"| {t['episode'][-6:]} | {t['family']}/{t['variant']} | {values} | {'yes' if t['metrics']['mp_sr'] else 'no'} | {'recomputed' if t['independently_recomputed'] else 'reported; pending audit'} | {video} |")
    lines += ['', 'Run `code/summarize_refinement_audits.py` with the workspace virtualenv to refresh the JSON and tables. Detailed per-object contact windows, immutable input hashes, replay checks and video paths are recorded in `metrics_review.json` and the linked audit files.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=B/'refinement_20261017')
    a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    report=summarize()
    (a.output/'metrics_review.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (a.output/'metrics_review.md').write_text(markdown(report))
    print(json.dumps(dict(proxy_success_count=report['proxy_success_count'], output=str(a.output))))


if __name__=='__main__':
    main()
