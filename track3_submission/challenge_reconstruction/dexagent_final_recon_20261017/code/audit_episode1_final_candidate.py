"""Hash-bound evidence for the repaired episode-1 learned physical controller.

This audit keeps original aligned and declared-frame metrics separate. It does
not promote a custom evaluator result to an official leaderboard result.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

B = Path(__file__).resolve().parents[1]
REPAIR = B / 'refinement_20261018/episode_000001/reconstruction_repair'
ROOT = REPAIR / 'patch_height094'
RUN = ROOT / 'learned_carry_refinement_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    base = load(RUN / 'evaluation/result.json')
    training = load(RUN / 'training_report.json')
    joint = load(RUN / 'evaluation/audit/full_joint_limits.json')
    support = load(RUN / 'evaluation/audit/object_support.json')
    video = load(RUN / 'videos/validation.json')
    assert base['metrics']['reference_proxy_mp_sr'] == 1
    assert base['execution']['physics_verified']
    assert base['execution']['causal_policy_action_reproduction_max_error'] == 0
    assert base['execution']['independent_control_replay']['max_replay_qpos_error'] == 0
    assert all(v['max_command_violation'] == 0 for v in joint['joints'].values())
    assert video['decoded_frames'] == 168
    source = Path(base['metrics']['provenance']['reference'])
    assert sha(source) == base['metrics']['provenance']['reference_sha256']
    assert sha(ROOT / 'repaired_adapter/metric_geometry.json') == sha(B / 'episodes/episode_000001/metric_geometry.json')
    assert sha(ROOT / 'repaired_adapter/tracking_reference.npz') == sha(B / 'episodes/episode_000001/tracking_reference.npz')

    arrays = ('qpos', 'qvel', 'substep_qpos', 'substep_qvel', 'substep_ctrl', 'substep_time_s')
    crosshost = {}
    with np.load(RUN / 'evaluation/execution/trajectory.npz', allow_pickle=False) as a:
        for host in ('node01', 'node200'):
            other = load(RUN / f'crosshost_{host}/result.json')
            assert other['policy_checkpoint_sha256'] == base['policy_checkpoint_sha256']
            assert other['execution']['hashes']['scene'] == base['execution']['hashes']['scene']
            assert other['metrics']['reference_proxy_mp_sr'] == 1
            with np.load(RUN / f'crosshost_{host}/execution/trajectory.npz', allow_pickle=False) as c:
                errors = {key: float(np.max(np.abs(a[key]-c[key]))) for key in arrays}
            crosshost[host] = dict(result=str(RUN / f'crosshost_{host}/result.json'),
                reference_proxy_mp_sr=other['metrics']['reference_proxy_mp_sr'],
                maximum_array_differences=errors, bitwise_identical=all(v == 0 for v in errors.values()),
                independent_replay=other['execution']['independent_control_replay'])

    end = [row for row in support['rows'] if row['time_s'] >= 16.3]
    end_support = dict(window_seconds=[16.3, 16.7], sampled_at_hz=20,
        maximum_hand_object_normal_force_n=max(sum(sum(v['fingers'].values()) for v in row['objects'].values()) for row in end),
        minimum_pot_table_normal_force_n=min(row['objects']['white_pot']['support'].get('table', 0) for row in end),
        minimum_lid_pot_normal_force_n=min(row['objects']['white_pot_lid']['support'].get('white_pot', 0) for row in end),
        full_500hz_search_maximum_object_speed_m_s=training['best']['max_final_object_speed_m_s'])
    assert end_support['maximum_hand_object_normal_force_n'] == 0
    assert end_support['minimum_pot_table_normal_force_n'] > .5
    assert end_support['minimum_lid_pot_normal_force_n'] > .1
    top = sorted((dict(name=name, **v) for name, v in joint['joints'].items() if v['max_actual_violation_deg'] is not None),
        key=lambda row: -row['max_actual_violation_deg'])[:5]
    rejected = load(ROOT / 'learned_carry_precision6/evaluation/result.json')
    report = dict(episode='episode_000001', updated_utc=datetime.now(timezone.utc).isoformat(),
        status='Verified learned physical candidate; packaging and official harness acceptance are separate',
        ready_for_official_submission=False,
        evaluation=str(RUN / 'evaluation/result.json'), checkpoint=str(RUN / 'policy.npz'),
        checkpoint_sha256=sha(RUN / 'policy.npz'), trained_actor=True,
        method='Phase/proprioception radial-basis imitation feedback actor, with robot-only finger margins and physically optimized phase weights; not FlashSAC',
        original_aligned_proxy=base['metrics']['official_style_reference_proxy'],
        declared_source_frame_proxy=base['metrics']['declared_frame_reference_proxy'],
        original_reference_unchanged=True, original_reference_sha256=sha(source),
        original_metric_geometry_unchanged=True, actor_has_no_runtime_object_reference=True,
        immutable_repaired_scene_sha256=base['execution']['hashes']['scene'],
        reconstructed_scene_changed_from_original=True,
        reconstruction_evidence=str(REPAIR / 'status.json'),
        repaired_mesh_export_audit=str(ROOT / 'repaired_adapter/mesh_export_audit.json'),
        full_horizon_seconds=[0.,16.7], policy_calls=base['execution']['policy_inference_calls'],
        independent_replay=base['execution']['independent_control_replay'],
        causal_policy_action_reproduction_max_error=base['execution']['causal_policy_action_reproduction_max_error'],
        crosshost=crosshost, final_support=end_support,
        all_commands_within_source_joint_limits=True, actual_soft_limit_extrema=top,
        physical_quality_review='Largest actual soft-limit excursion is 1.577 degrees at the right index PIP during second lid acquisition; it is disclosed, with original joint bounds and actuator limits retained.',
        video=video, video_sha256=sha(video['video']),
        visually_reviewed_seconds=[2.4,3.6,4.5,6.,9.,12.,13.5,14.5,15.5,16.7],
        visual_observations='Actual recorded physics completes lid removal, pot motion, lid replacement and stable clear release. Original lid visual mesh remains visibly nonmanifold/speckled; no render-only geometry replacement was used.',
        rejected_precision_variant=dict(path=str(ROOT / 'learned_carry_precision6'),
            decimal_places=6,reference_proxy_mp_sr=rejected['metrics']['reference_proxy_mp_sr'],
            reason='Micrometer-scale command rounding changed contact acquisition and worsened tracking and joint quality; preserve the exact unrounded checkpoint/output contract.'),
        limitations=['Local reconstructed poses are proxy targets, not confidential ground truth.',
            'Custom MuJoCo physical evaluator; this is not an official leaderboard or Isaac harness score.',
            'Deterministic single-episode controller; tested portability is not robustness to arbitrary initial states.',
            'Pot visual bowl height was source-supported/versioned at 0.94 scale; original mass/inertia retained.',
            'Source lid topology and triangle-patch collision approximation are disclosed reconstruction limitations.'])
    (RUN / 'final_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    status = load(REPAIR / 'status.json')
    status['status'] = 'Learned physical controller passes unchanged original and declared-frame proxy, exact replay and cross-host checks; packaging remains separate'
    status['accepted_for_submission'] = False
    status['updated_utc'] = report['updated_utc']
    status['active_search'] = None
    status['current_learned_policy'] = dict(checkpoint=report['checkpoint'], evaluation=report['evaluation'],
        final_validation=str(RUN / 'final_validation.json'), original_aligned_proxy=report['original_aligned_proxy'],
        declared_source_frame_proxy=report['declared_source_frame_proxy'], final_support=end_support,
        actual_joint_limit_maximum_degrees=joint['maximum_angular_violation_deg'], crosshost=crosshost,
        video=video['video'], learned_method=report['method'])
    status['metric_interpretation'] = 'Both metric views retain original immutable targets. The original common initial alignment has a documented 13.438-degree tilt bias. This candidate now passes both views through robot-control refinement; the metric formula and source targets were not changed.'
    (REPAIR / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(dict(report=str(RUN / 'final_validation.json'), crosshost=crosshost, final_support=end_support)))


if __name__ == '__main__':
    main()
