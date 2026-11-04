"""Summarize bounded seeking, closure timing, table blockage and replay identity."""
from pathlib import Path
import argparse,json
import numpy as np
B=Path(__file__).resolve().parents[1]


def run(trial, prefix_cutoff=None):
    path=Path(trial)
    report=json.loads((path/'report.json').read_text())
    seeking=json.loads((path/'execution/contact_seeking_audit.json').read_text())
    events=json.loads((path/'execution/acquisition_events.json').read_text())['events']
    samples=seeking['samples'];config=seeking['config']
    baseline=B/'refinement_20261017/lid_collision_feedback/episode_000001/candidate_00_open10/execution/trajectory.npz'
    old=np.load(baseline);new=np.load(path/'execution/trajectory.npz')
    cutoff=(config['acquisition_start_s']+config['approach_s']
            if prefix_cutoff is None else float(prefix_cutoff))
    mask=new['substep_time_s']<=cutoff+1e-10
    prefix={k:bool(np.array_equal(new[k][mask],old[k][mask])) for k in ('substep_qpos','substep_qvel','substep_ctrl')}
    dt=float(np.median(np.diff(new['substep_time_s'])))
    contact=next((e['time_seconds'] for e in events if e['phase']=='seeking_stopped_at_contact'),None)
    close=next((e['time_seconds'] for e in events if e['phase']=='close_thumb'),None)
    closing_end=None if close is None else close+config['thumb_closure_s']
    force_threshold=config['min_contact_force_n']
    loss=next((s['time_seconds'] for s in samples if contact is not None and s['time_seconds']>contact
               and s['measured_index_normal_force_n']<force_threshold),None)
    closure_samples=[] if close is None else [s for s in samples if close<=s['time_seconds']<closing_end]
    summary=dict(events=events,physics_verified=report['execution']['physics_verified'],
        physical_initial_state_bitwise_unchanged=bool(np.array_equal(new['initial_qpos'],old['initial_qpos'])),
        prefix_through_seconds=cutoff,prefix_bitwise_equal=prefix,
        maximum_extra_path_m=max(s['accumulated_additional_travel_m'] for s in samples),
        maximum_extra_seek_speed_m_s=max(np.linalg.norm(s['additional_world_step_m']) for s in samples)/dt,
        maximum_hand_table_normal_force_n=max(s['hand_table']['total_normal_force_n'] for s in samples),
        maximum_index_normal_force_n=max(s['measured_index_normal_force_n'] for s in samples),
        first_force_contact_s=contact,thumb_closure_start_s=close,thumb_closure_complete_nominal_s=closing_end,
        first_index_contact_loss_s=loss,
        index_contact_fraction_during_thumb_closure=(float(np.mean([s['measured_index_normal_force_n']>=force_threshold for s in closure_samples])) if closure_samples else None),
        metrics=report['metrics']['official_style_reference_proxy'],
        retention={k:v for k,v in report['retention'].items() if k!='table_contact_and_height_samples'})
    assert summary['maximum_extra_path_m']<=.010000001
    assert summary['maximum_extra_seek_speed_m_s']<=.020000001
    (path/'controlled_test_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('trial',type=Path)
    parser.add_argument('--prefix-cutoff',type=float,default=None)
    args=parser.parse_args()
    print(json.dumps(run(args.trial,args.prefix_cutoff),indent=2))
