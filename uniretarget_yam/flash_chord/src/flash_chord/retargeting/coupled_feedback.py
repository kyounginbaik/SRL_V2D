"""Choose a diagnostic coupled-refinement center from independent repetitions.

Discovery ranks feasible forward directions. Held-out data can reject that
fixed choice, never choose an alternative. This does not accept a full episode
reference or certify a deployable learned policy.
"""
import numpy as np


def choose_coupled_center(candidates, report, *, minimum_retention=.9, minimum_tracking=.9,
                          minimum_improvement=.1, minimum_worlds=16):
    if (not 0<minimum_retention<=1 or not 0<minimum_tracking<=1
            or not 0<minimum_improvement<1 or minimum_worlds<2):
        raise ValueError('positive physical and independent improvement gates required')
    identities=[i for i,c in enumerate(candidates) if c.get('operation')=='coupled_command_transport' and c.get('coupled_case')=='identity']
    if len(identities)!=1:raise ValueError('one paired native identity control required')
    baseline=identities[0];rows={}
    for row in report['rows']:
        key=(row['trial'],row['repeat'])
        if key in rows:raise ValueError('duplicate physical comparison row')
        rows[key]=row
    def executed(row):
        commands=row.get('command_execution',[])
        return bool(commands) and all(
            np.isfinite([c['command_position_max_error_m'],c['command_rotation_max_error_rad'],c['command_finger_max_error_rad']]).all()
            and c['command_position_max_error_m']<=1e-5 and c['command_rotation_max_error_rad']<=1e-4
            and c['command_finger_max_error_rad']<=1e-4 for c in commands)
    for repeat in range(2):
        row=rows.get((baseline,repeat))
        if (row is None or not executed(row) or row['prefix_finite_fraction']!=1.
                or row['receiver_retention']['supported_retention_proxy_fraction']<minimum_retention):
            return dict(accepted=False,reason='identity_control_not_reproduced',baseline=baseline)
    def eligible(i,repeat):
        row=rows.get((i,repeat))
        if row is None:return False
        values=[row['prefix_loss'],row['prefix_tracking_fraction'],row['prefix_finite_fraction'],
            row['receiver_retention']['supported_retention_proxy_fraction']]
        return (executed(row) and np.isfinite(values).all() and row['worlds']>=minimum_worlds
            and row['prefix_finite_fraction']==1. and row['prefix_tracking_fraction']>=minimum_tracking
            and row['receiver_retention']['supported_retention_proxy_fraction']>=minimum_retention)
    discovery=[]
    for i,c in enumerate(candidates):
        if (c.get('operation')!='coupled_command_transport' or c.get('coupled_case') not in ('half','full')
                or not eligible(i,0)):continue
        before=rows.get((baseline,0),{}).get('prefix_loss',np.nan);after=rows[(i,0)]['prefix_loss']
        if not np.isfinite(before) or before<=0 or after>before*(1-minimum_improvement):continue
        discovery.append((after,i))
    if not discovery:return dict(accepted=False,reason='no_feasible_discovery_direction',baseline=baseline)
    _,selected=min(discovery)
    reason='validated_diagnostic_center'
    if not eligible(selected,1):reason='fixed_choice_fails_held_out_physical_gate'
    for repeat in range(2):
        old=rows.get((baseline,repeat));new=rows.get((selected,repeat))
        if (old is None or new is None or old['worlds']<minimum_worlds or old['prefix_finite_fraction']!=1.
                or not np.isfinite(old['prefix_loss']) or old['prefix_loss']<=0
                or new['prefix_loss']>old['prefix_loss']*(1-minimum_improvement)):
            reason='fixed_choice_fails_independent_improvement'
    return dict(accepted=reason=='validated_diagnostic_center',reason=reason,trial=selected,baseline=baseline,
        discovery_selection='minimum task loss among physically feasible forward directions; held-out can only reject',
        prefix_only=True,full_episode_accepted=False,
        repeat_relative_losses=[rows[(selected,r)]['prefix_loss']/rows[(baseline,r)]['prefix_loss']
            if (selected,r) in rows and (baseline,r) in rows and np.isfinite(rows[(selected,r)]['prefix_loss'])
            and np.isfinite(rows[(baseline,r)]['prefix_loss']) and rows[(baseline,r)]['prefix_loss']>0 else None for r in range(2)])
