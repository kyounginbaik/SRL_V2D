"""Select a diagnostic refinement center while preserving physical ownership.

Discovery chooses one direction; the second repeat can only reject it.
Continuous task progress may become a new search center without being called
episode success or a verified learned policy/reference pair.
"""
import numpy as np


def _physical_owner_row(row,owner,minimum_support,minimum_worlds):
    if row is None:return False
    support=row['phases']['owner_hold']['hands'][owner]['worlds_with_80pct_sole_free_support']
    commands=row.get('commands',[])
    values=[support,row['official_loss'],row['official_finite_fraction'],row['official_episode_mp_sr']]
    return (np.isfinite(values).all() and row['worlds']>=minimum_worlds and row['official_finite_fraction']==1.
        and support>=minimum_support and bool(commands) and all(
            np.isfinite([c['position_max_error_m'],c['rotation_max_error_rad'],c['finger_max_error_rad']]).all()
            and c['position_max_error_m']<=1e-5 and c['rotation_max_error_rad']<=1e-4 and c['finger_max_error_rad']<=1e-4 for c in commands))


def choose_owner_center(candidates,report,owner,*,minimum_support=.9,minimum_improvement=.05,minimum_worlds=16):
    if not 0<minimum_support<=1 or not 0<minimum_improvement<1 or minimum_worlds<2:
        raise ValueError('positive independent physical and improvement gates required')
    identities=[i for i,c in enumerate(candidates) if c.get('owner_case')=='identity']
    if len(identities)!=1:raise ValueError('one exact native identity control required')
    baseline=identities[0];rows={}
    for row in report['rows']:
        key=(row['trial'],row['repeat'])
        if key in rows:raise ValueError('duplicate physical comparison row')
        rows[key]=row
    def physical(row):
        return _physical_owner_row(row,owner,minimum_support,minimum_worlds)
    def improves(i,repeat):
        old=rows.get((baseline,repeat));new=rows.get((i,repeat))
        if not physical(new) or not physical(old) or old['official_loss']<=0:return False
        if new['official_episode_mp_sr']<old['official_episode_mp_sr']:return False
        for key in ['object_position_tolerance_fraction','object_orientation_tolerance_fraction']:
            before=np.asarray(old[key]);after=np.asarray(new[key])
            if (before.shape!=after.shape or not np.isfinite(before).all() or not np.isfinite(after).all()
                    or np.any(after<np.minimum(before,minimum_support))):return False
        return new['official_loss']<=old['official_loss']*(1-minimum_improvement)
    if not all(physical(rows.get((baseline,r))) for r in range(2)):
        return dict(accepted=False,reason='native_identity_not_reproduced',baseline=baseline)
    eligible=[i for i,c in enumerate(candidates) if c.get('owner_case') in ('half','full') and improves(i,0)]
    if not eligible:return dict(accepted=False,reason='no_feasible_discovery_improvement',baseline=baseline)
    selected=min(eligible,key=lambda i:(rows[(i,0)]['official_loss'],i))
    accepted=improves(selected,1)
    return dict(accepted=accepted,reason='validated_diagnostic_owner_center' if accepted else 'fixed_choice_fails_independent_validation',
        trial=selected,baseline=baseline,full_episode_accepted=False,canonical_evidence=False,
        official_episode_mp_sr=[rows[(selected,r)]['official_episode_mp_sr'] for r in range(2)],
        repeat_relative_losses=[rows[(selected,r)]['official_loss']/rows[(baseline,r)]['official_loss'] for r in range(2)])


def choose_owner_challenger(candidates,report,owner,*,minimum_support=.9,minimum_joint_gain=.05,minimum_improvement=.05,minimum_worlds=16):
    """Explore verified joint-task progress without replacing the incumbent.

    Component tradeoffs may create a challenger ONLY if full official task
    success improves. Already solved objects and physical ownership remain
    constraints. Discovery fixes the branch; held-out may only reject it.
    """
    if not 0<minimum_support<=1 or not 0<minimum_joint_gain<1 or not 0<minimum_improvement<1 or minimum_worlds<2:
        raise ValueError('positive physical, joint-progress and improvement gates required')
    identities=[i for i,c in enumerate(candidates) if c.get('owner_case')=='identity']
    if len(identities)!=1:raise ValueError('one exact native identity control required')
    baseline=identities[0];rows={}
    for row in report['rows']:
        key=(row['trial'],row['repeat'])
        if key in rows:raise ValueError('duplicate physical comparison row')
        rows[key]=row
    def physical(row):return _physical_owner_row(row,owner,minimum_support,minimum_worlds)
    if not all(physical(rows.get((baseline,r))) for r in range(2)):
        return dict(accepted=False,reason='native_identity_not_reproduced',baseline=baseline)
    def eligible(i,repeat):
        old=rows.get((baseline,repeat));new=rows.get((i,repeat))
        if not physical(new) or not physical(old):return False
        before=np.asarray(old['object_joint_tolerance_fraction']);after=np.asarray(new['object_joint_tolerance_fraction'])
        if (before.shape!=after.shape or not np.isfinite(before).all() or not np.isfinite(after).all()
                or np.any(before<0) or np.any(before>1) or np.any(after<0) or np.any(after>1)
                or np.any(after[before>=minimum_support]<minimum_support)):return False
        return (new['official_episode_mp_sr']>=old['official_episode_mp_sr']+minimum_joint_gain
                and new['official_loss']<=old['official_loss']*(1-minimum_improvement))
    discovery=[i for i,c in enumerate(candidates) if c.get('owner_case') in ('half','full') and eligible(i,0)]
    if not discovery:return dict(accepted=False,reason='no_physical_joint_progress_challenger',baseline=baseline)
    selected=min(discovery,key=lambda i:(-rows[(i,0)]['official_episode_mp_sr'],rows[(i,0)]['official_loss'],i))
    accepted=eligible(selected,1)
    return dict(accepted=accepted,reason='validated_diagnostic_challenger' if accepted else 'fixed_challenger_fails_independent_validation',
        trial=selected,baseline=baseline,branch_only=True,incumbent_replaced=False,full_episode_accepted=False,canonical_evidence=False,
        official_episode_mp_sr=[rows[(selected,r)]['official_episode_mp_sr'] for r in range(2)])
