"""Reset-compatible, reference-conditioned lower-policy adaptation banks."""
import numpy as np


def validate_bank(q, target, baseline_q, baseline_target):
    q=np.asarray(q,np.float32);target=np.asarray(target,np.float32)
    base=np.asarray(baseline_q,np.float32);bt=np.asarray(baseline_target,np.float32)
    if (q.ndim!=3 or target.ndim!=3 or q.shape[:2]!=target.shape[:2]
            or q.shape[1:]!=base.shape or target.shape[1:]!=bt.shape or len(q)<1
            or not np.isfinite(q).all() or not np.isfinite(target).all()):
        raise ValueError('finite bank with baseline-matching frame/joint dimensions required')
    for frame in (0,-1):
        if (not np.allclose(q[:,frame],base[frame],atol=1e-6,rtol=0)
                or not np.allclose(target[:,frame],bt[frame],atol=1e-6,rtol=0)):
            raise ValueError('bank must preserve baseline reset and terminal joint configurations/targets')
    controls=np.flatnonzero(np.all(q==base,axis=(1,2))&np.all(target==bt,axis=(1,2)))
    if len(controls)!=1:raise ValueError('bank must contain exactly one unchanged control')
    return q,target,int(controls[0])


def bank_assignment(groups,worlds,control,*,control_only=False):
    """Deterministic near-equal coverage, with every reference represented."""
    if not 0<=control<groups or worlds<groups:raise ValueError('all bank groups need at least one world')
    if control_only:return np.full(worlds,control,int)
    return np.arange(worlds,dtype=int)%groups


def supported_shortlist(trials,*,maximum=4,minimum_load=.5):
    """Choose repeat-supported hypotheses for a NEW release experiment.

    This is proposal selection, not acceptance or a grasp certificate. Every
    selected controller/reference combination must have at least two distinct
    repeats, finite transfer stages and receiver half-weight coverage above the
    threshold in every repeat. Later instability remains a required final guard.
    """
    if maximum<1 or not 0<=minimum_load<=1:raise ValueError('valid shortlist size/load threshold required')
    groups={}
    for row in trials:
        if row['condition']!='intervention':continue
        if row['component'] not in ('wrist','fingers','all'):raise ValueError('unknown controller component')
        key=(row['component'],int(row['group']));groups.setdefault(key,{})
        if row['repeat'] in groups[key]:raise ValueError('duplicate experimental repeat')
        groups[key][row['repeat']]=row
    eligible=[]
    for (component,group),repeats in groups.items():
        rows=list(repeats.values())
        if len(rows)<2 or any(r['stage_finite']!=1. or not np.isfinite(r['receiver_half_weight_fraction'])
                or r['receiver_half_weight_fraction']<minimum_load for r in rows):continue
        eligible.append(dict(component=component,group=group,repeats=len(rows),
            minimum_loaded_fraction=min(r['receiver_half_weight_fraction'] for r in rows),
            minimum_contact_fraction=min(r['receiver_contact_fraction'] for r in rows)))
    return sorted(eligible,key=lambda r:(-r['minimum_loaded_fraction'],-r['minimum_contact_fraction'],r['component'],r['group']))[:maximum]
