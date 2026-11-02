"""Physical acquisition statistics and independently checked exploration branches."""
import numpy as np


def acquisition_evaluation_windows(manifest,frames):
    """One shared diagnostic window after every proposed closure/lift event.

    Derived from reference timing before inspecting outcomes; official scoring
    always uses the full unchanged episode. Advancing/delaying one candidate
    never lets it choose its own easier holding interval.
    """
    seg=manifest['grasp_probe_segment'];fps=manifest['fps']
    base=manifest.get('contact_search_windows',dict(closure=[seg['start'],seg['start']+round(.6*fps)],
        hold=[seg['start']+round(.6*fps),min(seg['end']+1,seg['start']+round(2.6*fps))]))
    if 'digit_timing_order' not in manifest:return {key:list(value) for key,value in base.items()}
    settle=max(1,round(.25*fps));last=base['hold'][0]-settle
    for candidate in manifest['candidates']:
        if 'wrist_event' in candidate:last=max(last,candidate['wrist_event']+max(0,candidate['wrist_shift_frames']))
        for spec in candidate.get('digit_time_maps',[]):
            if spec['digit'] not in (manifest['digit_timing_order']['early'],manifest['digit_timing_order']['late']):
                raise ValueError('timing probe digit is absent from measured contact order')
            digit_key='early' if spec['digit']==manifest['digit_timing_order']['early'] else 'late'
            source_time=manifest['digit_timing_order'][digit_key+'_frame']
            last=max(last,int(np.ceil(source_time))+max(0,spec['shift']))
    start=int(last+settle);end=min(frames,seg['end']+1,start+round(2*fps))
    if not seg['start']<start<end:raise ValueError('no common post-event holding window')
    return dict(closure=[seg['start'],start],hold=[start,end])


def acquisition_statistics(data, trial, hand, object_id, closure, hold):
    ids=np.flatnonzero(data['world_reference_group']==trial)
    sides=list(data['physical_contact_sides']);h=sides.index(hand)
    a,b=closure;c,d=hold
    if len(ids)<2 or not 0<=a<b<=c<d<=len(data['object_pose']):
        raise ValueError('replicated worlds and ordered nonempty acquisition windows required')
    # Select the scalar object first: separated NumPy advanced indices can
    # silently transpose frame/world axes, even when their lengths happen to match.
    counts=data['hand_object_contact_count'][...,object_id][:,ids]
    load=data['hand_object_support_mg'][...,object_id][:,ids]
    external=data['object_external_force_norm_sum_n'][...,object_id][:,ids]
    invalid=data['object_external_invalid_force_count'][...,object_id][:,ids]
    environment=data['hand_environment_force_norm_sum_n'][:,ids,h]
    invalid_environment=data['hand_environment_invalid_force_count'][:,ids,h]
    mg=data['object_mass_kg'][ids,object_id]*np.linalg.norm(data['gravity_w'][ids],axis=-1)
    poses=data['object_pose'][:,:,object_id,:7][:,ids]
    finite=(np.isfinite(load).all() and np.isfinite(external).all() and np.isfinite(environment).all()
        and np.isfinite(poses).all() and (np.linalg.norm(poses[...,3:],axis=-1)>1e-8).all()
        and np.isfinite(mg).all() and (mg>0).all() and (invalid==0).all() and (invalid_environment==0).all())
    if not finite:return dict(trial=int(trial),worlds=len(ids),finite=False)
    other=[j for j in range(len(sides)) if j!=h]
    free=((counts[:,:,h]>0)&(load[:,:,h]>.5)&~(counts[:,:,other]>0).any(-1)&(external/mg<.1))
    return dict(trial=int(trial),worlds=len(ids),finite=True,
        contact_fraction=float((counts[a:b,:,h]>0).mean()),
        environment_mean_mg=float((environment[a:b]/mg).mean()),
        environment_p95_mg=float(np.quantile(environment[a:b]/mg,.95)),
        free_hold_fraction=float(free[c:d].mean()),
        stable_world_fraction=float((free[c:d].mean(0)>=.8).mean()))


def choose_acquisition_branch(candidates, rows, *, minimum_worlds=16, minimum_force_improvement=.15):
    """Choose a clearance/contact exploration branch, never an episode success.

    Mean force counts every sample. A zero median cannot conceal intermittent
    large collisions. Discovery chooses one candidate; held-out only rejects it.
    """
    identities=[i for i,c in enumerate(candidates) if c.get('grasp_probe')=='identity']
    if len(identities)!=1 or not 0<minimum_force_improvement<1:raise ValueError('one identity and positive improvement required')
    baseline=identities[0];lookup={}
    for row in rows:
        key=(row['trial'],row['repeat'])
        if key in lookup:raise ValueError('duplicate acquisition evidence row')
        lookup[key]=row
    def valid(row):
        return row is not None and row['finite'] and row['worlds']>=minimum_worlds and np.isfinite([
            row['contact_fraction'],row['environment_mean_mg'],row['environment_p95_mg'],row['stable_world_fraction']]).all()
    def improves(i,r):
        old=lookup.get((baseline,r));new=lookup.get((i,r))
        return (valid(old) and valid(new) and old['environment_mean_mg']>0
            and new['contact_fraction']>=old['contact_fraction']+.02
            and new['stable_world_fraction']>=old['stable_world_fraction']
            and new['environment_mean_mg']<=old['environment_mean_mg']*(1-minimum_force_improvement))
    eligible=[i for i,c in enumerate(candidates) if c.get('probe_axis') in (0,1,2) and improves(i,0)]
    if not eligible:return dict(accepted=False,reason='no_clearance_and_contact_exploration_branch',baseline=baseline)
    selected=min(eligible,key=lambda i:(-lookup[(i,0)]['stable_world_fraction'],-lookup[(i,0)]['contact_fraction'],lookup[(i,0)]['environment_mean_mg'],i))
    accepted=improves(selected,1)
    return dict(accepted=bool(accepted),reason='validated_clearance_contact_exploration_branch' if accepted else 'fixed_branch_failed_validation',
        trial=selected,baseline=baseline,branch_only=True,incumbent_replaced=False,
        full_episode_accepted=False,canonical_success_verified=False)


def digit_contact_timing(data,trial,hand,object_id,window,digits):
    """Measure force-bearing digit overlap; contact alone is not a grasp certificate."""
    ids=np.flatnonzero(data['world_reference_group']==trial);h=list(data['physical_contact_sides']).index(hand)
    a,b=window;slots=np.flatnonzero((data['complete_slot_hand_ids']==h)&(data['complete_slot_object_ids']==object_id))
    if not 0<=a<b<=len(data['complete_link_force_on_object_w']):
        raise ValueError('nonempty digit contact window required')
    forces=data['complete_link_force_on_object_w'][a:b][:,ids][:,:,slots]
    active=data['complete_link_contact_active'][a:b][:,ids][:,:,slots]
    if not len(ids) or not len(slots) or not np.isfinite(forces).all():raise ValueError('finite per-link force evidence required')
    names=[str(data['complete_link_names'][s]).removeprefix(hand+'_') for s in slots]
    bearing=active&(np.linalg.norm(forces,axis=-1)>1e-4);by_digit=[];fractions={};centers={}
    for digit in digits:
        chosen=[j for j,name in enumerate(names) if name.startswith(digit+'_')]
        contact=bearing[:,:,chosen].any(-1) if chosen else np.zeros(bearing.shape[:2],bool)
        by_digit.append(contact);fractions[digit]=float(contact.mean())
        frames=np.nonzero(contact)[0];centers[digit]=float(np.median(frames)+a) if len(frames) else None
    count=np.stack(by_digit,axis=-1).sum(-1)
    return dict(digit_contact_fraction=fractions,digit_contact_frame_median=centers,
        observed_digit_count=sum(v>=.02 for v in fractions.values()),two_digit_contact_fraction=float((count>=2).mean()))


def choose_digit_contact_branch(candidates,rows,*,minimum_worlds=16):
    """Explore better multidigit contact without accepting an unheld reference."""
    baseline=next(i for i,c in enumerate(candidates) if c.get('contact_search')=='branch_identity')
    lookup={(r['trial'],r['repeat']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('duplicate digit contact evidence')
    def improves(i,r):
        old=lookup.get((baseline,r));new=lookup.get((i,r))
        return (old is not None and new is not None and old['finite'] and new['finite']
            and min(old['worlds'],new['worlds'])>=minimum_worlds
            and np.isfinite([row[key] for row in (old,new) for key in ['contact_fraction','environment_mean_mg','stable_world_fraction','two_digit_contact_fraction']]).all()
            and new['observed_digit_count']>=max(2,old['observed_digit_count'])
            and new['contact_fraction']>=old['contact_fraction']+.1
            and new['environment_mean_mg']<=old['environment_mean_mg']*1.05
            and new['stable_world_fraction']>=old['stable_world_fraction'])
    eligible=[i for i,c in enumerate(candidates) if c.get('placement_offset_w') and np.linalg.norm(c['placement_offset_w'])>0 and improves(i,0)]
    if not eligible:return dict(accepted=False,reason='no_multidigit_contact_exploration_branch')
    selected=min(eligible,key=lambda i:(-lookup[(i,0)]['two_digit_contact_fraction'],-lookup[(i,0)]['contact_fraction'],lookup[(i,0)]['environment_mean_mg'],i))
    accepted=improves(selected,1)
    return dict(accepted=bool(accepted),reason='validated_multidigit_contact_exploration_branch' if accepted else 'fixed_digit_branch_failed_validation',
        trial=selected,baseline=baseline,branch_only=True,incumbent_replaced=False,full_episode_accepted=False,canonical_success_verified=False)
