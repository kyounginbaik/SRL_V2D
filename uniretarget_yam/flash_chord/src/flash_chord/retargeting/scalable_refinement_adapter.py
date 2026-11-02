"""Failure-driven adapter for the floating-hand parallel editor.

Installed by scripts/install_scalable_refinement.py into a fresh runtime. The
algorithm lives in the repository; existing sealed experiments are untouched.
"""
from dataclasses import replace
import glob
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_refinement import (
    RefinementConfig, anchor_hand, blend_hand, extract_segments, generate_phase_edits,
    phase_mask, project_edit, squeeze_hand,
    grasp_transition_mask,grasp_transition,
)
from .contact_refinement_adapter import channels, reference_contacts
from .scalable_refinement import FeedbackConfig, feedback_costs, independent_gate, phase_feedback, official_scores


def config(editor):
    return FeedbackConfig(**{k: editor.options['scalable_' + k] for k in FeedbackConfig.__dataclass_fields__
                             if 'scalable_' + k in editor.options})


def proposal_budget(editor):
    worlds=getattr(editor,'upper_worlds',None)
    replicas=int(editor.options.get('replicas',1));baseline=int(editor.options.get('baseline_replicas',0))
    return max(1,(int(worlds)-baseline)//replicas) if worlds is not None and replicas>0 else None


def grasp_family(path,prototype_names=(),*,selection=None,segment_key=None):
    if selection and segment_key and selection.get(segment_key):
        return str(selection[segment_key]).removeprefix('dexonomy_')
    stem=Path(path).stem
    names=[str(n).removeprefix('dexonomy_') for n in prototype_names if str(n)]
    matches=[n for n in names if '_'+n+'_' in '_'+stem+'_']
    return max(matches,key=len) if matches else re.sub(r'(_sl\d+)?_sq[^_]+_b\d+$','',stem)


def _phase(editor, packed=None):
    segment = getattr(editor, '_scalable_segment', None)
    if segment is None:
        return None
    sides = list(getattr(editor, '_palm_sides', [h.side for h in editor.projector.layout.hands]))
    hand = (packed if packed is not None else editor.binding).hand(segment.hand)
    planned = np.concatenate([hand.wrist_pos_w, hand.wrist_quat_w[:, [1, 2, 3, 0]]], axis=1)
    if packed is not None:
        planned = planned.reshape(len(editor.binding.joint_q), editor.upper_worlds, 7)
    planned = planned[:-1]
    result = phase_feedback(editor._object_pose, editor._palm_pose, sides.index(segment.hand),
                            segment, editor._scalable_segments, editor.baseline.fps, config(editor),
                            reference_palms=planned)
    if getattr(editor, '_scalable_mode', None) is not None:
        result['mode'] = editor._scalable_mode
    return result


def select_window(editor, learner, step):
    from .parallel_rollout import rollout_candidates
    q, t = editor.binding.joint_q, editor.binding.joint_target
    observer=None;old_observer=getattr(editor,'physical_response_observer',None)
    if editor.options.get('scalable_causal_phases'):
        from .physical_contact_feedback import ContactFeedbackRecorder
        observer=ContactFeedbackRecorder()
        def observe(env,frame):
            observer(env,frame)
            if old_observer is not None:old_observer(env,frame)
        editor.physical_response_observer=observe
    try:
        rollout_candidates(editor, learner, np.repeat(q[None], editor.upper_worlds, 0),
                           np.repeat(t[None], editor.upper_worlds, 0))
    finally:
        if observer is not None:
            if old_observer is not None:editor.physical_response_observer=old_observer
            else:del editor.physical_response_observer
    if editor._object_pose.shape[0] != len(q) - 1:
        raise ValueError('scalable refinement requires full-horizon command-indexed rollouts')
    frames = getattr(editor, '_rollout_reference_frames', np.arange(len(q) - 1))
    if not np.array_equal(frames, np.arange(len(q) - 1)):
        raise ValueError('partial or shifted feedback cannot drive scalable refinement')
    segments = extract_segments(reference_contacts(editor.setup.reference, len(q)), editor.baseline.fps)
    editor._scalable_segments = segments
    editor._scalable_segment = None
    editor._scalable_mode = None
    reports = []
    for segment in segments:
        editor._scalable_segment = segment
        report = _phase(editor)
        reports.append({k: v for k, v in report.items() if k not in ('retained', 'local_loss')})
        if not report['passed']:
            break  # Fix acquisition before interpreting downstream falling poses.
    else:
        editor._scalable_segment = None
    editor._scalable_anchor=None;editor._scalable_causal_info=None
    if observer is not None and segments:
        from .physical_response import select_causal_phase,supported_object_anchor
        from .contact_refinement import find_giver
        chosen,causal=select_causal_phase(editor._object_pose,segments,editor.baseline.fps)
        giver=find_giver(chosen,segments,editor.baseline.fps) if chosen is not None and causal.get('mode')=='handover' else None
        if giver is not None:
            contacts=observer.arrays();h=contacts['physical_contact_sides'].tolist().index(giver.hand);o=chosen.object_id
            support=(contacts['hand_object_contact_count'][:,:,h,o]>0)&(contacts['hand_object_support_mg'][:,:,h,o]>.5)
            anchor,evidence=supported_object_anchor(editor._object_pose,support,chosen,causal)
            causal['anchor_evidence']=evidence
            if anchor is not None:
                editor._scalable_segment=chosen;editor._scalable_anchor=anchor
                editor._scalable_causal_info=causal
                report=_phase(editor)
                reports.append({k:v for k,v in report.items() if k not in ('retained','local_loss')})
    editor._scalable_phase = _phase(editor)
    editor._scalable_mode = editor._scalable_phase['mode'] if editor._scalable_phase else None
    editor._scalable_reports = reports
    attempts = getattr(editor, '_scalable_attempts', {})
    # The online callback clears _last_record before entering a new round.
    # Keep an independent completed-round record for retry/cycle decisions.
    previous = getattr(editor, '_scalable_previous_record', None) or getattr(editor, '_last_record', {}) or {}
    if previous.get('scalable_feedback') and not previous.get('accepted'):
        key = previous['scalable_feedback'].get('segment')
        if key:
            attempts[key] = attempts.get(key, 0) + 1
    editor._scalable_attempts = attempts
    selected = editor._scalable_segment
    mask = np.zeros(len(q))
    info = dict(detect='physical_feedback', phases=reports, window=[1, 1], reason='all_contact_phases_pass')
    official = official_scores(editor._object_pose, config(editor))
    info['mp_sr'] = float(official['passed'].mean())
    if selected is None and not official['passed'].all():
        info['reason'] = 'task_error_outside_actionable_contact_phases'
    if selected is not None:
        mask, bounds = phase_mask(len(q), selected.start, selected.end, editor.baseline.fps)
        causal=editor._scalable_causal_info
        if causal is not None:
            from .refinement_window import window_mask
            bounds=tuple(causal['edit_window']);mask=window_mask(len(q),bounds,editor.baseline.fps)
        report = editor._scalable_phase
        info = dict(detect='physical_feedback', window=list(bounds), key=selected.key,
                    mode=report['mode'], retention_fraction=report['retention_fraction'], phases=reports,
                    mp_sr=float(official['passed'].mean()),
                    unsuccessful_rounds=attempts.get(selected.key, 0),
                    needs_lower_skill=attempts.get(selected.key, 0) >= 3 and report['mode'] != 'precision')
        if causal is not None:info['causal_phase']=causal
    editor._win_info = info
    editor._scalable_mask = mask
    editor._scalable_edit_envelope=mask>0
    if selected is not None and editor._scalable_mode!='precision':
        grasp_mask,grasp_info=grasp_transition_mask(len(q),selected,segments)
        info['grasp_preparation_available']=grasp_info
        info['protected_object_prefix_end']=info['window'][0]
        if grasp_mask is not None:editor._scalable_edit_envelope|=grasp_mask>0
    editor._win_chunks = (editor.basis * mask[:, None]).max(0) > .3
    print('[scalable-window]', step, json.dumps(info), flush=True)
    return mask


def _targets(editor, q):
    t = editor.binding.joint_target.copy()
    t[:, editor.projector.dids] = q[:, editor.projector.qids]
    for qi, di in editor.projector.ball_joints:
        t[:, di:di + 3] = Rotation.from_quat(q[:, qi:qi + 4]).as_rotvec()
    return t


def proposals(editor, learner):
    selected = editor._scalable_segment
    editor._ilc_feedback = dict(scalable=True, phases=editor._scalable_reports)
    if selected is None:
        return []
    hands = channels(editor); hand = hands[selected.hand]
    base = editor.binding.joint_q; mask = editor._scalable_mask
    phase = editor._scalable_phase; mode = phase['mode']; fps = editor.baseline.fps
    raw = []
    # Use offsets only while retention is established. Contact failures need a
    # new grasp/timing hypothesis, not a correction toward a falling object.
    if mode == 'precision':
        a, b = phase['interval']; o = selected.object_id
        pos = np.asarray(editor._object_pose[a:b, :, o], float)
        finite = np.isfinite(pos).all(-1)
        if finite.any():
            dp = np.median((pos[..., :3] - pos[..., 7:10])[finite], axis=0)
            quats = pos[finite]
            good = (np.linalg.norm(quats[:, 3:7], axis=-1) > 1e-8) & (np.linalg.norm(quats[:, 10:14], axis=-1) > 1e-8)
            rv = (Rotation.from_quat(quats[good, 3:7]) * Rotation.from_quat(quats[good, 10:14]).inv()).mean().as_rotvec() if good.any() else np.zeros(3)
            for gain in (.25, .5, 1.):
                raw.append((dict(op='retained_pose_compensation', gain=gain),
                            anchor_hand(base, hand, -gain * dp, -gain * rv, mask,
                                        pivot=np.asarray(editor.setup.reference.object_body_pos_w())[:, o])))
    else:
        # The library contains grasps fitted to THIS task and embodiment. Cycle
        # the finite catalog instead of evaluating its first entries forever.
        pattern = editor.options.get('scalable_grasp_glob', '')
        files = sorted(glob.glob(str(pattern))) if pattern else []
        if pattern and not files:
            raise ValueError(f'configured same-task grasp catalog is empty: {pattern}')
        families = {};prototype_names=[];segment_key=None;family_by_path={}
        catalog=Path(pattern).parent/'dexfit_types.npz' if pattern else None
        if catalog is not None and catalog.exists():
            with np.load(catalog,allow_pickle=False) as fits:
                if 'prototype_names' in fits:prototype_names=fits['prototype_names'].tolist()
                matches=[key[:-5] for key in fits.files if key.startswith(selected.hand+'_') and key.endswith('_meta')
                    and np.asarray(fits[key]).shape==(4,) and int(fits[key][3])==selected.object_id
                    and int(fits[key][0])==selected.start and int(fits[key][1])==selected.end]
                if len(matches)==1:segment_key=matches[0]
        for path in files:
            selection=None
            if segment_key is not None:
                with np.load(path,allow_pickle=False) as data:
                    header=json.loads(str(data['header'])) if 'header' in data else {}
                    selection=header.get('provenance',{}).get('selection')
            family = grasp_family(path,prototype_names,selection=selection,segment_key=segment_key)
            family_by_path[path]=family
            families.setdefault(family, []).append(path)
        editor._ilc_feedback['library_grasp_family_count']=len(families)
        files = []
        while any(families.values()):
            for family in families.values():
                if family:
                    files.append(family.pop(0))
        editor._ilc_feedback['library_catalog_size'] = len(files)
        limit = int(editor.options.get('scalable_library_candidates', 12))
        if limit < 1:
            raise ValueError('positive library candidate budget required')
        budget=proposal_budget(editor);variants=2 if getattr(editor,'_scalable_anchor',None) is not None else 1
        if budget is not None and budget>=2*variants:
            limit=min(limit,max(1,(budget//2)//variants))
        editor._ilc_feedback['library_file_budget']=limit
        cursor = editor._scalable_attempts.get(selected.key, 0) * limit
        files = files[cursor % len(files):] + files[:cursor % len(files)] if files else []
        library_mask,transition=grasp_transition_mask(len(base),selected,editor._scalable_segments)
        editor._ilc_feedback['grasp_preparation_available']=transition
        if library_mask is not None:editor._scalable_edit_envelope=(mask>0)|(library_mask>0)
        for path in files[:limit] if library_mask is not None else []:
            with np.load(path, allow_pickle=False) as data:
                template = np.asarray(data['joint_q'], float)
                expected = getattr(editor, 'command_sha256', None)
                if expected is not None:
                    header = json.loads(str(data['header'])) if 'header' in data.files else {}
                    actual = header.get('provenance', {}).get('object_contact_sha256')
                    if actual != expected:
                        raise ValueError(f'grasp library must match the immutable task digest: {path}')
            if template.shape != base.shape or not np.isfinite(template).all():
                raise ValueError(f'grasp library layout/frames mismatch: {path}')
            metadata=dict(op='replace_grasp',library=str(Path(path).resolve()),grasp_family=family_by_path[path],grasp_transition=transition,
                          sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
            swapped,trajectory=grasp_transition(base,template,hand,selected,editor._scalable_segments,fps)
            metadata['grasp_transition']=trajectory
            raw.append((metadata,swapped))
            anchor=getattr(editor,'_scalable_anchor',None)
            if anchor is not None:
                anchored_template=anchor_hand(template,hand,*anchor,np.ones(len(base)),
                    pivot=np.asarray(editor.setup.reference.object_body_pos_w())[:,selected.object_id])
                anchored,anchored_trajectory=grasp_transition(base,anchored_template,hand,selected,editor._scalable_segments,fps)
                raw.append((dict(metadata,op='replace_grasp_supported_anchor',
                    grasp_transition=anchored_trajectory,anchor_translation_m=anchor[0].tolist(),anchor_rotation_rad=anchor[1].tolist()),anchored))
        cfg = RefinementConfig(max_candidates=40)
        edit_args=dict(anchor=getattr(editor,'_scalable_anchor',None))
        if getattr(editor,'_scalable_causal_info',None) is not None:
            edit_args.update(edit_mask=mask,edit_bounds=editor._win_info['window'])
        raw += generate_phase_edits(base, hands, editor._scalable_segments, selected,
            np.asarray(editor.setup.reference.object_body_pos_w()),
            np.asarray(editor.setup.reference.object_body_quat_w())[..., [1, 2, 3, 0]], fps,
            holding=False, config=cfg,**edit_args)
        # Explore discrete object-frame placements, not world-axis episode fixes.
        ro = Rotation.from_quat(np.asarray(editor.setup.reference.object_body_quat_w())[selected.start, selected.object_id, [1, 2, 3, 0]])
        for axis in range(3):
            for shift in (-.02, -.01, .01, .02):
                d = np.zeros(3); d[axis] = shift
                q = anchor_hand(base, hand, ro.apply(d), np.zeros(3), mask)
                raw.append((dict(op='object_frame_regrasp', axis=axis, shift_m=shift), squeeze_hand(q, hand, .2, mask)))
    output = []; seen = set()
    ordinary = RefinementConfig()
    # Library swaps may change the complete finger posture. They still obey
    # physical limits, absolute cumulative wrist bounds and added-speed limits.
    library_translation = editor.options.get('scalable_grasp_translation_m')
    library_rotation = float(editor.options.get('scalable_grasp_rotation_rad', np.pi))
    for info, q in raw:
        requested=q
        fitted=info['op'].startswith('replace_grasp')
        a,b=selected.start,min(len(q),selected.end+1)
        if fitted:
            def linear_speed(values):return float(np.linalg.norm(np.diff(values,axis=0),axis=-1).max(initial=0))*fps
            def angular_speed(values):
                rotations=Rotation.from_quat(values)
                return float((rotations[1:]*rotations[:-1].inv()).magnitude().max(initial=0))*fps
            def finger_speed(values):return float(np.abs(np.diff(values,axis=0)).max(initial=0))*fps
            # Discrete grasps use their task-fitted contact geometry and motion
            # envelope. A continuous-nudge trust box must not erase the mode
            # change. Explicit user displacement caps still apply.
            translation_bound=float(library_translation) if library_translation is not None else max(
                ordinary.max_translation_m,float(np.linalg.norm(q[:,hand.position]-base[:,hand.position],axis=-1).max())+1e-7)
            motion_cfg=replace(ordinary,max_translation_m=translation_bound,max_rotation_rad=library_rotation,max_finger_delta_rad=np.pi,
                max_translation_speed_m_s=max(ordinary.max_translation_speed_m_s,linear_speed(base[a:b,hand.position])+linear_speed(q[a:b,hand.position])),
                max_rotation_speed_rad_s=max(ordinary.max_rotation_speed_rad_s,angular_speed(base[a:b,hand.quaternion])+angular_speed(q[a:b,hand.quaternion])),
                max_finger_speed_rad_s=max(ordinary.max_finger_speed_rad_s,finger_speed(base[a:b,hand.fingers])+finger_speed(q[a:b,hand.fingers])))
            info=dict(info,projection_policy='task_fitted_contact_envelope',motion_limits=dict(
                translation_m=translation_bound,translation_speed_m_s=motion_cfg.max_translation_speed_m_s,
                rotation_speed_rad_s=motion_cfg.max_rotation_speed_rad_s,finger_speed_rad_s=motion_cfg.max_finger_speed_rad_s))
        else:motion_cfg=ordinary
        q, scale = project_edit(base, q, list(hands.values()), editor.projector.qids,
                                editor.projector.lower, editor.projector.upper, fps,
                                motion_cfg)
        ok = True
        for h in hands.values():
            delta = q[:, h.position] - editor.baseline.joint_q[:, h.position]
            angle = (Rotation.from_quat(q[:, h.quaternion]) * Rotation.from_quat(editor.baseline.joint_q[:, h.quaternion]).inv()).magnitude()
            translation_bound = (float(library_translation) if library_translation is not None else
                max(ordinary.max_translation_m,float(np.linalg.norm(requested[:,h.position]-editor.baseline.joint_q[:,h.position],axis=-1).max())+1e-7)) if fitted else float(editor.options.get('wrist_translation_offset_m', .08))
            rotation_bound = library_rotation if info['op'].startswith('replace_grasp') else float(editor.options.get('wrist_rotation_offset_rad', .7))
            ok &= np.linalg.norm(delta, axis=1).max() <= translation_bound + 1e-6
            ok &= angle.max() <= rotation_bound + 1e-6
        key = hashlib.sha256(np.round(q, 6).tobytes()).hexdigest()
        if not ok or key in seen or np.max(np.abs(q - base)) < 1e-6:
            continue
        seen.add(key)
        if info['op'].startswith('replace_grasp'):
            a,b=selected.start,min(len(q),selected.end+1)
            original_r=Rotation.from_quat(base[a:b,hand.quaternion])
            wanted_r=Rotation.from_quat(requested[a:b,hand.quaternion])
            actual_r=Rotation.from_quat(q[a:b,hand.quaternion])
            info=dict(info,realization=dict(
                requested_rotation_change_deg=float(np.median((wanted_r*original_r.inv()).magnitude())*180/np.pi),
                realized_rotation_change_deg=float(np.median((actual_r*original_r.inv()).magnitude())*180/np.pi),
                rotation_goal_error_deg=float(np.median((actual_r*wanted_r.inv()).magnitude())*180/np.pi),
                position_goal_error_m=float(np.median(np.linalg.norm(q[a:b,hand.position]-requested[a:b,hand.position],axis=-1))),
                finger_goal_error_rad=float(np.max(np.abs(q[a:b,hand.fingers]-requested[a:b,hand.fingers]),initial=0))))
        output.append((dict(info, segment=selected.key, mode=mode, projection_scale=scale), q, _targets(editor, q)))
    # Small world budgets must still explore grasp, timing, and placement
    # alternatives; a large library must not consume every candidate slot.
    groups = {}
    for item in output:
        groups.setdefault(item[0]['op'], []).append(item)
    diverse = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                diverse.append(group.pop(0))
    # Reserve space for every catalog file whose cursor is advanced this round.
    # Balancing only by operator name can truncate library items at the runtime
    # trial limit, silently skipping grasp families on the following round.
    budget=proposal_budget(editor)
    variants=2 if getattr(editor,'_scalable_anchor',None) is not None else 1
    if budget is not None and budget>=2*variants:
        library=[p for p in output if p[0]['op'].startswith('replace_grasp')]
        others=[p for p in diverse if not p[0]['op'].startswith('replace_grasp')]
        diverse=[]
        for index in range(max(len(library),len(others))):
            if index<len(library):diverse.append(library[index])
            if index<len(others):diverse.append(others[index])
    editor._ilc_feedback['operations'] = [p[0] for p in diverse]
    if mode != 'precision':
        editor._ilc_feedback['library_proposals'] = sum(info['op'].startswith('replace_grasp') for info,_,_ in diverse)
    return diverse


def discovery_costs(editor, packed=None):
    return feedback_costs(editor._object_pose, _phase(editor, packed), config(editor))


def validate_shortlist(editor, learner, cand_q, cand_t, valid, costs):
    """Evaluate a fixed shortlist and current reference in one NEW sim batch."""
    from .parallel_rollout import rollout_candidates
    eligible = np.flatnonzero(valid)
    editor._scalable_validation = dict(independent=True, reason='no_improving_proposal')
    editor._accepted_stats = None
    if not len(eligible):
        return None
    k = min(int(editor.options.get('scalable_validation_top_k', 3)), len(eligible))
    if k < 1:
        raise ValueError('positive validation shortlist size required')
    ids = eligible[np.argsort(np.asarray(costs)[eligible], kind='stable')[:k]]
    replicas = editor.upper_worlds // (k + 1)
    if replicas < 2:
        raise ValueError('validation needs >=2 worlds per candidate and baseline')
    n = k * replicas; nb = editor.upper_worlds - n
    q = np.asarray([cand_q[i] for i in ids for _ in range(replicas)] + [editor.binding.joint_q] * nb)
    t = np.asarray([cand_t[i] for i in ids for _ in range(replicas)] + [editor.binding.joint_target] * nb)
    # Stop logic and search evidence refer to discovery ordering (C*R + NB).
    # Fresh validation must not silently replace those buffers with a new layout.
    names = [key for key in vars(editor) if key.startswith(('_object_', '_palm_', '_tip_', '_signed_', '_contact_', '_physical_', '_rollout_', '_hand_keypoints'))]
    saved = {key: getattr(editor, key) for key in names}
    try:
        rows, packed, _ = rollout_candidates(editor, learner, q, t)
        phase = _phase(editor, packed)
        fresh = feedback_costs(editor._object_pose, phase, config(editor))
        ok, report = independent_gate(editor._object_pose, fresh, k, replicas,
            prefix_end=int(editor._win_info['window'][0]),
            retained=phase['retained'] if phase is not None and phase['mode'] != 'precision' else None,
            config=config(editor))
        full = np.asarray([all(m['full_horizon'] and m['all_errors_finite'] for m in rows[j*replicas:(j+1)*replicas]) for j in range(k)])
        baseline_horizon = all(m['full_horizon'] for m in rows[n:])
        ok &= full & baseline_horizon
        report.update(trials=ids.tolist(), passed=ok.tolist(), full_horizon=full.tolist(), baseline_full_horizon=baseline_horizon,
                      selection='shortlist fixed before fresh rollout; corrected independent comparison')
        selected = int(np.flatnonzero(ok)[np.argmin(np.asarray(report['candidate_cost'])[ok])]) if ok.any() else None
        if selected is not None:
            keys = ('object_position_robust_m', 'object_orientation_robust_rad', 'object_success_robust_fraction', 'unstable_world_fraction')
            def aggregate(batch):
                return {key: float(np.mean([m[key] for m in batch])) for key in keys}
            report['candidate'] = aggregate(rows[selected*replicas:(selected+1)*replicas])
            report['baseline'] = aggregate(rows[n:])
        editor._scalable_validation = report
        return int(ids[selected]) if selected is not None else None
    finally:
        for name, value in saved.items():
            setattr(editor, name, value)


def record_feedback(editor):
    phase = getattr(editor, '_scalable_phase', None)
    return dict(segment=phase['segment'] if phase else None, mode=phase['mode'] if phase else None,
                phases=editor._scalable_reports, unsuccessful_rounds=dict(editor._scalable_attempts))


def install(cls):
    if getattr(cls, '_scalable_installed', False):
        return
    init, window, candidates, blend = cls.__init__, cls._failure_window, cls._ilc_candidates, cls._window_blend
    original_round = cls._one_round
    def initialize(self, **kwargs):
        options = dict(kwargs['options'])
        if options.get('scalable_refinement'):
            if options.get('structured_refinement') or options.get('window_checkpoint') or options.get('train_on_proposals'):
                raise ValueError('scalable refinement requires full-horizon evaluation and a single published reference')
            if options.get('upper_update', 'search') != 'search':
                raise ValueError('scalable refinement supplies its own proposals; use upper_update=search')
            if int(options.get('replicas', 1)) < 2 or int(options.get('baseline_replicas', 0)) < 2:
                raise ValueError('scalable refinement requires candidate and baseline replicas')
            if options.get('first_success') == 'stop':
                raise ValueError('a single lucky world cannot stop scalable refinement; use the validated pair stop rule')
            config(type('Options', (), {'options': options})())
            options.update(failure_window=True, ilc_candidates=True, official_metric=True)
            kwargs['options'] = options
        init(self, **kwargs)
        if options.get('scalable_refinement'):
            path = self.output.parent / 'ttsa_manifest.json'
            manifest = json.loads(path.read_text())
            manifest.update(engine='scalable_physical_feedback',
                upper_learning='failure-directed reference search; directed trials receive no upper actor credit',
                confirmation='fixed shortlist on fresh simulator replicas and unchanged baseline; corrected uncertainty margin',
                success_metric='official per-world first-pose-aligned episode means; every object',
                scalable_options={k:v for k,v in self.options.items() if k.startswith('scalable_')})
            path.write_text(json.dumps(manifest, indent=2) + '\n')
    def choose(self, learner, step):
        return select_window(self, learner, step) if self.options.get('scalable_refinement') else window(self, learner, step)
    def propose(self, learner):
        return proposals(self, learner) if self.options.get('scalable_refinement') else candidates(self, learner)
    def apply_window(self, q, mask):
        if not self.options.get('scalable_refinement'):
            return blend(self, q, mask)
        # Directed proposals already contain the phase taper. Random proposals
        # are disabled by the installer for this mode; do not taper a second time.
        out = q.copy(); outside = ~np.asarray(getattr(self,'_scalable_edit_envelope',np.asarray(mask)>0),bool)
        out[outside] = self.binding.joint_q[outside]
        return out, _targets(self, out)
    def one_round(self, *args, **kwargs):
        result = original_round(self, *args, **kwargs)
        if self.options.get('scalable_refinement'):
            self._scalable_previous_record = self._last_record
        return result
    cls.__init__, cls._failure_window, cls._ilc_candidates, cls._window_blend = initialize, choose, propose, apply_window
    cls._one_round = one_round
    cls._scalable_installed = True
