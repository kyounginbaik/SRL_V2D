"""Reusable contact-phase edits for precise manipulation reference refinement.

This module contains no simulator, episode paths, or robot joint-name assumptions.
An embodiment adapter supplies wrist/finger channels. Object poses and contact
labels are immutable inputs; only robot references are returned.
"""
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class HandChannels:
    name: str
    position: tuple[int, int, int]
    quaternion: tuple[int, int, int, int]  # xyzw
    fingers: tuple[int, ...]
    closure: tuple[float, ...]

    def __post_init__(self):
        ids = (*self.position, *self.quaternion, *self.fingers)
        if len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError('hand channels must be distinct nonnegative indices')
        if len(self.fingers) != len(self.closure) or not np.isfinite(self.closure).all():
            raise ValueError('one finite closure weight per finger channel is required')


@dataclass(frozen=True)
class ContactSegment:
    hand: str
    object_id: int
    start: int
    end: int
    index: int

    @property
    def key(self):
        return f'{self.hand}_obj{self.object_id}_seg{self.index}'


@dataclass(frozen=True)
class RefinementConfig:
    position_tolerance_m: float = .03
    orientation_tolerance_rad: float = np.pi / 6
    max_translation_m: float = .04
    max_rotation_rad: float = .35
    max_finger_delta_rad: float = .4
    max_translation_speed_m_s: float = .3
    max_rotation_speed_rad_s: float = 2.
    max_finger_speed_rad_s: float = 3.
    approach_s: float = .5
    release_s: float = .5
    max_candidates: int = 20

    def __post_init__(self):
        values = [v for k, v in vars(self).items() if k != 'max_candidates']
        if not np.isfinite(values).all() or min(values) <= 0 or self.max_candidates < 1:
            raise ValueError('finite positive refinement limits are required')


def extract_segments(contacts: Mapping[str, np.ndarray], fps: float, *, gap_s=.5,
                     min_duration_s=.15) -> list[ContactSegment]:
    """Extract independent per-hand/per-object phases from [frame, object] labels.

    A hand changing objects creates separate phases even without a contact gap.
    Small label holes are merged using seconds, making behavior FPS independent.
    """
    if fps <= 0 or gap_s < 0 or min_duration_s < 0:
        raise ValueError('positive fps and nonnegative segment durations required')
    segments = []
    for hand, values in contacts.items():
        active = np.asarray(values, bool)
        if active.ndim != 2:
            raise ValueError('contact arrays must have shape [frame, object]')
        for obj in range(active.shape[1]):
            idx = np.flatnonzero(active[:, obj])
            groups = np.split(idx, np.flatnonzero(np.diff(idx) > max(1, round(gap_s * fps))) + 1)
            for number, group in enumerate(g for g in groups if len(g)):
                if group[-1] - group[0] + 1 >= max(1, round(min_duration_s * fps)):
                    segments.append(ContactSegment(hand, obj, int(group[0]), int(group[-1]), number))
    return sorted(segments, key=lambda s: (s.start, s.hand, s.object_id))


def find_giver(receiver: ContactSegment, segments: Sequence[ContactSegment], fps: float):
    """A giver must support the same object around the receiver's contact onset."""
    eligible = [s for s in segments if s.hand != receiver.hand and s.object_id == receiver.object_id
                and s.start < receiver.start <= s.end + round(.25 * fps)]
    return max(eligible, key=lambda s: s.end) if eligible else None


def phase_mask(frames, start, end, fps, *, lead_s=.5, release_s=.5):
    a, b = max(1, start - round(lead_s * fps)), min(frames - 2, end + round(release_s * fps))
    if not 0 < a < b < frames - 1:
        raise ValueError('a phase requires a nonempty editable interior')
    t = np.arange(frames, dtype=float)
    ramp = max(1., .25 * fps)
    v = np.clip(np.minimum((t - a) / ramp, (b - t) / ramp), 0., 1.)
    return .5 - .5 * np.cos(np.pi * v), (a, b)


def event_time_map(frames, bounds, event, shift_frames):
    """Monotone time map moving one event while fixing both window boundaries.

    A positive shift delays the event. Time outside bounds is unchanged.
    """
    a, b = bounds
    if not 0 <= a < event < b < frames or not a < event + shift_frames < b:
        raise ValueError('retimed event must remain strictly inside the window')
    t = np.arange(frames, dtype=float)
    inside = (t > a) & (t < b)
    t[inside] = PchipInterpolator([a, event + shift_frames, b], [a, event, b])(t[inside])
    if np.any(np.diff(t) <= 0):
        raise ValueError('time map must be strictly increasing')
    return t


def _sample(values, times):
    values = np.asarray(values, float)
    shape = values.shape
    flat = values.reshape(shape[0], -1)
    return np.stack([np.interp(times, np.arange(shape[0]), col) for col in flat.T], axis=-1).reshape(shape)


def continuous_rotvec(rotations,active=None):
    """Choose temporally adjacent logarithm branches, including crossings of pi.

    Principal rotation vectors jump by 2*pi at the branch cut even for slow
    physical rotation. Inactive blend intervals reset the branch choice.
    """
    vectors=rotations.as_rotvec()
    if vectors.ndim!=2 or vectors.shape[1]!=3:raise ValueError('rotation trajectory required')
    enabled=np.ones(len(vectors),bool) if active is None else np.asarray(active,bool)
    if enabled.shape!=(len(vectors),):raise ValueError('one active flag per rotation required')
    result=np.zeros_like(vectors)
    for i,v in enumerate(vectors):
        if not enabled[i]:continue
        if i==0 or not enabled[i-1]:result[i]=v;continue
        angle=np.linalg.norm(v);previous=result[i-1]
        if angle<1e-10:
            # Identity is an exact anchor. Keeping an unwrapped 2*pi here
            # would turn an unchanged pose into a nonzero blended correction.
            result[i]=0.
        else:
            axis=v/angle;turns=round((previous@axis-angle)/(2*np.pi))
            result[i]=v+2*np.pi*turns*axis
    return result


def grasp_transition_mask(frames,segment,segments):
    """Prepare/restore a grasp over this hand's free neighboring intervals.

    The full fitted posture is reached during contact. Episode endpoints and
    other contacts of the same hand stay fixed, including other objects.
    """
    others=[s for s in segments if s.hand==segment.hand and s!=segment]
    if any(s.start<=segment.end and s.end>=segment.start for s in others):
        return None,dict(reason='overlapping_hand_contacts')
    a=max([1]+[s.end+1 for s in others if s.end<segment.start])
    b=min([frames-2]+[s.start-1 for s in others if s.start>segment.end])
    if not a<segment.start<=segment.end<b:
        return None,dict(reason='no_free_grasp_transition_interval')
    t=np.arange(frames);rise=np.clip((t-a)/(segment.start-a),0,1)
    fall=np.clip((b-t)/(b-segment.end),0,1)
    mask=(.5-.5*np.cos(np.pi*rise))*(.5-.5*np.cos(np.pi*fall))
    return mask,dict(reason='free_hand_grasp_transition',window=[a,b],
        approach_frames=segment.start-a,return_frames=b-segment.end)


def grasp_transition(current,template,hand,segment,segments,fps,config=RefinementConfig(),*,minimum_approach_frames=0,minimum_return_frames=0,return_config=None):
    """Extend a fitted contact grasp through a smooth free-hand preparation.

    Existing library approach/return motion may already contain abrupt jumps.
    Use only its fitted contact correction and extend the endpoint corrections
    through the available free intervals. Contact poses are exact.
    """
    if not np.isfinite(fps) or fps<=0:raise ValueError('positive finite control rate required')
    if any(not isinstance(v,int) or v<0 for v in (minimum_approach_frames,minimum_return_frames)):
        raise ValueError('nonnegative integer minimum transition durations required')
    base=np.asarray(current,float);target=np.asarray(template,float)
    mask,info=grasp_transition_mask(len(base),segment,segments)
    if mask is None:return None,info
    if target.shape!=base.shape or not np.isfinite(target).all():raise ValueError('finite matching grasp template required')
    old=Rotation.from_quat(base[:,hand.quaternion]);new=Rotation.from_quat(target[:,hand.quaternion])
    contact=np.clip(np.arange(len(base)),segment.start,segment.end)
    # A wrist rotation does not orbit a distant free hand around the object.
    # Freeze world position offsets outside contact rather than making a free
    # hand orbit an object that may already belong to the next manipulation.
    offset=(target[:,hand.position]-base[:,hand.position])[contact]
    # Right-hand correction cancels common object motion and inherits the
    # baseline hand's free motion. A world/object-left correction can invent a
    # large extra spin when the original hand turns during its return.
    relative=old.inv()*new
    vector=continuous_rotvec(relative[contact],mask>0)
    finger=(target[:,hand.fingers]-base[:,hand.fingers])[contact]
    def preparation_frames(index,limits):
        duration=np.pi/2*max(np.linalg.norm(offset[index])/limits.max_translation_speed_m_s,
            np.linalg.norm(vector[index])/limits.max_rotation_speed_rad_s,
            np.abs(finger[index]).max(initial=0)/limits.max_finger_speed_rad_s)
        # One control frame of slack covers discrete angular increment effects.
        return max(1,int(np.ceil(duration*fps))+1)
    lead=max(minimum_approach_frames,preparation_frames(segment.start,config))
    tail=max(minimum_return_frames,preparation_frames(segment.end,return_config or config))
    free=info['window'];a=max(free[0],segment.start-lead);b=min(free[1],segment.end+tail)
    t=np.arange(len(base));rise=np.clip((t-a)/(segment.start-a),0,1);fall=np.clip((b-t)/(b-segment.end),0,1)
    mask=(.5-.5*np.cos(np.pi*rise))*(.5-.5*np.cos(np.pi*fall))
    info=dict(info,available_window=free,window=[a,b],approach_frames=segment.start-a,return_frames=b-segment.end,
              required_approach_frames=lead,required_return_frames=tail)
    result=base.copy()
    result[:,hand.position]+=mask[:,None]*offset
    result[:,hand.quaternion]=(old*Rotation.from_rotvec(mask[:,None]*vector)).as_quat()
    result[:,hand.fingers]+=mask[:,None]*finger
    result[mask==0]=base[mask==0]
    return result,dict(info,space='fitted_contact_correction',rotation_space='hand_relative',position_space='world',contact_window=[segment.start,segment.end])


def blend_hand(current, proposed, hand, weight):
    """Blend on SE(3); quaternion sign changes never interpolate through zero."""
    q = np.array(current, dtype=float, copy=True)
    w = np.asarray(weight, float)
    if w.shape != (len(q),) or not np.isfinite(w).all() or (w < 0).any() or (w > 1).any():
        raise ValueError('blend weight must be finite in [0,1] for every frame')
    ids = (*hand.position, *hand.fingers)
    q[:, ids] += w[:, None] * (proposed[:, ids] - q[:, ids])
    rb = Rotation.from_quat(current[:, hand.quaternion])
    delta = continuous_rotvec(Rotation.from_quat(proposed[:, hand.quaternion])*rb.inv(),w>0)
    q[:, hand.quaternion] = (Rotation.from_rotvec(w[:, None] * delta) * rb).as_quat()
    q[w == 0] = current[w == 0]
    return q


def retime_hand(current, hand, object_positions, object_quaternions, object_id, times, weight):
    """Retime one hand in the object's frame, preserving a moving-object grasp."""
    base = np.asarray(current, float)
    times = np.asarray(times, float)
    if times.shape != (len(base),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('one finite increasing sample time per reference frame is required')
    if times[0] < 0 or times[-1] > len(base) - 1:
        raise ValueError('retiming cannot sample outside the reference')
    op = np.asarray(object_positions)[:, object_id]
    ro = Rotation.from_quat(np.asarray(object_quaternions)[:, object_id])
    relative_p = ro.inv().apply(base[:, hand.position] - op)
    relative_r = ro.inv() * Rotation.from_quat(base[:, hand.quaternion])
    q = base.copy()
    q[:, hand.position] = op + ro.apply(_sample(relative_p, times))
    q[:, hand.quaternion] = (ro * Slerp(np.arange(len(base)), relative_r)(times)).as_quat()
    if hand.fingers:
        q[:, hand.fingers] = _sample(base[:, hand.fingers], times)
    return blend_hand(base, q, hand, weight)


def anchor_hand(current, hand, translation, rotation_vector, weight, *, pivot=None):
    """Move one hand toward a supported measured object pose; never move both blindly."""
    q = np.array(current, dtype=float, copy=True)
    d, rv = np.asarray(translation, float), np.asarray(rotation_vector, float)
    if d.shape != (3,) or rv.shape != (3,) or not np.isfinite([d, rv]).all():
        raise ValueError('finite 3D translation and rotation required')
    if pivot is not None:
        center = np.broadcast_to(np.asarray(pivot, float), (len(q), 3))
        q[:, hand.position] = center + Rotation.from_rotvec(rv).apply(q[:, hand.position] - center) + d
    else:
        q[:, hand.position] += d
    q[:, hand.quaternion] = (Rotation.from_rotvec(rv) * Rotation.from_quat(q[:, hand.quaternion])).as_quat()
    return blend_hand(current, q, hand, weight)


def squeeze_hand(current, hand, radians, weight):
    q = np.array(current, dtype=float, copy=True)
    q[:, hand.fingers] += float(radians) * np.asarray(hand.closure)[None]
    return blend_hand(current, q, hand, weight)


def project_edit(current, proposed, hands, scalar_ids, lower, upper, fps, config):
    """Project translation, rotation, and fingers independently within each hand.

    A fast finger transition must not erase a valid receiver position correction.
    Each group preserves its limits. Rotation contractions are checked using
    physical world angular increments rather than rotation-vector differences.
    """
    base = np.asarray(current, float); q = np.array(proposed, dtype=float, copy=True)
    if q.shape != base.shape or not np.isfinite(q).all() or fps <= 0:
        raise ValueError('finite matching references and positive fps required')
    if np.any(base[:, scalar_ids] < lower - 1e-5) or np.any(base[:, scalar_ids] > upper + 1e-5):
        raise ValueError('current reference violates joint limits')
    q[:, scalar_ids] = np.clip(q[:, scalar_ids], lower, upper)
    q[[0, -1]] = base[[0, -1]]
    scales = []
    result = base.copy()
    for hand in hands:
        dp = q[:, hand.position] - base[:, hand.position]
        base_rotation=Rotation.from_quat(base[:,hand.quaternion])
        correction=Rotation.from_quat(q[:,hand.quaternion])*base_rotation.inv()
        rv = continuous_rotvec(correction)
        df = q[:, hand.fingers] - base[:, hand.fingers]
        def factor(peak, bound, speed, speed_bound):
            return min(1., bound / max(peak, 1e-12), speed_bound / max(speed, 1e-12))
        sp = factor(np.linalg.norm(dp, axis=-1).max(), config.max_translation_m,
                    np.linalg.norm(np.diff(dp, axis=0), axis=-1).max() * fps, config.max_translation_speed_m_s)
        # Compare actual world angular increments. Differentiating the edit's
        # log invents a spike at pi and penalizes a fixed object-relative grasp
        # simply because the object rotates. Neither is extra wrist velocity.
        base_increment=(base_rotation[1:]*base_rotation[:-1].inv()).as_rotvec()
        def rotation_feasible(amount):
            edited=Rotation.from_rotvec(amount*rv)*base_rotation
            increments=(edited[1:]*edited[:-1].inv()).as_rotvec()
            speed=np.linalg.norm(increments-base_increment,axis=-1).max()*fps
            peak=(edited*base_rotation.inv()).magnitude().max()
            return speed<=config.max_rotation_speed_rad_s*(1-1e-6) and peak<=config.max_rotation_rad+1e-10
        sr=1.
        if not rotation_feasible(sr):
            # Keep the largest tested feasible step. Bounds need not be
            # monotone on SO(3); every returned lower bracket is measured.
            high=1.;low=.5
            while low>1e-9 and not rotation_feasible(low):high=low;low*=.5
            if low<=1e-9:low=0.
            for _ in range(24):
                middle=(low+high)/2
                if rotation_feasible(middle):low=middle
                else:high=middle
            sr=low
        sf = factor(np.abs(df).max(initial=0), config.max_finger_delta_rad,
                    np.abs(np.diff(df, axis=0)).max(initial=0) * fps, config.max_finger_speed_rad_s)
        result[:, hand.position] += sp * dp
        result[:, hand.fingers] += sf * df
        result[:, hand.quaternion] = (Rotation.from_rotvec(sr * rv) * base_rotation).as_quat()
        unchanged_rotation = np.linalg.norm(rv, axis=-1) < 1e-12
        result[np.ix_(unchanged_rotation, hand.quaternion)] = base[np.ix_(unchanged_rotation, hand.quaternion)]
        scales.extend([sp, sr, sf])
    # Preserve exact unchanged rows; important for deterministic prefix comparison.
    result[np.all(q == base, axis=1)] = base[np.all(q == base, axis=1)]
    result[[0, -1]] = base[[0, -1]]
    return result.astype(np.float32), float(min(scales, default=1.))


def supported_anchor(actual, target, supported, frame, *, radius=3):
    """Estimate pose mismatch near contact, using supported finite samples only.

    Inputs are already robust world summaries [frame,7]. A dropped object must
    not create a slip or approach correction from its later falling trajectory.
    """
    actual, target = np.asarray(actual, float), np.asarray(target, float)
    if actual.shape != target.shape or actual.ndim != 2 or actual.shape[1] != 7:
        raise ValueError('pose summaries must have shape [frame,7]')
    ids = np.arange(max(0, frame - radius), min(len(actual), frame + radius + 1))
    good = (np.asarray(supported, bool)[ids] & np.isfinite(actual[ids]).all(1)
            & np.isfinite(target[ids]).all(1) & (np.linalg.norm(actual[ids, 3:], axis=1) > 1e-8)
            & (np.linalg.norm(target[ids, 3:], axis=1) > 1e-8))
    ids = ids[good]
    if not len(ids):
        return None
    delta = np.median(actual[ids, :3] - target[ids, :3], axis=0)
    rotation = (Rotation.from_quat(actual[ids, 3:]) * Rotation.from_quat(target[ids, 3:]).inv()).mean().as_rotvec()
    return delta, rotation


def generate_phase_edits(current, hands: Mapping[str, HandChannels], segments, selected,
                         object_positions, object_quaternions, fps, *, anchor=None,
                         holding=False, config=RefinementConfig(), edit_mask=None,
                         edit_bounds=None, event_frame=None):
    """A bounded candidate menu for acquisition, handover, regrasp, and carry.

    Timing is expressed in seconds and coordinated by hand/object relationships.
    Returns (metadata, robot_q); callers project, then evaluate physics replicas.
    """
    receiver = hands[selected.hand]; giver = find_giver(selected, segments, fps)
    if edit_mask is None and edit_bounds is None:
        mask, bounds = phase_mask(len(current), selected.start, selected.end, fps,
                                  lead_s=config.approach_s, release_s=config.release_s)
    else:
        if edit_mask is None or edit_bounds is None:
            raise ValueError('local edits require both a mask and window bounds')
        mask, bounds = np.asarray(edit_mask, float), tuple(edit_bounds)
        a, b = bounds
        if (mask.shape != (len(current),) or not np.isfinite(mask).all()
                or not 0 < a < b < len(current) - 1 or mask.min() < 0 or mask.max() > 1
                or np.any(mask[:a + 1]) or np.any(mask[b:])):
            raise ValueError('edit mask must be zero at/outside the bounded window')
        # A handover at acquisition does not make every later manipulation
        # interval another handover. Keep transition operators near their event.
        if giver is not None and not (a < selected.start < b or a < giver.end < b):
            giver = None
    proposals = []
    pivot = np.asarray(object_positions)[:, selected.object_id]

    def add(op, q, **kw):
        if np.isfinite(q).all() and np.max(np.abs(q - current)) > 1e-6:
            proposals.append((dict(op=op, segment=selected.key, edit_window=list(bounds), receiver=selected.hand,
                                   giver=giver.hand if giver else None, object_id=selected.object_id, **kw), q))

    def timing(q, hand, event, seconds):
        # Retiming a release must not shift a later regrasp by the same hand.
        from .refinement_window import window_mask
        next_contact=min((s.start for s in segments if s.hand==hand.name and s.start>event),default=len(current)-1)
        local_bounds = (max(bounds[0], event - round(.5 * fps)),
                        min(bounds[1], event + round((abs(seconds) + .75) * fps),next_contact))
        times = event_time_map(len(current), local_bounds, event, round(seconds * fps))
        return retime_hand(q, hand, object_positions, object_quaternions,
                           selected.object_id, times, mask*window_mask(len(current),local_bounds,fps))

    for seconds in (-.2, -.4):
        try: add('close_earlier', timing(current, receiver, selected.start, seconds), shift_s=seconds)
        except ValueError: continue
    for seconds in (.2, .4):
        try: add('close_later', timing(current, receiver, selected.start, seconds), shift_s=seconds)
        except ValueError: continue
    if event_frame is not None and not bounds[0] < selected.start < bounds[1]:
        # Retiming relative hand motion can refine an in-hand adjustment during
        # a long grasp. Object commands and distant acquisition stay fixed.
        for seconds in (-.2, .2):
            try: add('relative_phase_shift', timing(current, receiver, event_frame, seconds), shift_s=seconds)
            except ValueError: continue
    if giver is not None:
        for seconds in (.3, .6, .9):
            try: add('release_later', timing(current, hands[giver.hand], giver.end, seconds), shift_s=seconds)
            except ValueError: continue
        # Actual support can persist after the nominal release. Explicitly open
        # the giver while aligning the receiver; physics scoring decides whether
        # a transfer was established before allowing this release.
        t = np.arange(len(current), dtype=float)
        smooth = lambda v: .5 - .5 * np.cos(np.pi * np.clip(v, 0, 1))
        next_giver=min((s.start for s in segments if s.hand==giver.hand and s.start>giver.end),default=len(current))
        release_fall_start=min(giver.end+.8*fps,next_giver-.3*fps)
        release_weight = (smooth((t - (giver.end - 2)) / max(1., .2 * fps))
                          * (1 - smooth((t - release_fall_start) / max(1., .3 * fps)))) * mask
        for opening in (.3, .6):
            q = squeeze_hand(current, hands[giver.hand], -opening, release_weight)
            add('giver_open', q, radians=-opening)
            if anchor is not None:
                q = anchor_hand(q, receiver, *anchor, mask, pivot=pivot)
                add('receiver_anchor_giver_open', squeeze_hand(q, receiver, .2, mask), radians=-opening)
        away = current[min(giver.end, len(current) - 1), hands[giver.hand].position] - pivot[min(giver.end, len(current) - 1)]
        if np.linalg.norm(away) > 1e-8:
            q = squeeze_hand(current, hands[giver.hand], -.6, release_weight)
            q = anchor_hand(q, hands[giver.hand], .02 * away / np.linalg.norm(away), np.zeros(3), release_weight)
            if anchor is not None:
                q = anchor_hand(q, receiver, *anchor, mask, pivot=pivot)
            add('receiver_anchor_giver_withdraw', squeeze_hand(q, receiver, .2, mask))
        for early, late in ((-.2, .3), (-.2, .6), (-.4, .6)):
            try:
                q = timing(current, receiver, selected.start, early)
                q = timing(q, hands[giver.hand], giver.end, late)
                add('handover_overlap', q, close_shift_s=early, release_shift_s=late)
            except ValueError: continue
    if anchor is not None:
        dp, rv = anchor
        for gain in (.5, 1.):
            add('receiver_anchor', anchor_hand(current, receiver, gain * dp, gain * rv, mask, pivot=pivot), gain=gain)
        if holding:
            # With sustained receiver support, move the held object toward the
            # task goal. Acquisition uses the opposite sign to reach the object.
            for gain in (.25, .5, 1.):
                add('held_pose_compensation', anchor_hand(current, receiver, -gain * dp, -gain * rv,
                    mask, pivot=pivot), gain=gain)
        if giver is not None:
            try:
                q = timing(current, receiver, selected.start, -.2)
                q = timing(q, hands[giver.hand], giver.end, .6)
                add('overlap_and_anchor', anchor_hand(q, receiver, dp, rv, mask, pivot=pivot))
            except ValueError: pass
    for delta in (.1, .2, .35):
        add('receiver_squeeze', squeeze_hand(current, receiver, delta, mask), radians=delta)
    if giver is not None:
        try:
            q = timing(current, receiver, selected.start, -.2)
            q = timing(q, hands[giver.hand], giver.end, .6)
            add('overlap_and_squeeze', squeeze_hand(q, receiver, .2, mask), radians=.2)
            if anchor is not None:
                add('overlap_anchor_squeeze', anchor_hand(squeeze_hand(q, receiver, .2, mask), receiver, *anchor, mask, pivot=pivot))
        except ValueError: pass
    # Cover operator families before spending budget on amplitude variants.
    groups = {}
    for proposal in proposals:
        groups.setdefault(proposal[0]['op'], []).append(proposal)
    balanced = []
    for index in range(max((len(g) for g in groups.values()), default=0)):
        for group in groups.values():
            if index < len(group): balanced.append(group[index])
    return balanced[:config.max_candidates]


def precision_cost(position_error, orientation_error, config=RefinementConfig()):
    """Per-world local pose loss; preserves objects instead of averaging them away."""
    p, r = np.asarray(position_error, float), np.asarray(orientation_error, float)
    if p.shape != r.shape or p.ndim != 3:
        raise ValueError('pose errors must have shape [frame, world, object]')
    p = np.nan_to_num(p, nan=1., posinf=1., neginf=1.)
    r = np.nan_to_num(r, nan=np.pi, posinf=np.pi, neginf=np.pi)
    excess = p / config.position_tolerance_m + r / config.orientation_tolerance_rad
    # Mean tracks accuracy; a modest tail term discourages transient drops.
    return excess.mean(0).max(1) + .25 * np.quantile(excess, .95, axis=0).max(1)
