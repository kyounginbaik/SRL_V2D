"""Causal acquisition references; outputs are robot setpoints, never object edits."""
import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_feedback import correct_planned_hand_target, task_trajectory_in_execution_frame


def blend_pose(a, b, weight):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rotation = Rotation.from_rotvec(weight * (Rotation.from_quat(b[3:]) * Rotation.from_quat(a[3:]).inv()).as_rotvec()) * Rotation.from_quat(a[3:])
    return np.r_[a[:3] + weight * (b[:3] - a[:3]), rotation.as_quat()]


def smooth(value):
    x = np.clip(value, 0., 1.)
    return x*x*x*(10+x*(-15+6*x))


class CausalGraspReference:
    def __init__(self, planned_wrist, object_task, nominal_wrist, fingers, *,
                 bounds, contact, carry_blend, finger_bounds, release_start,
                 initial_reference_object0, fps, open_fingers, closed_fingers,
                 gated_closure=False, model=None, position_model=None,
                 max_speed=.5, max_rotation_speed=5., finger_speed=3.,
                 seating_position=.02, seating_rotation=.2, seating_duration=.15):
        self.planned = np.array(planned_wrist, float, copy=True)
        self.task = np.array(object_task, float, copy=True)
        self.nominal = np.array(nominal_wrist, float, copy=True)
        self.fingers = np.array(fingers, float, copy=True)
        n = len(self.planned)
        if (self.planned.shape != (n, 7) or self.task.shape != (n, 7) or self.nominal.shape != (n, 7)
                or self.fingers.ndim != 2 or len(self.fingers) != n
                or not all(np.isfinite(x).all() for x in [self.planned, self.task, self.nominal, self.fingers])
                or any(np.any(np.linalg.norm(x[:, 3:], axis=1) < 1e-8) for x in [self.planned, self.task, self.nominal])):
            raise ValueError('matching finite pose and finger trajectories required')
        a, b = bounds; c, d = contact; s, e = carry_blend; fa, fb = finger_bounds
        if (any(not isinstance(x, (int, np.integer)) for x in [a,b,c,d,s,e,fa,fb,release_start])
                or not 1 <= a < c <= s < e <= d < b < n
                or not 1 <= fa <= a < release_start <= fb < n):
            raise ValueError('protected acquisition/carry/release intervals required')
        self.open = np.asarray(open_fingers, float); self.closed = np.asarray(closed_fingers, float)
        self.initial_reference0 = np.asarray(initial_reference_object0, float)
        if (self.open.shape != self.fingers.shape[1:] or self.closed.shape != self.open.shape
                or not np.isfinite(self.open).all() or not np.isfinite(self.closed).all()
                or self.initial_reference0.shape != (7,) or not np.isfinite(self.initial_reference0).all()
                or np.linalg.norm(self.initial_reference0[3:]) < 1e-8):
            raise ValueError('valid named finger postures and initial object0 pose required')
        settings = [fps,max_speed,max_rotation_speed,finger_speed,seating_position,seating_rotation,seating_duration]
        if not np.isfinite(settings).all() or min(settings) <= 0:
            raise ValueError('positive finite feedback/rate settings required')
        self.bounds, self.contact, self.carry_blend = bounds, contact, carry_blend
        self.finger_bounds, self.release_start = finger_bounds, release_start
        self.fps, self.max_speed, self.max_rotation_speed, self.finger_speed = fps,max_speed,max_rotation_speed,finger_speed
        self.seating_position, self.seating_rotation = seating_position,seating_rotation
        self.seating_frames = max(1, round(seating_duration*fps)); self.gated = gated_closure
        self.model, self.position_model = model, position_model
        self.reset()

    def reset(self):
        self.last_frame = -1; self.initial0 = None; self.last_goal = None; self.previous_goal = None
        self.seating_count = 0; self.seated_at = -1; self.closed_at = -1

    def step(self, frame, object_before, wrist_before, initial_object0_before, finger_command_before):
        if frame == 0:
            self.reset()
        if frame != self.last_frame+1 or not 0 <= frame < len(self.planned):
            raise ValueError('causal controller requires contiguous pre-action frames')
        for pose in [object_before, wrist_before, initial_object0_before]:
            if np.shape(pose) != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
                raise ValueError('finite measured BEFORE poses required')
        if frame == 1:
            # AFTER command0 is the official first-pose alignment anchor.
            self.initial0 = np.array(initial_object0_before, copy=True)
        a,b = self.bounds; c,d = self.contact; s,e = self.carry_blend
        goal = self.planned[frame].copy(); nominal = self.nominal[frame].copy()
        if a < frame < b:
            if self.initial0 is None:raise ValueError('missing initial execution frame')
            acquisition = correct_planned_hand_target(self.planned[frame], self.task[frame-1], object_before)
            carry = task_trajectory_in_execution_frame(self.planned[frame:frame+1], self.initial0, self.initial_reference0)[0]
            desired = blend_pose(acquisition, carry, smooth((frame-s)/(e-s)))
            influence = smooth((frame-a)/(c-a))*smooth((b-frame)/(b-d))
            goal = blend_pose(self.planned[frame], desired, influence)
            last = self.last_goal
            displacement = np.linalg.norm(goal[:3]-last[:3]); angle = (Rotation.from_quat(goal[3:])*Rotation.from_quat(last[3:]).inv()).magnitude()
            fraction = min(1., self.max_speed/self.fps/max(displacement,1e-12), self.max_rotation_speed/self.fps/max(angle,1e-12))
            goal = blend_pose(last, goal, fraction)
            command = goal.copy()
            if self.model is not None:
                command = self.model.inverse(last, goal)
                if self.position_model is not None:
                    command[:3] = self.position_model.inverse(self.previous_goal[:3], last[:3], goal[:3])
            nominal = blend_pose(self.nominal[frame], command, influence)
        finger = self.fingers[frame].copy()
        fa,fb = self.finger_bounds
        if self.gated and fa < frame < fb:
            if c <= frame < self.release_start:
                # Seat against the instantaneous object, not a future state or
                # the task pose. The command-i goal still contains planned lead.
                seat = correct_planned_hand_target(self.planned[frame], self.task[frame], object_before)
                dp = np.linalg.norm(np.asarray(wrist_before)[:3]-seat[:3])
                dr = (Rotation.from_quat(wrist_before[3:])*Rotation.from_quat(seat[3:]).inv()).magnitude()
                self.seating_count = self.seating_count+1 if dp < self.seating_position and dr < self.seating_rotation else 0
                if self.seated_at < 0 and self.seating_count >= self.seating_frames:self.seated_at = frame
            if frame < self.release_start:
                finger = self.open if self.seated_at < 0 else self.closed
            before = np.asarray(finger_command_before, float)
            if before.shape != finger.shape or not np.isfinite(before).all():raise ValueError('finite previous finger commands required')
            finger = before + np.clip(finger-before, -self.finger_speed/self.fps, self.finger_speed/self.fps)
            if self.seated_at >= 0 and self.closed_at < 0 and np.max(np.abs(finger-self.closed)) < 1e-6:self.closed_at = frame
        self.previous_goal = self.last_goal.copy() if self.last_goal is not None else goal.copy()
        self.last_goal = goal.copy(); self.last_frame = frame
        return nominal, finger, dict(goal=goal, seated_at=self.seated_at, closed_at=self.closed_at)
