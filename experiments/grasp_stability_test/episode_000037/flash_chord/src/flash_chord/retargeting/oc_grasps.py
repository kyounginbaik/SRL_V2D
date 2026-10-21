"""Object-centric grasp search space for the UniRetarget upper level.

Candidates (tools/object_centric_grasps.py): per (hand, contact segment) and prototype k, a palm pose in the grasped
object's frame + a finger posture, fitted to the reconstruction's contacts. Physical feedback, not label fit, chooses.
Per (hand, segment) action in [-1, 1]:
  K logits -> w = softmax(temperature * logits); candidate = argmax w (hard choice; palm poses are not averaged)
  alpha    = max(0, a)        blend strength; alpha <= 0 KEEPS the currently committed grasp of that segment
                              (accepted grasps persist across rounds; see commit())
  squeeze  = 0.5 * max(0, a)  flexion past the posture (position-controlled grip force)
  aperture = 0.4 * max(0, a)  opening during approach
  lead     = 0.3 * (a+1)/2 s  when closing starts after contact onset (closing lasts 0.3 s)
Palm target in the window: T_obj(t) * T_palm_in_obj[k]; outside windows: unchanged. Zero action = unchanged.
"""
import numpy as np
from scipy.spatial.transform import Rotation


class ObjectCentricGrasps:
    def __init__(self, grasp_path, reference, joint_names, qids, ball_q_starts, fps, num_frames, *, temperature=8.0,
                 structured_fraction=.25, seed=0, preshape_s=.4, release_s=.3, close_s=.3):
        g = np.load(grasp_path)
        self.names = [str(n) for n in g['prototype_names']]
        self.K = len(self.names)
        self.temperature, self.structured_fraction = float(temperature), float(structured_fraction)
        self.rng = np.random.default_rng(seed)
        F, self.fps = int(num_frames), float(fps)
        order = ['thumb_CMC_FE', 'thumb_CMC_AA', 'thumb_MCP_FE', 'thumb_MCP_AA', 'thumb_IP', 'index_MCP_FE', 'index_MCP_AA', 'index_PIP', 'index_DIP',
                 'middle_MCP_FE', 'middle_MCP_AA', 'middle_PIP', 'middle_DIP', 'ring_MCP_FE', 'ring_MCP_AA', 'ring_PIP', 'ring_DIP',
                 'pinky_CMC', 'pinky_MCP_FE', 'pinky_MCP_AA', 'pinky_PIP', 'pinky_DIP']
        self.flex = np.array([not (j.endswith('AA') or j in ('thumb_CMC_FE', 'pinky_CMC')) for j in order], float)
        self.obj_pos = np.asarray(reference.object_body_pos_w(), np.float64)
        self.obj_quat = np.asarray(reference.object_body_quat_w(), np.float64)[..., [1, 2, 3, 0]]
        self.close_frames = max(1, int(round(close_s * fps)))
        t = np.arange(F)
        self.t = t
        self.segments = []
        pre, rel = int(round(preshape_s * fps)), int(round(release_s * fps))
        for key in sorted(k[:-5] for k in g.files if k.endswith('_meta')):
            start, end, kf, o = map(int, g[key + '_meta'])
            side = key.split('_')[0]
            env = np.zeros(F)
            env[start:end + 1] = 1
            up = (t >= start - pre) & (t < start); env[up] = .5 - .5 * np.cos(np.pi * (t[up] - (start - pre)) / pre)
            dn = (t > end) & (t <= end + rel); env[dn] = .5 + .5 * np.cos(np.pi * (t[dn] - end) / rel)
            env[[0, -1]] = 0
            self.segments.append(dict(
                key=key, side=side, start=start, end=end, object=o, env=env,
                pos_cols=[qids[joint_names.index(f'{side}_wrist_{a}')] for a in 'xyz'], ball=ball_q_starts[side],
                finger_cols=[qids[joint_names.index(f'{side}_{j}')] for j in order],
                palm_pos=g[key + '_palm_in_obj_pos'].astype(np.float64),
                palm_quat=g[key + '_palm_in_obj_quat_xyzw'].astype(np.float64),
                fingers=g[key + '_fingers'].astype(np.float64)))
        self.per_segment = self.K + 4
        self.action_dim = len(self.segments) * self.per_segment
        self.current = [None] * len(self.segments)   # committed (accepted) grasp per segment

    def decode(self, extra):
        a = np.clip(np.asarray(extra, np.float64), -1, 1).reshape(len(self.segments), self.per_segment)
        out = []
        for row in a:
            logits = self.temperature * row[:self.K]
            out.append(dict(k=int(np.argmax(logits)), alpha=max(0., row[self.K]), squeeze=.5 * max(0., row[self.K + 1]),
                            aperture=.4 * max(0., row[self.K + 2]), lead=.3 * (row[self.K + 3] + 1) / 2))
        return out

    def effective(self, extra):
        """Decoded choices with alpha<=0 replaced by the committed grasp (or None = no grasp)."""
        return [d if d['alpha'] > 0 else c for d, c in zip(self.decode(extra), self.current)]

    def commit(self, extra):
        self.current = [dict(d) if d is not None else None for d in self.effective(extra)]

    def proposal(self, extra, base_q):
        """Full joint_q proposal (frames x nq) toward the chosen (or committed) object-centric grasps."""
        q = np.array(base_q, dtype=np.float64, copy=True)
        for seg, d in zip(self.segments, self.effective(extra)):
            if d is None or d['alpha'] <= 0:
                continue
            k, o, env = d['k'], seg['object'], d['alpha'] * seg['env']
            R_obj = Rotation.from_quat(self.obj_quat[:, o])
            target_pos = self.obj_pos[:, o] + R_obj.apply(seg['palm_pos'][k])
            target_rot = R_obj * Rotation.from_quat(seg['palm_quat'][k])
            pc, b = seg['pos_cols'], seg['ball']
            q[:, pc] = (1 - env[:, None]) * q[:, pc] + env[:, None] * target_pos
            base_rot = Rotation.from_quat(q[:, b:b + 4])
            q[:, b:b + 4] = (Rotation.from_rotvec((target_rot * base_rot.inv()).as_rotvec() * env[:, None]) * base_rot).as_quat()
            closed = seg['fingers'][k] + d['squeeze'] * self.flex
            opened = seg['fingers'][k] - d['aperture'] * self.flex
            c0 = seg['start'] + int(round(d['lead'] * self.fps))
            closing = np.clip((self.t - c0) / self.close_frames, 0, 1)
            target = opened[None] + closing[:, None] * (closed - opened)[None]
            fc = seg['finger_cols']
            q[:, fc] = (1 - env[:, None]) * q[:, fc] + env[:, None] * target
        return q

    def structured(self, actions, offset):
        actions = np.array(actions, copy=True)
        rows = int(round(self.structured_fraction * len(actions)))
        extra = actions[:rows, offset:].reshape(rows, len(self.segments), self.per_segment)
        for r in range(rows):
            for s in range(len(self.segments)):
                extra[r, s, :self.K] = -1.0
                extra[r, s, self.rng.integers(self.K)] = 1.0
                extra[r, s, self.K] = self.rng.choice([.5, .8, 1.0])
                extra[r, s, self.K + 1:self.K + 4] = self.rng.uniform(-.2, 1, size=3)
        actions[:rows, offset:] = extra.reshape(rows, -1)
        return actions

    def describe(self, extra):
        return [dict(segment=s['key'], grasp=None if d is None else self.names[d['k']],
                     alpha=None if d is None else round(d['alpha'], 3), squeeze=None if d is None else round(d['squeeze'], 3),
                     aperture=None if d is None else round(d['aperture'], 3), lead_s=None if d is None else round(d['lead'], 3))
                for s, d in zip(self.segments, self.effective(extra))]
