"""Slow neural trajectory editor / fast motor FlashSAC, with fixed object targets.

The upper problem is deliberately an episodic contextual bandit: one action
specifies a whole smooth robot-reference correction, one frozen-controller
Newton rollout returns its reward, and the episode ends. FlashSAC's critic and
actor learn from that reward (including wrench support), NOT simulator gradients.
This is alternating two-time-scale learning, not an exact bilevel hypergradient
or a claim of TTSA convergence. A penetration-guided proposal may join the
upper actor's candidates, but only a fresh physical validation can publish it.
"""
from dataclasses import replace
from copy import deepcopy
import json
import time

import jax
import jax.numpy as jnp
import newton
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from flash_chord.embodiments.binding import build_robot_reference_binding
from flash_chord.envs.rl import RLEnv
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import (
    TrainingConfig, NetworkConfig, ReplayConfig, ExplorationConfig,
    OptimizationConfig, RewardNormalizationConfig,
)
from flash_chord.training.flash_sac.learner import create_learner, Transition
from flash_chord.training.flash_sac.replay import clear_replay
from flash_chord.training.flash_sac.checkpoint import ActorInferenceState, save_learner_checkpoint
from flash_chord.training.flash_sac.evaluation import create_actor_template, compile_policy_action
from flash_chord.retargeting.manipulation_bilevel import (
    ManipulationBilevel, assert_device_binding, passes_task_gate,
)
from flash_chord.retargeting.reference_snapshot import command_digest, save_snapshot
from flash_chord.retargeting.penetration import VisualHullPenetration, penetration_summary
from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY
from flash_chord.retargeting.granularity import granularity_basis


def temporal_basis(frames):
    """Three smooth modes; exactly zero at both episode boundaries."""
    phase = np.linspace(0, 1, frames, dtype=np.float32)
    return np.stack([np.sin(np.pi * k * phase) for k in (1, 2, 3)], axis=1)


def study_basis(frames, variant, *, local_centers=None):
    """Basis for matched A/B/C shared-reference experiments.

    B deliberately has one degree of freedom per interior frame. The common
    projection below, not this basis, enforces the motion trust region.
    """
    if variant == "smooth":
        return temporal_basis(frames)
    if variant == "per_frame":
        return np.eye(frames, dtype=np.float32)[:, 1:-1]
    if variant == "global_local":
        if local_centers is None:
            return temporal_basis(frames)
        phase = np.linspace(0, 1, frames, dtype=np.float32)
        local = np.exp(-.5 * ((phase[:, None] - np.asarray(local_centers)[None]) / .08)**2)
        local *= np.sin(np.pi * phase)[:, None]
        return np.concatenate((temporal_basis(frames), local.astype(np.float32)), axis=1)
    raise ValueError(f"unknown editor variant: {variant}")


def choose_local_centers(frame_scores, *, count=4, exclusion_fraction=.10):
    """Select separated peaks of repeated signed error, away from endpoints."""
    scores = np.asarray(frame_scores, dtype=np.float64).copy()
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("finite frame scores required")
    margin = max(1, int(len(scores) * .04))
    separation = max(1, int(len(scores) * exclusion_fraction))
    scores[:margin] = -np.inf
    scores[-margin:] = -np.inf
    centers = []
    for _ in range(count):
        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]) or scores[index] <= 0:
            break
        centers.append(index / (len(scores)-1))
        scores[max(0,index-separation):min(len(scores),index+separation+1)] = -np.inf
    return sorted(centers)


def robust_world_summary(position_error, orientation_error, wrench_error=None, wrench_active=None):
    """Equal-weight worlds, with a declared finite cap for exploding rollouts."""
    pos = np.asarray(position_error, dtype=np.float64)
    ori = np.asarray(orientation_error, dtype=np.float64)
    if pos.ndim != 2 or ori.shape != pos.shape:
        raise ValueError("expected matching time x world error arrays")
    if wrench_error is None:
        wrench = np.zeros_like(pos)
    else:
        wrench = np.asarray(wrench_error, dtype=np.float64)
        if wrench.shape != pos.shape:
            raise ValueError("wrench shape must match position error")
    if wrench_active is None:
        active = np.ones_like(pos, dtype=bool)
    else:
        active = np.asarray(wrench_active, dtype=bool)
        if active.shape != pos.shape:
            raise ValueError("wrench active mask must match position error")
    # One catastrophic physics world must count as a failure, not dominate the
    # numerical scale for every other world or destroy candidate ranking.
    pos = np.nan_to_num(pos, nan=1., posinf=1., neginf=1.)
    ori = np.nan_to_num(ori, nan=np.pi, posinf=np.pi, neginf=np.pi)
    wrench = np.nan_to_num(wrench, nan=10., posinf=10., neginf=10.)
    world_position = np.mean(np.clip(pos, 0, 1.), axis=0)
    world_orientation = np.mean(np.clip(ori, 0, np.pi), axis=0)
    world_wrench = np.sum(np.where(active,np.clip(wrench,0,10.),0.),axis=0) / np.maximum(active.sum(axis=0),1)
    world_success = np.mean((pos < .1) & (ori < .5), axis=0)
    return dict(object_position_robust_m=float(np.mean(world_position)),
                object_orientation_robust_rad=float(np.mean(world_orientation)),
                wrench_robust_mse=float(np.mean(world_wrench)),
                object_success_robust_fraction=float(np.mean(world_success)),
                unstable_world_fraction=float(np.mean(np.max(pos, axis=0) > 1.)))


def signed_feedback_summary(residual, *, bins=16):
    """Time-resolved mean AND spread of hand-object error across worlds.

    residual: time x world x (left/right keypoints) x xyz. Both directions
    survive aggregation, and large opposing errors show up in the spread.
    """
    error = np.asarray(residual, dtype=np.float64)
    if error.ndim != 4 or error.shape[-1] != 3 or error.shape[2] % 2:
        raise ValueError("expected time x world x even-keypoints x xyz")
    error = np.nan_to_num(error, nan=0., posinf=1., neginf=-1.)
    error = np.clip(error, -1., 1.)
    halves = np.stack((error[:, :, :error.shape[2]//2].mean(2),
                       error[:, :, error.shape[2]//2:].mean(2)), axis=2)
    world_mean = halves.mean(axis=1)
    world_std = halves.std(axis=1)
    pooled = []
    for frame_ids in np.array_split(np.arange(len(error)), bins):
        pooled.append(np.stack((world_mean[frame_ids].mean(0), world_std[frame_ids].mean(0))))
    return np.asarray(pooled, dtype=np.float32)  # bins x (mean,std) x hand x xyz


def contact_directed_action(direction_bins, joint_names, qids, joint_basis, translation_bound):
    """Put measured contact directions into the editable wrist channels.

    Each proposal still goes through the usual smoothing, bounds, physical
    rollout, and independent publication gate. This supplies an informative
    trial when a high-dimensional random upper action cannot find the object.
    """
    directions = np.asarray(direction_bins, dtype=np.float64)
    if directions.ndim != 4 or directions.shape[1] != 2 or directions.shape[3] != 3:
        raise ValueError("expected chunk x hand x object x xyz contact directions")
    if directions.shape[0] < 1 or not np.isfinite(directions).all() or translation_bound <= 0:
        raise ValueError("invalid contact directions or wrist bound")
    coefficients = np.zeros((len(directions), len(joint_basis)), dtype=np.float32)
    for hand, side in enumerate(("left", "right")):
        for axis, letter in enumerate("xyz"):
            name = f"{side}_wrist_{letter}"
            if name not in joint_names:
                raise ValueError(f"missing floating wrist channel: {name}")
            qid = qids[joint_names.index(name)]
            channels = np.flatnonzero(np.asarray(joint_basis)[:, qid])
            if len(channels) != 1:
                raise ValueError(f"ambiguous wrist action channel: {name}")
            for chunk in range(len(directions)):
                vectors = directions[chunk, hand]
                active = np.linalg.norm(vectors, axis=-1) > 1e-8
                if active.any():
                    displacement = vectors[active].mean(axis=0)[axis]
                    coefficients[chunk, channels[0]] = np.clip(
                        displacement / translation_bound, -1.0, 1.0)
    return coefficients.reshape(-1)


def consensus_frame_scores(residual):
    """Signed systematic error discounted by cross-world disagreement."""
    error=np.asarray(residual,dtype=np.float64)
    if error.ndim!=4 or error.shape[-1]!=3:
        raise ValueError("expected time x world x keypoint x xyz residual")
    error=np.clip(np.nan_to_num(error,nan=0.,posinf=1.,neginf=-1.),-1.,1.)
    systematic=np.linalg.norm(error.mean(axis=1),axis=-1)
    world_rms=np.sqrt(np.mean(np.sum(error**2,axis=-1),axis=1))
    return np.mean(systematic**2/np.maximum(world_rms,1e-6),axis=-1)


def bounded_correction(action, basis, joint_basis, *, bound, fps, max_velocity):
    coefficients = np.asarray(action, dtype=np.float32).reshape(basis.shape[1], joint_basis.shape[0])
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("nonfinite upper action")
    delta = basis @ np.clip(coefficients, -1, 1) @ joint_basis
    # Global rescaling preserves smoothness; unlike per-frame clipping it adds no corners.
    scale = min(1., bound / max(float(np.max(np.abs(delta))), 1e-9))
    velocity = float(np.max(np.abs(np.diff(delta, axis=0))) * fps)
    scale = min(scale, max_velocity / max(velocity, 1e-9))
    delta *= scale
    delta[[0, -1]] = 0
    return delta


def upper_cost(metrics, delta, fps, wrench_weight, *, robust=False, penetration_weight=0.0,
               contact_proximity_weight=0.0):
    """Dimensionless task cost, with the SAME fixed scales in both adaptive arms."""
    position = metrics["object_position_robust_m"] if robust else metrics["object_position_mean_m"]
    orientation = metrics["object_orientation_robust_rad"] if robust else metrics["object_orientation_mean_rad"]
    success = metrics["object_success_robust_fraction"] if robust else metrics["object_success_fraction"]
    wrench = metrics["wrench_robust_mse"] if robust else metrics["wrench_mse"]
    penetration = 0.0
    if penetration_weight:
        # The reference term is the actual UniRetarget objective; the
        # rollout term prevents an editor from moving overlap into physics.
        penetration = penetration_weight * (
            metrics["reference_penetration_mean_m"] / .005
            + .5 * metrics["rollout_penetration_mean_m"] / .005
        )
    proximity = 0.0
    if contact_proximity_weight:
        proximity = contact_proximity_weight * metrics["contact_proximity_mean_m"] / .05
    return (position / .03
            + orientation / .3
            + 2 * (1 - success)
            + wrench_weight * wrench
            + proximity
            + penetration
            + (3 * metrics["unstable_world_fraction"] if robust else 0)
            + .05 * float(np.mean((delta / .12)**2))
            + .05 * float(np.mean((np.diff(delta, axis=0) * fps / .5)**2)))


def contact_proximity_gate(candidate, baseline, minimum_gain_m=.001, per_object_tolerance_m=.002):
    if candidate["contact_proximity_mean_m"] > baseline["contact_proximity_mean_m"] - minimum_gain_m:
        return False
    for after, before in zip(candidate["per_object_contact_proximity_mean_m"],
                             baseline["per_object_contact_proximity_mean_m"], strict=True):
        if before is None:
            if after is not None:
                return False
        elif after is None or after > before + per_object_tolerance_m:
            return False
    for after_hand, before_hand in zip(candidate["per_hand_object_contact_proximity_mean_m"],
                                       baseline["per_hand_object_contact_proximity_mean_m"], strict=True):
        for after, before in zip(after_hand, before_hand, strict=True):
            if before is None:
                if after is not None:
                    return False
            elif after is None or after > before + per_object_tolerance_m:
                return False
    return True


def passes_study_gate(candidate, baseline, tolerance_m=.002, *, penetration_aware=False,
                      rollout_penetration_tolerance_m=.0001):
    """Do not publish a candidate by sacrificing object tracking or stability."""
    task_gate = (candidate["object_position_robust_m"] <= baseline["object_position_robust_m"] + tolerance_m
            and candidate["object_orientation_robust_rad"] <= baseline["object_orientation_robust_rad"] + .01
            and candidate["object_success_robust_fraction"] >= baseline["object_success_robust_fraction"] - .01
            and candidate["unstable_world_fraction"] <= baseline["unstable_world_fraction"])
    for key, tolerance in (("per_object_position_mean_m", tolerance_m),
                           ("per_object_orientation_mean_rad", .01)):
        if key in baseline:
            if key not in candidate:
                return False
            cand, base = np.asarray(candidate[key]), np.asarray(baseline[key])
            task_gate = task_gate and cand.shape == base.shape and bool(np.all(cand <= base + tolerance))
    if not penetration_aware:
        return task_gate
    if not 0 <= rollout_penetration_tolerance_m <= .001:
        raise ValueError("rollout penetration tolerance must be in [0, 1] mm")
    return (task_gate
            and candidate["reference_penetration_mean_m"] <= baseline["reference_penetration_mean_m"] + .0005
            and candidate["reference_penetration_peak_m"] <= baseline["reference_penetration_peak_m"] + .0005
            and candidate["rollout_penetration_mean_m"] <= baseline["rollout_penetration_mean_m"] + rollout_penetration_tolerance_m)


class ProposalRejected(ValueError):
    """An unsafe finite proposal; retain the published reference and keep training."""


def project_reference(base, delta, qids, lower, upper, fps):
    """Clip joint limits, then contract toward the feasible immutable baseline.

    Clipping q can introduce sharp changes in q-base when the baseline moves
    near a limit. Recheck the *post-clipping* correction and globally contract
    it. Convex contraction preserves joint limits without increasing velocity.
    """
    base = np.asarray(base, dtype=np.float64)
    if not np.all(np.isfinite(delta)):
        raise ValueError("nonfinite reference correction")
    if np.any(base[:, qids] < lower - 1e-6) or np.any(base[:, qids] > upper + 1e-6):
        raise ValueError("immutable source reference violates joint limits")
    q = base + delta
    q[:, qids] = np.clip(q[:, qids], lower, upper)
    q[[0, -1]] = base[[0, -1]]
    correction = q - base
    peak = float(np.max(np.abs(correction)))
    speed = float(np.max(np.abs(np.diff(correction, axis=0))) * fps)
    scale = min(1., .11999 / max(peak, 1e-12), .4999 / max(speed, 1e-12))
    q = (base + scale * correction).astype(np.float32)
    correction = q - base
    if (np.max(np.abs(correction)) > .12001
            or np.max(np.abs(np.diff(correction, axis=0))) * fps > .50001
            or np.any(q[:, qids] < lower - 1e-6)
            or np.any(q[:, qids] > upper + 1e-6)):
        raise ProposalRejected("projected proposal failed joint/offset/velocity bounds")
    return q


class TTSAReferenceEditor(ManipulationBilevel):
    def publish(self, step):
        path=self.output/f"reference_{self.version:04d}.npz"
        checksum=save_snapshot(path,self.binding,self.setup.scene.layout,provenance=dict(
            mode=self.mode,engine=self.options.get("engine", "ttsa_flashsac"),version=self.version,environment_steps=step,
            source_parquet=self.config["task"]["parquet"],object_contact_sha256=self.command_sha256))
        self.config["task"]["robot_reference_path"]=str(path)
        self.config["task"]["robot_reference_sha256"]=checksum
        self.metadata[CONFIG_METADATA_KEY]=json.dumps(self.config,sort_keys=True)
        self.metadata["robot_reference_version"]=str(self.version)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.penetration_aware = bool(self.options.get("penetration_aware", False))
        self.guided_penetration_proposal = bool(self.options.get("guided_penetration_proposal", False))
        self.select_best_trial = bool(self.options.get("select_best_trial", False))
        self.penetration_weight = float(self.options.get("penetration_weight", 1.0)) if self.penetration_aware else 0.0
        self.contact_proximity_weight = float(self.options.get("contact_proximity_weight", 0.0))
        self.guided_contact_proposal = bool(self.options.get("guided_contact_proposal", False))
        if not np.isfinite(self.contact_proximity_weight) or self.contact_proximity_weight < 0:
            raise ValueError("contact_proximity_weight must be finite and nonnegative")
        if self.penetration_weight < 0 or not np.isfinite(self.penetration_weight):
            raise ValueError("penetration_weight must be finite and nonnegative")
        self.study_variant = self.options.get("study_variant")
        if self.guided_contact_proposal and (not self.contact_proximity_weight or self.study_variant != "chunk"):
            raise ValueError("contact-guided proposal requires chunked contact proximity")
        if self.study_variant not in (None, "smooth", "per_frame", "global_local", "episode", "timestep", "chunk"):
            raise ValueError("unknown study_variant")
        if self.penetration_aware and self.study_variant is None:
            raise ValueError("penetration-aware editor requires a time-resolved study_variant")
        if self.penetration_aware and self.options.get("penetration_measure", "collision") not in ("collision", "visual_hull"):
            raise ValueError("penetration_measure must be collision or visual_hull")
        if self.guided_penetration_proposal and (not self.penetration_aware
                or self.options.get("penetration_measure") != "visual_hull"
                or self.study_variant not in ("smooth", "global_local")
                or not self.select_best_trial):
            raise ValueError("guided proposal requires visual-hull penetration, smooth/global_local basis, and best-trial selection")
        self.local_centers = None
        self.trials = int(self.options.get("upper_trials", 8))
        self.blend = float(self.options.get("blend", .25))
        if self.trials < 4 or not 0 < self.blend <= 1:
            raise ValueError("upper_trials >= 4 and blend in (0,1] required")
        self.upper = None
        self.upper_round = 0
        if self.projector is not None:
            names = self.projector.layout.scalar_joints.names
            rows = []
            # The original 24-channel arm/finger basis is specific to the
            # Vega/Sharpa pilot. For other embodiments (including G1+Dex3),
            # give each named actuated joint one correction channel. This
            # keeps chunk-wise reference editing usable without silently
            # dropping legs, torso, or embodiment-specific hand joints.
            is_vega_sharpa = any(name.startswith("L_arm_") for name in names) and any(
                name.startswith("R_arm_") for name in names
            )
            if is_vega_sharpa:
                for side, prefix in (("left", "L_arm_"), ("right", "R_arm_")):
                    # Seven independent arm corrections, plus five finger flexion synergies.
                    for index, name in enumerate(names):
                        if name.startswith(prefix):
                            row = np.zeros(self.baseline.num_joint_q, np.float32)
                            row[self.projector.qids[index]] = 1
                            rows.append(row)
                    for digit in ("thumb", "index", "middle", "ring", "pinky"):
                        row = np.zeros(self.baseline.num_joint_q, np.float32)
                        for index, name in enumerate(names):
                            if name.startswith(f"{side}_{digit}_") and not name.endswith("AA"):
                                row[self.projector.qids[index]] = 1
                        if not np.any(row):
                            raise ValueError(f"missing finger synergy: {side}/{digit}")
                        rows.append(row)
                self.joint_basis_kind = "vega_sharpa_arm_and_finger_synergies"
            else:
                for index, _name in enumerate(names):
                    row = np.zeros(self.baseline.num_joint_q, np.float32)
                    row[self.projector.qids[index]] = 1
                    rows.append(row)
                self.joint_basis_kind = "all_named_scalar_joints"
                if self.guided_penetration_proposal:
                    raise ValueError(
                        "geometry-guided finger proposals require the Vega/Sharpa finger-synergy basis"
                    )
            if not rows:
                raise ValueError("reference editor has no actuated joint correction channels")
            for q_start, _ in self.projector.ball_joints:
                for axis in range(3):
                    row = np.zeros(self.baseline.num_joint_q, np.float32)
                    row[q_start + axis] = 1
                    rows.append(row)
            if self.options.get("edit_scope") == "wrists_only":
                wrist_ids = {self.projector.qids[i] for i,name in enumerate(names)
                             if name.endswith(("wrist_x", "wrist_y", "wrist_z"))}
                wrist_ids.update(q_start+axis for q_start,_ in self.projector.ball_joints for axis in range(3))
                rows = [row for row in rows if set(np.flatnonzero(row)).issubset(wrist_ids)]
                if len(rows) != 12: raise ValueError("Expected 12 floating-wrist correction channels")
            self.joint_basis = np.stack(rows)
            if self.projector.ball_joints:
                self.joint_basis_kind = "floating_wrist_only" if self.options.get("edit_scope") == "wrists_only" else "floating_wrist_translation_rotation_and_fingers"
            if self.study_variant == "global_local":
                # Keep the upper action dimension fixed. These placeholders are
                # masked in round one and replaced only after the global stage.
                self.basis = study_basis(self.baseline.num_frames,"global_local",
                                         local_centers=[.2,.4,.6,.8])
            elif self.study_variant in ("episode", "timestep", "chunk"):
                self.basis = granularity_basis(self.baseline.num_frames, self.study_variant,
                    fps=self.baseline.fps, chunk_seconds=float(self.options.get("chunk_seconds", .5)),
                    smoothing_seconds=float(self.options.get("temporal_smoothing_seconds", .1)),
                    boundary_seconds=float(self.options.get("boundary_seconds", .25)))
            else:
                self.basis = study_basis(self.baseline.num_frames, self.study_variant or "smooth")
        manifest = dict(engine="upper_flashsac_contextual_bandit", lower="FlashSAC",
            object_reference="fixed", contact_reference="fixed", warmup=self.next_update,
            lower_block_transitions=self.interval, upper_trials=self.trials,
            joint_basis_kind=getattr(self, "joint_basis_kind", None),
            joint_basis_names=list(self.projector.layout.scalar_joints.names) if self.projector is not None else [],
            upper_replay="cleared on lower-policy version change", lower_replay="cleared on every matched block boundary",
            publication_blend=self.blend, max_joint_offset_rad=float(self.options.get("finger_offset_rad",.52)) if self.projector is not None and self.projector.ball_joints else .12, max_added_velocity_rad_s=float(self.options.get("finger_correction_speed_rad_s",2.)) if self.projector is not None and self.projector.ball_joints else .5,
            proposal_projection="joint_clip_then_global_contraction_v2",
            wrench="contact-wrench SUPPORT mismatch, not measured net wrench",
            upper_evaluation_voc=0., exact_hypergradient=False)
        manifest.update(study_variant=self.study_variant, robust_upper_score=bool(self.study_variant),
                        signed_feedback_bins=16 if self.study_variant else 0,
                        per_frame_action_smoothing_sigma_frames=2 if self.study_variant=="per_frame" else 0,
                        local_center_selection="after_global_round_from_remaining_signed_error" if self.study_variant=="global_local" else None)
        if self.projector is not None and self.study_variant in ("episode", "timestep", "chunk"):
            manifest.update(temporal_modes=int(self.basis.shape[1]),
                upper_action_dimension=int(self.basis.shape[1]*self.joint_basis.shape[0]),
                chunk_seconds=float(self.options.get("chunk_seconds", .5)) if self.study_variant=="chunk" else None,
                temporal_smoothing_seconds=float(self.options.get("temporal_smoothing_seconds", .1)),
                boundary_seconds=float(self.options.get("boundary_seconds", .25)))
        manifest.update(floating_wrist_translation_bound_m=float(self.options.get("wrist_translation_offset_m", .03)) if self.projector and self.projector.ball_joints else None,
                        floating_wrist_rotation_bound_rad=float(self.options.get("wrist_rotation_offset_rad", .52)) if self.projector and self.projector.ball_joints else None,
                        per_object_acceptance_gate=True,
                        penetration_aware=self.penetration_aware,
                        contact_proximity_weight=self.contact_proximity_weight,
                        contact_proximity_geometry="labeled object-attached points to intended hand fingertips" if self.contact_proximity_weight else None,
                        contact_proximity_cap_m=2.0 if self.contact_proximity_weight else None,
                        contact_proximity_pooling="equal active hand/object" if self.contact_proximity_weight else None,
                        contact_direction_feedback="per chunk, hand, object, xyz" if self.contact_proximity_weight else None,
                        guided_contact_proposal=self.guided_contact_proposal,
                        guided_penetration_proposal=self.guided_penetration_proposal,
                        select_best_trial=self.select_best_trial,
                        penetration_weight=self.penetration_weight,
                        penetration_measure=self.options.get("penetration_measure", "collision"),
                        penetration_reference_stride=int(self.options.get("penetration_reference_stride",4)),
                        penetration_rollout_stride=int(self.options.get("penetration_rollout_stride",4)),
                        rollout_penetration_gate_tolerance_m=.0001,
                        penetration_geometry=("visible finger vertices against convex hull of visible object mesh"
                                              if self.options.get("penetration_measure") == "visual_hull"
                                              else "Newton hand-contact/object collision shapes; 2mm labeled-contact tolerance"))
        (self.output.parent / "ttsa_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

    def candidate_joint_q(self, action):
        if self.projector.ball_joints:
            from flash_chord.retargeting.floating_chunk import floating_candidate
            return floating_candidate(self, action)
        coefficients = np.asarray(action, dtype=np.float32).reshape(self.basis.shape[1], self.joint_basis.shape[0]).copy()
        if self.study_variant == "per_frame":
            coefficients = gaussian_filter1d(coefficients,2.,axis=0)
        if self.study_variant == "global_local" and self.upper_round <= 1:
            coefficients[3:] = 0  # First round is exactly the global editor.
        delta = bounded_correction(coefficients, self.basis, self.joint_basis,
            bound=.12, fps=self.baseline.fps, max_velocity=.5)
        delta = (1-self.blend) * (self.binding.joint_q-self.baseline.joint_q) + self.blend * delta
        return project_reference(self.baseline.joint_q, delta, self.projector.qids,
            self.projector.lower, self.projector.upper, self.baseline.fps)

    def make_binding(self, action):
        q = self.candidate_joint_q(action)
        target = self.baseline.joint_target.copy()
        target[:,self.projector.dids] = q[:,self.projector.qids]
        for q_start, dof_start in self.projector.ball_joints:
            target[:, dof_start:dof_start+3] = Rotation.from_quat(q[:, q_start:q_start+4]).as_rotvec()
        return build_robot_reference_binding(self.projector.builder,self.projector.layout,q,target,self.baseline.fps)

    def guided_action(self):
        """Search finger-synergy actions using static FK and the box visual proxy.

        This proposal uses no lower-policy reward. It consumes the same first
        upper trial as an actor proposal, and is scored by the frozen lower
        policy in Newton before it can be selected for validation.
        """
        stride = int(self.options.get("guided_search_stride", 32))
        amplitude = float(self.options.get("guided_action_amplitude", .32))
        if stride < 1 or not 0 < amplitude <= 1:
            raise ValueError("positive guided stride and action amplitude in (0,1] required")
        env = RLEnv(replace(self.eval_setup.scene, robot_reference=self.binding),
                    self.eval_setup.reference, config=deepcopy(self.eval_config))
        monitor = VisualHullPenetration(env.scene, samples_per_shape=int(
            self.options.get("penetration_samples_per_shape", 8)))
        frames = np.unique(np.r_[np.arange(0, self.baseline.num_frames, stride), self.baseline.num_frames-1])
        q_per_world = env.model.joint_coord_count // env.world_count
        robot_q = self.baseline.num_joint_q

        def measure(q):
            depths = np.empty((len(frames), env.world_count, len(monitor.sides)))
            for slot, frame in enumerate(frames):
                env.reset(frame_id=int(frame))
                state_q = env.state_0.joint_q.numpy()
                for world in range(env.world_count):
                    state_q[world*q_per_world:world*q_per_world+robot_q] = q[frame]
                env.state_0.joint_q.assign(np.asarray(state_q, dtype=np.float32))
                newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)
                depths[slot] = monitor.frame_depths(env.state_0)
            return penetration_summary(depths)

        def score(metrics):
            return metrics["penetration_mean_m"]/.005 + .5*metrics["penetration_peak_m"]/.05

        original = measure(self.binding.joint_q)
        best_score, best_action, best_metrics = score(original), None, original
        best_choice = None
        modes = range(self.basis.shape[1]) if self.upper_round > 1 else range(min(3,self.basis.shape[1]))
        fingers = tuple(i for i in range(len(self.joint_basis)) if i % 12 >= 7)
        evaluated = 0
        for mode in modes:
            for finger in fingers:
                for sign in (-1.,1.):
                    action = np.zeros((self.basis.shape[1],len(self.joint_basis)),dtype=np.float32)
                    action[mode,finger] = sign*amplitude
                    try:
                        metrics = measure(self.candidate_joint_q(action))
                    except ProposalRejected:
                        continue
                    evaluated += 1
                    if metrics["penetration_peak_m"] > original["penetration_peak_m"] + .0005:
                        continue
                    if score(metrics) < best_score - 1e-4:
                        best_score, best_action, best_metrics = score(metrics), action.ravel(), metrics
                        best_choice = dict(mode=int(mode),finger_synergy=int(finger),sign=int(sign))
        return best_action, dict(baseline=original, candidate=best_metrics,
                                 evaluated=evaluated, improved=best_action is not None,
                                 choice=best_choice)

    def observation(self, metrics, feedback=None):
        # A trajectory editor sees the immutable source and current lower capability.
        # It does not observe simulator state at deployment or act as a motor policy.
        values = np.r_[self.baseline.joint_q.mean(0), self.baseline.joint_q.std(0),
            (self.binding.joint_q-self.baseline.joint_q).mean(0),
            (metrics["object_position_robust_m"] if self.study_variant else metrics["object_position_mean_m"])/.03,
            (metrics["object_orientation_robust_rad"] if self.study_variant else metrics["object_orientation_mean_rad"])/.3,
            metrics["object_success_robust_fraction"] if self.study_variant else metrics["object_success_fraction"],
            metrics["wrench_robust_mse"] if self.study_variant else metrics["wrench_mse"]]
        if self.study_variant:
            if feedback is None:
                raise ValueError("study variant requires signed hand-object feedback")
            current = self.binding.joint_q-self.baseline.joint_q
            if self.study_variant == "chunk":
                from flash_chord.retargeting.chunk_feedback import correction_chunk_means
                correction_bins=correction_chunk_means(current,self.baseline.fps,float(self.options.get("chunk_seconds",.5)))
                if len(correction_bins)!=len(feedback["signed_bins"]):
                    raise ValueError("Feedback and action chunks differ")
            else:
                correction_bins = np.stack([current[ids].mean(0) for ids in np.array_split(np.arange(len(current)),16)])
            values = np.r_[values, np.clip(feedback["signed_bins"],-.25,.25).ravel()/.05,
                np.clip(correction_bins,-.12,.12).ravel()/.12,
                metrics["object_position_robust_m"]/.03, metrics["object_orientation_robust_rad"]/.3,
                metrics["object_success_robust_fraction"],metrics["unstable_world_fraction"]]
        if self.study_variant == "chunk":
            counts=np.asarray(feedback["chunk_sample_counts"],dtype=float)
            values=np.r_[values,counts/max(float(counts.sum()),1.)]
        if self.contact_proximity_weight:
            values=np.r_[values, metrics["contact_proximity_mean_m"]/.05,
                [0.0 if value is None else value/.05 for value in metrics["per_object_contact_proximity_mean_m"]],
                [0.0 if value is None else value/.05 for row in metrics["per_hand_object_contact_proximity_mean_m"]
                 for value in row],
                np.clip(feedback["contact_direction_bins"], -.25, .25).ravel()/.05,
                feedback["contact_direction_counts"] / max(float(feedback["contact_direction_counts"].sum()), 1.)]
        if self.penetration_aware:
            values = np.r_[values,
                metrics["reference_penetration_mean_m"]/.005,
                metrics["rollout_penetration_mean_m"]/.005,
                metrics["reference_penetration_peak_m"]/.02,
                np.clip(feedback["penetration_bins"],0,.02).ravel()/.02]
        return jnp.asarray(values[None], dtype=jnp.float32)

    def prepare_upper(self, obs):
        worlds = getattr(self, "upper_worlds", 1)
        context = jnp.tile(jnp.asarray([[0.,0.,1.]], dtype=jnp.float32), (worlds, 1))
        if self.upper is None:
            self.upper_config = TrainingConfig(seed=self.training.seed+20000, world_count=worlds,
                total_environment_steps=((100000 + worlds - 1)//worlds)*worlds, updates_per_collection=int(self.options.get("upper_updates", 2)), mixed_precision=False,
                network=NetworkConfig(actor_blocks=2,actor_hidden_dim=64,critic_blocks=2,critic_hidden_dim=128,
                    actor_head_mode="residual",residual_mean_head="zero_dense",initial_normalized_std=float(self.options.get("upper_initial_std", .15))),
                optimization=OptimizationConfig(learning_rate_initial=1e-4,learning_rate_peak=1e-4,learning_rate_end=1e-4,
                    learning_rate_schedule_updates=100000),
                replay=ReplayConfig(capacity=max(1024, worlds*2),minimum_size=4,batch_size=min(256, max(16, worlds)),n_step=1),
                exploration=ExplorationConfig(warmup_action="policy",maximum_noise_repeat=1),
                reward_normalization=RewardNormalizationConfig(enabled=False,normalized_return_bound=10.))
            self.upper,self.executables = create_learner(self.upper_config, obs, context,
                action_dim=self.basis.shape[1]*self.joint_basis.shape[0],objective_term_count=1)
            self.upper_actor,_ = create_actor_template(self.upper_config,obs.shape[1],self.basis.shape[1]*self.joint_basis.shape[0])
            self.upper_policy = compile_policy_action(self.upper_actor,
                ActorInferenceState(self.upper.actor.params,self.upper.actor.batch_stats),obs,self.upper.key,deterministic=True)
        else:
            # The learner donates its state. Never alias its observation/context
            # with the separately supplied Transition buffers at a TTSA boundary.
            self.upper = self.upper.replace(replay=self.upper.replay if self.options.get("keep_frozen_replay", False) else clear_replay(self.upper.replay),
                observation=jnp.array(obs,copy=True),critic_context=jnp.array(context,copy=True))
        action, exploration, key = self.executables.initial_action(self.upper.actor.params,self.upper.actor.batch_stats,
            self.upper.replay.size,obs,self.upper.exploration,self.upper.key)
        self.upper = self.upper.replace(exploration=exploration,key=key)
        return action, context

    def __call__(self, env, learner, step, stage):
        if step < self.next_update or step >= self.training.total_environment_steps:
            return None
        self.next_update += self.interval
        self.upper_round += 1
        jax.block_until_ready(learner)
        started=time.perf_counter()
        previous_steps=self.validation_steps
        seed=self.training.seed+step
        if self.study_variant:
            baseline,_,_,feedback=self.rollout(self.binding,learner,seed,return_feedback=True)
            if self.study_variant == "global_local" and self.upper_round >= 2 and self.local_centers is None:
                centers = choose_local_centers(feedback["penetration_frame_scores"] if self.penetration_aware
                                               else feedback["frame_scores"])
                if len(centers) < 4:
                    centers = [.2,.4,.6,.8]
                self.local_centers = centers
                self.basis = study_basis(self.baseline.num_frames,"global_local",local_centers=centers)
        else:
            baseline,_,_=self.rollout(self.binding,learner,seed)
        weight=float(self.options.get("wrench_weight",1.)) if self.mode=="reactor_wrench" else 0.
        baseline_cost=upper_cost(baseline,self.binding.joint_q-self.baseline.joint_q,self.baseline.fps,weight,
                                 robust=bool(self.study_variant),penetration_weight=self.penetration_weight,
                                 contact_proximity_weight=self.contact_proximity_weight)
        record=dict(step=step,mode=self.mode,engine="upper_flashsac_contextual_bandit",
            version_before=self.version,baseline=baseline,candidates=[],upper_round=self.upper_round,
            lower_policy_frozen=True,object_contact_sha256=self.command_sha256)
        if self.study_variant:
            record.update(study_variant=self.study_variant, local_centers=self.local_centers,
                          signed_feedback_bins=feedback["signed_bins"].tolist())
            if self.contact_proximity_weight:
                record["contact_direction_bins"] = feedback["contact_direction_bins"].tolist()
        accepted=False
        best_trial = None
        best_trial_cost = baseline_cost
        if self.mode!="fixed":
            obs=self.observation(baseline,feedback if self.study_variant else None)
            action,context=self.prepare_upper(obs)
            original_params=jax.tree.map(lambda x:np.array(x,copy=True),self.upper.actor.params)
            if self.guided_penetration_proposal:
                guided, search = self.guided_action()
                record["guided_search"] = search
                if guided is not None:
                    action = jnp.asarray(guided[None], dtype=jnp.float32)
            if self.guided_contact_proposal:
                guided = contact_directed_action(
                    feedback["contact_direction_bins"],
                    list(self.projector.layout.scalar_joints.names), self.projector.qids,
                    self.joint_basis,
                    float(self.options.get("wrist_translation_offset_m", .03)),
                )
                action = jnp.asarray(guided[None], dtype=jnp.float32)
        # Fixed control matches rollout/reset overhead, but never modifies its reference.
        for trial in range(self.trials):
            if self.study_variant == "global_local" and self.upper_round == 1:
                # The action recorded in upper replay must match the applied
                # global-only proposal; otherwise the critic sees fictitious
                # variation in disabled local channels.
                action = action.at[:,3*self.joint_basis.shape[0]:].set(0)
            rejection = None
            try:
                candidate=self.binding if self.mode=="fixed" else self.make_binding(np.asarray(action)[0])
            except ProposalRejected as error:
                candidate=self.binding
                rejection=str(error)
            metrics,_,_=self.rollout(candidate,learner,seed)
            cost=upper_cost(metrics,candidate.joint_q-self.baseline.joint_q,self.baseline.fps,weight,
                            robust=bool(self.study_variant),penetration_weight=self.penetration_weight,
                            contact_proximity_weight=self.contact_proximity_weight)
            reward=float(np.clip(baseline_cost-cost,-5.,5.))
            if rejection is not None:
                reward=-5.0
            row=dict(trial=trial,reward=reward,metrics=metrics,cost=cost)
            if trial == 0 and self.guided_contact_proposal:
                row["proposal_kind"] = "contact_direction"
            elif trial == 0 and self.guided_penetration_proposal:
                row["proposal_kind"] = "guided" if guided is not None else "actor"
            if self.select_best_trial and rejection is None and cost < best_trial_cost:
                trial_gate = passes_study_gate(metrics,baseline,penetration_aware=self.penetration_aware)
                if self.contact_proximity_weight:
                    trial_gate = trial_gate and contact_proximity_gate(metrics,baseline)
                if trial_gate:
                    best_trial = (candidate, trial)
                    best_trial_cost = cost
                    row["best_training_candidate"] = True
            if rejection is not None:
                row["rejected"]=rejection
            if self.mode!="fixed":
                # Whole-trajectory outcome is terminal: no fictitious bootstrapping
                # across reference edits or across different frozen motor policies.
                transition=Transition(action=action,objective_terms=jnp.asarray([[reward]],jnp.float32),
                    terminated=jnp.ones(1,jnp.int32),truncated=jnp.zeros(1,jnp.int32),
                    episode_reference_progress=jnp.ones(1,jnp.float32),termination_diagnostics=jnp.zeros((1,0),jnp.float32),
                    next_observation=obs,next_critic_context=context,post_reset_observation=obs,post_reset_critic_context=context)
                self.upper,action,losses=self.executables.consume_transition(self.upper,transition,
                    jnp.ones(1,jnp.float32),jnp.float32(1),jnp.float32(0),jnp.bool_(False),jnp.bool_(True))
                jax.block_until_ready(self.upper)
                row.update(actor_loss=float(losses.actor_loss_mean_per_update),critic_loss=float(losses.critic_loss_mean_per_update),
                           updates=int(self.upper.global_update_step))
            record["candidates"].append(row)
            print(f"[ttsa] step={step} upper_trial={trial+1}/{self.trials} reward={reward:.5f}",flush=True)
        if self.contact_proximity_weight:
            distances=[row["metrics"]["contact_proximity_mean_m"] for row in record["candidates"]]
            record["contact_proximity_candidate_spread_m"]=float(np.ptp(distances))
            record["contact_proximity_baseline_m"]=baseline["contact_proximity_mean_m"]
        if self.mode!="fixed":
            rejection = None
            if self.select_best_trial and best_trial is not None:
                candidate, selected_trial = best_trial
                proposal_source = f"upper_trial_{selected_trial}"
            else:
                deterministic,_=self.upper_policy(ActorInferenceState(self.upper.actor.params,self.upper.actor.batch_stats),obs,self.upper.key)
                proposal_source = "deterministic_actor"
                try:
                    candidate=self.make_binding(np.asarray(deterministic)[0])
                except ProposalRejected as error:
                    candidate=self.binding
                    rejection=str(error)
            control,_,_=self.rollout(self.binding,learner,seed+100000)
            check,_,_=self.rollout(candidate,learner,seed+100000)
            gate = passes_study_gate if self.study_variant else passes_task_gate
            gate_passes = (gate(check,control,penetration_aware=self.penetration_aware)
                           if self.study_variant else gate(check,control))
            if self.contact_proximity_weight:
                # Publish only a contact-reaching edit; never trade away either
                # object's reach merely to improve another score component.
                gate_passes = gate_passes and contact_proximity_gate(check,control)
            accepted=(rejection is None and gate_passes and
                upper_cost(check,candidate.joint_q-self.baseline.joint_q,self.baseline.fps,weight,
                           robust=bool(self.study_variant),penetration_weight=self.penetration_weight,
                           contact_proximity_weight=self.contact_proximity_weight)<
                upper_cost(control,self.binding.joint_q-self.baseline.joint_q,self.baseline.fps,weight,
                           robust=bool(self.study_variant),penetration_weight=self.penetration_weight,
                           contact_proximity_weight=self.contact_proximity_weight))
            record["validation"]=dict(baseline=control,candidate=check,proposal_source=proposal_source)
            if self.contact_proximity_weight:
                record["validation"]["contact_proximity_gain_m"]=(
                    control["contact_proximity_mean_m"]-check["contact_proximity_mean_m"]
                )
                record["validation"]["contact_proximity_gate_passed"]=contact_proximity_gate(check,control)
            if rejection is not None:
                record["validation"]["rejected"]=rejection
            differences=jax.tree.leaves(jax.tree.map(lambda a,b:np.max(np.abs(np.asarray(a)-b)),self.upper.actor.params,original_params))
            record["upper_actor_parameter_max_change"]=float(max(differences))
            record["upper_optimizer_updates"]=int(self.upper.global_update_step)
            save_learner_checkpoint(self.output.parent/"upper",self.upper,metadata={"lower_step":step,"engine":"trajectory_bandit"},save_replay=True)
            if accepted:
                self.binding=candidate
                self.version+=1
                self.publish(step)
        else:
            self.rollout(self.binding,learner,seed+100000)
            self.rollout(self.binding,learner,seed+100000)
        if command_digest(self.setup.reference)!=self.command_sha256:
            raise RuntimeError("upper editor mutated immutable object/contact commands")
        record.update(accepted=bool(accepted),version_after=self.version,seconds=time.perf_counter()-started,
            upper_simulation_transitions=self.validation_steps-previous_steps,
            cumulative_upper_simulation_transitions=self.validation_steps,
            max_joint_change_rad=float(np.max(np.abs(self.binding.joint_q-self.baseline.joint_q))))
        with (self.output.parent/"upper_updates.jsonl").open("a") as stream:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
        print(f"[ttsa] step={step} mode={self.mode} accepted={accepted} version={self.version}",flush=True)
        if not accepted:
            return env
        replacement=RLEnv(replace(self.setup.scene,robot_reference=self.binding),self.setup.reference,config=deepcopy(env.env.config))
        assert_device_binding(replacement,self.binding)
        np.testing.assert_array_equal(replacement.command.reference.body_pos_w.numpy(),env.env.command.reference.body_pos_w.numpy())
        np.testing.assert_array_equal(replacement.command.reference.body_quat_w.numpy(),env.env.command.reference.body_quat_w.numpy())
        np.testing.assert_array_equal(replacement.objective.ref_wrench_support.numpy(),env.env.objective.ref_wrench_support.numpy())
        replacement.reset(); replacement.capture_step(); replacement.reset()
        return WarpRLEnv(replacement,publish_diagnostics=False,publish_step_metrics=False)
