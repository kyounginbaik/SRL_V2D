# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train the reference-tracking policy with native JAX FlashSAC."""

from __future__ import annotations

import os
import json
import time
from pathlib import Path

import hydra
import jax
import warp as wp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from flash_chord.configuration import instantiate_typed, resolve_path, resolved_dict
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint_metadata import (
    build_checkpoint_metadata,
    critic_schema,
    policy_schema,
    resolved_reference_config,
)
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.runner import WandbLogger, metric_definitions, run_training
from flash_chord.training.progress import ConsoleTrainingProgress

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["TZ"] = "PST8PDT,M3.2.0,M11.1.0"
time.tzset()


class LocalMetricLogger:
    """Preserve numerical-health evidence when remote logging is disabled."""
    def __init__(self, path):
        self.stream = Path(path).open("a")

    def log(self, metrics, step):
        self.stream.write(json.dumps({"step": step, **metrics}) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


@hydra.main(version_base="1.3", config_path="../src/flash_chord/configs", config_name="train_flash_sac")
def main(cfg: DictConfig) -> None:
    """Compose one FlashSAC experiment and run its compiled Warp/JAX training loop."""
    if cfg.get("adaptive_voc") is not None and cfg.adaptive_voc.get("mode") == "step_schedule":
        # The original contact-focused arm has no direct position tracking.
        # A separate reference-guided arm may use hand tracking so the lower
        # controller can actually respond to edits in its hand reference.
        guided_hands = bool(cfg.get("reference_guided_hands", False))
        for index, stage in enumerate(cfg.curriculum.stages):
            terms = ("object_keypoints",) if guided_hands else (
                "object_keypoints", "hand_keypoints", "hand_joint_pos")
            for term in terms:
                if float(stage.objective_weights[term]) != 0.0:
                    raise ValueError(
                        f"step_schedule curriculum stage {index} has nonzero {term} reward"
                    )
            if guided_hands and any(float(stage.objective_weights[term]) <= 0.0
                                    for term in ("hand_keypoints", "hand_joint_pos")):
                raise ValueError(f"reference-guided curriculum stage {index} lacks hand tracking")
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    training = instantiate_typed(cfg.training, TrainingConfig)
    output_dir = Path(HydraConfig.get().runtime.output_dir) if cfg.output_dir is None else resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with wp.ScopedDevice(training.device):
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=env_config.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=cfg.scene.world_count,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            object_scale_min=cfg.scene.object_scale_min,
            object_scale_max=cfg.scene.object_scale_max,
            object_scale_seed=cfg.scene.object_scale_seed,
            contact_friction=cfg.scene.get("contact_friction", 1.0),
            object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=bool(cfg.task.source_frame_playback),
            motion_start_frame=int(cfg.task.motion_start_frame),
            motion_end_frame=int(cfg.task.motion_end_frame),
            robot_reference_path=cfg.task.get("robot_reference_path"),
            robot_reference_sha256=cfg.task.get("robot_reference_sha256"),
        )
        parquet = setup.parquet_path
        reference = setup.reference
        scene = setup.scene
        env = RLEnv(scene, reference, config=env_config)
        env.reset()
        env.capture_step()
        env.reset()
        jax_env = WarpRLEnv(
            env,
            publish_diagnostics=False,
            publish_step_metrics=False,
        )
        environment_config = resolved_dict(cfg)
        environment_config["task"]["parquet"] = str(parquet)
        environment_config["reference"] = resolved_reference_config(reference, str(parquet))
        checkpoint_metadata = build_checkpoint_metadata(
            environment_config,
            policy_schema(env.action, env.observation_strategy),
            critic_schema(env.action, env.observation_strategy, jax_env.critic_context_names),
        )
        logger = LocalMetricLogger(output_dir / "training_metrics.jsonl") if cfg.logging.mode == "disabled" else None
        if cfg.logging.mode != "disabled":
            logger = WandbLogger(
                training,
                metric_definitions=metric_definitions(jax_env),
                environment_config=environment_config,
                upload_checkpoints=cfg.logging.upload_checkpoints,
                upload_training_state=cfg.logging.upload_training_state,
                mode=cfg.logging.mode,
                dir=str(output_dir),
            )
        reference_update = None
        if cfg.get("bilevel") is not None:
            from flash_chord.retargeting.manipulation_bilevel import ManipulationBilevel
            update_class = ManipulationBilevel
            if cfg.bilevel.get("engine") == "ttsa_flashsac":
                from flash_chord.retargeting.ttsa_editor import TTSAReferenceEditor
                update_class = TTSAReferenceEditor
            if cfg.bilevel.get("engine") == "response_bilevel":
                from flash_chord.retargeting.response_editor import ResponseBilevelEditor
                update_class = ResponseBilevelEditor
            elif cfg.bilevel.get("engine") == "adaptive_window_flashsac":
                from flash_chord.retargeting.adaptive_editor import AdaptiveWindowEditor
                update_class = AdaptiveWindowEditor
            elif cfg.bilevel.get("engine") == "local_flashsac":
                from flash_chord.retargeting.local_editor import LocalReferenceEditor
                update_class = LocalReferenceEditor
            elif cfg.bilevel.get("engine") == "parallel_flashsac":
                from flash_chord.retargeting.parallel_editor import ParallelReferenceEditor
                update_class = ParallelReferenceEditor
            reference_update = update_class(
                options=resolved_dict(cfg.bilevel), setup=setup, embodiment=embodiment,
                collision=collision, training=training, environment_config=environment_config,
                checkpoint_metadata=checkpoint_metadata, output_dir=output_dir,
            )
        voc_controller=None
        if cfg.get("adaptive_voc") is not None:
            if reference_update is None:raise ValueError("Adaptive VOC requires the rollout evaluator")
            from flash_chord.lifecycle.performance_voc import PerformanceVOCController, StepVOCController
            voc_options=resolved_dict(cfg.adaptive_voc)
            voc_mode=voc_options.get("mode", "performance_gate")
            if voc_mode == "step_schedule":
                voc_controller=StepVOCController(reference_update,voc_options,output_dir)
            elif voc_mode == "performance_gate":
                voc_controller=PerformanceVOCController(reference_update,voc_options,output_dir)
            else:
                raise ValueError(f"Unknown adaptive_voc.mode: {voc_mode}")
        learner = run_training(
            jax_env,
            training,
            logger=logger,
            checkpoint_dir=output_dir,
            log_interval=cfg.logging.log_interval,
            voc_scale_override=cfg.voc_scale_override,
            checkpoint_metadata=checkpoint_metadata,
            progress=ConsoleTrainingProgress(training.total_environment_steps, output_dir),
            reference_update=reference_update,
            voc_controller=voc_controller,
        )
    print(f"completed {int(jax.device_get(learner.environment_steps))} environment steps; checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
