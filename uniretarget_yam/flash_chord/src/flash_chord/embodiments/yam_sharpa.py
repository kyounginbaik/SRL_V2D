"""YAM arms + SharpaWave hands (two fixed bases, two 6-DOF arms + Sharpa fingers). Same layout rules as Vega, without its torso."""

from __future__ import annotations

from flash_chord.embodiments.base import register_embodiment
from flash_chord.embodiments.vega_sharpa import VegaSharpa, VegaSharpaConfig


class YamSharpaConfig(VegaSharpaConfig):
    """YAM + Sharpa controls and generated-URDF path; values come from configs/embodiment/yam_sharpa.yaml."""


@register_embodiment("yam_sharpa")
class YamSharpa(VegaSharpa):
    """No-gripper YAM with the hand on the wrist; the URDF is generated per episode because the bases follow its scene."""

    name = "yam_sharpa"
    arm_joint_count = 6
