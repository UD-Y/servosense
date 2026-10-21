"""Isaac Lab observation terms exposing STS3215 telemetry (LeRobot-decoded register ints as float tensors).

Values come from each :class:`~servosense.isaaclab.actuator.STS3215Actuator`'s :class:`TelemetryBus`, which
already applies the read rate, latency, quantization, noise and dropout. Joints not driven by an STS3215 group read 0.

    ObsTerm(func=servo_load, params={"asset_cfg": SceneEntityCfg("robot")})
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from .actuator import STS3215Actuator


def _gather(env, asset_cfg: SceneEntityCfg, key: str) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    out = torch.zeros(env.num_envs, asset.num_joints, device=env.device)
    for name in asset.actuators:
        group = asset.actuators[name]
        if isinstance(group, STS3215Actuator):
            idx = group.joint_indices
            out[:, idx] = group.telemetry[key].to(out.dtype)
    return out[:, asset_cfg.joint_ids]


def servo_load(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Present_Load per joint: signed, 0.1 % PWM duty per count (+-1000)."""
    return _gather(env, asset_cfg, "Present_Load")


def servo_current(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Present_Current per joint: unsigned, 6.5 mA per count."""
    return _gather(env, asset_cfg, "Present_Current")
