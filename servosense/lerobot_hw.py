"""Read the same telemetry from a real SO-101 through LeRobot, so sim and real observations match.

Mirrors LeRobot's ``MotorCurrentProcessorStep`` (``lerobot/rl/joint_observations_processor.py``) but writes the
registers to their own keys instead of extending ``observation.state``:

- ``observation.servo_load``: ``Present_Load``, signed. LeRobot's Feetech bus decodes the sign-magnitude word
  (bit 10) in ``sync_read``, exactly as :func:`servosense.registers.decode_sign_magnitude` does.
- ``observation.servo_current``: ``Present_Current``, unsigned 6.5 mA counts (LeRobot does not decode a sign).

Both are float32 tensors of shape (1, num_motors) in the bus's motor order. lerobot is imported only by this module,
and the module still imports without it (the step is then a plain class; :func:`read_servo_telemetry` works on any
object with a LeRobot-style ``sync_read`` and ``motors``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

OBS_SERVO_LOAD = "observation.servo_load"
OBS_SERVO_CURRENT = "observation.servo_current"
_REGISTERS = {OBS_SERVO_LOAD: "Present_Load", OBS_SERVO_CURRENT: "Present_Current"}

try:  # lerobot is optional
    from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
    from lerobot.processor import ObservationProcessorStep, ProcessorStepRegistry

    _register = ProcessorStepRegistry.register("servo_telemetry_processor")
except ImportError:  # pragma: no cover - exercised only without lerobot
    ObservationProcessorStep = object
    FeatureType = PipelineFeatureType = PolicyFeature = None

    def _register(cls):
        return cls


def read_servo_telemetry(bus: Any, motors: list[str] | None = None) -> dict[str, torch.Tensor]:
    """``sync_read`` Present_Load and Present_Current; returns ``{obs key: (1, num_motors) float32}``."""
    names = list(motors) if motors is not None else list(bus.motors)
    out = {}
    for key, register in _REGISTERS.items():
        values = bus.sync_read(register, names, normalize=False)
        out[key] = torch.tensor([values[n] for n in names], dtype=torch.float32).unsqueeze(0)
    return out


@_register
@dataclass
class ServoTelemetryProcessorStep(ObservationProcessorStep):
    """Adds ``observation.servo_load`` and ``observation.servo_current`` read from ``robot.bus``."""

    robot: Any = None

    def observation(self, observation: dict) -> dict:
        if self.robot is None:
            raise ValueError("Robot is not set")
        new_observation = dict(observation)
        new_observation.update(read_servo_telemetry(self.robot.bus))
        return new_observation

    def transform_features(self, features):
        if self.robot is None or PolicyFeature is None:
            return features
        n = len(self.robot.bus.motors)
        for key in _REGISTERS:
            features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(type=FeatureType.STATE, shape=(n,))
        return features
