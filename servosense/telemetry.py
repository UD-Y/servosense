"""Feetech bus model: what LeRobot's ``sync_read`` returns, sampled at the control rate (CLAUDE.md section 4.6).

Each control tick the bus samples the servo registers, quantizes them to register ints, adds +-``noise_counts``
uniform integer noise, and delivers them ``latency_steps`` control ticks later. A dropped read (probability
``dropout_p`` per env and tick) repeats that env's last delivered values. Values are LeRobot-decoded ints:
signed ``Present_Load`` (0.1 % duty), unsigned ``Present_Current`` (6.5 mA), signed ``Present_Speed`` (counts/s),
``Present_Position`` (counts, joint zero at :data:`~servosense.registers.POSITION_CENTER`).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch

from . import registers

TELEMETRY_KEYS = ("Present_Position", "Present_Speed", "Present_Load", "Present_Current")
_NOISY = ("Present_Load", "Present_Current")


class TelemetryBus:
    """Batched register reads for ``(num_envs, num_joints)`` servos, in the joint order of the inputs.

    Call :meth:`step` once per physics step; it returns the readings currently held by the host (updated on control
    ticks only). ``read_hz`` must divide the physics rate.
    """

    def __init__(
        self,
        num_envs: int,
        num_joints: int,
        physics_dt: float,
        read_hz: float = 30.0,
        latency_steps: int = 1,
        noise_counts: int = 1,
        dropout_p: float = 0.0,
        position_noise: bool = False,
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ):
        self.decimation = max(1, round(1.0 / (read_hz * physics_dt)))
        if abs(self.decimation * read_hz * physics_dt - 1.0) > 1e-6:
            raise ValueError(f"read_hz={read_hz} does not divide the physics rate {1 / physics_dt:.3f} Hz")
        self.num_envs, self.num_joints = num_envs, num_joints
        self.latency_steps = latency_steps
        self.noise_counts = noise_counts
        self.dropout_p = dropout_p
        self.noisy = _NOISY + (("Present_Position",) if position_noise else ())
        self.generator = generator
        self.device = torch.device(device)
        self._step_count = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._queue: deque[dict[str, torch.Tensor]] = deque(maxlen=latency_steps + 1)
        self._held = self._zeros()

    def _zeros(self) -> dict[str, torch.Tensor]:
        return {k: torch.zeros(self.num_envs, self.num_joints, dtype=torch.int64, device=self.device)
                for k in TELEMETRY_KEYS}  # fmt: skip

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._step_count[ids] = 0
        for sample in self._queue:
            for v in sample.values():
                v[ids] = 0
        for v in self._held.values():
            v[ids] = 0

    def sample(self, duty: torch.Tensor, current: torch.Tensor, q: torch.Tensor, qd: torch.Tensor):
        """Quantize the servo state to decoded register ints (no noise, no latency)."""
        return {
            "Present_Position": registers.present_position(q),
            "Present_Speed": registers.present_speed(qd),
            "Present_Load": registers.present_load(duty),
            "Present_Current": registers.present_current(current),
        }

    def step(self, duty: torch.Tensor, current: torch.Tensor, q: torch.Tensor, qd: torch.Tensor):
        """Advance one physics step; returns the held readings (a dict of int64 ``(num_envs, num_joints)``)."""
        tick = (self._step_count % self.decimation) == 0
        self._step_count += 1
        if not bool(tick.any()):
            return self.readings
        sample = self.sample(duty, current, q, qd)
        if self.noise_counts > 0:
            for k in self.noisy:
                n = torch.randint(-self.noise_counts, self.noise_counts + 1, sample[k].shape, generator=self.generator)
                sample[k] = sample[k] + n.to(self.device)
            sample["Present_Load"] = sample["Present_Load"].clamp(-registers.LOAD_FULL_SCALE, registers.LOAD_FULL_SCALE)
            sample["Present_Current"] = sample["Present_Current"].clamp(min=0)
        if len(self._queue) == 0:  # prime the pipe so the first deliveries are the first sample
            for _ in range(self.latency_steps):
                self._queue.append({k: v.clone() for k, v in sample.items()})
        self._queue.append(sample)
        delivered = self._queue[0]
        update = tick
        if self.dropout_p > 0:
            update = update & (torch.rand(self.num_envs, generator=self.generator).to(self.device) >= self.dropout_p)
        for k in TELEMETRY_KEYS:
            self._held[k] = torch.where(update.unsqueeze(-1), delivered[k], self._held[k])
        return self.readings

    @property
    def readings(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._held.items()}
