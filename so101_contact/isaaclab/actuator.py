"""Explicit Isaac Lab actuator running :class:`~so101_contact.servo.STS3215Model` on PhysX.

Written against Isaac Lab ``bb0c8e1`` (IsaacLab-Arena ``aa36f19`` submodule):

- ``ActuatorCollection._build_groups`` constructs the actuator with ``cfg, joint_names, joint_ids, num_envs, device,
  stiffness, damping, actuator_velocity_limit, actuator_effort_limit``, then writes the cfg's ``armature``,
  ``friction`` (static friction *effort* [N.m] on Isaac Sim >= 5), ``dynamic_friction`` and ``viscous_friction`` to
  PhysX, and zero drive stiffness/damping for explicit groups.
- ``compute(control_action, joint_pos, joint_vel)`` runs every physics step; ``control_action.joint_positions`` is the
  Goal_Position. We return efforts only.

BAM's load-dependent friction needs the external torque on each joint. After :meth:`STS3215Actuator.bind` the
actuator estimates it from the articulation with :func:`~so101_contact.external_torque.estimate_external_torque`:
``(M qdd)_j + armature_j*qdd_j - tau_j(previous step)`` (gravity + contacts + Coriolis; ``joint_acc`` is
finite-differenced by Isaac Lab, ``mass_matrix`` from PhysX, which excludes the armature), minus the PhysX joint
friction applied during that step (``physx_joint`` mode; otherwise it is zero). Unbound, it uses zero.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions

from ..external_torque import estimate_external_torque
from ..params import STS3215_7V4, RandomizationRanges, STS3215Params, randomize, so101_follower, to_tensors
from ..servo import STS3215Model
from ..telemetry import TelemetryBus

SO101_SIM_TO_LEROBOT = {
    "Rotation": "shoulder_pan",
    "Pitch": "shoulder_lift",
    "Elbow": "elbow_flex",
    "Wrist_Pitch": "wrist_flex",
    "Wrist_Roll": "wrist_roll",
    "Jaw": "gripper",
}
"""isaaclab_so101 joint names -> LeRobot motor names (ENV_AUDIT section 5)."""


class STS3215Actuator(ActuatorBase):
    """Explicit STS3215 servo group: Goal_Position in, joint torque out, telemetry kept for observations."""

    cfg: STS3215ActuatorCfg

    def __init__(self, cfg: STS3215ActuatorCfg, *args, stiffness=None, damping=None, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._params = [_resolve_params(cfg.servo_params, name) for name in self.joint_names]
        self._generator = torch.Generator().manual_seed(cfg.seed)
        self._ranges = cfg.randomization if cfg.randomization is not None else None
        p = self._sample_params(self._num_envs) if cfg.randomize else to_tensors(self._params, self._num_envs)
        dt = cfg.physics_dt or _physics_dt()
        self.servo = STS3215Model(
            p, self._num_envs, dt, device=self._device, friction_mode=cfg.friction_mode,
            friction_smoothing_velocity=cfg.friction_smoothing_velocity, generator=self._generator,
            max_lag=max(int(x.max_delay) for x in self._params),
        )  # fmt: skip
        self.bus = TelemetryBus(
            self._num_envs, self.num_joints, dt, read_hz=cfg.read_hz, latency_steps=cfg.latency_steps,
            noise_counts=cfg.noise_counts, dropout_p=cfg.dropout_p, generator=self._generator, device=self._device,
        )  # fmt: skip
        shape = (self._num_envs, self.num_joints)
        self.duty = torch.zeros(shape, device=self._device)
        self.current = torch.zeros(shape, device=self._device)
        self.friction_budget = torch.zeros(shape, device=self._device)
        self.overload_active = torch.zeros(shape, dtype=torch.bool, device=self._device)
        self.external_torque = torch.zeros(shape, device=self._device)
        self.telemetry = self.bus.readings
        self._last_tau = torch.zeros(shape, device=self._device)
        self._articulation = None
        self._step = 0

    def bind(self, articulation) -> None:
        """Give the actuator its articulation, for the external-torque estimate and PhysX friction writes."""
        self._articulation = articulation

    @staticmethod
    def bind_all(articulation) -> list[STS3215Actuator]:
        """Bind every STS3215 group of ``articulation``; returns them."""
        groups = [articulation.actuators[n] for n in articulation.actuators]
        groups = [g for g in groups if isinstance(g, STS3215Actuator)]
        for g in groups:
            g.bind(articulation)
        return groups

    def reset(self, env_ids: Sequence[int] | None):
        ids = None if env_ids is None else torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        n = self._num_envs if ids is None else len(ids)
        if self.cfg.randomize and n > 0:
            self.servo.set_params(self._sample_params(n), ids)
        self.servo.reset(ids)
        self.bus.reset(ids)
        sel = slice(None) if ids is None else ids
        self._last_tau[sel] = 0.0
        self.external_torque[sel] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self.external_torque = self._estimate_external_torque(joint_vel)
        out = self.servo.step(control_action.joint_positions, joint_pos, joint_vel, self.external_torque)
        tau = out["tau"]
        self.duty, self.current = out["duty"], out["current"]
        self.friction_budget, self.overload_active = out["friction_budget"], out["overload_active"]
        self.telemetry = self.bus.step(self.duty, self.current, joint_pos, joint_vel)
        if self.cfg.friction_mode == "physx_joint" and self._articulation is not None:
            if self._step % self.cfg.friction_sync_decimation == 0:
                self._write_friction_budget()
        self._step += 1
        self.computed_effort = out["motor_tau"]
        self.applied_effort = tau
        self._last_tau = tau
        control_action.joint_efforts = tau
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    # --- internals ---

    def _sample_params(self, n: int) -> dict[str, torch.Tensor]:
        return randomize(self._params, n, self._generator, self._ranges or RandomizationRanges(), self._device)

    def _estimate_external_torque(self, joint_vel: torch.Tensor) -> torch.Tensor:
        art = self._articulation
        if art is None or self.cfg.external_torque == "none":
            return torch.zeros_like(self._last_tau)
        idx = self.joint_indices
        d = art.data
        gravity = -d.gravity_compensation_forces.torch[:, idx]
        if self.cfg.external_torque == "gravity":
            return gravity
        qdd = d.joint_acc.torch
        mqdd = torch.einsum("eij,ej->ei", d.mass_matrix.torch, qdd)[:, idx]
        # PhysX friction in effect during the last step = the coefficients last written (before this step's write)
        ext = estimate_external_torque(
            mqdd, qdd[:, idx], self._last_tau, d.joint_armature.torch[:, idx], qd=joint_vel, prior=gravity,
            static_friction=d.joint_friction_coeff.torch[:, idx],
            dynamic_friction=d.joint_dynamic_friction_coeff.torch[:, idx],
            viscous_friction=d.joint_viscous_friction_coeff.torch[:, idx],
        )  # fmt: skip
        a = self.cfg.external_torque_filter
        return ext if a >= 1.0 else a * ext + (1 - a) * self.external_torque

    def _write_friction_budget(self) -> None:
        idx = self.joint_indices
        ids = torch.arange(self._articulation.num_joints, device=self._device)[idx]
        b = self.friction_budget.float().contiguous()
        self._articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=b, joint_dynamic_friction_coeff=b, joint_ids=ids
        )


@configclass
class STS3215ActuatorCfg(ActuatorBaseCfg):
    """Configuration of :class:`STS3215Actuator`. Use :func:`sts3215_actuator_cfg` to fill the joint properties."""

    class_type: type = STS3215Actuator
    stiffness: dict[str, float] | float | None = 0.0
    """Unused: the firmware gain is ``servo_params.kp_fw``."""
    damping: dict[str, float] | float | None = 0.0
    """Unused: damping is back-EMF plus ``friction_viscous``."""
    servo_params: dict[str, STS3215Params] | STS3215Params = STS3215_7V4
    """One parameter set for all joints, or ``{joint-name regex: params}`` (first full match wins)."""
    friction_mode: str = "explicit"
    """``explicit``: all friction in the torque. ``physx_joint``: Coulomb budget written to PhysX joint friction."""
    friction_smoothing_velocity: float = 0.1
    """``w_s`` of the explicit tanh friction [rad/s]."""
    friction_sync_decimation: int = 1
    """``physx_joint``: write the budget to PhysX every N physics steps (each write goes through the CPU)."""
    external_torque: str = "dynamics"
    """``dynamics`` (M qdd + armature qdd - tau - PhysX joint friction, needs :meth:`STS3215Actuator.bind`),
    ``gravity`` or ``none``."""
    external_torque_filter: float = 1.0
    """EMA weight of the newest external-torque estimate (1 = no filtering)."""
    randomize: bool = False
    """Draw per env-joint parameters on construction and on every reset."""
    randomization: RandomizationRanges | None = None
    seed: int = 0
    physics_dt: float | None = None
    """Physics step [s]; None reads it from the running simulation."""
    read_hz: float = 30.0
    latency_steps: int = 1
    noise_counts: int = 1
    dropout_p: float = 0.0


def sts3215_actuator_cfg(
    joint_names_expr: list[str],
    params: dict[str, STS3215Params] | STS3215Params = STS3215_7V4,
    friction_mode: str = "explicit",
    joint_names: Sequence[str] | None = None,
    **kwargs,
) -> STS3215ActuatorCfg:
    """Cfg with armature, PhysX friction and solver limits filled from ``params``.

    ``joint_names`` (the concrete names) are needed when ``params`` is a dict, to write per-joint properties.
    """
    names = list(joint_names) if joint_names is not None else list(joint_names_expr)
    per = {re.escape(n): _resolve_params(params, n) for n in names}
    physx = friction_mode == "physx_joint"
    return STS3215ActuatorCfg(
        joint_names_expr=joint_names_expr,
        servo_params=params,
        friction_mode=friction_mode,
        armature={k: p.armature for k, p in per.items()},
        friction={k: (p.friction_base if physx else 0.0) for k, p in per.items()},
        dynamic_friction={k: (p.friction_base if physx else 0.0) for k, p in per.items()},
        viscous_friction={k: (p.friction_viscous if physx else 0.0) for k, p in per.items()},
        # the servo clamps its own torque at V*kt/R; keep the solver limit out of the way
        joint_effort_limit={k: 2.0 * p.effort_limit for k, p in per.items()},
        actuator_effort_limit={k: 2.0 * p.effort_limit for k, p in per.items()},
        **kwargs,
    )


def so101_sts3215_cfg(friction_mode: str = "explicit", **kwargs) -> dict[str, STS3215ActuatorCfg]:
    """Actuator cfgs for an isaaclab_so101 SO-101 follower: arm and gripper groups with LeRobot settings."""
    follower = so101_follower()
    sim_params = {sim: follower[lr] for sim, lr in SO101_SIM_TO_LEROBOT.items()}
    arm = [n for n in SO101_SIM_TO_LEROBOT if n != "Jaw"]
    return {
        "arm": sts3215_actuator_cfg(arm, {re.escape(n): sim_params[n] for n in arm}, friction_mode, arm, **kwargs),
        "gripper": sts3215_actuator_cfg(["Jaw"], sim_params["Jaw"], friction_mode, ["Jaw"], **kwargs),
    }


def _resolve_params(params, joint_name: str) -> STS3215Params:
    if isinstance(params, STS3215Params):
        return params
    for pattern, p in params.items():
        if re.fullmatch(pattern, joint_name):
            return p
    raise ValueError(f"no STS3215 parameters match joint {joint_name!r} (patterns: {list(params)})")


def _physics_dt() -> float:
    from isaaclab_physx.physics import PhysxManager

    return float(PhysxManager.get_physics_dt())
