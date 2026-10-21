"""Batched STS3215 servo model: firmware position loop -> PWM duty -> DC motor -> gearbox friction -> joint torque.

The motor and friction equations are BAM's (Isaac Lab ``develop`` @ ``0ec7c29``,
``isaaclab_newton/actuators/bam_kernels.py``: ``bam_delay_kernel``, ``bam_motor_kernel``, ``bam_friction_kernel``).
Added on top, from the Feetech firmware: Goal_Position quantization, the internal target rate limiter (Rhoban BAM
``STS3215Actuator``), the error in encoder counts with a dead zone, a minimum start-up duty and overload protection.

Per physics step, all tensors ``(num_envs, num_joints)``:

    delayed   = target from ``lag`` steps ago                                   (lag fixed per episode)
    q_int     = clamp(round_counts(delayed), q_int -+ max_velocity*dt)          (firmware rate limiter)
    err_c     = counts(q_int) - counts(q);  0 if |err_c| <= dead zone
    duty      = err_c*(2pi/4096)*kp_fw*error_gain + sign(err_c)*min_startup_duty
    duty      = clamp(duty, kt*w/V -+ R*Imax/V)  if max_current > 0                (firmware current window)
    duty      = clamp(duty, -max_pwm, max_pwm);  clamp(+-holding) while overloaded
    I         = (V*duty - kt*w)/R;   motor_tau = kt*I;   applied = clamp(motor_tau, +-V*kt/R)
    budget    = base + s*stribeck + |ext*mu_e - prev_applied*mu_m| + s*|ext*mu_es - prev_applied*mu_ms|,
                s = exp(-(|w|/dtheta)^alpha)

``friction_mode="physx_joint"``: ``tau = applied``; the engine applies ``budget`` as joint Coulomb friction
(a constraint, like BAM's MJWarp ``frictionloss``) and ``friction_viscous`` as joint viscous friction.
``friction_mode="explicit"``: ``tau = applied - budget*tanh(w/w_s) - friction_viscous*w`` (smooth, no stiction).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from .params import STS3215_7V4, STS3215Params, _as_list, to_tensors

# BAM_MODEL_FLAGS (actuator_bam_cfg.py): which friction terms each model enables.
MODEL_FLAGS = {"m1": (False, False), "m2": (True, False), "m5": (True, True)}  # (stribeck, load_dependent)


class STS3215Model:
    """Batched STS3215 servos.

    Args:
        params: one :class:`STS3215Params` (all joints), a sequence / mapping per joint, or a dict of
            ``(num_envs, num_joints)`` tensors from :func:`~servosense.params.randomize`.
        num_envs: number of environments.
        dt: physics step [s]; :meth:`step` must be called once per physics step.
        num_joints: needed only when ``params`` is a single :class:`STS3215Params`.
        friction_mode: ``"explicit"`` or ``"physx_joint"`` (see module docstring).
        friction_smoothing_velocity: ``w_s`` of the explicit tanh friction [rad/s].
        model: BAM friction model, ``"m1"``, ``"m2"`` or ``"m5"``.
        generator: CPU generator for the per-episode command lag.
        max_lag: longest command lag the history must hold [physics steps]; default: the largest ``max_delay``.
    """

    def __init__(
        self,
        params: STS3215Params | Sequence[STS3215Params] | Mapping = STS3215_7V4,
        num_envs: int = 1,
        dt: float = 1 / 240,
        num_joints: int | None = None,
        device: str | torch.device = "cpu",
        friction_mode: str = "explicit",
        friction_smoothing_velocity: float = 0.1,
        model: str = "m5",
        generator: torch.Generator | None = None,
        max_lag: int | None = None,
    ):
        if friction_mode not in ("explicit", "physx_joint"):
            raise ValueError(f"friction_mode must be 'explicit' or 'physx_joint', got {friction_mode!r}")
        if model not in MODEL_FLAGS:
            raise ValueError(f"unsupported friction model {model!r}; expected one of {list(MODEL_FLAGS)}")
        self.device = torch.device(device)
        if isinstance(params, Mapping) and params and isinstance(next(iter(params.values())), torch.Tensor):
            self.p = {k: v.to(self.device, torch.float32) for k, v in params.items()}
        else:
            plist = _as_list(params)
            if len(plist) == 1 and num_joints:
                plist = plist * num_joints
            self.p = to_tensors(plist, num_envs, self.device)
        self.num_envs, self.num_joints = self.p["kt"].shape
        self.dt = dt
        self.friction_mode = friction_mode
        self.w_s = friction_smoothing_velocity
        self.stribeck, self.load_dependent = MODEL_FLAGS[model]
        self.generator = generator
        self.effort_limit = self.p["vin"] * self.p["kt"] / self.p["resistance"]

        shape = (self.num_envs, self.num_joints)
        self.max_lag = max(int(self.p["max_delay"].max().item()), max_lag or 0)
        self._hist = torch.zeros(*shape, self.max_lag + 1, device=self.device)
        self._lag = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self._q_int = torch.zeros(shape, device=self.device)
        self._prev_applied = torch.zeros(shape, device=self.device)
        self._overload_timer = torch.zeros(shape, device=self.device)
        self._overload = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._needs_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear the servo state; the next :meth:`step` re-seeds the firmware target from the joint position."""
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._prev_applied[ids] = 0.0
        self._overload_timer[ids] = 0.0
        self._overload[ids] = False
        self._needs_init[ids] = True
        lo, hi = self.p["min_delay"][ids], self.p["max_delay"][ids]
        u = torch.rand(lo.shape, generator=self.generator).to(self.device)
        self._lag[ids] = torch.minimum(torch.floor(lo + u * (hi - lo + 1)), hi).to(torch.int64)

    def set_params(self, params: Mapping[str, torch.Tensor], env_ids: Sequence[int] | torch.Tensor | None = None):
        """Overwrite per-env parameters (e.g. a fresh :func:`randomize` draw on reset)."""
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for k, v in params.items():
            self.p[k][ids] = v.to(self.device, torch.float32)
        if int(self.p["max_delay"].max().item()) > self.max_lag:
            raise ValueError("set_params cannot raise max_delay above the value the model was built with")
        self.effort_limit = self.p["vin"] * self.p["kt"] / self.p["resistance"]

    def step(
        self,
        target: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        tau_ext: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Advance one physics step.

        Args:
            target: Goal_Position [rad], ``(num_envs, num_joints)``.
            q, qd: joint position [rad] and velocity [rad/s].
            tau_ext: external torque on the joint from the previous solve (gravity + contacts, excluding this
                actuator's own friction; BAM's ``-qfrc_bias + qfrc_constraint - own_friction``). None = 0.

        Returns:
            dict of ``(num_envs, num_joints)`` tensors: ``tau`` (to apply), ``motor_tau``, ``duty``, ``current``,
            ``friction_budget``, ``overload_active``, ``internal_target``, ``delayed_target``.
        """
        p, dt = self.p, self.dt
        target, q, qd = (torch.as_tensor(x, device=self.device, dtype=torch.float32) for x in (target, q, qd))
        ext = torch.zeros_like(q) if tau_ext is None else torch.as_tensor(tau_ext, device=self.device).float()

        init = self._needs_init
        if init.any():
            self._hist[init] = target[init].unsqueeze(-1)
            self._q_int[init] = q[init]
            self._needs_init[:] = False

        # Command delay (bam_delay_kernel): column k holds the command from k steps ago.
        self._hist = torch.roll(self._hist, 1, dims=-1)
        self._hist[..., 0] = target
        delayed = self._hist.gather(-1, self._lag.unsqueeze(-1)).squeeze(-1)

        # Goal_Position is an integer count; the firmware moves its internal target at most max_velocity.
        cpr = p["counts_per_rev"]
        rad_per_count = 2 * math.pi / cpr
        goal = torch.round(delayed / rad_per_count) * rad_per_count
        step_max = p["max_velocity"] * dt
        self._q_int = torch.minimum(torch.maximum(goal, self._q_int - step_max), self._q_int + step_max)

        # Firmware P loop on integer counts, with dead zone and minimum start-up duty.
        err_c = torch.round(self._q_int / rad_per_count) - torch.round(q / rad_per_count)
        err_c = torch.where(err_c.abs() <= p["dead_zone_counts"], torch.zeros_like(err_c), err_c)
        duty = err_c * rad_per_count * p["kp_fw"] * p["error_gain"] + torch.sign(err_c) * p["min_startup_duty"]

        # Firmware current window (bam_motor_kernel), then the PWM limit.
        vin = p["vin"]
        center = p["kt"] * qd / vin
        span = p["resistance"] * p["max_current"] / vin
        duty = torch.where(p["max_current"] > 0, torch.clamp(duty, center - span, center + span), duty)
        duty = torch.clamp(duty, -p["max_pwm"], p["max_pwm"])

        # Overload protection: |duty| above Overload_Torque for Protection_Time -> limit to the holding torque
        # until the requested duty drops back under the threshold.
        over = duty.abs() > p["overload_torque_pct"] / 100
        self._overload_timer = torch.where(over, self._overload_timer + dt, torch.zeros_like(duty))
        self._overload = torch.where(self._overload, over, self._overload_timer >= p["protection_time_s"])
        hold = torch.minimum(p["holding_pct"] / 100, p["max_pwm"])
        duty = torch.where(self._overload, torch.clamp(duty, -hold, hold), duty)

        # DC motor, quasi-static (L/R << dt), joint side.
        current = (vin * duty - p["kt"] * qd) / p["resistance"]
        motor_tau = p["kt"] * current

        # Gearbox friction budget (bam_friction_kernel), using the previous applied torque.
        prev = self._prev_applied
        budget = p["friction_base"].clone()
        s = torch.zeros_like(qd)
        if self.stribeck:
            s = torch.exp(-torch.pow(qd.abs() / p["dtheta_stribeck"], p["alpha"]))
            budget = budget + s * p["friction_stribeck"]
        if self.load_dependent:
            budget = budget + (ext * p["load_friction_external"] - prev * p["load_friction_motor"]).abs()
            if self.stribeck:
                budget = budget + s * (
                    ext * p["load_friction_external_stribeck"] - prev * p["load_friction_motor_stribeck"]
                ).abs()

        applied = torch.clamp(motor_tau, -self.effort_limit, self.effort_limit)
        self._prev_applied = applied
        if self.friction_mode == "explicit":
            tau = applied - budget * torch.tanh(qd / self.w_s) - p["friction_viscous"] * qd
        else:
            tau = applied
        return {
            "tau": tau,
            "motor_tau": applied,
            "duty": duty,
            "current": current,
            "friction_budget": budget,
            "overload_active": self._overload.clone(),
            "internal_target": self._q_int.clone(),
            "delayed_target": delayed,
        }
