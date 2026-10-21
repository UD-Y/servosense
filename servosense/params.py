"""STS3215 servo parameters in BAM-compatible names, presets, datasheet derivation and randomization.

Field names follow Isaac Lab's BAM drive (``BamMotorCfg`` / ``BamActuatorCfg`` on Isaac Lab ``develop`` @ ``0ec7c29``,
``source/isaaclab/isaaclab/actuators/actuator_bam_cfg.py``) so that :func:`to_bam_kwargs` drops into BAM on Newton
unchanged. Feetech firmware behaviour that BAM does not model (rate limiter, dead zone, overload protection) has extra
fields, listed in :data:`FIRMWARE_ONLY_FIELDS`.

Sources, cited per value:
  [DS]    Feetech STS3215 7.4 V datasheet (CLAUDE.md section 4.2).
  [BAM]   Rhoban BAM, hardware identification of the STS3215 7.4 V, model m5:
          github.com/Rhoban/bam ``bam/params/feetech_sts3215_7_4V/m5.json`` @ ``31e5b97`` (2026-08-12).
  [BAMFW] Rhoban BAM ``bam/feetech/actuator.py`` @ ``e9a619d``: firmware constants measured with an oscilloscope.
  [LR]    LeRobot SO-101 follower configuration (P=16, I=0, D=32; gripper Max_Torque_Limit=500,
          Protection_Current=250, Overload_Torque=25).
  [FT]    Feetech STS memory table (register defaults).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

COUNTS_PER_REV = 4096  # [FT] Present_Position resolution
CURRENT_LSB_A = 0.0065  # [FT] Present_Current / Protection_Current unit: 6.5 mA per count

LEROBOT_MOTOR_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
"""Motor order of LeRobot's SO-101 follower bus (and of every per-joint tensor in this package)."""

BAM_MOTOR_FIELDS = (
    "model", "kt", "resistance", "error_gain", "max_pwm", "max_current", "friction_base", "friction_viscous",
    "friction_stribeck", "dtheta_stribeck", "alpha", "load_friction_motor", "load_friction_external",
    "load_friction_motor_stribeck", "load_friction_external_stribeck",
)  # fmt: skip
"""``BamMotorCfg`` fields we fill (m5; the m6 quadratic terms stay at their default 0)."""

BAM_ACTUATOR_FIELDS = ("kp_fw", "vin", "min_delay", "max_delay")
"""``BamActuatorCfg`` fields we fill. ``armature`` is an ``ActuatorBaseCfg`` field."""

FIRMWARE_ONLY_FIELDS = (
    "max_velocity", "dead_zone_counts", "min_startup_duty", "overload_torque_pct", "protection_time_s",
    "holding_pct", "counts_per_rev",
)  # fmt: skip
"""Feetech firmware behaviour with no BAM equivalent (dropped by :func:`to_bam_kwargs`)."""


@dataclass(frozen=True)
class Datasheet:
    """Feetech STS3215 datasheet values at the output shaft (gear 1/345). [DS]"""

    vin: float = 7.4  # V
    stall_torque: float = 1.912  # N.m (19.5 kg.cm)
    stall_current: float = 2.5  # A
    no_load_speed: float = 5.445  # rad/s (0.192 s / 60 deg = 52 rpm)
    no_load_current: float = 0.15  # A
    kt_effective: float = 0.785  # N.m/A (8 kg.cm/A, the datasheet "Kt": output torque per amp at stall)


DATASHEET_7V4 = Datasheet()


@dataclass
class STS3215Params:
    """One STS3215 servo, joint-side SI units. Defaults: [BAM] identified m5 fit of a real 7.4 V unit."""

    # --- BamMotorCfg ---
    model: str = "m5"
    kt: float = 1.2717402321162192  # N.m/A [BAM] (datasheet back-EMF derivation gives 1.24-1.28)
    resistance: float = 2.74580169718352  # Ohm [BAM]
    error_gain: float = 0.166 * 1.1558830785746244  # 1/rad per unit kp_fw: [BAMFW] 0.166 x [BAM] error_gain_ratio
    max_pwm: float = 0.97  # [BAMFW] max duty, oscilloscope, at Max_Torque_Limit=1000
    max_current: float = 0.0  # A, 0 = no firmware current window ([BAM] fit has none; arm keeps the register default)
    friction_base: float = 0.05286249893910219  # N.m Coulomb [BAM]
    friction_viscous: float = 0.02875750675061869  # N.m.s/rad [BAM] (on top of back-EMF damping kt^2/R)
    friction_stribeck: float = 7.453456393946817e-09  # N.m [BAM] (fit ~0)
    dtheta_stribeck: float = 0.3714206936630544  # rad/s [BAM]
    alpha: float = 9.999957365013934  # [BAM] Stribeck exponent (~10: a sharp near-rest band)
    load_friction_motor: float = 0.03890079636876726  # [BAM] gearbox friction per N.m of motor torque
    load_friction_external: float = 0.24078091046614414  # [BAM] gearbox friction per N.m of external torque
    load_friction_motor_stribeck: float = 1.103364634592961e-08  # [BAM] (fit ~0)
    load_friction_external_stribeck: float = 0.14920078266091807  # [BAM] extra near-rest external-load friction
    # --- BamActuatorCfg ---
    kp_fw: float = 16.0  # [LR] P_Coefficient written by LeRobot (Feetech default 32)
    vin: float = 7.4  # V [DS]
    min_delay: int = 0  # physics steps; [BAM] command_delay = 5.2 ms ~ 1.25 steps at 240 Hz -> randomize 0..2
    max_delay: int = 2
    # --- ActuatorBaseCfg ---
    armature: float = 0.02160563221423854  # kg.m^2 reflected rotor inertia [BAM]
    # --- Feetech firmware, not in BAM ---
    max_velocity: float = 5.1109355980623965  # rad/s rate limit on the firmware's internal target [BAM]/[BAMFW]
    dead_zone_counts: int = 1  # [FT] CW/CCW dead zone registers 26/27, default 1
    min_startup_duty: float = 0.0  # [FT] Minimum_Startup_Force (reg 24); unknown for LeRobot arms, [BAM] fit has none
    overload_torque_pct: float = 80.0  # [FT] Overload_Torque (reg 36) default
    protection_time_s: float = 2.0  # [FT] Protection_Time (reg 35) default 200 x 10 ms
    holding_pct: float = 20.0  # [FT] protective torque after overload (reg 34) default
    counts_per_rev: int = COUNTS_PER_REV

    @property
    def stall_torque_motor(self) -> float:
        """Motor torque at zero speed and full duty, before gearbox friction [N.m]."""
        return self.kt * self.max_pwm * self.vin / self.resistance

    @property
    def effort_limit(self) -> float:
        """BAM's effort clamp ``vin * kt / R`` [N.m] (``DriveBam`` default ``max_effort``)."""
        return self.vin * self.kt / self.resistance

    @property
    def backemf_damping(self) -> float:
        """Electrical damping ``kt^2 / R`` [N.m.s/rad]."""
        return self.kt**2 / self.resistance

    @property
    def duty_saturation_angle(self) -> float:
        """Position error at which the P loop reaches ``max_pwm`` [rad]."""
        return self.max_pwm / (self.kp_fw * self.error_gain)


def quasi_static_stall_torque(p: STS3215Params) -> float:
    """Largest opposing load the servo can still push at zero speed [N.m].

    With BAM's m5 friction budget at rest (Stribeck coefficient 1) and an external torque ``-T`` opposing a motor
    torque ``tau``: ``budget = base + stribeck + T*(mu_e + mu_es) + tau*(mu_m + mu_ms)``. Motion in the motor's
    direction needs ``tau - T > budget``, hence ``T = (tau*(1 - mu_m - mu_ms) - base - stribeck) / (1 + mu_e + mu_es)``.
    This is how a datasheet stall torque is measured (load raised until the shaft stops).
    """
    tau = p.stall_torque_motor
    if p.max_current > 0:
        tau = min(tau, p.kt * p.max_current)
    mu_m = p.load_friction_motor + p.load_friction_motor_stribeck
    mu_e = p.load_friction_external + p.load_friction_external_stribeck
    return (tau * (1 - mu_m) - p.friction_base - p.friction_stribeck) / (1 + mu_e)


def no_load_speed(p: STS3215Params) -> float:
    """Motor-limited free speed at full duty, ignoring the firmware rate limiter [rad/s].

    Balance ``kt*(pV - kt*w)/R * (1 - mu_m) = base + viscous*w`` (moving, so the Stribeck terms vanish for alpha~10).
    """
    v = p.max_pwm * p.vin
    a = p.kt * v / p.resistance * (1 - p.load_friction_motor)
    b = p.kt**2 / p.resistance * (1 - p.load_friction_motor) + p.friction_viscous
    return (a - p.friction_base) / b


def derive_from_datasheet(ds: Datasheet = DATASHEET_7V4, base: STS3215Params | None = None) -> STS3215Params:
    """Solve R, kt, friction_base and load_friction_motor from the four datasheet numbers.

    The datasheet gives no information on the external-load, Stribeck and viscous terms, so they (and the firmware
    constants) are taken from ``base`` ([BAM] fit by default). Equations (p = max_pwm):

    1. stall current   ``I_s = p*V/R``                              -> R
    2. no-load current ``I_nl = (p*V - kt*w_nl)/R``                 -> kt
    3. no-load balance ``kt*I_nl = base + mu_m*kt*I_nl + c*w_nl``    -> friction_base(mu_m)
    4. stall torque    :func:`quasi_static_stall_torque` = T_stall  -> mu_m (linear)

    Raises ``ValueError`` if the solution is unphysical (negative friction), i.e. the datasheet and the assumed
    external-load friction are inconsistent.
    """
    b = base or STS3215Params()
    v = b.max_pwm * ds.vin
    r = v / ds.stall_current
    kt = (v - ds.no_load_current * r) / ds.no_load_speed
    mu_e = b.load_friction_external + b.load_friction_external_stribeck
    rhs = (
        ds.stall_torque * (1 + mu_e)
        + b.friction_stribeck
        - b.friction_viscous * ds.no_load_speed
        + kt * ds.stall_current * b.load_friction_motor_stribeck
    )
    mu_m = 1 - rhs / (kt * (ds.stall_current - ds.no_load_current))
    f_base = kt * ds.no_load_current * (1 - mu_m) - b.friction_viscous * ds.no_load_speed
    if not (0 <= mu_m < 1) or f_base < 0:
        raise ValueError(f"datasheet inconsistent with assumed friction: mu_m={mu_m:.3f}, friction_base={f_base:.4f}")
    return dataclasses.replace(
        b, vin=ds.vin, resistance=r, kt=kt, friction_base=f_base, load_friction_motor=mu_m,
        # The datasheet rate is 52 rpm; the firmware limiter must not cap the derived speed.
        max_velocity=max(b.max_velocity, ds.no_load_speed),
    )  # fmt: skip


# --- Presets -------------------------------------------------------------------------------------------------------

STS3215_7V4 = STS3215Params()
"""Default: [BAM] hardware-identified m5 fit of a 7.4 V STS3215, LeRobot firmware gains."""

STS3215_7V4_DATASHEET = derive_from_datasheet(DATASHEET_7V4)
"""Datasheet-only derivation (cross-check and an alternative prior)."""


def _stall_matched_resistance(p: STS3215Params, stall_torque: float) -> float:
    mu_m = p.load_friction_motor + p.load_friction_motor_stribeck
    mu_e = p.load_friction_external + p.load_friction_external_stribeck
    tau = (stall_torque * (1 + mu_e) + p.friction_base + p.friction_stribeck) / (1 - mu_m)
    return p.kt * p.max_pwm * p.vin / tau


STS3215_12V = dataclasses.replace(STS3215_7V4, vin=12.0)
STS3215_12V = dataclasses.replace(STS3215_12V, resistance=_stall_matched_resistance(STS3215_12V, 2.942))
"""PROVISIONAL 12 V variant (C018/C047, 30 kg.cm = 2.942 N.m). Same kt and friction as the 7.4 V fit, R chosen to
match the 12 V datasheet stall torque. The 12 V unit has a different winding: identify it before trusting this."""


def so101_follower(base: STS3215Params = STS3215_7V4) -> dict[str, STS3215Params]:
    """Per-joint parameters of a LeRobot SO-101 follower, in :data:`LEROBOT_MOTOR_ORDER`. [LR]

    All six joints are STS3215 1/345. The gripper gets LeRobot's lower limits: Max_Torque_Limit=500 (half of
    ``max_pwm``), Protection_Current=250 (x 6.5 mA = 1.625 A current window), Overload_Torque=25 %.
    """
    gripper = dataclasses.replace(
        base, max_pwm=base.max_pwm * 500 / 1000, max_current=250 * CURRENT_LSB_A, overload_torque_pct=25.0
    )
    return {name: (gripper if name == "gripper" else dataclasses.replace(base)) for name in LEROBOT_MOTOR_ORDER}


SO101_FOLLOWER = so101_follower()


# --- Tensors and randomization -------------------------------------------------------------------------------------

TENSOR_FIELDS = tuple(f.name for f in dataclasses.fields(STS3215Params) if f.name != "model")


@dataclass
class RandomizationRanges:
    """Per env-joint multiplicative scale ranges (uniform; ``error_gain`` log-uniform). Delay: integer steps."""

    scales: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "kt": (0.9, 1.1),  # unit-to-unit motor spread
            "resistance": (0.9, 1.1),  # winding temperature (+4 %/10 K copper) and spread
            "error_gain": (0.5, 2.0),  # CLAUDE.md 4.4; P=16 has not been verified on hardware
            "friction_base": (0.5, 2.0),  # gearbox wear / lubrication
            "friction_viscous": (0.5, 1.5),
            "load_friction_motor": (0.5, 2.0),
            "load_friction_external": (0.5, 1.5),
            "load_friction_external_stribeck": (0.5, 1.5),
            "armature": (0.8, 1.2),
            "vin": (7.0 / 7.4, 7.6 / 7.4),  # supply 7.0-7.6 V
        }
    )
    log_uniform: tuple[str, ...] = ("error_gain",)
    delay: tuple[int, int] | None = None  # None: each joint's [min_delay, max_delay]


def to_tensors(
    params: STS3215Params | Sequence[STS3215Params] | Mapping[str, STS3215Params],
    num_envs: int,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Nominal parameters as ``(num_envs, num_joints)`` float tensors, one entry per numeric field."""
    plist = _as_list(params)
    out = {}
    for name in TENSOR_FIELDS:
        row = torch.tensor([float(getattr(p, name)) for p in plist], dtype=torch.float32, device=device)
        out[name] = row.unsqueeze(0).repeat(num_envs, 1)
    return out


def randomize(
    params: STS3215Params | Sequence[STS3215Params] | Mapping[str, STS3215Params],
    num_envs: int,
    generator: torch.Generator | None = None,
    ranges: RandomizationRanges | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Per env-joint randomized parameters, ``(num_envs, num_joints)`` tensors (independent draw per servo).

    ``min_delay`` / ``max_delay`` are both set to one sampled integer lag per env-joint.
    """
    ranges = ranges or RandomizationRanges()
    out = to_tensors(params, num_envs, "cpu")
    shape = out["kt"].shape
    for name, (lo, hi) in ranges.scales.items():
        u = torch.rand(shape, generator=generator)
        if name in ranges.log_uniform:
            s = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        else:
            s = lo + u * (hi - lo)
        out[name] = out[name] * s
    lo_d, hi_d = (out["min_delay"], out["max_delay"]) if ranges.delay is None else ranges.delay
    u = torch.rand(shape, generator=generator)
    lag = torch.floor(torch.as_tensor(lo_d) + u * (torch.as_tensor(hi_d) - torch.as_tensor(lo_d) + 1))
    lag = torch.minimum(lag, torch.as_tensor(hi_d, dtype=lag.dtype))
    out["min_delay"] = lag
    out["max_delay"] = lag.clone()
    return {k: v.to(device) for k, v in out.items()}


def to_bam_kwargs(p: STS3215Params) -> dict:
    """Keyword arguments for Isaac Lab ``BamActuatorCfg(motor=BamMotorCfg(**out["motor"]), **rest)``.

    Firmware-only behaviour (:data:`FIRMWARE_ONLY_FIELDS`) has no BAM field and is dropped: on Newton the target rate
    limiter, dead zone and overload protection are not simulated.
    """
    motor = {name: getattr(p, name) for name in BAM_MOTOR_FIELDS}
    out = {"motor": motor, **{name: getattr(p, name) for name in BAM_ACTUATOR_FIELDS}, "armature": p.armature}
    out["min_delay"], out["max_delay"] = int(p.min_delay), int(p.max_delay)
    return out


def _as_list(params) -> list[STS3215Params]:
    if isinstance(params, STS3215Params):
        return [params]
    if isinstance(params, Mapping):
        return list(params.values())
    return list(params)
