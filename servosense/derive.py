"""Derivation table: ``python -m servosense.derive [--vin 7.4] [--markdown docs/DERIVATION.md]``.

Prints the datasheet-only derivation next to the hardware-identified BAM fit, and asserts that the datasheet preset
reproduces the datasheet (stall torque, stall current, no-load speed) through the closed-form m5 balance.
"""

from __future__ import annotations

import argparse
import dataclasses
import math

from .params import (
    BAM_ACTUATOR_FIELDS,
    BAM_MOTOR_FIELDS,
    COUNTS_PER_REV,
    FIRMWARE_ONLY_FIELDS,
    STS3215_7V4,
    Datasheet,
    STS3215Params,
    derive_from_datasheet,
    no_load_speed,
    quasi_static_stall_torque,
)


def rows(ds: Datasheet, ident: STS3215Params, derived: STS3215Params) -> list[tuple[str, str, str, str, str]]:
    """(quantity, unit, datasheet-derived, BAM-identified, how)"""
    kp4 = (COUNTS_PER_REV / (2 * math.pi)) / 1000 / 4

    def f(x, n=4):
        return f"{x:.{n}f}"

    return [
        ("supply voltage V", "V", f(derived.vin, 2), f(ident.vin, 2), "datasheet"),
        ("max duty p", "-", f(derived.max_pwm, 2), f(ident.max_pwm, 2), "oscilloscope (Rhoban BAM)"),
        ("winding resistance R", "Ohm", f(derived.resistance), f(ident.resistance), "R = pV / I_stall"),
        ("torque constant kt (joint side)", "N.m/A", f(derived.kt), f(ident.kt), "kt = (pV - I_nl R) / w_nl"),
        ("ideal stall torque kt pV/R", "N.m", f(derived.stall_torque_motor, 3), f(ident.stall_torque_motor, 3), ""),
        ("Coulomb friction_base", "N.m", f(derived.friction_base), f(ident.friction_base), "no-load balance"),
        ("load_friction_motor mu_m", "-", f(derived.load_friction_motor), f(ident.load_friction_motor),
         "stall balance (m5)"),
        ("load_friction_external mu_e (+ Stribeck)", "-",
         f(derived.load_friction_external + derived.load_friction_external_stribeck),
         f(ident.load_friction_external + ident.load_friction_external_stribeck), "BAM fit (no datasheet info)"),
        ("friction_viscous", "N.m.s/rad", f(derived.friction_viscous), f(ident.friction_viscous), "BAM fit"),
        ("back-EMF damping kt^2/R", "N.m.s/rad", f(derived.backemf_damping, 3), f(ident.backemf_damping, 3), ""),
        ("armature", "kg.m^2", f(derived.armature), f(ident.armature), "BAM fit"),
        ("quasi-static stall torque", "N.m", f(quasi_static_stall_torque(derived), 3),
         f(quasi_static_stall_torque(ident), 3), f"datasheet {ds.stall_torque}"),
        ("stall current pV/R", "A", f(derived.max_pwm * derived.vin / derived.resistance, 3),
         f(ident.max_pwm * ident.vin / ident.resistance, 3), f"datasheet {ds.stall_current}"),
        ("no-load speed (motor)", "rad/s", f(no_load_speed(derived), 3), f(no_load_speed(ident), 3),
         f"datasheet {ds.no_load_speed}"),
        ("firmware rate limit max_velocity", "rad/s", f(derived.max_velocity, 3), f(ident.max_velocity, 3), "BAM fit"),
        ("error_gain (per unit P, per rad)", "1/rad", f(derived.error_gain), f(ident.error_gain),
         f"0.166 (scope) x ratio; Kp/4 hypothesis {kp4:.4f}"),
        ("kp_fw (P_Coefficient)", "-", f(derived.kp_fw, 0), f(ident.kp_fw, 0), "LeRobot writes 16"),
        ("duty saturation angle at P", "deg", f(math.degrees(derived.duty_saturation_angle), 1),
         f(math.degrees(ident.duty_saturation_angle), 1), "p / (kp_fw * error_gain)"),
        ("effective Kt at stall", "N.m/A", f(quasi_static_stall_torque(derived) / ds.stall_current, 3),
         f(quasi_static_stall_torque(ident) / (ident.max_pwm * ident.vin / ident.resistance), 3),
         f"datasheet {ds.kt_effective}"),
    ]  # fmt: skip


def check(ds: Datasheet, derived: STS3215Params) -> list[str]:
    """Datasheet-consistency asserts; returns human-readable lines."""
    stall = quasi_static_stall_torque(derived)
    i_stall = derived.max_pwm * derived.vin / derived.resistance
    w_nl = no_load_speed(derived)
    i_nl = (derived.max_pwm * derived.vin - derived.kt * w_nl) / derived.resistance
    checks = [
        ("stall torque", stall, ds.stall_torque),
        ("stall current", i_stall, ds.stall_current),
        ("no-load speed", w_nl, ds.no_load_speed),
        ("no-load current", i_nl, ds.no_load_current),
    ]
    out = []
    for name, got, want in checks:
        assert abs(got - want) <= 1e-6 * max(1.0, abs(want)), f"{name}: derived {got} != datasheet {want}"
        out.append(f"OK  {name}: {got:.4f} (datasheet {want})")
    return out


def table_text(r):
    w = [max(len(x[i]) for x in r) for i in range(5)]
    head = ("quantity", "unit", "datasheet", "BAM-identified", "how")
    w = [max(w[i], len(head[i])) for i in range(5)]
    lines = ["  ".join(h.ljust(w[i]) for i, h in enumerate(head)), "  ".join("-" * x for x in w)]
    lines += ["  ".join(c.ljust(w[i]) for i, c in enumerate(x)) for x in r]
    return "\n".join(lines)


def table_md(r):
    lines = ["| quantity | unit | datasheet-derived | BAM-identified (default) | how |", "|---|---|---|---|---|"]
    lines += ["| " + " | ".join(x) + " |" for x in r]
    return "\n".join(lines)


def markdown(ds, ident, derived, r, checks) -> str:
    return f"""# STS3215 derivation

Generated by `python -m servosense.derive --markdown docs/DERIVATION.md`; every number below comes from
`servosense/params.py`.

## Sources

- **Datasheet**: Feetech STS3215, {ds.vin} V, gear 1/345. Stall {ds.stall_torque} N.m at {ds.stall_current} A,
  no-load {ds.no_load_speed} rad/s at {ds.no_load_current} A, "Kt" {ds.kt_effective} N.m/A.
- **BAM fit**: Rhoban BAM hardware identification of a real STS3215 7.4 V on a pendulum bench, model m5
  ([`bam/params/feetech_sts3215_7_4V/m5.json`](https://github.com/Rhoban/bam/blob/31e5b97b3cabfe2b909850d483c4e84afa9781c4/bam/params/feetech_sts3215_7_4V/m5.json),
  commit `31e5b97`, 2026-08-12). Firmware constants (`error_gain = 0.166`, `max_pwm = 0.97`, target rate limiter)
  were measured with an oscilloscope (`bam/feetech/actuator.py` @ `e9a619d`).
- **Equations and names**: Isaac Lab BAM drive, branch `develop` @ `0ec7c29` (2026-10-19):
  `actuators/actuator_bam_cfg.py`, `isaaclab_newton/actuators/bam_kernels.py`.

## Model (joint side, per physics step)

```
I         = (V*duty - kt*w) / R             motor_tau = kt*I        (inductance neglected, L/R << dt)
budget    = friction_base + s*friction_stribeck
          + |tau_ext*mu_e - tau_prev*mu_m| + s*|tau_ext*mu_es - tau_prev*mu_ms|,   s = exp(-(|w|/dtheta)^alpha)
```

`budget` is a Coulomb friction limit (a constraint in the physics engine), `tau_ext` is the torque the environment
applies to the joint (gravity, contacts). At rest and against an opposing load `T` the shaft only moves forward if
`tau - T > budget`, so the quasi-static stall torque is

```
T_stall = (tau*(1 - mu_m - mu_ms) - friction_base - friction_stribeck) / (1 + mu_e + mu_es)
```

## Datasheet-only derivation

The datasheet has four numbers; they fix four unknowns. The external-load, Stribeck and viscous terms and the
firmware constants are not in the datasheet and are taken from the BAM fit.

1. stall current `I_s = pV/R` gives R
2. no-load current `I_nl = (pV - kt*w_nl)/R` gives kt
3. no-load balance `kt*I_nl = friction_base + mu_m*kt*I_nl + c*w_nl` gives friction_base(mu_m)
4. stall balance above with `T_stall` = datasheet gives mu_m

## Results

{table_md(r)}

Consistency (datasheet preset, closed form; the pytest suite repeats these through the time-stepped model):

```
{chr(10).join(checks)}
```

## Reading the numbers

- **Motor constants agree.** kt and R from the datasheet are within 3 % and 5 % of the hardware fit. The
  datasheet-derived kt (back-EMF) and the oscilloscope-measured error gain (0.166, vs. 0.163 from the
  "Kp/4 in 0.1 %-duty per count" hypothesis) independently confirm the electrical model.
- **Friction does not.** The datasheet stall torque ({ds.stall_torque} N.m) implies more gearbox loss than the
  measured unit: the BAM fit gives a quasi-static stall torque of {quasi_static_stall_torque(ident):.2f} N.m
  ({100 * (quasi_static_stall_torque(ident) / ds.stall_torque - 1):+.0f} %). Datasheet stall figures are usually
  conservative; the fit is one servo on one bench. The default preset is the measured fit, and the randomization
  ranges (`params.RandomizationRanges`) span both.
- **Earlier prior corrected.** CLAUDE.md section 4.4 put all the loss on the motor side (`mu_m = 0.40`,
  `friction_base = 0.19`). In BAM's m5 form the external-side term dominates: `mu_e + mu_es =
  {ident.load_friction_external + ident.load_friction_external_stribeck:.2f}` at rest vs `mu_m =
  {ident.load_friction_motor:.3f}`. The datasheet derivation with the BAM external terms gives `mu_m =
  {derived.load_friction_motor:.3f}`.
- **Asymmetry (the contact claim).** With a load `T` opposing the motor and the shaft moving slowly, the motor must
  supply `T + budget` when it drives the load and only `T - budget` when the load back-drives it. Present_Load
  (duty) therefore rises faster with force when the arm pushes into something than when something pushes the arm,
  and at rest it lies anywhere in the band between the two (gearbox stiction hysteresis).
- **LeRobot gains.** LeRobot writes P = 16 (Feetech default 32, used in the BAM fit). With the measured error gain
  the duty saturates at {math.degrees(ident.duty_saturation_angle):.1f} deg of error. `error_gain` is randomized
  x[0.5, 2] until the calibration protocol confirms it at P = 16.

## Not in the datasheet or the fit

Minimum start-up duty (register 24) and the overload protection thresholds use register defaults; the Present_Load
direction bit and Present_Current's sign must be checked on hardware.

## Field mapping to Isaac Lab BAM

- `BamMotorCfg`: {", ".join(f"`{x}`" for x in BAM_MOTOR_FIELDS)}
- `BamActuatorCfg`: {", ".join(f"`{x}`" for x in BAM_ACTUATOR_FIELDS)}, plus `armature` (`ActuatorBaseCfg`)
- Feetech firmware only (dropped by `to_bam_kwargs`): {", ".join(f"`{x}`" for x in FIRMWARE_ONLY_FIELDS)}

BAM's `error_gain` is the product of Rhoban's `error_gain` (0.166) and the fitted `error_gain_ratio`.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vin", type=float, default=7.4, help="supply voltage (only 7.4 V has datasheet values)")
    ap.add_argument("--markdown", metavar="PATH", help="also write the derivation as Markdown to PATH")
    args = ap.parse_args(argv)
    if abs(args.vin - 7.4) > 1e-9:
        raise SystemExit("only the 7.4 V datasheet is encoded; the 12 V preset (params.STS3215_12V) is provisional")
    ds = Datasheet(vin=args.vin)
    ident = dataclasses.replace(STS3215_7V4, vin=args.vin)
    derived = derive_from_datasheet(ds, ident)
    r = rows(ds, ident, derived)
    checks = check(ds, derived)
    print(table_text(r))
    print()
    print("\n".join(checks))
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(markdown(ds, ident, derived, r, checks))
        print(f"\nwrote {args.markdown}")


if __name__ == "__main__":
    main()
