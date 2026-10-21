# Calibration protocol (real SO-101)

The default parameters come from Rhoban BAM's identification of one STS3215 at Feetech's default gain (P = 32) and
from the datasheet (see [DERIVATION.md](DERIVATION.md)). Five things cannot be known without a real arm. This
protocol measures them with the arm's own registers, through LeRobot, in about an hour, with no extra sensors
except a kitchen scale.

**Status: written, not yet run on hardware.**

| # | Measures | Why it matters | Equipment |
|---|---|---|---|
| 1 | `error_gain` at LeRobot's P = 16 | sets duty per degree of error, i.e. stiffness and saturation angle | a hard stop (padded block) |
| 2 | Present_Load direction bit, Present_Current scale, `resistance` | sign convention of the telemetry; current vs duty | same as 1 |
| 3 | gearbox friction `friction_base`, `load_friction_external` | the push vs back-drive asymmetry | 2 known masses (e.g. 100 g, 200 g) |
| 4 | Present_Load per newton at the gripper tip | the contact signal itself: the sim-to-real check | kitchen scale (0.1 N or better) |
| 5 | sim and real joint zeros | to replay sim actions on the real arm | the LeRobot calibration file |

## Setup

- LeRobot SO-101 follower, calibrated with `lerobot-calibrate` (LeRobot writes P = 16, I = 0, D = 32).
- Read registers with `servosense.lerobot_hw.read_servo_telemetry(robot.bus)`. It returns Present_Load
  (signed, LeRobot-decoded) and Present_Current (raw counts), decoded exactly like the simulated bus. Read
  Present_Position and Present_Voltage with `robot.bus.sync_read(name, normalize=False)`.
- Log at 30 Hz; average 30 samples (1 s) per measurement point. The ±1 count noise then drops to about 0.2 counts.
- **Safety:** a stall above Overload_Torque (80 %) for Protection_Time (2 s) triggers overload protection. Keep every
  stall point under 1.5 s, or set Max_Torque_Limit to 500 for tests 1 and 2. Pad all hard stops.

## 1. Firmware error gain at P = 16 (stall slope)

1. Drive the elbow (`elbow_flex`) slowly onto a padded hard stop and stop when it touches. Read `Present_Position`
   = `p0` (counts).
2. For `k = 0, 5, 10, ..., 150` counts (up to ~13°, below duty saturation): write `Goal_Position = p0 + k`, wait
   0.3 s, average Present_Load over 1 s, then return to `p0` for 1 s (protection timer reset).
3. The joint does not move (ω = 0), so `duty = err · kp_fw · error_gain`, with `err = k · 2π/4096` rad.
   Fit `Present_Load = a · k + b` for the points above the dead zone:

   ```
   error_gain = a · 4096 / (2π · 1000 · 16)      # 1/rad per unit P
   ```

   Default: 0.192 (Rhoban oscilloscope value 0.166 × fitted ratio 1.156, measured at P = 32).
   The randomization range is ×[0.5, 2] until this is measured.
4. Read off the zero-crossing too: `|b|` > 0 at small `k` gives `min_startup_duty` (Present_Load / 1000), and the
   largest `k` with Present_Load = 0 gives `dead_zone_counts`.
5. Repeat on `shoulder_lift` (against the base stop) to check that the gain is the same per servo.

## 2. Sign convention and current scale

From the same stall points:

- **Direction bit:** record the sign of the decoded Present_Load for a positive `k` (Goal above Present). The sim
  assumes positive duty means bit 10 is clear (`registers.present_load`). If the real servo is the opposite, flip the sign
  in `registers.present_load` and say so in the release notes.
- **Current:** at ω = 0, `I = V · duty / R`. Fit `Present_Current · 0.0065` (A) against
  `Present_Voltage · 0.1 · Present_Load / 1000` (V): the slope is `1/R`. Default `resistance` = 2.75 Ω (BAM fit);
  datasheet derivation 2.87 Ω.
- Check whether Present_Current ever reports a sign (bit 15) when the load reverses. LeRobot reads it raw; the sim
  reports magnitude.

## 3. Gearbox friction and the push / back-drive asymmetry

1. Hold the arm with the forearm horizontal and hang mass `m` at a measured lever `r` from the elbow axis
   (gravity torque `T = m g r`, plus the arm's own, measured with no mass).
2. **Lift:** ramp `Goal_Position` of `elbow_flex` upward at 5°/s through ±10° around horizontal; average Present_Load
   while moving through horizontal. The motor drives the load.
3. **Lower:** ramp downward through the same angle at 5°/s. The load drives the motor.
4. The model predicts motor torque `T + budget` when lifting and `T - budget` when lowering, with
   `budget ≈ friction_base + μ_e·T + μ_m·τ` (stribeck terms vanish at 5°/s). With two masses (and none), solve the
   three equations for `friction_base`, `μ_e` and the duty-to-torque scale. Defaults: `friction_base` = 0.053 N·m,
   `μ_e` = 0.24 (+0.15 near rest), `μ_m` = 0.039.
5. Report the ratio `ΔLoad(lower) / ΔLoad(lift)` for the same mass change. The sim predicts about 0.4 to 0.6. This is the
   asymmetry the benchmark's telemetry controller relies on.

## 4. Contact signal at the gripper (sim-to-real check)

1. Put a kitchen scale under the gripper tip, about 1 cm below it, with the arm in the press-test pose of
   `latchbench/scripts/hold_test.py`.
2. Ramp `shoulder_lift` down at 2°/s until the scale reads ~8 N, then back up. Log the scale reading (by hand or
   with a phone camera on the display, synced to the log) and Present_Load on `shoulder_lift` and `elbow_flex`.
3. Fit Present_Load vs force on the pressing branch: counts per newton, and the noise floor at zero force. The
   simulated SO-101 gives about 55 to 90 counts/N on `shoulder_lift` (results/hold_test*), and the 1-DOF push-wall
   example 84.5 counts/N at 0.2 m. The smallest detectable force is `3 counts / slope`.
4. Compare the unloading branch: the hysteresis between pressing and releasing is the stiction band.

## 5. Joint zeros (sim and real)

The sim's joint zero is the USD pose; the real arm's zero comes from `lerobot-calibrate` (homing offsets and ranges
in `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<id>.json`). `isaaclab_so101.mapping`
(`motor_norm_to_sim_radians`, `sim_radians_to_motor_norm`) converts between them. Record the per-joint offset
between the real arm at its calibration pose and the sim at the same pose, and store the calibration file next to
any dataset meant for the real arm. Don't fake it: a dataset without the real calibration file can't be replayed.

## Writing the results back

```python
import dataclasses
from servosense import STS3215_7V4, so101_follower

measured = dataclasses.replace(STS3215_7V4, error_gain=..., resistance=..., friction_base=...,
                               load_friction_external=..., min_startup_duty=..., dead_zone_counts=...)
params = so101_follower(measured)   # gripper overrides applied on top
```

Then rerun `pytest -q` and `python examples/push_wall.py`, and narrow `RandomizationRanges` around the measured
values (e.g. `error_gain` ×[0.8, 1.25]). Commit the raw logs with the fit so the numbers can be audited.
