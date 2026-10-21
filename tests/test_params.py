import dataclasses
import math

import pytest
import torch

from servosense import params as P


def test_datasheet_preset_reproduces_datasheet():
    p, ds = P.STS3215_7V4_DATASHEET, P.DATASHEET_7V4
    assert P.quasi_static_stall_torque(p) == pytest.approx(ds.stall_torque, rel=1e-6)
    assert p.max_pwm * p.vin / p.resistance == pytest.approx(ds.stall_current, rel=1e-6)
    assert P.no_load_speed(p) == pytest.approx(ds.no_load_speed, rel=1e-6)


def test_identified_preset_is_close_to_datasheet():
    """The hardware fit is not tuned to the datasheet; record how far apart they are."""
    p, ds = P.STS3215_7V4, P.DATASHEET_7V4
    assert P.quasi_static_stall_torque(p) / ds.stall_torque == pytest.approx(1.18, abs=0.02)
    assert P.no_load_speed(p) / ds.no_load_speed == pytest.approx(0.97, abs=0.02)
    # electrical constants agree with the datasheet derivation within 5 %
    d = P.STS3215_7V4_DATASHEET
    assert p.kt / d.kt == pytest.approx(1.0, abs=0.05)
    assert p.resistance / d.resistance == pytest.approx(1.0, abs=0.05)


def test_effective_kt_matches_datasheet_kt():
    """The datasheet's 8 kg.cm/A is not used in the derivation: it is an independent check."""
    p = P.STS3215_7V4_DATASHEET
    assert P.quasi_static_stall_torque(p) / P.DATASHEET_7V4.stall_current == pytest.approx(0.785, rel=0.05)


def test_error_gain_matches_kp_over_4_hypothesis():
    kp4 = (4096 / (2 * math.pi)) / 1000 / 4
    assert 0.166 == pytest.approx(kp4, rel=0.03)  # Rhoban's oscilloscope value vs CLAUDE.md hypothesis


def test_inconsistent_datasheet_raises():
    with pytest.raises(ValueError):
        P.derive_from_datasheet(dataclasses.replace(P.DATASHEET_7V4, stall_torque=4.0))


def test_12v_preset_matches_its_stall_torque():
    assert P.quasi_static_stall_torque(P.STS3215_12V) == pytest.approx(2.942, rel=1e-6)


def test_so101_follower_gripper_overrides():
    f = P.so101_follower()
    assert tuple(f) == P.LEROBOT_MOTOR_ORDER
    g, arm = f["gripper"], f["shoulder_lift"]
    assert g.max_pwm == pytest.approx(arm.max_pwm / 2)
    assert g.max_current == pytest.approx(250 * 0.0065)
    assert g.overload_torque_pct == 25.0
    assert arm.kp_fw == 16.0 and arm.max_current == 0.0


def test_randomize_shapes_ranges_and_determinism():
    n = 64
    a = P.randomize(P.SO101_FOLLOWER, n, torch.Generator().manual_seed(0))
    b = P.randomize(P.SO101_FOLLOWER, n, torch.Generator().manual_seed(0))
    nominal = P.to_tensors(P.SO101_FOLLOWER, n)
    rr = P.RandomizationRanges()
    for k, v in a.items():
        assert v.shape == (n, 6)
        assert torch.equal(v, b[k])
    for k, (lo, hi) in rr.scales.items():
        s = a[k] / nominal[k]
        assert s.min() >= lo - 1e-6 and s.max() <= hi + 1e-6, k
        assert s.std() > 0, k  # independent draw per env-joint
    lag = a["min_delay"]
    assert torch.equal(lag, a["max_delay"]) and torch.equal(lag, lag.round())
    assert set(lag.unique().tolist()) == {0.0, 1.0, 2.0}


def test_to_bam_kwargs_uses_bam_names():
    # BamMotorCfg / BamActuatorCfg field names on Isaac Lab develop @ 0ec7c29
    bam_motor = {
        "model", "kt", "resistance", "error_gain", "max_pwm", "max_current", "friction_base", "friction_viscous",
        "friction_stribeck", "dtheta_stribeck", "alpha", "load_friction_motor", "load_friction_external",
        "load_friction_motor_stribeck", "load_friction_external_stribeck", "load_friction_motor_quad",
        "load_friction_external_quad",
    }  # fmt: skip
    bam_actuator = {"motor", "kp_fw", "vin", "vin_range", "min_delay", "max_delay", "armature"}
    kw = P.to_bam_kwargs(P.STS3215_7V4)
    assert set(kw["motor"]) <= bam_motor
    assert set(kw) <= bam_actuator
    assert kw["motor"]["model"] == "m5"
    assert isinstance(kw["min_delay"], int)
    assert not set(P.FIRMWARE_ONLY_FIELDS) & (set(kw) | set(kw["motor"]))
