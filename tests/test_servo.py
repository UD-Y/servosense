import dataclasses
import math

import pytest
import torch
from conftest import DT, Rig, no_delay

from so101_contact.params import DATASHEET_7V4, STS3215_7V4, quasi_static_stall_torque
from so101_contact.servo import STS3215Model

COUNT = 2 * math.pi / 4096


def push_into_wall(p, seconds=8.0, ramp=0.2):
    """Ramp the goal slowly through a stiff wall at 0.05 rad; return the settled wall torque [N.m]."""
    rig = Rig(p, wall=(0.05, 200.0, 2.0))
    for i in range(int(seconds / DT)):
        rig.step(min(ramp * i * DT, 1.0))
    return -rig.ext.item(), rig.out


def free_run(p, seconds=2.0):
    rig = Rig(p)
    for _ in range(int(seconds / DT)):
        rig.step(100.0)
    return rig.qd.item()


def test_stall_torque_datasheet_preset_within_5pct(datasheet):
    tau, out = push_into_wall(datasheet)
    assert tau == pytest.approx(DATASHEET_7V4.stall_torque, rel=0.05)
    assert out["duty"].item() == pytest.approx(datasheet.max_pwm)
    assert out["current"].item() == pytest.approx(DATASHEET_7V4.stall_current, rel=0.01)


def test_stall_torque_identified_matches_closed_form(ident):
    tau, _ = push_into_wall(ident)
    assert tau == pytest.approx(quasi_static_stall_torque(ident), rel=0.02)


@pytest.mark.parametrize("preset", ["datasheet", "ident"])
def test_no_load_speed_within_10pct(preset, request):
    p = request.getfixturevalue(preset)
    assert free_run(p) == pytest.approx(DATASHEET_7V4.no_load_speed, rel=0.10)


def test_rate_limiter_caps_internal_target(ident):
    m = STS3215Model(ident, 1, DT, num_joints=1)
    z = torch.zeros(1, 1)
    m.step(z, z, z)
    out = m.step(torch.full((1, 1), 1.0), z, z)
    assert out["internal_target"].item() == pytest.approx(ident.max_velocity * DT, rel=1e-4)


def test_dead_zone_gives_zero_duty(ident):
    m = STS3215Model(ident, 3, DT, num_joints=1)
    q = torch.zeros(3, 1)
    m.step(q, q, q)
    target = torch.tensor([[1 * COUNT], [-1 * COUNT], [2 * COUNT]])
    for _ in range(5):
        out = m.step(target, q, q)
    assert out["duty"][:2].abs().max().item() == 0.0
    assert out["duty"][2].item() == pytest.approx(2 * COUNT * ident.kp_fw * ident.error_gain, rel=1e-5)


def test_duty_saturates_at_max_pwm(ident):
    m = STS3215Model(ident, 2, DT, num_joints=1)
    q = torch.zeros(2, 1)
    m.step(q, q, q)
    for _ in range(100):
        out = m.step(torch.tensor([[1.0], [-1.0]]), q, q)
    assert out["duty"].squeeze().tolist() == pytest.approx([ident.max_pwm, -ident.max_pwm])
    assert (out["motor_tau"].abs() <= m.effort_limit + 1e-6).all()


def test_current_window_respected():
    p = no_delay(STS3215_7V4, max_current=1.0)
    n = 512
    g = torch.Generator().manual_seed(1)
    m = STS3215Model(p, n, DT, num_joints=1)
    q = torch.zeros(n, 1)
    m.step(q, q, q)
    target = (torch.rand(n, 1, generator=g) - 0.5) * 2.0
    qd = (torch.rand(n, 1, generator=g) - 0.5) * 4.0  # |w| <= 2 rad/s keeps the window inside +-max_pwm
    for _ in range(300):
        out = m.step(target, q, qd)
    assert out["current"].abs().max().item() <= 1.0 + 1e-5
    # without the window the same states exceed 1 A
    m2 = STS3215Model(no_delay(STS3215_7V4), n, DT, num_joints=1)
    m2.step(q, q, q)
    for _ in range(300):
        out2 = m2.step(target, q, qd)
    assert out2["current"].abs().max().item() > 2.0


def test_overload_protection_triggers_and_releases():
    p = no_delay(STS3215_7V4, protection_time_s=0.5)
    rig = Rig(p, wall=(0.0, 2000.0, 5.0))
    over_t = trigger_t = None
    for i in range(int(1.0 / DT)):
        out = rig.step(0.5)  # 0.5 rad into a rigid wall: duty saturates above 80 %
        if over_t is None and out["duty"].abs().item() > p.overload_torque_pct / 100:
            over_t = i * DT  # the rate-limited goal needs a few steps to build the error
        if trigger_t is None and out["overload_active"].item():
            trigger_t = i * DT
    assert trigger_t - over_t == pytest.approx(0.5, abs=2 * DT)
    assert out["duty"].abs().item() == pytest.approx(p.holding_pct / 100, rel=1e-5)
    # the motor drops to the holding torque; the wall stays loaded because the gearbox does not back-drive
    assert out["motor_tau"].item() == pytest.approx(p.holding_pct / 100 * p.kt * p.vin / p.resistance, rel=1e-3)
    assert -rig.ext.item() > out["motor_tau"].item()
    for _ in range(int(0.2 / DT)):
        out = rig.step(rig.q.item())  # error drops below the overload threshold
    assert not out["overload_active"].item()


def test_command_delay_is_exact():
    p = dataclasses.replace(STS3215_7V4, min_delay=2, max_delay=2)
    m = STS3215Model(p, 1, DT, num_joints=1)
    z = torch.zeros(1, 1)
    seen = [m.step(torch.full((1, 1), float(k)), z, z)["delayed_target"].item() for k in range(6)]
    assert seen == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_lag_sampled_per_env_and_reset_reseeds():
    m = STS3215Model(STS3215_7V4, 256, DT, num_joints=6, generator=torch.Generator().manual_seed(0))
    assert set(m._lag.unique().tolist()) == {0, 1, 2}
    q = torch.full((256, 6), 0.3)
    m.step(q, q, torch.zeros_like(q))
    m.reset([0])
    out = m.step(torch.zeros_like(q), torch.full((256, 6), -0.2), torch.zeros_like(q))
    assert out["internal_target"][0].tolist() != pytest.approx([0.3] * 6)


def test_explicit_mode_applies_friction_in_tau(ident):
    m = STS3215Model(ident, 1, DT, num_joints=1, friction_mode="explicit")
    z = torch.zeros(1, 1)
    m.step(z, z, z)
    qd = torch.full((1, 1), 1.0)
    out = m.step(z, z, qd)
    expect = out["motor_tau"] - out["friction_budget"] * math.tanh(1.0 / 0.1) - ident.friction_viscous * 1.0
    assert out["tau"].item() == pytest.approx(expect.item(), rel=1e-5)


def test_batched_envs_are_independent(ident):
    n = 8
    m = STS3215Model(ident, n, DT, num_joints=6)
    q = torch.zeros(n, 6)
    m.step(q, q, q)
    target = torch.linspace(-0.2, 0.2, n * 6).reshape(n, 6)
    out = m.step(target, q, q)
    single = STS3215Model(ident, 1, DT, num_joints=6)
    single.step(q[:1], q[:1], q[:1])
    out1 = single.step(target[3:4], q[:1], q[:1])
    assert torch.allclose(out["duty"][3], out1["duty"][0])


def test_bad_friction_mode():
    with pytest.raises(ValueError):
        STS3215Model(STS3215_7V4, 1, DT, num_joints=1, friction_mode="nope")
