import torch

from servosense.telemetry import TelemetryBus

DT = 1 / 240  # decimation 8 at 30 Hz


def run(bus, steps):
    """Feed duty = (control tick index)/1000 so each sample is identifiable; return readings at each tick."""
    z = torch.zeros(1, 1)
    out = []
    for i in range(steps):
        tick = i // bus.decimation
        r = bus.step(torch.full((1, 1), tick / 1000), z, z, z)
        if i % bus.decimation == 0:
            out.append(r["Present_Load"].item())
    return out


def test_latency_is_exactly_one_control_step():
    bus = TelemetryBus(1, 1, DT, read_hz=30, latency_steps=1, noise_counts=0)
    assert bus.decimation == 8
    ticks = run(bus, 8 * 10)
    assert ticks == [0] + list(range(0, 9))


def test_zero_latency_and_hold_between_ticks():
    bus = TelemetryBus(1, 1, DT, latency_steps=0, noise_counts=0)
    z = torch.zeros(1, 1)
    held = [bus.step(torch.full((1, 1), (i // 8) / 1000), z, z, z)["Present_Load"].item() for i in range(24)]
    assert held == [0] * 8 + [1] * 8 + [2] * 8


def test_noise_is_bounded_integer():
    bus = TelemetryBus(256, 6, DT, latency_steps=0, noise_counts=1, generator=torch.Generator().manual_seed(0))
    duty = torch.full((256, 6), 0.3)
    cur = torch.full((256, 6), 1.0)
    z = torch.zeros(256, 6)
    r = bus.step(duty, cur, z, z)
    d = r["Present_Load"] - 300
    assert set(d.unique().tolist()) == {-1, 0, 1}
    assert r["Present_Current"].dtype == torch.int64
    assert (r["Present_Position"] == 2048).all()  # position noise is off by default


def test_dropout_repeats_last_value():
    bus = TelemetryBus(1, 1, DT, latency_steps=0, noise_counts=0, dropout_p=1.0)
    assert run(bus, 8 * 5) == [0] * 5
    bus = TelemetryBus(512, 1, DT, latency_steps=0, noise_counts=0, dropout_p=0.5,
                       generator=torch.Generator().manual_seed(0))  # fmt: skip
    z = torch.zeros(512, 1)
    bus.step(torch.full((512, 1), 0.1), z, z, z)
    r = bus.step(torch.full((512, 1), 0.1), z, z, z)  # not a tick: unchanged
    frac = (r["Present_Load"] == 100).float().mean().item()
    assert 0.4 < frac < 0.6


def test_reset_clears_env():
    bus = TelemetryBus(2, 1, DT, latency_steps=0, noise_counts=0)
    z = torch.zeros(2, 1)
    bus.step(torch.full((2, 1), 0.2), z, z, z)
    bus.reset([0])
    assert bus.readings["Present_Load"].squeeze().tolist() == [0, 200]
