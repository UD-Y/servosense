"""ServoTelemetryProcessorStep against a fake LeRobot Feetech bus (no hardware, lerobot optional)."""

import torch

from so101_contact import registers
from so101_contact.lerobot_hw import OBS_SERVO_CURRENT, OBS_SERVO_LOAD, ServoTelemetryProcessorStep
from so101_contact.params import LEROBOT_MOTOR_ORDER


class FakeFeetechBus:
    """Holds raw register words and decodes them the way LeRobot's FeetechMotorsBus.sync_read does."""

    sign_bits = {"Present_Load": 10}  # STS_SMS_SERIES_ENCODINGS_TABLE

    def __init__(self, raw: dict[str, dict[str, int]]):
        self.motors = {name: i + 1 for i, name in enumerate(LEROBOT_MOTOR_ORDER)}
        self.raw = raw
        self.calls = []

    def sync_read(self, data_name, motors=None, *, normalize=True):
        self.calls.append((data_name, tuple(motors), normalize))
        bit = self.sign_bits.get(data_name)
        out = {}
        for m in motors:
            word = self.raw[data_name][m]
            out[m] = int(registers.decode_sign_magnitude(torch.tensor(word), bit)) if bit else word
        return out


class FakeRobot:
    def __init__(self, bus):
        self.bus = bus


def test_step_matches_sim_encoding():
    duty = torch.tensor([0.12, -0.34, 0.0, 0.97, -0.005, 0.25])
    amps = torch.tensor([0.4, 1.1, 0.0, 2.6, 0.01, 0.3])
    raw_load = registers.present_load_raw(duty)
    raw_cur = registers.present_current(amps)
    bus = FakeFeetechBus({
        "Present_Load": dict(zip(LEROBOT_MOTOR_ORDER, raw_load.tolist())),
        "Present_Current": dict(zip(LEROBOT_MOTOR_ORDER, raw_cur.tolist())),
    })  # fmt: skip
    step = ServoTelemetryProcessorStep(robot=FakeRobot(bus))
    obs = step.observation({"observation.state": torch.zeros(1, 6)})
    assert obs[OBS_SERVO_LOAD].shape == (1, 6) and obs[OBS_SERVO_LOAD].dtype == torch.float32
    # identical to what the sim bus would report for the same duty / current
    assert obs[OBS_SERVO_LOAD][0].tolist() == registers.present_load(duty).float().tolist()
    assert obs[OBS_SERVO_CURRENT][0].tolist() == registers.present_current(amps).float().tolist()
    assert obs["observation.state"].shape == (1, 6)  # state untouched
    assert [c[0] for c in bus.calls] == ["Present_Load", "Present_Current"]
    assert all(c[1] == LEROBOT_MOTOR_ORDER and c[2] is False for c in bus.calls)


def test_requires_robot():
    import pytest

    with pytest.raises(ValueError):
        ServoTelemetryProcessorStep().observation({})
