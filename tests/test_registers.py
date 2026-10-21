import math

import torch

from servosense import registers as R


# LeRobot's scalar reference (lerobot/motors/encoding_utils.py), copied for a byte-level comparison.
def lr_encode(value, bit):
    return ((1 if value < 0 else 0) << bit) | abs(value)


def lr_decode(word, bit):
    mag = word & ((1 << bit) - 1)
    return -mag if (word >> bit) & 1 else mag


def test_sign_magnitude_matches_lerobot():
    for bit in (10, 11, 15):
        vals = torch.arange(-((1 << bit) - 1), (1 << bit), 7)
        enc = R.encode_sign_magnitude(vals, bit)
        assert enc.tolist() == [lr_encode(int(v), bit) for v in vals]
        assert R.decode_sign_magnitude(enc, bit).tolist() == vals.tolist()
        assert [lr_decode(int(w), bit) for w in enc] == vals.tolist()


def test_present_load_bit10_and_saturation():
    duty = torch.tensor([0.0, 0.0004, 0.0006, 0.5, -0.5, 0.97, -1.2])
    load = R.present_load(duty)
    assert load.tolist() == [0, 0, 1, 500, -500, 970, -1000]
    raw = R.present_load_raw(duty)
    assert raw[4].item() == (1 << 10) | 500
    assert R.decode_sign_magnitude(raw, 10).tolist() == load.tolist()
    assert torch.allclose(R.load_to_duty(load[3]), torch.tensor(0.5))


def test_present_current_units_and_magnitude():
    amps = torch.tensor([0.0, 0.0065, -0.0065, 1.0, 2.5])
    c = R.present_current(amps)
    assert c.tolist() == [0, 1, 1, 154, 385]
    assert torch.allclose(R.current_to_amps(c[-1]), torch.tensor(385 * 0.0065))


def test_position_and_speed_counts():
    q = torch.tensor([0.0, 2 * math.pi / 4096 * 0.49, 2 * math.pi / 4096 * 0.51, math.pi / 2, -math.pi / 2])
    assert R.present_position(q).tolist() == [2048, 2048, 2049, 3072, 1024]
    assert R.present_speed(torch.tensor([2 * math.pi, -math.pi])).tolist() == [4096, -2048]
    assert torch.allclose(R.counts_to_rad(R.quantize_counts(torch.tensor([1.0]))), torch.tensor([1.0]), atol=8e-4)


def test_register_addresses():
    assert R.REGISTERS["Present_Load"] == (60, 2, 10)
    assert R.REGISTERS["Present_Current"] == (69, 2, None)
    assert R.REGISTERS["Present_Position"][0] == 56
