"""Feetech STS register encodings (batched torch), byte-compatible with LeRobot's Feetech bus.

LeRobot (``lerobot/motors/feetech/tables.py``, ``STS_SMS_SERIES_ENCODINGS_TABLE``) decodes sign-magnitude with
sign bit 10 for ``Present_Load`` and bit 15 for positions and speeds; ``Present_Current`` is not in that table, so
LeRobot returns it raw (unsigned magnitude). The helpers below produce the same raw words and the same decoded ints.
All SI <-> register conversions in the package live here.
"""

from __future__ import annotations

import math

import torch

from .params import COUNTS_PER_REV, CURRENT_LSB_A

# name: (address, bytes, sign bit or None)  -- Feetech STS memory table (CLAUDE.md section 4.1)
REGISTERS: dict[str, tuple[int, int, int | None]] = {
    "Max_Torque_Limit": (16, 2, None),
    "P_Coefficient": (21, 1, None),
    "D_Coefficient": (22, 1, None),
    "I_Coefficient": (23, 1, None),
    "CW_Dead_Zone": (26, 1, None),
    "CCW_Dead_Zone": (27, 1, None),
    "Protection_Current": (28, 2, None),
    "Protective_Torque": (34, 1, None),
    "Protection_Time": (35, 1, None),
    "Overload_Torque": (36, 1, None),
    "Goal_Position": (42, 2, 15),
    "Present_Position": (56, 2, 15),
    "Present_Speed": (58, 2, 15),
    "Present_Load": (60, 2, 10),
    "Present_Voltage": (62, 1, None),
    "Present_Temperature": (63, 1, None),
    "Present_Current": (69, 2, None),
}

LOAD_FULL_SCALE = 1000  # Present_Load: 0.1 % duty per count
POSITION_CENTER = COUNTS_PER_REV // 2  # joint zero of a calibrated arm sits near mid-range


def encode_sign_magnitude(value: torch.Tensor, sign_bit: int) -> torch.Tensor:
    """Signed ints -> sign-magnitude words (LeRobot ``encode_sign_magnitude``), magnitude saturated."""
    value = torch.as_tensor(value).to(torch.int64)
    max_mag = (1 << sign_bit) - 1
    mag = value.abs().clamp(max=max_mag)
    return torch.where(value < 0, mag | (1 << sign_bit), mag)


def decode_sign_magnitude(word: torch.Tensor, sign_bit: int) -> torch.Tensor:
    """Sign-magnitude words -> signed ints (LeRobot ``decode_sign_magnitude``)."""
    word = torch.as_tensor(word).to(torch.int64)
    mag = word & ((1 << sign_bit) - 1)
    return torch.where(((word >> sign_bit) & 1).bool(), -mag, mag)


def rad_to_counts(q: torch.Tensor, cpr: int = COUNTS_PER_REV) -> torch.Tensor:
    """Joint angle [rad] -> continuous encoder counts (unquantized, no offset)."""
    return q * (cpr / (2 * math.pi))


def counts_to_rad(c: torch.Tensor, cpr: int = COUNTS_PER_REV) -> torch.Tensor:
    return torch.as_tensor(c, dtype=torch.float32) * (2 * math.pi / cpr)


def quantize_counts(q: torch.Tensor, cpr: int = COUNTS_PER_REV) -> torch.Tensor:
    """Joint angle [rad] -> integer counts (round to nearest), as the firmware sees it."""
    return torch.round(rad_to_counts(q, cpr)).to(torch.int64)


def present_position(q: torch.Tensor, offset: int = POSITION_CENTER) -> torch.Tensor:
    """Decoded Present_Position [counts]. ``offset`` is the calibrated arm's count at joint zero."""
    return quantize_counts(q) + offset


def present_speed(qd: torch.Tensor) -> torch.Tensor:
    """Decoded Present_Speed [counts/s], signed, saturated to the 15-bit magnitude."""
    return torch.round(rad_to_counts(qd)).to(torch.int64).clamp(-(1 << 15) + 1, (1 << 15) - 1)


def present_load(duty: torch.Tensor) -> torch.Tensor:
    """Decoded Present_Load (signed, 0.1 % duty units): ``round(duty * 1000)`` saturated to +-1000.

    Sign convention: negative duty <-> direction bit set. Which physical direction sets the bit must be verified on
    hardware (CALIBRATION_PROTOCOL.md).
    """
    return torch.round(duty * LOAD_FULL_SCALE).to(torch.int64).clamp(-LOAD_FULL_SCALE, LOAD_FULL_SCALE)


def present_load_raw(duty: torch.Tensor) -> torch.Tensor:
    """Raw Present_Load word: magnitude in bits 0-9, direction in bit 10."""
    return encode_sign_magnitude(present_load(duty), REGISTERS["Present_Load"][2])


def load_to_duty(load: torch.Tensor) -> torch.Tensor:
    """Decoded Present_Load -> duty fraction."""
    return torch.as_tensor(load, dtype=torch.float32) / LOAD_FULL_SCALE


def present_current(current: torch.Tensor) -> torch.Tensor:
    """Present_Current [counts of 6.5 mA]: ``round(|I| / 0.0065)``, unsigned (as LeRobot returns it)."""
    return torch.round(current.abs() / CURRENT_LSB_A).to(torch.int64).clamp(max=(1 << 16) - 1)


def current_to_amps(counts: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(counts, dtype=torch.float32) * CURRENT_LSB_A
