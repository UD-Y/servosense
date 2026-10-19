"""1-DOF push-wall demo: what Present_Load and Present_Current say about a contact force.

A servo drives a 0.2 m lever (STS3215, LeRobot gains, BAM-identified parameters) into a spring wall (2 kN/m at the
lever tip). The goal moves slowly 20 deg past the wall and back. Registers are read like LeRobot does (30 Hz,
+-1 count noise). The rising branch is the motor pushing into the wall; the falling branch is the wall pushing the
joint back as the goal retracts (back-drive). Writes docs/img/push_wall.png and prints the smallest tip force the
telemetry can detect.

    python examples/push_wall.py [--out docs/img/push_wall.png] [--datasheet]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import numpy as np
import torch

from so101_contact import registers
from so101_contact.params import STS3215_7V4, STS3215_7V4_DATASHEET
from so101_contact.servo import STS3215Model
from so101_contact.sim1d import joint_step, spring_wall_torque
from so101_contact.telemetry import TelemetryBus

DT = 1 / 240
LEVER = 0.2  # m
K_WALL = 2000.0  # N/m at the lever tip
MASS = 0.1  # kg at the tip
NOISE = 1  # counts
DETECT_COUNTS = 2 * NOISE + 1  # a change larger than the +-noise band and one quantization step


def run(p, max_deg=20.0, speed=0.05, seed=0):
    p = dataclasses.replace(p, min_delay=1, max_delay=1, protection_time_s=1e9)
    m = STS3215Model(p, 1, DT, num_joints=1, friction_mode="physx_joint")
    bus = TelemetryBus(1, 1, DT, read_hz=30, latency_steps=0, noise_counts=NOISE,
                       generator=torch.Generator().manual_seed(seed))  # fmt: skip
    inertia = p.armature + MASS * LEVER**2
    q = torch.zeros(1, 1)
    qd = torch.zeros(1, 1)
    ext = torch.zeros(1, 1)
    goal, direction = math.radians(-2.0), 1.0
    rows = []
    i = 0
    while True:
        goal += direction * speed * DT
        if goal >= math.radians(max_deg):
            direction = -1.0
        if direction < 0 and goal <= math.radians(-2.0):
            break
        out = m.step(torch.full((1, 1), goal), q, qd, ext)
        tick = i % bus.decimation == 0
        reg = bus.step(out["duty"], out["current"], q, qd)
        ext = spring_wall_torque(q, qd, 0.0, K_WALL * LEVER**2, 0.5)
        q, qd = joint_step(q, qd, out["tau"] + ext, inertia, DT, out["friction_budget"], p.friction_viscous)
        if tick:
            rows.append((math.degrees(goal), direction, -ext.item() / LEVER,
                         registers.present_load(out["duty"]).item(), registers.present_current(out["current"]).item(),
                         reg["Present_Load"].item(), reg["Present_Current"].item()))  # fmt: skip
        i += 1
    return np.array(rows)


def slope(force, counts, lo=0.5, hi=None):
    sel = (force > lo) & ((force < hi) if hi else True)
    return np.polyfit(force[sel], counts[sel], 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs/img/push_wall.png"))
    ap.add_argument("--datasheet", action="store_true", help="use the datasheet-derived preset")
    args = ap.parse_args()
    p = STS3215_7V4_DATASHEET if args.datasheet else STS3215_7V4
    d = run(p)
    goal, up, force, load_true, cur_true, load, cur = d.T
    rise, fall = up > 0, up < 0
    sat_force = force[rise][load_true[rise] >= 0.95 * p.max_pwm * 1000].min()
    s_load = slope(force[rise], load_true[rise], hi=0.8 * sat_force)
    s_cur = slope(force[rise], cur_true[rise], hi=0.8 * sat_force)
    s_load_back = slope(force[fall], load_true[fall], hi=0.8 * sat_force)
    at5 = rise & (np.abs(goal - 5.0) < 0.05)
    f5, l5, c5 = force[at5].mean(), load_true[at5].mean(), cur_true[at5].mean()

    print(f"preset: {'datasheet' if args.datasheet else 'BAM-identified'}; lever {LEVER} m, wall {K_WALL:.0f} N/m")
    print(f"goal 5 deg past the wall: tip force {f5:.2f} N, Present_Load {l5:.0f}, Present_Current {c5:.0f}")
    print(f"Present_Load slope: {s_load:.1f} counts/N pushing, {s_load_back:.1f} counts/N back-driven "
          f"(ratio {s_load_back / s_load:.2f})")
    print(f"Present_Current slope: {s_cur:.1f} counts/N pushing")
    print(f"duty saturates at {sat_force:.1f} N tip force")
    print(f"minimum detectable tip force ({DETECT_COUNTS} counts at +-{NOISE} count noise): "
          f"{DETECT_COUNTS / s_load:.2f} N from Present_Load, {DETECT_COUNTS / s_cur:.2f} N from Present_Current")
    band = np.interp(f5, force[fall][::-1], load_true[fall][::-1])
    print(f"stiction hysteresis at {f5:.1f} N: Present_Load {l5:.0f} pushing vs {band:.0f} back-driven")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, true, noisy, name, s in ((axes[0], load_true, load, "Present_Load (0.1 % duty)", s_load),
                                     (axes[1], cur_true, cur, "Present_Current (6.5 mA)", s_cur)):  # fmt: skip
        ax.scatter(force[rise], noisy[rise], s=6, c="tab:blue", alpha=0.35, label="read, pushing (motor drives)")
        ax.scatter(force[fall], noisy[fall], s=6, c="tab:orange", alpha=0.35, label="read, retracting (back-drive)")
        ax.step(force[rise], true[rise], where="post", c="tab:blue", lw=1)
        ax.step(force[fall], true[fall], where="post", c="tab:orange", lw=1)
        ax.axvline(f5, c="gray", ls=":", lw=1)
        ax.text(f5, ax.get_ylim()[1] * 0.05, " goal 5° past wall", color="gray", fontsize=8)
        ax.set_xlabel("wall force at lever tip [N]")
        ax.set_ylabel(name + " [register counts]")
        ax.set_title(f"{name.split()[0]}: {s:.1f} counts/N, min detectable {DETECT_COUNTS / s:.2f} N")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("STS3215 (LeRobot P=16, 7.4 V) pushing a 0.2 m lever into a 2 kN/m wall; 30 Hz reads, ±1 count noise",
                 fontsize=10)  # fmt: skip
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
