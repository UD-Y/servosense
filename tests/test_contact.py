"""The two claims the benchmark rests on: Present_Load tracks contact force, and it does so asymmetrically."""

import math

import torch
from conftest import DT, Rig

from servosense import registers


def settle(rig, target, seconds):
    for _ in range(int(seconds / DT)):
        rig.step(target)


def test_present_load_increases_monotonically_with_wall_force(ident):
    """Lever r = 0.2 m against a 2 kN/m wall; goal stepped deeper past the wall, approached slowly."""
    r_lever, k_lin = 0.2, 2000.0
    rig = Rig(ident, inertia=ident.armature + 0.1 * r_lever**2, wall=(0.0, k_lin * r_lever**2, 1.0))
    forces, loads = [], []
    goal = 0.0
    for deg in range(0, 17, 2):
        while goal < math.radians(deg):  # slow approach: 0.1 rad/s
            goal = min(goal + 0.1 * DT, math.radians(deg))
            rig.step(goal)
        settle(rig, goal, 0.5)
        forces.append(-rig.ext.item() / r_lever)
        loads.append(registers.present_load(rig.out["duty"]).item())
    assert all(b > a for a, b in zip(forces[1:], forces[2:])), forces
    assert all(b > a for a, b in zip(loads[1:], loads[2:])), loads
    assert loads[-1] > 400 and forces[-1] > 5.0


def mean_duty_while_moving(p, direction, load, speed=1.0, seconds=1.5):
    """Goal ramps at ``speed`` in ``direction`` against a constant external torque ``load``; mean duty in steady motion."""
    rig = Rig(p)
    duties = []
    goal = 0.0
    n = int(seconds / DT)
    for i in range(n):
        goal += direction * speed * DT
        out = rig.step(goal, load=load)
        if i > n // 2:
            duties.append(out["duty"].item())
    return sum(duties) / len(duties)


def test_backdrive_asymmetry(ident):
    """Same external torque: the motor pushing into the load changes Present_Load more than the load back-driving it.

    The load opposes +motion (-T). Driving: the shaft moves +, the motor lifts the load. Back-driving: the shaft moves
    -, the load wins and the gearbox friction helps the motor hold it.
    """
    T = 0.8
    d_drive = mean_duty_while_moving(ident, +1, -T) - mean_duty_while_moving(ident, +1, 0.0)
    d_back = mean_duty_while_moving(ident, -1, -T) - mean_duty_while_moving(ident, -1, 0.0)
    assert d_drive > 0 and d_back > 0
    mu_e = ident.load_friction_external
    expected = (1 - mu_e) / (1 + mu_e)  # moving fast enough that the Stribeck terms vanish
    ratio = d_back / d_drive
    assert abs(ratio - expected) < 0.08, (ratio, expected)
    assert ratio < 0.7
    # in register counts the difference is resolvable many times over
    assert (d_drive - d_back) * 1000 > 20


def test_load_rises_with_contact_in_batch(ident):
    """Batched: 4 envs, walls at different stiffness, same goal -> force and load ordered the same way."""
    n = 4
    ks = torch.tensor([[20.0], [40.0], [80.0], [160.0]])
    from servosense.servo import STS3215Model
    from servosense.sim1d import joint_step, spring_wall_torque

    m = STS3215Model(ident, n, DT, num_joints=1, friction_mode="physx_joint")
    q = torch.zeros(n, 1)
    qd = torch.zeros(n, 1)
    ext = torch.zeros(n, 1)
    goal = torch.zeros(n, 1)
    for i in range(int(3.0 / DT)):
        goal = torch.clamp(goal + 0.1 * DT, max=0.15)
        out = m.step(goal, q, qd, ext)
        ext = -torch.clamp(ks * q, min=0) * (q > 0)
        q, qd = joint_step(q, qd, out["tau"] + ext, ident.armature, DT, out["friction_budget"], ident.friction_viscous)
    f = -ext.squeeze()
    load = registers.present_load(out["duty"]).squeeze()
    assert torch.all(f[1:] > f[:-1]) and torch.all(load[1:] >= load[:-1])
