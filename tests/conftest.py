import dataclasses

import pytest
import torch

from servosense.params import STS3215_7V4, STS3215_7V4_DATASHEET
from servosense.servo import STS3215Model
from servosense.sim1d import joint_step, spring_wall_torque

DT = 1 / 240


def no_delay(p, **kw):
    """Deterministic variant: no command lag, protection off unless asked."""
    return dataclasses.replace(p, min_delay=0, max_delay=0, **{"protection_time_s": 1e9, **kw})


class Rig:
    """1-DOF joint driven by one servo, Coulomb budget applied as a constraint (BAM / MJWarp convention)."""

    def __init__(self, p, n=1, inertia=None, dt=DT, wall=None):
        self.p, self.dt = p, dt
        self.m = STS3215Model(p, n, dt, num_joints=1, friction_mode="physx_joint")
        self.J = inertia or p.armature
        self.q = torch.zeros(n, 1)
        self.qd = torch.zeros(n, 1)
        self.ext = torch.zeros(n, 1)
        self.wall = wall  # (q_wall, k, c)
        self.out = None

    def step(self, target, load=0.0):
        """``load``: extra external torque (added to the wall's)."""
        target = torch.as_tensor(target, dtype=torch.float32).expand_as(self.q)
        self.out = self.m.step(target, self.q, self.qd, self.ext)
        ext = torch.as_tensor(load, dtype=torch.float32).expand_as(self.q).clone()
        if self.wall:
            ext = ext + spring_wall_torque(self.q, self.qd, *self.wall)
        self.q, self.qd = joint_step(
            self.q, self.qd, self.out["tau"] + ext, self.J, self.dt, self.out["friction_budget"], self.p.friction_viscous
        )
        self.ext = ext
        return self.out


@pytest.fixture
def ident():
    return no_delay(STS3215_7V4)


@pytest.fixture
def datasheet():
    return no_delay(STS3215_7V4_DATASHEET)
