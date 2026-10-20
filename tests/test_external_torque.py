import math

import pytest
import torch
from conftest import DT

from so101_contact.external_torque import estimate_external_torque
from so101_contact.servo import STS3215Model
from so101_contact.sim1d import joint_step

B = torch.tensor([[0.3]])  # engine static = dynamic friction [N.m]
CV = 0.03
G = torch.tensor([[0.4]])  # gravity torque on the joint [N.m]


def est(ext_true, f, tau, qd, qdd=0.0, arm=0.02, prior=G):
    tau = torch.as_tensor([[tau]])
    qdd_t = torch.as_tensor([[qdd]])
    # equation of motion (M without armature): M qdd + arm qdd = tau + f + ext
    mqdd = (tau + f + ext_true) - arm * qdd_t
    return estimate_external_torque(
        mqdd, qdd_t, tau, arm, qd=torch.as_tensor([[qd]]), prior=prior,
        static_friction=B, dynamic_friction=B, viscous_friction=CV,
    )  # fmt: skip


@pytest.mark.parametrize("tau", [-0.7, -0.55, -0.4, -0.1])  # every motor torque the friction can hold
def test_stuck_without_contact_returns_gravity_whatever_the_motor_torque(tau):
    f = -(torch.tensor([[tau]]) + G)  # stuck: the engine friction cancels tau + gravity
    assert f.abs().item() <= B.item() + 1e-6
    assert est(G, f, tau, qd=0.0).item() == pytest.approx(G.item(), abs=1e-6)


@pytest.mark.parametrize("qd", [-2.0, -0.01, 0.01, 2.0])
def test_sliding_removes_dynamic_and_viscous_friction_exactly(qd):
    contact = torch.tensor([[-0.7]])
    f = -(B * math.copysign(1.0, qd) + CV * qd)
    got = est(G + contact, f, 0.9, qd=qd, qdd=3.0)
    assert got.item() == pytest.approx((G + contact).item(), abs=1e-6)


def test_stuck_with_contact_underestimates_by_at_most_twice_the_budget():
    contact = torch.tensor([[-1.5]])
    tau = 1.2
    for share in (0.0, 0.5, 1.0):  # how much of the holding torque PhysX puts on the friction constraint
        f = -share * B
        c = -(tau + G + f)  # the contact takes the rest (static)
        got = est(G + c, f, tau, qd=0.0)
        assert (got - (G + c)).abs().item() <= 2 * B.item() + 1e-6
        assert got.item() < 0.0  # the contact is still seen


def test_no_engine_friction_returns_raw_with_armature():
    tau, qdd, arm = torch.tensor([[0.5]]), torch.tensor([[10.0]]), 0.02
    ext = torch.tensor([[0.4]])
    mqdd = tau + ext - arm * qdd
    got = estimate_external_torque(mqdd, qdd, tau, arm)
    assert got.item() == pytest.approx(ext.item(), abs=1e-6)
    with pytest.raises(ValueError):
        estimate_external_torque(mqdd, qdd, tau, arm, static_friction=B)


def run_physx_like(p, mode, goal_fn, seconds, gravity=0.43, j_link=0.0104):
    """1-DOF joint as PhysX runs it in ``physx_joint`` mode: the servo's budget is an engine friction constraint,
    the mass matrix excludes the armature, and qdd is finite-differenced. ``mode`` picks the external-torque input:
    ``oracle`` (true gravity), ``new`` (this estimator) or ``old`` (M qdd - tau, the 19 Oct actuator)."""
    m = STS3215Model(p, 1, DT, num_joints=1, friction_mode="physx_joint")
    q, qd, ext = torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1)
    g = torch.full((1, 1), gravity)
    log = []
    for i in range(int(seconds / DT)):
        out = m.step(torch.full((1, 1), goal_fn(i * DT)), q, qd, ext)
        tau, budget = out["tau"], out["friction_budget"]
        q, qd_new = joint_step(q, qd, tau + g, j_link + p.armature, DT, budget, p.friction_viscous)
        qdd = (qd_new - qd) / DT
        qd = qd_new
        if mode == "oracle":
            ext = g.clone()
        elif mode == "old":
            ext = j_link * qdd - tau
        else:
            ext = estimate_external_torque(
                j_link * qdd, qdd, tau, p.armature, qd=qd, prior=g,
                static_friction=budget, dynamic_friction=budget, viscous_friction=p.friction_viscous,
            )  # fmt: skip
        log.append((i * DT, q.item(), ext.item(), budget.item()))
    return log


def test_stuck_joint_breaks_away_like_the_oracle(ident):
    """Held against gravity, then the goal ramps away: the joint must restart at the oracle's command error."""
    goal = lambda t: 0.0 if t < 0.5 else -math.radians(10) * (t - 0.5) / 2.0  # noqa: E731

    def break_away(mode):
        log = run_physx_like(ident, mode, goal, 3.0)
        q0 = log[int(0.5 / DT)][1]
        t = next(r[0] for r in log if r[0] > 0.5 and abs(r[1] - q0) > math.radians(0.2))
        return t, log

    t_oracle, lo = break_away("oracle")
    t_new, ln = break_away("new")
    t_old, _ = break_away("old")
    assert abs(t_new - t_oracle) <= 2 * DT
    assert ln[-1][2] == pytest.approx(lo[-1][2], abs=0.01)  # estimate == gravity on the stuck joint
    assert t_old > t_oracle + 0.05  # the 19 Oct estimate held the joint longer (self-feedback)


def test_step_settles_like_the_oracle(ident):
    """30 deg step against gravity: final error and friction budget match the oracle (the old estimate stuck 0.6 deg
    further out with a 50 % larger budget, as in hold_test_physx_joint)."""
    goal = lambda t: math.radians(30)  # noqa: E731
    lo = run_physx_like(ident, "oracle", goal, 2.0)
    ln = run_physx_like(ident, "new", goal, 2.0)
    lold = run_physx_like(ident, "old", goal, 2.0)
    assert math.degrees(abs(ln[-1][1] - lo[-1][1])) < 0.05
    assert ln[-1][3] == pytest.approx(lo[-1][3], rel=0.02)
    assert lold[-1][3] > 1.3 * lo[-1][3]
