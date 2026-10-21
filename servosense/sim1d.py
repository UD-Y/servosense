"""Minimal batched 1-DOF joint integrator for tests and examples (pure torch, no physics engine).

Coulomb friction is applied as an implicit velocity constraint, the way MuJoCo's ``frictionloss`` (and therefore
BAM on MJWarp) treats the friction budget: the joint sticks when the friction impulse can cancel the free velocity.
"""

from __future__ import annotations

import torch


def joint_step(
    q: torch.Tensor,
    qd: torch.Tensor,
    tau: torch.Tensor,
    inertia: torch.Tensor | float,
    dt: float,
    friction_budget: torch.Tensor | float = 0.0,
    viscous: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Semi-implicit Euler step with implicit Coulomb and viscous friction. Returns ``(q, qd)``."""
    v_free = (qd + dt * tau / inertia) / (1 + dt * torch.as_tensor(viscous) / inertia)
    dv = dt * torch.as_tensor(friction_budget) / inertia
    v = torch.sign(v_free) * torch.clamp(v_free.abs() - dv, min=0.0)
    return q + dt * v, v


def spring_wall_torque(q: torch.Tensor, qd: torch.Tensor, q_wall: float, k: float, c: float = 0.0) -> torch.Tensor:
    """Torque of a one-sided spring-damper wall at ``q_wall`` that resists ``q > q_wall`` (never pulls)."""
    pen = (q - q_wall).clamp(min=0.0)
    return torch.where(pen > 0, -(k * pen + c * qd).clamp(min=0.0), torch.zeros_like(q))
