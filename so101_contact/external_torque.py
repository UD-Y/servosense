"""BAM's external joint torque estimated from articulation dynamics, without the engine's own joint friction.

BAM's load-dependent friction needs the torque the environment applies to each joint (gravity, Coriolis, contacts,
joint limits), *excluding* the actuator's own friction: on MJWarp BAM reads it as
``-qfrc_bias + qfrc_constraint - own_friction`` (``gather_external_torque_kernel``). PhysX exposes neither the bias
split nor the friction constraint force, so we rebuild it from the joint equation of motion, per joint ``j``::

    (M qdd)_j + armature_j * qdd_j = tau_j + f_j + ext_j

``M`` is PhysX's generalized mass matrix, which does **not** contain the joint armature (measured on the SO-101,
Isaac Lab ``bb0c8e1``: ``M_ii`` of the shoulder is 0.0104 kg.m^2 with 0.0216 armature set; without the armature term
the residual against gravity has sd 0.27 N.m during a 30 deg step, with it 0.002 N.m). ``tau`` is the effort the
actuator applied during the last physics step and ``f`` the PhysX joint friction. So ``raw = (M qdd) + armature*qdd
- tau = ext + f``.

When the friction budget is written to PhysX (``friction_mode="physx_joint"``), ``f`` is the friction *reaction*:
on a stuck joint it cancels the motor torque, so ``raw`` grows with the command, the load-dependent budget grows with
``raw`` and the joint never breaks away. ``f`` is removed as PhysX applies it (measured, same probe):

- sliding (``|qd| > stick_velocity``): ``f = -(dynamic * sign(qd) + viscous * qd)``, exact to 0.001 N.m;
- stuck (PhysX holds ``|qd| < 1e-5`` rad/s): ``f`` is any value in ``[-static, static]``. It is estimated as the part
  of the residual against a friction-free ``prior`` (gravity) that the friction can hold:
  ``f = clamp(raw - prior, -static, static)``. Without contact this returns the prior exactly. With a contact on a
  stuck joint, the split between the contact and the friction constraint is not observable; the estimate then
  under-reads the contact torque by at most ``2*static``, and never feeds the motor torque back into the budget.

Pure torch, so it runs (and is tested) without Isaac Sim.
"""

from __future__ import annotations

import torch

STICK_VELOCITY = 1e-3
"""rad/s. PhysX holds a stuck joint at ``|qd| < 1e-5``; every sample above 1e-3 had saturated, sliding friction."""


def estimate_external_torque(
    mqdd: torch.Tensor,
    qdd: torch.Tensor,
    tau_prev: torch.Tensor,
    armature: torch.Tensor | float = 0.0,
    qd: torch.Tensor | None = None,
    prior: torch.Tensor | None = None,
    static_friction: torch.Tensor | float | None = None,
    dynamic_friction: torch.Tensor | float | None = None,
    viscous_friction: torch.Tensor | float = 0.0,
    stick_velocity: float = STICK_VELOCITY,
) -> torch.Tensor:
    """External torque on each joint during the last physics step [N.m], ``(num_envs, num_joints)``.

    Args:
        mqdd: ``(M qdd)`` rows of the actuated joints (PhysX mass matrix, armature excluded).
        qdd: joint acceleration over the last step [rad/s^2].
        tau_prev: effort applied by the actuator during the last step [N.m].
        armature: joint armature as set in the engine [kg.m^2].
        qd: joint velocity at the end of the last step [rad/s]; needed with ``static_friction``.
        prior: friction-free part of the external torque the caller can compute (e.g. ``-gravity_compensation``);
            needed with ``static_friction``.
        static_friction, dynamic_friction, viscous_friction: the engine's joint friction during the last step
            (PhysX static / dynamic friction effort [N.m], viscous coefficient [N.m.s/rad]). ``static_friction=None``
            means the engine applies no joint friction: the raw estimate is returned.
        stick_velocity: below this ``|qd|`` the joint is treated as stuck [rad/s].
    """
    raw = mqdd + armature * qdd - tau_prev
    if static_friction is None:
        return raw
    if qd is None or prior is None:
        raise ValueError("qd and prior are needed to remove the engine's joint friction")
    static = torch.as_tensor(static_friction, dtype=raw.dtype, device=raw.device)
    dynamic = static if dynamic_friction is None else torch.as_tensor(dynamic_friction).to(raw)
    f_slide = -(dynamic * torch.sign(qd) + viscous_friction * qd)
    f_stick = torch.maximum(torch.minimum(raw - prior, static), -static)
    f = torch.where(qd.abs() > stick_velocity, f_slide, f_stick)
    return raw - f
