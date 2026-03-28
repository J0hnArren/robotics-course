from pathlib import Path

import numpy as np
import sympy
import torch
import pytorch_kinematics as pk
from pytorch_kinematics.transforms import rotation_conversions

SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

URDF_PATH = Path(__file__).resolve().parent.parent / "assets" / "so101" / "robot.urdf"

# --------------- Simplified model constants (from geometry diagram) ---------------
_D_BASE = 0.0388353       # horizontal base offset (world → shoulder_pan)
_H_BASE = 0.0624          # vertical base offset
_D_SHOULDER = 0.0303992   # shoulder_pan → shoulder_lift radial offset
_H_SHOULDER = 0.0542      # shoulder_pan → shoulder_lift height offset
_L_UPPER = 0.11257        # upper arm link length
_D_ELBOW = 0.028          # elbow perpendicular offset
_L_LOWER = 0.1349         # lower arm link length
_D_WRIST = 0.0052         # wrist perpendicular offset
_L_WRIST = 0.0611         # wrist_flex → wrist_roll distance
_DELTA = 0.0486795        # extra rotation offset at wrist_roll
_L_GRIPPER = 0.1044       # wrist_roll → gripper_frame distance (simplified model)


def so101_downturned_ik_symbolic(
    x: sympy.Symbol,
    y: sympy.Symbol,
    z: sympy.Symbol,
    yaw: sympy.Symbol,
) -> dict[str, sympy.Expr]:
    """
    Return a dict mapping each joint name to a sympy expression in (x, y, z, yaw).

    Parameters
    ----------
    x, y, z, yaw : sympy.Symbol
        Symbols for end-effector position and yaw.

    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a sympy expression (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    d_base = sympy.Float(_D_BASE)
    d_shoulder = sympy.Float(_D_SHOULDER)
    h_base = sympy.Float(_H_BASE)
    h_shoulder = sympy.Float(_H_SHOULDER)
    L_upper = sympy.Float(_L_UPPER)
    d_elbow = sympy.Float(_D_ELBOW)
    L_lower = sympy.Float(_L_LOWER)
    d_wrist = sympy.Float(_D_WRIST)
    L_wrist = sympy.Float(_L_WRIST)
    delta = sympy.Float(_DELTA)
    L_gripper = sympy.Float(_L_GRIPPER)

    # Link lengths and offset angles from perpendicular offsets
    L1 = L_upper
    L2 = L_lower
    alpha1 = sympy.atan2(d_elbow, L_upper)
    alpha2 = sympy.atan2(d_wrist, L_lower)

    # Shoulder_lift position
    z0 = h_base + h_shoulder

    # Tool z-offset (constant for all downturned configs)
    L_tool = L_wrist + L_gripper

    # --- theta1: shoulder_pan ---
    theta1 = sympy.atan2(-y, x - d_base)

    # --- theta5: wrist_roll ---
    theta5 = theta1 + delta - yaw

    # --- 2-link planar IK in the arm plane ---
    R = sympy.sqrt((x - d_base)**2 + y**2)
    dr = R - d_shoulder
    dz = z + L_tool - z0

    D_sq = dr**2 + dz**2
    cos_beta = (D_sq - L1**2 - L2**2) / (2 * L1 * L2)
    beta = -sympy.acos(cos_beta)

    A = sympy.atan2(dz, dr) - sympy.atan2(L2 * sympy.sin(beta), L1 + L2 * sympy.cos(beta))

    theta2 = sympy.pi / 2 - alpha1 - A
    theta3 = alpha1 + alpha2 - sympy.pi / 2 - beta
    theta4 = sympy.pi / 2 - theta2 - theta3

    return {
        "shoulder_pan": theta1,
        "shoulder_lift": theta2,
        "elbow_flex": theta3,
        "wrist_flex": theta4,
        "wrist_roll": theta5,
    }


def analytical_ik_so101_downturned(
    x: float, y: float, z: float, yaw: float
) -> dict[str, float]:
    """
    Evaluate the analytical IK formulas numerically and check joint limits.

    Parameters
    ----------
    x, y, z: float
        Desired end-effector position (x, y, z) in base frame.
    yaw : float
        Desired yaw angle (radians) in the downturned end-effector plane.

    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a float (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    x_sym, y_sym, z_sym, yaw_sym = sympy.symbols("x y z yaw", real=True)
    formulas = so101_downturned_ik_symbolic(x_sym, y_sym, z_sym, yaw_sym)
    func = sympy.lambdify(
        (x_sym, y_sym, z_sym, yaw_sym),
        [formulas[k] for k in SO101_JOINT_NAMES],
        "numpy",
    )
    vals = func(x, y, z, yaw)
    if any(np.isnan(v) for v in vals):
        return None
    q = dict(zip(SO101_JOINT_NAMES, [float(v) for v in vals]))

    chain = pk.build_chain_from_urdf(open(URDF_PATH, mode="rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    low, high = serial_chain.get_joint_limits()
    low, high = np.asarray(low), np.asarray(high)
    for idx, name in enumerate(SO101_JOINT_NAMES):
        if q[name] < low[idx] or q[name] > high[idx]:
            return None
    return q


def numerical_ik_so101_downturned(
    x: float, y: float, z: float, yaw: float
) -> dict[str, float] | None:
    """
    Numerical IK for a downturned SO101 pose.

    Parameters
    ----------
    x, y, z: float
        Desired end-effector position (x, y, z) in base frame.
    yaw : float
        Desired yaw (radians) in the downturned end-effector plane.

    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a float (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    chain = pk.build_chain_from_urdf(open(URDF_PATH, mode="rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")

    # Build target 4x4 transform: downturned with z-axis=[0,0,-1], desired yaw
    # R_target = Rz(-yaw) @ Ry(pi)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz_neg_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry_pi = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    R = Rz_neg_yaw @ Ry_pi

    T_target = np.eye(4, dtype=np.float64)
    T_target[:3, :3] = R
    T_target[:3, 3] = [x, y, z]

    target_tf = pk.Transform3d(
        matrix=torch.tensor(T_target, dtype=torch.float32).unsqueeze(0)
    )

    low, high = serial_chain.get_joint_limits()
    lim = torch.tensor(np.column_stack([low, high]), dtype=torch.float32)

    try:
        ik = pk.PseudoInverseIK(
            serial_chain,
            joint_limits=lim,
            num_retries=100,
            max_iterations=300,
            pos_tolerance=1e-4,
            rot_tolerance=1e-3,
            regularlization=1e-6,
        )
        result = ik.solve(target_tf)
    except Exception:
        return None

    if not result.converged.any():
        return None

    # Pick the best converged solution (lowest error)
    converged_mask = result.converged.squeeze(0)
    all_sols = result.solutions.squeeze(0)

    # Verify via FK and pick the best
    best_q = None
    best_err = float("inf")

    for i in range(all_sols.shape[0]):
        if not converged_mask[i]:
            continue
        q_candidate = all_sols[i:i+1]
        try:
            T_fk = serial_chain.forward_kinematics(q_candidate)
        except Exception:
            continue
        M = T_fk.get_matrix()[0].detach().numpy()
        pos_err = np.linalg.norm(M[:3, 3] - np.array([x, y, z]))
        euler = rotation_conversions.matrix_to_euler_angles(
            torch.tensor(M[:3, :3], dtype=torch.float32).unsqueeze(0), "YZX"
        )
        yaw_fk = float(euler[0, 1])
        cos_diff = np.cos(yaw_fk) * np.cos(yaw) + np.sin(yaw_fk) * np.sin(yaw)
        yaw_err = float(np.arccos(np.clip(cos_diff, -1.0, 1.0)))

        total_err = pos_err + yaw_err
        if pos_err < 1e-3 and yaw_err < 5e-2 and total_err < best_err:
            best_err = total_err
            best_q = q_candidate[0].detach().numpy()

    if best_q is None:
        return None

    return dict(zip(SO101_JOINT_NAMES, [float(v) for v in best_q]))
