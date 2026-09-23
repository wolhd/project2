"""
Example: UKF tracking a 2D target with a Cartesian constant-velocity state,
observed through a range-bearing sensor.

State:  x = [px, vx, py, vy]          (Cartesian position + velocity)
Meas:   z = [range, bearing]          (polar, relative to a sensor position)

This file only defines the *problem-specific* pieces (fx, hx, angle-aware
mean/residual for the measurement) and wires them into the generic
`ukf.py` core. To track a different state (e.g. add acceleration, or track
in 3D) or use a different sensor (e.g. pure bearing, or range/bearing/range-
rate), you only need to swap out this file's fx/hx/mean/residual functions
and adjust dim_x / dim_z accordingly -- ukf.py itself never changes.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ukf import (
    MerweScaledSigmaPoints,
    UnscentedKalmanFilter,
    unscented_transform,
    normalize_angle,
)


# ---------------------------------------------------------------------------
# Problem-specific models
# ---------------------------------------------------------------------------

def fx_constant_velocity(x, dt):
    """State transition for [px, vx, py, vy] under constant velocity."""
    F = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1],
    ])
    return F @ x


def hx_range_bearing(x, sensor_pos=(0.0, 0.0)):
    """Measurement model: Cartesian state -> [range, bearing]."""
    dx = x[0] - sensor_pos[0]
    dy = x[2] - sensor_pos[1]
    r = np.hypot(dx, dy)
    b = np.arctan2(dy, dx)
    return np.array([r, b])


def z_mean_range_bearing(sigmas, Wm):
    """Weighted mean of [range, bearing] sigma points; bearing is circular."""
    r = np.dot(Wm, sigmas[:, 0])
    sum_sin = np.dot(Wm, np.sin(sigmas[:, 1]))
    sum_cos = np.dot(Wm, np.cos(sigmas[:, 1]))
    b = np.arctan2(sum_sin, sum_cos)
    return np.array([r, b])


def residual_range_bearing(a, b):
    """a - b for [range, bearing], with bearing difference wrapped."""
    y = a - b
    y[1] = normalize_angle(y[1])
    return y


# ---------------------------------------------------------------------------
# Filter construction helper
# ---------------------------------------------------------------------------

def build_range_bearing_ukf(dt, q_var, sensor_pos=(0.0, 0.0)):
    """
    Build a ready-to-run UKF for the Cartesian-state / range-bearing-
    measurement problem.
    """
    dim_x, dim_z = 4, 2
    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2.0, kappa=3 - dim_x)

    ukf = UnscentedKalmanFilter(
        dim_x=dim_x,
        dim_z=dim_z,
        fx=fx_constant_velocity,
        hx=lambda x: hx_range_bearing(x, sensor_pos=sensor_pos),
        points=points,
        z_mean_fn=z_mean_range_bearing,
        residual_z=residual_range_bearing,
        # state is plain Cartesian -> default mean/residual (subtraction) is fine
    )

    # Discretized white-noise-acceleration process noise, one block per axis.
    q = q_var * np.array([
        [dt ** 4 / 4, dt ** 3 / 2],
        [dt ** 3 / 2, dt ** 2],
    ])
    Q = np.zeros((dim_x, dim_x))
    Q[0:2, 0:2] = q          # px, vx
    Q[2:4, 2:4] = q          # py, vy
    ukf.Q = Q

    return ukf


# ---------------------------------------------------------------------------
# Quick correctness check: a UKF-propagated *linear* transform must match
# the exact closed-form linear result (see the "verify a UT" discussion).
# ---------------------------------------------------------------------------

def _sanity_check_linear():
    n = 4
    points = MerweScaledSigmaPoints(n=n, alpha=0.1, beta=2.0, kappa=3 - n)
    mu = np.array([1.0, 2.0, -3.0, 0.5])
    Sigma = np.diag([1.0, 0.5, 2.0, 0.3])

    A = np.array([
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ])
    b = np.array([0.1, -0.1, 0.2, 0.0])
    f = lambda x: A @ x + b

    sigmas = points.sigma_points(mu, Sigma)
    sigmas_f = np.array([f(s) for s in sigmas])
    mean_ut, cov_ut = unscented_transform(sigmas_f, points.Wm, points.Wc)

    mean_true = A @ mu + b
    cov_true = A @ Sigma @ A.T

    assert np.allclose(mean_ut, mean_true, atol=1e-9), "UT mean != exact linear mean"
    assert np.allclose(cov_ut, cov_true, atol=1e-9), "UT cov != exact linear cov"
    print("Sanity check passed: UT reproduces exact mean/cov for a linear map.")


# ---------------------------------------------------------------------------
# Simulation / demo
# ---------------------------------------------------------------------------

def run_demo():
    _sanity_check_linear()

    rng = np.random.default_rng(0)

    dt = 1.0
    n_steps = 60
    sensor_pos = (0.0, 0.0)

    # True target: starts at (5, 20), moves with constant velocity + small
    # random accelerations (so it's not perfectly linear-model-matched).
    true_x = np.array([5.0, 1.2, 20.0, -0.3])
    q_true = 0.01

    range_std = 5.0          # meters
    bearing_std = np.radians(2.0)
    R = np.diag([range_std ** 2, bearing_std ** 2])

    ukf = build_range_bearing_ukf(dt, q_var=0.05, sensor_pos=sensor_pos)
    ukf.x = np.array([0.0, 0.0, 15.0, 0.0])   # deliberately off initial guess
    ukf.P = np.diag([25.0, 4.0, 25.0, 4.0])
    ukf.R = R

    truth_hist, est_hist, meas_hist = [], [], []

    for _ in range(n_steps):
        # propagate truth with small process noise
        accel = rng.normal(0, np.sqrt(q_true), size=2)
        true_x = fx_constant_velocity(true_x, dt)
        true_x[1] += accel[0]
        true_x[3] += accel[1]
        truth_hist.append(true_x.copy())

        # generate noisy range-bearing measurement
        z_true = hx_range_bearing(true_x, sensor_pos=sensor_pos)
        z = z_true + rng.normal(0, [range_std, bearing_std])
        z[1] = normalize_angle(z[1])
        meas_hist.append(z.copy())

        # filter
        ukf.predict(dt=dt)
        ukf.update(z)
        est_hist.append(ukf.x.copy())

    truth_hist = np.array(truth_hist)
    est_hist = np.array(est_hist)
    meas_hist = np.array(meas_hist)

    pos_err = est_hist[:, [0, 2]] - truth_hist[:, [0, 2]]
    rmse = np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1)))
    print(f"Position RMSE over {n_steps} steps: {rmse:.3f} m")

    # Convert measurements back to Cartesian (relative to sensor) just for plotting
    meas_xy = np.column_stack([
        sensor_pos[0] + meas_hist[:, 0] * np.cos(meas_hist[:, 1]),
        sensor_pos[1] + meas_hist[:, 0] * np.sin(meas_hist[:, 1]),
    ])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(truth_hist[:, 0], truth_hist[:, 2], "g-", label="Truth", linewidth=2)
    ax.plot(est_hist[:, 0], est_hist[:, 2], "b-", label="UKF estimate", linewidth=2)
    ax.scatter(meas_xy[:, 0], meas_xy[:, 1], c="r", s=15, alpha=0.5, label="Measurements")
    ax.scatter(*sensor_pos, c="k", marker="^", s=100, label="Sensor")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"UKF: Cartesian state, range-bearing measurement (RMSE={rmse:.2f} m)")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("/home/claude/ukf_range_bearing_demo.png", dpi=150)
    print("Saved plot to ukf_range_bearing_demo.png")


if __name__ == "__main__":
    run_demo()
