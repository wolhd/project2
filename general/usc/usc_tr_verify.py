"""
Verify an unscented transform (UT) implementation against Monte Carlo (MC).

Test case: polar -> Cartesian, the classic nonlinear UT benchmark.

    x = [r, theta]  ~ N(mu, Sigma)
    f(x) = [r*cos(theta), r*sin(theta)]

We compare three ways of propagating the input distribution through f:
    1. Unscented transform (using ukf.py's MerweScaledSigmaPoints +
       unscented_transform -- the code under test)
    2. Monte Carlo with a large number of samples (ground truth)
    3. First-order linearization (Jacobian) -- included only as a baseline
       to show how much the UT buys you over naive linearization

Two plots are produced:
    ut_vs_mc_scatter.png
        MC sample cloud in the transformed space, with the UT and MC
        mean/covariance ellipses overlaid (and the linearized ellipse for
        reference).
    ut_vs_mc_convergence.png
        Mean/covariance error (UT and linearization, vs. MC) as the input
        angular spread grows, showing the UT staying close to MC while
        linearization degrades.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ukf import MerweScaledSigmaPoints, unscented_transform


# ---------------------------------------------------------------------------
# The nonlinear function under test, and its Jacobian (for the linearized
# baseline only -- the UT itself never needs a Jacobian).
# ---------------------------------------------------------------------------

def polar_to_cartesian(x):
    r, theta = x
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def polar_to_cartesian_jacobian(x):
    r, theta = x
    return np.array([
        [np.cos(theta), -r * np.sin(theta)],
        [np.sin(theta),  r * np.cos(theta)],
    ])


# ---------------------------------------------------------------------------
# The three propagation methods
# ---------------------------------------------------------------------------

def transform_ut(mu, Sigma, f, alpha=0.1, beta=2.0, kappa=None):
    n = len(mu)
    kappa = 3 - n if kappa is None else kappa
    points = MerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa)
    sigmas = points.sigma_points(mu, Sigma)
    sigmas_f = np.array([f(s) for s in sigmas])
    mean, cov = unscented_transform(sigmas_f, points.Wm, points.Wc)
    return mean, cov, sigmas_f


def transform_mc(mu, Sigma, f, n_samples=200_000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    samples = rng.multivariate_normal(mu, Sigma, size=n_samples)
    y = np.array([f(s) for s in samples])
    mean = y.mean(axis=0)
    cov = np.cov(y.T)
    return mean, cov, y


def transform_linearized(mu, Sigma, f, jacobian):
    mean = f(mu)
    J = jacobian(mu)
    cov = J @ Sigma @ J.T
    return mean, cov


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def draw_covariance_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """Draw an n_std confidence ellipse for a 2D Gaussian (mean, cov)."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.clip(vals, 0, None))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                       fill=False, linewidth=2.5, **kwargs)
    ax.add_patch(ellipse)


def error_metrics(mean, cov, mean_ref, cov_ref):
    mean_err = np.linalg.norm(mean - mean_ref)
    cov_err = np.linalg.norm(cov - cov_ref, ord="fro")
    return mean_err, cov_err


# ---------------------------------------------------------------------------
# Plot 1: scatter + ellipse comparison at one representative setting
# ---------------------------------------------------------------------------

def plot_scatter_comparison(mu, Sigma, rng):
    mean_ut, cov_ut, sigmas_f = transform_ut(mu, Sigma, polar_to_cartesian)
    mean_mc, cov_mc, mc_samples = transform_mc(mu, Sigma, polar_to_cartesian, rng=rng)
    mean_lin, cov_lin = transform_linearized(mu, Sigma, polar_to_cartesian,
                                              polar_to_cartesian_jacobian)

    mean_err_ut, cov_err_ut = error_metrics(mean_ut, cov_ut, mean_mc, cov_mc)
    mean_err_lin, cov_err_lin = error_metrics(mean_lin, cov_lin, mean_mc, cov_mc)

    print("Reference (Monte Carlo, N={:,}):".format(len(mc_samples)))
    print(f"  mean = {mean_mc}")
    print(f"  cov  =\n{cov_mc}\n")
    print("Unscented transform:")
    print(f"  mean = {mean_ut}   |mean err| = {mean_err_ut:.5f}")
    print(f"  cov  =\n{cov_ut}")
    print(f"  ||cov err||_F = {cov_err_ut:.5f}\n")
    print("Linearization (Jacobian):")
    print(f"  mean = {mean_lin}   |mean err| = {mean_err_lin:.5f}")
    print(f"  cov  =\n{cov_lin}")
    print(f"  ||cov err||_F = {cov_err_lin:.5f}\n")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(mc_samples[:, 0], mc_samples[:, 1], s=2, alpha=0.15,
               color="gray", label=f"MC samples (N={len(mc_samples):,})", rasterized=True)
    ax.scatter(sigmas_f[:, 0], sigmas_f[:, 1], color="black", marker="x",
               s=70, zorder=5, label="Transformed sigma points")

    draw_covariance_ellipse(ax, mean_mc, cov_mc, n_std=2, edgecolor="tab:gray",
                             linestyle="-", label="MC mean/cov (2σ)")
    draw_covariance_ellipse(ax, mean_ut, cov_ut, n_std=2, edgecolor="tab:blue",
                             linestyle="--", label="UT mean/cov (2σ)")
    draw_covariance_ellipse(ax, mean_lin, cov_lin, n_std=2, edgecolor="tab:red",
                             linestyle=":", label="Linearized mean/cov (2σ)")

    ax.scatter(*mean_mc, color="tab:gray", marker="o", s=90, zorder=6)
    ax.scatter(*mean_ut, color="tab:blue", marker="o", s=90, zorder=6)
    ax.scatter(*mean_lin, color="tab:red", marker="o", s=90, zorder=6)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "Polar\u2192Cartesian UT: sanity check against Monte Carlo\n"
        f"mean err (UT/lin) = {mean_err_ut:.4f} / {mean_err_lin:.4f}   "
        f"cov err (UT/lin) = {cov_err_ut:.4f} / {cov_err_lin:.4f}"
    )
    ax.legend(loc="best", fontsize=9)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("/home/claude/ut_vs_mc_scatter.png", dpi=150)
    print("Saved plot: ut_vs_mc_scatter.png")


# ---------------------------------------------------------------------------
# Plot 2: error vs. degree of nonlinearity (angular spread)
# ---------------------------------------------------------------------------

def plot_convergence_vs_nonlinearity(r_mean, theta_mean, rng):
    theta_stds_deg = np.array([1, 3, 5, 8, 12, 18, 25, 35, 45, 60])
    r_std = 1.0

    mean_err_ut, cov_err_ut = [], []
    mean_err_lin, cov_err_lin = [], []

    for std_deg in theta_stds_deg:
        theta_std = np.radians(std_deg)
        mu = np.array([r_mean, theta_mean])
        Sigma = np.diag([r_std ** 2, theta_std ** 2])

        mean_mc, cov_mc, _ = transform_mc(mu, Sigma, polar_to_cartesian,
                                           n_samples=200_000, rng=rng)
        mean_ut, cov_ut, _ = transform_ut(mu, Sigma, polar_to_cartesian)
        mean_lin, cov_lin = transform_linearized(mu, Sigma, polar_to_cartesian,
                                                  polar_to_cartesian_jacobian)

        me_ut, ce_ut = error_metrics(mean_ut, cov_ut, mean_mc, cov_mc)
        me_lin, ce_lin = error_metrics(mean_lin, cov_lin, mean_mc, cov_mc)

        mean_err_ut.append(me_ut)
        cov_err_ut.append(ce_ut)
        mean_err_lin.append(me_lin)
        cov_err_lin.append(ce_lin)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(theta_stds_deg, mean_err_ut, "o-", color="tab:blue", label="UT")
    axes[0].plot(theta_stds_deg, mean_err_lin, "s--", color="tab:red", label="Linearized")
    axes[0].set_xlabel("Bearing std. dev. (degrees)")
    axes[0].set_ylabel("|| mean error || vs. Monte Carlo")
    axes[0].set_title("Mean error vs. input nonlinearity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(theta_stds_deg, cov_err_ut, "o-", color="tab:blue", label="UT")
    axes[1].plot(theta_stds_deg, cov_err_lin, "s--", color="tab:red", label="Linearized")
    axes[1].set_xlabel("Bearing std. dev. (degrees)")
    axes[1].set_ylabel("|| cov error ||_F vs. Monte Carlo")
    axes[1].set_title("Covariance error vs. input nonlinearity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "More spread-out input (more nonlinear regime) \u2192 UT should track\n"
        "Monte Carlo much better than a first-order linearization",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("/home/claude/ut_vs_mc_convergence.png", dpi=150)
    print("Saved plot: ut_vs_mc_convergence.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # A moderately nonlinear case: decent range uncertainty and a nontrivial
    # bearing spread (~15 deg std), so linearization is visibly worse than UT
    # but not so extreme that either falls apart completely.
    mu = np.array([10.0, np.radians(30.0)])          # r=10, theta=30 deg
    Sigma = np.diag([1.0 ** 2, np.radians(15.0) ** 2])

    plot_scatter_comparison(mu, Sigma, rng)
    plot_convergence_vs_nonlinearity(r_mean=10.0, theta_mean=np.radians(30.0), rng=rng)
