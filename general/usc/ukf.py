"""
Generic Unscented Kalman Filter (UKF) core.

This module is deliberately agnostic about:
    - state dimension / meaning
    - measurement dimension / meaning
    - the process model fx(x, dt, *args) -> x
    - the measurement model hx(x, *args) -> z

Angle-valued states/measurements (e.g. bearing) need special averaging and
differencing, so you can plug in custom `mean_fn` / `residual_fn` callables
per state and per measurement. If you don't supply them, plain
weighted-average / subtraction is used, which is correct for ordinary
Cartesian quantities.

To adapt this to a new problem: write a new fx, hx, and (if any component
wraps around, like an angle) a mean_fn/residual_fn for that block. Nothing
else in this file needs to change.
"""

import numpy as np
from scipy.linalg import cholesky


def normalize_angle(angle):
    """Wrap an angle (or array of angles) to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class MerweScaledSigmaPoints:
    """
    Van der Merwe scaled sigma point generator.

    Parameters
    ----------
    n : int
        Dimension of the state.
    alpha : float
        Spread of sigma points around the mean. Small positive value,
        typically 1e-3 to 1.
    beta : float
        Incorporates prior knowledge of the distribution. beta=2 is
        optimal for Gaussian distributions.
    kappa : float, optional
        Secondary scaling parameter, usually 0 or 3-n. Defaults to 3-n.
    """

    def __init__(self, n, alpha=1e-3, beta=2.0, kappa=None):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = 3.0 - n if kappa is None else kappa
        self._compute_weights()

    def num_sigmas(self):
        return 2 * self.n + 1

    def _compute_weights(self):
        n, alpha, beta, kappa = self.n, self.alpha, self.beta, self.kappa
        lambda_ = alpha ** 2 * (n + kappa) - n
        self.lambda_ = lambda_

        c = 0.5 / (n + lambda_)
        Wm = np.full(2 * n + 1, c)
        Wc = np.full(2 * n + 1, c)
        Wm[0] = lambda_ / (n + lambda_)
        Wc[0] = lambda_ / (n + lambda_) + (1 - alpha ** 2 + beta)

        self.Wm = Wm
        self.Wc = Wc

    def sigma_points(self, x, P):
        """Generate 2n+1 sigma points for mean x and covariance P."""
        n = self.n
        x = np.asarray(x, dtype=float).flatten()
        if x.shape[0] != n:
            raise ValueError(f"x has length {x.shape[0]}, expected {n}")

        U = cholesky((n + self.lambda_) * P)  # upper triangular, U.T @ U = P

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for k in range(n):
            sigmas[k + 1] = x + U[k]
            sigmas[n + k + 1] = x - U[k]
        return sigmas


def unscented_transform(sigmas, Wm, Wc, noise_cov=None,
                         mean_fn=None, residual_fn=None):
    """
    Compute mean and covariance of a set of (already-transformed) sigma
    points.

    sigmas : (2n+1, dim) array
    Wm, Wc : (2n+1,) weight arrays
    noise_cov : additive process/measurement noise covariance, or None
    mean_fn(sigmas, Wm) -> mean : optional custom mean (e.g. for angles)
    residual_fn(a, b) -> a - b : optional custom subtraction (e.g. angles)
    """
    n_sigmas, dim = sigmas.shape

    if mean_fn is None:
        x = np.dot(Wm, sigmas)
    else:
        x = mean_fn(sigmas, Wm)

    P = np.zeros((dim, dim))
    if residual_fn is None:
        y = sigmas - x
        P = y.T @ np.diag(Wc) @ y
    else:
        for i in range(n_sigmas):
            y = residual_fn(sigmas[i], x)
            P += Wc[i] * np.outer(y, y)

    if noise_cov is not None:
        P = P + noise_cov

    return x, P


class UnscentedKalmanFilter:
    """
    General discrete-time UKF.

    Parameters
    ----------
    dim_x, dim_z : int
        State and measurement dimensions.
    fx : callable(x, dt, *fx_args) -> x
        Process model. If your process model doesn't need dt, it's fine to
        ignore the argument in your function signature.
    hx : callable(x, *hx_args) -> z
        Measurement model.
    points : MerweScaledSigmaPoints
        Sigma point generator configured with n = dim_x.
    x_mean_fn, z_mean_fn : callable(sigmas, Wm) -> mean, optional
        Custom mean functions for state / measurement, needed if a
        component is an angle.
    residual_x, residual_z : callable(a, b) -> a-b, optional
        Custom residual (subtraction) functions, needed if a component is
        an angle.
    """

    def __init__(self, dim_x, dim_z, fx, hx, points,
                 x_mean_fn=None, z_mean_fn=None,
                 residual_x=None, residual_z=None):
        if points.n != dim_x:
            raise ValueError("points was configured for a different dim_x")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.hx = hx
        self.points_fn = points
        self.Wm = points.Wm
        self.Wc = points.Wc

        self.x_mean_fn = x_mean_fn
        self.z_mean_fn = z_mean_fn
        self.residual_x = residual_x if residual_x is not None else (lambda a, b: a - b)
        self.residual_z = residual_z if residual_z is not None else (lambda a, b: a - b)

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

        n_sig = points.num_sigmas()
        self.sigmas_f = np.zeros((n_sig, dim_x))
        self.sigmas_h = np.zeros((n_sig, dim_z))

        # last update diagnostics
        self.K = None
        self.y = None
        self.Pz = None

    def predict(self, dt=None, fx_args=()):
        """Propagate state and covariance through the process model."""
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i, s in enumerate(sigmas):
            if dt is None:
                self.sigmas_f[i] = self.fx(s, *fx_args)
            else:
                self.sigmas_f[i] = self.fx(s, dt, *fx_args)

        self.x, self.P = unscented_transform(
            self.sigmas_f, self.Wm, self.Wc, self.Q,
            mean_fn=self.x_mean_fn, residual_fn=self.residual_x,
        )

    def update(self, z, hx_args=(), R=None):
        """Incorporate a measurement z. Call predict() first."""
        R = self.R if R is None else R

        for i, s in enumerate(self.sigmas_f):
            self.sigmas_h[i] = self.hx(s, *hx_args)

        zp, Pz = unscented_transform(
            self.sigmas_h, self.Wm, self.Wc, R,
            mean_fn=self.z_mean_fn, residual_fn=self.residual_z,
        )

        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(len(self.sigmas_f)):
            dx = self.residual_x(self.sigmas_f[i], self.x)
            dz = self.residual_z(self.sigmas_h[i], zp)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(Pz)
        y = self.residual_z(np.asarray(z, dtype=float), zp)

        self.x = self.x + K @ y
        self.P = self.P - K @ Pz @ K.T

        self.K, self.y, self.Pz = K, y, Pz
