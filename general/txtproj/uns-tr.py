"""
Unscented Transform (UT) for an n = 3 dimensional random vector,
propagated through a generic nonlinear function f(x).
"""

import numpy as np


def unscented_transform(x_bar, P, f, alpha=1e-3, beta=2.0, kappa=0.0):
    """
    Compute the mean and covariance of y = f(x) via the Unscented Transform.

    Parameters
    ----------
    x_bar : (n,) array_like
        Mean of the input random vector x.
    P : (n, n) array_like
        Covariance of the input random vector x.
    f : callable
        Nonlinear function, f(x) -> (m,) array. Must accept a single
        n-dimensional vector and return an m-dimensional vector.
    alpha : float
        Spread of sigma points around x_bar (small positive, e.g. 1e-3).
    beta : float
        Incorporates prior knowledge of the distribution (2.0 is optimal
        for Gaussian inputs).
    kappa : float
        Secondary scaling parameter (often 0).

    Returns
    -------
    y_bar : (m,) ndarray
        Estimated mean of y = f(x).
    P_y : (m, m) ndarray
        Estimated covariance of y.
    P_xy : (n, m) ndarray
        Cross-covariance between x and y (useful for Kalman gain in UKF).
    sigma_X : (2n+1, n) ndarray
        The sigma points in x-space (for inspection/debugging).
    sigma_Y : (2n+1, m) ndarray
        The propagated sigma points in y-space.
    """
    x_bar = np.atleast_1d(np.asarray(x_bar, dtype=float))
    P = np.atleast_2d(np.asarray(P, dtype=float))
    n = x_bar.shape[0]

    # --- Scaling parameter ---
    lam = alpha**2 * (n + kappa) - n

    # --- Weights ---
    Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    Wc = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    # --- Sigma points ---
    # Matrix square root via Cholesky: S @ S.T = (n + lam) * P
    S = np.linalg.cholesky((n + lam) * P)

    sigma_X = np.zeros((2 * n + 1, n))
    sigma_X[0] = x_bar
    for i in range(n):
        col = S[:, i]
        sigma_X[i + 1] = x_bar + col
        sigma_X[n + i + 1] = x_bar - col

    # --- Propagate through the nonlinear function ---
    sigma_Y = np.array([f(pt) for pt in sigma_X])
    m = sigma_Y.shape[1] if sigma_Y.ndim > 1 else 1
    sigma_Y = sigma_Y.reshape(2 * n + 1, m)

    # --- Recombine statistics ---
    y_bar = np.sum(Wm[:, None] * sigma_Y, axis=0)

    dy = sigma_Y - y_bar
    P_y = np.zeros((m, m))
    for i in range(2 * n + 1):
        P_y += Wc[i] * np.outer(dy[i], dy[i])

    dx = sigma_X - x_bar
    P_xy = np.zeros((n, m))
    for i in range(2 * n + 1):
        P_xy += Wc[i] * np.outer(dx[i], dy[i])

    return y_bar, P_y, P_xy, sigma_X, sigma_Y


if __name__ == "__main__":
    # --- Example: 3D state, generic nonlinear function ---
    x_bar = np.array([1.0, 2.0, 0.5])
    P = np.diag([0.1, 0.2, 0.05])  # input covariance

    def f(x):
        # Example nonlinear map R^3 -> R^3
        x1, x2, x3 = x
        return np.array([
            x1 * np.cos(x3),
            x2 * np.sin(x3),
            x1 + x2 + x3**2
        ])

    # Note: alpha is typically small (e.g. 1e-3) for real filtering, but
    # that makes (n+lambda) -> 0 for n=3, kappa=0, causing huge weights
    # that are hard to sanity-check by eye. alpha=1, kappa=0 keeps the
    # weights well-behaved for this demo; both are mathematically valid.
    y_bar, P_y, P_xy, sigma_X, sigma_Y = unscented_transform(
        x_bar, P, f, alpha=1.0, beta=2.0, kappa=0.0
    )

    np.set_printoptions(precision=4, suppress=True)
    print("Sigma points (x-space):")
    print(sigma_X)
    print("\nPropagated sigma points (y-space):")
    print(sigma_Y)
    print("\nEstimated mean y_bar:")
    print(y_bar)
    print("\nEstimated covariance P_y:")
    print(P_y)
    print("\nCross-covariance P_xy:")
    print(P_xy)
