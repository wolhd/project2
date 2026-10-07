import numpy as np


def kalman_smooth_interp(t_meas, z, t_query, q=1e-2, sigma_pos=10.0, sigma_vel=0.1):
    """
    Interpolate ECEF position/velocity with a Kalman filter + RTS smoother.

    Parameters
    ----------
    t_meas : (N,) array   measurement epochs, seconds (any common reference)
    z      : (N,6) array  [x, y, z, vx, vy, vz] in m and m/s (ECEF)
    t_query: (M,) array   epochs to interpolate to, same time reference as t_meas
    q      : float        process-noise acceleration PSD (m^2/s^3); tune this
    sigma_pos, sigma_vel : measurement 1-sigma noise (m, m/s)

    Returns
    -------
    (M,6) array of smoothed [x, y, z, vx, vy, vz] at t_query
    """
    t_meas = np.asarray(t_meas, float)
    t_query = np.asarray(t_query, float)
    z = np.asarray(z, float)

    # sort measurements
    order = np.argsort(t_meas)
    t_meas, z = t_meas[order], z[order]

    # merged timeline (query epochs become nodes without measurements)
    t_all = np.unique(np.concatenate([t_meas, t_query]))
    n = len(t_all)
    meas_idx = np.searchsorted(t_all, t_meas)
    has_meas = np.full(n, -1, int)
    has_meas[meas_idx] = np.arange(len(t_meas))

    I3, Z3 = np.eye(3), np.zeros((3, 3))
    H = np.eye(6)
    R = np.diag([sigma_pos**2] * 3 + [sigma_vel**2] * 3)

    def F_Q(dt):
        F = np.block([[I3, dt * I3], [Z3, I3]])
        Q = q * np.block([[dt**3 / 3 * I3, dt**2 / 2 * I3],
                          [dt**2 / 2 * I3, dt * I3]])
        return F, Q

    # storage
    x_f = np.zeros((n, 6)); P_f = np.zeros((n, 6, 6))   # filtered
    x_p = np.zeros((n, 6)); P_p = np.zeros((n, 6, 6))   # predicted
    Fs = np.zeros((n, 6, 6))                            # F from k-1 -> k

    # initial state: first measurement with a loose prior
    x = z[0].copy()
    P = np.diag([(1e3 * sigma_pos) ** 2] * 3 + [(1e3 * sigma_vel) ** 2] * 3)

    for k in range(n):
        if k == 0:
            xp, Pp = x, P
            Fs[k] = np.eye(6)
        else:
            F, Q = F_Q(t_all[k] - t_all[k - 1])
            xp = F @ x_f[k - 1]
            Pp = F @ P_f[k - 1] @ F.T + Q
            Fs[k] = F
        x_p[k], P_p[k] = xp, Pp

        j = has_meas[k]
        if j >= 0:
            S = H @ Pp @ H.T + R
            K = np.linalg.solve(S.T, (Pp @ H.T).T).T
            x = xp + K @ (z[j] - H @ xp)
            IKH = np.eye(6) - K @ H
            P = IKH @ Pp @ IKH.T + K @ R @ K.T      # Joseph form
        else:
            x, P = xp, Pp
        x_f[k], P_f[k] = x, P

    # RTS backward pass
    x_s = x_f.copy(); P_s = P_f.copy()
    for k in range(n - 2, -1, -1):
        C = P_f[k] @ Fs[k + 1].T @ np.linalg.inv(P_p[k + 1])
        x_s[k] = x_f[k] + C @ (x_s[k + 1] - x_p[k + 1])
        P_s[k] = P_f[k] + C @ (P_s[k + 1] - P_p[k + 1]) @ C.T

    return x_s[np.searchsorted(t_all, t_query)]


# example
if __name__ == "__main__":
    t = np.arange(0, 600, 60.0)
    z = np.column_stack([7e6 + 7000 * t, 1e6 + 500 * t, 2e5 + 100 * t,
                         np.full_like(t, 7000.0), np.full_like(t, 500.0), np.full_like(t, 100.0)])
    tq = np.array([30.0, 90.0, 275.5])
    print(kalman_smooth_interp(t, z, tq))
