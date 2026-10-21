import numpy as np
from dataclasses import dataclass
from scipy.stats import chi2

# assumes bistatic_measurement() and _wrap_pi() from the earlier snippets are in scope


@dataclass
class BistaticResidual:
    residual: np.ndarray    # [d_range (m), d_rate (m/s), d_az (rad)] = observed - model, az wrapped to [-pi, pi)
    cov: np.ndarray         # 3x3 residual (innovation) covariance S = R_obs + R_model
    z_model: np.ndarray     # model measurement [range, rate, az in [0, 2pi)]
    R_model: np.ndarray     # 3x3 model measurement covariance (from state uncertainty)
    R_obs: np.ndarray       # 3x3 observation covariance
    nis: float              # normalized innovation squared, r^T S^-1 r
    p_value: float          # P(chi2_3 >= nis)
    sigma: np.ndarray       # per-component r_i / sqrt(S_ii)


def bistatic_residual(z_obs, R_obs, target, tx, rx,
                      P_target=None, P_tx=None, P_rx=None, P_full=None,
                      cross_cov=None):
    """
    z_obs     : [excess_range (m), range_rate (m/s), azimuth (rad)]
    R_obs     : observation covariance, 3x3 or length-3 variances [var_r, var_rr, var_az]
    target, tx, rx, P_* / P_full : same as bistatic_measurement()
    cross_cov : optional 3x3 cross-covariance between observation error and model error
                (default: independent)

    residual = z_obs - z_model (azimuth wrapped)
    S        = R_obs + R_model  (- C - C^T if cross_cov C is given)
    """
    z_obs = np.asarray(z_obs, float)
    R_obs = np.asarray(R_obs, float)
    if R_obs.ndim == 1:
        R_obs = np.diag(R_obs)

    m = bistatic_measurement(target, tx, rx, P_target, P_tx, P_rx, P_full)
    z_model = np.array([m.excess_range, m.range_rate, m.azimuth])

    r = z_obs - z_model
    r[2] = _wrap_pi(r[2])

    S = R_obs + m.cov
    if cross_cov is not None:
        C = np.asarray(cross_cov, float)
        S = S - C - C.T

    nis = float(r @ np.linalg.solve(S, r))
    return BistaticResidual(
        residual=r, cov=S, z_model=z_model, R_model=m.cov, R_obs=R_obs,
        nis=nis, p_value=float(chi2.sf(nis, df=3)),
        sigma=r / np.sqrt(np.diag(S)),
    )


# ---- example ----
if __name__ == "__main__":
    tgt = np.array([1_100_000, -4_300_000, 4_400_000, 200, -50, 10.])
    tx  = np.array([1_080_000, -4_320_000, 4_400_000, 5, 3, 0.])
    rx  = np.array([1_130_000, -4_290_000, 4_390_000, 100, 0, 0.])
    P_t  = np.diag([1500**2]*3 + [30**2]*3)
    P_tx = np.diag([2000**2]*3 + [10**2]*3)
    P_rx = np.diag([300**2]*3 + [2**2]*3)
    R_obs = np.diag([50.**2, 1.0**2, np.radians(0.5)**2])

    z_obs = [2290.0 - 317.0, -50.7 + 7.2, np.radians(342.07 + 1.69)]
    res = bistatic_residual(z_obs, R_obs, tgt, tx, rx, P_t, P_tx, P_rx)
    print(res.residual, np.sqrt(np.diag(res.cov)), res.nis, res.p_value)
