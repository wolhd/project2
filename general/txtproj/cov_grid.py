import numpy as np
import pymap3d as pm


def covariance_ellipse_grid(lat_deg, lon_deg, alt_m, cov_enu,
                            confidence=0.95, spacing_m=None, max_points=1_000_000):
    """
    Parameters
    ----------
    lat_deg, lon_deg : float   center position in degrees
    alt_m            : float   ellipsoidal altitude in meters
    cov_enu          : (3,3) or (2,2) array, ENU covariance in m^2
    confidence       : probability mass enclosed by the 2D ellipse (e.g. 0.95)
    spacing_m        : grid spacing in meters (default: smallest semi-axis / 10)

    Returns
    -------
    (N, 2) array of [lat_deg, lon_deg] for grid points inside the ellipse.
    The grid is aligned to East/North and centered on the input position.
    """
    c_en = np.asarray(cov_enu, dtype=float)[:2, :2]   # East-North block
    c_en = 0.5 * (c_en + c_en.T)                      # enforce symmetry

    # Chi-square scale for 2 DOF: P = 1 - exp(-k^2/2)
    k2 = -2.0 * np.log(1.0 - confidence)

    # Principal axes of the ellipse
    evals, evecs = np.linalg.eigh(c_en)               # ascending eigenvalues
    evals = np.clip(evals, 1e-12, None)
    semi = np.sqrt(k2 * evals)                        # [minor, major] semi-axes (m)

    if spacing_m is None:
        spacing_m = semi[0] / 10.0

    # E/N bounding box of the rotated ellipse
    ne = int(np.floor(np.sqrt(k2 * c_en[0, 0]) / spacing_m))
    nn = int(np.floor(np.sqrt(k2 * c_en[1, 1]) / spacing_m))
    if (2 * ne + 1) * (2 * nn + 1) > max_points:
        raise ValueError("Grid too large; increase spacing_m.")

    e_vals = np.arange(-ne, ne + 1) * spacing_m
    n_vals = np.arange(-nn, nn + 1) * spacing_m
    ee, nn_grid = np.meshgrid(e_vals, n_vals)
    pts = np.column_stack([ee.ravel(), nn_grid.ravel()])

    # Inside test in the principal-axis frame: (u/a)^2 + (v/b)^2 <= 1
    uv = pts @ evecs
    inside = (uv[:, 0] / semi[0]) ** 2 + (uv[:, 1] / semi[1]) ** 2 <= 1.0
    pts = pts[inside]

    # ENU (up = 0) -> geodetic, WGS84 by default
    lat, lon, _ = pm.enu2geodetic(
        pts[:, 0], pts[:, 1], np.zeros(len(pts)),
        lat_deg, lon_deg, alt_m, deg=True,
    )
    return np.column_stack([lat, lon])


if __name__ == "__main__":
    cov = np.array([
        [25.0, 8.0, 0.0],
        [ 8.0, 9.0, 0.0],
        [ 0.0, 0.0, 4.0],
    ])
    grid = covariance_ellipse_grid(42.6334, -71.3162, 30.0, cov,
                                   confidence=0.95, spacing_m=1.0)
    print(f"{len(grid)} points")
    print(grid[:5])
