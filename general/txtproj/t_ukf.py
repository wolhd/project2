"""
Stage 4: Recursive fusion (UKF) for unknown bistatic transmitter geolocation.

State:      x = [X, Y, Z, Vx, Vy, Vz]   (Tx position/velocity, any consistent
                                          Cartesian frame - local ENU or ECEF)
Process:    constant-velocity (CV)
Measurements per aircraft i, per epoch (subset may be missing):
    - baseline-removed bistatic range     R_bi
    - bistatic range-rate                 Rdot_bi
    - azimuth at Tx from Tx->Rx to Tx->Target, clockwise-positive, viewed
      along a local "up" unit vector n_hat

The measurement vector at each epoch is built dynamically from whichever
aircraft have a valid detection that epoch (asynchronous / variable length).
"""

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


# --------------------------------------------------------------------------
# Geometry / measurement model
# --------------------------------------------------------------------------

def local_up(pos, mode="flat", flat_up=np.array([0.0, 0.0, 1.0])):
    """Local 'up' unit vector n_hat at a given position, used for the
    clockwise-positive azimuth sign convention.

    mode="flat": fixed vertical (valid in a local ENU/NED tangent frame).
    mode="geocentric": pos / |pos| (valid in ECEF, spherical-Earth approx).
    """
    if mode == "flat":
        return flat_up
    elif mode == "geocentric":
        n = np.linalg.norm(pos)
        return pos / n if n > 0 else flat_up
    else:
        raise ValueError("mode must be 'flat' or 'geocentric'")


def bistatic_measurement(tx_pos, tx_vel, target_pos, target_vel,
                          rx_pos, rx_vel, up_mode="flat"):
    """h_i(x): predicted (range, range_rate, azimuth) from a hypothesized
    Tx state to a single known aircraft, given the known receiver state.
    """
    r1 = tx_pos - target_pos          # Tx -> Target
    r2 = target_pos - rx_pos          # Target -> Rx
    rL = tx_pos - rx_pos              # Tx -> Rx

    n1, n2, nL = np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(rL)

    # --- range (baseline removed) ---
    R_bi = n1 + n2 - nL

    # --- range-rate ---
    Rdot_bi = (np.dot(r1, tx_vel - target_vel) / n1
               + np.dot(r2, target_vel - rx_vel) / n2
               - np.dot(rL, tx_vel - rx_vel) / nL)

    # --- azimuth: angle at Tx from (Tx->Rx) to (Tx->Target), CW positive ---
    u = rx_pos - tx_pos                # Tx -> Rx
    v = target_pos - tx_pos            # Tx -> Target
    n_hat = local_up(tx_pos, mode=up_mode)
    az = np.arctan2(-np.dot(n_hat, np.cross(u, v)), np.dot(u, v))

    return np.array([R_bi, Rdot_bi, az])


# --------------------------------------------------------------------------
# Process model (constant velocity)
# --------------------------------------------------------------------------

def fx_cv(x, dt):
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F @ x


def Q_cv(dt, sigma_a):
    """Discretized white-noise-acceleration process noise, per axis, then
    assembled into the 6x6 block form for [pos, vel]."""
    q = sigma_a ** 2
    Qb = q * np.array([[dt**3 / 3, dt**2 / 2],
                        [dt**2 / 2, dt]])
    Q = np.zeros((6, 6))
    for i in range(3):          # X, Y, Z occupy indices (i, i+3)
        idx = [i, i + 3]
        Q[np.ix_(idx, idx)] = Qb
    return Q


# --------------------------------------------------------------------------
# Epoch measurement assembly (handles variable / asynchronous aircraft set)
# --------------------------------------------------------------------------

class EpochBuilder:
    """
    Builds z_k, R_k, and a matching hx_k(x) for one epoch from a list of
    per-aircraft detections. Each detection dict must supply the aircraft's
    known kinematics (already time-aligned/interpolated -- Stage 1 output)
    and the measured values with their 1-sigma noise.

    detection = {
        'target_pos': np.array([X,Y,Z]),
        'target_vel': np.array([Vx,Vy,Vz]),
        'range': float, 'sigma_range': float,
        'range_rate': float, 'sigma_range_rate': float,
        'azimuth': float, 'sigma_azimuth': float,   # radians
        # any subset of ('range','range_rate','azimuth') may be omitted
    }
    """

    MEAS_ORDER = ("range", "range_rate", "azimuth")
    SIGMA_KEY = {"range": "sigma_range",
                 "range_rate": "sigma_range_rate",
                 "azimuth": "sigma_azimuth"}
    ANGLE_FLAGS = (False, False, True)  # which components wrap at +-pi

    def __init__(self, rx_pos, rx_vel, up_mode="flat"):
        self.rx_pos = rx_pos
        self.rx_vel = rx_vel
        self.up_mode = up_mode

    def build(self, detections):
        """detections: list of per-aircraft detection dicts (see docstring).
        Returns (z, R_diag, hx_fn, angle_mask) or (None, ...) if empty.
        """
        z_list, sig_list, mask_list = [], [], []
        specs = []  # (target_pos, target_vel, which_components)

        for det in detections:
            present = [m for m in self.MEAS_ORDER if m in det]
            if not present:
                continue
            specs.append((det["target_pos"], det["target_vel"], present))
            for m in present:
                z_list.append(det[m])
                sig_list.append(det[self.SIGMA_KEY[m]])
                mask_list.append(self.ANGLE_FLAGS[self.MEAS_ORDER.index(m)])

        if not z_list:
            return None, None, None, None

        z = np.array(z_list, dtype=float)
        R_diag = np.array(sig_list, dtype=float) ** 2
        angle_mask = np.array(mask_list, dtype=bool)

        def hx(x):
            tx_pos, tx_vel = x[0:3], x[3:6]
            out = []
            for target_pos, target_vel, present in specs:
                full = bistatic_measurement(tx_pos, tx_vel, target_pos,
                                             target_vel, self.rx_pos,
                                             self.rx_vel, self.up_mode)
                for m in present:
                    out.append(full[self.MEAS_ORDER.index(m)])
            return np.array(out)

        return z, np.diag(R_diag), hx, angle_mask


def wrapped_residual(angle_mask):
    """Residual function for filterpy's UKF update: ordinary subtraction,
    except angle components get wrapped to (-pi, pi]."""
    def residual(a, b):
        y = a - b
        y[angle_mask] = (y[angle_mask] + np.pi) % (2 * np.pi) - np.pi
        return y
    return residual


def angle_aware_mean(angle_mask):
    """Sigma-point mean function for filterpy: ordinary weighted sum for
    non-angular components, circular (sin/cos) weighted mean for angular
    ones. Needed because a plain weighted average of angles is wrong near
    the +-pi wrap (and near the atan2 branch cut in general)."""
    def mean_fn(sigmas, Wm):
        x = np.dot(Wm, sigmas)
        if np.any(angle_mask):
            s = np.dot(Wm, np.sin(sigmas[:, angle_mask]))
            c = np.dot(Wm, np.cos(sigmas[:, angle_mask]))
            x[angle_mask] = np.arctan2(s, c)
        return x
    return mean_fn


# --------------------------------------------------------------------------
# UKF wrapper for this problem
# --------------------------------------------------------------------------

class TxUKF:
    def __init__(self, x0, P0, sigma_a=1.0, rx_state_fn=None, up_mode="flat"):
        """
        x0, P0      : initial state (6,) and covariance (6,6) -- from the
                      Stage-2 closed-form seed (validated across several
                      consistent epochs; inflate P0 beyond the raw linearized
                      covariance per the seeding discussion).
        sigma_a     : process-noise tuning knob (accel spectral density),
                      same units as position/time^2.
        rx_state_fn : callable(t) -> (rx_pos, rx_vel), known receiver track.
        up_mode     : 'flat' (local ENU) or 'geocentric' (ECEF).
        """
        points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2.0, kappa=0.0)
        self.ukf = UKF(dim_x=6, dim_z=1, dt=1.0, fx=fx_cv, hx=lambda x: x,
                        points=points)
        self.ukf.x = np.array(x0, dtype=float)
        self.ukf.P = np.array(P0, dtype=float)
        self.sigma_a = sigma_a
        self.rx_state_fn = rx_state_fn
        self.up_mode = up_mode
        self.last_t = None
        self.nis_history = []   # (t, NIS, dof)

    def step(self, t, detections):
        """Advance filter to time t with a predict, and (if detections is
        non-empty) a measurement update built from the active aircraft.
        detections: list of per-aircraft detection dicts (target kinematics
        + measured range/range_rate/azimuth + their sigmas), or [].
        """
        dt = 0.0 if self.last_t is None else (t - self.last_t)
        self.last_t = t

        # --- predict ---
        self.ukf.Q = Q_cv(max(dt, 1e-6), self.sigma_a)
        self.ukf.predict(dt=dt, fx=fx_cv)

        if not detections:
            return self.ukf.x.copy(), self.ukf.P.copy()

        rx_pos, rx_vel = self.rx_state_fn(t)
        builder = EpochBuilder(rx_pos, rx_vel, up_mode=self.up_mode)
        z, R, hx, angle_mask = builder.build(detections)
        if z is None:
            return self.ukf.x.copy(), self.ukf.P.copy()

        residual_fn = wrapped_residual(angle_mask)
        mean_fn = angle_aware_mean(angle_mask)

        # filterpy reads these as attributes inside update(), not as update()
        # kwargs -- must set them per call since angle_mask/dim changes epoch
        # to epoch with the active aircraft set.
        self.ukf.residual_z = residual_fn
        self.ukf.z_mean = mean_fn
        self.ukf._dim_z = len(z)

        # --- update (variable dimension: rebuild sigma points/hx each call) ---
        self.ukf.update(z, R=R, hx=hx)

        # --- NIS bookkeeping (health monitoring, see Stage 6) ---
        zhat = hx(self.ukf.x_prior)
        innov = residual_fn(z, zhat)
        S = self.ukf.S  # innovation covariance from the last update
        try:
            nis = float(innov @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError:
            nis = np.nan
        self.nis_history.append((t, nis, len(z)))

        return self.ukf.x.copy(), self.ukf.P.copy()

  #-----
  #demo
  """
Synthetic end-to-end test of tx_ukf.py:
  - simulate a moving Tx, 3 moving aircraft, and a moving receiver
  - generate noisy bistatic (range, range_rate, azimuth) detections
  - seed the UKF with a deliberately imperfect guess
  - run the filter and report position error + NIS consistency

Everything is in a flat local frame (up_mode="flat"), consistent with the
"local ENU tangent plane" simplification discussed for a scenario that
doesn't span a large enough area for Earth curvature to matter.
"""

import numpy as np
from tx_ukf import TxUKF, bistatic_measurement

rng = np.random.default_rng(42)

# --------------------------------------------------------------------------
# Truth trajectories (simple constant-velocity, km and km/s)
# --------------------------------------------------------------------------

dt = 1.0          # seconds between epochs
n_steps = 200

tx0 = np.array([40.0, 60.0, 8.0])
tx_vel_true = np.array([-0.05, 0.02, 0.0])          # slow-moving transmitter

rx0 = np.array([0.0, 0.0, 0.5])
rx_vel = np.array([0.10, 0.0, 0.0])                  # known receiver track

aircraft0 = {
    1: (np.array([20.0, -10.0, 10.0]), np.array([0.15, 0.05, 0.0])),
    2: (np.array([-15.0, 30.0, 9.0]),  np.array([0.05, -0.12, 0.0])),
    3: (np.array([50.0, 5.0, 11.0]),   np.array([-0.02, 0.18, 0.0])),
}

sigma_range = 0.05        # km
sigma_rr = 0.01            # km/s
sigma_az = np.deg2rad(0.5)  # rad


def truth_state(t):
    tx_pos = tx0 + tx_vel_true * t
    return np.concatenate([tx_pos, tx_vel_true])


def rx_state_fn(t):
    return rx0 + rx_vel * t, rx_vel.copy()


def aircraft_state(aid, t):
    p0, v = aircraft0[aid]
    return p0 + v * t, v


# --------------------------------------------------------------------------
# Generate noisy detections. Not every aircraft reports every epoch -- this
# exercises the variable-dimension / asynchronous measurement handling.
# --------------------------------------------------------------------------

def make_detections(t):
    tx_pos, tx_vel = truth_state(t)[0:3], truth_state(t)[3:6]
    rx_pos, rx_vel_now = rx_state_fn(t)
    dets = []
    for aid in (1, 2, 3):
        if rng.random() < 0.2:          # ~20% miss rate per aircraft/epoch
            continue
        tp, tv = aircraft_state(aid, t)
        truth = bistatic_measurement(tx_pos, tx_vel, tp, tv, rx_pos,
                                      rx_vel_now, up_mode="flat")
        det = {
            "target_pos": tp,
            "target_vel": tv,
            "range": truth[0] + rng.normal(0, sigma_range),
            "sigma_range": sigma_range,
            "range_rate": truth[1] + rng.normal(0, sigma_rr),
            "sigma_range_rate": sigma_rr,
            "azimuth": truth[2] + rng.normal(0, sigma_az),
            "sigma_azimuth": sigma_az,
        }
        dets.append(det)
    return dets


# --------------------------------------------------------------------------
# Stage-2-style seed: deliberately offset from truth to test convergence
# (see the "bad seed" discussion -- inflated P0, not a point-trust seed)
# --------------------------------------------------------------------------

x0 = truth_state(0.0) + np.array([5.0, -4.0, 2.0, 0.03, -0.02, 0.0])
P0 = np.diag([25.0, 25.0, 9.0, 0.05, 0.05, 0.01])   # deliberately generous

ukf = TxUKF(x0=x0, P0=P0, sigma_a=1e-4, rx_state_fn=rx_state_fn,
            up_mode="flat")

pos_err_hist = []
nees_hist = []

for k in range(n_steps):
    t = k * dt
    dets = make_detections(t)
    x_est, P_est = ukf.step(t, dets)

    truth = truth_state(t)
    err = x_est - truth
    pos_err_hist.append(np.linalg.norm(err[0:3]))
    nees = float(err @ np.linalg.solve(P_est, err))
    nees_hist.append(nees)

pos_err_hist = np.array(pos_err_hist)
nees_hist = np.array(nees_hist)
nis_hist = np.array([n for (_, n, _) in ukf.nis_history if not np.isnan(n)])

print(f"Initial seed position error: {np.linalg.norm(x0[0:3] - truth_state(0)[0:3]):.2f} km")
print(f"Final position error:        {pos_err_hist[-1]:.4f} km")
print(f"Mean position error (last 50 epochs): {pos_err_hist[-50:].mean():.4f} km")
print()
print(f"Mean NEES (state dim=6, expect ~6):  {nees_hist.mean():.2f}")
print(f"Mean NEES (last 50 epochs):          {nees_hist[-50:].mean():.2f}")
print()
print(f"Mean NIS (varies with #active aircraft this epoch): {nis_hist.mean():.2f}")
print(f"Number of updates with a measurement: {len(nis_hist)} / {n_steps}")
