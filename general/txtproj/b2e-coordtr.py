Using authoritative geospatial libraries like pyproj or pymap3d replaces manual WGS84 math with verified industry-standard transformations.
Here is the complete Python solution utilizing pyproj for geodetic-to-ECEF transformations and coordinate system rotations, along with scipy to solve the geometric intersections.
## Code Implementation

import numpy as npfrom pyproj import CRS, Transformerfrom scipy.optimize import fsolve
def bistatic_alt_to_ecef(r_bistatic, r_dot_bistatic, 
                          tx_pos_ecef, tx_vel_ecef, 
                          rx_lat, rx_lon, rx_alt, rx_vel_ecef,
                          az_deg, target_alt):
    """
    Transforms bistatic tracking measurements into an ECEF State Vector 
    using standard pyproj transformations.
    """
    # 1. Initialize Standard Geospatial Transformers
    # EPSG:4978 is the official standard EPSG code for WGS84 ECEF 3D Coordinates
    ecef_crs = CRS.from_epsg(4978)
    geodetic_crs = CRS.from_epsg(4979) # WGS84 3D Geodetic (Lat, Lon, Alt)
    
    geo_to_ecef = Transformer.from_crs(geodetic_crs, ecef_crs, always_xy=True)
    ecef_to_geo = Transformer.from_crs(ecef_crs, geodetic_crs, always_xy=True)
    
    # 2. Compute Receiver ECEF Position via pyproj
    # Note: pyproj always_xy=True expects (Longitude, Latitude, Altitude)
    rx_x, rx_y, rx_z = geo_to_ecef.transform(rx_lon, rx_lat, rx_alt)
    rx_pos_ecef = np.array([rx_x, rx_y, rx_z])
    
    # Calculate bistatic baseline constants
    L_baseline = np.linalg.norm(rx_pos_ecef - tx_pos_ecef)
    r_sum = r_bistatic + L_baseline
    
    # 3. Create Local Topocentric Frame (ENU) Rotation Matrix using pyproj
    # We transform local ENU basis vectors (1,0,0), (0,1,0), (0,0,1) at Rx to ECEF
    # to perfectly handle the Earth's curvature without manual trig.
    # ENU origins in Geodetic
    # We construct a transformation from a local topocentric coordinate system
    # custom to our specific receiver lat/lon.
    local_enu_crs = CRS(f"+proj=topocentric +lat_0={rx_lat} +lon_0={rx_lon} +h_0={rx_alt} +ellps=WGS84")
    enu_to_ecef_trans = Transformer.from_crs(local_enu_crs, ecef_crs, always_xy=True)
    ecef_to_enu_trans = Transformer.from_crs(ecef_crs, local_enu_crs, always_xy=True)

    # 4. Core Geometric Equation Solver Block
    def target_equations(vars):
        p_tgt = vars
        
        # Constraint 1: Bistatic Ellipsoid
        r_rx = np.linalg.norm(p_tgt - rx_pos_ecef)
        r_tx = np.linalg.norm(p_tgt - tx_pos_ecef)
        eq1 = (r_rx + r_tx) - r_sum
        
        # Constraint 2: Target Altitude (Via authoritative pyproj ellipsoid evaluation)
        tgt_lon, tgt_lat, tgt_alt = ecef_to_geo.transform(*p_tgt)
        eq2 = tgt_alt - target_alt
        
        # Constraint 3: Azimuth Line of Sight
        tgt_e, tgt_n, tgt_u = ecef_to_enu_trans.transform(*p_tgt)
        az_rad = np.radians(az_deg)
        eq3 = tgt_e * np.cos(az_rad) - tgt_n * np.sin(az_rad)
        
        return [eq1, eq2, eq3]
    
    # 5. Numerical Guess along the local horizon
    az_rad = np.radians(az_deg)
    # Estimate target 20km out along azimuth at target altitude relative to Rx ENU
    guess_e = 20000 * np.sin(az_rad)
    guess_n = 20000 * np.cos(az_rad)
    guess_u = target_alt - rx_alt
    
    initial_ecef_guess = np.array(enu_to_ecef_trans.transform(guess_e, guess_n, guess_u))
    
    # Solve exact target ECEF coordinates
    target_pos_ecef, info, ier, mesg = fsolve(target_equations, initial_ecef_guess, full_output=True)
    if ier != 1:
        raise ValueError(f"Geospatial geometry solver failed to converge: {mesg}")
        
    # 6. Compute Target Velocity Vector
    vec_tx_target = target_pos_ecef - tx_pos_ecef
    vec_rx_target = target_pos_ecef - rx_pos_ecef
    u_tx = vec_tx_target / np.linalg.norm(vec_tx_target)
    u_rx = vec_rx_target / np.linalg.norm(vec_rx_target)
    
    sum_u = u_tx + u_rx
    rhs = r_dot_bistatic + np.dot(u_tx, tx_vel_ecef) + np.dot(u_rx, rx_vel_ecef)
    v_mag_projected = rhs / np.dot(sum_u, sum_u)
    target_vel_ecef = v_mag_projected * sum_u
    
    return target_pos_ecef, target_vel_ecef
# --- Verification & Execution ---if __name__ == "__main__":
    # Sample Operational Inputs
    tx_pos = np.array([1115000.0, -4845000.0, 3980000.0])  
    tx_vel = np.array([-3000.0, 4000.0, 5000.0])
    
    rx_lat, rx_lon, rx_alt = 34.05, -118.24, 100.0        
    rx_vel = np.array([0.0, 0.0, 0.0])
    
    azimuth_deg = 45.0             
    target_altitude_m = 12000.0    # Target cruising at 12 km
    
    bistatic_range_m = 250000.0    
    bistatic_rate_ms = -150.0      
    
    pos_ecef, vel_ecef = bistatic_alt_to_ecef(
        bistatic_range_m, bistatic_rate_ms,
        tx_pos, tx_vel,
        rx_lat, rx_lon, rx_alt, rx_vel,
        azimuth_deg, target_altitude_m
    )
    
    print(f"Target Position [X, Y, Z] ECEF (m):\n{pos_ecef}")
    print(f"Target Velocity [Vx, Vy, Vz] ECEF (m/s):\n{vel_ecef}")

## Why this approach is preferred over manual matrix math:

* True Topocentric Reference Frame (+proj=topocentric): Instead of relying on spherical or simplified rotation matrix code, pyproj configures an exact spatial transformation relative to the local WGS84 tangential horizon.
* Exact Ellipsoidal Altitudes: In step 4, ecef_to_geo.transform precisely evaluates the target's height normal to the true WGS84 reference ellipsoid surface at every internal solver iteration loop, completely accounting for earth flattening effects.

If you would like to run this code over a dataframe series using pandas vectorization arrays, let me know and we can adjust the execution loop!

