import numpy as np
import pymap3d as pm

def generate_segmented_flight_path(start_lla, initial_heading_deg, target_heading_deg, 
                                   speed_mps, climb_rate_mps, first_leg_dist_m, 
                                   second_leg_dist_m, bank_deg=20.0, dt=1.0):
    """
    Generates a flight path that flies straight, performs a smooth banking turn to 
    a target heading, and flies straight again.
    
    Parameters:
    - start_lla: tuple (lat, lon, alt)
    - initial_heading_deg: Heading for the first straight leg (0 = North)
    - target_heading_deg: Heading for the final straight leg
    - speed_mps: Constant horizontal ground speed (m/s)
    - climb_rate_mps: Constant vertical speed (m/s)
    - first_leg_dist_m: Distance to fly straight on the first leg (meters)
    - second_leg_dist_m: Distance to fly straight on the final leg (meters)
    - bank_deg: Bank angle used strictly during the turning maneuver
    - dt: Time step interval (seconds)
    """
    lat0, lon0, alt0 = start_lla
    g = 9.80665
    
    # Calculate flight durations for straight segments
    t_leg1 = first_leg_dist_m / speed_mps
    t_leg2 = second_leg_dist_m / speed_mps
    
    # Calculate turn mechanics
    psi1 = np.radians(initial_heading_deg)
    psi2 = np.radians(target_heading_deg)
    
    # Determine shortest turn direction and angle difference
    delta_psi = (psi2 - psi1 + np.pi) % (2 * np.pi) - np.pi
    turn_direction = np.sign(delta_psi)  # +1 for right turn, -1 for left turn
    
    # Standard turn rate equation: omega = (g * tan(bank)) / v
    turn_rate_rad = (g * np.tan(np.radians(bank_deg))) / speed_mps
    t_turn = abs(delta_psi) / turn_rate_rad
    
    # Timeline thresholds
    end_leg1 = t_leg1
    end_turn = end_leg1 + t_turn
    total_duration = end_turn + t_leg2
    
    time_steps = np.arange(0, total_duration + dt, dt)
    trajectory_data = []
    
    # Cumulative displacements in local tangent plane (ENU framework)
    e_pos, n_pos, u_pos = 0.0, 0.0, 0.0
    current_heading_rad = psi1
    
    for i, t in enumerate(time_steps):
        if i == 0:
            current_phase = "Leg 1 (Straight)"
        else:
            # Determine flight phase and update heading dynamically
            if t <= end_leg1:
                current_phase = "Leg 1 (Straight)"
                current_heading_rad = psi1
            elif t <= end_turn:
                current_phase = "Transition Turn"
                # Incrementally rotate heading throughout the turn phase
                current_heading_rad = (psi1 + turn_direction * turn_rate_rad * (t - end_leg1)) % (2 * np.pi)
            else:
                current_phase = "Leg 2 (Straight)"
                current_heading_rad = psi2
            
            # Calculate velocity vector components for this step
            v_east = speed_mps * np.sin(current_heading_rad)
            v_north = speed_mps * np.cos(current_heading_rad)
            v_up = climb_rate_mps
            
            # Integrate positions step-by-step
            e_pos += v_east * dt
            n_pos += v_north * dt
            u_pos += v_up * dt
            
        # 1. Transform current cumulative local displacement into absolute ECEF
        x, y, z = pm.enu2ecef(e_pos, n_pos, u_pos, lat0, lon0, alt0)
        
        # 2. Convert absolute ECEF back to LLA coordinates
        lat, lon, alt = pm.ecef2lla(x, y, z)
        
        trajectory_data.append({
            'time': t,
            'phase': current_phase,
            'heading': np.degrees(current_heading_rad),
            'ecef': (x, y, z),
            'lla': (lat, lon, alt)
        })
        
    return trajectory_data

# --- Example Setup ---
if __name__ == "__main__":
    start_position = (40.6397, -73.7789, 1000.0) # JFK Airport starting at 1000m altitude
    
    flight_points = generate_segmented_flight_path(
        start_lla=start_position,
        initial_heading_deg=0.0,     # Heading due North initially
        target_heading_deg=90.0,    # Turning to head due East
        speed_mps=200.0,            # 200 m/s airspeed (~390 knots)
        climb_rate_mps=3.0,         # Constant uniform climb of 3 m/s
        first_leg_dist_m=3000.0,    # Fly straight for 3,000 meters
        second_leg_dist_m=3000.0,   # Fly straight for 3,000 meters after turn
        bank_deg=25.0,              # Comfortably bank at 25 degrees during turn
        dt=2.0                      # Output structural point every 2 seconds
    )
    
    # Print formatted output displaying the transition steps
    header = f"{'Time (s)':<10}{'Phase':<20}{'Heading':<10}{'ECEF X (m)':<14}{'ECEF Y (m)':<14}{'Lat (deg)':<11}{'Lon (deg)':<11}{'Alt (m)':<10}"
    print(header)
    print("-" * len(header))
    
    for pt in flight_points:
        t = pt['time']
        phase = pt['phase']
        hdg = pt['heading']
        x, y, _ = pt['ecef'] # ommiting Z to fit standard screen printing layout cleanly
        lat, lon, alt = pt['lla']
        print(f"{t:<10.1f}{phase:<20}{hdg:<10.1f}{x:<14.1f}{y:<14.1f}{lat:<11.5f}{lon:<11.5f}{alt:<10.1f}")
