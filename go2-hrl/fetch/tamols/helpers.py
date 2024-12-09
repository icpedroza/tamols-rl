import numpy as np
from pydrake.all import cos, sin
from pydrake.symbolic import if_then_else, Expression
from pydrake.symbolic import cos, sin


# SPLINE

def evaluate_spline_position(tmls, coeffs, tau):
    return np.sum([coeffs[:, i] * tau**i for i in range(tmls.spline_order)], axis=0)

def evaluate_spline_velocity(tmls, coeffs, tau):
    # returns vel and euler rates (0:3 and 3:6 respectively)
    return np.sum([i * coeffs[:, i] * tau**(i - 1) for i in range(1, tmls.spline_order)], axis=0)
    
def evaluate_spline_acceleration(tmls, coeffs, tau):
    return np.sum([i * (i - 1) * coeffs[:, i] * tau**(i - 2) for i in range(2, tmls.spline_order)], axis=0)

def evaluate_angular_momentum_derivative(tmls, coeffs, tau):
    # Extract Euler angles, rates, and second derivatives

    # throwing error because the type is no longer symbolic.Expression (come back to this)
    # NOTE: GPTed
    pos = evaluate_spline_position(tmls, coeffs, tau)               # [x, y, z, phi, theta, psi]
    vel = evaluate_spline_velocity(tmls, coeffs, tau)               # [vx, vy, vz, phi_dot, theta_dot, psi_dot]
    accel_full = evaluate_spline_acceleration(tmls, coeffs, tau)    # [ax, ay, az, phi_ddot, theta_ddot, psi_ddot]

    # Unpack Euler angles and their derivatives
    phi, theta, psi = pos[3], pos[4], pos[5]
    phi_dot, theta_dot, psi_dot = vel[3], vel[4], vel[5]
    phi_ddot, theta_ddot, psi_ddot = accel_full[3], accel_full[4], accel_full[5]

    # Convert Euler angle rates to body angular velocity (assuming Z-Y-X or similar Euler angle sequence)
    # The formulas here match a common rotation sequence. Adjust if your Euler angle definition differs.
    # For a Z-Y-X sequence (yaw=psi, pitch=theta, roll=phi):
    # omega_x = phi_dot - psi_dot * sin(theta)
    # omega_y = theta_dot * cos(phi) + psi_dot * sin(phi) * cos(theta)
    # omega_z = psi_dot * cos(phi)*cos(theta) - theta_dot * sin(phi)
    
    sin_phi, cos_phi = sin(phi), cos(phi)
    sin_theta, cos_theta = sin(theta), cos(theta)

    omega_x = phi_dot - psi_dot * sin_theta
    omega_y = theta_dot * cos_phi + psi_dot * sin_phi * cos_theta
    omega_z = psi_dot * cos_phi * cos_theta - theta_dot * sin_phi
    omega = np.array([omega_x, omega_y, omega_z])

    # Compute angular acceleration (dot_omega)
    # Taking the time derivative of the above expressions:
    # dot_omega_x = phi_ddot - [psi_ddot * sin(theta) + psi_dot * cos(theta)*theta_dot]
    dot_omega_x = phi_ddot - (psi_ddot * sin_theta + psi_dot * cos_theta * theta_dot)

    # dot_omega_y is a bit more complex:
    # dot_omega_y = theta_ddot*cos(phi) - theta_dot*phi_dot*sin(phi)
    #              + psi_ddot*sin(phi)*cos(theta)
    #              + psi_dot[cos(phi)*cos(theta)*phi_dot - sin(phi)*sin(theta)*theta_dot]
    dot_omega_y = (theta_ddot*cos_phi 
                   - theta_dot*phi_dot*sin_phi
                   + psi_ddot*sin_phi*cos_theta
                   + psi_dot*(cos_phi*cos_theta*phi_dot - sin_phi*sin_theta*theta_dot))

    # dot_omega_z:
    # dot_omega_z = psi_ddot*cos(phi)*cos(theta)
    #              - psi_dot[ sin(phi)*cos(theta)*phi_dot + cos(phi)*sin(theta)*theta_dot ]
    #              - theta_ddot*sin(phi)
    #              - theta_dot*phi_dot*cos(phi)
    dot_omega_z = (psi_ddot*cos_phi*cos_theta
                   - psi_dot*(sin_phi*cos_theta*phi_dot + cos_phi*sin_theta*theta_dot)
                   - theta_ddot*sin_phi
                   - theta_dot*phi_dot*cos_phi)

    # I = tmls.moment_of_inertia
    dot_omega = np.array([dot_omega_x, dot_omega_y, dot_omega_z])
    # dot_omega = np.array([I[0] * dot_omega_x, I[1] * dot_omega_y, I[2] * dot_omega_z])

    # Compute angular momentum derivative
    # L = I * omega (for diagonal I)
    # L_dot = I * dot_omega + omega x (I * omega)
    I = tmls.moment_of_inertia  # Diagonal matrix
    I_omega = I @ omega
    L_dot = I @ dot_omega + np.cross(omega, I_omega)

    return L_dot


# GAIT

def get_num_contacts(tmls, phase):
    return sum(tmls.gait_pattern['contact_states'][phase])

def get_contact_pairs(tmls, stance_feet):
    return [(i, j) for i in stance_feet for j in stance_feet if i < j]

def get_stance_feet(tmls, phase):
    return [i for i, in_contact in enumerate(tmls.gait_pattern['contact_states'][phase]) if in_contact]


# HEIGHT MAP

def evaluate_height_at_symbolic_xy(tmls, height_map, x, y):
    m, n = height_map.shape # NOTE: SHOULD BE SQUARE
    cell_size = tmls.cell_size
    offset = tmls.cell_size * tmls.map_size / 2.0 # move zero to the center
    i = (x + offset) / cell_size
    j = (y + offset) / cell_size
    total_height = 0

    for k in range(m):
        for l in range(n):
            partial_height = if_then_else(abs(i - k) < 1, 
                                          if_then_else(abs(j - l) < 1, 
                                                       height_map[k, l] * (1 - abs(i - k)) * (1 - abs(j - l)),
                                                         0), 
                                          0)
            # partial_height = if_then_else(abs(i - k) < 1, height_map[k, l] * (1 - abs(i - k)) / (m), 0)
            # partial_height = if_then_else(abs(i - k) < 1, 
            #                               if_then_else(abs(j - l) < 1, 
            #                                            height_map[k, l],
            #                                              0), 
            #                               0)
            total_height += partial_height
            # total_height = 0.05
    return total_height


# TODO: finished?
def evaluate_smoothed_height_gradient(tmls, x, y):
    if tmls.height_map_smoothed is None:
        return np.zeros(2)
    _, grad = tmls.height_map_smoothed([x, y])
    return np.array(grad)

def evaluate_height_gradient(tmls, x, y):
    if tmls.height_map is None:
        return np.zeros(2)
    _, grad = tmls.height_map([x, y])
    return np.array(grad)

# math

def determinant(a, b, c):
    return a.dot(np.cross(b, c))

def get_R_B(phi_B):
    """
    Calculate rotation matrix R_B from ZYX-Euler angles
    Args:
        phi_B: [psi, theta, phi] ZYX-Euler angles of the base
    Returns:
        3x3 rotation matrix R_B
    """
    # Create RollPitchYaw object from Euler angles
    # Note: RollPitchYaw expects [roll, pitch, yaw] = [phi, theta, psi]
    # So we need to reverse the order from [psi, theta, phi]
    psi, theta, phi = phi_B  # [Yaw, Pitch, Roll]

    # Define rotation matrices for each axis
    R_z = np.array([
        [cos(psi), -sin(psi), 0],
        [sin(psi), cos(psi), 0],
        [0, 0, 1]
    ])
    R_y = np.array([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)]
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, cos(phi), -sin(phi)],
        [0, sin(phi), cos(phi)]
    ])

    # Combine rotations (ZYX convention)
    R_B = R_z @ R_y @ R_x

    return R_B


def get_R_B_numerical(phi_B):
    """
    Calculate rotation matrix R_B from ZYX-Euler angles
    Args:
        phi_B: [psi, theta, phi] ZYX-Euler angles of the base
    Returns:
        3x3 rotation matrix R_B
    """
    # Create RollPitchYaw object from Euler angles
    # Note: RollPitchYaw expects [roll, pitch, yaw] = [phi, theta, psi]
    # So we need to reverse the order from [psi, theta, phi]
    psi, theta, phi = phi_B  # [Yaw, Pitch, Roll]

    assert isinstance(phi_B, np.ndarray)
    assert phi_B.dtype == np.float64

    # Define rotation matrices for each axis
    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    # Combine rotations (ZYX convention)
    R_B = R_z @ R_y @ R_x

    return R_B

