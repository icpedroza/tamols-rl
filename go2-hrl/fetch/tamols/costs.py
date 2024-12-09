import numpy as np
from .tamols import TAMOLSState
from pydrake.symbolic import if_then_else, Expression
from pydrake.math import exp, abs, sqrt
from .helpers import (
    evaluate_spline_position, evaluate_spline_velocity, 
    evaluate_height_at_symbolic_xy,
    get_R_B, get_stance_feet, evaluate_angular_momentum_derivative
)
from pydrake.symbolic import floor, ExtractVariablesFromExpression, Expression
from pydrake.solvers import QuadraticConstraint, Cost

def add_tracking_cost(tmls: TAMOLSState):
    """Cost to track reference trajectory"""
    print("Adding tracking cost...") 
    if tmls.ref_vel is None:
        raise ValueError("Reference velocity not set")

    num_phases = len(tmls.phase_durations)
    total_cost = 0

    for phase in range(num_phases):
        a_k = tmls.spline_coeffs[phase]
        T_k = tmls.phase_durations[phase]

        for tau in np.linspace(0, T_k, tmls.tau_sampling_rate+1)[:tmls.tau_sampling_rate]:
            vel = evaluate_spline_velocity(tmls, a_k, tau)[0:3]

            for dim in range(1): # ONLY TRACK X VEL
                weight = 2 * T_k / tmls.tau_sampling_rate
                total_cost += weight * (vel[dim] - tmls.ref_vel[dim])**2

    c = tmls.prog.AddQuadraticCost(total_cost)
    tmls.tracking_costs.append(c)
    
def add_foot_collision_cost(tmls: TAMOLSState):
    print("Adding foot collision cost...")
    raise NotImplementedError("add_foot_collision_cost is not yet implemented")

#     num_legs = tmls.num_legs
#     min_foot_distance = tmls.min_foot_distance
#     total_cost = Expression(0)


#     # test with leg 1 and 2
#     distance = sqrt((tmls.p[0] - tmls.p[1]).dot(tmls.p[0] - tmls.p[1]) + 1e-6) 

#     penalty = (distance - min_foot_distance)**2
#     total_cost += penalty


#     # for i in range(num_legs):
#     #     for j in range(i + 1, num_legs):
#     #         distance = np.linalg.norm(tmls.p[i] - tmls.p[j])
#     #         penalty = if_then_else(distance > min_foot_distance, (distance - min_foot_distance)**2, 0)
#     #         total_cost += penalty

#     tmls.prog.AddCost(total_cost)

def add_test_cost(tmls: TAMOLSState):
    print("Adding test cost...")
    total_cost = 0

    # Encourage the model to rotate 90 degrees left in ZYX Euler coordinates at the end of the base pose trajectory
    num_phases = len(tmls.phase_durations)
    a_k = tmls.spline_coeffs[-1]  # Get the coefficients for the last phase
    T_k = tmls.phase_durations[-1]  # Get the duration for the last phase

    # Get the base pose at the end of the last phase
    phi_B_end = evaluate_spline_position(tmls, a_k, T_k)[3:6]

    # Desired rotation is 90 degrees left around the Z-axis in ZYX Euler coordinates
    desired_rotation = np.array([np.pi / 2, 0, 0])

    # Calculate the cost as the squared difference between the current and desired rotation
    rotation_cost = np.sum((phi_B_end - desired_rotation) ** 2)
    total_cost += rotation_cost

    tmls.prog.AddCost(total_cost)


def add_foothold_on_ground_cost(tmls: TAMOLSState):
    """Cost to keep footholds on ground"""
    print("Adding foothold cost...")

    total_cost = 0

    for i in range(tmls.num_legs):
        # Create continuous height approximation using bilinear interpolation
        h_pi = evaluate_height_at_symbolic_xy(tmls, tmls.h, tmls.p[i][0], tmls.p[i][1])
        
        # Add to cost using interpolated height
        cost = (h_pi - tmls.p[i][2])**2
        # if i is 0:
        #     cost = (0.05 - tmls.p[i][2])**2
        total_cost += 10**2 * cost

    c = tmls.prog.AddCost(total_cost)
    tmls.foothold_on_ground_costs.append(c)

def add_nominal_kinematic_cost(tmls: TAMOLSState):
    """Cost for nominal kinematics"""
    print("Adding nominal kinematic cost...")

    total_cost = 0

    l_des = np.array([0., 0., tmls.h_des])

    num_phases = len(tmls.phase_durations)
    for phase in [-1]: #range(num_phases):
        a_k = tmls.spline_coeffs[phase]
        T_k = tmls.phase_durations[phase]

        p_B = evaluate_spline_position(tmls, a_k, T_k / 2.0)[:3]
        phi_B = evaluate_spline_position(tmls, a_k, T_k / 2.0)[3:6]

        R_B = get_R_B(phi_B)

        stance_feet = get_stance_feet(tmls, phase)
        p_alr_at_des_pos = tmls.gait_pattern['at_des_position'][phase]

        for i in range(4): #stance_feet:
            base_minus_leg = p_B + R_B.dot(tmls.hip_offsets[i]) - l_des
            p_i = tmls.p[i] if p_alr_at_des_pos[i] else tmls.p_meas[i]
            cost = np.dot(base_minus_leg - p_i, base_minus_leg - p_i)
            weight = 20
            total_cost += weight * cost

    c = tmls.prog.AddCost(total_cost)
    tmls.nominal_kinematic_costs.append(c)

def add_base_pose_alignment_cost(tmls: TAMOLSState):
    """Cost to align base on ground"""
    print("Adding base pose alignment cost...")

    e_z = np.array([0., 0., 1.])
    total_cost = 0

    l_des = np.array([0., 0., tmls.h_des])

    num_phases = len(tmls.phase_durations)
    for phase in range(num_phases):
        a_k = tmls.spline_coeffs[phase]
        T_k = tmls.phase_durations[phase]

        for tau in np.linspace(0, T_k, tmls.base_pose_sampling_rate+1)[:tmls.base_pose_sampling_rate]:
            p_B = evaluate_spline_position(tmls, a_k, tau)[:3]
            phi_B = evaluate_spline_position(tmls, a_k, tau)[3:6]

            R_B = get_R_B(phi_B)

            hs2_pB = evaluate_height_at_symbolic_xy(tmls, tmls.h_s2, p_B[0], p_B[1])
            for i in range(tmls.num_legs):
                base_minus_leg = p_B + R_B.dot(tmls.hip_offsets[i]) - l_des
                # base_minus_leg = p_B - l_des
                cost = (e_z.dot(base_minus_leg) - hs2_pB)**2
                weight = 0.01 * T_k / tmls.tau_sampling_rate
                total_cost += weight * cost

    c = tmls.prog.AddCost(total_cost)
    tmls.base_pose_alignment_costs.append(c)


def add_edge_avoidance_cost(tmls: TAMOLSState):
    """Cost to avoid edges"""
    print("Adding edge avoidance cost...")

    total_cost = 0

    for i in range(tmls.num_legs):
        # Create continuous height approximation using bilinear interpolation
        h_grad_x_pi = evaluate_height_at_symbolic_xy(tmls, tmls.h_grad_x, tmls.p[i][0], tmls.p[i][1])
        h_grad_y_pi = evaluate_height_at_symbolic_xy(tmls, tmls.h_grad_y, tmls.p[i][0], tmls.p[i][1])
        h_s1_grad_x_pi = evaluate_height_at_symbolic_xy(tmls, tmls.h_s1_grad_x, tmls.p[i][0], tmls.p[i][1])
        h_s1_grad_y_pi = evaluate_height_at_symbolic_xy(tmls, tmls.h_s1_grad_y, tmls.p[i][0], tmls.p[i][1])

        h_grad_pi = np.array([h_grad_x_pi, h_grad_y_pi])
        h_s1_grad_pi = np.array([h_s1_grad_x_pi, h_s1_grad_y_pi])
        
        # Add to cost using interpolated height
        cost = 3 * h_grad_pi.dot(h_grad_pi) + h_s1_grad_pi.dot(h_s1_grad_pi)
        total_cost += cost

    c = tmls.prog.AddCost(total_cost)
    tmls.edge_avoidance_costs.append(c)

def add_previous_solution_cost(tmls: TAMOLSState):
    """Cost to keep relative closeness to previous solution"""
    print("Adding previous solution cost...")

    total_cost = 0

    for i in range(tmls.num_legs):
        difference_squared = (tmls.p[i] - tmls.p_meas[i])**2
        cost = np.sum(difference_squared)
        
        total_cost += 0.01 * cost

    c = tmls.prog.AddQuadraticCost(total_cost)
    tmls.previous_solution_costs.append(c)

def add_previous_solution_cost(tmls: TAMOLSState):
    """Cost to keep relative closeness to previous solution"""
    print("Adding previous solution cost...")

    total_cost = 0

    for i in range(tmls.num_legs):
        difference_squared = (tmls.p[i] - tmls.p_meas[i])**2
        cost = np.sum(difference_squared)
        
        total_cost += 0.01 * cost

    c = tmls.prog.AddQuadraticCost(total_cost)
    tmls.previous_solution_costs.append(c)

def add_smoothness_cost(tmls: TAMOLSState):
    """Cost to keep relative closeness to previous solution"""
    print("Adding previous solution cost...")

    total_cost = 0

    num_phases = len(tmls.phase_durations)
    for phase in range(num_phases):
        a_k = tmls.spline_coeffs[phase]
        T_k = tmls.phase_durations[phase]

        for tau in np.linspace(0, T_k, tmls.tau_sampling_rate+1)[:tmls.tau_sampling_rate]:
            L_dot_B = evaluate_angular_momentum_derivative(tmls, a_k, tau)[0:3]

            cost = L_dot_B.dot(L_dot_B)
            weight = 0.001 * T_k / tmls.tau_sampling_rate
            total_cost += weight * cost

    c = tmls.prog.AddCost(total_cost)
    tmls.smoothness_costs.append(c)

