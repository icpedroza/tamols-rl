import numpy as np
from .tamols import TAMOLSState
from .helpers import (
    get_num_contacts, get_stance_feet, get_contact_pairs, 
    evaluate_spline_position, evaluate_spline_acceleration, 
    evaluate_angular_momentum_derivative, evaluate_height_at_symbolic_xy, 
    determinant, get_R_B
)
from pydrake.solvers import QuadraticConstraint

def add_initial_constraints(tmls: TAMOLSState, log: bool = False):
    """Add initial and junction constraints on the spline coefficients"""
    print("Adding initial constraints...")
    if tmls.base_pose is None or tmls.p_meas is None or tmls.base_vel is None:
        raise ValueError("Initial state not set")
    
    # SPLINE INITIAL CONSTRAINTS
    a0 = tmls.spline_coeffs[0] 
    T0 = tmls.phase_durations[0]
    for dim in range(tmls.base_dims):
        tmls.prog.AddLinearConstraint(
            a0[dim,0] == tmls.base_pose[dim]
        )
        tmls.prog.AddLinearConstraint(
            a0[dim,1] == tmls.base_vel[dim]
        )

    # SPLINE JUNCTION CONSTRAINTS
    num_phases = len(tmls.phase_durations)
    for phase in range(num_phases - 1):
        ak = tmls.spline_coeffs[phase]      # Current phase
        ak1 = tmls.spline_coeffs[phase+1]   # Next phase
        Tk = tmls.phase_durations[phase]    # Current phase duration
        Tk1 = tmls.phase_durations[phase+1] # Next phase duration
        
        for dim in range(tmls.base_dims):
            # Position continuity: evaluate end of current = start of next
            pos_k = sum(ak[dim,i] * Tk**i for i in range(tmls.spline_order))  
            pos_k1 = ak1[dim,0]                                  
            tmls.prog.AddLinearConstraint(pos_k == pos_k1)
            
            # Velocity continuity
            vel_k = sum(i * ak[dim,i] * Tk**(i-1) for i in range(1, tmls.spline_order))   
            vel_k1 = ak1[dim,1]                                               # τ = 0
            tmls.prog.AddLinearConstraint(vel_k == vel_k1)

    # NOTE: check if there need to be any intial conditions on feet

def add_dynamics_constraints(tmls: TAMOLSState):
    """Add GIAC stability constraints at continuously sampled points"""
    print("Adding dynamics constraints...")

    # Constants
    num_phases = len(tmls.phase_durations)
    e_z = np.array([0., 0., 1.])
    mu = tmls.mu
    m = tmls.mass

    for phase in range(num_phases):
        a_k = tmls.spline_coeffs[phase]
        T_k = tmls.phase_durations[phase]

        N = get_num_contacts(tmls, phase)
        stance_feet = get_stance_feet(tmls, phase) # returns the indices of the feet in contact (but does not account for p_meas vs p)
        p_alr_at_des_pos = tmls.gait_pattern['at_des_position'][phase]

        eps = tmls.epsilon[phase]

        tmls.prog.AddConstraint(eps >= 0)

        weight = 1
        tmls.prog.AddQuadraticCost(weight * eps * eps)
        
        for tau in np.linspace(0, T_k, tmls.tau_sampling_rate+1)[:tmls.tau_sampling_rate]:
            p_B = evaluate_spline_position(tmls, a_k, tau)[:3]
            a_B = evaluate_spline_acceleration(tmls, a_k, tau)[0:3]
            L_dot_B = evaluate_angular_momentum_derivative(tmls, a_k, tau)[0:3]
            
            if N > 0: # Eq 17a: Friction cone constraint - FIXED
                LHS = (mu * a_B[2])**2 - a_B[0]**2 - a_B[1]**2 
                RHS = 0
                tmls.prog.AddConstraint(LHS >= RHS)

 
            if N >= 3: # Eq 17b: Multiple contact GIAC constraints
                for i, j in get_contact_pairs(tmls, stance_feet):

                    p_i = tmls.p[i] if p_alr_at_des_pos[i] else tmls.p_meas[i]
                    p_j = tmls.p[j] if p_alr_at_des_pos[j] else tmls.p_meas[j]
                    p_ij = p_j - p_i
                    
                    LHS = m * determinant(p_ij, p_B - p_i, a_B) - p_ij.dot(L_dot_B)
                    RHS = eps 
                    LHS = m * determinant(p_ij, p_B - p_i, a_B) - p_ij.dot(L_dot_B)
                    RHS = eps 


                    tmls.prog.AddConstraint(LHS <= RHS)
                
                    
            if N == 2: 
                # Eq 17c,d: Double support constraints
                i, j = stance_feet
                p_i = tmls.p[i] if p_alr_at_des_pos[i] else tmls.p_meas[i]
                p_j = tmls.p[j] if p_alr_at_des_pos[j] else tmls.p_meas[j]
                p_ij = p_j - p_i
                
                # 17c: Equality constraint
                print(f"adding N={N} constraint 17c")
                tmls.prog.AddConstraint(
                    m * determinant(p_ij, p_B - p_i, a_B) - p_ij.dot(L_dot_B) <= eps
                )
                tmls.prog.AddConstraint(
                    -(m * determinant(p_ij, p_B - p_i, a_B) - p_ij.dot(L_dot_B)) <= eps
                )
                
                
                # 17d: Moment constraint
                # print(f"adding N={N} constraint 17d")
                g = np.array([0, 0, -9.81])
                M_i = np.cross(p_B - p_i, g - a_B) - L_dot_B / m
                cost = determinant(e_z, p_ij, M_i)
                LHS = -cost
                RHS = eps
                
                tmls.prog.AddConstraint(LHS <= RHS)

            # SKIP N < 2 CASES FOR NOW

def add_kinematic_constraints(tmls: TAMOLSState):
    """Max distance between hip and foot"""
    print("Adding kinematic constraints...")

    for phase_idx, at_des_pos in enumerate(tmls.gait_pattern['at_des_position']):
        for leg_idx in range(tmls.num_legs):
            if at_des_pos[leg_idx]:

                a_k = tmls.spline_coeffs[phase_idx]
                T_k = tmls.phase_durations[phase_idx]
                
                for tau in np.linspace(0, T_k, tmls.tau_sampling_rate+1)[:tmls.tau_sampling_rate]:
                    base_pos = evaluate_spline_position(tmls, tmls.spline_coeffs[phase_idx], tau)[0:3]
                    phi_B = evaluate_spline_position(tmls, a_k, tau)[3:6]

                    R_B = get_R_B(phi_B)

                    diff = base_pos + R_B @ tmls.hip_offsets[leg_idx]- tmls.p[leg_idx]
                    total = diff.dot(diff)

                    c = tmls.prog.AddConstraint(total <= tmls.l_max**2)
                    tmls.kinematic_constraints.append(c)

                    c = tmls.prog.AddConstraint(total >= tmls.l_min**2)
                    tmls.kinematic_constraints.append(c)


                    # difference vector should sit in a l_max ball (convex set)
                    # c =tmls.prog.AddQuadraticConstraint(
                    #     diff.dot(diff),
                    #     -np.inf,
                    #     tmls.l_max**2
                    # )
                    # # if leg_idx == 1:
                    # tmls.kinematic_constraints.append(c)


                    # # base pose should be l_min above the foot
                    # c = tmls.prog.AddLinearConstraint(
                    #     base_pos[2] >= tmls.p[leg_idx][2] + .1 # TODO: REMOVE HARDCODE
                    # )
                    # tmls.kinematic_constraints.append(c)

def add_friction_cone_constraints(tmls: TAMOLSState):
    """Friction cone constraints"""
    print("Adding friction cone constraints...")

    for phase_idx in range(len(tmls.phase_durations)):
        T_k = tmls.phase_durations[phase_idx]
        
        for tau in np.linspace(0, T_k, tmls.tau_sampling_rate+1)[:tmls.tau_sampling_rate]:
            base_acc = evaluate_spline_acceleration(tmls, tmls.spline_coeffs[phase_idx], tau)[2]
            
            # Ensure base pose acceleration is always greater than gravity in the z direction
            c = tmls.prog.AddLinearConstraint(base_acc >= -9.81)
            tmls.friction_cone_constraints.append(c)

def add_giac_constraints(tmls: TAMOLSState):
    """Enforce feet form a convex polygon"""
    print("Adding GIAC constraints...")

    # Get the final foot locations
    # p_1 = tmls.p_meas[0]
    # p_2 = tmls.p_meas[1]
    # p_3 = tmls.p_meas[2]
    # p_4 = tmls.p_meas[3]
    p_1 = tmls.p[0]
    p_2 = tmls.p[1]
    p_3 = tmls.p[2]
    p_4 = tmls.p[3]

    # Constraint (p12 × p13) z-component ≤ 0 for p_meas
    p_12 = p_2 - p_1
    p_13 = p_3 - p_1
    constraint_1 = tmls.prog.AddConstraint(p_12[0] * p_13[1] - p_12[1] * p_13[0] <= 0)
    tmls.giac_constraints.append(constraint_1)

    # Constraint (p14 × p13) z-component ≤ 0 for p_meas
    p_14 = p_4 - p_1
    constraint_2 = tmls.prog.AddConstraint(p_14[0] * p_13[1] - p_14[1] * p_13[0] <= 0)
    tmls.giac_constraints.append(constraint_2)

    # Constraint (p24 × p23) z-component ≤ 0 for p_meas
    p_23 = p_3 - p_2
    p_24 = p_4 - p_2
    constraint_3 = tmls.prog.AddConstraint(p_24[0] * p_23[1] - p_24[1] * p_23[0] <= 0)
    tmls.giac_constraints.append(constraint_3)

    # Constraint (p24 × p21) z-component ≤ 0 for p_meas
    p_21 = p_1 - p_2
    constraint_4 = tmls.prog.AddConstraint(p_24[0] * p_21[1] - p_24[1] * p_21[0] <= 0)
    tmls.giac_constraints.append(constraint_4)



    # TODO ADD GIAC CONSTRAINTS FOR INTERMEDIATE GAIT