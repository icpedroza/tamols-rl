import numpy as np
from pydrake.all import MathematicalProgram, Solve
from pydrake.solvers import SnoptSolver
from pydrake.all import Solve
from .tamols import TAMOLSState, setup_variables
from .constraints import (
    add_initial_constraints, 
    add_dynamics_constraints, 
    add_giac_constraints,
    add_friction_cone_constraints,
    add_kinematic_constraints,
)
from .costs import (
    add_tracking_cost, 
    add_foot_collision_cost,
    add_test_cost,
    add_foothold_on_ground_cost,
    add_base_pose_alignment_cost,
    add_nominal_kinematic_cost,
    add_edge_avoidance_cost,
    add_previous_solution_cost,
    add_smoothness_cost
)
from .plotting_helpers import *
from .map_processing import *
import manual_heightmaps as mhm

def setup_test_state(tmls: TAMOLSState):
     # Create a TAMOLSState instance

    tmls.base_pose = np.array([0, 0, 0.25, 0, 0, 0])  # Example initial base pose
    tmls.base_vel = np.array([0, 0, 0, 0, 0, 0])   # Example initial base velocity
    tmls.p_meas = np.array([
        [0.1934, 0.0465, 0],  # Front left leg
        [0.1934, -0.0465, 0], # Front right leg
        [-0.1934, 0.0465, 0], # Rear left leg
        [-0.1934, -0.0465, 0] # Rear right leg
    ])  # Reasonable initial foot positions

    # elevation_map = mhm.get_heightmap_stairs(tmls)
    elevation_map = mhm.get_heightmap_with_holes(tmls)
    
    h_s1, h_s2, gradients = process_height_maps(elevation_map)

    tmls.h = elevation_map
    tmls.h_s1 = h_s1
    tmls.h_s2 = h_s2

    tmls.h_grad_x, tmls.h_grad_y = gradients['h']
    tmls.h_s1_grad_x, tmls.h_s1_grad_y = gradients['h_s1']
    tmls.h_s2_grad_x, tmls.h_s2_grad_y = gradients['h_s2']

    tmls.ref_vel = np.array([0.5, 0, 0])
    tmls.ref_angular_momentum = np.array([0, 0, 0])

  
    # single spline / phase
    tmls.gait_pattern = {
        'phase_timing': [0, 0.4, 0.8, 1.2, 1.6, 2.0],  # Adjusted phase timings
        'contact_states': [
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
        ],
        
        # boolean array of whether the foot is at the final position in the i-th phase
        # used to determine if p or p_meas should be used
        'at_des_position': [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
        ],
    }

def setup_costs_and_constraints(tmls: TAMOLSState):
    # CONSTRAINTS
    add_initial_constraints(tmls)
    add_dynamics_constraints(tmls)
    add_kinematic_constraints(tmls) # for some reason problem becomes infeasible without this
    add_giac_constraints(tmls)
    add_friction_cone_constraints(tmls)

    
    # COSTS
    add_tracking_cost(tmls)
    add_foothold_on_ground_cost(tmls)
    add_nominal_kinematic_cost(tmls)
    # add_base_pose_alignment_cost(tmls)
    # add_edge_avoidance_cost(tmls)
    # add_previous_solution_cost(tmls)
    # add_smoothness_cost(tmls)

def run_single_optimization(tmls: TAMOLSState):
    solver = SnoptSolver()

    open('snopt.out', 'w').close()
    open('snopt.txt', 'w').close()

    print("Starting solve")
    tmls.prog.SetSolverOption(SnoptSolver().solver_id(), "Print file", "snopt.out")
    tmls.prog.SetSolverOption(SnoptSolver().solver_id(), "Summary file", "snopt.txt")
    tmls.prog.SetSolverOption(SnoptSolver().solver_id(), "Print frequency", 1)
    tmls.prog.SetSolverOption(SnoptSolver().solver_id(), "Major feasibility tolerance", 1e-6)
    tmls.prog.SetSolverOption(SnoptSolver().solver_id(), "Major optimality tolerance", 2e-2)

    tmls.result = solver.Solve(tmls.prog)
    
    if tmls.result.is_success():
        print("Optimization problem is feasible.")
        tmls.optimal_footsteps = tmls.result.GetSolution(tmls.p)
        num_phases = len(tmls.gait_pattern['phase_timing']) - 1
        tmls.optimal_spline_coeffs = [tmls.result.GetSolution(tmls.spline_coeffs[i]) for i in range(num_phases)]
        return True
    else:
        print("Optimization problem is not feasible.")
        print("Solver result code:", tmls.result.GetInfeasibleConstraints(tmls.prog))
        return False
    
def setup_next_optimization(prev_tmls: TAMOLSState):
    """Creates new TAMOLS state using results from previous optimization"""
    new_tmls = TAMOLSState()
    
    # Copy over static parameters and maps
    new_tmls.h = prev_tmls.h
    new_tmls.h_s1 = prev_tmls.h_s1
    new_tmls.h_s2 = prev_tmls.h_s2
    new_tmls.h_grad_x = prev_tmls.h_grad_x
    new_tmls.h_grad_y = prev_tmls.h_grad_y
    new_tmls.h_s1_grad_x = prev_tmls.h_s1_grad_x
    new_tmls.h_s1_grad_y = prev_tmls.h_s1_grad_y
    new_tmls.h_s2_grad_x = prev_tmls.h_s2_grad_x
    new_tmls.h_s2_grad_y = prev_tmls.h_s2_grad_y
    
    # Use final state from previous optimization as initial state
    final_phase_coeff = prev_tmls.optimal_spline_coeffs[-1]
    T_final = prev_tmls.phase_durations[-1]
    
    # Get final base pose and velocity
    final_pose = evaluate_spline_position(prev_tmls, final_phase_coeff, T_final)
    final_vel = evaluate_spline_velocity(prev_tmls, final_phase_coeff, T_final)
    
    # Offset x position for continuing motion
    new_tmls.base_pose = final_pose
    new_tmls.base_vel = final_vel
    
    # Use final footsteps as initial footsteps
    new_tmls.p_meas = prev_tmls.optimal_footsteps.copy()
    
    # Copy over gait pattern and other parameters
    new_tmls.gait_pattern = prev_tmls.gait_pattern
    new_tmls.ref_vel = prev_tmls.ref_vel
    new_tmls.ref_angular_momentum = prev_tmls.ref_angular_momentum
    
    return new_tmls
    

if __name__ == "__main__":

    # SETUP
    tmls1 = TAMOLSState()
    setup_test_state(tmls1)
    setup_variables(tmls1)
    setup_costs_and_constraints(tmls1)

    if run_single_optimization(tmls1):
        plot_optimal_solutions_interactive(tmls1, filepath='out/interactive_optimal_base_pose_and_footsteps1.html')
        save_optimal_solutions(tmls1, filepath='out/optimal_solution1.txt')

        tmls2 = setup_next_optimization(tmls1)
        setup_variables(tmls2)
        setup_costs_and_constraints(tmls2)

        if run_single_optimization(tmls2):
            plot_optimal_solutions_interactive(tmls2, filepath='out/interactive_optimal_base_pose_and_footsteps2.html')
            save_optimal_solutions(tmls2, filepath='out/optimal_solution2.txt')
   