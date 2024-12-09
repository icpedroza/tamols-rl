from dataclasses import dataclass, field
import numpy as np
from pydrake.all import (
    MathematicalProgram,
    Constraint, Cost
)
from pydrake.solvers import MathematicalProgramResult
from typing import List, Dict, Union

@dataclass
class TAMOLSState:
    # Time and horizon parameters
    dt: float = 0.01
    horizon_steps: int = 100

    # Height map data
    cell_size: float = 0.05
    map_size: int = 27
    h: np.ndarray = None
    h_s1: np.ndarray = None
    h_s2: np.ndarray = None
    h_grad_x: np.ndarray = None
    h_grad_y: np.ndarray = None
    h_s1_grad_x: np.ndarray = None
    h_s1_grad_y: np.ndarray = None
    h_s2_grad_x: np.ndarray = None
    h_s2_grad_y: np.ndarray = None

    # Robot configuration
    num_legs: int = 4
    spline_order: int = 4
    base_dims: int = 6

    # Physical parameters
    mass: float = 6.921
    mu: float = 0.7
    # TODO: find something reasonable for this
    moment_of_inertia: np.ndarray = field(default_factory=lambda: np.diag([0.02448, 0.098077, 0.107]))

    # Leg configuration
    hip_offsets: np.ndarray = field(default_factory=lambda: np.array([
        [0.1934, 0.0465, 0],   # FL - from FL_hip_joint origin
        [0.1934, -0.0465, 0],  # FR - from FR_hip_joint origin
        [-0.1934, 0.0465, 0],  # RL - from RL_hip_joint origin
        [-0.1934, -0.0465, 0], # RR - from RR_hip_joint origin
    ]))
    l_min: float = 0.15
    l_max: float = 0.4
    h_des: float = 0.25

    # Current state
    p_meas: np.ndarray = None
    base_pose: np.ndarray = None
    base_vel: np.ndarray = None

    # Optimization hyper-parameters TODO
    tau_sampling_rate: int = 6
    base_pose_sampling_rate: int = 1
    weights: Dict[str, float] = field(default_factory=lambda: {
        'robustness_margin': 1.0,
        'footholds_on_ground': 1.0,
        'leg_collision_avoidance': 1.0,
        'nominal_kinematics': 1.0,
        'base_alignment': 1.0,
        'edge_avoidance': 1.0,
        'previous_solution_tracking': 1.0,
        'reference_trajectory_tracking': 1.0,
        'smoothness': 1.0,
    })

    # Optimization variables
    prog: MathematicalProgram = None
    spline_coeffs: List[np.ndarray] = None
    p: np.ndarray = None
    epsilon: np.ndarray = None


    # Program Constraint Bindings
    test_constraints: List[Constraint] = field(default_factory=list)
    initial_constraints: List[Constraint] = field(default_factory=list)
    dynamics_constraints: List[Constraint] = field(default_factory=list)
    kinematic_constraints: List[Constraint] = field(default_factory=list)
    giac_constraints: List[Constraint] = field(default_factory=list)
    friction_cone_constraints: List[Constraint] = field(default_factory=list)
    # Program Cost Bindings
    tracking_costs: List[Cost] = field(default_factory=list)
    foothold_on_ground_costs: List[Cost] = field(default_factory=list)
    nominal_kinematic_costs: List[Cost] = field(default_factory=list)
    base_pose_alignment_costs: List[Cost] = field(default_factory=list)
    edge_avoidance_costs: List[Cost] = field(default_factory=list)
    previous_solution_costs: List[Cost] = field(default_factory=list)
    smoothness_costs: List[Cost] = field(default_factory=list)
    
    # Optimization results
    result: MathematicalProgramResult = None
    optimal_footsteps: np.ndarray = None
    optimal_spline_coeffs: List[np.ndarray] = None

    # Desired configuration
    nominal_height: float = 0.4
    foot_radius: float = 0.02
    min_foot_distance: float = 1
    desired_height: float = 0.4

    ref_vel: np.ndarray = None
    ref_angular_momentum: np.ndarray = None
    gait_pattern: Dict[str, List[Union[float, List[int]]]] = None
    num_phases: int = None # setup by setup_variables
    
    # Internal variables
    phase_durations: List[float] = None
    
def setup_variables(tmls: TAMOLSState):
    """Setup optimization variables based on gait phases"""

    tmls.prog = MathematicalProgram()
    
    phase_times = tmls.gait_pattern['phase_timing']
    tmls.num_phases = len(phase_times) - 1  # Number of intervals between timestamps
    tmls.phase_durations = [ phase_times[i+1] - phase_times[i] for i in range(tmls.num_phases) ]

    # Spline coefficients (26-27)
    tmls.spline_coeffs = []
    for i in range(tmls.num_phases):
        coeffs = tmls.prog.NewContinuousVariables(
            tmls.base_dims, 
            tmls.spline_order, 
            f'a_{i}'
        )
        tmls.spline_coeffs.append(coeffs)

    
    # Foothold plan (27)
    tmls.p = tmls.prog.NewContinuousVariables(
        tmls.num_legs, 3, 
        'p'
    )
    
    # Stability constraints slack variables (27)
    tmls.epsilon = tmls.prog.NewContinuousVariables(
        tmls.num_phases, 
        'epsilon'
    )

