import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pydrake.symbolic import Expression, Evaluate
from .tamols import TAMOLSState
from .helpers import evaluate_spline_velocity, evaluate_spline_position, get_R_B_numerical


def plot_optimal_solutions_interactive(tmls: TAMOLSState, filepath='out/interactive_optimal_base_pose_and_footsteps.html'):
    optimal_footsteps = tmls.optimal_footsteps
    optimal_spline_coeffs = tmls.optimal_spline_coeffs
    num_phases = tmls.num_phases
    
    # Create figure
    fig = go.Figure()
    
    # Plot initial foot positions (p_meas)
    for i in range(tmls.p_meas.shape[0]):
        fig.add_trace(go.Scatter3d(
            x=[tmls.p_meas[i, 0]],
            y=[tmls.p_meas[i, 1]],
            z=[tmls.p_meas[i, 2]],
            mode='markers',
            name=f'p_meas {i+1}',
            marker=dict(size=8, color='black')
        ))
    
    # Colors for alternating steps
    colors = ['red', 'green', 'green', 'red']
    
    # Plot optimal footsteps
    for i in range(optimal_footsteps.shape[0]):
        fig.add_trace(go.Scatter3d(
            x=[optimal_footsteps[i, 0]],
            y=[optimal_footsteps[i, 1]],
            z=[optimal_footsteps[i, 2]],
            mode='markers+text',
            name=f'Footstep {i+1}',
            marker=dict(size=8, color=colors[i % len(colors)]),
            text=[f'Footstep {i+1}'],
            textposition='top center'
        ))
    
    # Plot lines between each adjacent footstep to indicate convex outline
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[optimal_footsteps[i, 0], optimal_footsteps[j, 0]],
            y=[optimal_footsteps[i, 1], optimal_footsteps[j, 1]],
            z=[optimal_footsteps[i, 2], optimal_footsteps[j, 2]],
            mode='lines',
            name=f'Edge {i+1}-{j+1}',
            line=dict(color='purple', width=2),  # Make lines purple and thinner
            showlegend=False
        ))

    
    # Plot splines for each phase
    for phase in range(num_phases):
        a_k = optimal_spline_coeffs[phase]
        T_k = tmls.phase_durations[phase]
        
        # Generate points along the spline
        tau_values = np.linspace(0, T_k, 100)
        spline_points = np.array([evaluate_spline_position(tmls, a_k, tau) for tau in tau_values])
        
        # Plot the spline
        fig.add_trace(go.Scatter3d(
            x=spline_points[:, 0],
            y=spline_points[:, 1],
            z=spline_points[:, 2],
            mode='lines',
            name=f'Spline Phase {phase+1}',
            line=dict(color='blue', width=4),
            showlegend=True
        ))

        # Add endpoints for each spline
        fig.add_trace(go.Scatter3d(
            x=[spline_points[0, 0], spline_points[-1, 0]],
            y=[spline_points[0, 1], spline_points[-1, 1]],
            z=[spline_points[0, 2], spline_points[-1, 2]],
            mode='markers',
            name=f'Spline Endpoints Phase {phase+1}',
            marker=dict(size=6, color='orange'),
            showlegend=True
        ))

        # Add beginning point for spline one
        if phase == 0:
            fig.add_trace(go.Scatter3d(
                x=[spline_points[0, 0]],
                y=[spline_points[0, 1]],
                z=[spline_points[0, 2]],
                mode='markers',
                name='Spline Start Point',
                marker=dict(size=7, color='orange'),
                showlegend=True
            ))
    

    # Plot rectangles centered at the initial and final base poses
    for phase in [0, num_phases - 1]:
        a_k = optimal_spline_coeffs[phase]
        T_k = tmls.phase_durations[phase]
        
        # Get the base pose at the start and end of the spline
        if phase == 0:
            base_pose = evaluate_spline_position(tmls, a_k, 0)[:3]
            phi_B = evaluate_spline_position(tmls, a_k, 0)[3:6]
        else:
            base_pose = evaluate_spline_position(tmls, a_k, T_k)[:3]
            phi_B = evaluate_spline_position(tmls, a_k, T_k)[3:6]

        R_B = get_R_B_numerical(phi_B)
  

        # Calculate the corners of the rectangle
        corners = []
        for offset in tmls.hip_offsets:
            corner = base_pose + R_B.dot(offset)
            corners.append(corner)
        
        # Create the rectangle by connecting the corners in the order 1-2-4-3
        corner_order = [0, 1, 3, 2, 0] 
        for i in range(len(corner_order) - 1):
            fig.add_trace(go.Scatter3d(
                x=[corners[corner_order[i]][0], corners[corner_order[i + 1]][0]],
                y=[corners[corner_order[i]][1], corners[corner_order[i + 1]][1]],
                z=[corners[corner_order[i]][2], corners[corner_order[i + 1]][2]],
                mode='lines',
                name=f'Rectangle Phase {phase+1}',
                line=dict(color='green', width=2),
                showlegend=False
            ))



    # Update layout - THIS IS WHERE SIZE OF PLOT SET
    fig.update_layout(
        title='Optimal Base Pose and Footsteps',
        scene=dict(
            xaxis=dict(range=[-1, 1], title='X'),
            yaxis=dict(range=[-1, 1], title='Y'),
            zaxis=dict(range=[-0.4, 1], title='Z'),  # Adjusted to plot down to -0.3
            aspectmode='cube'  # This ensures equal scaling
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        margin=dict(r=250),  # Add right margin for legend
        showlegend=True
    )
    
    # Add grid lines
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray'
            ),
            zaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray'
            )
        )
    )


    # Plot height map
    height_map = tmls.h
    grid_size = height_map.shape[0]
    boundary = grid_size * tmls.cell_size / 2
    x = np.linspace(-boundary, boundary, grid_size)
    y = np.linspace(-boundary, boundary, grid_size)
    x, y = np.meshgrid(x, y, indexing='ij')
    z = height_map

    # Add height map surface plot
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Viridis', opacity=0.7, showscale=False, name='Height Map'))


    # Save as HTML file for interactive viewing
    fig.write_html(filepath)


def save_optimal_solutions(tmls: TAMOLSState, filepath='out/optimal_solution.txt'):
    optimal_footsteps = tmls.optimal_footsteps
    optimal_spline_coeffs = tmls.optimal_spline_coeffs
    num_phases = tmls.num_phases

    with open(filepath, 'w') as f:
        # SOLUTION
        f.write("Optimal Footsteps:\n")
        for i in range(optimal_footsteps.shape[0]):
            f.write(f"Footstep {i+1}: {optimal_footsteps[i, 0]}, {optimal_footsteps[i, 1]}, {optimal_footsteps[i, 2]}\n")
        
        # f.write("\nOptimal Spline Coefficients:\n")
        # for i in range(num_phases):
        #     f.write(f"Spline Phase {i+1} Coefficients:\n")
        #     np.savetxt(f, optimal_spline_coeffs[i], fmt='%.6f')
        #     f.write("\n")

        f.write("\nObjective Function Optimal Value:\n")
        optimal_value = tmls.result.get_optimal_cost()  # Assuming this method exists
        f.write(f"Optimal Value: {optimal_value:.6f}\n")

        f.write("\nOptimal Spline End Points and Euler Angles:\n")
        for i in range(num_phases):
            T_k = tmls.phase_durations[i]
            start_values = evaluate_spline_position(tmls, optimal_spline_coeffs[i], 0)
            end_values = evaluate_spline_position(tmls, optimal_spline_coeffs[i], T_k)

            f.write(f"Phase {i+1} Start Point: {start_values[0]: >10.6f}, {start_values[1]: >10.6f}, {start_values[2]: >10.6f}, {start_values[3]: >10.6f}, {start_values[4]: >10.6f}, {start_values[5]: >10.6f}\n")
            if i == num_phases - 1:
                f.write(f"Phase {i+1} End Point:   {end_values[0]: >10.6f}, {end_values[1]: >10.6f}, {end_values[2]: >10.6f}, {end_values[3]: >10.6f}, {end_values[4]: >10.6f}, {end_values[5]: >10.6f}\n")

        f.write("\nOptimal Spline Velocities and Euler Derivatives:\n")
        for i in range(num_phases):
            T_k = tmls.phase_durations[i]
            start_velocity_values = evaluate_spline_velocity(tmls, optimal_spline_coeffs[i], 0)
            end_velocity_values = evaluate_spline_velocity(tmls, optimal_spline_coeffs[i], T_k)

            f.write(f"Phase {i+1} Start Velocity: {start_velocity_values[0]: >10.6f}, {start_velocity_values[1]: >10.6f}, {start_velocity_values[2]: >10.6f}, {start_velocity_values[3]: >10.6f}, {start_velocity_values[4]: >10.6f}, {start_velocity_values[5]: >10.6f}\n")
            if i == num_phases - 1:
                f.write(f"Phase {i+1} End Velocity:   {end_velocity_values[0]: >10.6f}, {end_velocity_values[1]: >10.6f}, {end_velocity_values[2]: >10.6f}, {end_velocity_values[3]: >10.6f}, {end_velocity_values[4]: >10.6f}, {end_velocity_values[5]: >10.6f}\n")

        # VELOCITY
        f.write(f"\n\nReference Velocity: {tmls.ref_vel}\n")








        # FOOT DISTANCES AND COSTS
        f.write("\nFoot Distances to Base:\n")
        num_legs = tmls.num_legs
        base_pos = tmls.base_pose[:3]

        for i in range(num_legs):
            dist = np.linalg.norm(optimal_footsteps[i] - base_pos)
            f.write(f"Foot {i+1} to Base: {dist:.6f}\n")


        f.write("\n\n\n\n")
        f.write("------- CONSTRAINT INFORMATION --------\n")
        f.write("\nTest Constraints Information:\n")
        for idx, constraint in enumerate(tmls.test_constraints):
            constraint_value = tmls.result.GetSolution(constraint.evaluator().Eval(tmls.result.GetSolution(constraint.variables())))
            f.write(f"{constraint}{constraint_value}\n\n")

        f.write("\nGIAC Constraints Information:\n")
        print(len(tmls.giac_constraints))
        for idx, constraint in enumerate(tmls.giac_constraints):
            constraint_value = tmls.result.GetSolution(constraint.evaluator().Eval(tmls.result.GetSolution(constraint.variables())))
            f.write(f"{constraint}{constraint_value}\n\n")

        f.write("\nKinematic Constraints Information:\n")
        for idx, constraint in enumerate(tmls.kinematic_constraints):
            constraint_value = tmls.result.GetSolution(constraint.evaluator().Eval(tmls.result.GetSolution(constraint.variables())))
            f.write(f"{constraint}{constraint_value}\n\n")



        f.write("------- COST INFORMATION --------\n")
        
        f.write("\nTracking Costs Information:\n")
        for idx, cost in enumerate(tmls.tracking_costs):
            cost_value = tmls.result.GetSolution(cost.evaluator().Eval(tmls.result.GetSolution(cost.variables())))
            f.write(f"Cost {idx+1}: {cost_value}\n")


        f.write("\nFoothold On Ground Costs Information:\n")
        for idx, cost in enumerate(tmls.foothold_on_ground_costs):
            cost_value = tmls.result.GetSolution(cost.evaluator().Eval(tmls.result.GetSolution(cost.variables())))
            f.write(f"Cost {idx+1}: {cost_value}\n")

        