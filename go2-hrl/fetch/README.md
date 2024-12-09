# fetch
Repo for MEAM 5170 project using hierarchical model-based motion planner and RL whole body controller for the Unitree Go2


Look at this document for notes on which of the costs/constraints have been implemented / are wip
https://docs.google.com/document/d/1sHdM0nYUNMaIfw7tmxlh4CA-Nb2IrAO7sX3AtRb7L1k/edit?usp=sharing


As of now (Tues Dec 3rd 4:22pm) I think that add_initial_constraints and add_dynamics_constraints are good. The next thing to do would be to add a cost for tracking a reference tracectory if we want to implement the costs/contraints in order of most to least important.

To run this install `pydrake` and `plotly` (or whatever it asks for you to install), and then run python `/tamols/test.py` from the root dir/. If a feasible solution is found to the test problem, this will output an html file that is an interactive 3d view of the motion plan.