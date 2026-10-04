"""The 2D exploration simulator.

`simulator` is the one the reported runs use and casts its rays on the GPU.
`simulator_astar` is an older copy of it. Only its
`optimized_astar_path_planner` is used, by the policy and the baselines.
"""
