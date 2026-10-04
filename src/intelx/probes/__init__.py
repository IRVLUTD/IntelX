"""Timing and smoke checks.

`tier1` times the three prediction baselines against a running service, and
`astar` times A*. Neither is used by the benchmark. Both read a map and spawn
that do not ship, so point `MAP_PGM` and `SPAWN_XY` in `tier1` at your own.
"""
