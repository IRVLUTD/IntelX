"""IntelX, autonomous indoor exploration by local coverage gain prediction.

The stack is four pieces. `sim` carries the 2D simulator and its CUDA ray
casting, `policy` the planner that turns a predicted coverage field into a
goal, `planners` the junction and branch bookkeeping the global layer uses,
and `baselines` every method the paper compares against. `explorer` composes
them into one class and `cli` runs it.
"""

__version__ = "0.1.0"

__all__ = ["paths", "config"]
