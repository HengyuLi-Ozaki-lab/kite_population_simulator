"""Intervention operators: explicit, provenance-carrying changes applied to a kernel's response distributions.

Research plan v3 §4.4. An operator never trains anything; it states what a cell's response should be and moves
the kernel's distributions there in a way that keeps their support and can be undone.
"""

from kite.operators.tilt import ALPHA_MAX, TiltResult, cell_mean, floored, solve, tilt, tilt_cell

__all__ = ["ALPHA_MAX", "TiltResult", "cell_mean", "floored", "solve", "tilt", "tilt_cell"]
