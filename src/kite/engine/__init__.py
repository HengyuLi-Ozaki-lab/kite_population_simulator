"""The population execution engine: runs agents over kernel response distributions, never over the kernel itself.

Research plan v3 §4.3. The tabulated tier executes a population from a table of unique states to response
distributions; randomness is keyed by event, so two conditions share their random numbers (common random
numbers) and a result never depends on batch boundaries or execution order.
"""

from kite.engine.random import event_generator, event_uniform
from kite.engine.tabulated import Table, expectation, run

__all__ = ["Table", "event_generator", "event_uniform", "expectation", "run"]
