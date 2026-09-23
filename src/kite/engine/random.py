"""Event-keyed randomness: every random draw is a pure function of (seed, event key).

A draw for "agent 17 decides about headline 4 at step 2" is the same number whichever batch it lands in, in
whichever order the batches run, and in every experimental condition - so the conditions are coupled by
common random numbers and a between-condition difference is never sampling noise. A run is reproduced from
its seed alone; no generator state is carried between events.
"""

from __future__ import annotations

import hashlib

import numpy as np

_SCALE = float(2**53)


def _digest(seed: int, keys: tuple) -> bytes:
    text = f"{int(seed)}|" + "|".join(str(k) for k in keys)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def event_uniform(seed: int, *keys) -> float:
    """One uniform draw in [0, 1) for this event."""
    bits = int.from_bytes(_digest(seed, keys)[:8], "little") >> 11  # 53 random bits, as a double can hold exactly
    return bits / _SCALE


def event_generator(seed: int, *keys) -> np.random.Generator:
    """A fresh generator for events that need more than one draw; seeded from the same digest."""
    return np.random.default_rng(int.from_bytes(_digest(seed, keys), "little"))
