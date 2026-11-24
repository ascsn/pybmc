"""Package-wide random-number generation.

All stochastic pybmc routines (the MCMC samplers in
:mod:`pybmc.inference_utils` and the posterior-predictive draws in
:mod:`pybmc.sampling_utils`) obtain their generator through
:func:`get_rng`, so the whole pipeline is driven by one seeded state:

- With ``seed=None`` a function draws from the shared package-wide
  generator, which is seeded with `DEFAULT_SEED` at import time. A fresh
  session that performs the same sequence of calls is therefore fully
  reproducible, training included.
- With an explicit ``seed`` a function uses an independent generator
  seeded with that value, so a single call is reproducible in isolation
  regardless of what ran before it.

Use `set_seed` to re-seed the shared generator mid-session (e.g. at the
top of a script or between repetitions of an experiment).
"""

import numpy as np

#: Seed for the shared package-wide generator (and the default for the
#: per-call reproducible posterior-predictive draws).
DEFAULT_SEED = 142858

_global_rng = np.random.default_rng(DEFAULT_SEED)


def get_rng(seed=None):
    """
    Returns the generator to use for a stochastic routine.

    Args:
        seed (int | numpy.random.Generator | None): If None, the shared
            package-wide generator (seeded with `DEFAULT_SEED` at import,
            or the last `set_seed` call). If an integer, a fresh
            independent generator seeded with it. A ready-made
            `numpy.random.Generator` is returned unchanged.

    Returns:
        numpy.random.Generator: The generator to draw from.
    """
    if seed is None:
        return _global_rng
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def set_seed(seed=DEFAULT_SEED):
    """
    Re-seeds the shared package-wide generator.

    Args:
        seed (int, optional): New seed (default: `DEFAULT_SEED`).

    Returns:
        numpy.random.Generator: The freshly seeded shared generator.
    """
    global _global_rng
    _global_rng = np.random.default_rng(seed)
    return _global_rng
