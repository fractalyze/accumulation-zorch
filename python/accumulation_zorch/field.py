"""Field-element helpers shared across the prover: canonical-integer decode of a
jit kernel's output, and the small jax ``fr`` reductions the host orchestration
needs.

Field *identity* (which scalar/base dtype, its modulus and capacity) lives on
:class:`curve.Curve` now — construct a field element with the dtype itself
(``cv.fr(v)``) and serialize it with ``cv.fr(v).tobytes()`` (32-byte canonical
LE). The helpers here are curve-agnostic: they take an already-built array (or a
``Curve`` for the ``fr`` reductions), so they name no curve.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from .curve import Curve


def fe_value(arr: Any) -> int:
    """Canonical integer of a 1-element field array (Montgomery-decoded). Accepts
    a numpy or jax array (`np.asarray` normalizes)."""
    return int(np.asarray(arr).astype(object).reshape(()))


def fe_values(arr: Any) -> list[int]:
    """Canonical integers of a 1-D field array — the host boundary that turns a
    jit kernel's ``fr``/``fq`` output back into the int lists the (still host-side)
    serialization consumes."""
    a = np.asarray(arr)
    return [int(a[i].astype(object)) for i in range(a.shape[0])]


def fr_mul(cv: Curve, *vals: int) -> int:
    """Product of canonical ``fr`` values, reduced mod r — a jax ``fr`` reduction
    (no numpy field arithmetic on the prove path)."""
    if not vals:
        return 1
    return fe_value(jnp.prod(jnp.asarray(np.array(vals, dtype=cv.fr))))


def fr_add(cv: Curve, *vals: int) -> int:
    """Sum of canonical ``fr`` values, reduced mod r — a jax ``fr`` reduction."""
    if not vals:
        return 0
    return fe_value(jnp.sum(jnp.asarray(np.array(vals, dtype=cv.fr))))
