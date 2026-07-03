"""Field-element serialization boundary: canonical-integer decode of a field
dtype array back into the int lists the (still host-side) byte serialization
consumes.

Field *identity* (which scalar/base dtype, its modulus and capacity) lives on
:class:`curve.Curve` now — construct a field element with the dtype itself
(``cv.fr(v)``) and serialize it with ``cv.fr(v).tobytes()`` (32-byte canonical
LE). The helpers here are curve-agnostic: they take an already-built array, so
they name no curve. Field *arithmetic* stays on the ``zk_dtypes`` arrays
themselves (native ``*``/``+`` auto-reduce mod p — the zorch idiom); these
helpers only cross the dtype→int boundary at the edge.
"""

from typing import Any

import numpy as np


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
