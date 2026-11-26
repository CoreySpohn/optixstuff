"""Precision-preserving staging of static interpolation axes."""

import jax.numpy as jnp
import numpy as np


def _host_knot_coordinates(knots, dtype):
    """Split host knots into working anchors and binary64 differences."""
    anchor = knots.astype(dtype)
    residual = knots - anchor
    following = np.nextafter(anchor, dtype.type(np.inf))
    preceding = np.nextafter(anchor[-1], dtype.type(-np.inf))
    # Compare against the first representable query on or above each knot
    # and the last representable query on or below the upper endpoint.
    lower = np.where(residual > 0, following, anchor)
    upper = preceding if residual[-1] < 0 else anchor[-1]
    return (
        jnp.asarray(anchor),
        jnp.asarray(residual, dtype=dtype),
        jnp.asarray(lower),
        jnp.asarray(upper),
        jnp.asarray(residual == 0),
    )


def _interp_zero(query, knots, values):
    """Linearly read host knots with zero extrapolation."""
    if isinstance(query, (list, tuple)):
        query = jnp.asarray(query)
    dtype = jnp.result_type(query, values, 1.0)
    query = jnp.asarray(query, dtype=dtype)
    values = jnp.asarray(values, dtype=dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise ValueError("interpolation wavelengths must be real")
    anchor, residual, lower, upper, _ = _host_knot_coordinates(knots, dtype)
    widths = jnp.asarray(np.diff(knots), dtype=dtype)
    i = jnp.clip(jnp.searchsorted(lower, query, side="right") - 1, 0, len(knots) - 2)
    delta = (query - anchor[i]) - residual[i]
    # Guard degenerate widths for finite gradients when a tiny host interval
    # is too small for the working dtype.
    tiny = widths[i] <= np.spacing(np.finfo(dtype).eps)
    outside = (query < lower[0]) | (query > upper)
    fraction = jnp.where(tiny | outside, 0, delta) / jnp.where(tiny, 1, widths[i])
    value = values[i] + fraction * (values[i + 1] - values[i])
    return jnp.where(outside, 0, value)
