"""Static photometric aperture rules on per-pixel source-flux Images.

Source positions are (x, y) in the Image's lambda/D frame. ``weights`` returns
weights on that Image's grid. Each hard-edged rule represents an idealized
aperture in the limit of infinitely fine detector pixels; detector pixelization
is not modelled here. Its native weights report coverage. Photometry integrates
the reconstructed planet and stellar intensities over the geometric region.
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Circular:
    """Circle of radius ``radius_lod`` with a named centre convention.

    ``centre="source"`` (default) uses the supplied source position exactly,
    without a peak search: one aperture fixed in angle per band must have a
    centre independent of pixelization. ``centre="peak"`` follows the PSF's
    refined local peak instead.
    """

    radius_lod: float
    centre: str = "source"

    def __post_init__(self):
        """Validate the static rule parameter."""
        if not math.isfinite(self.radius_lod) or self.radius_lod <= 0:
            raise ValueError("radius_lod must be finite and positive")
        if not isinstance(self.centre, str) or self.centre not in ("source", "peak"):
            raise ValueError('centre must be "source" or "peak"')

    def weights(self, psf, source_position, *, oversample=10):
        """Fractional native-pixel coverage, using exact circle-square overlaps."""
        from .performance import _rule_weights

        return _rule_weights(self, psf, source_position, oversample)


#: The photometric core used when a caller names no rule: a 0.7 lambda/D circle.
DEFAULT_APERTURE_RULE = Circular(0.7)


@dataclass(frozen=True)
class HalfMaximum:
    """Region about the PSF peak by construction, above ``fraction`` of it.

    Only the connected component of the refined source-local peak is used.
    """

    fraction: float = 0.5

    def __post_init__(self):
        """Validate the static rule parameter."""
        if not math.isfinite(self.fraction) or not 0 < self.fraction < 1:
            raise ValueError("fraction must be between zero and one")

    def weights(self, psf, source_position, *, oversample=10):
        """Fractional native-pixel coverage from the band-limited level set."""
        from .performance import _rule_weights

        return _rule_weights(self, psf, source_position, oversample)


@dataclass(frozen=True)
class Matched:
    """Weighting about the PSF peak by construction, using the PSF template.

    For w=p/peak, where peak is the band-limited subpixel maximum, the planet mean
    and variance fractions are sum(w*p) and sum(w**2*p). The collecting and variance
    areas are the pixel area times sum(w) and sum(w**2), respectively. A zero
    template has zero moments.
    Reference: Kasdin & Braems (2006), ApJ 646, 1260,
    https://doi.org/10.1086/505017 (PSF matched-filter photometry).
    Matched has no mask. The mean area and mean leakage grow with the field
    of view while the squared-weight variance moments converge.
    """

    def weights(self, psf, source_position, *, oversample=10):
        """Peak-normalized template on the supplied grid (no mask or recentering)."""
        from .performance import _template_peak, _validate_photometry_image

        _validate_photometry_image(psf)
        data = jnp.asarray(psf.data, dtype=float)
        peak = _template_peak(data)
        return jnp.where(peak > 0, data / jnp.where(peak > 0, peak, 1), 0)


ApertureRule = Circular | HalfMaximum | Matched
