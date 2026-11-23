"""Static photometric aperture rules on per-pixel source-flux Images.

Source positions are (x, y) in the Image's lambda/D frame. ``weights`` returns
weights on that Image's grid. Each hard-edged rule represents an idealized
aperture in the limit of infinitely fine detector pixels; detector pixelization
is not modelled here. Its native weights report coverage. Photometry integrates
the reconstructed planet and stellar intensities over the geometric region.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Circular:
    """Circle of radius ``radius_lod`` with a named centre convention.

    ``centre="source"`` (default) uses the supplied source position exactly,
    without a peak search: one aperture fixed in angle per band must have a
    centre independent of pixelization. ``centre="peak"`` follows the PSF's
    refined local peak instead.

    Attributes:
        radius_lod: Circle radius in lambda/D units of the Image's frame. Must be
            finite and positive.
        centre: Center convention, ``"source"`` (the supplied source position) or
            ``"peak"`` (the refined local PSF peak).
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
        """Fractional native-pixel coverage, using exact circle-square overlaps.

        Args:
            psf: Per-pixel source-flux Image the weights are laid on.
            source_position: Source position (x, y) in lambda/D units of the Image's
                frame.
            oversample: Subpixel sampling factor per axis used to refine the peak and
                the aperture edge.

        Returns:
            Fractional native-pixel coverage on the PSF Image's grid.
        """
        from .performance import _rule_weights

        return _rule_weights(self, psf, source_position, oversample)


#: The photometric core used when a caller names no rule: a 0.7 lambda/D circle.
DEFAULT_APERTURE_RULE = Circular(0.7)


@dataclass(frozen=True)
class HalfMaximum:
    """Region about the PSF peak by construction, above ``fraction`` of it.

    Only the connected component of the refined source-local peak is used.

    Attributes:
        fraction: Level, as a fraction of the refined local peak, above which the
            region is kept. Strictly between zero and one.
    """

    fraction: float = 0.5

    def __post_init__(self):
        """Validate the static rule parameter."""
        if not math.isfinite(self.fraction) or not 0 < self.fraction < 1:
            raise ValueError("fraction must be between zero and one")

    def weights(self, psf, source_position, *, oversample=10):
        """Fractional native-pixel coverage from the band-limited level set.

        Args:
            psf: Per-pixel source-flux Image the weights are laid on.
            source_position: Source position (x, y) in lambda/D units of the Image's
                frame.
            oversample: Subpixel sampling factor per axis used to refine the peak and
                the level set.

        Returns:
            Fractional native-pixel coverage on the PSF Image's grid.
        """
        from .performance import _rule_weights

        return _rule_weights(self, psf, source_position, oversample)


ApertureRule = Circular | HalfMaximum
