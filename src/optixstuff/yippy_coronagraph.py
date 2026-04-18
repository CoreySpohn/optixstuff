"""YippyCoronagraph -- AbstractCoronagraph backed by a yippy EqxCoronagraph."""


from typing import final

from jax.typing import ArrayLike
from jaxtyping import Array
from yippy import EqxCoronagraph

from optixstuff.coronagraph import AbstractCoronagraph


@final
class YippyCoronagraph(AbstractCoronagraph):
    """Coronagraph performance model backed by a yippy YIP interpolation table.

    Wraps a yippy ``EqxCoronagraph`` via composition, adapting its methods
    to the ``AbstractCoronagraph`` interface.  The ``_backend`` field is
    itself an ``eqx.Module``, so its internal JAX arrays flow through
    ``filter_jit`` and ``filter_grad`` normally.

    Construction mirrors ``EqxCoronagraph`` -- pass either a YIP path or
    an existing ``EqxCoronagraph`` instance::

        coro = YippyCoronagraph("/path/to/yip")
        coro = YippyCoronagraph(backend=existing_eqx_coro)
    """

    _backend: EqxCoronagraph

    def __init__(
        self,
        yip_path: str | None = None,
        *,
        backend: EqxCoronagraph | None = None,
        **kwargs,
    ) -> None:
        """Create a YippyCoronagraph from a YIP path or existing backend.

        Args:
            yip_path: Path to a Yield Input Package directory.
            backend: Pre-built EqxCoronagraph.  Takes precedence over yip_path.
            **kwargs: Forwarded to ``EqxCoronagraph`` when building from yip_path.
        """
        if backend is not None:
            self._backend = backend
        elif yip_path is not None:
            self._backend = EqxCoronagraph(yip_path, **kwargs)
        else:
            msg = "Provide either yip_path or backend"
            raise ValueError(msg)

    # -- AbstractVar fields satisfied via properties ----------------------

    @property
    def pixel_scale_lod(self) -> float:
        """Native pixel scale in lambda/D per pixel."""
        return self._backend.pixel_scale_lod

    @property
    def IWA(self) -> float:
        """Inner working angle in lambda/D."""
        return self._backend.IWA

    @property
    def OWA(self) -> float:
        """Outer working angle in lambda/D."""
        return self._backend.OWA

    # -- Scalar interface -------------------------------------------------

    def throughput(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Core throughput from the YIP interpolation table."""
        return self._backend.throughput(separation_lod)

    def core_area(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Photometric aperture area from the YIP interpolation table."""
        return self._backend.core_area(separation_lod)

    def core_mean_intensity(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Mean stellar leakage from the YIP interpolation table."""
        return self._backend.core_mean_intensity(separation_lod)

    def occulter_transmission(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Sky transmission from the YIP interpolation table."""
        return self._backend.occulter_transmission(separation_lod)

    # -- Image interface --------------------------------------------------

    def on_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """Stellar leakage PSF from the YIP stellar intensity model."""
        return self._backend.stellar_intens(0.0)

    def off_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        separation_lod: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """Off-axis planet PSF from the YIP PSF interpolator.

        Places the planet along the +x axis by convention.
        """
        return self._backend.create_psf(separation_lod, 0.0, npixels)

    # -- Convenience methods (not on AbstractCoronagraph) ------------------

    def noise_floor_ayo(
        self,
        separation_lod: ArrayLike,
        ppf: float = 30.0,
    ) -> ArrayLike:
        """AYO noise floor: core_mean_intensity / ppf.

        This is a convenience passthrough to the backend. Not part of the
        AbstractCoronagraph contract -- downstream ETCs should compute
        noise floors as pure functions.
        """
        return self._backend.noise_floor_ayo(separation_lod, ppf)

    def raw_contrast(self, separation_lod: ArrayLike) -> ArrayLike:
        """Raw contrast from the YIP interpolation table."""
        return self._backend.raw_contrast(separation_lod)

    def stellar_intens(self, stellar_diam_lod: float) -> Array:
        """Stellar intensity map for a given stellar angular diameter."""
        return self._backend.stellar_intens(stellar_diam_lod)

    @property
    def psf_shape(self) -> tuple[int, int]:
        """Shape of the PSF arrays from the YIP file."""
        return self._backend.psf_shape

    @property
    def sky_trans(self) -> Array:
        """Full sky transmission map."""
        return self._backend.sky_trans

    def create_psfs(self, x_lod: ArrayLike, y_lod: ArrayLike) -> Array:
        """Batched off-axis PSFs at (x_lod, y_lod) source positions.

        Delegates to the backend yippy create_psfs closure. Returns
        a stack of PSF images, one per input source coordinate.

        Args:
            x_lod: Source x-coordinates in lambda/D, shape (K,).
            y_lod: Source y-coordinates in lambda/D, shape (K,).

        Returns:
            PSF stack of shape (K, ny, nx) where (ny, nx) == self.psf_shape.
        """
        return self._backend.create_psfs(x_lod, y_lod)

    @property
    def psf_datacube(self) -> Array | None:
        """Pre-computed quarter-symmetric PSF datacube from the backend.

        Returns None if the backing EqxCoronagraph was not built
        with ensure_psf_datacube=True. Consumers that need this for
        disk convolution should construct the backend with the flag set.
        """
        return self._backend.psf_datacube
