"""Coronagraph abstractions."""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from hwoutils.transforms import resample_flux
from jax.typing import ArrayLike
from jaxtyping import Array


def _convolve_quadrants(flux, psf_datacube):
    """Convolve flux with a quarter-symmetric PSF datacube via fold-and-sum.

    Handles padding dynamically to ensure all quadrants match the shape of the
    first quadrant (which defines the PSF datacube shape).
    """
    ny, nx = flux.shape
    cy, cx = (ny - 1) // 2, (nx - 1) // 2

    # Q1: top-right (includes center pixel and axes) -> reference shape
    q1 = flux[cy:, cx:]
    target_h, target_w = q1.shape

    # Q2: top-left -- flip X, pad inner-left + outer-right to target width
    q2_raw = flux[cy:, :cx]
    q2_flipped = q2_raw[:, ::-1]
    pad_q2_right = max(0, target_w - (q2_flipped.shape[1] + 1))
    q2 = jnp.pad(q2_flipped, ((0, 0), (1, pad_q2_right)))

    # Q3: bottom-left -- flip both, pad inner-top, inner-left, outer-bottom/right
    q3_raw = flux[:cy, :cx]
    q3_flipped = q3_raw[::-1, ::-1]
    pad_q3_bottom = max(0, target_h - (q3_flipped.shape[0] + 1))
    pad_q3_right = max(0, target_w - (q3_flipped.shape[1] + 1))
    q3 = jnp.pad(q3_flipped, ((1, pad_q3_bottom), (1, pad_q3_right)))

    # Q4: bottom-right -- flip Y, pad inner-top + outer-bottom
    q4_raw = flux[:cy, cx:]
    q4_flipped = q4_raw[::-1, :]
    pad_q4_bottom = max(0, target_h - (q4_flipped.shape[0] + 1))
    q4 = jnp.pad(q4_flipped, ((1, pad_q4_bottom), (0, 0)))

    flux_stack = jnp.stack([q1, q2, q3, q4])
    partial_images = jnp.einsum("qij,ijxy->qxy", flux_stack, psf_datacube)

    img_q1 = partial_images[0]
    img_q2 = jnp.fliplr(partial_images[1])
    img_q3 = jnp.flipud(jnp.fliplr(partial_images[2]))
    img_q4 = jnp.flipud(partial_images[3])

    return img_q1 + img_q2 + img_q3 + img_q4


class AbstractCoronagraph(eqx.Module):
    """Abstract interface for coronagraph performance models.

    Provides scalar performance curves (for ETC use), single-PSF
    generation, and a sampling-explicit image contract (for image
    simulation): ``stellar_map`` / ``source_psfs`` /
    ``background_transmission`` / ``extended_scene`` serve maps at the
    caller's requested pixel scale and shape, so backend-internal grids
    never leak downstream. Implementations can be backed by pre-computed
    interpolation tables (subclass :class:`AbstractTableCoronagraph`,
    which serves the image contract from native-grid tables), physical
    wavefront propagation, or analytical models.

    All wavelength arguments are in nanometres throughout.
    All separations and pixel scales are in lambda/D units -- the
    contract is dimensionless optics; angle conversions are the
    telescope primary's business.
    """

    pixel_scale_lod: AbstractVar[float]
    """Native pixel scale in lambda/D per pixel."""

    IWA: AbstractVar[float]
    """Inner working angle in lambda/D."""

    OWA: AbstractVar[float]
    """Outer working angle in lambda/D."""

    # ------------------------------------------------------------------
    # Scalar interface -- consumed by jaxEDITH and yield estimators
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def throughput(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Core (off-axis planet) throughput.

        Args:
            separation_lod: Angular separation in lambda/D.
            wavelength_nm: Wavelength in nanometres.
            time_s: Time since mission start in seconds.

        Returns:
            Fractional throughput in [0, 1].
        """
        ...

    @abc.abstractmethod
    def core_area(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Photometric aperture area in (lambda/D)^2.

        Args:
            separation_lod: Angular separation in lambda/D.
            wavelength_nm: Wavelength in nanometres.
            time_s: Time since mission start in seconds.

        Returns:
            Core area in (lambda/D)^2.
        """
        ...

    @abc.abstractmethod
    def core_mean_intensity(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Mean stellar intensity within the photometric aperture.

        Args:
            separation_lod: Angular separation in lambda/D.
            wavelength_nm: Wavelength in nanometres.
            time_s: Time since mission start in seconds.

        Returns:
            Mean stellar leakage intensity in (lambda/D)^-2.
        """
        ...

    @abc.abstractmethod
    def occulter_transmission(
        self,
        separation_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        time_s: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Off-axis (sky/zodi) transmission through the occulter.

        Args:
            separation_lod: Angular separation in lambda/D.
            wavelength_nm: Wavelength in nanometres.
            time_s: Time since mission start in seconds.

        Returns:
            Fractional sky transmission in [0, 1].
        """
        ...

    # ------------------------------------------------------------------
    # Single-PSF image interface
    # ------------------------------------------------------------------
    #
    # Subsumed by the sampling-explicit contract below (``stellar_map``
    # additionally carries the stellar diameter; ``source_psfs`` the 2D
    # position) but retained for existing single-PSF consumers.

    @abc.abstractmethod
    def on_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """On-axis (stellar leakage) PSF.

        Returns the coronagraphic PSF for an on-axis point source,
        normalized to unit stellar flux before the coronagraph. For a
        finite stellar diameter use :meth:`stellar_map`, which carries
        the diameter axis this signature lacks.

        Args:
            wavelength_nm: Wavelength in nanometres.
            pixel_scale_rad: Output pixel scale in radians/pixel.
            npixels: Output array side length in pixels. Must be a
                Python int (not a JAX array) as it determines the
                output shape at compile time.

        Returns:
            2D float array of shape (npixels, npixels).
        """
        ...

    @abc.abstractmethod
    def off_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        separation_lod: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """Off-axis PSF at a given angular separation.

        Args:
            wavelength_nm: Wavelength in nanometres.
            separation_lod: Source separation in lambda/D.
            pixel_scale_rad: Output pixel scale in radians/pixel.
            npixels: Output array side length in pixels. Must be a
                Python int (not a JAX array) as it determines the
                output shape at compile time.

        Returns:
            2D float array of shape (npixels, npixels).
        """
        ...

    # ------------------------------------------------------------------
    # Sampling-explicit image contract -- consumed by coronagraphoto
    # ------------------------------------------------------------------
    #
    # Maps are served AT the requested target sampling (in practice the
    # detector grid), so callers never see a backend's internal grid.

    @abc.abstractmethod
    def stellar_map(
        self,
        wavelength_nm: ArrayLike,
        stellar_diam_lod: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
    ) -> Array:
        """On-axis stellar leakage map at the requested sampling.

        Args:
            wavelength_nm: Wavelength in nanometres.
            stellar_diam_lod: Stellar angular diameter in lambda/D.
            pixel_scale_lod: Target pixel scale in lambda/D per pixel.
            shape: Target array shape (ny, nx).

        Returns:
            Map of shape ``shape`` in fraction of pre-coronagraph
            stellar photons per target pixel.
        """
        ...

    @abc.abstractmethod
    def source_psfs(
        self,
        wavelength_nm: ArrayLike,
        x_lod: ArrayLike,
        y_lod: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
    ) -> Array:
        """Off-axis point-source PSFs at the requested sampling.

        Args:
            wavelength_nm: Wavelength in nanometres.
            x_lod: Source x-coordinates in lambda/D, shape (K,).
            y_lod: Source y-coordinates in lambda/D, shape (K,).
            pixel_scale_lod: Target pixel scale in lambda/D per pixel.
            shape: Target array shape (ny, nx).

        Returns:
            Stack of shape (K, ny, nx) in fraction of source photons
            per target pixel.
        """
        ...

    @abc.abstractmethod
    def background_transmission(
        self,
        wavelength_nm: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
    ) -> Array:
        """Per-pixel transmission map for a spatially uniform background.

        Args:
            wavelength_nm: Wavelength in nanometres.
            pixel_scale_lod: Target pixel scale in lambda/D per pixel.
            shape: Target array shape (ny, nx).

        Returns:
            Dimensionless transmission map of shape ``shape`` (value
            semantics, not a flux).
        """
        ...

    @abc.abstractmethod
    def extended_scene(
        self,
        scene_map: Array,
        map_pixel_scale_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
        rotation_deg: ArrayLike = 0.0,
    ) -> Array:
        """Render an extended incoherent scene through the coronagraph.

        Args:
            scene_map: Per-pixel source intensity in any units; the
                output carries the same units per target pixel.
            map_pixel_scale_lod: Pixel scale of ``scene_map`` in
                lambda/D per pixel.
            wavelength_nm: Wavelength in nanometres.
            pixel_scale_lod: Target pixel scale in lambda/D per pixel.
            shape: Target array shape (ny, nx).
            rotation_deg: CCW rotation applied when mapping the scene
                into the coronagraph frame (sky-to-detector roll).

        Returns:
            The scene redistributed by the per-position off-axis PSFs,
            shape ``shape``.
        """
        ...


class AbstractTableCoronagraph(AbstractCoronagraph):
    """Base for coronagraphs backed by native-grid interpolation tables.

    Declares the table SPI -- the native-grid maps a table backend holds
    (``stellar_intens``, ``create_psfs``, ``sky_trans``, ``psf_shape``,
    ``psf_datacube``) -- and serves the sampling-explicit image contract
    from it with flux-conserving resamples to the requested target grid.
    Scalar performance curves remain the subclass's business (a table
    backend interpolates its own curve tables).

    The tables are monochromatic by construction (a YIP is a per-band
    file), so the served maps ignore ``wavelength_nm`` beyond the
    caller's own lambda/D sampling conversion; stack per-band instances
    in a :class:`MultiBandCoronagraph` for a broadband model.
    """

    # -- table SPI ---------------------------------------------------------
    # AbstractVar members may be satisfied by a dataclass field or a
    # property (a concrete base property would shadow subclass fields).

    psf_shape: AbstractVar[tuple]
    """Native map shape (ny, nx) shared by the table members."""

    sky_trans: AbstractVar[Array]
    """Native-grid transmission map for a spatially uniform background."""

    psf_datacube: AbstractVar[Array | None]
    """Optional per-source-position PSF library for extended scenes.

    ``None`` means the table was built without one; :meth:`extended_scene`
    then raises. Full ``(ny, nx, ny, nx)`` and quarter-symmetric
    ``(ny//2+1, nx//2+1, ny, nx)`` layouts are supported.
    """

    @abc.abstractmethod
    def stellar_intens(self, stellar_diam_lod: ArrayLike) -> Array:
        """Native-grid stellar leakage map for a stellar angular diameter.

        The diameter axis is a genuine table capability (a YIP tabulates
        finite-size stellar maps); implementations interpolate over it.
        """
        ...

    @abc.abstractmethod
    def create_psfs(self, x_lod: ArrayLike, y_lod: ArrayLike) -> Array:
        """Native-grid off-axis PSF stack at (x_lod, y_lod), shape (K, ny, nx)."""
        ...

    # -- single-PSF interface served from the tables -----------------------

    def on_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """Point-source stellar leakage map on the native grid.

        Table backends serve their native sampling; the requested
        ``pixel_scale_rad`` / ``npixels`` are accepted for interface
        conformance only. Use :meth:`stellar_map` for target-sampled
        maps and finite stellar diameters.
        """
        return self.stellar_intens(0.0)

    def off_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        separation_lod: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """Off-axis PSF at +x separation on the native grid.

        Sampling arguments are accepted for interface conformance only;
        use :meth:`source_psfs` for target-sampled maps.
        """
        return self.create_psfs(jnp.atleast_1d(separation_lod), jnp.zeros(1))[0]

    # -- sampling-explicit contract served from the tables -----------------

    def stellar_map(
        self,
        wavelength_nm: ArrayLike,
        stellar_diam_lod: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
    ) -> Array:
        """Native ``stellar_intens`` resampled (flux-conserving) to target."""
        return resample_flux(
            self.stellar_intens(stellar_diam_lod),
            self.pixel_scale_lod,
            pixel_scale_lod,
            shape,
            0.0,
        )

    def source_psfs(
        self,
        wavelength_nm: ArrayLike,
        x_lod: ArrayLike,
        y_lod: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
    ) -> Array:
        """Native ``create_psfs`` stack resampled (flux-conserving) to target."""

        def _to_target(psf):
            return resample_flux(psf, self.pixel_scale_lod, pixel_scale_lod, shape, 0.0)

        return jax.vmap(_to_target)(self.create_psfs(x_lod, y_lod))

    def background_transmission(
        self,
        wavelength_nm: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
    ) -> Array:
        """Native ``sky_trans`` resampled to target with value semantics.

        A transmission is a per-pixel value, not a flux, so the
        pixel-area scaling of the flux-conserving resample is undone.
        """
        resampled = resample_flux(
            self.sky_trans, self.pixel_scale_lod, pixel_scale_lod, shape, 0.0
        )
        return resampled * (self.pixel_scale_lod / pixel_scale_lod) ** 2

    def extended_scene(
        self,
        scene_map: Array,
        map_pixel_scale_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
        rotation_deg: ArrayLike = 0.0,
    ) -> Array:
        """Scene convolved with the ``psf_datacube`` PSF library.

        Resamples the scene to the native source grid (applying the
        sky-to-detector rotation), convolves via the full or
        quarter-symmetric datacube, and resamples the result to the
        target grid.

        Raises:
            ValueError: if :attr:`psf_datacube` is ``None``, or its
                source grid matches neither the full nor the quarter
                PSF shape.
        """
        datacube = self.psf_datacube
        if datacube is None:
            raise ValueError(
                "extended_scene requires a coronagraph with a PSF "
                f"datacube; got {type(self).__name__}.psf_datacube=None. "
                "The table implementation convolves the scene with the "
                "per-source-position PSFs and cannot run without it."
            )
        ny, nx = self.psf_shape
        src = resample_flux(
            scene_map,
            map_pixel_scale_lod,
            self.pixel_scale_lod,
            (ny, nx),
            rotation_deg,
        )
        n_src_y, n_src_x = datacube.shape[:2]
        if (n_src_y, n_src_x) == (ny, nx):
            native = jnp.einsum("ij,ijxy->xy", src, datacube)
        elif (n_src_y, n_src_x) == (ny // 2 + 1, nx // 2 + 1):
            native = _convolve_quadrants(src, datacube)
        else:
            raise ValueError(
                "extended_scene: psf_datacube source-grid shape "
                f"({n_src_y}, {n_src_x}) does not match either the full "
                f"PSF shape ({ny}, {nx}) or the quarter PSF shape "
                f"({ny // 2 + 1}, {nx // 2 + 1}). Coronagraphs must "
                "publish a full or quarter datacube."
            )
        return resample_flux(native, self.pixel_scale_lod, pixel_scale_lod, shape, 0.0)


class AbstractScalarCoronagraph(AbstractCoronagraph):
    """Base for ETC-only coronagraph models that lack 2D PSF generation.

    Stubs out the single-PSF interface and the sampling-explicit image
    contract with zero arrays so the class satisfies AbstractCoronagraph
    without requiring a full optical model. Do NOT use this base for a
    model that carries native-grid tables -- subclass
    :class:`AbstractTableCoronagraph` instead, or the image pipeline
    will silently render zeros.
    """

    def on_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """Return a zero PSF (not implemented for scalar-only models)."""
        return jnp.zeros((npixels, npixels))

    def off_axis_psf(
        self,
        wavelength_nm: ArrayLike,
        separation_lod: ArrayLike,
        pixel_scale_rad: float,
        npixels: int,
    ) -> Array:
        """Return a zero PSF (not implemented for scalar-only models)."""
        return jnp.zeros((npixels, npixels))

    def stellar_map(
        self,
        wavelength_nm: ArrayLike,
        stellar_diam_lod: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
    ) -> Array:
        """Return a zero map (not implemented for scalar-only models)."""
        return jnp.zeros(shape)

    def source_psfs(
        self,
        wavelength_nm: ArrayLike,
        x_lod: ArrayLike,
        y_lod: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
    ) -> Array:
        """Return zero PSFs (not implemented for scalar-only models)."""
        k = jnp.atleast_1d(jnp.asarray(x_lod)).shape[0]
        return jnp.zeros((k, *shape))

    def background_transmission(
        self,
        wavelength_nm: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
    ) -> Array:
        """Return a zero map (not implemented for scalar-only models)."""
        return jnp.zeros(shape)

    def extended_scene(
        self,
        scene_map: Array,
        map_pixel_scale_lod: ArrayLike,
        wavelength_nm: ArrayLike,
        *,
        pixel_scale_lod: ArrayLike,
        shape: tuple[int, int],
        rotation_deg: ArrayLike = 0.0,
    ) -> Array:
        """Return a zero map (not implemented for scalar-only models)."""
        return jnp.zeros(shape)


class MultiBandCoronagraph(AbstractCoronagraph):
    """A broadband coronagraph as a stack of per-band models.

    Every contract call dispatches to the band whose declared center is
    nearest the requested ``wavelength_nm`` and forwards the call
    unchanged. Because the contract is sampling-explicit in lambda/D,
    chromatic magnification needs no handling here: the caller already
    converts its detector grid to lambda/D at each wavelength.

    The per-band models can be anything implementing
    ``AbstractCoronagraph`` -- per-band YIP tables
    (:class:`optixstuff.YippyCoronagraph`) or per-wavelength compiled
    views of a live propagation model -- and can be mixed.

    ``wavelength_nm`` must be concrete (a Python/NumPy scalar, not a
    traced JAX value) at dispatch time: bands are distinct pytrees with
    distinct shapes, so the selection is Python control flow. This
    matches the per-bin structure of the rate pipeline, where the
    wavelength is a static observation parameter.

    Scalar metadata (``pixel_scale_lod`` / ``IWA`` / ``OWA``) is served
    from the band nearest ``reference_wavelength_nm`` (default: the
    first band).
    """

    band_centers_nm: tuple = eqx.field(static=True)
    bands: tuple
    reference_wavelength_nm: float = eqx.field(static=True)

    def __init__(self, band_centers_nm, bands, *, reference_wavelength_nm=None):
        """Build a multi-band stack.

        Args:
            band_centers_nm: Band center wavelengths in nanometres, one
                per band.
            bands: The per-band ``AbstractCoronagraph`` models, aligned
                with ``band_centers_nm``.
            reference_wavelength_nm: Wavelength whose band serves the
                scalar metadata. Default: the first band's center.
        """
        centers = tuple(float(c) for c in band_centers_nm)
        bands = tuple(bands)
        if len(centers) != len(bands):
            raise ValueError(
                f"{len(centers)} band centers but {len(bands)} band models"
            )
        if not bands:
            raise ValueError("MultiBandCoronagraph needs at least one band")
        self.band_centers_nm = centers
        self.bands = bands
        self.reference_wavelength_nm = float(
            centers[0] if reference_wavelength_nm is None else reference_wavelength_nm
        )

    def band(self, wavelength_nm) -> AbstractCoronagraph:
        """The band model whose center is nearest ``wavelength_nm``."""
        wl = float(wavelength_nm)
        index = min(
            range(len(self.band_centers_nm)),
            key=lambda i: abs(self.band_centers_nm[i] - wl),
        )
        return self.bands[index]

    # -- scalar metadata from the reference band ---------------------------

    @property
    def pixel_scale_lod(self) -> float:
        """Native pixel scale of the reference band."""
        return self.band(self.reference_wavelength_nm).pixel_scale_lod

    @property
    def IWA(self) -> float:
        """Inner working angle of the reference band."""
        return self.band(self.reference_wavelength_nm).IWA

    @property
    def OWA(self) -> float:
        """Outer working angle of the reference band."""
        return self.band(self.reference_wavelength_nm).OWA

    # -- every interface dispatches on wavelength --------------------------

    def throughput(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Core throughput from the nearest band."""
        return self.band(wavelength_nm).throughput(
            separation_lod, wavelength_nm, time_s=time_s
        )

    def core_area(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Photometric aperture area from the nearest band."""
        return self.band(wavelength_nm).core_area(
            separation_lod, wavelength_nm, time_s=time_s
        )

    def core_mean_intensity(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Mean stellar leakage from the nearest band."""
        return self.band(wavelength_nm).core_mean_intensity(
            separation_lod, wavelength_nm, time_s=time_s
        )

    def occulter_transmission(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Sky transmission from the nearest band."""
        return self.band(wavelength_nm).occulter_transmission(
            separation_lod, wavelength_nm, time_s=time_s
        )

    def on_axis_psf(self, wavelength_nm, pixel_scale_rad, npixels):
        """On-axis PSF from the nearest band."""
        return self.band(wavelength_nm).on_axis_psf(
            wavelength_nm, pixel_scale_rad, npixels
        )

    def off_axis_psf(self, wavelength_nm, separation_lod, pixel_scale_rad, npixels):
        """Off-axis PSF from the nearest band."""
        return self.band(wavelength_nm).off_axis_psf(
            wavelength_nm, separation_lod, pixel_scale_rad, npixels
        )

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, pixel_scale_lod, shape):
        """Stellar leakage map from the nearest band."""
        return self.band(wavelength_nm).stellar_map(
            wavelength_nm,
            stellar_diam_lod,
            pixel_scale_lod=pixel_scale_lod,
            shape=shape,
        )

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, pixel_scale_lod, shape):
        """Off-axis PSF stack from the nearest band."""
        return self.band(wavelength_nm).source_psfs(
            wavelength_nm, x_lod, y_lod, pixel_scale_lod=pixel_scale_lod, shape=shape
        )

    def background_transmission(self, wavelength_nm, *, pixel_scale_lod, shape):
        """Background transmission map from the nearest band."""
        return self.band(wavelength_nm).background_transmission(
            wavelength_nm, pixel_scale_lod=pixel_scale_lod, shape=shape
        )

    def extended_scene(
        self,
        scene_map,
        map_pixel_scale_lod,
        wavelength_nm,
        *,
        pixel_scale_lod,
        shape,
        rotation_deg=0.0,
    ):
        """Extended-scene render from the nearest band."""
        return self.band(wavelength_nm).extended_scene(
            scene_map,
            map_pixel_scale_lod,
            wavelength_nm,
            pixel_scale_lod=pixel_scale_lod,
            shape=shape,
            rotation_deg=rotation_deg,
        )
