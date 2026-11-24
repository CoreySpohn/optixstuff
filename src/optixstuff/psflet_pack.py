"""PSFlet template pack: the array waist between a PSFlet generator and a renderer.

A pack tabulates the pixel-integrated PSFlet response on a fine sub-pixel
offset grid, per field anchor and per wavelength: ``templates[f, l, iy, ix]``
is the flux falling in a unit detector pixel whose center sits at offset
``(dy, dx) = (offsets[iy], offsets[ix])`` px from the PSFlet centroid. A renderer
places it at the dispersion-model centroid plus the optional ``centroids``
correction. A measured pack can use its wavelength-calibrated centroid even
when the template has a nonzero first moment. ``LensletDisperser.psflet_pack_path``
points at the pack file.

The npz format requires ``format_version`` as the scalar integer ``2``.

- ``templates``: ``(n_field, n_lam, n_off, n_off)`` template planes.
- ``offsets``: ``(n_off,)`` uniform ascending pixel-offset axis, shared by the
  dy (rows) and dx (columns) axes of each plane.
- ``wavelengths_nm``: ``(n_lam,)`` ascending tabulation wavelengths.
- ``field_xy``: ``(n_field, 2)`` field-anchor positions in lenslet-index
  coordinates; each lenslet uses its nearest anchor's templates.
- ``template_origin``: required string declaring the template origin.
  ``"centroid"`` is the only origin supported by format 2.
- ``centroids`` (optional): ``(n_field, n_lam, 2)`` per-anchor ``(dx, dy)``
  pixel corrections to the dispersion-model centroid (the wavecal residual).
- ``meta_json`` (optional): provenance string, not interpreted.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import HostGrid
from jax import Array

from optixstuff.spectral import as_knots

PACK_FORMAT_VERSION = 2
# The only declared template origin in format 2.
TEMPLATE_ORIGIN = "centroid"


def _pack_grid(wavelengths_nm) -> HostGrid:
    """Return the static wavelength grid of a pack.

    Two or more wavelengths are validated :class:`~optixstuff.spectral.Knots`.
    A single wavelength is a one-point ``HostGrid`` (a pack tabulated at one
    wavelength is allowed: its template is read at every wavelength).
    """
    n = (
        len(wavelengths_nm)
        if isinstance(wavelengths_nm, HostGrid)
        else np.size(wavelengths_nm)
    )
    if n != 1:
        return as_knots(wavelengths_nm)
    grid = HostGrid(wavelengths_nm)
    if grid.values[0] <= 0:
        raise ValueError("the PSFlet wavelength must be positive")
    return grid


def _optional_asarray(x):
    return None if x is None else jnp.asarray(x)


class PsfletPack(eqx.Module):
    """Tabulated pixel-integrated PSFlet templates (see the module docstring).

    Attributes:
        templates: Template planes, shape ``(n_field, n_lam, n_off, n_off)``.
        offsets: Uniform ascending pixel-offset axis, shape ``(n_off,)``.
        wavelengths_nm: Ascending tabulation wavelengths in nm, shape ``(n_lam,)``,
            a static grid: :class:`~optixstuff.spectral.Knots` (each at least
            the wavelength tolerance from the next), or a one-point
            ``HostGrid`` for a pack tabulated at one wavelength; ``.values``
            is the float64 array and ``.array()`` a JAX constant.
        field_xy: Field-anchor positions in lenslet-index coordinates, shape
            ``(n_field, 2)``.
        template_origin: Required origin declaration, equal to ``TEMPLATE_ORIGIN``.
        centroids: Optional ``(dx, dy)`` pixel corrections to the dispersion-model
            centroid, shape ``(n_field, n_lam, 2)``.
        meta_json: Provenance string, not interpreted.
    """

    templates: Array = eqx.field(converter=jnp.asarray)
    offsets: Array = eqx.field(converter=jnp.asarray)
    wavelengths_nm: HostGrid = eqx.field(static=True, converter=_pack_grid)
    field_xy: Array = eqx.field(converter=jnp.asarray)
    template_origin: str = eqx.field(static=True)
    centroids: Array | None = eqx.field(converter=_optional_asarray, default=None)
    meta_json: str = eqx.field(static=True, default="")

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the wavelengths where template and centroid interpolation changes.

        Interpolation is linear between the tabulated wavelengths, so they are
        the breaks. A one-wavelength pack is constant, with no change to mark.

        Returns:
            The tabulation wavelengths in nm; empty for a one-wavelength pack.
        """
        if len(self.wavelengths_nm) == 1:
            return np.empty(0)
        return self.wavelengths_nm.values

    def __check_init__(self):
        """Validate shapes and grid conventions at construction (eager only)."""
        if self.template_origin != TEMPLATE_ORIGIN:
            raise ValueError(
                f"unsupported PSFlet template origin {self.template_origin!r}. "
                "For legacy origin-relative templates, regenerate the pack"
            )
        if (
            self.templates.ndim != 4
            or self.templates.shape[2] != self.templates.shape[3]
        ):
            raise ValueError(
                "templates must be (n_field, n_lam, n_off, n_off) with square planes"
            )
        n_field, n_lam, n_off, _ = self.templates.shape
        if self.offsets.shape != (n_off,):
            raise ValueError("offsets must match the template plane size")
        steps = jnp.diff(self.offsets)
        if not bool(jnp.all(steps > 0)):
            raise ValueError("offsets must be ascending")
        if not bool(jnp.allclose(steps, steps[0], rtol=1e-6, atol=0.0)):
            raise ValueError("offsets must be uniform")
        if len(self.wavelengths_nm) != n_lam:
            raise ValueError("wavelengths_nm must match the template plane count")
        if self.field_xy.shape != (n_field, 2):
            raise ValueError("field_xy must be (n_field, 2)")
        if self.centroids is not None and self.centroids.shape != (n_field, n_lam, 2):
            raise ValueError("centroids must be (n_field, n_lam, 2)")


def save_psflet_pack(path, pack):
    """Write a pack to the documented npz format.

    Args:
        path: Destination of the npz file.
        pack: Pack to write.
    """
    data = dict(
        format_version=np.int64(PACK_FORMAT_VERSION),
        templates=np.asarray(pack.templates),
        offsets=np.asarray(pack.offsets),
        wavelengths_nm=pack.wavelengths_nm.values,
        field_xy=np.asarray(pack.field_xy),
        template_origin=np.asarray(pack.template_origin),
    )
    if pack.centroids is not None:
        data["centroids"] = np.asarray(pack.centroids)
    if pack.meta_json:
        data["meta_json"] = np.asarray(pack.meta_json)
    np.savez(path, **data)


def load_psflet_pack(path):
    """Load a pack written by :func:`save_psflet_pack` or a conforming emitter.

    Args:
        path: Path of the npz file.

    Returns:
        The pack read from the file.
    """
    with np.load(path, allow_pickle=False) as data:
        version = None
        if "format_version" in data:
            version_tag = data["format_version"]
            if version_tag.ndim != 0 or version_tag.dtype.kind not in "iu":
                raise ValueError("PSFlet pack format_version must be a scalar integer")
            version = int(version_tag)
        if version != PACK_FORMAT_VERSION:
            if version == 1:
                raise ValueError(
                    "unsupported PSFlet pack format 1: the template convention "
                    "changed, regenerate the pack"
                )
            raise ValueError(
                f"unsupported PSFlet pack format {version!r}; this reader "
                f"supports format {PACK_FORMAT_VERSION}"
            )
        if "template_origin" not in data:
            raise ValueError(
                "PSFlet pack declares no template origin. Please regenerate the pack"
            )
        return PsfletPack(
            templates=data["templates"],
            offsets=data["offsets"],
            wavelengths_nm=data["wavelengths_nm"],
            field_xy=data["field_xy"],
            template_origin=str(data["template_origin"]),
            centroids=data["centroids"] if "centroids" in data else None,
            meta_json=str(data["meta_json"]) if "meta_json" in data else "",
        )
