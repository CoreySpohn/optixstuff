"""PSFlet template pack: the array waist between a PSFlet generator and a renderer.

A pack tabulates the pixel-integrated PSFlet response on a fine sub-pixel
offset grid, per field anchor and per wavelength: ``templates[f, l, iy, ix]``
is the flux falling in a unit detector pixel whose center sits at offset
``(dy, dx) = (offsets[iy], offsets[ix])`` px from the PSFlet centroid. A
wave-optics generator writes the npz format below and a spectrograph model
reads it; ``LensletDisperser.psflet_pack_path`` points at such a file.

npz format, ``format_version = 1``:

- ``templates``: ``(n_field, n_lam, n_off, n_off)`` template planes.
- ``offsets``: ``(n_off,)`` uniform ascending pixel-offset axis, shared by the
  dy (rows) and dx (columns) axes of each plane.
- ``wavelengths_nm``: ``(n_lam,)`` ascending tabulation wavelengths.
- ``field_xy``: ``(n_field, 2)`` field-anchor positions in lenslet-index
  coordinates; each lenslet uses its nearest anchor's templates.
- ``centroids`` (optional): ``(n_field, n_lam, 2)`` per-anchor ``(dx, dy)``
  pixel corrections to the dispersion-model centroid (the wavecal residual).
- ``meta_json`` (optional): provenance string, not interpreted.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

PACK_FORMAT_VERSION = 1


def _optional_asarray(x):
    return None if x is None else jnp.asarray(x)


class PsfletPack(eqx.Module):
    """Tabulated pixel-integrated PSFlet templates (see the module docstring)."""

    templates: Array = eqx.field(converter=jnp.asarray)
    offsets: Array = eqx.field(converter=jnp.asarray)
    wavelengths_nm: Array = eqx.field(converter=jnp.asarray)
    field_xy: Array = eqx.field(converter=jnp.asarray)
    centroids: Array | None = eqx.field(converter=_optional_asarray, default=None)
    meta_json: str = eqx.field(static=True, default="")

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the wavelengths where template and centroid interpolation changes."""
        return np.asarray(self.wavelengths_nm)

    def __check_init__(self):
        """Validate shapes and grid conventions at construction (eager only)."""
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
        if self.wavelengths_nm.shape != (n_lam,):
            raise ValueError("wavelengths_nm must match the template plane count")
        if n_lam > 1 and not bool(jnp.all(jnp.diff(self.wavelengths_nm) > 0)):
            raise ValueError("wavelengths_nm must be ascending")
        if self.field_xy.shape != (n_field, 2):
            raise ValueError("field_xy must be (n_field, 2)")
        if self.centroids is not None and self.centroids.shape != (n_field, n_lam, 2):
            raise ValueError("centroids must be (n_field, n_lam, 2)")


def save_psflet_pack(path, pack):
    """Write a pack to the documented npz format (``format_version`` 1)."""
    data = dict(
        format_version=np.int64(PACK_FORMAT_VERSION),
        templates=np.asarray(pack.templates),
        offsets=np.asarray(pack.offsets),
        wavelengths_nm=np.asarray(pack.wavelengths_nm),
        field_xy=np.asarray(pack.field_xy),
    )
    if pack.centroids is not None:
        data["centroids"] = np.asarray(pack.centroids)
    if pack.meta_json:
        data["meta_json"] = np.asarray(pack.meta_json)
    np.savez(path, **data)


def load_psflet_pack(path):
    """Load a pack written by :func:`save_psflet_pack` or a conforming emitter."""
    with np.load(path, allow_pickle=False) as data:
        version = int(data["format_version"]) if "format_version" in data else None
        if version != PACK_FORMAT_VERSION:
            raise ValueError(
                f"unsupported PSFlet pack format {version!r}; this reader "
                f"supports format {PACK_FORMAT_VERSION}"
            )
        return PsfletPack(
            templates=data["templates"],
            offsets=data["offsets"],
            wavelengths_nm=data["wavelengths_nm"],
            field_xy=data["field_xy"],
            centroids=data["centroids"] if "centroids" in data else None,
            meta_json=str(data["meta_json"]) if "meta_json" in data else "",
        )
