"""Tests for the PSFlet template pack record and its npz format."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optixstuff import PsfletPack, load_psflet_pack, save_psflet_pack


def _pack(lam, sigma=0.7, **kwargs):
    offsets = jnp.linspace(-2.0, 2.0, 17)
    vv, uu = jnp.meshgrid(offsets, offsets, indexing="ij")
    plane = jnp.exp(-0.5 * (uu**2 + vv**2) / sigma**2)
    lam = jnp.asarray(lam)
    templates = jnp.broadcast_to(plane, (1, lam.size, *plane.shape))
    return PsfletPack(
        templates=templates,
        offsets=offsets,
        wavelengths_nm=lam,
        field_xy=jnp.zeros((1, 2)),
        **kwargs,
    )


def test_pack_roundtrip(tmp_path):
    """save/load round-trips every pack field, including provenance."""
    lam = jnp.array([640.0, 660.0, 680.0])
    centroids = jnp.zeros((1, 3, 2)).at[:, :, 0].set(1.5)
    pack = _pack(lam, centroids=centroids, meta_json='{"generator": "test"}')
    path = tmp_path / "pack.npz"
    save_psflet_pack(path, pack)
    loaded = load_psflet_pack(path)
    assert jnp.allclose(loaded.templates, pack.templates)
    assert jnp.allclose(loaded.offsets, pack.offsets)
    assert loaded.wavelengths_nm == pack.wavelengths_nm
    assert jnp.allclose(loaded.field_xy, pack.field_xy)
    assert jnp.allclose(loaded.centroids, pack.centroids)
    assert loaded.meta_json == pack.meta_json


def test_pack_roundtrip_without_centroids(tmp_path):
    """A pack without centroid corrections loads with centroids=None."""
    pack = _pack(jnp.array([640.0, 680.0]))
    assert pack.centroids is None
    path = tmp_path / "pack.npz"
    save_psflet_pack(path, pack)
    assert load_psflet_pack(path).centroids is None


def test_pack_validation_rejects_bad_shapes():
    """Malformed packs fail at construction, not deep inside a build."""
    lam = jnp.array([640.0, 680.0])
    good = _pack(lam)
    with pytest.raises(ValueError, match="templates"):
        PsfletPack(
            templates=good.templates[0],
            offsets=good.offsets,
            wavelengths_nm=lam,
            field_xy=good.field_xy,
        )
    with pytest.raises(ValueError, match="uniform"):
        PsfletPack(
            templates=good.templates[:, :, :3, :3],
            offsets=jnp.array([0.0, 1.0, 3.0]),
            wavelengths_nm=lam,
            field_xy=good.field_xy,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        PsfletPack(
            templates=good.templates,
            offsets=good.offsets,
            wavelengths_nm=lam[::-1],
            field_xy=good.field_xy,
        )
    with pytest.raises(ValueError, match="centroids"):
        PsfletPack(
            templates=good.templates,
            offsets=good.offsets,
            wavelengths_nm=lam,
            field_xy=good.field_xy,
            centroids=jnp.zeros((2, 2, 2)),
        )


def test_loader_rejects_unknown_format(tmp_path):
    """A pack file with an unknown format version fails loudly."""
    path = tmp_path / "pack.npz"
    save_psflet_pack(path, _pack(jnp.array([640.0, 680.0])))
    data = dict(np.load(path, allow_pickle=False))
    data["format_version"] = np.int64(99)
    np.savez(path, **data)
    with pytest.raises(ValueError, match="format"):
        load_psflet_pack(path)


def test_wavelength_grid_is_static():
    """Exact basis (bitwise): the tabulation wavelengths are static knots.

    Freeing a template value leaves ``spectral_breaks_nm`` bitwise unchanged
    and the grid is not a leaf. Fails if the grid is stored as a leaf.
    """
    pack = _pack(jnp.array([640.0, 660.0, 680.0]))
    moved = eqx.tree_at(lambda p: p.templates, pack, pack.templates * 2.0)
    np.testing.assert_array_equal(moved.spectral_breaks_nm(), [640.0, 660.0, 680.0])
    assert not any(leaf is pack.wavelengths_nm for leaf in jax.tree.leaves(pack))
    with pytest.raises(TypeError, match="not a leaf"):
        eqx.tree_at(lambda p: p.wavelengths_nm, pack, pack.wavelengths_nm)
