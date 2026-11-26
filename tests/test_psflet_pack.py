"""Tests for the PSFlet template pack record and its npz format."""

from contextlib import nullcontext

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optixstuff import TEMPLATE_ORIGIN, PsfletPack, load_psflet_pack, save_psflet_pack


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
        template_origin=TEMPLATE_ORIGIN,
        **kwargs,
    )


def test_v2_pack_roundtrip_bitwise(tmp_path):
    """A calibration-defined origin can give the template a nonzero moment."""
    pack = PsfletPack(
        templates=np.array([[[[0, 0, 0], [0, 0.7, 0.3], [0, 0, 0]]]], dtype=np.float32),
        offsets=[-1.0, 0.0, 1.0],
        wavelengths_nm=[650.25],
        field_xy=[[2, -1]],
        template_origin=TEMPLATE_ORIGIN,
        centroids=[[[0.3, -0.0]]],
        meta_json='{"generator": "test"}',
    )
    path = tmp_path / "pack.npz"
    save_psflet_pack(path, pack)
    with np.load(path, allow_pickle=False) as data:
        assert data["format_version"].item() == 2
        assert data["template_origin"].item() == TEMPLATE_ORIGIN
    loaded = load_psflet_pack(path)
    for actual, expected in (
        (loaded.templates, pack.templates),
        (loaded.offsets, pack.offsets),
        (loaded.wavelengths_nm.values, pack.wavelengths_nm.values),
        (loaded.field_xy, pack.field_xy),
        (np.asarray(loaded.template_origin), np.asarray(pack.template_origin)),
        (loaded.centroids, pack.centroids),
    ):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
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
            template_origin=TEMPLATE_ORIGIN,
        )
    with pytest.raises(ValueError, match="uniform"):
        PsfletPack(
            templates=good.templates[:, :, :3, :3],
            offsets=jnp.array([0.0, 1.0, 3.0], dtype=jnp.float32),
            wavelengths_nm=lam,
            field_xy=good.field_xy,
            template_origin=TEMPLATE_ORIGIN,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        PsfletPack(
            templates=good.templates,
            offsets=good.offsets,
            wavelengths_nm=lam[::-1],
            field_xy=good.field_xy,
            template_origin=TEMPLATE_ORIGIN,
        )
    with pytest.raises(ValueError, match="centroids"):
        PsfletPack(
            templates=good.templates,
            offsets=good.offsets,
            wavelengths_nm=lam,
            field_xy=good.field_xy,
            template_origin=TEMPLATE_ORIGIN,
            centroids=jnp.zeros((2, 2, 2)),
        )


@pytest.mark.parametrize(
    ("offsets", "error"),
    [
        *((step * np.arange(24, 27), None) for step in (0.1, 0.2, 1 / 3)),
        ((np.arange(24, 27) / 3).astype(np.float32), None),
        ((0.5, 1.0, 1.5000014), None),
        ((0.50000003, 1.00000003, 1.50000143), "uniform"),
        ((0.5, 1.0, 1.50000146), "uniform"),
        ((1.0, 1.0 + 2e-8, 1.0 + 4e-8), "ascending"),
    ],
)
def test_pack_offset_admission(offsets, error, precision):
    """Original host steps must be uniform and remain distinct in float32.

    At offsets near 1.5, the eight-epsilon uniformity allowance is about
    1.43e-6. The last step is perturbed on either side of that cutoff.
    """
    offsets = np.asarray(offsets)
    with pytest.raises(ValueError, match=error) if error else nullcontext():
        pack = PsfletPack(
            templates=np.ones((1, 1, 3, 3), dtype=np.float32),
            offsets=offsets,
            wavelengths_nm=[650.0],
            field_xy=[[0, 0]],
            template_origin=TEMPLATE_ORIGIN,
        )
        np.testing.assert_array_equal(pack.offsets, offsets.astype(precision))


def test_pack_requires_template_origin():
    """A writer must explicitly declare the template origin."""
    with pytest.raises(TypeError, match="template_origin"):
        PsfletPack(
            templates=np.ones((1, 1, 3, 3)),
            offsets=[-1.0, 0.0, 1.0],
            wavelengths_nm=[650.0],
            field_xy=[[0, 0]],
        )


def test_pack_rejects_legacy_template_origin():
    """Origin-relative templates are refused at construction."""
    with pytest.raises(ValueError, match=r"legacy origin-relative.*regenerate"):
        PsfletPack(
            templates=np.ones((1, 1, 3, 3)),
            offsets=[-1.0, 0.0, 1.0],
            wavelengths_nm=[650.0],
            field_xy=[[0, 0]],
            template_origin="origin",
        )


@pytest.fixture
def pack_data():
    """Return the required entries for a pack from an external writer."""
    return {
        "format_version": np.int64(2),
        "templates": np.ones((1, 1, 3, 3)),
        "offsets": np.array([-1.0, 0.0, 1.0]),
        "wavelengths_nm": np.array([650.0]),
        "field_xy": np.array([[0, 0]]),
        "template_origin": np.asarray("centroid"),
    }


@pytest.mark.parametrize(
    ("version", "message"),
    [
        (None, "unsupported PSFlet pack format None; this reader supports format 2"),
        (1, "template convention changed.*regenerate"),
        (99, "format"),
    ],
)
def test_loader_rejects_unsupported_format(tmp_path, pack_data, version, message):
    """Preserve format compatibility errors, including a missing tag."""
    path = tmp_path / "pack.npz"
    if version is None:
        del pack_data["format_version"]
    else:
        pack_data["format_version"] = np.int64(version)
    np.savez(path, **pack_data)
    with pytest.raises(ValueError, match=message):
        load_psflet_pack(path)


def test_loader_requires_template_origin(tmp_path, pack_data):
    """Earlier format-2 archives must be regenerated with a declaration."""
    path = tmp_path / "pack.npz"
    del pack_data["template_origin"]
    np.savez(path, **pack_data)
    with pytest.raises(ValueError, match=r"declares no template origin.*regenerate"):
        load_psflet_pack(path)


def test_loader_rejects_legacy_template_origin(tmp_path, pack_data):
    """A format-2 tag alone does not make an origin-relative template valid."""
    path = tmp_path / "pack.npz"
    pack_data["template_origin"] = np.asarray("origin")
    np.savez(path, **pack_data)
    with pytest.raises(ValueError, match=r"legacy origin-relative.*regenerate"):
        load_psflet_pack(path)


@pytest.mark.parametrize(
    "version",
    [2.9, 2.0, np.array([2]), np.timedelta64(2, "ns")],
    ids=["fractional", "float", "vector", "timedelta"],
)
def test_loader_rejects_non_scalar_integer_version(tmp_path, pack_data, version):
    """Require both a scalar shape and an integer dtype for the version."""
    path = tmp_path / "pack.npz"
    pack_data["format_version"] = version
    np.savez(path, **pack_data)
    with pytest.raises(ValueError, match="format_version must be a scalar integer"):
        load_psflet_pack(path)


def test_loader_accepts_int32_version(tmp_path, pack_data):
    """An external writer may use an integer dtype other than int64."""
    path = tmp_path / "pack.npz"
    pack_data["format_version"] = np.int32(2)
    np.savez(path, **pack_data)
    assert load_psflet_pack(path).template_origin == "centroid"


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


def test_one_wavelength_pack_is_allowed_and_static(tmp_path):
    """Exact basis: a one-point grid is its single value; no break marks it.

    A pack tabulated at one wavelength builds, round-trips through npz with an
    equal static grid, declares no breaks (its template is constant, so the
    interpolation never changes) and a nonpositive wavelength is refused.
    Fails if a single wavelength is rejected as too short a grid.
    """
    pack = _pack(jnp.array([650.0]))
    assert len(pack.wavelengths_nm) == 1
    assert pack.spectral_breaks_nm().size == 0
    assert not any(leaf is pack.wavelengths_nm for leaf in jax.tree.leaves(pack))
    path = tmp_path / "one.npz"
    save_psflet_pack(path, pack)
    assert load_psflet_pack(path).wavelengths_nm == pack.wavelengths_nm
    with pytest.raises(ValueError, match="positive"):
        _pack(jnp.array([-1.0]))
