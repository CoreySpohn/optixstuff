"""Verification of the linear speckle law, native geometry and time clock."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import SinusoidalSpeckle, scalar_coronagraph
from hwoutils.constants import s2d
from hwoutils.grids import Frame, Image, PixelGrid
from hwoutils.spectral import Quadrature

import optixstuff as ox


def test_focal_response_integrates_density_over_the_grid():
    """Fails if either amplitude skips or repeats the pixel-area conversion."""
    grid = PixelGrid.focal(2, 0.25, Frame(500.0, 1.0, "circumscribed"))
    nominal = jnp.array([[1 + 2j, -3j], [2 - 1j, 0.5j]])
    jacobian = jnp.array([[[0.5j, 1.0], [-2 * (1 + 1j), 0.0]]])
    response = ox.FocalResponse(nominal, jacobian, grid, 12.0, 500.0)
    e_pixel, g_pixel = response.per_pixel()
    np.testing.assert_allclose(
        [jnp.sum(jnp.abs(e_pixel) ** 2), jnp.sum(jnp.abs(g_pixel) ** 2)],
        [19.25 / 16.0, 9.25 / 16.0],
        rtol=1e-7,
    )


def test_law_and_cancellation(linear_speckle):
    """Fails if conjugation, energy normalization or stable cross term is lost.

    Float64 intensity difference: absolute error <= 1e-10 of nominal peak.
    At eps=1e-18 with E=G=1, the naive difference is exactly zero; the
    nonzero 2e-18 cross term must instead survive to 16 float64 eps relative.
    """
    field = linear_speckle
    time = 0.17
    eps = np.array([0.03, -0.02]) + np.array([0.05, 0.04]) * np.sin(
        time * np.array([2.0, 3.0])
    )
    nominal, g = np.asarray(field.response.e_nom), np.asarray(field.response.G)
    delta_e = np.einsum("m,myx->yx", eps, g)
    expected = (
        (np.abs(nominal + delta_e) ** 2 - np.abs(nominal) ** 2)
        * field.grid.pixel_area
        / 7.3
    )
    got = field.realize(wavelength_nm=500.0, time_d=time).data
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-10 * np.max(np.abs(nominal) ** 2) / 7.3
    )
    tiny = eqx.tree_at(
        lambda f: (f.response.e_nom, f.response.G, f.offset_nm, f.amplitude_nm),
        field,
        (
            jnp.ones_like(field.response.e_nom),
            jnp.stack(
                [
                    jnp.ones_like(field.response.e_nom),
                    jnp.zeros_like(field.response.e_nom),
                ]
            ),
            jnp.array([1e-18, 0.0]),
            jnp.zeros(2),
        ),
    )
    naive = (np.abs(np.complex128(1) + 1e-18) ** 2 - 1) / 7.3
    assert naive == 0
    np.testing.assert_allclose(
        tiny.realize(wavelength_nm=500.0).data,
        (2e-18 + 1e-36) * field.grid.pixel_area / 7.3,
        rtol=16 * np.finfo(float).eps,
        atol=0,
    )


def test_chromatic_linear_and_quadratic_terms(linear_speckle):
    """Fails if G ignores wavelength or if e_nom is scaled with G.

    Exact algebra: at twice lambda_ref the linear term halves and quadratic
    term quarters, within 32 float64 eps of the per-pixel term scale.
    """
    field = linear_speckle
    halo = eqx.tree_at(
        lambda f: f.response.e_nom, field, jnp.zeros_like(field.response.e_nom)
    )
    wavelengths = jnp.array([500.0, 1000.0])
    delta_e = np.einsum("m,myx->yx", [0.03, -0.02], np.asarray(field.response.G))
    linear = (
        2
        * np.real(np.conj(np.asarray(field.response.e_nom)) * delta_e)
        * field.grid.pixel_area
        / 7.3
    )
    quadratic = np.abs(delta_e) ** 2 * field.grid.pixel_area / 7.3
    got = field.realize(wavelength_nm=wavelengths).data
    quadratic_got = halo.realize(wavelength_nm=wavelengths).data
    floor = 32 * np.finfo(float).eps
    for i, scale in enumerate([1.0, 0.5]):
        np.testing.assert_allclose(quadratic_got[i], scale**2 * quadratic, rtol=floor)
        np.testing.assert_allclose(
            got[i] - quadratic_got[i],
            scale * linear,
            rtol=0,
            atol=floor * np.max(np.abs(linear)),
        )


def test_broadened_node_integral(linear_speckle):
    """Fails if a band-center sample replaces the normalized node integral.

    Weights in nm sum to widths 100, 200 nm; exact weighted algebra to 32
    float64 eps. For zero nominal field, center error is B*(mean(r^2)-r_c^2),
    where B=|G eps|^2/E_in and r=500/lambda; it is nonzero in both bands.
    """
    field = linear_speckle
    nodes = np.array([[400.0, 500.0, 600.0], [600.0, 700.0, 800.0]])
    weights = np.array([[20.0, 60.0, 20.0], [50.0, 100.0, 50.0]])
    quadrature = Quadrature(nodes, weights)
    expected = np.stack(
        [
            sum(
                w * np.asarray(field.realize(wavelength_nm=node).data)
                for node, w in zip(band, weight, strict=True)
            )
            / weight.sum()
            for band, weight in zip(nodes, weights, strict=True)
        ]
    )
    got = field.broadened(quadrature=quadrature)
    np.testing.assert_allclose(got.data, expected, rtol=32 * np.finfo(float).eps)
    assert got.leading_axes == ("band",)
    compiled = eqx.filter_jit(field.broadened)(quadrature=quadrature)
    np.testing.assert_allclose(compiled.data, got.data, rtol=32 * np.finfo(float).eps)
    halo = eqx.tree_at(
        lambda f: f.response.e_nom, field, jnp.zeros_like(field.response.e_nom)
    )
    b = (
        np.abs(np.einsum("m,myx->yx", [0.03, -0.02], np.asarray(field.response.G))) ** 2
        * field.grid.pixel_area
        / 7.3
    )
    correction = (weights * (500 / nodes) ** 2).sum(axis=1) / weights.sum(axis=1)
    correction -= (500 / np.array([500.0, 700.0])) ** 2
    assert np.all(correction > 0.01)
    difference = (
        halo.broadened(quadrature=quadrature).data
        - halo.realize(wavelength_nm=jnp.array([500.0, 700.0])).data
    )
    np.testing.assert_allclose(
        difference, correction[:, None, None] * b, rtol=1e-13, atol=0
    )


def test_epoch_and_float32_offset_resolution(linear_speckle):
    """Fails if epoch subtraction or float64 absolute-clock precision is lost.

    A float32 1e-6 day offset added AFTER float64 promotion at epoch 10 days
    survives subtraction to 2e-9 relative (binary64 roundoff / 1e-6).
    Adding in float32 has >1% error. This does not promise recovery from an
    already rounded float32 absolute timestamp.
    """
    field = replace(linear_speckle, epoch_d=np.float64(10.0))
    assert type(field.epoch_d) is float
    assert not any(leaf is field.epoch_d for leaf in jax.tree.leaves(field))
    time = 10.23
    np.testing.assert_array_equal(
        field.realize(wavelength_nm=500.0, time_d=time).data,
        linear_speckle.realize(wavelength_nm=500.0, time_d=time - 10.0).data,
    )
    offset = jnp.asarray(1e-6, dtype=jnp.float32)
    timestamp = jnp.asarray(field.epoch_d, dtype=jnp.float64) + offset
    resolved = timestamp - field.epoch_d
    np.testing.assert_allclose(resolved, offset, rtol=2e-9, atol=0)
    rounded = (jnp.float32(field.epoch_d) + offset) - jnp.float32(field.epoch_d)
    assert abs(float(rounded / offset) - 1) > 0.01
    expected_change = np.array([0.05, 0.04]) * np.sin(
        float(offset) * np.array([2.0, 3.0])
    )
    actual_change = field.coefficients(timestamp) - field.coefficients(field.epoch_d)
    np.testing.assert_allclose(actual_change, expected_change, rtol=2e-9, atol=0)
    np.testing.assert_allclose(
        eqx.filter_jit(field.realize)(wavelength_nm=500.0, time_d=timestamp).data,
        linear_speckle.realize(wavelength_nm=500.0, time_d=float(offset)).data,
        rtol=0,
        atol=1e-15,
    )


def test_image_geometry_jit_and_time_batching(linear_speckle):
    """Fails if native sampling is stamped, metadata lost, or time axes collide.

    Geometry uses 6 m and lambda[nm]*1e-9 *180/pi*3600/D; 16 float64 eps.
    JIT/vmap agree within 32 eps of the maximum image value (operation order).
    """
    field = linear_speckle
    nodes = jnp.array([[400.0, 600.0], [700.0, 800.0]])
    image = field.realize(wavelength_nm=nodes, time_d=0.2)
    assert isinstance(image, Image)
    assert image.grid == field.grid
    assert image.grid.step == 0.37
    assert not hasattr(field, "pixel_scale_lod")
    assert image.leading_axes == ("band", "quadrature")
    assert (image.measure, image.quantity, image.reconstruction, image.outside) == (
        "per_pixel",
        "stellar_flux_fraction",
        "band_limited",
        "reflect",
    )
    for wavelength in [500.0, 1000.0]:
        angular_grid = image.grid.to_arcsec(wavelength)
        expected_step = 0.37 * wavelength * 1e-9 / 6 * 180 / np.pi * 3600
        np.testing.assert_allclose(
            angular_grid.step, expected_step, rtol=16 * np.finfo(float).eps
        )
        assert angular_grid.center_x == 1.5
        assert angular_grid.center_y == 1.0
    compiled = eqx.filter_jit(field.realize)(wavelength_nm=nodes, time_d=jnp.array(0.2))
    np.testing.assert_allclose(compiled.data, image.data, rtol=32 * np.finfo(float).eps)
    assert compiled.grid == image.grid
    times = jnp.array([0.0, 0.2, 0.7])
    loop = np.stack([field.realize(wavelength_nm=nodes, time_d=t).data for t in times])
    mapped = jax.vmap(lambda t: field.realize(wavelength_nm=nodes, time_d=t).data)(
        times
    )
    batched = field.realize(wavelength_nm=nodes, time_d=times)
    assert batched.leading_axes == ("band", "quadrature", "time_0")
    floor = 32 * np.finfo(float).eps * np.max(np.abs(loop))
    np.testing.assert_allclose(mapped, loop, rtol=0, atol=floor)
    np.testing.assert_allclose(
        batched.data, np.moveaxis(loop, 0, 2), rtol=0, atol=floor
    )


def test_coefficient_gradient(linear_speckle):
    """Fails if eps is detached or the interference derivative is incorrect.

    Central difference h=1e-5 is exact for this quadratic apart from roundoff;
    1e-10 absolute covers O(eps_machine/h) in these order-one field units.
    Compare each mode's derivative at every pixel, avoiding sum cancellation.
    """
    field = linear_speckle

    def realize_eps(eps):
        changed = eqx.tree_at(lambda f: f.offset_nm, field, eps)
        return changed.realize(wavelength_nm=650.0).data

    eps = jnp.array([0.03, -0.02])
    gradient = jax.jacfwd(realize_eps)(eps)
    h = 1e-5
    finite = np.stack(
        [
            (
                np.asarray(realize_eps(eps + direction * h))
                - np.asarray(realize_eps(eps - direction * h))
            )
            / (2 * h)
            for direction in np.eye(2)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(gradient, finite, rtol=0, atol=1e-10)


def test_exposure_average(linear_speckle):
    """Fails if averaging coefficients, seconds/days confusion or metadata loss.

    Midpoint sum/JIT: 32 float64 eps. Three refinements (4, 8, 16) of a
    smooth sinusoidal quadratic must give second-order ratios in [3.8,4.2].
    """
    field = linear_speckle
    # Shared conversion is rounded by a few binary64 ULPs.
    np.testing.assert_allclose(s2d, 1 / 86400, rtol=4 * np.finfo(float).eps)
    nodes = jnp.array([[450.0, 550.0]])
    keywords = dict(wavelength_nm=nodes, exposure_s=86400.0, start_time_d=0.1)
    expected = np.mean(
        [
            field.realize(wavelength_nm=nodes, time_d=0.1 + (i + 0.5) / 8).data
            for i in range(8)
        ],
        axis=0,
    )
    average = field.realize_average(**keywords)
    np.testing.assert_allclose(average.data, expected, rtol=32 * np.finfo(float).eps)
    assert average.grid == field.grid
    assert average.leading_axes == ("band", "quadrature")
    assert average.reconstruction == "band_limited" and average.outside == "reflect"
    compiled = eqx.filter_jit(field.realize_average)(**keywords)
    np.testing.assert_allclose(compiled.data, expected, rtol=32 * np.finfo(float).eps)
    np.testing.assert_allclose(
        field.realize_average(**keywords, n_sub=1).data,
        field.realize(wavelength_nm=nodes, time_d=0.6).data,
        rtol=32 * np.finfo(float).eps,
    )
    levels = [field.realize_average(**keywords, n_sub=n).data for n in [4, 8, 16]]
    ratio = np.max(np.abs(levels[0] - levels[1])) / np.max(
        np.abs(levels[1] - levels[2])
    )
    assert 3.8 < ratio < 4.2
    # Averaging the coefficient trajectory omits its nonzero quadratic variance.
    mean_eps = jnp.mean(field.coefficients(0.1 + (jnp.arange(8) + 0.5) / 8), axis=0)
    mean_field = eqx.tree_at(
        lambda f: (f.offset_nm, f.amplitude_nm), field, (mean_eps, jnp.zeros(2))
    )
    assert (
        np.max(np.abs(average.data - mean_field.realize(wavelength_nm=nodes).data))
        > 1e-5
    )
    static = eqx.tree_at(lambda f: f.amplitude_nm, field, jnp.zeros(2))
    np.testing.assert_allclose(
        static.realize_average(**keywords).data,
        static.realize(wavelength_nm=nodes).data,
        rtol=32 * np.finfo(float).eps,
    )
    for bad in [0, -1, 1.5]:
        with pytest.raises(ValueError, match="n_sub"):
            field.realize_average(**keywords, n_sub=bad)


def test_construction_and_optical_path(linear_speckle):
    """Fails if abstractness, ingredient geometry or optional path wiring breaks.

    Structural assertions are exact; malformed fields must fail at construction.
    """
    field = linear_speckle
    for cls in [ox.AbstractSpeckleField, ox.LinearSpeckleField]:
        with pytest.raises(TypeError):
            cls()
    with pytest.raises(TypeError):
        field.realize(500.0)
    for changes in [
        {"input_energy": 0.0},
        {"input_energy": [1.0]},
        {"e_nom": jnp.ones((2, 4))},
        {"G": jnp.ones((2, 4))},
        {"reference_wavelength_nm": 600.0},
        {"grid": PixelGrid.regular((3, 4), 0.37, "arcsec")},
    ]:
        with pytest.raises(ValueError):
            replace(field, response=replace(field.response, **changes))
    with pytest.raises(ValueError, match="epoch_d"):
        replace(field, epoch_d=np.inf)
    assert isinstance(field, SinusoidalSpeckle)
    path = ox.OpticalPath.from_default_setup(scalar_coronagraph(), speckle=field)
    assert path.speckle is field
    assert any(line.startswith("  speckle:") for line in repr(path).splitlines())
    default = ox.OpticalPath.from_default_setup(scalar_coronagraph())
    assert default.speckle is None and "speckle:" not in repr(default)
