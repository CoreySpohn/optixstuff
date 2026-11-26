"""Verification of the linear speckle law, native geometry and time clock."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    SinusoidalSpeckle,
    make_instrument,
    make_linear_speckle,
    table_coronagraph,
)
from hwoutils.constants import d2s, s2d
from hwoutils.precision import PrecisionError
from hwoutils.testing import ulps

import optixstuff as ox
from optixstuff.grids import Frame, Image, Measure, PixelGrid, resample
from optixstuff.spectral import Band, Quadrature


def test_focal_response_integrates_density_over_the_grid():
    """Fails if either amplitude skips or repeats the pixel-area conversion."""
    grid = PixelGrid.focal(2, 0.25, Frame(500.0, 1.0, "circumscribed"))
    nominal = jnp.array([[1 + 2j, -3j], [2 - 1j, 0.5j]])
    jacobian = jnp.array([[[0.5j, 1.0], [-2 * (1 + 1j), 0.0]]])
    response = ox.FocalResponse(
        nominal, jacobian, grid, 12.0, 500.0, wavelength_law="opd"
    )
    e_pixel, g_pixel = response.per_pixel()
    np.testing.assert_allclose(
        [jnp.sum(jnp.abs(e_pixel) ** 2), jnp.sum(jnp.abs(g_pixel) ** 2)],
        [19.25 / 16.0, 9.25 / 16.0],
        rtol=1e-7,
    )


@pytest.mark.parametrize(("law", "expected"), [("amplitude", 1.0), ("opd", 0.25)])
def test_wavelength_law_halo_ratio(law, expected):
    """Squared OPD phase and achromatic transmission give analytic halo ratios.

    Scaling every sensitivity by
    reference wavelength over wavelength incorrectly quarters the amplitude halo.

    Floating-point basis. Allow 12 operations for amplitude scaling, complex
    magnitude squared, pixel area and forming the halo ratio.
    """
    grid = PixelGrid.focal(2, 0.25, Frame(500.0, 1.0, "circumscribed"))
    response = ox.FocalResponse(
        jnp.zeros(grid.shape, dtype=complex),
        jnp.full((1, *grid.shape), 0.3 + 0.2j),
        grid,
        1.0,
        500.0,
        wavelength_law=law,
    )
    field = SinusoidalSpeckle(response, jnp.ones(1), jnp.zeros(1), jnp.zeros(1))
    blue, red = field.realize(wavelength_nm=jnp.array([500, 1000])).data
    np.testing.assert_allclose(red / blue, expected, rtol=float(ulps(12, 1.0)), atol=0)


def test_focal_response_requires_law_matching_layout(linear_speckle):
    """The response layout contract is exact (mutation, no numeric tolerance).

    Fails if the law defaults to OPD, accepts unknown values, or permits
    scaling an already tabulated sensitivity.
    """
    r = linear_speckle.response
    with pytest.raises(TypeError, match="wavelength_law"):
        ox.FocalResponse(r.e_nom, r.G, r.grid, r.input_energy, 500.0)
    with pytest.raises(ValueError, match=r"opd.*amplitude"):
        replace(r, wavelength_law=None)
    with pytest.raises(ValueError, match=r"opd.*amplitude"):
        replace(r, wavelength_law="unknown")
    with pytest.raises(ValueError, match="wavelength_law=None"):
        replace(r, e_nom=r.e_nom[None], G=r.G[None])


def test_law_and_cancellation(linear_speckle):
    """Fails if conjugation, energy normalization or stable cross term is lost.

    At eps=1e-18 with E=G=1, the naive difference is exactly zero; the
    nonzero 2e-18 cross term must instead survive.

    Floating-point basis. Allow 32 operations at the nominal intensity scale
    for the phase, complex contraction and intensity calculation. Allow 16
    relative operations for the isolated nonzero cross term.
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
    got = field.realize(wavelength_nm=500.0, elapsed_s=time).data
    np.testing.assert_allclose(
        got,
        expected,
        rtol=0,
        atol=float(
            ulps(32, np.max(np.abs(nominal) ** 2) * field.grid.pixel_area / 7.3)
        ),
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
        rtol=float(ulps(16, 1.0)),
        atol=0,
    )


def test_chromatic_linear_and_quadratic_terms(linear_speckle, precision):
    """Fails if G ignores wavelength or if e_nom is scaled with G.

    Exact algebra: at twice lambda_ref the linear term halves and quadratic
    term quarters, within 32 eps of the per-pixel term scale.
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
    floor = 32 * np.finfo(precision).eps
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

    Weights in nm sum to widths 200, 200 nm. For zero nominal field,
    center error is B*(mean(r^2)-r_c^2),
    where B=|G eps|^2/E_in and r=500/lambda; it is nonzero in both bands.

    Floating-point basis. Allow 32 operations for the weighted node sum and
    coherent read. For the halo difference use the full halo scale before
    subtraction, with 32 operations for both operands.
    """
    field = linear_speckle
    nodes = np.array([[400.0, 500.0, 600.0], [600.0, 700.0, 800.0]])
    weights = np.array([[40.0, 120.0, 40.0], [50.0, 100.0, 50.0]])
    quadrature = Quadrature((Band(400, 600), Band(600, 800)), nodes, weights)
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
    np.testing.assert_allclose(got.data, expected, rtol=float(ulps(32, 1.0)), atol=0)
    assert got.leading_axes == ("band",)
    compiled = eqx.filter_jit(field.broadened)(quadrature=quadrature)
    np.testing.assert_allclose(
        compiled.data,
        got.data,
        rtol=0,
        atol=float(ulps(32, jnp.max(jnp.abs(got.data)))),
    )
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
        difference,
        correction[:, None, None] * b,
        rtol=0,
        atol=float(ulps(32, np.max(np.abs(b)))),
    )


_EPOCHS_D = (60000.123, 60000.125)


@pytest.mark.float32_only(reason="checks float32 epoch conditioning and traced refusal")
def test_elapsed_s_conditions_absolute_times_before_narrowing():
    """Fails if an absolute time is rounded before subtracting the epoch.

    The float64 reference subtracts before converting days to seconds.
    The only float32 rounding is storage of the offset, so compare exactly
    to the correctly rounded reference. A float64 JAX array retains the
    input when it leaves its construction scope; a trace still requires x64.
    """
    elapsed_under_jit = eqx.filter_jit(lambda field, time: field.elapsed_s(time))
    for epoch in _EPOCHS_D:
        field = make_linear_speckle(epoch_d=epoch)
        time = epoch + 10 * s2d
        reference = np.float32((np.float64(time) - epoch) * d2s)
        with jax.enable_x64():
            carried = jnp.asarray(time)
        for concrete in (time, np.asarray(time, ">f8"), carried):
            elapsed = field.elapsed_s(concrete)
            assert elapsed.dtype == jnp.float32
            np.testing.assert_array_equal(elapsed, reference)
        with pytest.raises(PrecisionError, match=r"pass elapsed_s.*Enable x64"):
            elapsed_under_jit(field, carried)


def test_elapsed_s_refuses_reduced_precision_at_a_nonzero_epoch(precision):
    """Subtracting a nonzero epoch requires the original absolute time.

    Host, device and traced float32 absolute times are refused in either
    lane. Complex values are refused even when they occupy eight bytes.
    At epoch zero, one day is exactly 86400 seconds in both output dtypes.
    """
    anchored = make_linear_speckle(epoch_d=_EPOCHS_D[0])
    time = np.float32(_EPOCHS_D[0] + 10 * s2d)
    compiled = eqx.filter_jit(lambda field, t: field.elapsed_s(t))
    remedy = r"elapsed_s formed on the host.*original"
    for reduced in (time, jnp.asarray(time), np.float16(1)):
        with pytest.raises(PrecisionError, match=remedy):
            anchored.elapsed_s(reduced)
    with pytest.raises(PrecisionError, match=remedy):
        compiled(anchored, jnp.asarray(time))
    with pytest.raises(ValueError, match="time_d"):
        anchored.elapsed_s(np.complex64(time))

    zero = make_linear_speckle()
    with jax.enable_x64():
        carried = jnp.asarray(np.float64(1.0))
    assert carried.dtype == np.float64
    for days in (np.float32(1), jnp.asarray(1.0, dtype=precision), carried):
        for elapsed in (zero.elapsed_s(days), compiled(zero, days)):
            assert elapsed.dtype == precision
            np.testing.assert_array_equal(elapsed, 86400)


@pytest.mark.float64_only(reason="absolute epoch must retain the sub-day offset")
def test_elapsed_s_eager_equals_jit_under_x64():
    """Fails if the host, device and traced paths of the kernel disagree.

    Times ``epoch + k * 2**-20`` d are exact in binary64, so ``time - epoch``
    is exact (Sterbenz) and the elapsed seconds are ``k * 86400 / 2**20``
    exactly, one rounding at most. Exact comparison for the host call, the
    eager call on a JAX array and the jit call; ``realize`` through each
    equals the field anchored at zero read at those seconds, eager against
    eager and compiled against compiled. Eager against compiled images agree
    to 64 float64 eps of the largest pixel, not bitwise: the compiled program
    fuses multiply-adds (8.7e-19 seen on 1e-3 pixels). Fails if the
    device path subtracts in float32 or forms ``time * d2s - epoch * d2s``.
    """
    epoch = _EPOCHS_D[0]
    field = make_linear_speckle(epoch_d=epoch)
    zero = make_linear_speckle()
    k = np.array([1.0, 3.0, 7.0, 121.0, 1001.0, 12345.0])
    times = epoch + k * 2.0**-20
    np.testing.assert_array_equal(times - epoch, k * 2.0**-20)
    exact = k * 2.0**-20 * 86400.0
    host = field.elapsed_s(times)
    device = field.elapsed_s(jnp.asarray(times))
    traced = jax.jit(lambda t: field.elapsed_s(t))(times)
    for got in (host, device, traced):
        assert got.dtype == jnp.float64
        np.testing.assert_array_equal(got, exact)
    render = eqx.filter_jit(lambda f, t: f.realize(wavelength_nm=500.0, elapsed_s=t))
    eager = zero.realize(wavelength_nm=500.0, elapsed_s=exact).data
    compiled = render(zero, exact).data
    for elapsed in (host, device, traced):
        np.testing.assert_array_equal(
            field.realize(wavelength_nm=500.0, elapsed_s=elapsed).data, eager
        )
        np.testing.assert_array_equal(render(field, elapsed).data, compiled)
    floor = 64 * np.finfo(np.float64).eps * np.max(np.abs(eager))
    np.testing.assert_allclose(compiled, eager, rtol=0, atol=floor)


def test_coefficients_eager_equals_jit(precision):
    """Fails if ``coefficients(elapsed_s=)`` branches on the argument's type.

    Truth: the eager call, against ``eqx.filter_jit`` for Python floats, NumPy
    arrays and JAX arrays, in each precision. Tolerance basis floating-point:
    the compiled program contracts ``offset + amplitude * sin`` into one fused
    multiply-add, which drops one rounding, so the two agree to 4 eps of the
    term scale ``|offset| + |amplitude|`` (0.08 nm), not bitwise. A branch that
    casts a JAX array through float32 moves the x64 result by about 1e-9 nm.
    """
    field = make_linear_speckle()
    coefficients = eqx.filter_jit(field.coefficients)
    floor = 4 * np.finfo(precision).eps * 0.08
    for elapsed in (0.1, np.array([0.1, 0.7, 2.3]), jnp.array([0.1, 0.7, 2.3])):
        eager = field.coefficients(elapsed_s=elapsed)
        assert eager.dtype == precision
        np.testing.assert_allclose(
            coefficients(elapsed_s=elapsed), eager, rtol=0, atol=floor
        )


def test_epoch_is_static_metadata_checked_by_the_kernel():
    """Fails if ``epoch_d`` becomes a leaf or the kernel stops checking its type.

    Two fields that differ only in ``epoch_d`` have the same leaves, bitwise,
    and different tree structures (a static field is part of the structure);
    with ``epoch_d`` a leaf the leaves differ and the structures agree. A field
    whose ``epoch_d`` was swapped for an array raises ``TypeError`` from
    ``elapsed_s``. Exact.
    """
    field = make_linear_speckle(epoch_d=np.float64(_EPOCHS_D[0]))
    other = make_linear_speckle(epoch_d=5.0)
    assert type(field.epoch_d) is float
    leaves = jax.tree_util.tree_leaves(field)
    assert not any(leaf is field.epoch_d for leaf in leaves)
    others = jax.tree_util.tree_leaves(other)
    assert len(leaves) == len(others)
    for a, b in zip(leaves, others, strict=True):
        np.testing.assert_array_equal(a, b)
    assert jax.tree_util.tree_structure(field) != jax.tree_util.tree_structure(other)
    with pytest.raises(TypeError):
        eqx.tree_at(lambda f: f.epoch_d, field, jnp.asarray(1.0))
    swapped = replace(field)
    object.__setattr__(swapped, "epoch_d", jnp.asarray(_EPOCHS_D[0]))
    with pytest.raises(TypeError, match="epoch_d"):
        swapped.elapsed_s(_EPOCHS_D[0] + 10 * s2d)


def test_image_geometry_jit_and_time_batching(linear_speckle, precision):
    """Fails if native sampling is stamped, metadata lost, or time axes collide.

    Geometry uses 6 m and lambda[nm]*1e-9 *180/pi*3600/D; 16 float64 eps.
    JIT/vmap agree within 32 eps of the maximum image value (operation order).
    """
    field = linear_speckle
    nodes = jnp.array([[400.0, 600.0], [700.0, 800.0]])
    image = field.realize(wavelength_nm=nodes, elapsed_s=0.2)
    assert isinstance(image, Image)
    assert image.grid == field.grid
    assert image.grid.step == 0.37
    assert not hasattr(field, "pixel_scale_lod")
    assert image.leading_axes == ("band", "quadrature")
    assert (image.measure, image.quantity, image.reconstruction) == (
        Measure.integral(),
        "stellar_flux_fraction",
        "band_limited",
    )
    for wavelength in [500.0, 1000.0]:
        angular_grid = image.grid.to_arcsec(wavelength)
        expected_step = 0.37 * wavelength * 1e-9 / 6 * 180 / np.pi * 3600
        np.testing.assert_allclose(
            angular_grid.step, expected_step, rtol=16 * np.finfo(float).eps
        )
        assert angular_grid.center_x == 1.5
        assert angular_grid.center_y == 1.0
    # The field lives on its footprint: a target wholly outside it reads zero,
    # not a reflected copy of the edge.
    shifted = PixelGrid.regular(
        field.grid.shape, 0.37, "lod", center=(40.0, 40.0), frame=field.grid.frame
    )
    far = resample(image, shifted)
    np.testing.assert_array_equal(far.data, 0.0)
    compiled = eqx.filter_jit(field.realize)(
        wavelength_nm=nodes, elapsed_s=jnp.array(0.2)
    )
    np.testing.assert_allclose(
        compiled.data, image.data, rtol=32 * np.finfo(precision).eps
    )
    assert compiled.grid == image.grid
    times = jnp.array([0.0, 0.2, 0.7])
    loop = np.stack(
        [field.realize(wavelength_nm=nodes, elapsed_s=t).data for t in times]
    )
    mapped = jax.vmap(lambda t: field.realize(wavelength_nm=nodes, elapsed_s=t).data)(
        times
    )
    batched = field.realize(wavelength_nm=nodes, elapsed_s=times)
    assert batched.leading_axes == ("band", "quadrature", "time_0")
    floor = 32 * np.finfo(precision).eps * np.max(np.abs(loop))
    np.testing.assert_allclose(mapped, loop, rtol=0, atol=floor)
    np.testing.assert_allclose(
        batched.data, np.moveaxis(loop, 0, 2), rtol=0, atol=floor
    )


@pytest.mark.float64_only(reason="finite-difference gradient reference")
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


def test_exposure_average(linear_speckle, precision):
    """Fails if averaging coefficients, a step left in days or metadata loss.

    Truth: the mean of ``realize`` at ``s0 + (i + 0.5) * T / n`` seconds.
    Midpoint sum/JIT: 32 eps. Three refinements (4, 8, 16) of a
    smooth sinusoidal quadratic must give second-order ratios in [3.8,4.2].
    """
    field = linear_speckle
    nodes = jnp.array([[450.0, 550.0]])
    keywords = dict(wavelength_nm=nodes, exposure_s=1.0, start_elapsed_s=0.1)
    expected = np.mean(
        [
            field.realize(wavelength_nm=nodes, elapsed_s=0.1 + (i + 0.5) / 8).data
            for i in range(8)
        ],
        axis=0,
    )
    average = field.realize_average(**keywords)
    np.testing.assert_allclose(
        average.data, expected, rtol=32 * np.finfo(precision).eps
    )
    assert average.grid == field.grid
    assert average.leading_axes == ("band", "quadrature")
    assert average.reconstruction == "band_limited"
    compiled = eqx.filter_jit(field.realize_average)(**keywords)
    np.testing.assert_allclose(
        compiled.data, expected, rtol=32 * np.finfo(precision).eps
    )
    np.testing.assert_allclose(
        field.realize_average(**keywords, n_sub=1).data,
        field.realize(wavelength_nm=nodes, elapsed_s=0.6).data,
        rtol=32 * np.finfo(precision).eps,
    )
    levels = [field.realize_average(**keywords, n_sub=n).data for n in [4, 8, 16]]
    ratio = np.max(np.abs(levels[0] - levels[1])) / np.max(
        np.abs(levels[1] - levels[2])
    )
    assert 3.8 < ratio < 4.2
    # Averaging the coefficient trajectory omits its nonzero quadratic variance.
    mean_eps = jnp.mean(
        field.coefficients(elapsed_s=0.1 + (jnp.arange(8) + 0.5) / 8), axis=0
    )
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
        rtol=32 * np.finfo(precision).eps,
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
    # Days passed positionally would be read as seconds; the keyword is required.
    with pytest.raises(TypeError):
        field.coefficients(1.0)
    with pytest.raises(TypeError):
        field.realize(wavelength_nm=500.0, time_d=1.0)
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
    path = make_instrument(table_coronagraph(), speckle=field)
    assert path.speckle is field
    assert any(line.startswith("  speckle:") for line in repr(path).splitlines())
    default = make_instrument(table_coronagraph())
    assert default.speckle is None and "speckle:" not in repr(default)
