"""Exact optical contraction and transformation checks for the GPU experiment."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optixstuff.coronagraph import _convolve_quadrants
from optixstuff.experimental.disk_convolution import convolve_disk


@pytest.fixture(autouse=True)
def enable_x64():
    with jax.enable_x64():
        yield


def direct_sum(source, cube):
    """Independent per-source oracle, including the existing even-grid convention."""
    ny, nx = source.shape
    cy, cx = (ny - 1) // 2, (nx - 1) // 2
    result = np.zeros(cube.shape[-2:])
    full = cube.shape[:2] == source.shape
    for y in range(ny):
        for x in range(nx):
            if full:
                psf = cube[y, x]
            else:
                psf = cube[abs(y - cy), abs(x - cx)]
                if y < cy:
                    psf = psf[::-1, :]
                if x < cx:
                    psf = psf[:, ::-1]
            result += source[y, x] * psf
    return result


@pytest.mark.parametrize("shape", [(5, 7), (6, 8), (5, 8), (1, 3)])
@pytest.mark.parametrize("quarter", [False, True])
@pytest.mark.parametrize("backend", ["jax", "reference"])
def test_contraction_and_batch(shape, quarter, backend):
    rng = np.random.default_rng(0)
    src_shape = tuple(n // 2 + 1 for n in shape) if quarter else shape
    cube = rng.uniform(size=(*src_shape, 7, 9))
    sources = rng.normal(size=(3, *shape))
    expected = np.stack([direct_sum(s, cube) for s in sources])
    actual = eqx.filter_jit(convolve_disk)(
        jnp.asarray(sources), jnp.asarray(cube), backend=backend
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    mapped = eqx.filter_jit(
        jax.vmap(lambda s, c: convolve_disk(s, c, backend=backend), in_axes=(0, None))
    )(jnp.asarray(sources), jnp.asarray(cube))
    np.testing.assert_allclose(mapped, expected, rtol=2e-12, atol=2e-12)
    if quarter and shape != (1, 3):
        np.testing.assert_allclose(
            actual[0],
            _convolve_quadrants(jnp.asarray(sources[0]), jnp.asarray(cube)),
            rtol=2e-12,
            atol=2e-12,
        )


@pytest.mark.parametrize("quarter", [False, True])
def test_adjoint_gradients_and_hessian(quarter):
    rng = np.random.default_rng(0)
    shape = (6, 7)
    cshape = (4, 4) if quarter else shape
    cube = jnp.asarray(rng.uniform(size=(*cshape, 5, 9)))
    basis = jnp.asarray(rng.normal(size=(3, *shape)))
    data = jnp.asarray(rng.normal(size=(5, 9)))
    z = jnp.array([0.2, -0.4, 0.5])

    def forward(z, c, backend):
        return convolve_disk(
            jnp.einsum("p,pij->ij", jnp.sin(z), basis), c, backend=backend
        )

    def loss(z, c, backend):
        return jnp.sum((forward(z, c, backend) - data) ** 2) / 2

    for transform in (jax.value_and_grad, jax.hessian):
        actual = eqx.filter_jit(transform(loss))(z, cube, "reference")
        expected = eqx.filter_jit(transform(loss))(z, cube, "jax")
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-10),
            actual,
            expected,
        )

    def f(s):
        return convolve_disk(s, cube, backend="reference")

    source = basis[0]
    _, pullback = jax.vjp(f, source)
    np.testing.assert_allclose(
        jnp.vdot(f(source), data), jnp.vdot(source, pullback(data)[0]), rtol=1e-12
    )
    direction = jnp.array([0.3, 0.1, -0.2])
    eps = 1e-5
    finite = (
        loss(z + eps * direction, cube, "jax") - loss(z - eps * direction, cube, "jax")
    ) / (2 * eps)
    analytical = jnp.vdot(jax.grad(loss)(z, cube, "reference"), direction)
    np.testing.assert_allclose(analytical, finite, rtol=2e-8)


def test_fixed_cube_derivative_is_rejected():
    with pytest.raises(NotImplementedError, match="fixed"):
        jax.grad(
            lambda c: convolve_disk(jnp.ones((3, 3)), c, backend="reference").sum()
        )(jnp.ones((3, 3, 2, 2)))


def test_invalid_backend_and_shapes():
    with pytest.raises(ValueError, match="backend"):
        convolve_disk(jnp.ones((3, 3)), jnp.ones((3, 3, 2, 2)), backend="typo")
    with pytest.raises(ValueError, match="source"):
        convolve_disk(jnp.ones((7, 7)), jnp.ones((2, 2, 2, 2)))


def test_explicit_triton_on_cpu_fails():
    if jax.default_backend() != "cpu":
        pytest.skip("CPU-only diagnostic")
    with pytest.raises(NotImplementedError, match="CUDA"):
        eqx.filter_jit(convolve_disk)(
            jnp.ones((3, 3)), jnp.ones((3, 3, 2, 2)), backend="triton"
        )


@pytest.mark.parametrize("quarter", [False, True])
@pytest.mark.parametrize("backend", ["reference", "jax-batched"])
def test_table_resampling_and_disk_rate(quarter, backend):
    from coronagraphoto import disk_rate
    from skyscapes.disk import GraterDisk
    from skyscapes.scene import FlatStar
    from test_coronagraph_contract import _TableCoronagraph

    from optixstuff import OpticalPath

    rng = np.random.default_rng(0)
    cs = (21, 21) if quarter else (41, 41)
    cube = jnp.asarray(rng.uniform(size=(*cs, 41, 41))) / (41 * 41)
    table = _TableCoronagraph(datacube=cube)
    path = OpticalPath.from_default_setup(
        table, detector_shape=(19, 21), pixel_scale_arcsec=0.025
    )
    other = eqx.tree_at(lambda p: p.coronagraph.disk_convolution_backend, path, backend)
    disk = GraterDisk(
        sma_AU=jnp.array(3.0),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-5.0),
        ksi0_AU=jnp.array(0.2),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.array(1.5),
        rmax_AU=jnp.array(4.5),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        g_HG_grid=jnp.array([0.3, 0.3]),
        Ag_grid=jnp.array([0.5, 0.5]),
        nx=17,
        ny=19,
        pixel_scale_arcsec=0.03,
        dist_pc=10.0,
        n_slices_los=7,
        n_scale_heights=3.0,
    )
    star = FlatStar(Ms_kg=1.0, dist_pc=10.0, flux_phot_per_nm_m2=1e9)

    def predict(incl, p):
        return disk_rate(
            disk,
            p,
            start_time_jd=0.0,
            wavelength_nm=500.0,
            bin_width_nm=50.0,
            telescope_pa_deg=17.0,
            star=star,
            incl_deg=incl,
            pa_deg=30.0,
        )

    for transform in (lambda f: f, jax.jacfwd):
        f = eqx.filter_jit(transform(predict))
        np.testing.assert_allclose(
            f(jnp.array(45.0), path), f(jnp.array(45.0), other), rtol=2e-11, atol=1e-18
        )


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="requires CUDA; run on Colab A100"
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("quarter", [False, True])
@pytest.mark.parametrize("grids", [(17, 19, 13, 15), (65, 67, 35, 37)])
def test_cuda_forward_adjoint_and_hessian(dtype, batch, quarter, grids):
    rng = np.random.default_rng(0)
    shape, detector = grids[:2], grids[2:]
    cs = tuple(n // 2 + 1 for n in shape) if quarter else shape
    cube = jnp.asarray(rng.uniform(size=(*cs, *detector)), dtype=dtype)
    source = jnp.asarray(rng.normal(size=(batch, *shape)), dtype=dtype)
    residual = jnp.asarray(rng.normal(size=(batch, *detector)), dtype=dtype)
    rtol, atol = (2e-5, 2e-4) if dtype == jnp.float32 else (2e-12, 2e-11)

    def forward(s, c, backend):
        return convolve_disk(s, c, backend=backend)

    actual = eqx.filter_jit(forward)(source, cube, "triton")
    expected = eqx.filter_jit(forward)(
        source.astype(jnp.float64), cube.astype(jnp.float64), "jax"
    )
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    def loss(z, c, backend):
        prediction = forward(source * jnp.exp(z[0]) + z[1], c, backend)
        return jnp.sum((prediction - residual) ** 2) / 2

    z = jnp.array([0.1, 0.2], dtype=dtype)
    for transform in (jax.value_and_grad, jax.hessian):
        a = eqx.filter_jit(transform(loss))(z, cube, "triton")
        b = eqx.filter_jit(transform(loss))(z, cube, "jax")
        jax.tree.map(
            lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol), a, b
        )
    mapped = eqx.filter_jit(jax.vmap(forward, in_axes=(0, None, None)))(
        source, cube, "triton"
    )
    np.testing.assert_allclose(mapped, actual, rtol=rtol, atol=atol)

    def adjoint(s, c):
        _, pullback = jax.vjp(lambda x: forward(x, c, "triton"), s)
        return pullback(residual)[0]

    adj = eqx.filter_jit(adjoint)(source, cube)
    np.testing.assert_allclose(
        jnp.vdot(actual, residual), jnp.vdot(source, adj), rtol=rtol, atol=atol
    )
