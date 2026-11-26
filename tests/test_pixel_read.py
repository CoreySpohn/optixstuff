"""The pixel read of point-sampled densities: footprint integral of the reconstruction.

A table stores the intensity at cell centers. ``on_grid`` reads a density as the
integral of its declared reconstruction over each target pixel; most tests here
use ``cubic_spline``. The truths are analytic: algebra on polynomials, the
closed-form pixel integral of a square-pupil PSF, a closed-form dark-hole
speckle model, and a pupil-limited speckle field integrated by converged
Gauss-Legendre quadrature. Each model is tabulated as a producer does: 256
by 256 cells of 0.25 lambda/D, the density at each cell center.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import arcsec2rad
from hwoutils.testing import ulps
from scipy import special
from scipy.interpolate import make_interp_spline

from optixstuff.coronagraph import on_grid
from optixstuff.grids import (
    Frame,
    Image,
    Measure,
    PixelGrid,
    SkyOrientation,
    _footprint_matrix,
    coverage,
    resample,
)

N_SRC, CELL = 256, 0.25
C_SRC = (N_SRC - 1) / 2
FRAME = Frame(500.0, 6.0, "circumscribed")
DENSITY = Measure.density("grid")
SOURCE = PixelGrid.regular((N_SRC, N_SRC), CELL, "lod", frame=FRAME)
SAMPLES_LOD = (np.arange(N_SRC) - C_SRC) * CELL

# (pitch in lambda/D, offset of the target center in lambda/D on both axes)
PITCHES = (
    (0.125, 0.0),
    (0.2275, 0.0),
    (0.25, 0.0),
    (0.25, 0.0925),
    (0.275, 0.0),
    (0.5, 0.0),
    (0.75, 0.0),
)


@pytest.fixture(autouse=True, scope="module")
def _clear_caches():
    yield
    jax.clear_caches()


def _target(pitch, offset=0.0, half_width=32.0, nominal=None):
    """Square lod grid of the given pitch spanning about +-half_width lambda/D.

    The pixel count follows ``nominal`` (default ``pitch``), so two pitches
    close to one nominal value give grids of one shape.
    """
    n = int(2 * half_width / (nominal or pitch)) // 2 * 2
    center = (n - 1) / 2 - offset / pitch
    grid = PixelGrid.regular((n, n), pitch, "lod", center=(center, center), frame=FRAME)
    centers = (np.arange(n) - (n - 1) / 2) * pitch + offset
    return grid, centers


def _density(table, reconstruction="cubic_spline", grid=SOURCE, measure=DENSITY):
    return Image(jnp.asarray(table), grid, (), measure, "fraction", reconstruction)


def _covered(target, reconstruction="cubic_spline"):
    return np.asarray(
        coverage(SOURCE, target, reconstruction=reconstruction, measure=DENSITY)
    )


# Analytic truths ------------------------------------------------------------


def _sinc2_antiderivative(u):
    """Antiderivative of sinc(u)**2, sinc(u) = sin(pi u) / (pi u)."""
    u = np.asarray(u, float)
    safe = np.where(u == 0, 1.0, u)
    tail = np.where(u == 0, 0.0, np.sin(np.pi * u) ** 2 / (np.pi**2 * safe))
    return special.sici(2 * np.pi * u)[0] / np.pi - tail


class SquarePupil:
    """sinc^2(x - x0) sinc^2(y - y0), x in lambda/D; closed-form pixel integral."""

    def __init__(self, x0, y0):
        self.x0, self.y0 = x0, y0

    def density(self, x, y):
        return np.sinc(y - self.y0)[:, None] ** 2 * np.sinc(x - self.x0)[None] ** 2

    def pixel_integral(self, xc, yc, pitch):
        def axis(c, c0):
            return _sinc2_antiderivative(c + pitch / 2 - c0) - _sinc2_antiderivative(
                c - pitch / 2 - c0
            )

        return axis(yc, self.y0)[:, None] * axis(xc, self.x0)[None]


class DarkHole:
    """Three incoherent speckle fields with a dark hole and a bright surround.

    E_j = sum_k c_jk exp(-|x - x_k|^2 / (2 s^2)) on a unit lambda/D lattice,
    with |c_jk|^2 following 1e-13 in the hole, a wall of 5.5 decades at 29.5
    lambda/D and a bright field beyond. Each product of two Gaussians is a
    Gaussian, so the pixel integral of sum_j |E_j|^2 is a finite sum of error
    functions.
    """

    def __init__(self, s=0.8, half=36, seed=0):
        self.s = s
        self.k = np.arange(-half, half + 1).astype(float)
        rng = np.random.default_rng(seed)
        ky, kx = np.meshgrid(self.k, self.k, indexing="ij")
        r = np.hypot(kx, ky)
        log_i = -13.0 + 0.5 * (r / 26) ** 2 + 5.5 / (1 + np.exp(-(r - 29.5) / 0.8))
        amplitude = 10 ** (0.5 * log_i)
        self.c = [
            amplitude
            * (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape))
            / np.sqrt(2)
            for _ in range(3)
        ]

    def density(self, x, y):
        gx = np.exp(-((x[:, None] - self.k) ** 2) / (2 * self.s**2))
        gy = np.exp(-((y[:, None] - self.k) ** 2) / (2 * self.s**2))
        return sum(np.abs(gy @ c @ gx.T) ** 2 for c in self.c)

    def _pairs(self, centers, pitch):
        k, s = self.k, self.s
        mean = 0.5 * (k[:, None] + k[None])
        scale = np.exp(-((k[:, None] - k[None]) ** 2) / (4 * s**2)) * s * np.sqrt(np.pi)
        low = (centers - pitch / 2)[:, None, None]
        high = (centers + pitch / 2)[:, None, None]
        return (
            0.5
            * scale
            * (special.erf((high - mean) / s) - special.erf((low - mean) / s))
        )

    def pixel_integral(self, xc, yc, pitch):
        ax, ay = self._pairs(xc, pitch), self._pairs(yc, pitch)
        out = np.zeros((yc.size, xc.size))
        for c in self.c:
            right = np.matmul(c, ax).reshape(xc.size, -1)
            left = np.matmul(ay, np.conj(c)).reshape(yc.size, -1)
            out += (left @ right.T).real
        return out


class PupilSpeckle:
    """Fully developed speckle limited by a circular pupil, power to the band edge.

    E = sum_u a(u) exp(2 pi i u.x) over pupil samples |u| <= 1/2 cycle per
    lambda/D, a complex normal: the intensity is band-limited at 1 cycle per
    lambda/D. Three realizations make the hole (1e-13 of a cell), three more an
    amplitude envelope with a wall of 5.5 decades at 29.5 lambda/D. The field is
    exact through the pupil samples; its pixel integral uses Gauss-Legendre with
    8 nodes per axis (converged: 8 against 12 nodes agree to 5e-12 at a pitch of
    0.75 lambda/D, the widest pixel).
    """

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.u = np.arange(-48, 49) / 97.0
        uu, vv = np.meshgrid(self.u, self.u)
        disk = np.hypot(uu, vv) <= 0.5

        def realization():
            normal = rng.standard_normal(disk.shape) + 1j * rng.standard_normal(
                disk.shape
            )
            return np.where(disk, normal, 0)

        self.hole = [realization() for _ in range(3)]
        self.bright = [realization() for _ in range(3)]
        self.norm = 6 * disk.sum()

    def density(self, x, y):
        fx = np.exp(2j * np.pi * x[:, None] * self.u)
        fy = np.exp(2j * np.pi * y[:, None] * self.u)
        hole = sum(np.abs(fy @ a @ fx.T) ** 2 for a in self.hole)
        bright = sum(np.abs(fy @ a @ fx.T) ** 2 for a in self.bright)
        r = np.hypot(x[None], y[:, None])
        envelope = 10 ** (5.5 / (1 + np.exp(-(r - 29.5) / 0.8)))
        wall = 0.5 * (1 + np.tanh((r - 29.5) / 0.8))
        return 1e-13 / CELL**2 * (hole + envelope * bright * wall) / self.norm

    def pixel_integral(self, xc, yc, pitch, nodes=8):
        g, w = np.polynomial.legendre.leggauss(nodes)
        x = (xc[:, None] + 0.5 * pitch * g).ravel()
        y = (yc[:, None] + 0.5 * pitch * g).ravel()
        values = self.density(x, y).reshape(yc.size, nodes, xc.size, nodes)
        weights = 0.5 * pitch * w
        return np.einsum("yqxp,q,p->yx", values, weights, weights)


def _table(model):
    return model.density(SAMPLES_LOD, SAMPLES_LOD)


@pytest.fixture(scope="module")
def dark_hole_cases():
    """Dark-hole table, and per pitch the target, centers and exact pixel integrals."""
    model = DarkHole()
    cases = {}
    for pitch, offset in PITCHES:
        # The closure test reads to the table edge at 0.125, 0.25 and 0.5.
        edge = pitch in (0.125, 0.25, 0.5)
        target, centers = _target(pitch, offset, half_width=32.0 if edge else 27.0)
        cases[pitch, offset] = (
            target,
            centers,
            model.pixel_integral(centers, centers, pitch),
        )
    return _table(model), cases


def _hole(centers, low=3.0, high=26.0):
    r = np.hypot(centers[None], centers[:, None])
    return (r > low) & (r < high)


def _relative_errors(read, truth, mask):
    return np.abs(read - truth)[mask] / truth[mask]


# Algebra ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "reconstruction", ["cubic_spline", "linear", "cubic", "nearest"]
)
def test_constant_and_ramp_integrate_exactly(reconstruction):
    """Fails if a constant or a ramp is not integrated exactly over a pixel.

    Truth: algebra. A constant c and the ramp a + b x + e y reconstruct exactly
    (every kernel here reproduces linear functions; the not-a-knot spline
    reproduces cubics over the whole footprint), so each supported pixel
    integrates to its value at the pixel center times the pixel area (nearest
    reproduces only the constant). A grid covering the table collects the sum
    of the table (partition of unity for the local kernels; for the spline the
    ramp integrates to N times its mean over the footprint).

    Floating-point basis. Allow 96 operations at the peak pixel scale for the
    coordinate transforms, two integrated kernel axes and their contractions,
    and 96 relative for summing the covering pixels and the source.
    """
    n, step = 24, 0.5
    source = PixelGrid.regular((n, n), step, "arcsec")
    x, y = source.coordinates()
    target = PixelGrid.regular((17, 17), 0.61, "arcsec", center=(8.2, 7.9))
    covering = PixelGrid.regular((4, 4), 4.0, "arcsec", center=(1.5, 1.5))
    tx, ty = target.coordinates()
    supported = np.asarray(
        coverage(source, target, reconstruction=reconstruction, measure=DENSITY)
    )
    assert supported.sum() > 100
    fields = [lambda u, v: 3.7 + 0 * u, lambda u, v: 2.0 + 0.3 * u - 0.45 * v]
    for index, field in enumerate(fields):
        image = Image(field(x, y), source, (), DENSITY, "flux", reconstruction)
        read = on_grid(image, target)
        assert read.measure == Measure.integral()
        expected = np.asarray(field(tx, ty)) * target.pixel_area
        if reconstruction != "nearest" or index == 0:
            np.testing.assert_allclose(
                np.asarray(read.data)[supported],
                expected[supported],
                rtol=0,
                atol=float(ulps(96, np.max(np.abs(expected[supported])))),
            )
        total = on_grid(image, covering).data.sum()
        np.testing.assert_allclose(
            total, image.integrated().data.sum(), rtol=float(ulps(96, 1.0)), atol=0
        )


def test_cubic_spline_integrates_a_cubic_polynomial_exactly():
    """Fails if the spline coefficients skip the prefilter or its end condition.

    Truth: algebra. The not-a-knot spline of a cubic is the cubic, so the read of
    the separable p(x) q(y) at every supported pixel is the product of the
    exact one-dimensional integrals, to the table edge. Without the
    prefilter the read smooths the cubic by the 1/6, 2/3, 1/6 filter and is off
    by 3.9e-3.

    Floating-point basis. Allow 192 operations at peak pixel flux for the
    two spline prefilters and box contractions. Allow four boundary strips
    at peak density for coordinate error, each of pixel width times eight
    operations at the largest coordinate. The centered cubic integral is
    evaluated in host float64 and remains well conditioned through zeros.
    """

    def p(t):
        return 1.0 + 0.02 * t - 3e-4 * t**2 + 2e-5 * t**3

    def q(t):
        return 0.8 - 0.01 * t + 1e-4 * t**2 - 1e-5 * t**3

    table = q(SAMPLES_LOD)[:, None] * p(SAMPLES_LOD)[None]
    target, centers = _target(0.37, 0.11)
    read = np.asarray(on_grid(_density(table), target).data)
    width = target.step
    # Only the even terms of the cubic about the cell center integrate.
    px = width * p(centers) + width**3 / 24 * (-6e-4 + 1.2e-4 * centers)
    qy = width * q(centers) + width**3 / 24 * (2e-4 - 6e-5 * centers)
    expected = qy[:, None] * px[None, :]
    supported = _covered(target)
    assert supported.sum() > 0.9 * supported.size
    density_scale = np.max(np.abs(table))
    coordinate_error = float(ulps(8, np.max(np.abs(centers))))
    tolerance = float(ulps(192, density_scale * width**2))
    tolerance += 4 * density_scale * width * coordinate_error
    np.testing.assert_allclose(
        read[supported], expected[supported], rtol=0, atol=tolerance
    )


@pytest.mark.parametrize(
    ("pitch", "offset", "tolerance"),
    [
        (0.125, 0.0, 3.5e-3),
        (0.2275, 0.0, 2.5e-3),
        (0.25, 0.0, 2.5e-3),
        (0.25, 0.0925, 3.3e-3),
        (0.275, 0.0, 2.5e-3),
        (0.5, 0.0, 2.5e-3),
        (0.75, 0.0, 4e-4),
    ],
)
def test_square_pupil_psf_matches_its_closed_form(pitch, offset, tolerance):
    """Fails if the read samples the PSF at pixel centers or loses the prefilter.

    Truth: the closed-form pixel integral of sinc^2 sinc^2 (sine integral), with
    the PSF displaced by (0.03, -0.02) lambda/D. Largest error over every
    supported pixel, in units of the brightest pixel. Tolerance: spike, at
    least twice the measured value (1.7e-3, 1.1e-3, 1.2e-3, 1.6e-3, 1.2e-3,
    6.8e-4 and 1.5e-4 of the peak, in the order of the cases). A center sample
    in place of the footprint is off by 6.8e-3 to 1.1e-1, and a read without
    the prefilter by 4.6e-2 to 1.2e-1.
    """
    model = SquarePupil(0.03, -0.02)
    target, centers = _target(pitch, offset)
    read = np.asarray(on_grid(_density(_table(model)), target).data)
    truth = model.pixel_integral(centers, centers, pitch)
    supported = _covered(target)
    error = np.abs(read - truth)[supported].max() / truth[supported].max()
    print(f"square pupil, pitch {pitch} offset {offset}: {error:.2e} of the peak")
    assert error < tolerance


def test_dark_hole_model_matches_its_closed_form(dark_hole_cases):
    """Fails if the hole is read with a wrong kernel, closure or pixel model.

    Truth: the closed-form pixel integral of the dark-hole model. Error over the
    pixel's own value in the hole (3 to 26 lambda/D) at seven pitches from
    0.125 to 0.75 lambda/D. Tolerance: spike, at least twice the measured
    (median 5.9e-5 to 1.1e-4, 95th percentile 2.6e-4 to 5.1e-4): median below
    2.5e-4 and 95th percentile below 1.2e-3. The Keys kernel under the
    cubic_spline name gives 1.5e-3 to 4.4e-3 at the 95th percentile.
    """
    table, cases = dark_hole_cases
    for (pitch, offset), (target, centers, truth) in cases.items():
        read = np.asarray(on_grid(_density(table), target).data)
        errors = _relative_errors(read, truth, _hole(centers))
        median, p95 = np.median(errors), np.percentile(errors, 95)
        print(f"dark hole, pitch {pitch} offset {offset}: {median:.2e} / {p95:.2e}")
        assert median < 2.5e-4
        assert p95 < 1.2e-3


@pytest.mark.float64_only(reason="float64 baseline for the float32 comparison")
def test_dark_hole_model_float32(dark_hole_cases):
    """Fails if a float32 table loses the hole to rounding in the read.

    Truth and region as the float64 test. The matrices are built in float64 and
    cast to the table's precision, and the product is accumulated in float32.
    Tolerance: from a float32 run, twice the measured, which equals the float64
    run to two digits: median below 2.5e-4 and 95th percentile below 1.2e-3.
    The float32 read differs from the float64 read by at most 7e-7.
    """
    table, cases = dark_hole_cases
    for (pitch, offset), (target, centers, truth) in cases.items():
        image = _density(np.asarray(table, np.float32))
        read32 = on_grid(image, target).data
        assert read32.dtype == jnp.float32
        read64 = np.asarray(on_grid(_density(table), target).data)
        hole = _hole(centers)
        errors = _relative_errors(np.asarray(read32, np.float64), truth, hole)
        floor = np.abs(np.asarray(read32, np.float64) - read64)[hole] / read64[hole]
        median, p95 = np.median(errors), np.percentile(errors, 95)
        print(
            f"dark hole float32, pitch {pitch} offset {offset}: {median:.2e} / "
            f"{p95:.2e}; float32 floor max {floor.max():.2e}"
        )
        assert median < 2.5e-4
        assert p95 < 1.2e-3


def test_no_switch_at_equal_pitch(dark_hole_cases):
    """Fails if the read changes form when the pixel reaches the cell size.

    Truth: the pixel integral is continuous in its limits, so the reads at
    pitches 0.25 (1 - 1e-8) and 0.25 (1 + 1e-8) lambda/D differ only by the
    motion of the pixel edges, at most 5e-7 lambda/D at 26 lambda/D, where the
    relative gradient of the speckle reaches about 5 per lambda/D: about 3e-6.
    Tolerance: 1e-5 relative in the hole. Sampling below the cell size and
    depositing above it differ by percent.
    """
    table, _ = dark_hole_cases
    image = _density(table)
    reads = []
    for factor in (1 - 1e-8, 1 + 1e-8):
        target, centers = _target(0.25 * factor, half_width=27.0, nominal=0.25)
        reads.append(np.asarray(on_grid(image, target).data))
    hole = _hole(centers)
    change = np.abs(reads[1] - reads[0])[hole] / reads[0][hole]
    print(f"no switch: largest relative change {change.max():.2e}")
    assert change.max() < 1e-5


@pytest.mark.parametrize("pitch", [0.125, 0.25, 0.5])
def test_margin_is_two_cells_from_the_pixel_corners(pitch):
    """Fails if the support boundary of the spline read is not 2 cells.

    Truth: geometry. A pixel whose nearest corner lies 1.9 cells inside the
    table edge is unsupported, and one at 2.1 cells is supported, on each of
    the four sides.
    """
    ratio = pitch / CELL
    for inset, expected in ((1.9, False), (2.1, True)):
        for side in ("left", "right", "bottom", "top"):
            # Pixel center in source pixels, with its corner at the inset.
            near = -0.5 + inset + ratio / 2
            far = N_SRC - 0.5 - inset - ratio / 2
            x, y = {
                "left": (near, C_SRC),
                "right": (far, C_SRC),
                "bottom": (C_SRC, near),
                "top": (C_SRC, far),
            }[side]
            target = PixelGrid(
                (1, 1),
                -(x - C_SRC) / ratio,
                -(y - C_SRC) / ratio,
                ((pitch, 0.0), (0.0, pitch)),
                "lod",
                FRAME,
            )
            assert bool(_covered(target)[0, 0]) is expected, (inset, side)


@pytest.mark.parametrize(
    "seed", [0, 1, 2], ids=["realization-a", "realization-b", "realization-c"]
)
def test_full_band_speckle_truth(seed):
    """Fails if the read is less accurate than the docs state on realistic power.

    Truth: a pupil-limited speckle field with power up to the band edge (exact
    through its pupil samples), integrated over each pixel by converged
    Gauss-Legendre quadrature. Three named realizations (seeds 0, 1 and 2), each
    with three hole and three bright fields.
    Pixels about every 0.5 lambda/D along each axis (every pixel at 0.5 and
    0.75) in the hole (3 to 26 lambda/D), pitches 0.125 to 0.75 lambda/D.
    Tolerance: spike, twice the largest measured with another seed (7.3e-4 and
    3.6e-3): median below 1.5e-3 and 95th percentile below 7e-3. Measured here
    2.8e-4 to 7.2e-4 and 1.0e-3 to 3.7e-3 over the three.
    """
    model = PupilSpeckle(seed)
    image = _density(_table(model))
    for pitch, offset in (
        (0.125, 0.0),
        (0.25, 0.0),
        (0.25, 0.0925),
        (0.275, 0.0),
        (0.5, 0.0),
        (0.75, 0.0),
    ):
        _check_speckle(model, image, pitch, offset)


def _check_speckle(model, image, pitch, offset):
    target, centers = _target(pitch, offset, half_width=27.0)
    read = np.asarray(on_grid(image, target).data)
    stride = max(1, round(0.5 / pitch))
    picked = centers[::stride]
    truth = model.pixel_integral(picked, picked, pitch)
    errors = _relative_errors(read[::stride, ::stride], truth, _hole(picked))
    median, p95 = np.median(errors), np.percentile(errors, 95)
    print(f"speckle, pitch {pitch} offset {offset}: {median:.2e} / {p95:.2e}")
    assert median < 1.5e-3
    assert p95 < 7e-3


@pytest.mark.float64_only(reason="finite-difference gradient reference")
def test_gradient_with_respect_to_pitch_and_offset():
    """Fails if the derivative is wrong where a pixel edge lies on a sample.

    Truth: central differences (step 1e-5 of the pitch and 1e-5 cell of the
    offset; the antiderivative of the B-spline is C3, so the difference is
    exact to about 1e-10). The pitch derivative goes through ``on_grid`` with a
    traced basis; the offset derivative through the per-axis matrix. Geometries:
    a pitch of 0.125 lambda/D aligned with the table, which puts pixel edges on
    samples, and an offset one. Tolerance: 1e-5 relative to the largest
    derivative. An antiderivative written as 0.5 + sign(t) G(|t|) has a zero
    derivative at a knot and fails by tens of percent.
    """
    rng = np.random.default_rng(0)
    table = rng.random((32, 32))
    source = PixelGrid.regular((32, 32), CELL, "lod", frame=FRAME)
    image = _density(table, grid=source)

    def read(pitch, center):
        target = PixelGrid(
            (16, 16), center, center, ((pitch, 0.0), (0.0, pitch)), "lod", FRAME
        )
        return on_grid(image, target).data[4:12, 4:12].ravel()

    for pitch, center in ((0.125, 7.5), (0.1371, 7.5 - 0.29)):
        derivative = jax.jacfwd(read)(pitch, center)
        h = 1e-5 * pitch
        difference = (read(pitch + h, center) - read(pitch - h, center)) / (2 * h)
        scale = np.abs(difference).max()
        np.testing.assert_allclose(derivative, difference, rtol=0, atol=1e-5 * scale)

    data = jnp.asarray(rng.random(32))
    for start in (11.25, 11.25 + 0.137):
        positions = start + 0.5 * np.arange(16)

        def pixels(shift, positions=positions):
            matrix = _footprint_matrix(
                32, positions + shift, 0.5, "cubic_spline", float
            )
            return matrix @ data

        derivative = jax.jacfwd(pixels)(0.0)
        h = 1e-5
        difference = (pixels(h) - pixels(-h)) / (2 * h)
        scale = np.abs(difference).max()
        np.testing.assert_allclose(derivative, difference, rtol=0, atol=1e-5 * scale)


def test_closure_near_the_table_edge(dark_hole_cases):
    """Fails if the end condition of the spline reflects or zeroes the table.

    Truth: the closed-form dark-hole model, whose table is cut inside the bright
    field. Pixels whose nearest corner lies 2 to 3 cells inside the table edge.
    Tolerance: spike, 2 percent of the pixel value, twice the largest measured
    on another realization (0.48, 0.29, 0.98 and 0.18 percent at pitches 0.25,
    0.25 offset, 0.125 and 0.5; here 0.30, 0.18, 1.03 and 0.11 percent). A
    mirror closure is off by 2.3e-2, 2.8e-2 and 1.2e-1 at the first three and
    passes at 0.5 (7.2e-3).
    """
    table, cases = dark_hole_cases
    for key in ((0.25, 0.0), (0.25, 0.0925), (0.125, 0.0), (0.5, 0.0)):
        target, centers, truth = cases[key]
        read = np.asarray(on_grid(_density(table), target).data)
        inset = (N_SRC * CELL / 2 - (np.abs(centers) + key[0] / 2)) / CELL
        inset = np.minimum(inset[None], inset[:, None])
        band = (inset >= 2 - 1e-9) & (inset < 3)
        assert band.sum() > 0
        assert np.array_equal(band, band & _covered(target))
        error = _relative_errors(read, truth, band).max()
        print(f"closure, pitch {key[0]} offset {key[1]}: {error:.2e}")
        assert error < 2e-2


@pytest.mark.parametrize("reconstruction", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize("pitch", [0.25, 0.5, 0.75])
def test_local_footprint_reads_have_finite_reach(reconstruction, pitch):
    """Fails if a local kernel's footprint read reaches past its sampling margin.

    Truth: finite reach. The table is embedded in a larger array whose two extra
    rings hold other values; every pixel ``coverage`` calls supported on the
    original table (density measure) reads exactly the same value from both.
    The count of changed values must be zero. Integrating first and depositing
    (the present read of a cell integral) changes values there for cubic at
    0.5 and 0.75 lambda/D and for linear at 0.75.
    """
    rng = np.random.default_rng(0)
    n, ring = 48, 2
    wide = rng.random((n + 2 * ring, n + 2 * ring))
    table = wide[ring:-ring, ring:-ring]
    small = PixelGrid.regular((n, n), CELL, "lod", frame=FRAME)
    large = PixelGrid.regular((n + 2 * ring,) * 2, CELL, "lod", frame=FRAME)
    # A dyadic shift of 13/16 pixel keeps every position exact and puts pixel
    # corners in the band a deposit reaches beyond the sampling margin.
    size = int(n * CELL / pitch) + 4
    center = (size - 1) / 2 + 0.8125
    target = PixelGrid.regular(
        (size, size), pitch, "lod", center=(center, center), frame=FRAME
    )
    reads = [
        np.asarray(on_grid(_density(data, reconstruction, grid), target).data)
        for data, grid in ((table, small), (wide, large))
    ]
    supported = np.asarray(
        coverage(small, target, reconstruction=reconstruction, measure=DENSITY)
    )
    assert supported.any()
    changed = int(np.count_nonzero(reads[0][supported] != reads[1][supported]))
    assert changed == 0


def test_absolute_scale_of_the_read():
    """Fails if the read integrates in the source cell area instead of the pixel.

    Truth: algebra. A uniform density L, read through ``on_grid`` onto a grid
    of a different pitch, is L times the target pixel's solid angle (for
    ``density("sr")``) or area (for ``density("grid")``) at every supported
    pixel, with an integral measure. The steradian is typed here from the
    arcsecond. ``resample`` of a uniform cubic_spline cell integral gives the
    value times the ratio of the pixel areas, at pixels 0.6, 1 and 3.4 times
    the cell.

    Floating-point basis. The conversion constant is checked in float64. Allow
    128 lane operations for geometry, spline prefiltering, two footprint
    contractions and converting from surface density to pixel flux.
    """
    source = PixelGrid.regular((40, 40), 0.004, "arcsec")
    target = PixelGrid.regular((23, 23), 0.0061, "arcsec", center=(11.3, 10.8))
    arcsec_in_rad = np.pi / (180 * 3600)
    assert arcsec2rad == pytest.approx(arcsec_in_rad, rel=1e-15)
    for measure, area in (
        (Measure.density("sr"), (0.0061 * arcsec_in_rad) ** 2),
        (Measure.density("grid"), 0.0061**2),
    ):
        image = Image(
            jnp.full((40, 40), 2.5e-3), source, (), measure, "ph", "cubic_spline"
        )
        read = on_grid(image, target)
        assert read.measure == Measure.integral()
        assert read.reconstruction == "cubic_spline"
        supported = np.asarray(
            coverage(source, target, reconstruction="cubic_spline", measure=measure)
        )
        assert supported.sum() > 100
        np.testing.assert_allclose(
            np.asarray(read.data)[supported],
            0.0025 * area,
            rtol=float(ulps(128, 1.0)),
            atol=0,
        )
    # resample of a cubic_spline cell integral: a uniform 3 per cell becomes 3
    # times the ratio of the pixel areas.
    cells = Image(
        jnp.full((40, 40), 3.0), source, (), Measure.integral(), "ph", "cubic_spline"
    )
    for ratio in (0.6, 1.0, 3.4):
        grid = PixelGrid.regular((9, 9), 0.004 * ratio, "arcsec", center=(4.3, 3.8))
        read = resample(cells, grid)
        assert read.measure == Measure.integral()
        supported = np.asarray(
            coverage(
                source,
                grid,
                reconstruction="cubic_spline",
                measure=Measure.integral(),
            )
        )
        assert supported.sum() > 20
        np.testing.assert_allclose(
            np.asarray(read.data)[supported],
            3.0 * ratio**2,
            rtol=float(ulps(128, 1.0)),
            atol=0,
        )


def test_resample_keeps_center_samples_and_on_grid_refuses_rotation():
    """Fails if the footprint read leaks into ``resample`` or rotation is admitted.

    Truth: algebra. ``resample`` of a density samples the reconstruction at the
    target centers and keeps the density measure: a linear ramp read on a grid
    rotated by 30 degrees (linear) and a cubic polynomial read on an aligned
    grid (cubic_spline) return the polynomial at the centers.
    ``on_grid`` of a cubic_spline or linear density on a rotated grid
    raises, because the footprint read is axis-aligned only.

    Floating-point basis. Allow 24 operations at peak sample scale for
    coordinate transforms, evaluating the polynomial, and separable
    interpolation. The signed cubic can cancel near zero.
    """
    source = PixelGrid.regular((40, 40), 0.5, "arcsec")
    x, y = source.coordinates()
    angle = np.radians(30.0)
    c, s = np.cos(angle), np.sin(angle)
    rotated = PixelGrid(
        (12, 12), 5.5, 5.5, ((0.7 * c, -0.7 * s), (0.7 * s, 0.7 * c)), "arcsec"
    )
    aligned = PixelGrid.regular((12, 12), 0.7, "arcsec", center=(5.2, 5.9))

    def ramp(u, v):
        return 1.0 + 0.1 * u - 0.05 * v

    def cubic(u, v):
        return 1.0 + 0.02 * u**3 - 0.01 * v**2 + 0.03 * u * v

    for reconstruction, grid, field in (
        ("linear", rotated, ramp),
        ("cubic_spline", aligned, cubic),
    ):
        image = Image(field(x, y), source, (), DENSITY, "flux", reconstruction)
        sampled = resample(image, grid)
        assert sampled.measure == DENSITY
        expected = field(*grid.coordinates())
        tolerance = float(ulps(24, jnp.max(jnp.abs(expected))))
        np.testing.assert_allclose(sampled.data, expected, rtol=0, atol=tolerance)
    for reconstruction in ("cubic_spline", "linear"):
        image = Image(cubic(x, y), source, (), DENSITY, "flux", reconstruction)
        with pytest.raises(NotImplementedError, match="rotated"):
            on_grid(image, rotated)


def _box(t):
    return ((t >= -0.5) & (t < 0.5)).astype(float)


def _tent(t):
    return np.clip(1 - np.abs(t), 0, None)


def _keys(t):
    a = np.abs(t)
    inner = 1.5 * a**3 - 2.5 * a**2 + 1
    outer = -0.5 * a**3 + 2.5 * a**2 - 4 * a + 2
    return np.where(a <= 1, inner, np.where(a <= 2, outer, 0.0))


def _periodic_sinc(u, period):
    """Interpolant of a zero-padded table of even period, cosine Nyquist term."""
    out = np.ones_like(u)  # the limit at u = m P is 1 for an even period
    regular = np.abs(np.sin(np.pi * u / period)) > 1e-12
    t = u[regular]
    out[regular] = np.sin(np.pi * t) / np.tan(np.pi * t / period) / period
    return out


def _reference_axis(reconstruction, samples, low, high):
    """Integral of one axis's reconstruction over each [low, high], in cells.

    Gauss-Legendre with 16 nodes on every piece between half-integers. There
    the local kernels and the spline are polynomials of degree at most 3, so
    the rule is exact; the band-limited interpolant (a period of 2N) turns by
    at most a quarter cycle on a piece, where the rule converges to rounding.
    The spline comes from scipy's not-a-knot interpolant; the spline and the
    band-limited interpolant are restricted to the footprint.
    """
    n = samples.size
    if reconstruction == "band_limited":
        low, high = np.clip(low, -0.5, n - 0.5), np.clip(high, -0.5, n - 0.5)

        def field(t):
            return _periodic_sinc(t[..., None] - np.arange(n), 2 * n) @ samples

    elif reconstruction == "cubic_spline":
        spline = make_interp_spline(np.arange(n), samples, k=3)
        low, high = np.clip(low, -0.5, n - 0.5), np.clip(high, -0.5, n - 0.5)

        def field(t):
            return spline(t)

    else:
        kernel = {"nearest": _box, "linear": _tent, "cubic": _keys}[reconstruction]

        def field(t):
            return kernel(t[..., None] - np.arange(n)) @ samples

    nodes, weights = np.polynomial.legendre.leggauss(16)
    out = np.zeros(low.size)
    for i, (a, b) in enumerate(zip(low, high, strict=True)):
        cuts = np.arange(np.ceil(2 * a), np.floor(2 * b) + 1) / 2
        edges = np.unique(np.concatenate([[a], cuts, [b]]))
        for left, right in itertools.pairwise(edges):
            half = (right - left) / 2
            out[i] += half * weights @ field((left + right) / 2 + half * nodes)
    return out


@pytest.mark.parametrize(
    "reconstruction", ["nearest", "linear", "cubic", "cubic_spline", "band_limited"]
)
def test_every_density_is_read_by_its_footprint_integral(reconstruction):
    """Fails if ``on_grid`` reads a declared density any other way.

    Truth: the integral of the declared reconstruction over each pixel, built in
    the test from the kernel (box, tent, Keys cubic), from scipy's not-a-knot
    spline, or from the closed-form periodic sinc of a table zero-padded to
    twice its length (band-limited), by Gauss-Legendre on half-cell pieces.
    The table is separable, so a pixel is the product of two one-dimensional
    integrals. Pitches 0.25 (aligned, equal to the cell), 0.15 and 0.6 lambda/D
    (offset). The
    midpoint read (cell integral, then sample or deposit) differs by percent
    at one of these pitches for each of nearest, linear and cubic, and so
    does a center sample of the band-limited interpolant.

    Floating-point basis. Allow 192 operations at peak pixel scale for
    coordinate differences across the 40-cell table, the two separable kernel
    integrals or Fourier synthesis, and the weighted contractions.
    """
    rng = np.random.default_rng(0)
    n = 40
    rows, columns = rng.random(n), rng.random(n)
    grid = PixelGrid.regular((n, n), CELL, "lod", frame=FRAME)
    image = _density(np.outer(rows, columns), reconstruction, grid)
    for pitch, offset in ((0.25, 0.0), (0.15, 0.031), (0.6, 0.13)):
        target, centers = _target(pitch, offset, half_width=n * CELL / 2 + 1.0)
        read = np.asarray(on_grid(image, target).data)
        positions = centers / CELL + (n - 1) / 2
        width = pitch / CELL
        axis = _reference_axis(
            reconstruction, columns, positions - width / 2, positions + width / 2
        )
        y_axis = _reference_axis(
            reconstruction, rows, positions - width / 2, positions + width / 2
        )
        truth = np.outer(y_axis, axis) * CELL**2
        np.testing.assert_allclose(
            read, truth, rtol=0, atol=float(ulps(192, np.max(np.abs(truth))))
        )


def test_non_square_table_and_anisotropic_flipped_pixels():
    """Fails if the read mixes the axes of a non-square table or pixel.

    Truth: algebra. A separable cubic q(y) p(x) with different coefficients on
    a 36 by 52 table of 0.22 by 0.3 arcsec cells, read onto pixels of 0.37 by
    0.29 arcsec, and the same with the x axis flipped, is the product of the
    exact one-dimensional integrals at every supported pixel.

    Floating-point basis. Allow 128 operations for the anisotropic coordinate
    map, two spline axes, and weighted footprint integration.
    """

    def p(t):
        return 1.0 + 0.05 * t - 4e-3 * t**2 + 3e-4 * t**3

    def p_integral(t):
        return t + 0.025 * t**2 - 4e-3 / 3 * t**3 + 7.5e-5 * t**4

    def q(t):
        return 0.7 - 0.03 * t + 2e-3 * t**2 - 5e-4 * t**3

    def q_integral(t):
        return 0.7 * t - 0.015 * t**2 + 2e-3 / 3 * t**3 - 1.25e-4 * t**4

    source = PixelGrid((36, 52), 25.5, 17.5, ((0.3, 0.0), (0.0, 0.22)), "arcsec")
    x, y = source.coordinates()
    image = Image(q(y) * p(x), source, (), DENSITY, "flux", "cubic_spline")
    for sign in (1.0, -1.0):
        target = PixelGrid(
            (22, 35), 17.2, 10.9, ((0.37 * sign, 0.0), (0.0, 0.29)), "arcsec"
        )
        read = np.asarray(on_grid(image, target).data)
        tx, ty = (np.asarray(v) for v in target.coordinates())
        expected = (q_integral(ty + 0.145) - q_integral(ty - 0.145)) * (
            p_integral(tx + 0.185) - p_integral(tx - 0.185)
        )
        supported = np.asarray(
            coverage(source, target, reconstruction="cubic_spline", measure=DENSITY)
        )
        assert supported.sum() > 300
        np.testing.assert_allclose(
            read[supported], expected[supported], rtol=float(ulps(128, 1.0)), atol=0
        )


def test_traced_orientation_that_differs_raises():
    """Fails if a rotation hidden behind a traced angle is read as axis-aligned.

    The footprint read and the cubic_spline branch of ``resample`` are
    axis-aligned only. Under ``jit`` a target whose orientation angle is traced
    and differs from the source's may hold any rotation, so both raise rather
    than return the unrotated read. A target that shares the source's
    orientation still reads under a traced pixel scale.

    Floating-point basis. Allow 128 operations for the inverse pixel geometry
    and both spline axes in the eager and fused footprint read.
    """
    orientation = SkyOrientation(0.0)
    source = PixelGrid.regular((40, 40), 0.5, "arcsec", orientation=orientation)
    x, y = source.coordinates()
    data = 1.0 + 0.01 * x - 0.02 * y

    def target(angle):
        return PixelGrid.regular(
            (12, 12), 0.7, "arcsec", orientation=SkyOrientation(angle)
        )

    for reconstruction in ("cubic_spline", "linear"):
        image = Image(data, source, (), DENSITY, "flux", reconstruction)
        with pytest.raises(NotImplementedError, match="orientation"):
            jax.jit(lambda angle, image=image: on_grid(image, target(angle)).data)(30.0)
    image = Image(data, source, (), DENSITY, "flux", "cubic_spline")
    with pytest.raises(NotImplementedError, match="orientation"):
        jax.jit(lambda angle: resample(image, target(angle)).data)(30.0)

    def scaled(step):
        grid = PixelGrid(
            (12, 12), 5.5, 5.5, ((step, 0.0), (0.0, step)), "arcsec", None, orientation
        )
        return on_grid(image, grid).data

    np.testing.assert_allclose(
        jax.jit(scaled)(0.7), scaled(0.7), rtol=float(ulps(128, 1.0)), atol=0
    )
