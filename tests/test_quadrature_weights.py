"""Named-weight quadrature against independent reference integrals.

Every expected value is a 12-point Gauss-Legendre sum over each interval
between the knots of the tabulated response, star and zodi, typed here with
NumPy. The build under test uses its own reference rule of another order.
All tests are code verification on a 50 percent band at 500 nm.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi

from optixstuff.spectral import Band, Probe, Quadrature, Transmission

LO_NM, HI_NM, CENTER_NM = 375.0, 625.0, 500.0
HALF_NM = (HI_NM - LO_NM) / 2
BAND = Band(LO_NM, HI_NM)
SECOND_RADIATION_NM_K = 1.438776877e7
EDGE_NM = (0.985 * CENTER_NM, 1.015 * CENTER_NM)


def planck_photon(wavelength_nm, temperature_K):
    """Blackbody photon spectral radiance up to a constant factor."""
    return wavelength_nm**-4 / np.expm1(
        SECOND_RADIATION_NM_K / (wavelength_nm * temperature_K)
    )


def thermal(wavelength_nm, temperature_K):
    return planck_photon(wavelength_nm, temperature_K) / planck_photon(
        CENTER_NM, temperature_K
    )


def contrast(wavelength_nm):
    return 0.3 * (wavelength_nm / CENTER_NM) ** -4


def albedo_edge(wavelength_nm):
    return np.interp(wavelength_nm, EDGE_NM, [0.1, 0.5])


class Hosts:
    """A line-blanketed 5800 K star, a coarse zodi table and a trapezoid response.

    The star is a blackbody carrying 300 seeded Gaussian absorption lines of
    depth up to 0.6 on a 0.1 nm table, which stands in for a stellar atmosphere
    grid without any download.
    """

    def __init__(self):
        rng = np.random.default_rng(0)
        self.star_nm = np.arange(350.0, 650.0, 0.1)
        self.star = planck_photon(self.star_nm, 5800.0) / planck_photon(500.0, 5800.0)
        lines = zip(
            rng.uniform(350, 650, 300),
            rng.uniform(0, 0.6, 300),
            rng.uniform(0.05, 0.3, 300),
            strict=True,
        )
        for center, depth, sigma in lines:
            near = np.abs(self.star_nm - center) < 5 * sigma
            self.star[near] *= 1 - depth * np.exp(
                -0.5 * ((self.star_nm[near] - center) / sigma) ** 2
            )
        self.zodi_nm = np.array([200.0, 300, 400, 500, 700, 900, 1000, 1200])
        self.zodi = np.array([0.1, 0.45, 0.8, 1.0, 0.95, 0.75, 0.68, 0.5])
        width = HI_NM - LO_NM
        self.response_nm = np.array(
            [LO_NM, LO_NM + 0.1 * width, HI_NM - 0.1 * width, HI_NM]
        )
        self.response = np.array([0.0, 1.0, 1.0, 0.0])
        knots = np.unique(
            np.concatenate((self.star_nm, self.zodi_nm, self.response_nm, EDGE_NM))
        )
        knots = knots[(knots >= LO_NM) & (knots <= HI_NM)]
        nodes, weights = leggauss(12)
        half = np.diff(knots)[:, None] / 2
        self.wave = (knots[:-1, None] + knots[1:, None]) / 2 + half * nodes
        self.width = half * weights
        self.factor = {
            "response": np.interp(self.wave, self.response_nm, self.response),
            "star": np.interp(self.wave, self.star_nm, self.star),
            "zodi": np.interp(self.wave, self.zodi_nm, self.zodi),
        }
        self.rules = {}

    def exact(self, name, fn):
        """Reference integral of response times the named host times fn."""
        host = 1.0 if name == "response" else self.factor[name]
        return np.sum(self.width * self.factor["response"] * host * fn(self.wave))

    def rule(self, order, **kwargs):
        """Build the rule with all three weight names, cached by order."""
        if kwargs:
            return self._build(order, **kwargs)
        if order not in self.rules:
            self.rules[order] = self._build(order)
        return self.rules[order]

    def _build(self, order, **kwargs):
        return Quadrature.build(
            (BAND,),
            order=order,
            response=Transmission(self.response_nm, self.response),
            spectra={
                "star": Transmission(self.star_nm, self.star),
                "zodi": Transmission(self.zodi_nm, self.zodi),
            },
            **kwargs,
        )

    def error(self, rule, name, fn, weights=None):
        """Relative error of the rule against the reference for one name."""
        value = float(rule.integrate(fn(np.asarray(rule.nodes_nm)), weights or name)[0])
        return abs(value / self.exact(name, fn) - 1)


@pytest.fixture(scope="module")
def hosts():
    with jax.enable_x64():
        yield Hosts()
    jax.clear_caches()


def test_weights_integrate_polynomials_to_their_stated_degree(hosts):
    """Fails if a weight row omits its host spectrum or the nodes are not Gauss.

    With t the wavelength mapped to [-1, 1], order 4 must match the moments of
    response times star to degree 7 and of response and response times zodi
    to degree 3. Floating-point: 1e-13 of the zeroth moment for sums of
    25000 reference terms. The next degree must miss by more than 1e-4,
    measured at 7e-4 to 3e-3, so a secretly finer rule also fails.
    """
    rule = hosts.rule(4)
    assert rule.nodes_nm.shape == (1, 4)
    assert rule.names == ("response", "star", "zodi")
    t = (np.asarray(rule.nodes_nm[0]) - CENTER_NM) / HALF_NM
    for name, degree in (("star", 7), ("response", 3), ("zodi", 3)):
        weights = np.asarray(rule.weights(name)[0])
        assert np.all(weights > 0)
        scale = hosts.exact(name, np.ones_like)

        def miss(k, weights=weights, name=name, scale=scale):
            moment = hosts.exact(name, lambda w: ((w - CENTER_NM) / HALF_NM) ** k)
            return abs(np.sum(weights * t**k) - moment) / scale

        assert max(miss(k) for k in range(degree + 1)) < 1e-13
        assert miss(degree + 1) > 1e-4


@pytest.mark.parametrize("order", [1, 3, 5])
def test_linear_spectrum_gives_gauss_jacobi_rule(order):
    """Fails if the Stieltjes recurrence or the Jacobi matrix is wrong.

    A spectrum 1 + t on a unit response is the Jacobi weight with
    alpha = 0, beta = 1, whose nodes and weights scipy computes independently.
    Floating-point: 1e-13 absolute on quantities of order one; measured 5e-16.
    """
    with jax.enable_x64():
        ramp = Transmission([LO_NM, HI_NM], [0.0, 2.0])
        rule = Quadrature.build((BAND,), order=order, spectra={"ramp": ramp})
        nodes, weights = roots_jacobi(order, 0, 1)
        np.testing.assert_allclose(
            (np.asarray(rule.nodes_nm[0]) - CENTER_NM) / HALF_NM, nodes, atol=1e-13
        )
        np.testing.assert_allclose(
            np.asarray(rule.weights("ramp")[0]) / HALF_NM, weights, atol=1e-13
        )
        np.testing.assert_allclose(
            float(rule.weights().sum()), BAND.width_nm, rtol=1e-13
        )


def test_accuracy_ladder_on_a_half_width_band(hosts):
    """Fails if nodes or weights lose the accuracy measured before the build.

    Spike basis, relative errors at orders 4, 6, 8 with a PHOENIX 5800 K star:
    contrast under star weights 7.9e-6, 5.8e-9, 3.1e-12; zodi under zodi
    weights 1.2e-4 at order 4; thermal 600 K under response weights 3.7e-2,
    2.0e-3, 7.5e-5. The synthetic star here measures 8.8e-6, 6.2e-9, 3.3e-12;
    6.4e-5; and 3.0e-2, 1.1e-3, 3.3e-4. Bounds allow a factor of about three,
    and the thermal case must stay above 1 percent at order 4: flux-form
    terms need order 6 to 8.
    """
    errors = {
        order: (
            hosts.error(hosts.rule(order), "star", contrast),
            hosts.error(hosts.rule(order), "zodi", lambda w: (CENTER_NM / w) ** 3),
            hosts.error(hosts.rule(order), "response", lambda w: thermal(w, 600.0)),
        )
        for order in (4, 6, 8)
    }
    star, zodi, heat = zip(*(errors[order] for order in (4, 6, 8)), strict=True)
    assert 3e-6 < star[0] < 3e-5
    assert star[1] < 2e-8
    assert star[2] < 1e-11
    assert zodi[0] < 3e-4
    assert zodi[1] < 3e-6
    assert 1e-2 < heat[0] < 1e-1
    assert heat[1] < 5e-3
    assert heat[2] < 1e-3


def test_wrong_weight_row_misses_by_percent(hosts):
    """Fails if the named rows are interchangeable, so the names carry nothing.

    Negative control at order 4 on the contrast integrand, whose star weights
    reach 8.8e-6. Sampling the star at the four nodes under response weights
    measures 9e-2, and the zodi row on the star integral is off by a factor
    of order one. Both must exceed 1 percent.
    """
    rule = hosts.rule(4)
    assert hosts.error(rule, "star", contrast) < 3e-5

    def sampled(wavelength_nm):
        return np.interp(wavelength_nm, hosts.star_nm, hosts.star) * contrast(
            wavelength_nm
        )

    value = float(rule.integrate(sampled(np.asarray(rule.nodes_nm)))[0])
    assert abs(value / hosts.exact("star", contrast) - 1) > 1e-2
    assert hosts.error(rule, "star", contrast, weights="zodi") > 1e-2


def test_declared_break_restores_a_narrow_albedo_edge(hosts):
    """Fails if declared breaks do not split the weighted rule into panels.

    A 3 percent wide linear albedo edge is narrower than the band over four
    nodes: undeclared it misses by 2e-2 (spike 2.5e-2), so above 5e-3 here.
    With both corners declared the albedo is linear on each of three panels
    and the 12-node rule is exact. Floating-point: 1e-13; measured 1e-16.
    The panels must also keep the total star measure.
    """
    whole = hosts.rule(4)
    assert hosts.error(whole, "star", albedo_edge) > 5e-3
    split = hosts.rule(4, breaks_nm=EDGE_NM)
    assert split.nodes_nm.shape == (1, 12)
    assert hosts.error(split, "star", albedo_edge) < 1e-13
    np.testing.assert_allclose(
        float(split.weights("star").sum()), float(whole.weights("star").sum()), 1e-13
    )


def test_envelope_error_bounds_the_declared_temperature_range(hosts):
    """Fails if the stored error is taken at one parameter value or unscaled.

    The thermal error under response weights rises steeply toward 300 K.
    The envelope over 300 to 1000 K must bound the error at temperatures
    off the 17-sample grid, and equal 1.1 times the error at 300 K where the
    maximum sits (exact algebra, rtol 1e-9). Negative control: a probe
    stored at 600 K alone underestimates the 300 K error more than fivefold
    (measured 11x).
    """
    rule = hosts.rule(
        4,
        probes={
            "range": Probe(thermal, "response", {"temperature_K": (300.0, 1000.0)}),
            "point": Probe(
                thermal, "response", {"temperature_K": (600.0, 600.0)}, samples=2
            ),
        },
    )
    envelope = float(rule.envelope_error["range"][0])
    point = float(rule.envelope_error["point"][0])

    def actual(temperature_K):
        return hosts.error(rule, "response", lambda w: thermal(w, temperature_K))

    for temperature_K in (311.0, 437.0, 600.0, 983.0):
        assert actual(temperature_K) < envelope
    np.testing.assert_allclose(envelope, 1.1 * actual(300.0), rtol=1e-9)
    assert actual(300.0) > 5 * point
    with pytest.raises(ValueError, match="unknown weights"):
        hosts.rule(4, probes={"bad": Probe(thermal, "reference")})


def test_negative_weights_and_invalid_hosts_are_refused():
    """Fails if a rule with a negative weight or a negative host is returned.

    A 300 K blackbody as the node spectrum crowds the nodes to the red edge,
    which makes the interpolatory response weights negative.
    """
    with jax.enable_x64():
        wave = np.arange(350.0, 650.0, 0.1)
        steep = Transmission(wave, thermal(wave, 300.0) / thermal(650.0, 300.0))
        with pytest.raises(ValueError, match="response weights are negative"):
            Quadrature.build((BAND,), order=4, spectra={"steep": steep})
        with pytest.raises(ValueError, match="nonnegative"):
            Quadrature.build((BAND,), response=lambda w: 500.0 - w)
        with pytest.raises(ValueError, match="nonnegative"):
            Quadrature((BAND,), [[400.0, 600.0]], [[1.0, -1.0]])
        with pytest.raises(ValueError, match="'response'"):
            Quadrature((BAND,), [[400.0]], {"star": [[1.0]]})
        with pytest.raises(ValueError, match="'response'"):
            Quadrature.build((BAND,), spectra={"response": steep})


def test_named_rows_survive_padding_slicing_and_jit(hosts):
    """Fails if padding carries weight under any name or a row is dropped.

    Two bands with one declared break in the first give rows of 8 and 4
    nodes. Each band of the joint rule must equal the rule built alone
    (same host arithmetic, rtol 1e-13), and jit must match eager.
    """
    bands = (BAND, Band(400.0, 500.0))
    response = Transmission(hosts.response_nm, hosts.response)
    spectra = {"star": Transmission(hosts.star_nm, hosts.star)}
    rule = Quadrature.build(
        bands, order=4, breaks_nm=(550.0,), response=response, spectra=spectra
    )
    assert rule.nodes_nm.shape == (2, 8)
    values = contrast(rule.nodes_nm)
    padded = jnp.where(rule.active, values, jnp.nan)
    joint = jax.jit(lambda q, v: q.integrate(v, "star"))(rule, padded)
    np.testing.assert_allclose(joint, rule.integrate(values, "star"), rtol=1e-15)
    for i, band in enumerate(bands):
        row = rule.band(i)
        assert row.names == rule.names
        alone = Quadrature.build(
            (band,), order=4, breaks_nm=(550.0,), response=response, spectra=spectra
        )
        expected = alone.integrate(contrast(alone.nodes_nm), "star")
        np.testing.assert_allclose(joint[i], expected[0], rtol=1e-13)
        np.testing.assert_allclose(row.integrate(values[i : i + 1], "star"), expected)
    np.testing.assert_array_equal(np.asarray(rule.active).sum(-1), [8, 4])
    np.testing.assert_array_equal(
        np.asarray(rule.weights("star"))[~np.asarray(rule.active)], 0.0
    )
    constant = jnp.full(rule.nodes_nm.shape, 2.5)
    np.testing.assert_allclose(rule.mean(constant, "star"), 2.5, rtol=1e-14)
    assert (
        rule.digest
        == Quadrature.build(
            bands, order=4, breaks_nm=(550.0,), response=response, spectra=spectra
        ).digest
    )
    assert rule.digest != rule.band(0).digest
    with pytest.raises(KeyError, match="zodi"):
        rule.weights("zodi")
