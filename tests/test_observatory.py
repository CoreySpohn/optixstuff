"""One telescope, named configurations and the derived selected view."""

import dataclasses

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import GaussianTableCoronagraph, make_instrument, make_telescope
from hwoutils import model_paths
from hwoutils.constants import c, h, k_B
from hwoutils.provenance import StampMismatchError
from hwoutils.testing import ulps

import optixstuff as ox
from optixstuff.spectral import Band, Quadrature, Transmission

AREA = 27.3125  # a distinctive area so that its copies can be counted


def _pupil(diameter_m=6.0, area=AREA, pupil_id=None):
    return ox.SimplePupil(
        circumscribed_diameter_m=diameter_m,
        collecting_area_m2=area,
        pupil_id=pupil_id,
    )


def _detector(**kwargs):
    return ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(64, 96), **kwargs)


def _config(name, elements=(), **kwargs):
    kwargs.setdefault("detector", _detector())
    return ox.Instrument(name, attenuating_elements=elements, **kwargs)


def _two_filters():
    """One telescope, a red and a blue filter configuration, as an Observatory."""
    return ox.Observatory(
        ox.Telescope(_pupil(), attenuating_elements=(ox.ConstantThroughput(0.9),)),
        (
            _config("red", (ox.ConstantThroughput(0.5),)),
            _config("blue", (ox.ConstantThroughput(0.25),), flux_scale=2.0),
        ),
    )


def _with_pupil_id(pupil_id, bound=False, **kwargs):
    """Table coronagraph that declares ``pupil_id``, optionally returning a copy."""

    class Declared(GaussianTableCoronagraph):
        @property
        def pupil_id(self):
            return pupil_id

        if bound:

            def for_telescope(self, telescope):
                super().for_telescope(telescope)
                return jax.tree.map(lambda leaf: leaf, self)

    return Declared(**kwargs)


@vv.case("observatory", "code-verification")
def test_t1_configurations_share_one_collecting_area():
    """Fails if a configuration keeps its own copy of the pupil area.

    Truth: the effective area is linear in the one pupil leaf. Doubling a power
    of two is exact, and the two products below are single roundings, so
    the basis is exact algebra within 2 ULP of the expected value.
    """
    observatory = _two_filters()
    doubled = eqx.tree_at(
        lambda o: o.telescope.pupil.collecting_area_m2, observatory, 2 * AREA
    )
    for name in observatory.names:
        before = observatory.select(name).effective_area_m2(550.0)
        after = doubled.select(name).effective_area_m2(550.0)
        np.testing.assert_allclose(
            after, 2 * before, rtol=0, atol=float(ulps(2, 2 * before))
        )


@vv.case("observatory", "code-verification")
def test_t2_gradient_reaches_the_one_area_leaf():
    """Fails if the area appears twice or its cotangent misses a configuration.

    Truth: d(sum A_i)/dA = sum of the throughput products (telescope 0.9 times
    configuration train times flux scale), a closed form. Basis: exact algebra,
    4 ULP of the expected sum; the leaf count is exact.
    """
    observatory = _two_filters()

    def total(area):
        edited = eqx.tree_at(
            lambda o: o.telescope.pupil.collecting_area_m2, observatory, area
        )
        return sum(
            edited.select(name).effective_area_m2(550.0) for name in edited.names
        )

    expected = 0.9 * 0.5 + 0.9 * 0.25 * 2.0
    np.testing.assert_allclose(
        jax.grad(total)(jnp.asarray(AREA)),
        expected,
        rtol=0,
        atol=float(ulps(4, expected)),
    )
    copies = [leaf for leaf in jax.tree.leaves(observatory) if leaf == AREA]
    assert len(copies) == 1


@vv.case("observatory", "code-verification")
def test_t3_trains_multiply_and_thermal_ignores_them():
    """Fails if a train is dropped from the total or enters the thermal rate.

    Truth: 0.9 * 0.8 in closed form, basis floating-point, 2 ULP. The thermal
    rate is the same expression whatever the trains or the flux scale are, so
    equality is exact.
    """
    warm = ox.WarmEmission(290.0, 0.5)

    def build(telescope_throughput, instrument_throughput, scale):
        return ox.Observatory(
            ox.Telescope(
                _pupil(),
                attenuating_elements=(ox.ConstantThroughput(telescope_throughput),),
            ),
            (
                _config(
                    "a",
                    (ox.ConstantThroughput(instrument_throughput),),
                    warm_emission=warm,
                    flux_scale=scale,
                ),
            ),
        ).select("a")

    base = build(0.9, 0.8, 1.0)
    np.testing.assert_allclose(
        base.system_throughput(550.0), 0.72, rtol=0, atol=float(ulps(2, 0.72))
    )
    np.testing.assert_allclose(
        base.effective_area_m2(550.0),
        AREA * 0.72,
        rtol=0,
        atol=float(ulps(2, AREA * 0.72)),
    )
    thermal = base.thermal_photon_rate_per_sr(jnp.array([800.0, 1200.0]))
    for other in (build(0.3, 0.8, 1.0), build(0.9, 0.1, 1.0), build(0.9, 0.8, 3.0)):
        np.testing.assert_array_equal(
            other.thermal_photon_rate_per_sr(jnp.array([800.0, 1200.0])), thermal
        )


@vv.case("observatory", "code-verification")
def test_t4_thermal_rate_is_planck_times_emissivity_times_area():
    """Fails if the thermal rate drops a factor or reads the throughput.

    Truth: 2 c / lambda^4 / expm1(h c / (lambda k T)) written from the
    constants, in host float64, times emissivity 0.638 and the geometric area.
    Basis: eight operations at the exponent scale (the exponential amplifies
    the error of the wavelength-temperature product) plus eight at unit scale.
    """
    wavelength = np.array([500.0, 1000.0, 2000.0])
    observatory = ox.Observatory(
        ox.Telescope(_pupil(), attenuating_elements=(ox.ConstantThroughput(0.4),)),
        (
            _config(
                "a",
                (ox.ConstantThroughput(0.7),),
                warm_emission=ox.WarmEmission(290.0, 0.638),
                flux_scale=1.7,
            ),
        ),
    )
    lam_m = wavelength * 1e-9
    exponent = h * c / (lam_m * k_B * 290.0)
    radiance = 2 * c / lam_m**4 / np.expm1(exponent) * 1e-9
    np.testing.assert_allclose(
        observatory.select("a").thermal_photon_rate_per_sr(wavelength),
        radiance * 0.638 * AREA,
        rtol=float(ulps(8, 1 + exponent.max())),
        atol=0,
    )


@vv.case("observatory", "code-verification")
def test_t5_telescope_breaks_reach_the_view_and_the_quadrature():
    """Fails if the view or the quadrature drops the telescope's breaks.

    Truth: set algebra on host floats. The telescope declares 500 and 600 nm,
    the configuration declares 520 nm; the view holds all three exactly and the
    band splits at each of them.
    """
    mirror = ox.TransmissiveElement(Transmission([500.0, 600.0], [0.9, 0.9]))
    filt = ox.TransmissiveElement(Transmission([520.0, 700.0], [0.5, 0.5]))
    observatory = ox.Observatory(
        ox.Telescope(_pupil(), attenuating_elements=(mirror,)),
        (_config("a", (filt,)),),
    )
    view = observatory.select("a")
    assert observatory.telescope.spectral_breaks_nm() == (500.0, 600.0)
    np.testing.assert_array_equal(
        view.spectral_breaks_nm(), [500.0, 520.0, 600.0, 700.0]
    )
    (pieces,) = Quadrature.build((Band(450.0, 650.0),), view).pieces
    edges = {edge for piece in pieces for edge in piece[:2]}
    assert {500.0, 520.0, 600.0} <= edges


@vv.case("observatory", "code-verification")
def test_t6_field_radius_follows_the_configuration_hook():
    """Fails if the view reads the field from anywhere but the instance hook.

    Truth: the released formula written out, half the detector diagonal over
    the lambda/D angle at the band center. Basis: floating-point, 2 ULP. An
    override returning 7.0 must be honored exactly by the view and by
    ``Quadrature.build`` (the same layout as passing 7.0 explicitly).
    """
    detector = _detector()
    band = Band(450.0, 550.0)
    expected = 0.5 * np.hypot(64, 96) * 0.01 / (500.0 * 1e-9 / 6.0 * 180 / np.pi * 3600)
    view = ox.Observatory(ox.Telescope(_pupil()), (_config("a", detector=detector),))
    np.testing.assert_allclose(
        view.select("a").spectral_field_lod(band),
        expected,
        rtol=0,
        atol=float(ulps(2, expected)),
    )

    class Constant(ox.Instrument):
        def spectral_field_lod(self, band, *, telescope, detector):
            return 7.0

    custom = ox.Observatory(
        ox.Telescope(_pupil()), (Constant("a", detector=detector),)
    ).select("a")
    assert custom.spectral_field_lod(band) == 7.0
    wide = Band(495.0, 605.0)
    built = Quadrature.build((wide,), custom, field_lod=None)
    assert built.pieces == Quadrature.build((wide,), field_lod=7.0).pieces


@vv.case("observatory", "code-verification")
def test_t7_curve_only_configuration():
    """Fails if no coronagraph breaks construction, the field or the verdict.

    Truth: with no coronagraph the field uses the circumscribed pupil diameter
    (closed form, 2 ULP) and nothing verifies the prescription.
    """
    observatory = ox.Observatory(ox.Telescope(_pupil(8.0)), (_config("curves"),))
    view = observatory.select("curves")
    assert view.coronagraph is None
    assert view.pupil_match == "unverified"
    expected = 0.5 * np.hypot(64, 96) * 0.01 / (500.0 * 1e-9 / 8.0 * 180 / np.pi * 3600)
    np.testing.assert_allclose(
        view.spectral_field_lod(Band(450.0, 550.0)),
        expected,
        rtol=0,
        atol=float(ulps(2, expected)),
    )


@vv.case("observatory", "code-verification")
def test_t8_frame_mismatch_names_the_configuration():
    """Fails if a coronagraph frame 1 percent off the pupil is accepted.

    Contract test, exact. A traced diameter is skipped: the same build under
    ``jit`` with a traced pupil diameter does not raise.
    """
    coronagraph = GaussianTableCoronagraph(diameter_m=6.06)
    with pytest.raises(ValueError, match="instrument 'imager' coronagraph"):
        ox.Observatory(
            ox.Telescope(_pupil(6.0)), (_config("imager", coronagraph=coronagraph),)
        )

    @jax.jit
    def traced(diameter):
        pupil = ox.SimplePupil(
            circumscribed_diameter_m=diameter, collecting_area_m2=AREA
        )
        observatory = ox.Observatory(
            ox.Telescope(pupil), (_config("imager", coronagraph=coronagraph),)
        )
        return observatory.select("imager").effective_area_m2(550.0)

    assert float(traced(jnp.asarray(6.0))) == AREA


@vv.case("observatory", "code-verification")
def test_t9_multiband_children_are_all_bound():
    """Fails if a multiband product binds only its reference child.

    Contract, exact identity. All children matched returns ``self``. A second
    child whose binding returns a distinct object lands in slot 1 and the first
    child stays in slot 0. A child of the wrong diameter raises when the
    product is bound directly.
    """
    telescope = make_telescope(6.0)
    first = GaussianTableCoronagraph(bandpass=Band(450.0, 550.0))
    plain = ox.MultiBandCoronagraph(
        [first, GaussianTableCoronagraph(bandpass=Band(550.0, 650.0))]
    )
    assert plain.for_telescope(telescope) is plain
    second = _with_pupil_id(None, bound=True, bandpass=Band(550.0, 650.0))
    mixed = ox.MultiBandCoronagraph([first, second])
    result = mixed.for_telescope(telescope)
    assert result is not mixed
    assert result.coronagraphs[0] is first
    assert result.coronagraphs[1] is not second
    assert result.reference_wavelength_nm == mixed.reference_wavelength_nm
    wrong = ox.MultiBandCoronagraph(
        [first, GaussianTableCoronagraph(diameter_m=6.06, bandpass=Band(550.0, 650.0))]
    )
    with pytest.raises(ValueError, match="frame"):
        wrong.for_telescope(telescope)


@vv.case("observatory", "code-verification")
def test_t10_pupil_identity_outcomes():
    """Fails if known, unknown or conflicting identifiers are mixed up.

    The comparison table of the provenance stamps, exact: equal known is
    matched, known different raises with both identifiers, either side unknown
    or both unknown is unverified.
    """

    def verdict(coronagraph_id, pupil_id):
        observatory = ox.Observatory(
            ox.Telescope(_pupil(pupil_id=pupil_id)),
            (_config("a", coronagraph=_with_pupil_id(coronagraph_id)),),
        )
        return observatory.select("a").pupil_match

    assert verdict("v1", "v1") == "matched"
    assert verdict(None, "v1") == "unverified"
    assert verdict("v1", None) == "unverified"
    assert verdict(None, None) == "unverified"
    with pytest.raises(StampMismatchError, match=r"v1.*v2|v2.*v1"):
        verdict("v1", "v2")
    multi = ox.MultiBandCoronagraph(
        [
            _with_pupil_id("v1", bandpass=Band(450.0, 550.0)),
            _with_pupil_id("v1", bandpass=Band(550.0, 650.0)),
        ]
    )
    selected = ox.Observatory(
        ox.Telescope(_pupil(pupil_id="v1")), (_config("a", coronagraph=multi),)
    ).select("a")
    assert selected.pupil_match == "matched"


@vv.case("observatory", "code-verification")
def test_t11_names_and_selection():
    """Fails if a bad or repeated name is accepted or select misroutes.

    Contract, exact. Duplicate, empty, all-digit and bracketed names raise;
    ``select`` of a missing name lists the names; ``select`` works under jit
    and vmap and picks the named configuration, not the first; a lone
    ``Instrument`` raises a ``TypeError`` that shows the tuple form.
    """
    telescope = ox.Telescope(_pupil())
    for bad in ("", "7", "a[0]", "a b", "1x"):
        with pytest.raises(ValueError, match="name"):
            _config(bad)
    with pytest.raises(ValueError, match="repeat"):
        ox.Observatory(telescope, (_config("a"), _config("a")))
    with pytest.raises(TypeError, match=r"\(instrument,\)"):
        ox.Observatory(telescope, _config("a"))
    with pytest.raises(ValueError, match="at least one"):
        ox.Observatory(telescope, ())
    observatory = _two_filters()
    assert observatory.names == ("red", "blue")
    with pytest.raises(ValueError, match=r"\['red', 'blue'\]"):
        observatory.select("missing")
    assert observatory.select("blue").name == "blue"
    assert observatory.select("blue") is not observatory.select("blue")
    expected = {"red": AREA * 0.9 * 0.5, "blue": AREA * 0.9 * 0.25 * 2.0}
    for name, value in expected.items():
        jitted = eqx.filter_jit(lambda o, n=name: o.select(n).effective_area_m2(550.0))
        np.testing.assert_allclose(
            jitted(observatory), value, rtol=0, atol=float(ulps(4, value))
        )
        scales = jnp.array([1.0, 2.0])
        mapped = jax.vmap(
            lambda s, n=name: (
                eqx.tree_at(lambda o: o.instruments[0].flux_scale, observatory, s)
                .select(n)
                .effective_area_m2(550.0)
            )
        )(scales)
        assert mapped.shape == (2,)


@vv.case("observatory", "code-verification")
def test_t12_the_view_owns_nothing():
    """Fails if the view stores a copy or an owner path points elsewhere.

    Identity checks, exact: the view's accessors return the owners' objects,
    every owner path resolves to the object it names, a private detector per
    configuration and a detector object shared by two configurations both
    select correctly, and a write through one configuration leaves the other.
    """
    shared = _detector()
    private = _detector()
    observatory = ox.Observatory(
        ox.Telescope(_pupil()),
        (
            _config("a", detector=shared),
            _config("b", detector=shared),
            _config("c", detector=private),
        ),
    )
    assert [f.name for f in dataclasses.fields(ox.SelectedInstrument)] == [
        "telescope",
        "instrument",
    ]
    for index, name in enumerate(observatory.names):
        view = observatory.select(name)
        assert view.detector is observatory.instruments[index].detector
        assert view.pupil is observatory.telescope.pupil
        leaves = jax.tree.leaves(view)
        assert len(leaves) == len(jax.tree.leaves(view.telescope)) + len(
            jax.tree.leaves(view.instrument)
        )
        for attribute, owner in (
            ("", view.instrument),
            ("detector", view.detector),
            ("coronagraph", None),
            ("disperser", None),
            ("speckle", None),
            ("warm_emission", None),
            ("flux_scale", view.flux_scale),
            ("pupil", view.pupil),
            ("telescope", view.telescope),
        ):
            path = view.owner_path(attribute)
            canonical = model_paths.canonical_path(observatory, path)
            assert model_paths.resolve(observatory, canonical) is owner
    assert observatory.select("c").owner_path("detector") == "instruments[c].detector"
    assert observatory.select("a").owner_path("pupil") == "telescope.pupil"
    with pytest.raises(ValueError, match="attribute"):
        observatory.select("a").owner_path("primary")
    replaced = eqx.tree_at(
        lambda o: o.instruments[0].detector, observatory, _detector()
    )
    shared_leaves = jax.tree.leaves(shared)
    kept = jax.tree.leaves(replaced.select("b").detector)
    assert all(x is y for x, y in zip(kept, shared_leaves, strict=True))
    changed = jax.tree.leaves(replaced.select("a").detector)
    assert not all(x is y for x, y in zip(changed, shared_leaves, strict=True))


@vv.case("observatory", "code-verification")
def test_t13_speckle_route_keeps_the_released_sign(linear_speckle):
    """Fails if the view returns another field or flips the speckle sign.

    Truth: the intensity change |E + dE|^2 - |E|^2 for the released field
    (computed here from its arrays), basis floating-point with 32 operations at
    the nominal scale, as in the released law test.
    """
    field = linear_speckle
    view = ox.Observatory(
        ox.Telescope(_pupil()), (_config("a", speckle=field),)
    ).select("a")
    assert view.speckle is field
    time = 0.17
    eps = np.array([0.03, -0.02]) + np.array([0.05, 0.04]) * np.sin(
        time * np.array([2.0, 3.0])
    )
    nominal, g = np.asarray(field.response.e_nom), np.asarray(field.response.G)
    expected = (
        (np.abs(nominal + np.einsum("m,myx->yx", eps, g)) ** 2 - np.abs(nominal) ** 2)
        * field.grid.pixel_area
        / 7.3
    )
    np.testing.assert_allclose(
        view.speckle.realize(wavelength_nm=500.0, elapsed_s=time).data,
        expected,
        rtol=0,
        atol=float(
            ulps(32, np.max(np.abs(nominal) ** 2) * field.grid.pixel_area / 7.3)
        ),
    )


def test_t14_empty_train_dtype_follows_the_wavelength():
    """Fails if an empty train returns a wider or strong one for a narrow input.

    Exact dtype basis. A float32 array gives float32 ones; a Python scalar gives
    a weak one, so it never widens a float32 product.
    """
    telescope = ox.Telescope(_pupil())
    narrow = telescope.throughput(jnp.asarray([500.0, 600.0], dtype=jnp.float32))
    assert narrow.dtype == jnp.float32
    np.testing.assert_array_equal(narrow, [1.0, 1.0])
    scalar = telescope.throughput(550.0)
    assert scalar.weak_type
    assert (scalar * jnp.float32(2)).dtype == jnp.float32
    view = ox.Observatory(telescope, (_config("a"),)).select("a")
    assert view.system_throughput(jnp.float32(550.0)).dtype == jnp.float32


@vv.case("observatory", "code-verification")
def test_t15_pupil_diameters_and_identifier(precision):
    """Fails if a pupil frame carries another diameter than its convention.

    Truth: closed forms. A simple pupil returns the two diameters as given; a
    two-ring segmented pupil with 1.65 m segments and no gap has circumscribed
    diameter 2 hypot(sqrt(3) n p / 2 + a, n p / 2) and inscribed diameter
    2 sqrt((sqrt(3) n p / 2 + a / 2)^2 + (sqrt(3) a / 2)^2) with a = 0.825,
    p = sqrt(3) a. Basis: 16 float64 epsilons. The identifier is static,
    defaults to None and rejects an empty string.
    """
    simple = ox.SimplePupil(
        circumscribed_diameter_m=6.0,
        inscribed_diameter_m=5.0,
        collecting_area_m2=20.0,
        pupil_id="p1",
    )
    assert simple.frame(500.0).diameter_m == 6.0
    assert simple.frame(500.0, "inscribed").diameter_m == 5.0
    n, a = 2, 0.825
    p = np.sqrt(3) * a
    segmented = ox.SegmentedPupil(n_rings=n, segment_point_to_point_m=2 * a)
    circ = 2 * np.hypot(np.sqrt(3) * n * p / 2 + a, n * p / 2)
    insc = 2 * np.hypot(np.sqrt(3) * n * p / 2 + a / 2, np.sqrt(3) * a / 2)
    np.testing.assert_allclose(
        segmented.frame(500.0).diameter_m, circ, rtol=16 * np.finfo(float).eps
    )
    np.testing.assert_allclose(
        segmented.frame(500.0, "inscribed").diameter_m,
        insc,
        rtol=16 * np.finfo(float).eps,
    )
    assert simple.pupil_id == "p1"
    assert segmented.pupil_id is None
    assert (
        ox.SegmentedPupil(
            n_rings=1, segment_point_to_point_m=1.0, pupil_id="seg"
        ).pupil_id
        == "seg"
    )
    assert all("p1" not in repr(leaf) for leaf in jax.tree.leaves(simple))
    for bad in ("", 3):
        with pytest.raises(ValueError, match="pupil_id"):
            ox.SimplePupil(
                circumscribed_diameter_m=6.0, collecting_area_m2=1.0, pupil_id=bad
            )


@vv.case("observatory", "code-verification")
def test_t16_replacing_the_pupil_replaces_the_identity():
    """Fails if the identity is stored away from the pupil it names.

    Contract, exact. Two pupils of equal diameter and different identifiers:
    swapping the pupil with ``eqx.tree_at`` turns a matched coronagraph into a
    ``StampMismatchError`` at ``select``, with no second write.
    """
    coronagraph = _with_pupil_id("v1")
    observatory = ox.Observatory(
        ox.Telescope(_pupil(pupil_id="v1")), (_config("a", coronagraph=coronagraph),)
    )
    assert observatory.select("a").pupil_match == "matched"
    swapped = eqx.tree_at(
        lambda o: o.telescope.pupil, observatory, _pupil(pupil_id="v2")
    )
    with pytest.raises(StampMismatchError, match=r"instruments\[a\]\.coronagraph"):
        swapped.select("a")


@vv.case("observatory", "code-verification")
def test_t17_geometry_reads_follow_the_resolved_detector():
    """Fails if a hook reads the configuration's inline detector.

    Truth: a view double whose resolved detector has another QE break (700 nm)
    and another pixel scale. The view's breaks and field radius follow it, exact
    for the break and floating-point (2 ULP) for the field.
    """

    class Resolved(ox.SelectedInstrument):
        @property
        def detector(self):
            return ox.Detector(
                pixel_scale_arcsec=0.02,
                shape=(64, 96),
                qe_curve=([650.0, 700.0, 750.0], [0.5, 0.9, 0.5]),
            )

    observatory = ox.Observatory(ox.Telescope(_pupil()), (_config("a"),))
    inline = observatory.select("a")
    double = Resolved(observatory.telescope, observatory.instruments[0])
    assert 700.0 not in inline.spectral_breaks_nm()
    assert 700.0 in double.spectral_breaks_nm()
    band = Band(450.0, 550.0)
    np.testing.assert_allclose(
        double.spectral_field_lod(band),
        2 * inline.spectral_field_lod(band),
        rtol=0,
        atol=float(ulps(2, 2 * inline.spectral_field_lod(band))),
    )


@vv.case("observatory", "code-verification")
def test_t18_element_names_are_optional_and_unique_when_given():
    """Fails if unnamed elements collide or a duplicate explicit name passes.

    Contract, exact. Two unnamed mirrors and a nameless filter construct on both
    trains and are addressed by position; a repeated explicit name raises with
    the same message on the telescope and on a configuration.
    """
    mirrors = (ox.ConstantThroughput(0.9), ox.ConstantThroughput(0.9))
    filt = ox.TransmissiveElement(Transmission([500.0, 600.0], [0.5, 0.6]))
    observatory = ox.Observatory(
        ox.Telescope(_pupil(), attenuating_elements=mirrors),
        (_config("a", (*mirrors, filt)),),
    )
    assert observatory.telescope.attenuating_elements[1].name is None
    path = "instruments[a].attenuating_elements[2]"
    assert model_paths.resolve(observatory, path) is filt
    twin = (ox.ConstantThroughput(0.9, name="m"), ox.ConstantThroughput(0.8, name="m"))
    with pytest.raises(ValueError, match="repeat") as on_telescope:
        ox.Telescope(_pupil(), attenuating_elements=twin)
    with pytest.raises(ValueError, match="repeat") as on_configuration:
        _config("a", twin)
    assert str(on_telescope.value) == str(on_configuration.value)


@vv.case("observatory", "code-verification")
def test_t19_thermal_dtype_is_the_same_with_and_without_emission(precision):
    """Fails if zero emission and nonzero emission differ in dtype.

    Exact dtype basis, asserted in each lane: float32 operands, and where x64 is
    available also float64 wavelengths and a float32 wavelength against a strong
    float64 area (the mixed operands that decide the joint dtype). The host field
    radius is computed eagerly as a float.
    """
    cases = [(np.float32, AREA)]
    if precision == np.float64:
        cases += [(np.float64, AREA), (np.float32, np.float64(AREA))]
    warm = ox.WarmEmission(290.0, 0.5)
    for dtype, area in cases:
        wavelength = np.asarray([800.0, 1200.0], dtype=dtype)
        telescope = ox.Telescope(_pupil(area=area))
        off = ox.Observatory(telescope, (_config("a"),)).select("a")
        on = ox.Observatory(telescope, (_config("a", warm_emission=warm),)).select("a")
        zero = off.thermal_photon_rate_per_sr(wavelength)
        emitted = on.thermal_photon_rate_per_sr(wavelength)
        assert zero.dtype == emitted.dtype == jnp.result_type(wavelength, area, 1.0)
    assert isinstance(off.spectral_field_lod(Band(450.0, 550.0)), float)


def test_make_instrument_fixture_builds_one_imager():
    """Fails if the shared fixture stops building one configuration named imager."""
    observatory = make_instrument(GaussianTableCoronagraph())
    assert observatory.names == ("imager",)
    assert observatory.select("imager").pupil_match == "unverified"
