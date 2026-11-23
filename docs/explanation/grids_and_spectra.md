# Grids, measures and spectral quadrature

## Spectral integration

Supply wavelength intervals by their endpoints in nm. Build the quadrature
once on the host, then evaluate each spectrum on the same array of nodes.
For example, the following integrates a piecewise-linear transmission curve
and returns its mean response in two bands.

```python
from optixstuff.spectral import Band, Quadrature, Transmission

bands = (Band(400, 550), Band(550, 700))
response = Transmission([400, 500, 700], [0, 1, 0])
rule = Quadrature.build(bands, response)
samples = response(rule.nodes_nm)
equivalent_widths_nm = rule.integrate(samples)
mean_transmissions = rule.mean(samples)
```

Models declare slope changes and jumps with a host method
`spectral_breaks_nm()` returning a one-dimensional NumPy array.
Smooth models return an empty array. Each band is split at all interior
breaks before applying Gauss-Legendre integration. The first two axes of
the samples index bands and nodes. Integration preserves all trailing axes.
`rule.band(i)` selects a single band while retaining the band axis.

Every node carries one plain weight, and the rule never depends on a source
spectrum. A rough tabulated spectrum, such as a stellar atmosphere model, is
integrated only as well as its samples at the nodes represent it: its lines
are narrower than any affordable node spacing, so more nodes do not converge.
Reduce such a table to node-effective values (below), or pass it as a model
so its knots become breaks. A factor that is evaluated at the nodes and has a
feature narrower than about the band width over the order needs a declared
break.

The node count of a piece follows the response it is read through. A pixel at
separation s from the axis, in lambda/D at the band center c, sees a
diffraction pattern slide across a piece from a to b by
`s * c * (1 / a - 1 / b)` chromatic cycles. `field_lod` is the largest separation at which the response is read.
A piece that holds no cycles keeps exactly `order` nodes; otherwise it gets
`max(order, 2 * ceil((order + 3 * cycles) / 2))` nodes. The default of zero
suits a response read at one band-center separation for every node, such as
a performance curve. An image out to the corner of a 128 by 128 detector at
0.25 lambda/D reads separations up to 22.6 lambda/D. An `Instrument` declares
the radius its detector spans through `spectral_field_lod(band)`, the detector
half-diagonal in lambda/D at the band center, and `field_lod=None` takes it.
For a smooth response, a caller who wants the rings of a source at separation
`q` resolved everywhere passes the half-diagonal plus `q`; on a tabulated
product the extra nodes do not remove the floor set by how the pixels read the
table:

```python
import math

from optixstuff.spectral import Band, Quadrature

band = Band(540, 660)
curves = Quadrature.build((band,), order=4)
image = Quadrature.build((band,), order=4, field_lod=16 * math.sqrt(2))
assert curves.pieces == (((540.0, 660.0, 4),),)
assert image.pieces == (((540.0, 660.0, 18),),)
```

`rule.pieces` records the edges and the order of every piece, and
`rule.layout_digest` identifies the bands and pieces; both are static, so
they can be read inside a traced program. `field_lod` also takes one value per
band, or `None` for the largest value that a model declares through
`spectral_field_lod(band)`. Order 4 serves smooth integrands. A steep
flux-form term, such as thermal emission on the Wien side, needs order 6 to 8
in bands 20 percent or wider.

### Rough tabulated sources

On a piece with nodes `lambda_k`, weights `w_k` and Lagrange polynomials
`l_k`, a source `phi` has the node-effective value
`phibar_k = (1 / w_k) * integral of phi * l_k`. The integral of `phi * R` over
the piece is then `sum_k w_k * phibar_k * R(lambda_k)`, exact in `phi` however
rough it is, and exact for any response `R` of degree below the piece's order.
A tabulated source is linear in its values through its own interpolant, so
`phibar` is a fixed linear map of the table values. `Reduction.build` builds
that map on the host from the nodes and the knots, never from a value, by
transposing the source's interpolant; `apply` acts on the live values inside a
traced program, so a fitted table or a scale keeps its gradient.

A source declares the protocol `SpectralTable`: its knots through
`spectral_breaks_nm()`, a static `knot_digest` made with `digest_knots`, its
polynomial degree between knots and, optionally, `spectral_interpolant()`.
`Resolution.build` builds one rule for every pair of a quadrature and a table
grid, and `Resolution.rule` finds it by the two static digests, also under
`jit`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from optixstuff.spectral import Band, Quadrature, Resolution, digest_knots


class Table(eqx.Module):
    wavelength_nm: jax.Array
    values: jax.Array
    knot_digest: str = eqx.field(static=True)
    spectral_degree = 1

    def __init__(self, wavelength_nm, values):
        self.wavelength_nm = jnp.asarray(wavelength_nm)
        self.values = jnp.asarray(values)
        self.knot_digest = digest_knots(self.wavelength_nm, "linear")

    def spectral_breaks_nm(self):
        return np.asarray(self.wavelength_nm)

    def spectral_interpolant(self):
        return lambda values, nm: jnp.interp(nm, self.wavelength_nm, values)


knots = np.linspace(480.0, 620.0, 1401)
lines = 1 - 0.5 * (np.sin(7 * knots) > 0.9)
star = Table(knots, lines)
rule = Quadrature.build((Band(500, 550), Band(550, 600)), order=4)
resolution = Resolution.build(rule, star)


@jax.jit
def band_flux(resolution, star):
    node_values = resolution.rule(rule, star).apply(star.values)
    return rule.integrate(node_values)


exact = [
    np.trapezoid(np.interp(g, knots, lines), g)
    for g in (np.linspace(500, 550, 50001), np.linspace(550, 600, 50001))
]
np.testing.assert_allclose(band_flux(resolution, star), exact, rtol=1e-6)
```

Every piece of every band must lie inside each table, from its first knot to
its last: a band that leaves a table, or straddles its end, is refused when the
rule is built. End the band at the table end, or load the table over the band.
A table whose knots are all piece edges takes the `"identity"` kind, which
holds no arrays, because a Gauss rule on each piece is already exact there; a
source that is not one table (a product of two tables) takes the `"fine"`
kind and is evaluated at `Reduction.fine_nm`. A rule belongs to the full
quadrature: a route that works band by band computes the node values once and
carries the rows of each band. `Exposure.resolution` carries the rules beside
the quadrature, and `Exposure.reads` names the knot leaves they were built
from.

`Quadrature.monochromatic(wavelength_nm)` represents an already integrated
rate at one wavelength. Its single node has unit weight. A mean density
is undefined for this zero-width channel.

## Measures and conservative integration

An `Image` has a numerator unit and a spatial measure. For example, photon
radiance on an arcsec grid has quantity `"ph s^-1 m^-2 nm^-1"` and measure
`Measure.density("sr")`. Its `integrated()` method multiplies the center
sample by the cell's solid angle in steradians. A density with denominator
`"grid"` instead uses the squared coordinate unit. To use steradians with a
lambda/D grid, convert the grid through its frame at the relevant wavelength.
Carry cell integrals through the coordinate change, then call `density("sr")`
on the resulting arcsec image. Transmission and contrast use
`Measure.intensive()` and retain their values under center sampling.

Use explicit source and target geometry to integrate a PSF. Here `psf` is a
100 by 100 array of photon rates integrated over source pixels.

```python notest
from optixstuff.grids import (
    Image,
    Measure,
    PixelGrid,
    footprint_overlap,
    integrate_footprints,
)

source = PixelGrid.regular(psf.shape, 1.0, "arcsec")
target = PixelGrid.regular((40, 40), 2.5, "arcsec")
image = Image(psf, source, (), Measure.integral(), "ph/s")
overlap = footprint_overlap(source, target)
rebinned = integrate_footprints(image, overlap)
```

Each target cell receives the fraction of every source integral covered by
its footprint. Density samples use the midpoint rule within their cells.
The field is zero beyond the source. A target that tiles the source conserves
the total flux to floating-point precision. Rotation, shear, different
centers, and leading array dimensions are supported. Construct overlap
weights once on the host and reuse them with traced image values. To
collect several grids, pass a sequence of target grids. The result has a
final cell axis in grid order, with each grid flattened in row-major order.

## Reconstructions and the pixel read

An `Image` names the reconstruction that fills the space between its sample
centers. The table lists each name, its margin in source cells (the value of
`reconstruction_margin`), whether its reach is finite, and how `on_grid` reads a
density image that declares it.

| Name | Kernel | Margin | Finite reach | `on_grid` read of a density |
|---|---|---|---|---|
| `nearest` | value of the cell | 0 | yes | midpoint cell integral, then `resample` |
| `linear` | tent | 0.5 | yes | midpoint cell integral, then `resample` |
| `cubic` | Keys cubic convolution, a = -1/2 | 2 | yes | midpoint cell integral, then `resample` |
| `cubic_spline` | not-a-knot cubic B-spline | 2 | no | integral of the spline over each pixel |
| `band_limited` | zero-padded Fourier interpolant | 2 | no | midpoint cell integral, then `resample` |

For a kernel of finite reach the margin is exact: a pixel whose footprint lies
that far inside the table never reads the zero beyond it. The cubic spline and
the band-limited interpolant have no finite reach, so for them the margin marks
the support boundary and is not an accuracy bound. `coverage` returns these
margins from the pixel corners.

### The cubic spline

A table that holds the intensity at each cell center, times the cell area, is a
set of point samples. A detector pixel measures the integral of the intensity
over its footprint, and the stored sample is not that integral, not even on the
native pitch. `on_grid` therefore reads a density that declares `cubic_spline`
as the integral of the reconstructed density over each target pixel, in one
form at every pitch, and returns a cell integral in the area the density is per
(grid units squared of the native grid, or steradians). The read is two matrix
products per image, one per axis, each built from the closed-form antiderivative
of the B-spline with the prefilter folded in. It needs axis-aligned grids and
raises `NotImplementedError` for a rotated or sheared target. `resample` keeps
its own contract for the same image: it samples the spline at the target centers
and returns a density.

The spline interpolates the samples, with coefficients that are a linear map of
the table alone: the not-a-knot end condition makes the third derivative
continuous at the second and the second-to-last sample. It reproduces cubic
polynomials over the whole footprint. It suits products that are band-limited
at the table pitch, including tables cut off inside a bright field, such as a
coronagraph stellar map. On a pupil-limited speckle field with power up to the
band edge, read between 3 and 26 lambda/D from a table at 0.25 lambda/D, the
read is wrong by about 0.05 percent of the pixel value at the median and 0.3
percent at the 95th percentile, over pixel pitches from 0.125 to 0.75 lambda/D.
In float32 the matrices are built in float64 and cast, and the read moves by
less than 1e-6 of the pixel value.

Its limits:

- Every coefficient depends on every sample, with a weight that falls by about
  0.268 per cell, so pixels near the table edge depend on the end condition. On
  a public stellar map tabulated at 0.25 lambda/D, replacing the not-a-knot
  condition by a mirror or a zero closure changes a pixel by at most the
  amounts below (over pitches from 0.125 to 0.5 lambda/D; the largest values
  are at 0.125), and the sum over the supported pixels by up to 1.5e-4.

  | Distance from the table edge | Largest change |
  |---|---|
  | 2 to 3 cells | 13 percent |
  | 3 to 4 cells | 6.2 percent |
  | 4 to 6 cells | 4.6 percent |
  | 6 to 8 cells | 0.55 percent |
  | 8 to 12 cells | 0.09 percent |

- A hard edge rings. Across a step the spline overshoots by up to 8 percent of
  the step and, beyond 2 cells, is less accurate than the Keys cubic, which is
  exact there. A product with a hard edge or a zero-filled data mask declares
  `linear` or `cubic`, never `cubic_spline`: zero-filling a stellar map beyond a
  straight mask edge moves the spline read by 2.7 percent 2 to 3 cells from the
  mask and by 0.8 percent 3 to 4 cells from it, where the Keys read is unchanged.
- Like any cubic it can undershoot below zero next to a steep wall. A consumer
  that needs a positive value applies its own floor.
