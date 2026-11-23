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
Pass the table as a model, so its knots become breaks, when such an integral
must be exact. A factor that is evaluated at the nodes and has a feature
narrower than about the band width over the order also needs a declared break.

The node count of a piece follows the response it is read through. A pixel at
separation s, in lambda/D at the band center c, sees a diffraction pattern
slide across a piece from a to b by `s * c * (1 / a - 1 / b)` chromatic
cycles. `field_lod` is the largest separation at which the response is read.
A piece that holds no cycles keeps exactly `order` nodes; otherwise it gets
`max(order, 2 * ceil((order + 3 * cycles) / 2))` nodes. The default of zero
suits a response read at one band-center separation for every node, such as
a performance curve. An image out to the corner of a 128 by 128 detector at
0.25 lambda/D reads separations up to 22.6 lambda/D:

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
