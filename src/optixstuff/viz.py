"""Plot functions for optixstuff types, built on eyepiece.

Requires the `viz` extra: `pip install "optixstuff[viz]"`. Import as
`from optixstuff import viz`; `import optixstuff` never imports matplotlib.
"""

_XLABELS = {
    "lod": r"$r$ [$\lambda/D$]",  # codespell:ignore lod
    "arcsec": r"$r$ [arcsec]",
    "au": r"$r$ [AU]",
    "meter": r"$r$ [m]",
    "pixel": r"$r$ [pixel]",
    "pupil_diameter": r"$r$ [D]",
}


def _eyepiece():
    """Import eyepiece, or raise with the install hint."""
    try:
        import eyepiece as ep
    except ImportError:
        raise ImportError(
            "optixstuff.viz requires eyepiece: pip install 'optixstuff[viz]'"
        ) from None
    return ep


def plot_radial_image(
    image,
    *,
    nbins=None,
    ax=None,
    label=None,
    color=None,
    log=False,
    line_kw=None,
    xlabel=None,
    ylabel=None,
):
    """Plot the radial mean of a 2D Image in its grid units.

    Args:
        image: Image with a PixelGrid declaring the sampling and optical axis.
        nbins: Number of radial bins. None uses the `radial_profile` default.
        ax: Axes to draw into. None creates a new figure and axes.
        label: Optional legend label for the curve.
        color: Line color override. None takes the next palette color.
        log: Whether to set a log y-scale.
        line_kw: Extra kwargs passed to ax.plot, applied last.
        xlabel: Optional x-axis label override.
        ylabel: Optional y-axis label.

    Returns:
        An eyepiece PlotResult with artist "line".
    """
    ep = _eyepiece()
    from optixstuff.radial import radial_profile

    separations, profile, _ = radial_profile(image, nbins=nbins)
    return ep.plot.plot_radial(
        separations,
        profile,
        ax=ax,
        label=label,
        color=color,
        log=log,
        line_kw=line_kw,
        xlabel=_XLABELS[image.grid.unit] if xlabel is None else xlabel,
        ylabel=ylabel,
    )
