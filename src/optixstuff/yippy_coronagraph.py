"""Shared-grid adapter over a yippy EqxCoronagraph and its aperture records."""

from typing import final

import jax
import jax.numpy as jnp
import numpy as np
from yippy import EqxCoronagraph
from yippy.performance import performance_curves

from optixstuff.coronagraph import _DEFAULT_RULE, AbstractTableCoronagraph


@final
class YippyCoronagraph(AbstractTableCoronagraph):
    """Adapt yippy Images and eagerly cached PerformanceCurves without unit changes.

    Existing backend records are retained. Additional ``rules`` are prepared
    during construction on the backend's separation axis and stellar map.
    Prepare every rule before JIT; curves() only selects stored records and
    raises for an unprepared rule/averaging combination. The constructor's
    leakage_rule selects the density convention for additional records.
    """

    _backend: EqxCoronagraph
    _records: tuple
    outer_working_angle_lod: float | None

    def __init__(
        self,
        yip_path=None,
        *,
        backend=None,
        rules=(),
        leakage_rule="aperture_mean",
        outer_working_angle_lod=None,
        **kwargs,
    ):
        """Load a YIP or wrap a backend, then prepare any extra aperture rules."""
        if backend is None and yip_path is None:
            raise ValueError("Provide either yip_path or backend")
        self._backend = (
            backend
            if backend is not None
            else EqxCoronagraph(yip_path, leakage_rule=leakage_rule, **kwargs)
        )
        self.outer_working_angle_lod = outer_working_angle_lod
        records = list(self._backend.curves_by_rule.values())
        for rule in rules:
            if any(r.rule == rule and r.leakage_rule == leakage_rule for r in records):
                continue
            record = performance_curves(
                lambda s: self._backend.create_psf(s, 0.0),
                self._backend.stellar_intens(0.0),
                rule,
                separations_lod=self._backend.performance.separation_lod,
                leakage_rule=leakage_rule,
                stellar_diameter=0.0,
            )
            records.append(
                jax.tree.map(
                    lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, record
                )
            )
        self._records = tuple(records)

    @property
    def grid(self):
        """Native off-axis lod grid with its explicit circumscribed Frame."""
        return self._backend.grid

    def curves(self, rule=_DEFAULT_RULE, *, leakage_rule="aperture_mean"):
        """Select a retained yippy record for this rule and density convention."""
        for record in self._records:
            if record.rule == rule and record.leakage_rule == leakage_rule:
                return record
        raise ValueError(
            "No prepared record; supply rules and leakage_rule at construction"
        )

    def occulter_transmission(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Dimensionless sky transmission from the backend's ancillary record."""
        return self._backend.occulter_transmission(separation_lod)

    def noise_floor_ayo(
        self,
        separation_lod,
        ppf=30.0,
        *,
        rule=_DEFAULT_RULE,
        leakage_rule="aperture_mean",
    ):
        """Record leakage_density_lod2 divided by ppf, in (lambda/D)^-2.

        No pixel-area conversion occurs: the record already stores density.
        """
        return (
            self.curves(rule, leakage_rule=leakage_rule).evaluate(
                "leakage_density_lod2", separation_lod
            )
            / ppf
        )

    def stellar_intens(self, stellar_diam_lod):
        """Native per-pixel stellar fraction Image at the requested diameter."""
        return self._backend.stellar_intens(stellar_diam_lod)

    def create_psfs(self, x_lod, y_lod):
        """Native per-pixel source fractions with leading axis 'source'."""
        return self._backend.create_psfs(x_lod, y_lod)

    @property
    def sky_trans(self):
        """Native dimensionless sampled transmission Image."""
        return self._backend.sky_trans

    @property
    def psf_datacube(self):
        """Full or quarter PSF cube; None when not requested at construction."""
        return self._backend.psf_datacube

    def __repr__(self):
        """Summarize native grid and available performance records."""
        record = self._records[0]
        return (
            f"YippyCoronagraph(iwa_lod={float(record.iwa_lod):.3g}, "
            f"data_extent_lod={record.data_extent_lod:.3g}, "
            f"grid={self.grid.shape}, step_lod={self.grid.step:.3g}, "
            f"rules={len(self._records)})"
        )
