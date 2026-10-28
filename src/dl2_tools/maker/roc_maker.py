"""
ROC curves, Q-factors and AUC for gamma/hadron separation in IACT analyses.

The entry point is :class:`ROCMaker`, a ctapipe ``Component``.  Call it on a
:class:`PointSourceObservationHandler` and it returns one :class:`ROCResult`
per signal handler.  In every reconstructed energy bin the full ROC curve is
computed from the weighted event sample, and the gammaness cut, the surviving
background and the Q-factor are all reported on the signal efficiency grid of
that curve.

The handler layer is a thin shell around :meth:`ROCMaker.from_tables`, which
works on bare event tables (``Table``/``QTable``/dict of arrays) and does not
touch the handler machinery at all.

Conventions
-----------
* signal efficiency     = weighted fraction of theta-cut signal passing the
                          gammaness cut, per energy bin.  With
                          ``apply_theta_cut=False`` it refers to all signal
                          events in the bin instead.
* background efficiency = weighted fraction of ring background passing the same
                          cut.  The alpha factor (on/off solid angle ratio)
                          cancels in this ratio, so it is not needed here.
* background rejection  = 1 - background efficiency
* Q-factor              = eff_signal / sqrt(eff_background)
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import vstack
from ctapipe.core import Component
from ctapipe.core.traits import Bool, Float, Int
from gammapy.maps import MapAxis
from sklearn.metrics import roc_auc_score, roc_curve

from ..handler.binning import IRFBinning
from ..handler.interpolated_cuts import InterpolatedCut
from ..handler.observation_handler import (
    DiffuseObservationHandler,
    PointSourceObservationHandler,
)

__all__ = ["ROCMaker", "ROCResult", "weighted_quantile"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def weighted_quantile(values, quantiles, weights=None):
    """Weighted quantiles of ``values`` (linear interpolation of the CDF)."""
    values = np.asarray(values, dtype=float)
    quantiles = np.atleast_1d(quantiles).astype(float)
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=float)

    order = np.argsort(values)
    values, weights = values[order], weights[order]

    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= np.sum(weights)
    return np.interp(quantiles, cdf, values)


def _values(table, column, unit=None):
    """Column as a plain numpy array, converted to ``unit`` if it has one."""
    col = table[column]
    if unit is None:
        return np.asarray(col)
    if isinstance(col, u.Quantity):
        return col.to_value(unit)
    if getattr(col, "unit", None) is not None:
        return col.quantity.to_value(unit)
    return np.asarray(col, dtype=float)


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class ROCResult:
    """Everything one ROC scan produced, plus plotting helpers.

    ``signal_efficiency`` is the 1D grid shared by every energy bin; the
    quantities defined on it (``background_efficiency``, ``gh_cut``,
    ``q_factor``) have shape ``(n_energy_bins, n_efficiency_points)``.  A NaN
    entry in ``theta_cut`` means no directional cut was applied in that bin.
    """

    energy_edges: u.Quantity
    signal_efficiency: np.ndarray
    background_efficiency: np.ndarray
    gh_cut: np.ndarray
    theta_cut: u.Quantity
    auc: np.ndarray
    n_signal: np.ndarray
    n_background: np.ndarray
    source_offset: u.Quantity = 0.0 * u.deg
    cut_column: str = "gh_score"

    # ---------------------------------------------------------------- derived
    @property
    def energy_center(self):
        e = self.energy_edges.to_value(u.TeV)
        return np.sqrt(e[:-1] * e[1:]) * u.TeV

    @property
    def background_rejection(self):
        return 1 - self.background_efficiency

    @property
    def q_factor(self):
        """Significance improvement factor eff_s / sqrt(eff_b)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            q = self.signal_efficiency / np.sqrt(self.background_efficiency)
        return np.where(self.background_efficiency > 0, q, np.nan)

    def _best_index(self, min_efficiency=0.0):
        """Index of the Q-factor maximum per energy bin, -1 where there is none."""
        q = np.where(self.signal_efficiency >= min_efficiency, self.q_factor, np.nan)
        idx = np.full(len(self.energy_center), -1, dtype=int)
        for i, row in enumerate(q):
            if not np.all(np.isnan(row)):
                idx[i] = np.nanargmax(row)
        return idx

    def best_efficiency(self, min_efficiency=0.0):
        """Signal efficiency maximising the Q-factor, per energy bin.

        The Q-factor keeps growing towards very tight cuts in bins where the
        background runs out of events, so restrict the search with
        ``min_efficiency`` if the maximum lands in that corner.

        Returns
        -------
        best_efficiency : np.ndarray
            Signal efficiency at the maximum, per energy bin.
        best_q_factor : np.ndarray
            Q-factor reached there, per energy bin.
        """
        idx = self._best_index(min_efficiency)
        best = np.full(len(self.energy_center), np.nan)
        best_q = np.full(len(self.energy_center), np.nan)
        for i, j in enumerate(idx):
            if j >= 0:
                best[i] = self.signal_efficiency[j]
                best_q[i] = self.q_factor[i, j]
        return best, best_q

    def best_cut(self, min_efficiency=0.0, fill_value=np.nan):
        """Gammaness cut maximising the Q-factor, per energy bin.

        This is the cut that goes with :meth:`best_efficiency`: it keeps the
        events with ``gh_score >= cut``.

        Parameters
        ----------
        min_efficiency : float
            Only consider cuts keeping at least this signal efficiency.
        fill_value : float
            Cut value for energy bins the scan skipped for lack of events.
            The default NaN makes those bins obvious; 1.0 rejects everything
            there, 0.0 keeps everything.

        Returns
        -------
        np.ndarray
            Gammaness cut per energy bin.
        """
        idx = self._best_index(min_efficiency)
        cuts = np.full(len(self.energy_center), float(fill_value))
        for i, j in enumerate(idx):
            if j >= 0:
                cuts[i] = self.gh_cut[i, j]
        return cuts

    def best_cut_table(self, min_efficiency=0.0, fill_value=np.nan):
        """Per-energy-bin table of the Q-optimal cut and what it achieves."""
        from astropy.table import QTable

        idx = self._best_index(min_efficiency)
        eff, q = self.best_efficiency(min_efficiency)
        eff_b = np.full(len(self.energy_center), np.nan)
        for i, j in enumerate(idx):
            if j >= 0:
                eff_b[i] = self.background_efficiency[i, j]

        return QTable(
            {
                "low": self.energy_edges[:-1],
                "high": self.energy_edges[1:],
                "center": self.energy_center,
                "cut": self.best_cut(min_efficiency, fill_value),
                "signal_efficiency": eff,
                "background_efficiency": eff_b,
                "q_factor": q,
                "theta_cut": self.theta_cut,
            },
            meta={
                "source_offset": self.source_offset,
                "cut_column": self.cut_column,
                "min_efficiency": min_efficiency,
            },
        )

    def to_interpolated_cut(self, min_efficiency=0.0, fill_value=1.0, **kwargs):
        """Q-optimal gammaness cut as an :class:`InterpolatedCut`.

        The cut is taken from :meth:`best_cut`, so in every reconstructed
        energy bin it is the gammaness threshold maximising the Q-factor among
        those keeping at least ``min_efficiency`` of the signal.

        The resulting cut has a single node on the offset axis, at this
        result's ``source_offset``, so it returns the same values at every
        offset.  Use :meth:`ROCMaker.to_interpolated_cut` to build one that
        interpolates between several source offsets.

        Parameters
        ----------
        min_efficiency : float
            Lower bound on the signal efficiency of the chosen cut.
        fill_value : float
            Cut used in energy bins the scan skipped for lack of events.
            Defaults to 1.0, which rejects essentially everything there.
        **kwargs
            Passed on to ``InterpolatedCut``: ``op`` (default
            ``operator.ge``), ``method`` (default ``"linear"``, which with a
            single offset node governs the energy axis), ``bounds_error``,
            and ``interp_fill_value`` for the interpolator's own fill value,
            renamed here to avoid the clash with ``fill_value`` above.
        """
        return _build_interpolated_cut(
            [self], min_efficiency=min_efficiency, fill_value=fill_value, **kwargs
        )

    def at(self, signal_efficiency):
        """Interpolate the curves at one signal efficiency, for all energy bins.

        Returns a table with the gammaness cut, the surviving background and
        the Q-factor per energy bin.
        """
        from astropy.table import QTable

        eff = float(signal_efficiency)
        grid = self.signal_efficiency

        def _interp(values):
            return np.array([np.interp(eff, grid, row) for row in values])

        return QTable(
            {
                "energy": self.energy_center,
                "signal_efficiency": np.full(len(self.energy_center), eff),
                "gh_cut": _interp(self.gh_cut),
                "background_efficiency": _interp(self.background_efficiency),
                "q_factor": _interp(self.q_factor),
                "theta_cut": self.theta_cut,
            },
            meta={"source_offset": self.source_offset},
        )

    def to_table(self):
        """Long-format table of the scan, one row per (energy bin, efficiency)."""
        from astropy.table import QTable

        n_e, n_eff = self.background_efficiency.shape
        return QTable(
            {
                "energy": np.repeat(self.energy_center, n_eff),
                "signal_efficiency": np.tile(self.signal_efficiency, n_e),
                "background_efficiency": self.background_efficiency.ravel(),
                "background_rejection": self.background_rejection.ravel(),
                "q_factor": self.q_factor.ravel(),
                "gh_cut": self.gh_cut.ravel(),
                "theta_cut": np.repeat(self.theta_cut, n_eff),
            },
            meta={"source_offset": self.source_offset},
        )

    # --------------------------------------------------------------- plotting
    def _energy_labels(self):
        edges = self.energy_edges.to_value(u.TeV)
        return [f"{lo:.2g} - {hi:.2g} TeV" for lo, hi in zip(edges[:-1], edges[1:])]

    def plot_roc(self, ax=None, **kwargs):
        """Signal efficiency vs. background efficiency, one curve per energy bin."""
        ax = ax or plt.gca()
        for i, label in enumerate(self._energy_labels()):
            ax.plot(
                self.background_efficiency[i],
                self.signal_efficiency,
                label=label,
                **kwargs,
            )

        ax.set_xlabel("Background efficiency")
        ax.set_ylabel("Signal efficiency")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
        return ax

    def plot_q_factor(self, ax=None, **kwargs):
        """Q-factor as a function of signal efficiency, one curve per energy bin."""
        ax = ax or plt.gca()
        q = self.q_factor
        for i, label in enumerate(self._energy_labels()):
            ax.plot(self.signal_efficiency, q[i], label=label, **kwargs)

        ax.set_xlabel("Signal efficiency")
        ax.set_ylabel(r"Q-factor  $\epsilon_s / \sqrt{\epsilon_b}$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
        return ax

    def plot_background_efficiency_vs_energy(
        self, ax=None, efficiencies=None, **kwargs
    ):
        """Background efficiency vs. energy for selected signal efficiencies."""
        ax = ax or plt.gca()
        if efficiencies is None:
            efficiencies = [0.5, 0.7, 0.9]
        for eff in np.atleast_1d(efficiencies):
            j = int(np.argmin(np.abs(self.signal_efficiency - eff)))
            ax.plot(
                self.energy_center,
                self.background_efficiency[:, j],
                marker="o",
                label=rf"$\epsilon_s$ = {self.signal_efficiency[j]:.2f}",
                **kwargs,
            )

        ax.set_xlabel("Reconstructed energy / TeV")
        ax.set_ylabel("Background efficiency")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
        return ax

    def plot_auc_vs_energy(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        kwargs.setdefault("marker", "o")
        ax.plot(self.energy_center, self.auc, **kwargs)
        ax.set_xlabel("Reconstructed energy / TeV")
        ax.set_ylabel("ROC AUC")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        return ax


# --------------------------------------------------------------------------- #
# cut extraction
# --------------------------------------------------------------------------- #
def _build_interpolated_cut(
    results,
    min_efficiency=0.0,
    fill_value=1.0,
    op=operator.ge,
    method="linear",
    bounds_error=False,
    interp_fill_value=None,
    offset_axis_name="reco_source_fov_offset",
    energy_axis_name="reco_energy",
):
    """Build an ``InterpolatedCut`` on the gammaness from one or more results.

    Each result contributes one node on the offset axis, at its own
    ``source_offset``; the cut values are the Q-optimal gammaness cuts per
    reconstructed energy bin.

    ``method`` applies to both axes of the interpolator, so with a single
    offset it really governs the energy axis: the default ``"linear"``
    interpolates between bin centres and extrapolates in log-energy beyond
    them, while ``"nearest"`` gives a step function in energy, clamped flat
    outside the node range.  The nodes sit at the bin *centres*, so the first
    and last half-bin are already extrapolated; pass
    ``interp_fill_value=np.nan`` or ``bounds_error=True`` to make
    out-of-range energies explicit instead.
    """
    if len(results) == 0:
        raise ValueError("need at least one ROCResult to build a cut")

    edges = u.Quantity(results[0].energy_edges)
    columns = {res.cut_column for res in results}
    if len(columns) > 1:
        raise ValueError(f"results use different cut columns: {sorted(columns)}")
    for res in results[1:]:
        if not u.allclose(u.Quantity(res.energy_edges), edges):
            raise ValueError("all results must share the same energy binning")

    offsets = u.Quantity([u.Quantity(res.source_offset) for res in results]).to(u.deg)
    order = np.argsort(offsets.to_value(u.deg))
    offsets = offsets[order]
    if len(offsets) > 1 and np.any(np.diff(offsets.to_value(u.deg)) <= 0):
        raise ValueError("results must have distinct source offsets")

    cut_values = np.array(
        [results[i].best_cut(min_efficiency, fill_value) for i in order]
    )
    if np.any(np.isnan(cut_values)):
        logger.warning(
            "%d of %d (offset, energy) cut values are NaN; pass a finite "
            "fill_value if the cut is going to be applied",
            int(np.sum(np.isnan(cut_values))),
            cut_values.size,
        )

    offset_axis = MapAxis(
        nodes=offsets, node_type="center", name=offset_axis_name
    )
    energy_axis = MapAxis(
        nodes=edges, node_type="edges", interp="log", name=energy_axis_name
    )

    return InterpolatedCut(
        offset_axis=offset_axis,
        cut_column=results[0].cut_column,
        cut_values=cut_values,
        bin_axes=[energy_axis],
        op=op,
        method=method,
        bounds_error=bounds_error,
        fill_value=interp_fill_value,
    )


# --------------------------------------------------------------------------- #
# the maker
# --------------------------------------------------------------------------- #
class ROCMaker(Component):
    """Background rejection, Q-factors and AUC for a point-source observation.

    Call the component on a ``PointSourceObservationHandler``.  One
    :class:`ROCResult` is produced per signal handler, i.e. per source offset;
    the events of all background handlers are stacked and restricted to a
    FoV-offset ring around that signal's offset.
    """

    n_efficiency_points = Int(
        help="Number of points of the signal efficiency grid the ROC curve and "
        "all quantities derived from it are sampled on",
        default_value=200,
    ).tag(config=True)

    apply_theta_cut = Bool(
        help="Apply a directional (theta) cut to the signal before scanning the "
        "gammaness cut. If False, signal efficiencies refer to all signal events "
        "in the energy bin",
        default_value=True,
    ).tag(config=True)

    radius_percentile = Float(
        help="Containment percentile of the per-energy-bin theta cut",
        default_value=68.0,
    ).tag(config=True)

    fixed_radius_cut = Float(
        help="Fixed theta cut in deg, overriding radius_percentile. "
        "If None, the per-energy-bin containment cut is used",
        default_value=None,
        allow_none=True,
    ).tag(config=True)

    bkg_ring_size = Float(
        help="Half-width of the FoV offset ring used for background estimation in deg",
        default_value=1.0,
    ).tag(config=True)

    n_reco_energy_bins = Int(
        help="Number of reconstructed energy bins to evaluate the cuts in",
        default_value=10,
    ).tag(config=True)

    min_events = Int(
        help="Energy bins with fewer signal or background events are filled with NaN",
        default_value=10,
    ).tag(config=True)

    @property
    def efficiency_grid(self):
        """Signal efficiencies every quantity is reported on, shared by all bins."""
        return np.linspace(0.0, 1.0, self.n_efficiency_points + 1)[1:]

    energy_column = "reco_energy"
    score_column = "gh_score"
    theta_column = "theta"
    offset_column = "reco_source_fov_offset"
    weight_column = "weight"

    # ------------------------------------------------------------ handler API
    def __call__(self, observation, energy_bins=None, apply_theta_cut=None):
        """Run the scan on an observation handler.

        Parameters
        ----------
        observation : PointSourceObservationHandler
            Signal and background handlers. Cuts already attached to the
            handlers are respected, since the events are read through
            ``get_masked_events()``.
        energy_bins : u.Quantity or None
            Reconstructed energy bin edges. Defaults to ``n_reco_energy_bins``
            log-spaced bins over the true energy range of the observation.
        apply_theta_cut : bool or None
            Overrides the ``apply_theta_cut`` trait for this call only.

        Returns
        -------
        list of ROCResult
            One entry per signal handler, in the same order.
        """
        if isinstance(observation, DiffuseObservationHandler):
            raise NotImplementedError(
                "Only point-source observations are supported for now"
            )
        if not isinstance(observation, PointSourceObservationHandler):
            raise TypeError("observation must be a PointSourceObservationHandler")
        if len(observation.background) < 1:
            raise ValueError("observation needs at least one background handler")

        self.binning = IRFBinning(observation, parent=self)
        self.energy_bins = (
            self._default_energy_bins() if energy_bins is None else energy_bins
        )

        background = vstack(
            [bkg.get_masked_events() for bkg in observation.background],
            metadata_conflicts="silent",
        )

        self.results = [
            self.from_tables(
                signal.get_masked_events(),
                background,
                source_offset=signal.offset,
                energy_bins=self.energy_bins,
                apply_theta_cut=apply_theta_cut,
            )
            for signal in observation.signal
        ]
        return self.results

    def to_interpolated_cut(
        self, results=None, min_efficiency=0.0, fill_value=1.0, **kwargs
    ):
        """Q-optimal gammaness cut as an :class:`InterpolatedCut`.

        In every reconstructed energy bin the cut is the gammaness threshold
        maximising the Q-factor among those keeping at least
        ``min_efficiency`` of the signal.  Every result contributes one node on
        the offset axis, at its own source offset, so a run over several signal
        handlers gives a cut that interpolates in offset as well as in energy.

        Parameters
        ----------
        results : list of ROCResult or None
            Defaults to the results of the last call.
        min_efficiency : float
            Lower bound on the signal efficiency of the chosen cut.
        fill_value : float
            Cut used in energy bins the scan skipped for lack of events.
            Defaults to 1.0, which rejects essentially everything there.
        **kwargs
            Passed on to ``InterpolatedCut``: ``op`` (default
            ``operator.ge``), ``method`` (default ``"linear"``, applied to
            both the energy and the offset axis), ``bounds_error`` and
            ``interp_fill_value``.
        """
        if results is None:
            results = getattr(self, "results", None)
        if not results:
            raise ValueError("no results to build a cut from; call the maker first")

        return _build_interpolated_cut(
            list(results),
            min_efficiency=min_efficiency,
            fill_value=fill_value,
            **kwargs,
        )

    def _default_energy_bins(self):
        edges = u.Quantity(self.binning.energy_true.edges)
        return (
            np.geomspace(
                edges[0].to_value(u.TeV),
                edges[-1].to_value(u.TeV),
                self.n_reco_energy_bins + 1,
            )
            * u.TeV
        )

    # -------------------------------------------------------------- table API
    def from_tables(
        self,
        signal,
        background,
        source_offset=0.0 * u.deg,
        energy_bins=None,
        apply_theta_cut=None,
    ):
        """Run the scan on bare event tables, with no handler involved.

        ``apply_theta_cut`` overrides the trait of the same name for this call,
        which is handy for comparing the two side by side.
        """
        if apply_theta_cut is None:
            apply_theta_cut = self.apply_theta_cut
        if energy_bins is None:
            energy_bins = getattr(self, "energy_bins", None)
        if energy_bins is None:
            raise ValueError("energy_bins must be given when running on tables")

        edges = u.Quantity(energy_bins).to_value(u.TeV)
        efficiencies = self.efficiency_grid
        n_e, n_eff = len(edges) - 1, len(efficiencies)

        sig = self._prepare(signal, is_signal=True)
        bkg = self._select_ring(self._prepare(background), source_offset)

        if apply_theta_cut and "theta" not in sig:
            self.log.warning(
                "apply_theta_cut is True but the signal table has no '%s' column, "
                "running without a directional cut",
                self.theta_column,
            )
            apply_theta_cut = False

        sig_bin = np.digitize(sig["energy"], edges) - 1
        bkg_bin = np.digitize(bkg["energy"], edges) - 1

        eff_b = np.full((n_e, n_eff), np.nan)
        gh_cuts = np.full((n_e, n_eff), np.nan)
        theta_cuts = np.full(n_e, np.nan)
        auc = np.full(n_e, np.nan)
        n_sig = np.zeros(n_e, dtype=int)
        n_bkg = np.zeros(n_e, dtype=int)

        for i in range(n_e):
            s = {k: v[sig_bin == i] for k, v in sig.items()}
            b = {k: v[bkg_bin == i] for k, v in bkg.items()}

            # directional cut on the signal, per energy bin
            theta_cut = self._get_theta_cut(s, apply_theta_cut)
            theta_cuts[i] = theta_cut
            if np.isfinite(theta_cut):
                keep = s["theta"] <= theta_cut
                s = {k: v[keep] for k, v in s.items()}

            n_sig[i], n_bkg[i] = len(s["score"]), len(b["score"])
            if n_sig[i] < self.min_events or n_bkg[i] < self.min_events:
                self.log.debug("Energy bin %d has too few events, skipping", i)
                continue

            auc[i], eff_b[i], gh_cuts[i] = self._roc(s, b, efficiencies)

        return ROCResult(
            energy_edges=u.Quantity(energy_bins),
            signal_efficiency=efficiencies,
            background_efficiency=eff_b,
            gh_cut=gh_cuts,
            theta_cut=theta_cuts * u.deg,
            auc=auc,
            n_signal=n_sig,
            n_background=n_bkg,
            source_offset=u.Quantity(source_offset),
            cut_column=self.score_column,
        )

    # -------------------------------------------------------------- internals
    def _prepare(self, table, is_signal=False):
        """Pull the columns we need out of an event table, as plain arrays."""
        energy = _values(table, self.energy_column, u.TeV)
        out = {
            "energy": energy,
            "score": _values(table, self.score_column).astype(float),
        }
        if self.weight_column in table.keys():
            out["weight"] = np.asarray(table[self.weight_column], dtype=float)
        else:
            out["weight"] = np.ones(len(energy))

        if is_signal and self.theta_column in table.keys():
            out["theta"] = _values(table, self.theta_column, u.deg)
        if not is_signal and self.offset_column in table.keys():
            out["offset"] = _values(table, self.offset_column, u.deg)
        return out

    def _select_ring(self, background, source_offset):
        """Restrict the background to a FoV offset ring around the source."""
        if "offset" not in background or self.bkg_ring_size <= 0:
            return background
        centre = u.Quantity(source_offset).to_value(u.deg)
        low = max(0.0, centre - self.bkg_ring_size)
        high = centre + self.bkg_ring_size
        keep = (background["offset"] >= low) & (background["offset"] <= high)
        return {k: v[keep] for k, v in background.items()}

    def _get_theta_cut(self, signal, apply_theta_cut):
        """Theta cut in deg for one energy bin, or NaN if none is to be applied."""
        if not apply_theta_cut:
            return np.nan
        if "theta" not in signal:
            return np.nan
        if self.fixed_radius_cut is not None:
            return float(self.fixed_radius_cut)
        if len(signal["theta"]) < self.min_events:
            return np.nan
        return weighted_quantile(
            signal["theta"], self.radius_percentile / 100, signal["weight"]
        )[0]

    @staticmethod
    def _roc(signal, background, efficiencies):
        """AUC, background efficiency and gammaness cut on the efficiency grid.

        The exact weighted ROC curve is computed once per energy bin, then the
        background efficiency (false positive rate) and the threshold are
        interpolated onto the common signal efficiency (true positive rate)
        grid, so every reported quantity lives on the same curve.
        """
        y_true = np.concatenate(
            [np.ones(len(signal["score"])), np.zeros(len(background["score"]))]
        )
        scores = np.concatenate([signal["score"], background["score"]])
        weights = np.concatenate([signal["weight"], background["weight"]])

        auc = roc_auc_score(y_true, scores, sample_weight=weights)
        fpr, tpr, thresholds = roc_curve(
            y_true, scores, sample_weight=weights, drop_intermediate=False
        )
        # the first entry carries threshold = inf, which interpolates badly
        fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]

        eff_b = np.interp(efficiencies, tpr, fpr)
        gh_cut = np.interp(efficiencies, tpr, thresholds)
        return auc, eff_b, gh_cut
"""

from ..handler.observation_handler import (
    PointSourceObservationHandler,
    DiffuseObservationHandler,
)
from ctapipe.core import Component
from ctapipe.core.traits import List, Int
from traitlets import Float
from ..handler.binning import IRFBinning
from .cut_optimizer import PercentileCutCalculator
import numpy as np
import astropy.units as u
from copy import deepcopy
from astropy.coordinates.angle_utilities import angular_separation
from gammapy.maps import MapAxis
from ..handler.interpolated_cuts import InterpolatedCut
import operator
from pyirf.sensitivity import estimate_background
from pyirf.binning import calculate_bin_indices

from scipy.integrate import quad
from scipy.interpolate import interp1d


def table_group_by(data, value_key, bins):
    bin_index, valid = calculate_bin_indices(data[value_key], bins)
    by_bin = data[valid].group_by(bin_index[valid])
    return by_bin


class ROCMaker(Component):

    loss_params = List(
        default_value=[0.01, 0.99, 99],
        minlen=3,
        maxlen=3,
        help="min, max and number of bins",
    ).tag(config=True)

    radius_percentile = Float(
        help="Percentile of events to keep after radial cut for pointsource observation",
        default_value=68,
    ).tag(config=True)

    bkg_ring_size = Float(
        help="Radius of FoV offset ring to consider for background estimation in deg",
        default_value=1,
    ).tag(config=True)

    n_reco_energy_bins = Int(
        help="Number of energy bins to evaluate cut in ",
        default_value=10,
    ).tag(config=True)

    def __init__(self, config=None, parent=None):
        super().__init__(config=config, parent=parent)
        self.loss_vector = np.linspace(
            self.loss_params[0],
            self.loss_params[1],
            self.loss_params[2],
        )

    def __call__(self, observation):

        assert len(observation.background) >= 1

        self.binning = IRFBinning(observation, parent=self)

        if self.n_reco_energy_bins >= 1:
            self.reco_energy_axis = MapAxis(
                nodes=np.geomspace(
                    self.binning.energy_true.edges[0],
                    self.binning.energy_true.edges[-1],
                    self.n_reco_energy_bins + 1,
                ),
                node_type="edges",
                name="reco_energy",
            )

        else:
            raise ValueError("You need to have at least one energy bin")

        self.gh_cut_calculator = PercentileCutCalculator()
        self.gh_cut_calculator.cut_variable = "gh_score"
        self.gh_cut_calculator.op = operator.ge
        self.gh_cut_calculator.fill_value = 1
        self.gh_cut_calculator.min_events = 1
        self.gh_cut_calculator.bin_axes = [self.reco_energy_axis]

        if isinstance(observation, PointSourceObservationHandler):
            self.radius_cut_calculator = PercentileCutCalculator()
            self.radius_cut_calculator.cut_variable = "theta"
            self.radius_cut_calculator.op = operator.le
            self.radius_cut_calculator.cut_op = lambda x: x.to_value(u.deg)
            self.radius_cut_calculator.fill_value = 0.1
            self.radius_cut_calculator.percentile = self.radius_percentile
            self.radius_cut_calculator.bin_axes = [self.reco_energy_axis]

            self.signal_loss, self.bkg_efficiencies = self._calc_roc_ps(
                observation,
            )

        elif isinstance(observation, DiffuseObservationHandler):

            self.signal_loss, self.bkg_efficiencies = self._calc_roc_diff(
                observation,
            )

        else:
            raise TypeError(
                "observation must be either a PointSourceObservationhandler or a DiffuseObservationHandler"
            )

    def _calc_roc_ps(self, observation):

        roc_sig_loss = np.empty(
            (
                len(self.loss_vector),
                len(observation.signal),
                len(self.reco_energy_axis.center),
            )
        )
        roc_bkg_efficiencies = np.empty(
            (len(self.loss_vector), len(self.reco_energy_axis.center))
        )

        for i, loss in enumerate(self.loss_vector):
            self.gh_cut_calculator.fill_value = loss
            self.gh_cut_calculator.percentile = 100 * (1 - loss)

            gh_cut = self.gh_cut_calculator(
                observation.signal, "reco_source_fov_offset"
            )

            (
                roc_sig_loss[i],
                roc_bkg_efficiencies[i],
            ) = self.calc_efficiencies_ps(observation, gh_cut)
        if len(observation.signal) > 1:
            return roc_sig_loss, roc_bkg_efficiencies
        else:
            return (
                roc_sig_loss.reshape(
                    len(self.loss_vector),
                    len(self.reco_energy_axis.center),
                ),
                roc_bkg_efficiencies,
            )

    def _calc_roc_diff(self, observation):

        roc_sig_loss = np.empty(
            (
                len(self.loss_vector),
                len(self.reco_energy_axis.center),
            )
        )
        roc_bkg_efficiencies = np.empty(
            (len(self.loss_vector), len(self.reco_energy_axis.center))
        )

        diff_offset_axis = self.binning.signal_offset
        for i, loss in enumerate(self.loss_vector):
            self.gh_cut_calculator.fill_value = loss
            self.gh_cut_calculator.percentile = 100 * (1 - loss)

            gh_cut = self.gh_cut_calculator(
                observation.signal, "reco_source_fov_offset"
            )

            (
                roc_sig_loss[i],
                roc_bkg_efficiencies[i],
            ) = self.calc_efficiencies_diff(observation, gh_cut, diff_offset_axis)

        return roc_sig_loss, roc_bkg_efficiencies

    def calc_efficiencies_ps(self, observation, gh_cut):

        sig_efficiencies = np.empty(
            (len(observation.signal), len(self.reco_energy_axis.center))
        )
        bkg_evt_rate = np.zeros(len(self.reco_energy_axis.center))
        uncut_bkg_evt_rate = np.zeros(len(self.reco_energy_axis.center))

        gh_cut_observation = deepcopy(observation)
        gh_cut_observation.add_cuts([gh_cut], "reco_source_fov_offset")
        radius_cut = self.radius_cut_calculator(
            gh_cut_observation.signal, "reco_source_fov_offset"
        )

        rate_observation = deepcopy(observation)
        rate_observation.add_signal_cuts([gh_cut, radius_cut], "reco_source_fov_offset")
        rate_observation.add_background_cuts([gh_cut], "reco_source_fov_offset")

        rad_cut_observation = deepcopy(observation)
        rad_cut_observation.add_signal_cuts([radius_cut], "reco_source_fov_offset")

        for j, (signal, uncut_signal) in enumerate(
            zip(rate_observation.signal, rad_cut_observation.signal)
        ):

            signal_by_bin = table_group_by(
                signal.get_masked_events(),
                "reco_energy",
                self.reco_energy_axis.edges,
            )
            uncut_signal_by_bin = table_group_by(
                uncut_signal.get_masked_events(),
                "reco_energy",
                self.reco_energy_axis.edges,
            )
            assert len(signal_by_bin.groups) == len(uncut_signal_by_bin.groups)
            for i, signal_group, uncut_signal_group in zip(
                signal_by_bin.groups.keys,
                signal_by_bin.groups,
                uncut_signal_by_bin.groups,
            ):
                uncut_sig_rate = np.sum(uncut_signal_group["weight"])
                sig_rate = np.sum(signal_group["weight"])
                sig_efficiencies[j, i] = sig_rate / uncut_sig_rate

        for j, (background, uncut_background) in enumerate(
            zip(rate_observation.background, rad_cut_observation.background)
        ):

            bg_hist_table = estimate_background(
                events=background.get_masked_events(),
                reco_energy_bins=self.reco_energy_axis.edges,
                theta_cuts=radius_cut.to_cut_table(
                    offset=0.0,
                    bin_axis=self.reco_energy_axis,
                    cut_column_unit=u.deg,
                ),
                alpha=1,
                fov_offset_min=max(
                    0.0 * u.deg, signal.offset - self.bkg_ring_size * u.deg
                ),
                fov_offset_max=signal.offset + self.bkg_ring_size * u.deg,
            )
            uncut_bg_hist_table = estimate_background(
                events=uncut_background.get_masked_events(),
                reco_energy_bins=self.reco_energy_axis.edges,
                theta_cuts=radius_cut.to_cut_table(
                    offset=0.0,
                    bin_axis=self.reco_energy_axis,
                    cut_column_unit=u.deg,
                ),
                alpha=1,
                fov_offset_min=max(
                    0.0 * u.deg, signal.offset - self.bkg_ring_size * u.deg
                ),
                fov_offset_max=signal.offset + self.bkg_ring_size * u.deg,
            )
            bkg_evt_rate += bg_hist_table["n_weighted"]
            uncut_bkg_evt_rate += uncut_bg_hist_table["n_weighted"]

        bkg_efficiencies = bkg_evt_rate / uncut_bkg_evt_rate

        return 1 - sig_efficiencies, bkg_efficiencies

    def calc_efficiencies_diff(self, observation, gh_cut, diff_offset_axis):

        sig_efficiencies = np.empty(
            (len(diff_offset_axis.center), len(self.reco_energy_axis.center))
        )
        bkg_efficiencies = np.empty(
            (len(diff_offset_axis.center), len(self.reco_energy_axis.center))
        )
        bkg_evt_rate = np.zeros(
            (len(diff_offset_axis.center), len(self.reco_energy_axis.center))
        )
        uncut_bkg_evt_rate = np.zeros(
            (len(diff_offset_axis.center), len(self.reco_energy_axis.center))
        )

        assert isinstance(observation, DiffuseObservationHandler)

        gh_cut_observation = deepcopy(observation)

        gh_cut_observation.add_cuts([gh_cut], "reco_source_fov_offset")

        for j, (offset_edge_low, offset_edge_high) in enumerate(
            zip(diff_offset_axis.edges[:-1], diff_offset_axis.edges[1:])
        ):
            low_radius_cut = InterpolatedCut(
                offset_axis=MapAxis(
                    nodes=[0.0 * u.deg],
                    node_type="center",
                    name="reco_source_fov_offset",
                ),
                cut_column="reco_source_fov_offset",
                cut_values=[offset_edge_low],
                method="nearest",
                bounds_error=False,
                fill_value=None,
            )
            high_radius_cut = InterpolatedCut(
                offset_axis=MapAxis(
                    nodes=[-0.01 * u.deg, 0.01 * u.deg],
                    node_type="edges",
                    name="reco_source_fov_offset",
                ),
                cut_column="reco_source_fov_offset",
                cut_values=[offset_edge_high],
                op=operator.le,
                method="nearest",
                bounds_error=False,
                fill_value=None,
            )
            rad_gh_observation = deepcopy(gh_cut_observation)
            rad_gh_observation.add_cuts(
                [low_radius_cut, high_radius_cut], "reco_source_fov_offset"
            )

            rad_observation = deepcopy(observation)
            rad_observation.add_cuts(
                [low_radius_cut, high_radius_cut], "reco_source_fov_offset"
            )

            signal_by_bin = table_group_by(
                rad_gh_observation.signal[0].get_masked_events(),
                "reco_energy",
                self.reco_energy_axis.edges,
            )
            uncut_signal_by_bin = table_group_by(
                rad_observation.signal[0].get_masked_events(),
                "reco_energy",
                self.reco_energy_axis.edges,
            )
            assert len(signal_by_bin.groups) == len(uncut_signal_by_bin.groups)
            for i, signal_group, uncut_signal_group in zip(
                signal_by_bin.groups.keys,
                signal_by_bin.groups,
                uncut_signal_by_bin.groups,
            ):
                uncut_sig_rate = np.sum(uncut_signal_group["weight"])
                sig_rate = np.sum(signal_group["weight"])
                sig_efficiencies[j, i] = sig_rate / uncut_sig_rate

            for bkg, uncut_bkg in zip(
                rad_gh_observation.background, rad_observation.background
            ):
                background_by_bin = table_group_by(
                    bkg.get_masked_events(),
                    "reco_energy",
                    self.reco_energy_axis.edges,
                )
                uncut_background_by_bin = table_group_by(
                    uncut_bkg.get_masked_events(),
                    "reco_energy",
                    self.reco_energy_axis.edges,
                )
                for i, bkg_group, uncut_bkg_group in zip(
                    background_by_bin.groups.keys,
                    background_by_bin.groups,
                    uncut_background_by_bin.groups,
                ):
                    uncut_bkg_evt_rate[j, i] += np.sum(uncut_bkg_group["weight"])
                    bkg_evt_rate[j, i] += np.sum(bkg_group["weight"])

        bkg_efficiencies = bkg_evt_rate / uncut_bkg_evt_rate

        return 1 - sig_efficiencies, bkg_efficiencies

    @staticmethod
    def make_plot(ax, signal_loss, bkg_efficiencies, **kwargs):

        sort_args = np.argsort(signal_loss)
        sorted_sig_effs = np.sort(signal_loss)
        sorted_bkg_effs = bkg_efficiencies[sort_args]

        ax.plot(sorted_sig_effs, sorted_bkg_effs, **kwargs)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Signal loss")
        ax.set_ylabel("Background efficiency")

    @staticmethod
    def make_energy_plot(ax, energies, bkg_efficiencies, **kwargs):

        ax.plot(energies, bkg_efficiencies, **kwargs)

        ax.set_ylabel("Background efficiency")
        ax.set_xlabel("Energy in TeV")
        ax.set_xscale("log")
        ax.set_yscale("log")

    @staticmethod
    def integrate_auc(signal_loss, bkg_efficiencies, **kwargs):

        sort_args = np.argsort(signal_loss)
        sorted_sig_effs = np.sort(signal_loss)
        sorted_bkg_effs = bkg_efficiencies[sort_args]

        f = lambda x: np.interp(
            x,
            sorted_sig_effs,
            sorted_bkg_effs,
        )
        auc = quad(f, 0, 1)

        return auc[0]

    def integrate_auc_all_energy_bins(self, offset):

        aucs = np.empty(len(self.reco_energy_axis.center))

        if len(np.shape(self.signal_loss)) > 2:
            offset_bin_id = self.binning.signal_offset.coord_to_idx(offset, clip=False)
            for i, (sig_loss, bkg_eff) in enumerate(
                zip(
                    self.signal_loss[:, offset_bin_id, :].T,
                    self.bkg_efficiencies[:, offset_bin_id, :].T,
                )
            ):
                aucs[i] = ROCMaker.integrate_auc(sig_loss, bkg_eff)

        else:
            assert len(self.signal_loss.T[0]) == len(self.loss_vector)
            for i, (sig_loss, bkg_eff) in enumerate(
                zip(
                    self.signal_loss.T,
                    self.bkg_efficiencies.T,
                )
            ):
                aucs[i] = ROCMaker.integrate_auc(sig_loss, bkg_eff)

        return aucs

    def integrate_auc_single_energy(self, energy, offset):
        bin_id = self.reco_energy_axis.coord_to_idx(energy, clip=True)

        if len(np.shape(self.signal_loss)) == 2:
            auc = ROCMaker.integrate_auc(
                self.signal_loss.T[bin_id].reshape(
                    len(self.loss_vector),
                ),
                self.bkg_efficiencies.T[bin_id].reshape(
                    len(self.loss_vector),
                ),
            )
        else:
            offset_bin_id = self.binning.signal_offset.coord_to_idx(offset, clip=False)
            auc = ROCMaker.integrate_auc(
                self.signal_loss[:, offset_bin_id, :]
                .T[bin_id]
                .reshape(
                    len(self.loss_vector),
                ),
                self.bkg_efficiencies[:, offset_bin_id, :]
                .T[bin_id]
                .reshape(
                    len(self.loss_vector),
                ),
            )

        return auc

    def roc_plot_all_energy_bins(self, ax, offset, **kwargs):

        if len(np.shape(self.signal_loss)) > 2:
            offset_bin_id = self.binning.signal_offset.coord_to_idx(offset, clip=False)
            for E, sig_loss, bkg_eff in zip(
                self.reco_energy_axis.center,
                self.signal_loss[:, offset_bin_id, :].T,
                self.bkg_efficiencies[:, offset_bin_id, :].T,
            ):
                ROCMaker.make_plot(ax, sig_loss, bkg_eff, label=E, **kwargs)

        else:
            assert len(self.signal_loss.T[0]) == len(self.loss_vector)
            for E, sig_loss, bkg_eff in zip(
                self.reco_energy_axis.center,
                self.signal_loss.T,
                self.bkg_efficiencies.T,
            ):
                ROCMaker.make_plot(ax, sig_loss, bkg_eff, label=E, **kwargs)

        ax.legend()

    def roc_plot_single_energy(self, ax, energy, offset, **kwargs):
        bin_id = self.reco_energy_axis.coord_to_idx(energy, clip=True)

        if len(np.shape(self.signal_loss)) == 2:
            ROCMaker.make_plot(
                ax,
                self.signal_loss.T[bin_id].reshape(
                    len(self.loss_vector),
                ),
                self.bkg_efficiencies.T[bin_id].reshape(
                    len(self.loss_vector),
                ),
                label=energy,
                **kwargs
            )
        else:
            offset_bin_id = self.binning.signal_offset.coord_to_idx(offset, clip=False)
            ROCMaker.make_plot(
                ax,
                self.signal_loss[:, offset_bin_id, :].T[bin_id],
                self.bkg_efficiencies[:, offset_bin_id, :].T[bin_id],
                label=energy,
                **kwargs
            )

    def energy_plot_efficiencies(self, ax, efficiency, offset, **kwargs):

        loss = 1 - efficiency

        assert loss > np.min(self.signal_loss)
        assert loss < np.max(self.signal_loss)

        if len(np.shape(self.signal_loss)) == 2:

            smaller_loss = np.max(
                np.where(self.signal_loss[:, 0] <= loss, self.signal_loss[:, 0], 0)
            )
            larger_loss = np.min(
                np.where(self.signal_loss[:, 0] > loss, self.signal_loss[:, 0], 1)
            )

            smaller_ind = np.argwhere(self.signal_loss[:, 0] == smaller_loss)
            larger_ind = np.argwhere(self.signal_loss[:, 0] == larger_loss)

            bkg_effs = (larger_loss - loss) / (
                larger_loss - smaller_loss
            ) * self.bkg_efficiencies[smaller_ind] + (loss - smaller_loss) / (
                larger_loss - smaller_loss
            ) * self.bkg_efficiencies[
                larger_ind
            ]

            ROCMaker.make_energy_plot(
                ax,
                self.reco_energy_axis.center,
                bkg_effs.reshape(
                    len(self.reco_energy_axis.center),
                ),
                label="Signal efficiency={}".format(efficiency),
                **kwargs
            )
        else:
            offset_bin_id = self.binning.signal_offset.coord_to_idx(offset, clip=False)
            smaller_loss = np.max(
                np.where(
                    self.signal_loss[:, offset_bin_id, 0] <= loss,
                    self.signal_loss[:, offset_bin_id, 0],
                    0,
                )
            )
            larger_loss = np.min(
                np.where(
                    self.signal_loss[:, offset_bin_id, 0] > loss,
                    self.signal_loss[:, offset_bin_id, 0],
                    1,
                )
            )

            smaller_ind = np.argwhere(
                self.signal_loss[:, offset_bin_id, 0] == smaller_loss
            )
            larger_ind = np.argwhere(
                self.signal_loss[:, offset_bin_id, 0] == larger_loss
            )

            bkg_effs = (larger_loss - loss) / (
                larger_loss - smaller_loss
            ) * self.bkg_efficiencies[smaller_ind, offset_bin_id] + (
                loss - smaller_loss
            ) / (
                larger_loss - smaller_loss
            ) * self.bkg_efficiencies[
                larger_ind, offset_bin_id
            ]

            ROCMaker.make_energy_plot(
                ax,
                self.reco_energy_axis.center,
                bkg_effs.reshape(
                    len(self.reco_energy_axis.center),
                ),
                label="Signal efficiency={}".format(efficiency),
                **kwargs
            )

        ax.legend()
"""