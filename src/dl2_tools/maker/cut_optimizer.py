import numpy as np
from copy import deepcopy
from ..handler.interpolated_cuts import InterpolatedCut
from gammapy.maps import MapAxis
import operator
import astropy.units as u
from pyirf.cuts import calculate_percentile_cut, weighted_quantile
from pyirf.cut_optimization import optimize_gh_cut
from ..handler.observation_handler import (
    PointSourceObservationHandler,
)
from astropy import table
from ..handler.data_lists import (
    DiffuseSetList,
    PointSourceSetList,
    PointSourceSignalSetList,
)
from ctapipe.core import Component
from ..handler.binning import IRFBinning
from traitlets import Float
from ctapipe.core.traits import List, Unicode, Int


class CutCalculator(Component):

    cut_variable = Unicode(
        default_value="gh_score",
        help="Name of the variable to cut on",
    ).tag(config=True)

    op_name = Unicode(
        default_value="operator.ge",
        allow_none=False,
        help="Name of operator to evaluate cut with",
    ).tag(config=True)

    cut_op_name = Unicode(
        default_value="lambda x: x",
        allow_none=False,
        help="Name of Operator to apply on cut_variable column before applying cut",
    ).tag(config=True)

    def __init__(self, config=None, parent=None) -> None:
        super().__init__(config=config, parent=parent)
        self.cut_op = eval(self.cut_op_name)
        self.op = eval(self.op_name)


class RecoEnergyPointSourceGHCutOptimizer(CutCalculator):

    energy_reco_bins = List(
        default_value=[0.01, 200, 20],
        help="Emin, Emax and number of bins in reco energy",
    ).tag(config=True)

    initial_cut_efficiency = Float(
        help="Percentile to cut on",
        default_value=0.8,
    ).tag(config=True)

    max_bkg_radius = Float(
        help="maximum bkg_radius to consider in degree",
        default_value=1,
        allow_none=False,
    ).tag(config=True)

    alpha = Float(
        help="Ratio of ON to OFF region",
        default_value=1,
        allow_none=False,
    ).tag(config=True)

    gh_cut_efficiency_params = List(
        default_value=[0.1, 0.8, 71],
        help="min, max and number of bins",
    ).tag(config=True)

    def __init__(self, config=None, parent=None) -> None:
        super().__init__(config=config, parent=parent)

        self.reco_energy_axis = MapAxis(
            np.geomspace(
                self.energy_reco_bins[0] * u.TeV,
                self.energy_reco_bins[1] * u.TeV,
                self.energy_reco_bins[2],
            ),
            interp="log",
            name="reco_energy",
            node_type="edges",
        )

        self.gh_cut_efficiencies = np.linspace(
            self.gh_cut_efficiency_params[0],
            self.gh_cut_efficiency_params[1],
            self.gh_cut_efficiency_params[2],
        )

        assert self.reco_energy_axis.unit.is_equivalent(u.TeV)

    def __call__(self, observation):
        assert isinstance(observation, PointSourceObservationHandler)

        for bg in observation.background:
            assert (
                bg.radius
                >= observation.signal.get_offsets()[-1] + self.max_bkg_radius * u.deg
            )

        background = deepcopy(observation.background)
        gh_cut_tabs = []
        theta_cut_tabs = []

        self.theta_cut_calculator = PercentileCutCalculator()
        self.theta_cut_calculator.cut_variable = "theta"
        self.theta_cut_calculator.fill_value = 0.32
        self.theta_cut_calculator.percentile = 68
        self.theta_cut_calculator.bin_axes = [self.reco_energy_axis]
        self.theta_cut_calculator.op = operator.le
        self.theta_cut_calculator.cut_op = lambda x: x.to_value(u.deg)
        self.theta_cut_calculator.min_value = 0.01
        self.theta_cut_calculator.max_value = 0.5

        self.init_gh_cut, self.coarse_theta_cut = self._calc_initial_cuts(
            observation.signal
        )

        for signal in deepcopy(observation.signal):
            gh_cut_tab, theta_cut_tab = self._calc_optimal_cut_single_offset_bin(
                signal, background
            )
            gh_cut_tabs.append(gh_cut_tab)
            theta_cut_tabs.append(theta_cut_tab)

        signal_offset_axis = IRFBinning.make_ps_signal_offset_bins(observation.signal)

        gh_cut = InterpolatedCut.from_cut_tables(
            cut_tables=gh_cut_tabs,
            offset_axis=signal_offset_axis,
            bin_axis_name="reco_energy",
            cut_column=self.cut_variable,
            op=self.op,
            cut_op=self.cut_op,
            method="nearest",
            bounds_error=False,
            fill_value=None,
        )

        theta_cut = InterpolatedCut.from_cut_tables(
            cut_tables=theta_cut_tabs,
            offset_axis=signal_offset_axis,
            bin_axis_name="reco_energy",
            cut_column="theta",
            op=operator.le,
            cut_op=lambda x: x.to_value(u.deg),
            method="nearest",
            bounds_error=False,
            fill_value=None,
        )

        return gh_cut, theta_cut

    def _calc_initial_cuts(self, signal):

        initial_gh_cut_calculator = PercentileCutCalculator()
        initial_gh_cut_calculator.cut_variable = self.cut_variable
        initial_gh_cut_calculator.fill_value = 0.8
        initial_gh_cut_calculator.percentile = 100 * self.initial_cut_efficiency
        initial_gh_cut_calculator.bin_axes = None
        initial_gh_cut_calculator.op = self.op
        initial_gh_cut_calculator.cut_op = self.cut_op

        initial_gh_cut = initial_gh_cut_calculator(signal, "true_source_fov_offset")

        signal_after_init_cut = deepcopy(signal)

        signal_after_init_cut.add_cut(
            initial_gh_cut, offset_column="true_source_fov_offset"
        )

        coarse_theta_cut = self.theta_cut_calculator(
            signal_after_init_cut, "true_source_fov_offset"
        )

        return initial_gh_cut, coarse_theta_cut

    def _calc_optimal_cut_single_offset_bin(self, signal, background):

        offset_axis = MapAxis(
            nodes=u.Quantity(
                [signal.offset - 0.01 * u.deg, signal.offset + 0.01 * u.deg]
            ),
            interp="lin",
            node_type="edges",
            name="true_source_fov_offset",
        )

        background_events = deepcopy(
            table.vstack([bkg.get_masked_events() for bkg in background])
        )

        _, gh_cuts_table = optimize_gh_cut(
            deepcopy(signal).get_masked_events(),
            background_events,
            reco_energy_bins=self.reco_energy_axis.edges,
            gh_cut_efficiencies=self.gh_cut_efficiencies,
            op=operator.ge,
            theta_cuts=self.coarse_theta_cut.to_cut_table(
                signal.offset, bin_axis=self.reco_energy_axis, cut_column_unit=u.deg
            ),
            alpha=self.alpha,
            fov_offset_max=signal.offset + self.max_bkg_radius * u.deg,
            fov_offset_min=max(signal.offset - self.max_bkg_radius * u.deg, 0 * u.deg),
        )

        gh_cut = InterpolatedCut.from_cut_tables(
            [gh_cuts_table],
            offset_axis=offset_axis,
            bin_axis_name="reco_energy",
            cut_column="gh_score",
            op=operator.ge,
            method="nearest",
            bounds_error=False,
            fill_value=None,
        )

        signal_after_final_cut = deepcopy(signal)

        signal_after_final_cut.add_cut(gh_cut, offset_column="true_source_fov_offset")

        fine_theta_cut = self.theta_cut_calculator(
            PointSourceSignalSetList([signal_after_final_cut]), "true_source_fov_offset"
        )

        fine_theta_cut_table = fine_theta_cut.to_cut_table(
            signal.offset, self.reco_energy_axis, cut_column_unit=u.deg
        )

        return gh_cuts_table, fine_theta_cut_table


class PercentileCutCalculator(CutCalculator):

    fill_value = Float(
        help="Fill value when bin has less than min_events entries",
    ).tag(config=True)

    percentile = Float(
        help="Percentile to cut on. Always describes the percentage of events that are kept after the cut.",
        default_value=68,
    ).tag(config=True)

    min_value = Float(
        help="Minimum value of cut", default_value=None, allow_none=True
    ).tag(config=True)

    max_value = Float(
        help="Maximum value of cut", default_value=None, allow_none=True
    ).tag(config=True)

    smoothing = Float(
        help="Gaussian Filter Smoothing", default_value=None, allow_none=True
    ).tag(config=True)

    min_events = Int(
        help="Minimum number of events in bin to evaluate cut",
        default_value=10,
        allow_none=False,
    ).tag(config=True)

    bin_axes_params = List(
        default_value=None,
        help="List of lists each containing the min, max, number of bins and the name of the axis",
        allow_none=True,
    ).tag(config=True)

    N_offset_bins = Int(
        default_value=1,
        help="Number of signal offset bins if signal is diffuse",
    ).tag(config=True)

    def __init__(self, config=None, parent=None):
        super().__init__(config=config, parent=parent)

        if self.bin_axes_params is None:
            self.bin_axes = None

        else:
            self.bin_axes = []
            for ax_params in self.bin_axes_params:

                if ax_params[0].unit.is_equivalent(u.TeV):
                    new_axis = MapAxis(
                        np.geomspace(ax_params[0], ax_params[1], ax_params[2]),
                        interp="log",
                        name=ax_params[3],
                        node_type="edges",
                    )
                else:
                    new_axis = MapAxis(
                        np.linspace(ax_params[0], ax_params[1], ax_params[2]),
                        interp="lin",
                        name=ax_params[3],
                        node_type="edges",
                    )

                self.bin_axes.append(new_axis)

    def __call__(self, datalist, offset_axis_name):

        if isinstance(datalist, PointSourceSetList):

            cut = self._calc_percentile_cut_ps(datalist, offset_axis_name)

        elif isinstance(datalist, DiffuseSetList):
            assert len(datalist) == 1
            # max_radius = np.max(datalist.get_radii())
            offset_axis = IRFBinning.make_diffuse_signal_offset_bins(
                datalist, self.N_offset_bins
            )
            offset_axis._name = offset_axis_name
            # offset_axis = MapAxis.from_bounds(
            #    0.0 * u.deg, max_radius, self.N_offset_bins, name=offset_axis_name
            # )

            cut = self._calc_percentile_cut_diffuse(datalist[0], offset_axis)
        return cut

    def _calc_percentile_cut_diffuse(self, dataset, offset_axis):

        events = dataset.get_masked_events()
        offset_name = offset_axis.name

        if self.bin_axes is None:

            cut_values = []

            for low_edge, high_edge in zip(offset_axis.edges[:-1], offset_axis[1:]):

                mask = np.logical_and(
                    events[offset_name] >= low_edge, events[offset_name] < high_edge
                )

                masked_events = events[mask]

                if self.op(1, -1):
                    cut_value = weighted_quantile(
                        self.cut_op(masked_events[self.cut_variable]),
                        masked_events["weight"],
                        1 - self.percentile / 100,
                    )

                elif self.op(-1, 1):
                    cut_value = weighted_quantile(
                        self.cut_op(masked_events[self.cut_variable]),
                        masked_events["weight"],
                        self.percentile / 100,
                    )

                cut_values.append(cut_value)

            cut = InterpolatedCut(
                offset_axis=offset_axis,
                cut_column=self.cut_variable,
                cut_values=cut_values,
                op=self.op,
                cut_op=self.cut_op,
                method="nearest",
                bounds_error=False,
                fill_value=cut_value,
            )
        elif len(self.bin_axes) == 1:

            cut_tabs = []

            for low_edge, high_edge in zip(
                offset_axis.edges[:-1], offset_axis.edges[1:]
            ):

                mask = np.logical_and(
                    events[offset_name] >= low_edge, events[offset_name] < high_edge
                )

                masked_events = events[mask]

                if self.op(1, -1):
                    cut_table = calculate_percentile_cut(
                        self.cut_op(masked_events[self.cut_variable]),
                        masked_events[self.bin_axes[0].name],
                        bins=self.bin_axes[0].edges,
                        min_value=self.min_value,
                        weights=masked_events["weight"],
                        fill_value=self.fill_value,
                        max_value=self.max_value,
                        percentile=100 - self.percentile,
                        smoothing=self.smoothing,
                        min_events=self.min_events,
                    )
                elif self.op(-1, 1):
                    cut_table = calculate_percentile_cut(
                        self.cut_op(masked_events[self.cut_variable]),
                        masked_events[self.bin_axes[0].name],
                        bins=self.bin_axes[0].edges,
                        min_value=self.min_value,
                        weights=masked_events["weight"],
                        fill_value=self.fill_value,
                        max_value=self.max_value,
                        percentile=self.percentile,
                        smoothing=self.smoothing,
                        min_events=self.min_events,
                    )

                cut_tabs.append(cut_table)

            cut = InterpolatedCut.from_cut_tables(
                cut_tables=cut_tabs,
                offset_axis=offset_axis,
                bin_axis_name=self.bin_axes[0].name,
                cut_column=self.cut_variable,
                op=self.op,
                cut_op=self.cut_op,
                method="nearest",
                bounds_error=False,
                fill_value=None,
            )

        else:
            # To be iplemented in the future using itertools.product()
            # Essentiallz, this involves a loop over an unknown number
            # of axes. The innermost axis is each time evaluated using
            # evaluate_binned_cut()
            pass
        return cut

    def _calc_percentile_cut_ps(self, datalist, offset_name):

        # events = dataset.get_masked_events()
        # offset_name = offset_axis.name

        offsets = datalist.get_offsets()

        if self.bin_axes is None:

            cut_values = []

            for dset in datalist:
                events = dset.get_masked_events()

                if self.op(1, -1):
                    cut_value = weighted_quantile(
                        self.cut_op(events[self.cut_variable]),
                        events["weight"],
                        1 - self.percentile / 100,
                    )

                elif self.op(-1, 1):
                    cut_value = weighted_quantile(
                        self.cut_op(events[self.cut_variable]),
                        events["weight"],
                        self.percentile / 100,
                    )

                cut_values.append(cut_value)

            cut = InterpolatedCut(
                offset_axis=MapAxis(
                    nodes=offsets, node_type="center", interp="lin", name=offset_name
                ),
                cut_column=self.cut_variable,
                cut_values=cut_values,
                op=self.op,
                cut_op=self.cut_op,
                method="nearest",
                bounds_error=False,
                fill_value=cut_value,
            )

        elif len(self.bin_axes) == 1:

            cut_tabs = []

            for dset in datalist:
                events = dset.get_masked_events()

                if self.op(1, -1):
                    cut_table = calculate_percentile_cut(
                        self.cut_op(events[self.cut_variable]),
                        events[self.bin_axes[0].name],
                        bins=self.bin_axes[0].edges,
                        min_value=self.min_value,
                        weights=events["weight"],
                        fill_value=self.fill_value,
                        max_value=self.max_value,
                        percentile=100 - self.percentile,
                        smoothing=self.smoothing,
                        min_events=self.min_events,
                    )
                elif self.op(-1, 1):
                    cut_table = calculate_percentile_cut(
                        self.cut_op(events[self.cut_variable]),
                        events[self.bin_axes[0].name],
                        bins=self.bin_axes[0].edges,
                        min_value=self.min_value,
                        weights=events["weight"],
                        fill_value=self.fill_value,
                        max_value=self.max_value,
                        percentile=self.percentile,
                        smoothing=self.smoothing,
                        min_events=self.min_events,
                    )

                cut_tabs.append(cut_table)

            cut = InterpolatedCut.from_cut_tables(
                cut_tables=cut_tabs,
                offset_axis=MapAxis(
                    nodes=offsets, node_type="center", interp="lin", name=offset_name
                ),
                bin_axis_name=self.bin_axes[0].name,
                cut_column=self.cut_variable,
                op=self.op,
                cut_op=self.cut_op,
                method="nearest",
                bounds_error=False,
                fill_value=None,
            )

        else:
            # To be iplemented in the future using itertools.product()
            # Essentiallz, this involves a loop over an unknown number
            # of axes. The innermost axis is each time evaluated using
            # evaluate_binned_cut()
            pass
        return cut
