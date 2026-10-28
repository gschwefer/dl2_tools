# This should take an observation handler after gamma hadron cuts and calculate all the IRFS from it
# These can then be passed to the cut optimizer for gammapy cut optimization or saved to disk
# It needs to cnatin conditions on the Observationhandler in order to produce irfs from it

from ..handler.irf_handler import IRFHandler
from pyirf.irf import (
    energy_dispersion,
    effective_area_per_energy,
    effective_area_per_energy_and_fov,
    psf_table,
    background_2d,
    BACKGROUND_UNIT,
)
from pyirf.gammapy import (
    create_effective_area_table_2d,
    create_energy_dispersion_2d,
    create_psf_3d,
    create_background_2d,
)

from pyirf.benchmarks import energy_bias_resolution_from_energy_dispersion
import copy
import astropy.units as u
import numpy as np

from ..handler.data_lists import DiffuseSignalSetList

from ctapipe.core import Component
from ctapipe.core.traits import Bool, Float

from scipy.interpolate import RegularGridInterpolator

from ..handler.binning import IRFBinning


def _cancelling_unit(model_unit, physical_type):
    """Find the unit in model_unit's denominator with the given physical type (e.g. TeV in s-1 TeV-1 sr-1)."""
    model_unit = u.Unit(model_unit)  # gammapy may store the unit as a string
    for base, power in zip(model_unit.bases, model_unit.powers):
        candidate = base ** -power
        if power < 0 and candidate.physical_type == physical_type:
            return candidate
    raise u.UnitsError(f"Model unit '{model_unit}' is not per {physical_type}")


def integrate_bkg_model(bkg_model, theta_max, average_over_sr=False, integrate_over_energy=False):
    """
    Integrate a Background2D model over the cone of half-angle theta_max,
    and optionally within each energy bin.

    Parameters
    ----------
    bkg_model : `~gammapy.irf.Background2D`
        Model with axes "energy" and "offset", in units like s-1 TeV-1 sr-1.
    theta_max : `~astropy.units.Quantity` or float
        Cone half-angle. A plain float is taken to be in degrees.
    average_over_sr : bool
        False: integrate over the solid angle (removes sr-1).
        True:  average over the solid angle (keeps sr-1).
    integrate_over_energy : bool
        False: return the differential rate per energy bin (keeps TeV-1).
        True:  integrate within each energy bin (removes TeV-1); same shape.

    Returns
    -------
    `~astropy.units.Quantity`
        One value per energy bin. For a model in s-1 TeV-1 sr-1:
            average_over_sr=False, integrate_over_energy=False -> s-1 TeV-1
            average_over_sr=True,  integrate_over_energy=False -> s-1 TeV-1 sr-1
            average_over_sr=False, integrate_over_energy=True  -> s-1
            average_over_sr=True,  integrate_over_energy=True  -> s-1 sr-1
    """
    theta_axis = bkg_model.axes["offset"]
    theta_max = u.Quantity(theta_max, "deg")

    # Exact solid angle of each offset ring, cut off at theta_max (zero for rings beyond it).
    # Expressed in the model's own solid-angle unit (sr or deg2) so it cancels cleanly.
    theta_lo = theta_axis.edges[:-1]
    theta_hi = np.minimum(theta_axis.edges[1:], theta_max)
    d_omega = 2 * np.pi * (np.cos(theta_lo) - np.cos(theta_hi)) * u.sr
    d_omega = np.maximum(d_omega, 0 * u.sr)
    d_omega = d_omega.to(_cancelling_unit(bkg_model.unit, "solid angle"))

    # Move the offset axis last so it doesn't matter how the model orders its axes
    offset_idx = bkg_model.axes.names.index("offset")
    rate = np.moveaxis(bkg_model.quantity, offset_idx, -1)

    # Integrate over solid angle -> one value per energy bin
    result = (rate * d_omega).sum(axis=-1)

    if average_over_sr:
        result = result / d_omega.sum()

    if integrate_over_energy:
        # Multiply each bin by its width: removes the energy unit, keeps one value per bin
        energy_axis = bkg_model.axes["energy"]
        e_width = energy_axis.bin_width.to(_cancelling_unit(bkg_model.unit, "energy"))
        result = result * e_width

    return result


def smooth_bkg_model(bkg_model, theta_max):
    """
    Build a new background model assuming the offset dependence is energy independent.

    The offset shape is the original model integrated over the full energy range.
    It is scaled so that, in every energy bin, the new model has the same rate
    inside theta_max as the original.

    Parameters
    ----------
    bkg_model : `~gammapy.irf.Background2D`
        Original model with axes "energy" and "offset".
    theta_max : `~astropy.units.Quantity` or float
        Cone half-angle within which the rate is preserved. A plain float is taken to be in degrees.

    Returns
    -------
    `~gammapy.irf.Background2D`
        New model with the same axes, unit and metadata.
    """
    theta_axis = bkg_model.axes["offset"]
    energy_axis = bkg_model.axes["energy"]
    theta_max = u.Quantity(theta_max, "deg")

    # Put the model in (energy, offset) order regardless of how it's stored
    energy_idx = bkg_model.axes.names.index("energy")
    rate = np.moveaxis(bkg_model.quantity, energy_idx, 0)

    # Energy-integrated offset profile; its absolute units cancel in the normalisation
    profile = (rate * energy_axis.bin_width[:, np.newaxis]).sum(axis=0)

    # Same ring solid angles as in integrate_bkg_model
    theta_lo = theta_axis.edges[:-1]
    theta_hi = np.minimum(theta_axis.edges[1:], theta_max)
    d_omega = np.maximum(2 * np.pi * (np.cos(theta_lo) - np.cos(theta_hi)), 0) * u.sr

    # Normalise the profile to a mean of 1 over the cone
    mean_profile = (profile * d_omega).sum() / d_omega.sum()
    shape = (profile / mean_profile).to_value(u.dimensionless_unscaled)

    # Solid-angle-averaged differential rate per energy bin (must stay per TeV)
    bkg_mean = integrate_bkg_model(bkg_model, theta_max, average_over_sr=True)

    # Outer product -> (energy, offset), then restore the model's axis order
    new_quantity = bkg_mean[:, np.newaxis] * shape[np.newaxis, :]
    new_quantity = np.moveaxis(new_quantity, 0, energy_idx)

    new_model = copy.deepcopy(bkg_model)  # keeps axes, FoV alignment and metadata
    new_model.quantity = new_quantity.to(bkg_model.unit)
    return new_model


class IRFMaker(Component):
    smooth_background = Bool(
        default_value=False,
        help=(
            "Smooth the background model by assuming its offset dependence is "
            "energy independent: the energy-integrated offset profile is scaled to "
            "the rate inside smooth_background_theta_max in each energy bin."
        ),
    ).tag(config=True)

    smooth_background_theta_max = Float(
        default_value=None,
        allow_none=True,
        help=(
            "Offset (deg) within which the background rate per energy bin is "
            "preserved when smoothing. None uses the full background offset axis."
        ),
    ).tag(config=True)

    def __init__(self, config=None, parent=None) -> None:
        super().__init__(config=config, parent=parent)

    def __call__(self, observation, binning=None):
        if binning is None:
            self.binning = IRFBinning(observation, parent=self)
        else:
            self.binning = binning
        # Check conditions on observations

        irfs = IRFHandler()

        if len(observation.signal) > 0:
            irfs.edisp, irfs.energy_bias, irfs.energy_resolution = self._make_Edisps(
                observation.signal
            )
            irfs.aeff = self._make_Aeff(observation.signal)
            irfs.psf = self._make_psf(observation.signal)
            irfs.binning = self.binning

        if len(observation.background) != 0:
            irfs.bkg_model = self._make_background_model(observation.background)

        return irfs

    def _make_background_model(self, background):
        pyirf_bkg = u.Quantity(
            np.zeros(
                (
                    len(self.binning.energy_reco.center),
                    len(
                        self.binning.background_offset.center,
                    ),
                )
            )
            * BACKGROUND_UNIT
        )

        for bkg in background:
            single_bkg = background_2d(
                bkg.get_masked_events(),
                self.binning.energy_reco.edges,
                self.binning.background_offset.edges,
                bkg.obs_time,
            )
            pyirf_bkg += single_bkg

        gammapy_bkg = create_background_2d(
            pyirf_bkg,
            self.binning.energy_reco.edges,
            self.binning.background_offset.edges,
        )

        if self.smooth_background:
            if self.smooth_background_theta_max is None:
                theta_max = self.binning.background_offset.edges[-1]
            else:
                theta_max = self.smooth_background_theta_max * u.deg
            gammapy_bkg = smooth_bkg_model(gammapy_bkg, theta_max)

        return gammapy_bkg

    def _make_Edisps(self, signal):
        if isinstance(signal, DiffuseSignalSetList):
            edisp_pyirf = self._make_Edisps_diffuse(signal[0])
            offset_points = self.binning.signal_offset.center.to_value(u.deg)

        else:
            edisp_pyirf = self._make_Edisps_pl(signal)
            offset_points = signal.get_offsets().to_value(u.deg)

        edisp = create_energy_dispersion_2d(
            edisp_pyirf,
            self.binning.energy_true.edges,
            self.binning.migration.edges,
            self.binning.signal_offset.edges,
        )

        (
            pyirf_energy_bias,
            pyirf_energy_resolution,
        ) = energy_bias_resolution_from_energy_dispersion(
            edisp_pyirf, self.binning.migration.edges
        )

        energy_bias_spline = RegularGridInterpolator(
            (
                np.log10(self.binning.energy_true.center.to_value(u.TeV)),
                offset_points,
            ),
            pyirf_energy_bias,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        energy_resolution_spline = RegularGridInterpolator(
            (
                np.log10(self.binning.energy_true.center.to_value(u.TeV)),
                offset_points,
            ),
            pyirf_energy_resolution,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        return edisp, energy_bias_spline, energy_resolution_spline

    def _make_Edisps_diffuse(self, signal):
        edisp_pyirf = energy_dispersion(
            signal.get_masked_events(),
            true_energy_bins=self.binning.energy_true.edges,
            migration_bins=self.binning.migration.edges,
            fov_offset_bins=self.binning.signal_offset.edges,
        )
        return edisp_pyirf

    def _make_Edisps_pl(self, signal):
        edisp_pyirf = np.empty(
            (
                len(self.binning.energy_true.center),
                len(self.binning.migration.center),
                len(self.binning.signal_offset.center),
            )
        )

        for i, _sig in enumerate(signal):
            edisp_pyirf[:, :, i] = energy_dispersion(
                _sig.get_masked_events(),
                true_energy_bins=self.binning.energy_true.edges,
                migration_bins=self.binning.migration.edges,
                fov_offset_bins=u.Quantity(
                    [
                        self.binning.signal_offset.edges[i],
                        self.binning.signal_offset.edges[i + 1],
                    ]
                ),
                use_event_weights=True
            ).reshape(
                len(self.binning.energy_true.center), len(self.binning.migration.center)
            )

        return edisp_pyirf

    def _make_Aeff(self, signal):
        if isinstance(signal, DiffuseSignalSetList):
            aeff_pyirf = self._make_Aeff_diffuse(signal[0])
        else:
            aeff_pyirf = self._make_Aeff_pl(signal)

        aeff = create_effective_area_table_2d(
            aeff_pyirf, self.binning.energy_true.edges, self.binning.signal_offset.edges
        )
        return aeff

    def _make_Aeff_pl(self, signal):
        aeff_pyirf = u.Quantity(
            np.empty(
                (
                    len(self.binning.energy_true.center),
                    len(self.binning.signal_offset.center),
                )
            ),
            u.m**2,
        )
        for i, _sig in enumerate(signal):
            aeff_pyirf[:, i] = effective_area_per_energy(
                _sig.get_masked_events(),
                _sig.simulation_info,
                self.binning.energy_true.edges,
                use_event_weights=False
            )

        return aeff_pyirf

    def _make_Aeff_diffuse(self, signal):
        aeff_pyirf = effective_area_per_energy_and_fov(
            signal.get_masked_events(),
            signal.simulation_info,
            self.binning.energy_true.edges,
            self.binning.signal_offset.edges,
            use_event_weights=False
        )
        return aeff_pyirf

    def _make_psf(self, signal):
        if isinstance(signal, DiffuseSignalSetList):
            psf_pyirf = self._make_psf_diffuse(signal[0])
        else:
            psf_pyirf = self._make_psf_pl(signal)

        psf = create_psf_3d(
            psf_pyirf,
            self.binning.energy_true.edges,
            self.binning.psf_offset.edges,
            self.binning.signal_offset.edges,
        )
        return psf

    def _make_psf_pl(self, signal):
        psf_pyirf = u.Quantity(
            np.empty(
                (
                    len(self.binning.energy_true.center),
                    len(self.binning.signal_offset.center),
                    len(self.binning.psf_offset.center),
                )
            ),
            unit=u.sr**-1,
        )

        for i, _sig in enumerate(signal):
            psf_pyirf[:, i, :] = np.swapaxes(
                psf_table(
                    _sig.get_masked_events(),
                    true_energy_bins=self.binning.energy_true.edges,
                    source_offset_bins=self.binning.psf_offset.edges,
                    fov_offset_bins=u.Quantity(
                        [
                            self.binning.signal_offset.edges[i],
                            self.binning.signal_offset.edges[i + 1],
                        ]
                    ),
                    use_event_weights=True
                ),
                1,
                2,
            ).reshape(
                len(self.binning.energy_true.center),
                len(self.binning.psf_offset.center),
            )
        return psf_pyirf

    def _make_psf_diffuse(self, signal):
        psf_pyirf = psf_table(
            signal.get_masked_events(),
            true_energy_bins=self.binning.energy_true.edges,
            source_offset_bins=self.binning.psf_offset.edges,
            fov_offset_bins=self.binning.signal_offset.edges,
        )

        return psf_pyirf
