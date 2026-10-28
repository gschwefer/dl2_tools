import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.data import Observation, observatory_locations
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import SensitivityEstimator
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom
from pyirf.binning import create_bins_per_decade
from regions import CircleSkyRegion

from ctapipe.core import Component
from ctapipe.core.traits import CaselessStrEnum, Float, Int, List, Unicode


class PSSensitivityCalculator(Component):
    """
    Point-source sensitivity for a single pointing.

    The ON region is shrunk to the PSF containment radius in each energy bin:
    the exposure is scaled by the containment fraction and the background,
    computed for a fixed ON region, is rescaled to the solid angle of the
    containment radius.
    """

    livetime = Unicode(
        default_value="50 h",
        help="Observation time, as a string parsable by astropy (e.g. '50 h')",
    ).tag(config=True)

    location_name = Unicode(
        default_value="cta_south",
        help="Name of the observatory location (key of gammapy observatory_locations)",
    ).tag(config=True)

    offset = Float(
        default_value=0.0,
        help="FoV offset of the source from the pointing position in deg",
    ).tag(config=True)

    on_region_radius = Float(
        default_value=0.1,
        help=(
            "Radius of the nominal ON region in deg. The background is computed for "
            "this region and then rescaled to the PSF containment radius."
        ),
    ).tag(config=True)

    containment = Float(
        default_value=0.68,
        help="PSF containment fraction defining the ON region (between 0 and 1)",
    ).tag(config=True)

    alpha = Float(
        default_value=0.2,
        allow_none=False,
        help="Ratio of ON to OFF exposure (acceptance_on / acceptance_off)",
    ).tag(config=True)

    gamma_min = Int(
        default_value=10,
        help="Minimum number of gamma-ray events for sensitivity",
    ).tag(config=True)

    n_sigma = Float(
        default_value=5.0,
        help="Number of sigma to use for the sensitivity curve",
    ).tag(config=True)

    bkg_syst_fraction = Float(
        default_value=0.05,
        help="Background relative systematic uncertainty",
    ).tag(config=True)

    true_energy_bounds = List(
        Unicode(),
        default_value=["0.05 TeV", "200 TeV"],
        minlen=2,
        maxlen=2,
        help="Minimum and maximum values of true energy",
    ).tag(config=True)

    true_energy_bins_per_decade = Int(
        default_value=5,
        help="Number of true energy bins per decade",
    ).tag(config=True)

    reco_energy_bounds = List(
        Unicode(),
        default_value=["0.05 TeV", "200 TeV"],
        minlen=2,
        maxlen=2,
        help="Minimum and maximum values of reco energy",
    ).tag(config=True)

    reco_energy_bins_per_decade = Int(
        default_value=5,
        help="Number of reco energy bins per decade to evaluate the sensitivity in",
    ).tag(config=True)

    energy_axis_interp = CaselessStrEnum(
        ["lin", "log"],
        default_value="log",
        help=(
            "Interpolation of the energy axes, which sets where the bin centers lie. "
            "'log' gives geometric bin centers; 'lin' gives arithmetic bin centers, "
            "as used in the reference notebook."
        ),
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

        if not 0 < self.containment < 1:
            raise ValueError(f"containment must be in (0, 1), got {self.containment}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

        self._livetime = u.Quantity(self.livetime)
        self.location = observatory_locations[self.location_name]
        self.pointing = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)

        self.energy_axis = self._make_energy_axis(
            self.reco_energy_bounds, self.reco_energy_bins_per_decade, "energy"
        )
        energy_axis_true = self._make_energy_axis(
            self.true_energy_bounds, self.true_energy_bins_per_decade, "energy_true"
        )

        source_position = self.pointing.directional_offset_by(
            0 * u.deg, self.offset * u.deg
        )
        self.on_region = CircleSkyRegion(
            source_position, radius=self.on_region_radius * u.deg
        )
        self.geom = RegionGeom.create(self.on_region, axes=[self.energy_axis])

        self.empty_dataset = SpectrumDataset.create(
            geom=self.geom, energy_axis_true=energy_axis_true
        )

        self.spectrum_maker = SpectrumDatasetMaker(
            selection=["exposure", "edisp", "background"],
            containment_correction=False,
        )

        self.sensitivity_estimator = SensitivityEstimator(
            gamma_min=self.gamma_min,
            n_sigma=self.n_sigma,
            bkg_syst_fraction=self.bkg_syst_fraction,
        )

    def _make_energy_axis(self, bounds, bins_per_decade, name):
        edges = create_bins_per_decade(
            u.Quantity(bounds[0]), u.Quantity(bounds[1]), bins_per_decade
        )
        return MapAxis.from_edges(edges, name=name, interp=self.energy_axis_interp)

    def __call__(self, irf_handler):
        missing = [
            name
            for name in ("aeff", "psf", "edisp", "bkg_model")
            if getattr(irf_handler, name, None) is None
        ]
        if missing:
            raise ValueError(f"IRF handler is missing: {', '.join(missing)}")

        irf_dict = {
            "aeff": irf_handler.aeff,
            "psf": irf_handler.psf,
            "edisp": irf_handler.edisp,
            "bkg": irf_handler.bkg_model,
        }

        obs = Observation.create(
            pointing=self.pointing,
            irfs=irf_dict,
            livetime=self._livetime,
            location=self.location,
        )
        dataset = self.spectrum_maker.run(self.empty_dataset, obs)

        # Only the contained fraction of the signal falls in the ON region
        dataset.exposure *= self.containment

        # Rescale background from the nominal ON region to the containment radius
        on_radii = obs.psf.containment_radius(
            energy_true=self.energy_axis.center,
            offset=self.offset * u.deg,
            fraction=self.containment,
        )
        factor = (1 - np.cos(on_radii)) / (1 - np.cos(self.on_region.radius))
        dataset.background *= factor.to_value(u.one).reshape((-1, 1, 1))

        dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
            dataset=dataset, acceptance=1, acceptance_off=1 / self.alpha
        )
        sensitivity_table = self.sensitivity_estimator.run(dataset_on_off)

        # gammapy < 1.2 only provides the bin centers ("energy"); add the
        # reference energy and bin edges so the output is the same for all versions
        if "e_ref" not in sensitivity_table.colnames:
            sensitivity_table["e_ref"] = self.energy_axis.center
        if "e_min" not in sensitivity_table.colnames:
            sensitivity_table["e_min"] = self.energy_axis.edges_min
        if "e_max" not in sensitivity_table.colnames:
            sensitivity_table["e_max"] = self.energy_axis.edges_max

        return sensitivity_table

    @staticmethod
    def plot_sensitivity_curve(ax, sensitivity_table, color=None, label_suffix=""):
        """Plot a sensitivity table, one marker style per limiting criterion."""
        t = sensitivity_table
        markers = {"significance": "s", "gamma": "*", "bkg": "v"}

        for criterion, marker in markers.items():
            sel = t[t["criterion"] == criterion]
            if len(sel) == 0:
                continue
            ax.errorbar(
                sel["e_ref"],
                sel["e2dnde"],
                xerr=0.5 * (sel["e_max"] - sel["e_min"]),
                linestyle="",
                marker=marker,
                color=color,
                label=f"{criterion}{label_suffix}",
            )

        ax.loglog()
        ax.set_xlabel(f"Energy [{t['e_ref'].unit.to_string('latex_inline')}]")
        ax.set_ylabel(f"Sensitivity [{t['e2dnde'].unit.to_string('latex_inline')}]")
        ax.legend()
