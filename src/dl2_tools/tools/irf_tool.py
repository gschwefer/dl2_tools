"""
Generate IRFs from dl2 data
"""
# pylint: disable=W0201

import os

from ctapipe.core import Tool, traits
from ..maker.irf_maker import IRFMaker
from ..maker.roc_maker import ROCMaker
from ctapipe.core.traits import List, Unicode, classes_with_traits, Bool, flag

from ..handler.sim_datasets import PointSourceSignalSet, DiffuseSignalSet, BackgroundSet
from ..handler.observation_handler import (
    PointSourceObservationHandler,
    DiffuseObservationHandler,
)
from ..handler.binning import IRFBinning
from traitlets import Float
import dill as pickle

import matplotlib.pyplot as plt
import astropy.units as u

from gammapy.maps import MapAxis

from pyirf.spectral import (
    PowerLaw,
    LogParabola,
    PowerLawWithExponentialGaussian,
    POINT_SOURCE_FLUX_UNIT,
    DIFFUSE_FLUX_UNIT,
    CRAB_HEGRA,
    CRAB_MAGIC_JHEAP2015,
    PDG_ALL_PARTICLE,
    IRFDOC_PROTON_SPECTRUM,
    DAMPE_P_He_SPECTRUM,
)


class IRFMakerTool(Tool):
    """
    Make IRFs from DL2 data files while applying a set of cuts.
    """

    name = "irf-maker"

    examples = """
    To process data with all default values:
    > ctapipe-process --input events.simtel.gz --output events.dl1.h5 --progress
    Or use an external configuration file, where you can specify all options:
    > ctapipe-process --config stage1_config.json --progress
    The config file should be in JSON or python format (see traitlets docs). For an
    example, see ctapipe/examples/stage1_config.json in the main code repo.
    """

    cuts_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Files of pickled InterpolatedCuts",
    ).tag(config=True)

    signal_cuts_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Files of pickled InterpolatedCuts applied only to the signal component",
    ).tag(config=True)

    signal_type = Unicode(
        default_value="PointSource",
        help="Type of observation",
    ).tag(config=True)

    signal_input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input dl2 files",
    ).tag(config=True)

    background_input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input dl2 files",
    ).tag(config=True)

    output_dir = traits.Path(
        default_value=None,
        help="Output directory",
        allow_none=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    energy_reco = Unicode(
        default_value="ImPACTReconstructor",
        help="Name of energy reco",
    ).tag(config=True)

    geometry_reco = Unicode(
        default_value="HillasReconstructor",
        help="Name of geometry reco",
    ).tag(config=True)

    gh_score = Unicode(
        help="Name of gh score",
    ).tag(config=True)

    signal_weight_name = Unicode(
        help="Name of the signal weights. Must be in pyirf spectral",
        default="CRAB_HEGRA",
        allow_none=False,
    ).tag(config=True)

    background_weight_name = Unicode(
        help="Name of the signal weights. Must be in pyirf spectral",
        default="IRFDOC_PROTON_SPECTRUM",
        allow_none=False,
    ).tag(config=True)

    signal_powerlaw_params = List(
        Float,
        default_value=[1e-11, -2],
        minlen=2,
        maxlen=2,
        help="Parameters to weight signal to a powerlaw spectrum",
    ).tag(config=True)

    background_powerlaw_params = List(
        Float,
        default_value=[1e-5, 2.7],
        minlen=2,
        maxlen=2,
        help="Parameters to weight baclground to a powerlaw spectrum",
    ).tag(config=True)

    signal_logparabola_params = List(
        Float,
        default_value=[1e-11, 2, -0.2],
        minlen=3,
        maxlen=3,
        help="Parameters to weight signal to a logparabola spectrum",
    ).tag(config=True)

    background_logparabola_params = List(
        Float,
        default_value=[1e-5, 2.7, 0.0],
        minlen=3,
        maxlen=3,
        help="Parameters to weight background to a logparabola spectrum",
    ).tag(config=True)

    signal_plwithexpgauss_params = List(
        Float,
        default_value=[1e-11, 2, 0, 1, 1],
        minlen=5,
        maxlen=5,
        help="Parameters to weight signal to a plwithexpgauss spectrum",
    ).tag(config=True)

    background_plwithexpgauss_params = List(
        Float,
        default_value=[1e-5, 2.7, 0.0, 1, 1],
        minlen=5,
        maxlen=5,
        help="Parameters to weight background to a plwithexpgauss spectrum",
    ).tag(config=True)

    save_plots = Bool(
        help="Save plots of the IRFs?",
        default_value=True,
    ).tag(config=True)

    make_roc = Bool(
        help="Also make the ROC?",
        default_value=False,
    ).tag(config=True)

    overwrite = Bool(
        help="Overwrite saved IRFs?",
        default_value=False,
    ).tag(config=True)

    save_suffix = Unicode(
        help="Suffix to add to the saved IRF files",
        default_value="",
    ).tag(config=True)

    aliases = {
        "signal-input": "IRFMakerTool.signal_input_files",
        "signal-type": "IRFMakerTool.signal_type",
        "background-input": "IRFMakerTool.background_input_files",
        ("o", "output"): "IRFMakerTool.output_dir",
        "cuts-files": "IRFMakerTool.cuts_files",
        "geometry-reco": "IRFMakerTool.geometry_reco",
        "energy-reco": "IRFMakerTool.energy_reco",
        "gh-score": "IRFMakerTool.gh_score",
    }

    flags = {
        **flag(
            "overwrite",
            "IRFMakerTool.overwrite",
            "Overwrite saved IRFs",
            "Do not overwrite saved IRFs",
        ),
        **flag(
            "save-plots",
            "IRFMakerTool.save_plots",
            "Save plots of the irfs together with files",
            "Only save IRFs to file",
        ),
        **flag(
            "make-roc",
            "IRFMakerTool.make_roc",
            "Make the g/h separation ROC together with the IRFs",
            "OCalcualte only the IRFs",
        ),
    }

    classes = (
        [IRFMaker, ROCMaker, IRFBinning]
        + classes_with_traits(IRFMaker)
        + classes_with_traits(ROCMaker)
        + classes_with_traits(IRFBinning)
    )

    def setup(self):

        if self.signal_type == "PointSource":
            self.observation = PointSourceObservationHandler()
            for inp_file in self.signal_input_files:
                self.observation.add_signal(
                    PointSourceSignalSet(
                        inp_file, reco_energy_name=self.energy_reco, geometry_reco_name=self.geometry_reco, gh_score_name=self.gh_score
                    )
                )

        elif self.signal_type == "Diffuse":
            self.observation = DiffuseObservationHandler()
            assert len(self.signal_input_files) == 1
            self.observation.add_signal(
                DiffuseSignalSet(
                    self.signal_input_files[0],
                    reco_energy_name=self.energy_reco,
                    geometry_reco_name=self.geometry_reco,
                    gh_score_name=self.gh_score,
                )
            )

        else:
            raise ValueError("signal_type must be either PointSource or Diffuse")

        match self.signal_weight_name:
            case "PowerLaw":
                if self.signal_type == "PointSource":
                    signal_spectrum = PowerLaw(
                        normalization=self.signal_powerlaw_params[0]
                        * POINT_SOURCE_FLUX_UNIT,
                        index=self.signal_powerlaw_params[1],
                    )
                else:
                    signal_spectrum = PowerLaw(
                        normalization=self.signal_powerlaw_params[0]
                        * DIFFUSE_FLUX_UNIT,
                        index=self.signal_powerlaw_params[1],
                    )
            case "LogParabola":
                if self.signal_type == "PointSource":
                    signal_spectrum = LogParabola(
                        normalization=self.signal_logparabola_params[0]
                        * POINT_SOURCE_FLUX_UNIT,
                        a=self.signal_logparabola_params[1],
                        b=self.signal_logparabola_params[2],
                    )
                else:
                    signal_spectrum = LogParabola(
                        normalization=self.signal_logparabola_params[0]
                        * DIFFUSE_FLUX_UNIT,
                        a=self.signal_logparabola_params[1],
                        b=self.signal_logparabola_params[2],
                    )
            case "PowerLawWithExponentialGaussian":
                if self.signal_type == "PointSource":
                    signal_spectrum = PowerLawWithExponentialGaussian(
                        normalization=self.signal_plwithexpgauss_params[0]
                        * POINT_SOURCE_FLUX_UNIT,
                        index=self.signal_plwithexpgauss_params[1],
                        f=self.signal_plwithexpgauss_params[2],
                        mu=self.signal_plwithexpgauss_params[3] * u.TeV,
                        sigma=self.signal_plwithexpgauss_params[4] * u.TeV,
                    )
                else:
                    signal_spectrum = PowerLawWithExponentialGaussian(
                        normalization=self.signal_plwithexpgauss_params[0]
                        * DIFFUSE_FLUX_UNIT,
                        index=self.signal_plwithexpgauss_params[1],
                        f=self.signal_plwithexpgauss_params[2],
                        mu=self.signal_plwithexpgauss_params[3] * u.TeV,
                        sigma=self.signal_plwithexpgauss_params[4] * u.TeV,
                    )
            case "CRAB_HEGRA":
                signal_spectrum = CRAB_HEGRA
            case "CRAB_MAGIC_JHEAP_2015":
                signal_spectrum = CRAB_MAGIC_JHEAP2015
            case _:
                raise ValueError("signal_weight_name is not among the allowed names")

        self.observation.signal.reweight_to(signal_spectrum)

        if len(self.background_input_files) > 0:
            for inp_file in self.background_input_files:
                self.observation.add_background(
                    BackgroundSet(
                        inp_file, reco_energy_name=self.energy_reco, geometry_reco_name=self.geometry_reco, gh_score_name=self.gh_score
                    )
                )

            match self.background_weight_name:
                case "PowerLaw":
                    background_spectrum = PowerLaw(
                        normalization=self.background_powerlaw_params[0]
                        * DIFFUSE_FLUX_UNIT,
                        index=self.background_powerlaw_params[1],
                    )
                case "LogParabola":
                    background_spectrum = LogParabola(
                        normalization=self.background_logparabola_params[0]
                        * DIFFUSE_FLUX_UNIT,
                        a=self.background_logparabola_params[1],
                        b=self.background_logparabola_params[2],
                    )
                case "PowerLawWithExponentialGaussian":
                    background_spectrum = PowerLawWithExponentialGaussian(
                        normalization=self.background_plwithexpgauss_params[0]
                        * DIFFUSE_FLUX_UNIT,
                        index=self.background_plwithexpgauss_params[1],
                        f=self.background_plwithexpgauss_params[2],
                        mu=self.background_plwithexpgauss_params[3] * u.TeV,
                        sigma=self.background_plwithexpgauss_params[4] * u.TeV,
                    )
                case "PDG_ALL_PARTICLE":
                    background_spectrum = PDG_ALL_PARTICLE
                case "IRFDOC_PROTON_SPECTRUM":
                    background_spectrum = IRFDOC_PROTON_SPECTRUM
                case "DAMPE_P_He_Spectrum":
                    background_spectrum = DAMPE_P_He_SPECTRUM
                case _:
                    raise ValueError(
                        "background_weight_name is not among the allowed names"
                    )

            self.observation.background.reweight_to(background_spectrum)

        cuts = []
        signal_cuts = []

        for cut_file in self.cuts_files:
            cuts.append(pickle.load(open(cut_file, "rb")))

        for signal_cut_file in self.signal_cuts_files:
            signal_cuts.append(pickle.load(open(signal_cut_file, "rb")))

        if len(cuts) > 0:

            self.observation.set_signal_cuts(cuts, "true_source_fov_offset")
            if len(self.background_input_files) > 0:
                self.observation.set_background_cuts(cuts, "reco_source_fov_offset")

        if len(signal_cuts) > 0:

            self.observation.set_signal_cuts(signal_cuts, "true_source_fov_offset")

        self.irf_maker = IRFMaker(parent=self)
        if self.make_roc:
            assert len(self.observation.background) > 0
            assert self.background_weight_name is not None
            self.roc_maker = ROCMaker(parent=self)

    def start(self):
        """
        Make IRFs
        """
        self.irfs = self.irf_maker(self.observation)
        if self.make_roc:
            self.roc_maker(self.observation)

    def finish(self):
        """
        Saving IRFs and make plots
        """

        self.irfs.save_Aeff_fits(self.output_dir, overwrite=self.overwrite, suffix=self.save_suffix)

        self.irfs.save_PSF_fits(self.output_dir, overwrite=self.overwrite, suffix=self.save_suffix)

        self.irfs.save_Edisp_fits(self.output_dir, overwrite=self.overwrite, suffix=self.save_suffix)

        self.irfs.save_bkg_model_fits(self.output_dir, overwrite=self.overwrite, suffix=self.save_suffix)

        if self.save_plots:

            aeff_fig, aeff_ax = plt.subplots()
            self.irfs.aeff.plot_energy_dependence(aeff_ax, offset=[0.0 * u.deg])
            aeff_ax.set_xlim(left=0.01 * u.TeV)
            # aeff_ax.set_ylim(bottom=1e-2)
            aeff_ax.set_yscale("log")
            aeff_ax.set_xscale("log")
            aeff_ax.grid()
            plt.savefig(os.path.join(self.output_dir, "Aeff_Edep_plot.pdf"))

            psf_fig, psf_ax = plt.subplots()
            self.irfs.psf.plot_containment_radius_vs_energy(
                psf_ax, offset=[0.0] * u.deg
            )
            psf_ax.set_xlim(left=0.01 * u.TeV)
            psf_ax.set_ylim(bottom=1e-2)
            psf_ax.set_yscale("log")
            psf_ax.set_xscale("log")
            psf_ax.grid()
            plt.savefig(os.path.join(self.output_dir, "PSF_Edep_plot.pdf"))

            energy_true_axis = MapAxis.from_energy_bounds(
                10 * u.GeV, 200 * u.TeV, 8, per_decade=True
            )
            energy_reco_axis = MapAxis.from_energy_bounds(
                1 * u.GeV, 1000 * u.TeV, 8, per_decade=True
            )
            edisp_kernel = self.irfs.edisp.to_edisp_kernel(
                offset=0.0 * u.deg,
                energy_true=energy_true_axis.edges,
                energy=energy_reco_axis.edges,
            )

            resolution = edisp_kernel.get_resolution(energy_true_axis.center)

            eres_fig, eres_ax = plt.subplots()

            eres_ax.plot(
                energy_true_axis.center,
                resolution,
                color="#006c66",
                linestyle="solid",
                marker="d",
            )

            eres_ax.set_xscale("log")
            eres_ax.set_yscale("log")
            eres_ax.set_xlabel("True energy in TeV")
            eres_ax.set_ylabel("Resolution")
            eres_ax.grid(visible=True)

            plt.savefig(os.path.join(self.output_dir, "Eres_Edep_plot.pdf"))

        if self.make_roc:

            roc_fig, roc_ax = plt.subplots()
            self.roc_maker.roc_plot_single_energy(
                roc_ax, energy=0.1 * u.TeV, offset=0.0 * u.deg
            )
            self.roc_maker.roc_plot_single_energy(
                roc_ax, energy=1 * u.TeV, offset=0.0 * u.deg
            )
            self.roc_maker.roc_plot_single_energy(
                roc_ax, energy=10 * u.TeV, offset=0.0 * u.deg
            )
            self.roc_maker.roc_plot_single_energy(
                roc_ax, energy=100 * u.TeV, offset=0.0 * u.deg
            )
            roc_ax.legend()
            plt.savefig(os.path.join(self.output_dir, "ROC_plot.pdf"))

            roc_fig_2, roc_ax_2 = plt.subplots()

            self.roc_maker.energy_plot_efficiencies(
                roc_ax_2, efficiency=0.5, offset=0.0 * u.deg
            )
            self.roc_maker.energy_plot_efficiencies(
                roc_ax_2, efficiency=0.6, offset=0.0 * u.deg
            )
            self.roc_maker.energy_plot_efficiencies(
                roc_ax_2, efficiency=0.7, offset=0.0 * u.deg
            )
            self.roc_maker.energy_plot_efficiencies(
                roc_ax_2, efficiency=0.8, offset=0.0 * u.deg
            )
            self.roc_maker.energy_plot_efficiencies(
                roc_ax_2, efficiency=0.9, offset=0.0 * u.deg
            )
            plt.savefig(os.path.join(self.output_dir, "ROC_energy_plot.pdf"))


def main():
    """run the tool"""
    tool = IRFMakerTool()
    tool.run()


if __name__ == "__main__":
    main()
