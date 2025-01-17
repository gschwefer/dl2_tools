from abc import ABC, abstractmethod
import numpy as np
from pyirf.spectral import calculate_event_weights, PowerLaw
from pyirf.simulations import SimulatedEventsInfo
from astropy.table import QTable
from collections import Counter
from astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import AltAz
import astropy.units as u
from ctapipe.io import TableLoader
from copy import copy
import uproot

class SimDataset(ABC):
    """
    Base class for handling dl2 datasets
    """

    def __init__(self, path,reco_energy_name, hess_root=False,obs_time=1 * u.s,siminfo_dict=None):
        """
        Constructor of the Base Class. Initializes Datset from ctapipe TableLoader

        Parameters
        ----------
        loader : ctapipie.io.TableLoader
            ctapipe TableLoader object conatining dl2 data
        reco_energy_name : string
            Prefix of the reconstructed energy columns in the table
        obs_time : u.Quantity time, optional
            Observation time of the dataset. Needed to make the weights.
            By default 1*u.s, this makes it easy to calculate weights in 1/s
        """
        self.obs_time = obs_time
        self.hess_root=hess_root
        if not self.hess_root:
            self.loader = TableLoader(
            path,
            load_true_images=True,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_simulated=True,
            load_dl2=True,
        )
            self.events=QTable(self.loader.read_subarray_events())
        else:
            self.events=QTable()
            file=uproot.open(path)
            self.par_tree = file["ParTree_Postselect;1"]
            self.events["true_energy"]=u.Quantity(np.array(self.par_tree["MCTrueEnergy"].array()),u.TeV)
            self.events["true_az"]=u.Quantity(np.array(self.par_tree["MCTrueAzimuth"].array()),u.deg)
            self.events["true_alt"]=u.Quantity(np.array(self.par_tree["MCTrueAlt"].array()),u.deg)
            self.events[f"{reco_energy_name}Energy"]=u.Quantity(np.array(self.par_tree[f"{reco_energy_name}Energy"].array()),u.TeV)
            self.events["mc_event_weight"]=np.array(self.par_tree["MCEventWeight"].array())
            self.events["true_core_x"]=np.array(self.par_tree["MCTrueCoreGroundX"].array())
            self.events["true_core_y"]=np.array(self.par_tree["MCTrueCoreGroundY"].array())
        self._set_reco_energy_column(reco_energy_name)
        

    @abstractmethod
    def _set_simulation_info(self, sim_config, obs_info):
        """Reads metadata from the TableLoader and creates SimulationInfo dictionary.

        Parameters
        ----------
        sim_config : astropy.table.QTable
            Created from TableLoader using read_simulation_configuration()
        obs_info : astropy.table.QTable
            Created from TableLoader using read_observation_information()
        """

        # The rows of sim_config and obs_info list the different simualtion files that went into them.
        # The asserts below make sure that certain items on the metadata are the same between them.
        assert (
            len(Counter(sim_config["energy_range_min"]).keys()) == 1
        ), "Minimum energies do not match between simulations"
        assert (
            len(Counter(sim_config["energy_range_max"]).keys()) == 1
        ), "Maximum energies do not match between simulations"
        assert (
            len(Counter(sim_config["spectral_index"]).keys()) == 1
        ), "Spectral indicies do not match between simulations"
        assert (
            len(Counter(sim_config["max_viewcone_radius"]).keys()) == 1
        ), "Viewcones do not match between simulations"
        assert (
            len(Counter(sim_config["max_scatter_range"]).keys()) == 1
        ), "Maximum impact distances do not match between simulations"
        assert (
            len(Counter(sim_config["diffuse"]).keys()) == 1
        ), "Diffuse and non-diffuse simulations are mixed here"
        assert (
            len(Counter(obs_info["subarray_pointing_lon"]).keys()) == 1
        ), "There are different pointing directions among the simulation"
        assert (
            len(Counter(obs_info["subarray_pointing_lat"]).keys()) == 1
        ), "There are different pointing directions among the simulation"
        assert (
            len(Counter(obs_info["subarray_pointing_lat"]).keys()) == 1
        ), "There are different pointing directions among the simulation"

        # If we later want to compare observations at different zeniths, we better make
        # the pointing direction a class attribute
        self.pointing_direction = AltAz(
            az=obs_info["subarray_pointing_lon"][0],
            alt=obs_info["subarray_pointing_lat"][0],
        )

    @abstractmethod
    def _set_simulation_info_hess_root(self, path, siminfo_dict):


        # If we later want to compare observations at different zeniths, we better make
        # the pointing direction a class attribute
        self.pointing_direction = AltAz(
            az=siminfo_dict["pointing_az"]*u.deg,
            alt=siminfo_dict["pointing_alt"]*u.deg,
        )

        file=uproot.open(path)
        sim_energy_histo=file.get("ThrownEnergy;1")

        self.simulation_info = SimulatedEventsInfo(
            n_showers=sim_energy_histo.values().sum(),
            max_impact=siminfo_dict["max_impact"]*u.m,
            viewcone_min=siminfo_dict["viewcone_min"]*u.deg,
            viewcone_max=siminfo_dict["viewcone_max"]*u.deg,
            energy_min=sim_energy_histo.axis().edges()[0]*u.TeV,
            energy_max=sim_energy_histo.axis().edges()[-1]*u.TeV,
            spectral_index=siminfo_dict["sim_spectral_index"],
        )

    def _set_simulation_weights(self):
        """
        Generates columns "weight" and "weight_rate" based on the simulated
        energy spectrum and specified observation time
        """
        self.spectrum = PowerLaw.from_simulation(self.simulation_info, self.obs_time)
        self.events["weight"] = np.ones(len(self.events))
        self.events["weight_rate"] = np.repeat(
            u.Quantity(1 / self.obs_time), len(self.events)
        )
        if "true_aweight" in self.events.keys():
            if np.all(~np.isnan(self.events["true_aweight"])):
                self.events["weight"]*=(self.events["true_aweight"]/(np.pi * self.simulation_info.max_impact ** 2))
                self.events["weight_rate"]*=(self.events["true_aweight"]/(np.pi * self.simulation_info.max_impact ** 2))
        if "mc_event_weight" in self.events.keys():
            self.events["weight"]*=self.events["mc_event_weight"]
            self.events["weight_rate"]*=self.events["mc_event_weight"]

    def _set_reco_energy_column(self, reco_energy_name):
        if self.hess_root:
            reco_energy_string = f"{reco_energy_name}Energy"
        else:
            reco_energy_string = f"{reco_energy_name}_energy"
        if reco_energy_name != "ImPACTReconstructor":
            reco_energy_valid_string = f"{reco_energy_name}_is_valid"
        else:
            reco_energy_valid_string = f"{reco_energy_name}_is_valid_1"
        self.events["reco_energy"] = self.events[reco_energy_string]
        if not self.hess_root:
            self.events["reco_energy_is_valid"] = self.events[reco_energy_valid_string]
        else:
            self.events["reco_energy_is_valid"] = np.repeat(True,len(self.events))
        self.events["rel_E_diff"]=((self.events["reco_energy"]-self.events["true_energy"])/self.events["true_energy"]).to_value(u.one)

    @abstractmethod
    def _set_gh_score_column(self, gh_score_name):
        if gh_score_name is not None:
            gh_score_string = f"{gh_score_name}_prediction"
            gh_score_valid_string = f"{gh_score_name}_is_valid"
        
            self.events["gh_score"] = self.events[gh_score_string]
            self.events["gh_score_is_valid"] = self.events[gh_score_valid_string]

    @abstractmethod
    def _set_geometry_columns(self, geometry_reco_name):
        if self.hess_root:
            self.events["reco_az"] = self.events[f"{geometry_reco_name}AzEvent"]
            self.events["reco_alt"] = self.events[f"{geometry_reco_name}AltEvent"]
        else:
            self.events["reco_az"] = self.events[f"{geometry_reco_name}_az"]
            self.events["reco_alt"] = self.events[f"{geometry_reco_name}_alt"]

        if not self.hess_root:
            if geometry_reco_name != "ImPACTReconstructor":
                self.events["geometry_reco_is_valid"] = self.events[
                    f"{geometry_reco_name}_is_valid"
                ]
            else:
                self.events["geometry_reco_is_valid"] = self.events[
                    f"{geometry_reco_name}_is_valid_1"
                ]
        else:
            self.events["geometry_reco_is_valid"]=np.repeat(True,len(self.events))


        self.events["reco_source_fov_offset"] = angular_separation(
            self.pointing_direction.az,
            self.pointing_direction.alt,
            self.events["reco_az"],
            self.events["reco_alt"],
        )

    def _set_valid_column(self):
        self.events["reco_is_valid"] = np.logical_and(
            self.events["geometry_reco_is_valid"],
            np.logical_and(
                self.events["gh_score_is_valid"], self.events["reco_energy_is_valid"]
            ),
        )

    #@classmethod
    #def from_path(cls, path, reco_energy_name, obs_time=1 * u.s):
    #    loader = TableLoader(
    #        path,
    #        load_true_images=True,
    #        load_dl1_images=False,
    #        load_dl1_parameters=True,
    #        load_simulated=True,
    #        load_dl2=True,
    #    )
    #    return cls( reco_energy_name,loader, obs_time)

    def reweight_to(self, new_spectrum):
        reweight_factors = calculate_event_weights(
            self.events["true_energy"], new_spectrum, self.spectrum
        )
        self.events["weight"] = self.events["weight"] * reweight_factors
        self.events["weight_rate"] = self.events["weight_rate"] * reweight_factors
        self.spectrum = new_spectrum

    def set_obs_time(self, obs_time):
        self.events["weight"] = (self.events["weight_rate"] * obs_time).to_value(u.one)
        self.obs_time = obs_time

    def set_cuts(self, cuts, offset_column=None):

        cuts_mask = np.ones(len(self.events))
        self.events["passes_cuts"] = cuts_mask
        if cuts is not None:
            assert offset_column is not None, "You need to specify an offset column"
            for cut in cuts:
                self.add_cut(cut, offset_column)

    def add_cut(self, cut, offset_column):
        """
        Add a cut to the dataset.
        Parameters
        ----------
        cut : dl2_tools.handler.InterpolatedCut
            The InterpolatedCut to add
        offset_column : str
            Name of the offset column to use to evaluate the cut
        """

        assert (
            cut.cut_column in self.events.keys()
        ), "The chosen cut parameter is not in the events table"

        if cut.bin_columns is not None:
            interp_cols = copy(cut.bin_columns)
            interp_cols.insert(0, offset_column)
        else:
            interp_cols = [offset_column]
        interp_array = []
        for i, col in enumerate(interp_cols):
            if u.Quantity(self.events[col]).unit.is_equivalent(u.TeV):
                interp_array.append(np.log10(self.events[col].to_value(u.TeV)))
            else:
                if i == 0:
                    # This is to assert that the offset column
                    # is interpolated in units of deg
                    interp_array.append(self.events[col].to_value(u.deg))
                else:
                    interp_array.append(self.events[col])

        interp_array = np.array(interp_array)
        if not cut.log_cut:
            cut_values = cut.cut_spline(interp_array.T)
        else:
            cut_values = 10 ** cut.cut_spline(interp_array.T)
        cuts_mask = np.logical_and(
            cut.op(cut.cut_op(self.events[cut.cut_column]), cut_values),
            self.events["passes_cuts"],
        )

        self.events["passes_cuts"] = cuts_mask

    def get_masked_events(self):
        mask = np.logical_and(self.events["passes_cuts"], self.events["reco_is_valid"])
        return self.events[mask]


class DiffuseDataset(SimDataset):
    def __init__(self, path, reco_energy_name,hess_root=False, obs_time=1 * u.s,siminfo_dict=None):
        super().__init__(path, reco_energy_name, hess_root, obs_time,siminfo_dict)
        if not self.hess_root:
            sim_config = QTable(self.loader.read_simulation_configuration())
            obs_info = QTable(self.loader.read_observation_information())
            self._set_simulation_info(sim_config, obs_info)
        else:
            assert siminfo_dict is not None
            self._set_simulation_info_hess_root(path,siminfo_dict)
        self._set_simulation_weights()

    def _set_simulation_info(self, sim_config, obs_info):
        super()._set_simulation_info(sim_config, obs_info)
        # We have to treat diffuse and point-like simulations differently
        # essentially because of how the Aeff per FoV is calcualted for each of them.
        # Also, the distance to the true source is of course different between them.

        assert sim_config["diffuse"][
            0
        ], "A diffuse dataset must be made from diffuse simulation"

        self.simulation_info = SimulatedEventsInfo(
            n_showers=np.sum(sim_config["n_showers"] * sim_config["shower_reuse"]),
            max_impact=sim_config["max_scatter_range"][0],
            viewcone_min=0*u.deg,
            viewcone_max=sim_config["max_viewcone_radius"][0],
            energy_min=sim_config["energy_range_min"][0],
            energy_max=sim_config["energy_range_max"][0],
            spectral_index=sim_config["spectral_index"][0],
        )

        self.source_position = AltAz(
            az=obs_info["subarray_pointing_lon"][0],
            alt=obs_info["subarray_pointing_lat"][0],
        )
        self.radius = self.simulation_info.viewcone_max

    def _set_simulation_info_hess_root(self,path,siminfo_dict):

        super()._set_simulation_info_hess_root(path,siminfo_dict)

        self.radius=siminfo_dict["viewcone_max"]*u.deg
        
        self.source_position = AltAz(
            az=np.max(self.events["true_az"]), alt=np.max(self.events["true_alt"])
        )


class PointSourceDataset(SimDataset):
    def __init__(self, path,reco_energy_name,hess_root=False, obs_time=1 * u.s,siminfo_dict=None):
        super().__init__(path,reco_energy_name,hess_root, obs_time,siminfo_dict)

        if not self.hess_root:
            sim_config = QTable(self.loader.read_simulation_configuration())
            obs_info = QTable(self.loader.read_observation_information())
            self._set_simulation_info(sim_config, obs_info)
        else:
            assert siminfo_dict is not None
            self._set_simulation_info_hess_root(path,siminfo_dict)

        self._set_simulation_weights()

    def _set_simulation_info(self, sim_config, obs_info):
        super()._set_simulation_info(sim_config, obs_info)
        # We have to treat diffuse and point-like simulations differently
        # essentially because of how the Aeff per FoV is calcualted for each of them.
        # Also, the distance to the true source is of course different between them.

        # We have to treat diffuse and point-like simulations differently
        # essentially because of how the Aeff per FoV is calcualted for each of them.
        # Also, the distance to the true source is of course different between them.

        assert not sim_config["diffuse"][
            0
        ], "A point source dataset must be made from point source simulation"

        offsets = (
            angular_separation(
                obs_info["subarray_pointing_lon"],
                obs_info["subarray_pointing_lat"],
                sim_config["max_az"],
                sim_config["max_alt"],
            )
            .to_value(u.deg)
            .round(1)
        )  # If we don't round we get issues with floating precision.
        # Also, if offsets are that close they may as well be the same

        assert (
            len(set(offsets)) == 1
        ), "There are different offsets among the simulations"

        self.offset = offsets[0] * u.deg
        self.source_position = AltAz(
            az=sim_config["max_az"][0], alt=sim_config["max_alt"][0]
        )

        self.simulation_info = SimulatedEventsInfo(
            n_showers=np.sum(sim_config["n_showers"] * sim_config["shower_reuse"]),
            max_impact=sim_config["max_scatter_range"][0],
            viewcone_min=0*u.deg,
            viewcone_max=sim_config["max_viewcone_radius"][0],
            energy_min=sim_config["energy_range_min"][0],
            energy_max=sim_config["energy_range_max"][0],
            spectral_index=sim_config["spectral_index"][0],
        )
    def _set_simulation_info_hess_root(self,path,siminfo_dict):

        super()._set_simulation_info_hess_root(path,siminfo_dict)

        self.offset=siminfo_dict["offset"]*u.deg
        
        self.source_position = AltAz(
            az=np.max(self.events["true_az"]), alt=np.max(self.events["true_alt"])
        )

class SignalSet(SimDataset):
    def __init__(
        self,
        path,
        reco_energy_name,
        geometry_reco_name,
        gh_score_name=None,
        hess_root=False,
        obs_time=1 * u.s,
        siminfo_dict=None
    ):
        super().__init__(path,reco_energy_name,hess_root, obs_time,siminfo_dict)
        if self.hess_root:
            self.events[f"{geometry_reco_name}AzEvent"]=u.Quantity(np.array(self.par_tree[f"{geometry_reco_name}AzEvent"].array()),u.deg)
            self.events[f"{geometry_reco_name}AltEvent"]=u.Quantity(np.array(self.par_tree[f"{geometry_reco_name}AltEvent"].array()),u.deg)
        self._set_gh_score_column(gh_score_name)
        self._set_geometry_columns(geometry_reco_name)
        self._set_valid_column()
        self.set_cuts(None)
    
    def _set_gh_score_column(self, gh_score_name):
        self.events["gh_score"] = np.ones(len(self.events))
        self.events["gh_score_is_valid"] = np.ones(len(self.events)).astype(bool)
        super()._set_gh_score_column(gh_score_name)

    def _set_geometry_columns(self, geometry_reco_name):
        super()._set_geometry_columns(geometry_reco_name)
        self.events["true_source_fov_offset"] = angular_separation(
            self.pointing_direction.az,
            self.pointing_direction.alt,
            self.events["true_az"],
            self.events["true_alt"],
        )

        self.events["theta"] = angular_separation(
            self.events["reco_az"],
            self.events["reco_alt"],
            self.events["true_az"],
            self.events["true_alt"],
        )

    #@classmethod
    #def from_path(
    #    cls, path, reco_energy_name, geometry_name, gh_score_name=None, obs_time=1 * u.s
    #):
    #    loader = TableLoader(
    #        path,
    #        load_true_images=True,
    #        load_dl1_images=False,
    #        load_dl1_parameters=True,
    #        load_simulated=True,
    #        load_dl2=True,
    #    )
    #    return cls(reco_energy_name, geometry_name, gh_score_name,loader, obs_time)

    #@classmethod
    #def from_hess_event_tree(
    #    cls, path, reco_energy_name, geometry_name, gh_score_name=None, obs_time=1 * u.s
    #):
    #    events=QTable()
    #    file=uproot.open(path)
    #    par_tree = file["ParTree_^Postselect;1"]
    #    events["true_energy"]=u.Quantity(np.array(par_tree["MCTrueEnergy"].array()),u.TeV)
    #    events["true_az"]=u.Quantity(np.array(par_tree["MCTrueAzimuth"].array()),u.TeV)
    #    events["true_alt"]=u.Quantity(np.array(par_tree["MCTrueAlt"].array()),u.TeV)
    #    return cls(events,reco_energy_name, geometry_name, gh_score_name,True, obs_time)

class PointSourceSignalSet(SignalSet, PointSourceDataset):
    def __init__(
        self,
        path,
        reco_energy_name,
        geometry_reco_name,
        gh_score_name=None,
        hess_root=False,
        obs_time=1 * u.s,
        siminfo_dict=None
    ):
        super().__init__(
            path, reco_energy_name, geometry_reco_name, gh_score_name,hess_root, obs_time,siminfo_dict
        )


class DiffuseSignalSet(SignalSet, DiffuseDataset):
    def __init__(
        self,
        path,
        reco_energy_name,
        geometry_reco_name,
        gh_score_name=None,
        hess_root=False,
        obs_time=1 * u.s,
        siminfo_dict=None
    ):
        super().__init__(
            path, reco_energy_name, geometry_reco_name, gh_score_name,hess_root, obs_time, siminfo_dict
        )


class BackgroundSet(DiffuseDataset):
    def __init__(
        self,
        path,
        reco_energy_name,
        geometry_reco_name,
        gh_score_name=None,
        hess_root=False,
        obs_time=1 * u.s,
        siminfo_dict=None
    ):
        super().__init__(path,reco_energy_name,hess_root, obs_time,siminfo_dict)
        if self.hess_root:
            self.events[f"{geometry_reco_name}AzEvent"]=u.Quantity(np.array(self.par_tree[f"{geometry_reco_name}AzEvent"].array()),u.deg)
            self.events[f"{geometry_reco_name}AltEvent"]=u.Quantity(np.array(self.par_tree[f"{geometry_reco_name}AltEvent"].array()),u.deg)
        self._set_gh_score_column(gh_score_name)
        self._set_geometry_columns(geometry_reco_name)
        self._set_valid_column()
        self.set_cuts(None)
    
    def _set_gh_score_column(self, gh_score_name):
        self.events["gh_score"] = np.zeros(len(self.events))
        self.events["gh_score_is_valid"] = np.ones(len(self.events)).astype(bool)
        super()._set_gh_score_column(gh_score_name)

    def _set_geometry_columns(self, geometry_reco_name):
        super()._set_geometry_columns(geometry_reco_name)

    #@classmethod
    #def from_path(
    #    cls, path, reco_energy_name, geometry_name, gh_score_name=None, obs_time=1 * u.s
    #):
    #    loader = TableLoader(
    #        path,
    #        load_true_images=True,
    #        load_dl1_images=False,
    #        load_dl1_parameters=True,
    #        load_simulated=True,
    #        load_dl2=True,
    #    )
    #    return cls(reco_energy_name, geometry_name, gh_score_name, loader, obs_time)
