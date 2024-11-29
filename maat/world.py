#!python
"""
#
# @file: world.py
# @brief: MAAT environmnet setup
# ---
# @website: 
# @repo: https://github.com/MLSM-at-DTU/floods_transport_rl
# @author: Miguel Costa, Morten William Petersen.
# MAAT environment
# ---
"""

# General imports
import branca.colormap as cmp
import copy
from enum import Enum
import fiona
import folium
import geopandas as gpd
from haversine import haversine
from ipfn import ipfn
import math
import matplotlib
import networkx as nx
import numpy as np 
import os
import osmnx as ox
import pandas as pd
import pickle
import scipy
import shapely as shp
from shapely.geometry import LineString

# Climada imports
from climada.hazard import Hazard
from climada.entity import Exposures
from climada.entity import ImpactFunc, ImpactFuncSet
from climada.entity.measures import Measure, MeasureSet
from climada.engine import ImpactCalc
import climada.util.lines_polys_handler as u_lp

# Filter out warnings from climada
import logging
logging.getLogger('climada').setLevel(logging.ERROR)

# Reinforcement learning imports
import gymnasium as gym
from gymnasium import spaces

# Custom imports
import maat.transport_utils as transp

# Constants
DATA_LOCATION_LOCAL = "data" # data in this folder does not reflect the original data used, but its rather just used for demonstration purposes
DATA_LOCATION_O_DRIVE = "data" # data in this folder does not reflect the original data used, but its rather just used for demonstration purposes

CRS_25832 = 'epsg:25832'
CRS_4326 = 'epsg:4326'


def _cost_function_haversine(
        zone_i: object, 
        zone_j: object, 
        beta: float=0.01,
    ):
    """Compute cost of going from zone_i to zone_j given its haversine distance."""

    return math.exp(-beta * haversine(zone_i['centroid_4326'], zone_j['centroid_4326'], unit='km'))


class HazardSampling(Enum):
    """
    Hazard (rainfall event) sampling method.

    IPCC : sampling follows a IPCC sampling distribution # TODO: Implement this
    DET : deterministic sampling, where the same hazard is used for all episodes
    RANDOM : random sampling of hazards
    """
    KLIMAATLAS = "klimaatlas"
    DETERMINISTIC = "deterministic"
    RANDOM = "random"
    CUSTOM = "custom"


class BasicEnvironment(gym.Env):
    
    metadata = {
        "render_modes": ["human", "ansi"],
        }

    # ==================================================================== #
    # INIT
    # ==================================================================== #
    def __init__(
            self: object,
            render_mode="ansi",
            assets_types=["network"],
            preload_hazards: bool=True,
            hazards_to_preload: list=[160],
            hazard_sampling_scheme: HazardSampling=HazardSampling.DETERMINISTIC,
            episode_time_steps: int=3,
            city_zones: str="Copenhagen",
            available_actions: list=[1] # ["_ElevateRoad1m", "_ElevateRoad2m", "_Resist25%", "_Resist50%"]
        ):
        
        # Define maximum episode step size
        self.max_time_steps = episode_time_steps

        # Step tracking
        self._observation_year = 0

        # Rainfall tracking
        self._observation_rain = 0

        # Action tracking
        self.action_taken = 0
        self.previous_actions = []
        
        # Rewards
        self.reward = 0
        self.cumulative_reward = 0
        self.reward_monetary_over_time = []

        # Reward function coeficients
        self.beta_damage = 1.
        self.beta_delay = 1.
        self.beta_notravel = 0.
        self.beta_action_cost = 1.

        # Reward function components
        self.impacts_direct_damage = 0
        self.impacts_delay = 0
        self.impacts_no_travel = 0
        self.impacts_action_cost = 0

        # Define environment geographic limits
        self._define_area_limits(area="Copenhagen")
        assert self.area_limits is not None, "Area limits not defined."
        
        # Define assets types to consider
        self.assets_types = assets_types
        if "network" in self.assets_types:
            self.assets_types += ["network_dis"]
        assert len(self.assets_types) > 0, "No assets types defined."

        # Define TAZ (Traffic Assignment Zones)
        self.city_zones = city_zones
        if city_zones == "Copenhagen":
            raise ValueError("Full Copenhagen not implemented yet. Choose another city area to run world.")
        elif city_zones == "IndreBy":
            self._zone_ids = [102122, 102222, 102221, 102224, 102142, 103131, 103141, 103143, 103142, 103144, 102171, 102121, 102231, 102111, 102162, 102151, 102141, 102223, 102131, 102172, 102351, 102152, 102213, 102211, 102212, 102214, 102161, 103132, 103133]
        elif city_zones == "IndreNord":
            self._zone_ids = [102122, 102325, 102222, 102422, 102433, 102342, 102221, 102224,
                              102142, 103131, 102333, 103141, 103143, 102183, 102441, 102321,
                              103142, 103144, 102413, 102352, 102412, 102311, 102171, 102121,
                              102231, 102111, 102162, 102324, 102151, 102141, 102223, 102455,
                              102131, 102172, 102323, 102322, 102411, 102343, 102421, 102423,
                              102442, 102445, 102432, 102344, 102341, 102444, 102443, 102452,
                              102453, 102351, 102152, 102332, 102213, 102211, 102212, # 102312
                              102181, 102182, 102184, 102451, 102431, 102434, 102331, 102336,
                              102414, 102353, 102214, 102161, 103132, 103133, 102334, 102335,
                              102454, 102337]
        elif city_zones == "IndreNordFredAma":
            self._zone_ids = [103121, 147131, 102122, 102325, 102222, 102731, 147242, 147111,
                              147232, 147223, 147141, 147121, 147213, 102422, 102433, 102342,
                              102221, 102224, 102142, 103111, 103131, 102333, 103141, 103143,
                              103172, 103174, 103181, 103281, 147122, 147151, 147152, 147233,
                              102183, 102441, 102321, 103161, 147142, 103243, 147251, 147252,
                              103142, 103144, 103231, 102413, 103212, 103214, 102741, 102722,
                              102821, 103253, 103251, 103221, 103151, 102851, 102352, 102412,
                              102311, 102711, 102841, 102831, 102725, 102732, 102171, 102121,
                              102231, 102111, 102162, 103244, 103262, 103241, 103252, 103223,
                              103232, 102324, 102811, 102151, 147112, 102141, 102223, 102455,
                              103211, 102131, 102172, 103193, 147221, 103191, 103192, 102712,
                              102323, 102322, 102411, 102343, 102421, 102423, 102442, 102445,
                              147222, 147212, 102432, 102344, 102341, 102444, 102443, 102452,
                              147161, 147234, 102453, 102351, 102152, 103182, 103282, 147224,
                              103112, 147133, 147231, 147162, 103152, 103272, 147211, 102332,
                              102213, 102211, 102212, 103173, 147132, 147241, 103162, 103271,
                              103292, 103291, 102724, 102721, 102723, 102822, 102181, 102812,
                              102182, 102184, 147114, 147113, 102871, 102451, 102431, 102434,
                              102331, 102336, 102414, 102353, 103213, 103224, 103242, 103261,
                              103222, 103233, 102214, 102161, 103132, 103133, 103171, 102334,
                              102335, 102454, 102852, 102337]
        else:
            raise ValueError("City zones not defined.")
        
        # Filter corresponding TAZ
        self.taz = self._define_taz(filter_taz=self._zone_ids)
        assert len(self.taz) > 0, "TAZ not defined."

        # Define transport network
        self.transport_network = self.define_transport_network(self.taz)
        assert self.transport_network is not None, "Transport network not defined."
        
        # Define demand and supply
        self.define_demand_supply()

        # Compute non-affected trip distribution and routing
        self.trip_distribution_original = self.distribute_trips(cost_function=_cost_function_haversine, beta=0.5)
        self.network_volume_original, self.trip_length_original = self.route_trips(self.trip_distribution_original)

        # Initialize damage per zone
        self._observation_impacts_direct_damage = np.zeros(self.taz.shape[0])
        self._observation_impacts_travel_delays = np.zeros(self.taz.shape[0])
        self._observation_impacts_no_travel = np.zeros(self.taz.shape[0])
        self._observation_impacts_action_cost = np.zeros(self.taz.shape[0])
        self.taz["impacts_direct_damage"] = self._observation_impacts_direct_damage
        self.taz["impacts_travel_delays"] = self._observation_impacts_travel_delays
        self.taz["impacts_no_travel"] = self._observation_impacts_no_travel
        self.taz["impacts_action_cost"] = self._observation_impacts_action_cost

        # Initialize hazards and a single hazard event
        self.hazard_sampling_scheme = hazard_sampling_scheme

        self.preload_hazards = preload_hazards
        if self.preload_hazards:
            self.hazard_set = self.initialize_hazards(hazards_to_load=hazards_to_preload, file_type="hdf5")
        self.hazard = self.hazard_set[160] if self.preload_hazards else self.sample_hazard(rain_event=160, sampling_scheme=self.hazard_sampling_scheme)

        # Assets
        self.exposures = self.get_assets_exposures(self.hazard, exposures_only=True)

        # Impact functions
        self.original_impact_function_ids = [1, 3, 5, 7]
        self.impact_functions = self.define_impact_functions()
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1                                          # year
                #    + 1                                        # action
                   + 1                                        # rain
                   + self.exposures["network"].gdf.shape[0] # mdr
                   + self.transport_network.number_of_nodes() # network_effective_water_depth
                   + self.transport_network.number_of_edges() # network_travel_time_impacted
                   + self.transport_network.number_of_edges() # network_volume
                   + self.taz.shape[0]                        # travel delays
                   ,),
            dtype=np.float64
            )
        
        # Action space
        self.available_action_types = available_actions
        self.total_available_actions = (self.taz.shape[0]*len(self.available_action_types)) + 1 # Assuming action_size is number of zones per types of actions + 1 (do nothing)
        self.measure_set = self.create_measure_set()
        self.action_space = spaces.Discrete(self.total_available_actions)  
 
        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    # ==================================================================== #
    # ASSETS
    # ==================================================================== #
    def _get_network_assets(
            self: object,
            network_type: str="drive",
        ):
        """Get network assets."""
        
        def _add_traffic_signals_to_edges(G_edges):
            """Add traffic signals to edges."""

            G_edges["traffic_signals"] = np.nan

            for (u,v,key), row in G_edges.iterrows():
                if G_nodes.loc[u]["highway"] == "traffic_signals" or G_nodes.loc[v]["highway"] == "traffic_signals":
                    G_edges.loc[(u,v,key), "traffic_signals"] = "yes"
                else:
                    G_edges.loc[(u,v,key), "traffic_signals"] = np.nan
            
            return G_edges
        
        def _add_impact_functions(G_edges):
            """Add impact functions to edges."""

            # Put all highway attributes in the same format (list)
            G_edges["highway"] = [highway if isinstance(highway, list) else [highway] for highway in G_edges["highway"]]

            # Add default impact function
            G_edges["impf_RainEvent"] = 7 # HZ
            G_edges["original_impf_RainEvent"] = 7 # HZ

            # Iterate over edges and add impact functions
            for (u,v,key), row in G_edges.iterrows():
                motorways_and_trunks = ["motorway", "motorway_link", "motorway_junction", "trunk", "trunk_link"]
                other_roads = ["primary", "primary_link", 
                            "secondary", "secondary_link", 
                            "tertiary", "tertiary_link", 
                            "unclassified", "residential", "living_street", "service", "pedestrian", "bus_guideway", "escape", "raceway", "road", "cycleway", "construction", "bus_stop", "crossing", "mini_roundabout", "passing_place", "rest_area", "turning_circle", "traffic_island", "yes", "emergency_bay"]

                # Check if the edge is a motorway or trunk
                if any([highway in motorways_and_trunks for highway in row["highway"]]):
                    # Check if the edge has traffic signals
                    if row["traffic_signals"] == "yes":
                        G_edges.loc[(u,v,key), "impf_RainEvent"] = 1 # C1
                        G_edges.loc[(u,v,key), "original_impf_RainEvent"] = 1 # C1
                   
                    # Check if the edge has no traffic signals
                    else:
                        G_edges.loc[(u,v,key), "impf_RainEvent"] = 3 # C3
                        G_edges.loc[(u,v,key), "original_impf_RainEvent"] = 3 # C3

                # Check if the edge is another type of road
                elif any([highway in other_roads for highway in row["highway"]]):
                    G_edges.loc[(u,v,key), "impf_RainEvent"] = 5 # C5
                    G_edges.loc[(u,v,key), "original_impf_RainEvent"] = 5 # C5

                # If unspecified, use the default impact function
                else:
                    pass
            
            return G_edges

        def _add_value_to_edges(
                G_edges: pd.DataFrame,
                euro_to_dkk: float=7.46,
                gdp_dk_2015: float=45500,
                gdp_dk_2023: float=52510,
                gdp_eu_2015: float=26760,
                value_highway_type: pd.DataFrame=pd.DataFrame({"highway": ["motorway", "trunk", "primary", "secondary", "tertiary", "other"], "value": [(35-3.5)/2 * 1e6, (7.5-2.5)/2 * 1e6, (3.0-1.0)/2 * 1e6, (1.5-0.5)/2 * 1e6, (.6-0.2)/2 * 1e6, (0.3-0.1)/2 * 1e6],}).set_index("highway", drop=True),
                factors_lanes: pd.DataFrame=pd.DataFrame({"highway": ["motorway", "trunk", "primary", "secondary", "tertiary", "other"],"1_lane": [.75, .75, .75, .75, .75, 1],"2_lane": [1., 1., 1., 1., 1., 1.25],"3_lane": [1.25, 1.25, 1.25, 1.25, 1.5, 1.5],"4_lane": [1.5, 1.5, 1.5, 1.5, 1.75, 1.75],"5_lane": [1.75, 1.75, 1.75, 1.75, 2., 2.],"6_lane": [2., 2., 2., 2., 2.25, 2.25]}).set_index("highway", drop=True),
                factor_signaling_motorway: float=1.22,
                factor_signaling_trunk: float=1.28,
                ):
            """
            Add value to edges. The value is calculated based on the length of the road and the type of road.
            
            Parameters:
            euro_to_dkk: 1 euro are equivalent to 7.46 dkk
            gdp_dk_2015: GDP per capita in Denmark in 2015, from https://ec.europa.eu/eurostat/databrowser/view/sdg_08_10/default/table?lang=en
            gdp_dk_2023: GDP per capita in Denmark in 2023, from https://ec.europa.eu/eurostat/databrowser/view/sdg_08_10/default/table?lang=en
            gdp_eu_2015: GDP per capita in EU28 in 2015, from https://ec.europa.eu/eurostat/databrowser/view/sdg_08_10/default/table?lang=en
            value_highway_type: Value per kilometer of road per type of road, from https://nhess.copernicus.org/articles/21/1011/2021/nhess-21-1011-2021-discussion.html
            factors_lanes: Factors for the number of lanes, from https://nhess.copernicus.org/articles/21/1011/2021/nhess-21-1011-2021-discussion.html
            factor_signaling_motorway: Factor for traffic signaling in motorways, from https://nhess.copernicus.org/articles/21/1011/2021/nhess-21-1011-2021-discussion.html
            factor_signaling_trunk: Factor for traffic signaling in trunks, from https://nhess.copernicus.org/articles/21/1011/2021/nhess-21-1011-2021-discussion.html
            """

            # Add default value to edges
            G_edges["value"] = 0

            # Set the number of lanes
            G_edges["lanes"] = [lanes if isinstance(lanes, list) else [lanes] for lanes in G_edges["lanes"]]
            G_edges["lanes"] = G_edges["lanes"].apply(lambda x: max(x))

            # Iterate over edges and add value
            for (u,v,key), row in G_edges.iterrows():
                motorways = ["motorway", "motorway_link", "motorway_junction"]
                trunk = ["trunk", "trunk_link"]
                primary = ["primary", "primary_link"]
                secondary = ["secondary", "secondary_link"]
                tertiary = ["tertiary", "tertiary_link"]
                # other_roads = ["unclassified", "residential", "living_street", "service", "pedestrian", "bus_guideway", "escape", "raceway", "road", "cycleway", "construction", "bus_stop", "crossing", "mini_roundabout", "passing_place", "rest_area", "turning_circle", "traffic_island", "yes", "emergency_bay"]

                # Check if the road is a motorway, trunk, primary, secondary, terciary or other
                if any([highway in motorways for highway in row["highway"]]):
                    highway_type = "motorway"
                
                elif any([highway in trunk for highway in row["highway"]]):
                    highway_type = "trunk"

                elif any([highway in primary for highway in row["highway"]]):
                    highway_type = "primary"
                
                elif any([highway in secondary for highway in row["highway"]]):
                    highway_type = "secondary"

                elif any([highway in tertiary for highway in row["highway"]]):
                    highway_type = "tertiary"

                else:
                    highway_type = "other"

                # Get the length of the road
                length = row["length"] / 1e3 # convert to km

                # Get the number of lanes
                lanes = row["lanes"] if not pd.isna(row["lanes"]) else 2 # get number of lanes, if not specified, assume 2 lanes

                # Get the lane value factor
                factor_lane = factors_lanes.loc[highway_type, f"{lanes}_lane"]

                # Get the value per type of road
                value = value_highway_type.loc[highway_type, "value"] * length

                # Factor if lanes
                value = value * factor_lane
                
                # Factor if signaling
                if highway_type == "motorway":
                    value = value * factor_signaling_motorway
                elif highway_type == "trunk":
                    value = value * factor_signaling_trunk

                # Value in DK in 2023
                value = ( value / (gdp_dk_2015/gdp_dk_2023) / (gdp_eu_2015/gdp_dk_2015) ) * euro_to_dkk

                # Set the value
                G_edges.loc[(u,v,key), "value"] = value

            return G_edges

        if network_type == "drive":
            # Get network graph
            G = ox.graph_from_polygon(self.area_limits, network_type='drive', retain_all=True)

            # Add speed to graph
            G = transp.add_speed_to_graph(G, transp.TransportMode.CAR)

            # Add travel times to graph
            G = ox.add_edge_travel_times(G)

            # Get graph nodes and edges
            G_nodes, G_edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

            # Project to EPSG:25832
            G_edges = G_edges.to_crs(CRS_25832)
            G_nodes = G_nodes.to_crs(CRS_25832)

            # Add traffic signals to edges if they exist i adjacent nodes
            G_edges = _add_traffic_signals_to_edges(G_edges)

            # Add impact function to edges
            G_edges = _add_impact_functions(G_edges)
            
            # Add value to edges
            G_edges["length"] = G_edges.length
            G_edges = _add_value_to_edges(G_edges)
            
            # Add region id to edges
            G_edges = gpd.sjoin(G_edges, self.taz[['zoneid', 'geometry']], how='inner', predicate='intersects')
            G_edges.rename(columns={'zoneid': 'region_id'}, inplace=True)
            G_edges['region_id'] = G_edges['region_id'].fillna(-1).astype(int)
            G_edges = G_edges[G_edges['region_id'] != -1]

            # Add water depth column
            G_edges["water_depth"] = 0.
            G_edges["mdr"] = 0.

            # Create exposure object
            G_edges = G_edges.reset_index()
            G_edges_exposure = Exposures(G_edges)

            return G_edges_exposure, G_edges, G_nodes
        
        else:
            raise ValueError("Network type not defined.")

        return 
    
    def _get_building_assets(
            self: object,
            impact_function: int=1,
        ):
        """Get buildings assets."""
        # Get buildings as assets
        buildings = gpd.read_file(os.path.join(DATA_LOCATION_O_DRIVE, "data", "assets", "bygninger.geojson")).to_crs(CRS_25832)
        prices = pd.read_csv(os.path.join(DATA_LOCATION_O_DRIVE, "data", "assets", "kvm_pris.csv"), 
                             header=None, names=["komkode","municipality", "kvm_pris"])

        # compute value for buildings and add exposure needed variables
        buildings = buildings.merge(prices, on='komkode', how='left')
        buildings['area'] = buildings.area
        buildings['value'] = buildings['kvm_pris'] * buildings['area']
        buildings["impf_RainEvent"] = impact_function

        buildings["latitude"] = buildings.apply(lambda row: row['geometry'].centroid.y, axis=1)
        buildings["longitude"] = buildings.apply(lambda row: row['geometry'].centroid.x, axis=1)

        # Perform spatial join
        buildings = gpd.sjoin(buildings, self.taz[['zoneid', 'geometry']], how='left', predicate='within')

        # Rename the column
        buildings.rename(columns={'zoneid': 'region_id'}, inplace=True)
        buildings['region_id'] = buildings['region_id'].fillna(-1).astype(int)
        buildings = buildings[buildings['region_id'] != -1]

        # Filter assets based on taz_zoneid column
        buildings = buildings[buildings['region_id'].isin(list(self.taz['zoneid']))]

        # Create exposure object
        buildings_exposures = Exposures(buildings)

        return buildings_exposures, buildings
        
    def disagregate_exposures(
            self: object,
            exposures: Exposures,
            hazard: Hazard,
            resolution: int=10,
            to_meters: bool=False):
        
        # Disaggregate exposures
        exp_pnt = u_lp.exp_geom_to_pnt(
            exp=exposures, 
            res=resolution,
            to_meters=to_meters,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None
            )
        exp_pnt.assign_centroids(hazard)

        return exp_pnt

    def get_assets_exposures(
            self: object,
            hazard: Hazard,
            res: float=10,
            to_meters: bool=True,
            exposures_only: bool=True,
        ):
        """Get assets and exposures."""
        
        # Initialize assets
        assets = {}

        if "network" in self.assets_types:
            exposures, edges, nodes = self._get_network_assets()
            if exposures_only:
                assets["network"] = exposures

            else:
                assets["network"] = (exposures, edges, nodes)

            # Disagregate exposures
            assets["network_dis"] = self.disagregate_exposures(exposures, hazard, res, to_meters)

        if "buildings" in self.assets_types:
            exposures, buildings = self._get_building_assets()
            
            if exposures_only:
                assets["buildings"] = exposures
            else:
                assets["buildings"] = (exposures, buildings)

        assert len(assets) > 0, "No assets found."

        return assets

    # ==================================================================== #
    # HAZARD
    # ==================================================================== #
    def initialize_hazards(
            self: object,
            hazards_to_load: list=[20, 80, 160],
            file_type: str="hdf5",
            ):
        """Initialize hazards by preloading them to memory."""
        hazard_set = {}

        if len(hazards_to_load) <= 0:
            raise ValueError("No hazards to load.")

        # Iterate over hazards to load
        for rain_event in hazards_to_load:
            # Load hazard from file
            hazard_set[rain_event] = self._load_hazard_from_file(rain_event=rain_event, file_type=file_type)
    
        return hazard_set
    
    def _load_hazard_from_file(
            self: object,
            rain_event: int=160,
            file_type: str="hdf5",
        ):
        """Load hazard from file."""

        if file_type == "pickle":
            # hazard_file = os.path.join(DATA_LOCATION_O_DRIVE, "data", "rain", "pickle", "Terrain_Buildings_København_V_Water_depth_Rain={}_mm.pkl".format(rain_event))
            hazard_file = os.path.join(DATA_LOCATION_LOCAL, "rain", "pickle", "Terrain_Buildings_København_V_Water_depth_Rain={}_mm.pkl".format(rain_event))
            hazard_event = pickle.load(open(hazard_file, 'rb'))
            hazard_event.intensity = hazard_event.intensity.tocsc()
        elif file_type == "hdf5":
            # hazard_file = os.path.join(DATA_LOCATION_O_DRIVE, "data", "rain", "hdf5", "{}mm.hdf5".format(rain_event))
            hazard_file = os.path.join(DATA_LOCATION_LOCAL, "rain", "hdf5", "{}mm.hdf5".format(rain_event))
            hazard_event = Hazard().from_hdf5(hazard_file)
            hazard_event.intensity = hazard_event.intensity.tocsc()
        
        return hazard_event

    def sample_rainfall_event_klimaatlas(
            self: object,):
        """Sample a rainfall event from the klimaatlas data."""

        # Define the return periods and corresponding rainfall amounts
        return_periods = np.array([1, 2, 5, 10, 20, 50, 100])
        rainfall_amounts_2011_2040 = np.array([35.15, 43.37, 55.67, 65.85, 77.15, 93.72,  107.50])
        rainfall_amounts_2041_2070 = np.array([37.39, 45.05, 57.26, 67.63, 78.89, 95.59,  109.30])
        rainfall_amounts_2071_2100 = np.array([40.65, 48.72, 62.17, 73.64, 86.23, 104.56, 119.95])

        # Calculate the exceedance probabilities
        exceedance_probabilities = 1 / return_periods

        # Check current year and select the corresponding rainfall amounts
        if self.observation_year >= 0 and self.observation_year <= 18: # 2011 - 2040
            rainfall_amounts = rainfall_amounts_2011_2040
        elif self.observation_year >= 19 and self.observation_year <= 48: # 2041 - 2070
            rainfall_amounts = rainfall_amounts_2041_2070
        elif self.observation_year >= 48 and self.observation_year <= 77: # 2071 - 2100
            rainfall_amounts = rainfall_amounts_2071_2100
        else:
            raise ValueError("Year out of range defined.")

        # Create the cumulative distribution function (CDF)
        cdf = 1 - exceedance_probabilities

        # Generate a uniform random number
        u = np.random.uniform(0, 1)

        # Inverse CDF Sampling - Find the rainfall amount corresponding to the random number
        rainfall_sampled = np.interp(u, cdf, rainfall_amounts)

        # Round the rainfall amount to the nearest 4 mm since thats how we got the data
        sampled_rainfall_rounded = round(rainfall_sampled / 4) * 4

        return sampled_rainfall_rounded

    def sample_hazard(
            self: object,
            sampling_scheme: HazardSampling=HazardSampling.DETERMINISTIC,
            rain_event: int=None,
            file_type: str="pickle",
        ):
        """Sample a rainfall event (hazard)."""

        # Rainfall event provided
        if sampling_scheme is HazardSampling.DETERMINISTIC:
            # use given rain event
            rain_event = rain_event

        # Random sampling
        elif sampling_scheme is HazardSampling.RANDOM:
            # Sample a random rainfall event
            rain_event = np.random.choice([20, 80, 160])        

        # IPCC sampling
        elif sampling_scheme is HazardSampling.KLIMAATLAS:
            rain_event = self.sample_rainfall_event_klimaatlas()
            
        # Custom sampling
        elif sampling_scheme is HazardSampling.CUSTOM:
            # Define the probabilities for each rainfall event
            probabilities = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]

            # Sample a rainfall event based on the probabilities
            rain_event = np.random.choice([0, 4, 8, 20, 80, 160], p=probabilities)

        else:
            raise ValueError("Sampling scheme not defined.")

        # If hazards were preloaded
        if self.preload_hazards:
            # Check if the hazard event was preloaded
            if rain_event in self.hazard_set.keys():
                hazard_event = self.hazard_set[rain_event]
                
                # Set observation rain
                self._observation_rain = rain_event
                return hazard_event
        
        # Load the hazard event corresponding to the sampled rainfall event
        hazard_event = self._load_hazard_from_file(rain_event=rain_event, file_type=file_type)

        # Set observation rain
        self._observation_rain = rain_event

        return hazard_event

    # ==================================================================== #
    # IMPACT FUNCTIONS
    # ==================================================================== #
    def define_impact_functions(
            self: object,
        ):
        """Define impact functions."""
        
        funcs = [] # placeholder for impact functions

        # Impact functions from "Flood risk assessment of the European road network"
        # Available on: https://nhess.copernicus.org/articles/21/1011/2021/nhess-21-1011-2021-discussion.html
        # We assume low flow all the time since it is a rain event, so we consider the C1, C3, C5 and HZ impact functions only
        # C1
        A = np.array([[0.0, 0.000],
                      [0.5, 0.010],
                      [1.0, 0.030],
                      [1.5, 0.075],
                      [2.0, 0.100],
                      [6.0, 0.200],
                      [10.0, 0.30]],)

        intensity_unit = "m"
        haz_type="RainEvent"
        id = 1
        name = "RainEvent Damage Function C1"
        intensity = A[:, 0]
        mdd = A[:, 1]
        paa = np.full(7, 1)

        C1 = ImpactFunc(
            id=id,                         # ID of the impact function
            name=name,                     # Name of the impact function
            intensity_unit=intensity_unit, # Intensity unit
            haz_type=haz_type,             # Hazard type
            intensity=intensity,           # Intensity values      
            mdd=mdd,                       # Mean damage (impact) degree
            paa=paa,                       # Percentage of affected assets
        )
        funcs.append(C1)

        # C3
        A = np.array([[0.0, 0.000],
                      [0.5, 0.002],
                      [1.0, 0.004],
                      [1.5, 0.025],
                      [2.0, 0.030],
                      [6.0, 0.040],
                      [10.0, 0.050]],)

        intensity_unit = "m"
        haz_type="RainEvent"
        id = 3
        name = "RainEvent Damage Function C3"
        intensity = A[:, 0]
        mdd = A[:, 1]
        paa = np.full(7, 1)

        C3 = ImpactFunc(
            id=id,                         # ID of the impact function
            name=name,                     # Name of the impact function
            intensity_unit=intensity_unit, # Intensity unit
            haz_type=haz_type,             # Hazard type
            intensity=intensity,           # Intensity values      
            mdd=mdd,                       # Mean damage (impact) degree
            paa=paa,                       # Percentage of affected assets
        )
        funcs.append(C3)

        # C5
        A = np.array([[0.0, 0.000],
                      [0.5, 0.015],
                      [1.0, 0.025],
                      [2.0, 0.035],
                      [6.0, 0.050],
                      [10.0, 0.065]],)

        intensity_unit = "m"
        haz_type="RainEvent"
        id = 5
        name = "RainEvent Damage Function C5"
        intensity = A[:, 0]
        mdd = A[:, 1]
        paa = np.full(6, 1)

        C5 = ImpactFunc(
            id=id,                         # ID of the impact function
            name=name,                     # Name of the impact function
            intensity_unit=intensity_unit, # Intensity unit
            haz_type=haz_type,             # Hazard type
            intensity=intensity,           # Intensity values      
            mdd=mdd,                       # Mean damage (impact) degree
            paa=paa,                       # Percentage of affected assets
        )
        funcs.append(C5)

        # HZ
        A = np.array([[0.0, 0.00],
                      [0.5, 0.25],
                      [1.0, 0.42],
                      [1.5, 0.55],
                      [2.0, 0.65],
                      [3.0, 0.80],
                      [4.0, 0.90],
                      [5.0, 1.00],
                      [6.0, 1.00],
                      [10.0, 1.00]],)

        intensity_unit = "m"
        haz_type="RainEvent"
        id = 7
        name = "RainEvent Damage Function HZ"
        intensity = A[:, 0]
        mdd = A[:, 1]
        paa = np.full(10, 1)

        HZ = ImpactFunc(
            id=id,                         # ID of the impact function
            name=name,                     # Name of the impact function
            intensity_unit=intensity_unit, # Intensity unit
            haz_type=haz_type,             # Hazard type
            intensity=intensity,           # Intensity values      
            mdd=mdd,                       # Mean damage (impact) degree
            paa=paa,                       # Percentage of affected assets
        )
        funcs.append(HZ)

        # Combine all functions
        impact_function_set = ImpactFuncSet(funcs)
        impact_function_set.check()

        def _add_impact_functions_from_actions(
                impact_function_set: ImpactFuncSet
                ):
            """Add impact functions related to possible actions."""

            original_funcs = self.original_impact_function_ids

            # Elevate road by 1m 
            action_measure_1 = 1
            for id in original_funcs:
                new_func = impact_function_set.get_func("RainEvent", id)
                new_func = copy.deepcopy(new_func)
                new_func.id = int("{}{}".format(id, action_measure_1))
                new_func.name = new_func.name + "_ElevateRoad1m"
                new_func.intensity = new_func.intensity + 1 # Elevate by 1m
                
                impact_function_set.append(new_func)

            # Elevate road by 2m
            action_measure_2 = 2
            for id in original_funcs:
                new_func = impact_function_set.get_func("RainEvent", id)
                new_func = copy.deepcopy(new_func)
                new_func.id = int("{}{}".format(id, action_measure_2))
                new_func.name = new_func.name + "_ElevateRoad2m"
                new_func.intensity = new_func.intensity + 2 # Elevate by 2m
                
                impact_function_set.append(new_func)

            # Increase resistance by 25%
            action_measure_3 = 3
            for id in original_funcs:
                new_func = impact_function_set.get_func("RainEvent", id)
                new_func = copy.deepcopy(new_func)
                new_func.id = int("{}{}".format(id, action_measure_3))
                new_func.name = new_func.name + "_Resist25%"
                new_func.mdd = new_func.mdd * 0.75 # Increase resitance by 25%
                
                impact_function_set.append(new_func)

            # Increase resistance by 50%
            action_measure_4 = 4
            for id in original_funcs:
                new_func = impact_function_set.get_func("RainEvent", id)
                new_func = copy.deepcopy(new_func)
                new_func.id = int("{}{}".format(id, action_measure_4))
                new_func.name = new_func.name + "_Resist50%"
                new_func.mdd = new_func.mdd * 0.50 # Increase resitance by 25%
                
                impact_function_set.append(new_func)

            return impact_function_set


        impact_function_set = _add_impact_functions_from_actions(impact_function_set)

        return impact_function_set

    # ==================================================================== #
    # Transport Model
    # ==================================================================== #
    def _define_area_limits(
            self: object,
            area: str="Copenhagen"):
        """Define area limits."""

        if area == "Copenhagen":
            # Read data on the municipality limits
            kommunes = gpd.read_file(os.path.join(DATA_LOCATION_LOCAL, "kommunes", "kommunes.shp")).to_crs('epsg:4326')

            area = shp.ops.unary_union([kommunes.iloc[0].geometry, # Copenhagen
                                        kommunes.iloc[1].geometry] # Frederiksberg
                                        )
            # Set limits
            self.area_limits = area

        else:
            raise ValueError("Area not defined.")

    def _define_taz(
            self: object,
            filter_taz=None):
        """Define Traffic Assignment Zones (TAZ)."""

        # Read TAZ data
        taz = pd.read_pickle(os.path.join("data", "taz", "taz.pkl"))

        # Compute TAZ centroids
        taz['centroid'] = taz.apply(lambda row: (row['geometry'].centroid.x, row['geometry'].centroid.y), axis=1)
        taz['centroid_4326'] = taz.to_crs(CRS_4326).apply(lambda row: (row['geometry'].centroid.x, row['geometry'].centroid.y), axis=1)

        # List of IDs to filter
        if filter_taz is not None:
            assert len(filter_taz) > 0, "No TAZ IDs to filter."

            taz = taz[taz['zoneid'].isin(self._zone_ids)]
    
        return taz
    
    def define_transport_network(
            self: object,
            taz: gpd.GeoDataFrame,
            speed: float=50,
        ):
        """Define transport network."""
        
        def _get_ratio_edge_in_polygons(
                G: nx.Graph,
                zone1: int, 
                zone2: int):
            """Get the ratio of the edge in the polygons."""

            # Get the position of nodes
            pos_u = G.nodes[zone1]['centroid']
            pos_v = G.nodes[zone2]['centroid']

            # Create a LineString from these positions
            line = LineString([pos_u, pos_v])

            # Get the two polygons
            polygon1 = self.taz[self.taz["zoneid"].isin([zone1])].iloc[0].geometry
            polygon2 = self.taz[self.taz["zoneid"].isin([zone2])].iloc[0].geometry

            # Determine the intersection of the edge with each polygon
            intersection1 = line.intersection(polygon1)
            intersection2 = line.intersection(polygon2)

            # Total length of the edge
            total_length = line.length

            # Calculate lengths of the intersections
            ratio1 = intersection1.length / total_length if not intersection1.is_empty else 0
            ratio2 = intersection2.length / total_length if not intersection2.is_empty else 0

            if ratio1 == 0 and ratio2 == 0:
                # no intersection, assume 50% of the edge is in each polygon    
                ratio1, ratio2 = .5, .5

            elif ratio1 == 0:
                ratio1, ratio2 = 1-ratio2, ratio2

            elif ratio2 == 0:
                ratio1, ratio2 = ratio1, 1-ratio1

            else:
                ratio1, ratio2 = ratio1, ratio2

            # Output the ratios
            return ratio1, ratio2

        # Initialize a transport network graph
        G = nx.Graph()

        # Add nodes
        for _, zone in taz.iterrows():
            G.add_node(zone["zoneid"], 
                       centroid=zone["centroid"], 
                       centroid_4326=zone["centroid_4326"], 
                       zonenavn=zone["zonenavn"],
                       effective_water_depth=0.,)

        # Add edges
        for _, zone1 in taz.iterrows():
            for _, zone2 in taz.iterrows():
                if zone1['geometry'].touches(zone2['geometry']):
                    distance = haversine(zone1['centroid_4326'], zone2['centroid_4326'], unit='km')
                    
                    # Get the ratio of the edge in each of the TAZ polygons
                    ratio1, ratio2 = _get_ratio_edge_in_polygons(G, zone1["zoneid"], zone2["zoneid"])

                    # calculate edge travel time in seconds 
                    speed_km_sec = speed / (60 * 60)

                    # travel time in zone 1
                    distance_km_1 = distance * ratio1
                    travel_time_1 = distance_km_1 / speed_km_sec

                    # travel time in zone 2
                    distance_km_2 = distance * ratio2
                    travel_time_2 = distance_km_2 / speed_km_sec

                    # Add edge to graph
                    G.add_edge(zone1["zoneid"], 
                               zone2["zoneid"], 
                               distance=distance,
                               travel_time=travel_time_1 + travel_time_2,
                               travel_time_impacted=travel_time_1 + travel_time_2,
                               mdr=None,
                               water_depth=None,
                               capacity=0.0,
                               volume=0.0
                               )
                    G[zone1["zoneid"]][zone2["zoneid"]].update(
                        {zone1["zoneid"]: ratio1, 
                         zone2["zoneid"]: ratio2}) 
                                       
        # Add non touching zones manually
        a = [102111, 102111,102111,102111,102111,102111,103143, 103143,103131,103132,103144, 102331, 102171, 102871, 103121]
        b = [103144, 102224,102121,102122,102131,102171,103131,103144,103132,103133, 103142, 102337, 103192, 103272, 103152]
        for i, j in zip(a, b):
            try:
                zone1, zone2 = taz[taz['zoneid'] == i].iloc[0], taz[taz['zoneid'] == j].iloc[0]

                # Get the ratio of the edge in each of the TAZ polygons
                ratio1, ratio2 = _get_ratio_edge_in_polygons(G, zone1["zoneid"], zone2["zoneid"])

                # calculate edge travel time in seconds 
                speed_km_sec = speed / (60 * 60)

                # travel time in zone 1
                distance_km_1 = distance * ratio1
                travel_time_1 = distance_km_1 / speed_km_sec

                # travel time in zone 2
                distance_km_2 = distance * ratio2
                travel_time_2 = distance_km_2 / speed_km_sec

                # Add edge to graph
                G.add_edge(zone1["zoneid"], 
                           zone2["zoneid"], 
                           distance=distance,
                           travel_time=travel_time_1 + travel_time_2,
                           travel_time_impacted=travel_time_1 + travel_time_2,
                           mdr=None,
                           water_depth=None,
                           capacity=0.0,
                           volume=0.0
                           )
                G[zone1["zoneid"]][zone2["zoneid"]].update({zone1["zoneid"]: ratio1, 
                                                            zone2["zoneid"]: ratio2}) 
            except IndexError:
                pass
            except nx.NetworkXError:
                pass
            
        # Remove edges manually
        a = [102121, 102142, 102152, 102171, 102131, 102231, 102151, 102444, 102443, 102431, 
            102433, 102432, 102443, 102441, 102351, 102421, 102343, 102344, 102414, 102411, 
            102324, 102231, 102213, 102445, 103243, 103251, 103142, 103112, 103133, 103141, 
            102454, 147161, 102455, 147233, 147162, 147234, 147252, 147251, 147241, 102732, 
            102724, 147241, 147232, 102723, 102721, 102831, 102821, 147132, 147131, 102721, 
            147114, 147114, 147112, 102731, 102821, 147141, 102722, 102732, 103131, 103152,
            103174, 103191, 103262, 103261, 103272, 103232, 103231, 103214, 103223, 103172,
            103244, 103174, 103162, ]
        b = [102231, 102122, 102161, 102182, 102161, 102445, 102442, 102451, 102453, 102433, 
            102414, 102413, 102432, 102423, 102421, 102342, 102411, 102414, 102412, 102324, 
            102322, 102442, 102441, 102443, 103252, 103233, 103191, 103133, 103152, 103162, 
            147212, 102444, 147121, 147162, 147122, 147251, 147232, 147241, 102732, 102723, 
            102722, 147231, 147142, 102741, 102723, 102183, 147133, 147114, 147122, 102725, 
            147121, 147112, 102152, 147142, 102184, 102725, 102841, 102724, 103152, 103141, 
            103162, 103193, 103244, 103243, 103282, 103241, 103222, 103223, 103222, 103242, 
            103191, 103142, 103172, ]

        for i, j in zip(a, b):
            try:
                G.remove_edge(i, j)
            except IndexError:
                pass
            except nx.NetworkXError:
                pass

        return G

    def define_demand_supply(
            self: object,
            from_daily_to_annual: bool=True):
        """Define demand and supply for the transport model."""

        # Compute demand and supply
        supply_per_zone = pd.read_pickle(os.path.join(DATA_LOCATION_LOCAL, "trips", "supply_per_zone.pkl"))
        demand_per_zone = pd.read_pickle(os.path.join(DATA_LOCATION_LOCAL, "trips", "demand_per_zone.pkl"))
        
        # Add them to our zones
        self.taz["supply"] = supply_per_zone
        self.taz["demand"] = demand_per_zone

        return 
    
    def _cost_matrix_generator(
            self: object,
            cost_function: callable=_cost_function_haversine, 
            beta: float=0.01, 
        ):

        originList = []
        for _, zone_i in self.taz.iterrows():
            destinationList = [cost_function(zone_i, zone_j, beta) for _, zone_j in self.taz.iterrows()]
            originList.append(destinationList)

        return pd.DataFrame(originList, index=self.taz.zoneid, columns=self.taz.zoneid)

    def distribute_trips(
            self: object,
            cost_function: callable=_cost_function_haversine,
            beta: float=0.01,

        ):
        """Distribute trips according to the demand and supply."""

        cost_matrix = (self._cost_matrix_generator(cost_function=cost_function, beta=beta)).to_numpy()
        aggregates = [self.taz["supply"].fillna(0).to_numpy(), self.taz["demand"].fillna(0).to_numpy()]
        dimensions = [[0], [1]]

        IPF = ipfn.ipfn(cost_matrix, aggregates, dimensions, convergence_rate=1e-6)
        trip_distribution = IPF.iteration()

        return pd.DataFrame(trip_distribution, index=self.taz.zoneid, columns=self.taz.zoneid).fillna(0).astype(int)

    def route_trips(
            self: object,
            trip_distribution: pd.DataFrame,
            weight: str='travel_time',
            cpus: int=1,
        ):
        """Route trips according to the trip distribution and available network."""

        trip_length = trip_distribution.copy() # placeholder for trip weight
        trip_length.loc[:, :] = np.inf

        for _, od_pair in trip_distribution.reset_index().melt(id_vars='zoneid', var_name='destination', value_name='volume').rename(columns={'zoneid': 'origin'}).iterrows():
    
            if nx.has_path(self.transport_network, od_pair.origin, od_pair.destination):
                path = nx.shortest_path(self.transport_network,          
                                        od_pair.origin, 
                                        od_pair.destination,
                                        weight=weight)
                path_edges = list(zip(path[:-1], path[1:]))

                # Compute total length of route
                path_length = nx.shortest_path_length(self.transport_network,
                                                      od_pair.origin, 
                                                      od_pair.destination,
                                                      weight=weight)
                
                trip_length.loc[od_pair.origin, od_pair.destination] = path_length          

                for edge in path_edges:
                    self.transport_network.edges[edge[0], edge[1]]['volume'] += od_pair.volume
        
        volume_per_edge = [{"edge": edge, "volume": self.transport_network.edges[edge]['volume']} for edge in self.transport_network.edges]
        
        return volume_per_edge, trip_length

    def impact_transport_network(
            self: object,
        ):
        """Update impact on transport network."""

        for edge in self.transport_network.edges:          
            # Get the ratio of the edge in each of the TAZ polygons
            ratio1, ratio2 = self.transport_network.edges[edge][edge[0]], self.transport_network.edges[edge][edge[1]]

            # Get maximum water depth between the two zones where there is an edge and percentage of affected assets
            water_depth = self.taz[self.taz["zoneid"].isin(edge)]["water_depth"]

            # Compute impacted travel time in Region 1
            # compute water depth in region
            water_depth_region1 = max(water_depth.iloc[0] + self.transport_network.nodes[edge[0]]['effective_water_depth'], 0)

            # compute the impacted travel time
            distance_km_1 = self.transport_network.edges[edge]['distance'] * ratio1
            speed_km_sec = transp.max_speed_on_edge_with_impacts(water_depth_region1) / (60 * 60)
            try:
                travel_time_impacted_1 = distance_km_1 / speed_km_sec
            except ZeroDivisionError:
                travel_time_impacted_1 = np.inf

            # Compute impacted travel time in Region 2
            # compute water depth in region
            water_depth_region2 = max(water_depth.iloc[1] + self.transport_network.nodes[edge[1]]['effective_water_depth'], 0)

            # compute the impacted travel time
            distance_km_2 = self.transport_network.edges[edge]['distance'] * ratio2
            speed_km_sec = transp.max_speed_on_edge_with_impacts(water_depth_region2) / (60 * 60)
            try:
                travel_time_impacted_2 = distance_km_2 / speed_km_sec
            except ZeroDivisionError:
                travel_time_impacted_2 = np.inf

            # Update the edge with the water depth
            self.transport_network.edges[edge]['water_depth'] = (water_depth_region1, water_depth_region2)

            # If the edge is not traversable, cut edge (i.e., set its travel time to infinity)
            if travel_time_impacted_1 == np.inf or travel_time_impacted_2 == np.inf:
                # self.transport_network.remove_edge(edge[0], edge[1])
                self.transport_network.edges[edge]['travel_time_impacted'] = np.inf
                self.transport_network.edges[edge]['volume'] = 0.0

            # Else, update its travel time and reset the volume
            else:
                self.transport_network.edges[edge]['travel_time_impacted'] = travel_time_impacted_1 + travel_time_impacted_2
                self.transport_network.edges[edge]['volume'] = 0.0

        return self.transport_network

    def compute_impacts_transport(
            self: object,
            value_of_time_delay_person_per_hour: float=213,
            value_of_time_no_travel: float=347.23 * 7.4, # average income per hour, 7.4 hours per day
        ):
        """Compute transport-related impacts."""

        # Setup placeholder for delay losses
        self.delay_costs_matrix = self.trip_distribution_impacted.copy()
        self.delay_costs_matrix.loc[:, :] = 0

        # Setup placeholder for no travel losses
        self.notravel_costs_matrix = self.trip_distribution_impacted.copy()
        self.notravel_costs_matrix.loc[:, :] = 0.

        for _, od_pair in self.trip_distribution_impacted.reset_index().melt(id_vars='zoneid', var_name='destination', value_name='volume').rename(columns={'zoneid': 'origin'}).iterrows():
            # 
            if od_pair.origin == od_pair.destination:
                impacted_volume = self.trip_distribution_impacted.loc[od_pair.origin, od_pair.destination] - self.trip_distribution_original.loc[od_pair.origin, od_pair.destination]
                self.notravel_costs_matrix.loc[od_pair.origin, od_pair.destination] = impacted_volume * value_of_time_no_travel

            else:
                # If there is no path between origin and destination, compute the costs of no travel
                if self.trip_length_impacted.loc[od_pair.origin, od_pair.destination] == np.inf:
                    self.notravel_costs_matrix.loc[od_pair.origin, od_pair.destination] = od_pair.volume * value_of_time_no_travel
                    
                # If there is a path between origin and destination, compute the costs of delays
                else:
                    self.delay_costs_matrix.loc[od_pair.origin, od_pair.destination] = ( 
                        od_pair.volume 
                        * value_of_time_delay_person_per_hour 
                        * ((self.trip_length_impacted.loc[od_pair.origin, od_pair.destination] - self.trip_length_original.loc[od_pair.origin, od_pair.destination]) / (60*60)))

        # Aggregate costs
        self.impacts_delay = self.delay_costs_matrix.sum().sum()
        self.impacts_no_travel = self.notravel_costs_matrix.sum().sum()

        # Set observation properties
        self._observation_impacts_travel_delays = self.delay_costs_matrix.sum().reindex(self._zone_ids).values
        self._observation_impacts_no_travel = self.notravel_costs_matrix.sum().reindex(self._zone_ids).values

        # Update taz with impacts
        self.taz["impacts_travel_delays"] = self._observation_impacts_travel_delays
        self.taz["impacts_no_travel"] = self._observation_impacts_no_travel


        return  

    def compute_impact_over_lines(
            self: object,
            disagregated_exp: Exposures,
            exp: Exposures=None,
            disagregated_only: bool=False):
        """Get impact over lines (exposures)."""
        
        if "value" not in disagregated_exp.gdf.columns and "mdr" not in disagregated_exp.gdf.columns:
            raise ValueError("MDR and value not available to computed EAI.")

        # get eai at each line segment centroid
        disagregated_exp.gdf["eai"] = disagregated_exp.gdf["value"] * disagregated_exp.gdf["mdr"]

        if not disagregated_only:
            # aggregate eai to original exposure geometry
            exp.gdf["eai"] = disagregated_exp.gdf["eai"].groupby(level=0).agg(["sum"]).values.flatten()

        return disagregated_exp, exp
    
    def update_water_depths(
            self: object,
            disagregated_exp: Exposures,
            haz: Hazard,
            exp: Exposures=None,
            disagregated_only: bool=False):
        """Update water depth over lines (exposures) given hazard."""
        
        # get depth from hazard at each line segment centroid
        # if matrix is in CSR format, slicing taking longer. so better to work in CSC format
        if isinstance(haz.intensity, scipy.sparse.csc_matrix):
            disagregated_exp.gdf["water_depth"] = haz.intensity[:, disagregated_exp.gdf["centr_RainEvent"].values].toarray().flatten()
        
        elif isinstance(haz.intensity, scipy.sparse.csr_matrix):
            disagregated_exp.gdf["water_depth"] = haz.intensity.tocsc()[:, disagregated_exp.gdf["centr_RainEvent"].values].toarray().flatten()


        if not disagregated_only:
            # aggregate water depth to original exposure geometry
            exp.gdf["water_depth"] = disagregated_exp.gdf["water_depth"].groupby(level=0).agg(["max"]).values.flatten()

        return disagregated_exp, exp

    def update_mdr(
            self: object,
            disagregated_exp: Exposures,
            imp_f: ImpactFuncSet,
            exp: Exposures=None,
            disagregated_only: bool=False):
        """Get Mean damage ratio over lines (disagreggated exposures)."""

        if "water_depth" not in disagregated_exp.gdf.columns:
            raise ValueError("Water depth not computed yet. Cannot compute MDR.")
        
        # get depth from hazard at each line segment centroid
        disagregated_exp.gdf["mdr"] = disagregated_exp.gdf.apply(lambda row: imp_f.get_func("RainEvent", row["impf_RainEvent"]).calc_mdr(row['water_depth']), axis=1)
        
        if not disagregated_only:
            # aggregate water depth to original exposure geometry
            exp.gdf["mdr"] = disagregated_exp.gdf["mdr"].groupby(level=0).agg(["max"]).values.flatten()
        
        return disagregated_exp, exp

    def compute_impacts_direct_damage(
            self: object,
        ):
        """Compute direct infrastructure damages."""

        if "network" in self.assets_types:

            # Compute impact (Expected Annual Impact) over lines
            self.exposures["network_dis"], self.exposures["network"] = self.compute_impact_over_lines(
                self.exposures["network_dis"],
                self.exposures["network"]
            )

            # group damages by zones
            self._observation_impacts_direct_damage = self.exposures["network"].gdf.groupby('region_id')['eai'].sum().reindex(self._zone_ids).values

            # Update taz damage per zone
            self.taz["impacts_direct_damage"] = self._observation_impacts_direct_damage

            # Direct damage
            direct_damage = self.exposures["network"].gdf["eai"]
            
        if "buildings" in self.assets_types: # TODO: a lot has changed here so this probably does not work anymore...
            self.impact_obj["buildings"] = ImpactCalc(
                self.exposures["buildings"], # Exposures
                self.impact_functions,       # Impact functions
                self.hazard                  # hazard
                ).impact(save_mat=True)      # Export impact matrix
            
            # Add damages to exposures
            self.exposures["buildings"].gdf["eai_exp"] = self.impact_obj["buildings"].eai_exp
            
            # group damages by zones
            self._observation_impacts_direct_damage = self.exposures["buildings"].gdf.groupby('region_id')['eai_exp'].sum().reindex(self._zone_ids).values

            # Update taz damage per zone
            self.taz["impacts_direct_damage"] = self._observation_impacts_direct_damage

            # Direct damage
            direct_damage = self.impact_obj["buildings"].eai_exp

        return direct_damage.sum()

    # ==================================================================== #
    # ACTIONS/POLICIES
    # ==================================================================== #
    def action_to_region(
            self: object,
            action: int):
        """Convert action to region_id."""
        region = self._zone_ids[action]
        
        return region

    def compute_cost_of_action(
            self: object,
            measure: Measure):
        """Compute cost of performing an action."""
        
        # Placeholder for costs. First key indicates action, second key the type of road
        costs = {
            1: {
                1: 0.22, # from https://nhess.copernicus.org/articles/21/1011/2021/nhess-21-1011-2021-discussion.html
                3: 0.22, # from https://nhess.copernicus.org/articles/21/1011/2021/nhess-21-1011-2021-discussion.html
                5: 0.22,
                7: 0.37, # from https://nhess.copernicus.org/articles/21/1011/2021/nhess-21-1011-2021-discussion.html
            },
            2: {
                1: 0.47, # from https://nhess.copernicus.org/articles/21/1011/2021/nhess-21-1011-2021-discussion.html
                3: 0.47, # from https://nhess.copernicus.org/articles/21/1011/2021/nhess-21-1011-2021-discussion.html
                5: 0.47,
                7: 0.79, # Adjusted from costs[1][7] * costs[2][1] / costs[1][1]
            },
            3: {
                1: 1.00,
                3: 1.00,
                5: 1.00,
                7: 1.00,
            },
            4: {
                1: 1.00,
                3: 1.00,
                5: 1.00,
                7: 1.00,
            },
        }

        # Get region where action is being applied
        regions = measure.exp_region_id

        # Get action type, road type
        prev_impf, new_impf = measure.imp_fun_map.split("to")
        road_type, action = int(new_impf[0]), int(new_impf[1])

        # Get assets where action is being applied
        gdf = self.exposures["network"].gdf

        # Iterate over regions where action is being applied
        for region in regions:
            region_index = self._zone_ids.index(region)
            assets_under_action = gdf[(gdf["region_id"].isin([region])) & 
                                      (gdf["impf_RainEvent"] == int(prev_impf)) & 
                                      (gdf["water_depth"] > 0)]
            
            # Get value of assets
            value = assets_under_action["value"]

            # Compute cost of action
            cost_of_action = (costs[action][road_type] * value).sum() # cost per km

            # Update cost of action
            self.impacts_action_cost += cost_of_action

            # Update observation with costs of action
            self._observation_impacts_action_cost[region_index] += cost_of_action
            

        # Update taz with impacts
        self.taz["impacts_action_cost"] = self._observation_impacts_action_cost

        return 

    def apply_action(
            self: object,
            action: int):
        """Apply action (measure)."""
   
        # if action is 0, do nothing
        if action == 0:
            return

        # else, apply it
        else:
            
            # Get action type
            action_type = ((action-1) // len(self._zone_ids)) + 1
            if action_type not in self.available_action_types:
                raise ValueError("Action type not available.")

            # Get action region
            try:
                action_region = self._zone_ids[(action-1) % len(self._zone_ids)]
            except KeyError:
                raise ValueError("Action region not available.")
            
            # Apply action on the transport network to later assess water depth impact on speed
            if action_type == 1:
                self.transport_network.nodes[action_region]["effective_water_depth"] = -1

            # Get measure names for this action type
            measure_names = [name for name in self.measure_set.get_names()["RainEvent"] if str(action_type) in name.split("_")[0]]

            # Iterate over available assets
            for asset_type in self.assets_types:
                
                # Iterate over measures for this action type
                for measure_name in measure_names:
                    
                    # Get measure
                    measure = self.measure_set.get_measure("RainEvent", measure_name)

                    # Set region for measure
                    measure.exp_region_id = [action_region]
                    
                    # Compute cost of action
                    self.compute_cost_of_action(measure)

                    # apply measure
                    self.exposures[asset_type], self.impact_functions, self.hazard = measure.apply(
                        self.exposures[asset_type], 
                        self.impact_functions, 
                        self.hazard)
                    
            return 

    def create_measure_set(
            self: object):
        """Create a set of measure in a specific region.

        Action types mapping:
        1: Elevate road by 1m
        2: Elevate road by 2m
        3: Increase resistance by 25%
        4: Increase resistance by 50%
        """
    
        region_id = -1 # region_id placeholder
        cost = 1 # cost placeholder

        # Create a measure set
        measure_set = MeasureSet()

        # Create measures for each impact function for each road type
        for function_id in self.original_impact_function_ids:

            # Create measures for each action type available
            for action_type in self.available_action_types:
                
                # Create measure
                measure = Measure(
                    name="{}_{}".format(action_type, function_id),
                    haz_type='RainEvent',
                    color_rgb=np.array([1, 1, 1]),
                    cost=cost,
                    imp_fun_map="{}to{}{}".format(function_id, function_id, action_type),
                    exp_region_id=[int(region_id)]
                )

                measure_set.append(measure)

        return measure_set

    def valid_action_mask(self):
        """Get valid action mask."""

        actions = np.ones(self.action_space.n)

        for action in self.previous_actions:
            if action != 0:
                actions[action] = 0

        return actions
    
    def action_masks(self):
        """Get action masks."""
        return self.valid_action_mask()

    # ==================================================================== #
    # REWARD
    # ==================================================================== #
    def compute_final_reward(
            self: object,
            scale_reward: bool=True):
        """Compute reward."""
        
        reward = -1 * (
            self.beta_delay       * self.impacts_delay + 
            self.beta_notravel    * self.impacts_no_travel + 
            self.beta_action_cost * self.impacts_action_cost + 
            self.beta_damage      * self.impacts_direct_damage
        )

        # Save monetary loss over time
        self.reward_monetary_over_time.append(reward)

        # Perform reward scaling
        if scale_reward:
            if self.city_zones == "Copenhagen":
                raise ValueError("Full Copenhagen not implemented yet. Choose another city area to run world.")
            elif self.city_zones == "IndreBy":
                scale_min_value = -7501413
            else:
                raise ValueError("City zones not defined.")
            
            R_clipped = np.maximum(reward, scale_min_value)
            scaled_reward = (R_clipped - scale_min_value) / (0 - scale_min_value) * 2 - 1
            return scaled_reward
        else:
            return reward

    # ==================================================================== #
    # OBSERVATION/STATE
    # ==================================================================== #
    def _get_obs(self):

        year = np.array([self._observation_year])
        action = np.array([self.action_taken])
        rain = np.array([self._observation_rain])
        mdr = self.exposures["network"].gdf["mdr"].values
        network_effective_water_depth = np.array([self.transport_network.nodes[node]["effective_water_depth"] for node in self.transport_network.nodes()])
        network_time_impacted = np.array([self.transport_network.edges[edge]["travel_time_impacted"] for edge in self.transport_network.edges()])
        network_volume = np.array([self.transport_network.edges[edge]["volume"] for edge in self.transport_network.edges()])
        travel_delays = self._observation_impacts_travel_delays


        # Concatenate arrays
        # obs = (year, rain, mdr, network_effective_water_depth)
        obs = (year, 
               rain,
               mdr,
            #    action,
               network_effective_water_depth, 
               network_time_impacted,
               network_volume,
               travel_delays)

        return np.concatenate(obs, axis=0)

        # return {
        #     "year": np.array([self._observation_year]),
        #     "rain": np.array([self._observation_rain]),
        #     # "last_action": np.array([self.action_taken]),
        #     "mdr": self.exposures["network"].gdf["mdr"].values,
        #     "network_effective_water_depth": np.array([self.transport_network.nodes[node]["effective_water_depth"] for node in self.transport_network.nodes()]),
        #     # "impacts_direct_damage": self._observation_impacts_direct_damage, # np.array([self._observation_impacts_direct_damage.sum()]),
        #     # "impacts_travel_delays": self._observation_impacts_travel_delays, # np.array([self._observation_impacts_travel_delays.sum()]),
        #     # "impacts_no_travel":     self._observation_impacts_no_travel, # np.array([self._observation_impacts_no_travel.sum()]),
        #     # "impacts_action_cost":   self._observation_impacts_action_cost, # np.array([self._observation_impacts_action_cost.sum()]),
        # }
    
    def _get_info(self):
        return {
            "extra_info": 'This can be anything useful for analysis or debugging',
            "hazard": self.hazard,
            "exposures": self.exposures,
            "impact_functions": self.impact_functions,
            "measures": self.measure_set,
            "reward": self.reward,
            "cumulative_reward": self.cumulative_reward,
            "reward_monetary_over_time": self.reward_monetary_over_time,
            "year": self._observation_year,
            "rain": self._observation_rain,
            "impacts_direct_damage": self._observation_impacts_direct_damage,
            "impacts_travel_delays": self._observation_impacts_travel_delays,
            "impacts_no_travel": self._observation_impacts_no_travel,
            "impacts_action_cost": self._observation_impacts_action_cost,
            "transport_network": self.transport_network,
            "action_taken": self.action_taken,
        }

    # ==================================================================== #
    # STEP
    # ==================================================================== #
    def step_transport_model(
            self: object,
            action: int,
        ):
        """Perform a step in the transport model."""
        
        # remove percentage if it already exists from previous iteration
        self.taz.drop(columns=['mdr'], inplace=True, errors="ignore")
        # Aggregate mean percentage loss to zones
        self.taz = self.taz.merge(self.exposures["network"].gdf.groupby('region_id')['mdr'].mean(), 
                                  left_on='zoneid', right_on="region_id", how='left')

        # remove water depth if it already exists from previous iteration
        self.taz.drop(columns=['water_depth'], inplace=True, errors="ignore")
        # Aggregate mean water depth to zones
        self.taz = self.taz.merge(self.exposures["network"].gdf.groupby('region_id')['water_depth'].mean(), 
                                  left_on='zoneid', right_on="region_id", how='left')
        
        
        # Update travel time on transport network
        self.transport_network = self.impact_transport_network()

        # Update trip distribution
        self.trip_distribution_impacted = self.distribute_trips(cost_function=_cost_function_haversine, beta=0.5) # _cost_function_withimpacts
        
        # Update the routing and edge volumes
        self.network_volume_impacted, self.trip_length_impacted = self.route_trips(self.trip_distribution_impacted, weight="travel_time_impacted")

        # Compute transport-related impacts
        self.compute_impacts_transport()

        return

    def step(self, action):
        # set defaults
        done = False      # placeholder for done flag
        truncated = False # placeholder for truncated flag
        self.reward = 0   # placeholder for reward
        info = {}         # placeholder for debugging information

        # Reset observation
        self._observation_impacts_direct_damage = np.zeros(self.taz.shape[0])
        self._observation_impacts_travel_delays = np.zeros(self.taz.shape[0])
        self._observation_impacts_no_travel = np.zeros(self.taz.shape[0])
        self._observation_impacts_action_cost = np.zeros(self.taz.shape[0])

        self.action_taken = action
        self.previous_actions.append(action)

        # sample a rain event (hazard)
        self.hazard = self.sample_hazard(rain_event=160, 
                                         sampling_scheme=self.hazard_sampling_scheme)

        # update water depths given new hazard
        self.exposures["network_dis"], self.exposures["network"] = self.update_water_depths(self.exposures["network_dis"], self.hazard, self.exposures["network"])

        # apply action (measure)
        self.impacts_action_cost = 0
        self.apply_action(action)

        # update mean damage ratio given new water depths
        self.exposures["network_dis"], self.exposures["network"] = self.update_mdr(self.exposures["network_dis"], self.impact_functions, exp=self.exposures["network"])

        # compute impacts of hazard on assets
        self.impacts_direct_damage = self.compute_impacts_direct_damage()

        # run transport model step and compute transport-related impacts
        self.step_transport_model(action)

        # compute reward
        self.reward = self.compute_final_reward()
        self.cumulative_reward += self.reward

        # Increment observation year
        self._observation_year += 1

        # check if the episode is done
        if self._observation_year >= self.max_time_steps:
            done=True

        observation = self._get_obs()
        reward = self.reward
        info = self._get_info()

        return observation, reward, done, truncated, info

    # ==================================================================== #
    # RESET
    # ==================================================================== #
    def reset(
            self: object,
            seed: int=None, 
            options: dict=None,
        ):
        """Reset the environment."""
        super().reset(seed=seed)

        # Time
        self._observation_year = 0

        # Rain
        self._observation_rain = 0

        # Actions
        self.action_taken = 0
        self.previous_actions = []

        # Reset exposures
        self.exposures["network"].gdf["mdr"] = 0.
        self.exposures["network"].gdf["water_depth"] = 0.
        self.exposures["network"].gdf["eai"] = 0.

        self.exposures["network_dis"].gdf["mdr"] = 0.
        self.exposures["network_dis"].gdf["water_depth"] = 0.
        self.exposures["network_dis"].gdf["eai"] = 0.

        self.exposures["network"].gdf["impf_RainEvent"] = self.exposures["network"].gdf["original_impf_RainEvent"]
        self.exposures["network_dis"].gdf["impf_RainEvent"] = self.exposures["network_dis"].gdf["original_impf_RainEvent"]

        # Reset transport network
        for node in self.transport_network.nodes:
            self.transport_network.nodes[node]["effective_water_depth"] = 0.
        for edge in self.transport_network.edges:
            self.transport_network.edges[edge]["travel_time_impacted"] = self.transport_network.edges[edge]["travel_time"]
            self.transport_network.edges[edge]["volume"] = 0.

        # Rewards
        self.reward = 0
        self.cumulative_reward = 0
        self.reward_monetary_over_time = []

        # Reward function components
        self.impacts_direct_damage = 0
        self.impacts_delay = 0
        self.impacts_no_travel = 0
        self.impacts_action_cost = 0

        # Impacts per zone
        self._observation_impacts_direct_damage = np.zeros(self.taz.shape[0])
        self._observation_impacts_travel_delays = np.zeros(self.taz.shape[0])
        self._observation_impacts_no_travel = np.zeros(self.taz.shape[0])
        self._observation_impacts_action_cost = np.zeros(self.taz.shape[0])
        
        return self._get_obs(), self._get_info()

    # ==================================================================== #
    # RENDER
    # ==================================================================== #
    def render(
            self: object):
        """Render the current state of the environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        
        if self.render_mode == "human":
            return self._render_human()

    def _render_ansi(
            self: object):
        """Renders the current state as text, suitable for console output."""
        _observation = self._get_obs()
        info = self._get_info()

        print("#", "="*76, "#")
        ordered_dict = sorted(info.items())
        for key, value in ordered_dict:
            print(key, ":", value)
        print()

    def _impacts_direct_per_zone_plot(
            self: object,
            map_render: folium.Map):
        """Plot direct impacts per zone."""

        folium.Choropleth(
            geo_data=self.taz.set_index("zoneid"),
            data=self.taz,
            columns=["zoneid", "impacts_direct_damage"],
            key_on="feature.id",
            fill_color="YlOrRd",
            fill_opacity=0.5,
            line_opacity=0.2,
            highlight=True,
            legend_name="Damage",
            name="Damage",
            show=False,
        ).add_to(map_render)

        return map_render
    
    def _impacts_travel_delays_plot(
            self: object,
            map_render: folium.Map):
        """Plot indirect impacts of travel delays per zone."""

        folium.Choropleth(
            geo_data=self.taz.set_index("zoneid"),
            data=self.taz,
            columns=["zoneid", "impacts_travel_delays"],
            key_on="feature.id",
            fill_color="YlOrRd",
            fill_opacity=0.5,
            line_opacity=0.2,
            highlight=True,
            legend_name="Travel Delay Impacts",
            name="Travel_Delay_Impacts",
            show=False,
        ).add_to(map_render)

        return map_render

    def _impacts_no_travel_plot(
            self: object,
            map_render: folium.Map):
        """Plot indirect impacts of not traveling per zone."""

        folium.Choropleth(
            geo_data=self.taz.set_index("zoneid"),
            data=self.taz,
            columns=["zoneid", "impacts_no_travel"],
            key_on="feature.id",
            fill_color="YlOrRd",
            fill_opacity=0.5,
            line_opacity=0.2,
            highlight=True,
            legend_name="No Travel Impacts",
            name="No_Travel_Impacts",
            show=False,
        ).add_to(map_render)

        return map_render
    
    def _impacts_action_cost_plot(
            self: object,
            map_render: folium.Map):
        """Plot indirect impacts of not traveling per zone."""

        folium.Choropleth(
            geo_data=self.taz.set_index("zoneid"),
            data=self.taz,
            columns=["zoneid", "impacts_action_cost"],
            key_on="feature.id",
            fill_color="YlOrRd",
            fill_opacity=0.5,
            line_opacity=0.2,
            highlight=True,
            legend_name="Action Costs Impacts",
            name="Action Costs Impacts",
            show=False,
        ).add_to(map_render)

        return map_render

    def _network_plot(
            self: object,
            map_render: folium.Map):
        """Plot network."""

        network_layer = folium.FeatureGroup(name="Network", 
                                            legend_name="Transport Network",
                                            show=False)
        cmap = matplotlib.colormaps["hot_r"]

        # Draw TAZ
        for node in self.transport_network.nodes:
            
            folium.GeoJson(
                self.taz[self.taz['zoneid'] == node]['geometry'],
                opacity=.2,
                fill_opacity=.1,
                ).add_to(network_layer)

        # Draw edges
        for edge in self.transport_network.edges:
            perc_diff = (self.transport_network.edges[edge]['travel_time_impacted'] - self.transport_network.edges[edge]['travel_time']) / self.transport_network.edges[edge]['travel_time']
            
            # Draw the edge
            folium.PolyLine(
                locations=[self.transport_network.nodes[edge[0]]['centroid_4326'][::-1],
                        self.transport_network.nodes[edge[1]]['centroid_4326'][::-1]],
                #color='black',
                weight=5,
                popup="Original: {:.2f}s<br>Impacted: {:.2f}s".format(self.transport_network.edges[edge]['travel_time'],
                                                self.transport_network.edges[edge]['travel_time_impacted']),
                color=matplotlib.colors.rgb2hex(cmap(perc_diff)),
            ).add_to(network_layer)

        # Draw nodes
        for node in self.transport_network.nodes:
            folium.CircleMarker(
                location=self.transport_network.nodes[node]['centroid_4326'][::-1],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=1.,
                popup="{}<br>{}".format(self.taz[self.taz['zoneid'] == node]['zoneid'].values,
                                        self.taz[self.taz['zoneid'] == node]['zonenavn'].values[0]),
            ).add_to(network_layer)

        # Draw colorbar
        step = cmp.StepColormap(
            [matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)],
            vmin=0, vmax=1,
            caption='Travel time increase (%)'
        );
        step.add_to(map_render);
        network_layer.add_to(map_render);

        return map_render

    def _render_human(
            self: object):
        """Renders the current state as an image, suitable for human viewing."""

        # Initialize the map
        map_render = folium.Map(location=[55.68539883157378, 12.58535859765494], zoom_start=14)

        # Add layers
        self._impacts_direct_per_zone_plot(map_render)
        self._impacts_travel_delays_plot(map_render)
        self._impacts_no_travel_plot(map_render)
        self._impacts_action_cost_plot(map_render)
        self._network_plot(map_render)

        # Add base layers and controls
        folium.TileLayer('cartodbpositron').add_to(map_render)
        folium.LayerControl(collapsed=False).add_to(map_render)

        # Display the map
        return map_render

    # ==================================================================== #
    # CLOSE
    # ==================================================================== #
    def close(self):
        
        print("TODO: closing...")


