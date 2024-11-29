#!python
"""
#
# @file: transport_utils.py
# @brief: MAAT environmnet setup
# ---
# @website: 
# @repo: https://github.com/MLSM-at-DTU/floods_transport_rl
# @author: Miguel Costa, Morten William Petersen.
# Auxiliary functions for the transport module within the MAAT environment.
# ---
"""

import networkx as nx
import osmnx as ox
import pandas as pd
from enum import Enum
import numpy as np

class TransportMode(Enum):
    """
    Transport mode.

    CAR : car
    BICYCLE : bicycle
    ON_FOOT : foot
    """
    CAR = "car"
    BICYCLE = "bicycle"
    ON_FOOT = "foot"


def add_speed_to_graph(
        G: nx.MultiDiGraph,
        mode: TransportMode.CAR,
    ):
    """Add speed attribute to graph edges."""
    if mode is TransportMode.CAR:
        G = _add_speed_to_graph_driving(G)
    elif mode is TransportMode.BICYCLE:
        G = _add_speed_to_graph_cycling(G)
    elif mode is TransportMode.ON_FOOT:
        G = _add_speed_to_graph_walking(G)
    else:
        raise TypeError("Transport mode specified not available: {}".format(mode))

    return G


def _add_speed_to_graph_driving(
        G: nx.MultiDiGraph,
    ):
    """Add default driving speeds to graph edges."""
    return ox.add_edge_speeds(G)


def _add_custom_speed_to_graph(
        G: nx.MultiDiGraph,
        fallback_speed: int,
        highway_types: list = ['busway', 'disused', 'living_street', 'motorway', 'motorway_link', 'primary', 'primary_link', 'residential', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 'unclassified'],

    ):
    # assign default highway speeds for all available highway types
    hwy_speeds = dict([(type, fallback_speed) for type in highway_types])

    # get graph edges
    edges = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=False)

    hwy_speed_avg = pd.Series(hwy_speeds).dropna()
    edges["highway"] = edges["highway"].map(lambda x: x[0] if isinstance(x, list) else x)
    edges["speed_kph"] = None

    # for each edge missing speed data, assign it the imputed value for itshighway type
    for hwy, group in edges.groupby("highway"):
        if hwy not in hwy_speed_avg:
            hwy_speed_avg.loc[hwy] = np.mean(group["speed_kph"])
    hwy_speed_avg = hwy_speed_avg.fillna(fallback_speed).fillna(np.mean(hwy_speed_avg))

    speed_kph = (
        edges[["highway", "speed_kph"]].set_index("highway").iloc[:, 0].fillna(hwy_speed_avg)
    )

    # add speed kph attribute to graph edges
    edges["speed_kph"] = speed_kph.to_numpy()
    nx.set_edge_attributes(G, values=edges["speed_kph"], name="speed_kph")

    return G


def _add_speed_to_graph_cycling(
        G: nx.MultiDiGraph,
        fallback_speed: int = 16.2,
    ):
    """Add custom cycling speeds to graph edges."""
    return _add_custom_speed_to_graph(G, fallback_speed)


def _add_speed_to_graph_walking(
        G: nx.MultiDiGraph,
        fallback_speed: int = 5.65,
    ):
    """Add custom walking speeds to graph edges."""
    return _add_custom_speed_to_graph(G, fallback_speed)


def max_speed_on_edge_with_impacts(
        water_depth: float,
        max_allowed_depth: float = 300,
        max_speed_noimpacts: float = 50,
    ):
    # depth in milimeters
    depth = water_depth * 1000
    
    # if no flooding
    if depth == 0:
        return max_speed_noimpacts

    # if depth larger than 300mm, we assume segment is untraversable
    elif depth > max_allowed_depth:
        return 0

    max_speed_withimpacts = 0.0009 * (depth ** 2) - 0.5529 * depth + 86.9448

    # TODO: this is a temporary fix, we should find a better way to handle this
    if max_speed_withimpacts > max_speed_noimpacts:
        return max_speed_noimpacts

    return max_speed_withimpacts
