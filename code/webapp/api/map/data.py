from typing import TypedDict

import geopandas as gpd
import pandas as pd


class MapData(TypedDict):
    """
    TypedDict for the map data structure.
    """

    parkingmeter_positions: pd.DataFrame
    roads_gdf: gpd.GeoDataFrame
    sensors: pd.DataFrame
