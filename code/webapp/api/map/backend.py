from typing import TypedDict

import geopandas as gpd
import pandas as pd

from api.general.data import ZoneDataMapping


class ZoneBoundary(TypedDict):
    """Data structure for the boundaries of a zone."""

    bounds: list[list[float]]
    label: str


class ParkingmeterPosition(TypedDict):
    """Data structure for parking meter positions."""

    lat: float
    lng: float
    road_id: int
    id: int
    zone_id: int


class ParkingslotPosition(TypedDict):
    """Data structure for parking meter positions."""

    lat: float
    lng: float
    road_id: int
    id: int
    zone_id: int


class PolygonData(TypedDict):
    """Data structure for polygon data."""

    type: str
    coordinates: tuple[tuple[tuple[float], ...], ...]


class LineStringData(TypedDict):
    """Data structure for line string data."""

    type: str
    coordinates: tuple[tuple[float, float], ...]


ShapeData = PolygonData | LineStringData


class RoadData(TypedDict):
    """Data structure for road data."""

    road_id: int
    zone_id: int
    geometry: ShapeData


class InnerOutMapData(TypedDict):
    """Data structure for the output of the map data."""

    zones: dict[str, ZoneBoundary]
    center: list[float]
    parking_meters: list[ParkingmeterPosition]
    parking_slots: list[ParkingslotPosition]
    roads: list[RoadData]


def get_map_data(
    meshgrid: ZoneDataMapping,
    parkingmeters_selection: pd.DataFrame,
    slots_selection: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> InnerOutMapData:
    import numpy as np

    zone_names = sorted(
        [zone_name for zone_name in meshgrid.keys() if zone_name != "all_map"],
        key=lambda v: int(v.split("_")[1]),
    )

    mapping_labels = {
        zone_name: zone_name.replace("_", " ").capitalize() for zone_name in zone_names
    }

    grid = meshgrid["all_map"]["grid"]

    lat_grid = grid[0]
    lon_grid = grid[1]

    center_lat = float(np.mean(lat_grid))
    center_lon = float(np.mean(lon_grid))

    zones: dict[str, ZoneBoundary] = {}

    for zone_name in zone_names:
        if zone_name == "all_map":
            continue

        zone_data = meshgrid[zone_name]

        lat_grid, lng_grid = zone_data["grid"]
        min_lat, max_lat = np.min(lat_grid).item(), np.max(lat_grid).item()
        min_lng, max_lng = np.min(lng_grid).item(), np.max(lng_grid).item()
        zone_label = mapping_labels[zone_name]

        zones[zone_name] = ZoneBoundary(
            bounds=[[min_lat, min_lng], [max_lat, max_lng]],
            label=zone_label,
        )

    zones = {k: zones[k] for k in sorted(zones)}
    parking_meters: list[ParkingmeterPosition] = list(
        parkingmeters_selection.to_dict("index").values()  # type: ignore
    )
    parking_slots: list[ParkingslotPosition] = list(
        slots_selection.to_dict("index").values()  # type: ignore
    )

    roads = [
        RoadData(
            road_id=row.road_id,  # type: ignore
            zone_id=row.zone_id,  # type: ignore
            geometry=row.geometry.__geo_interface__,  # type: ignore
        )
        for _, row in roads_gdf.iterrows()  # type: ignore
    ]

    return InnerOutMapData(
        zones=zones,
        center=[center_lat, center_lon],
        parking_meters=parking_meters,
        parking_slots=parking_slots,
        roads=roads,
    )
