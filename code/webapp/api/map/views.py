from typing import TypedDict

from api.general.views import get_area_name_mapping, get_street_name_map, get_zone_data

from .backend import (
    ParkingmeterPosition,
    ParkingslotPosition,
    ShapeData,
    ZoneBoundary,
    get_map_data,
)
from .startup import get_data


class OutRoadData(TypedDict):
    """Data structure for the output of the road data."""

    road_id: int
    zone_id: int
    geometry: ShapeData
    road_name: str


class OutMapData(TypedDict):
    """Data structure for the output of the map data."""

    zones: dict[int, ZoneBoundary]
    center: list[float]
    parkingMeters: list[ParkingmeterPosition]
    parkingSlots: list[ParkingslotPosition]
    roads: list[OutRoadData]


def do_get_map_data() -> OutMapData:
    meshgrid = get_zone_data()
    map_data = get_data()
    parkingmeters_selection = map_data["parkingmeter_positions"]
    slots_selection = map_data["sensors"]
    roads_gdf = map_data["roads_gdf"]

    out = get_map_data(meshgrid, parkingmeters_selection, slots_selection, roads_gdf)

    area_name_mapping = get_area_name_mapping()
    street_name_map = get_street_name_map()

    zones = {area_name_mapping[k]: v for k, v in out["zones"].items()}
    center = out["center"]
    parking_meters = out["parking_meters"]
    parking_slots = out["parking_slots"]
    roads = out["roads"]
    out_roads = [
        OutRoadData(
            road_id=road["road_id"],
            zone_id=road["zone_id"],
            geometry=road["geometry"],
            road_name=street_name_map[road["road_id"]],
        )
        for road in roads
    ]

    return OutMapData(
        zones=zones,
        center=center,
        parkingMeters=parking_meters,
        parkingSlots=parking_slots,
        roads=out_roads,
    )
