from typing import Any

import geopandas as gpd
import pandas as pd

from api.general.data import ZoneDictZoneDataMapping

from .data import MapData


# Private function to postprocess the data
def _postprocess_parkingmeter_positions(
    data: pd.DataFrame, zone_dict: ZoneDictZoneDataMapping, zone_names: list[str]
) -> pd.DataFrame:
    data["id_parcometro"] = data["id_parcometro"].astype(int)
    data = data.rename(columns={"id_parcometro": "id", "id_strada": "road_id"})
    data["zone_id"] = data["id"].apply(  # type: ignore
        lambda x: int(  # type: ignore
            [
                zone_name
                for zone_name in zone_names
                if x in zone_dict[zone_name]["parcometro"]
            ][0].split("_")[1]
        )
        + 1
    )
    return data


def _postprocess_roads_gdf(
    data: gpd.GeoDataFrame, zone_dict: ZoneDictZoneDataMapping, zone_names: list[str]
) -> gpd.GeoDataFrame:
    data.insert(  # type: ignore
        1,
        "zone_id",
        data["road_id"].apply(  # type: ignore
            lambda x: int(  # type: ignore
                [
                    zone_name
                    for zone_name in zone_names
                    if x in zone_dict[zone_name]["strade"]
                ][0].split("_")[1]
            )
            + 1
        ),
    )
    return data


def _postprocess_sensors(
    data: pd.DataFrame, zone_dict: ZoneDictZoneDataMapping, zone_names: list[str]
) -> pd.DataFrame:
    data = data.rename(columns={"id_strada": "road_id"})

    sensors_mapping = {
        zone_name: zone_dict[zone_name]["stalli"] for zone_name in zone_names
    }
    sensors_sets = [set(v) for v in sensors_mapping.values()]
    assert len(set.intersection(*sensors_sets)) == 0, (  # type: ignore
        "There are sensors that are present in multiple zones. "
        "This is not allowed and will cause issues in the application."
    )
    sensors_mapping_inv = {
        x: zone_name for zone_name, v in sensors_mapping.items() for x in v
    }
    data["zone_id"] = data["id"].apply(  # type: ignore
        lambda x:  # type: ignore
        int(sensors_mapping_inv[x].split("_")[1]) + 1
        if x in sensors_mapping_inv
        else -1
    )
    data = data[data["zone_id"] != -1]  # Filter out sensors not in any zone
    return data


def preprocess(data: dict[str, Any]) -> MapData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    from api.general.views import get_zone_dict

    zone_dict = get_zone_dict()
    zone_names = sorted(
        [zone_name for zone_name in zone_dict.keys() if zone_name != "all_map"],
        key=lambda v: int(v.split("_")[1]),
    )

    return MapData(
        parkingmeter_positions=_postprocess_parkingmeter_positions(
            data["parkingmeter_positions"], zone_dict, zone_names
        ),
        roads_gdf=_postprocess_roads_gdf(data["roads_gdf"], zone_dict, zone_names),
        sensors=_postprocess_sensors(data["sensors"], zone_dict, zone_names),
    )
