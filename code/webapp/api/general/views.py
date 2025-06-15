import datetime
from typing import Literal

import pandas as pd

from .data import (
    CityMapData,
    HourSlotsData,
    MacroAreaMapData,
    ZoneDataMapping,
    ZoneDictZoneDataMapping,
)
from .startup import get_data

LegalityStatus = Literal["occupied_regularly", "occupied_abusively"]


def get_city_mapping() -> CityMapData:
    return get_data()["citymap"]


def get_zone_data() -> ZoneDataMapping:
    return get_data()["zone"]


def get_zone_dict() -> ZoneDictZoneDataMapping:
    return get_data()["zone_dict"]


def get_zone_name(area_id: int) -> str:
    area_id_zone_map = get_city_mapping()["area_id_zone_map"]
    return area_id_zone_map[area_id]


def get_date(selected_date: str | None) -> pd.Timestamp | None:
    if not selected_date:
        return None

    return pd.to_datetime(selected_date, format="%Y-%m-%d")  # type: ignore


def get_area_name_mapping() -> dict[str, int]:
    return get_city_mapping()["area_name_map"]


def get_area_id(zone_name: str) -> int:
    area_name_map = get_area_name_mapping()
    return area_name_map[zone_name]


def get_area_id_label_mapping() -> dict[int, str]:
    return get_city_mapping()["id_label_map"]


def get_available_dates() -> dict[str, datetime.date]:
    available_dates = get_data()["available_dates"]
    return dict(
        min_date=available_dates["min_date"], max_date=available_dates["max_date"]
    )


def get_hour_slots() -> dict[int, HourSlotsData]:
    return get_data()["hour_slots"]


def get_area_street_map() -> dict[int, list[int]]:
    return get_city_mapping()["area_street_map"]


def get_street_name_map() -> dict[int, str]:
    return get_city_mapping()["street_name_map"]


def get_macroarea_map() -> MacroAreaMapData:
    return get_data()["macroarea_map"]


def get_macroarea_hourslot_map() -> dict[str, dict[tuple[int, int], bool]]:
    return get_macroarea_map()["macroarea_hourslot_map"]


def get_hour_slots_items() -> dict[int, str]:
    return {k: v["label"] for k, v in get_hour_slots().items()}


def get_hour_slot_range(hour_slot: int) -> list[int] | None:
    return get_hour_slots()[hour_slot]["range"]


def get_parkingmeters_map() -> dict[int, list[int]]:
    return get_city_mapping()["area_parkingmeter_map"]


def get_parkingmeters(area_id: int) -> dict[int, str]:
    return {
        0: "All parking meters",
        **{
            int(p): f"Parking meter {int(p)}"
            for p in sorted(get_parkingmeters_map()[area_id])
        },
    }


def get_parkingslot_map() -> dict[int, list[int]]:
    return get_city_mapping()["area_sensor_map"]


def get_parkingslots(area_id: int) -> dict[int, str]:
    return {
        0: "All parking slots",
        **{
            int(s): f"Parking slot {int(s)}"
            for s in sorted(get_parkingslot_map()[area_id])
        },
    }


def get_parkingmeter_name(parkingmeter: int) -> int | None:
    return None if parkingmeter == 0 else parkingmeter


def get_parkingslot_name(parkingslot: int) -> int | None:
    return None if parkingslot == 0 else parkingslot


def get_transactions_parkingmeters() -> pd.DataFrame:
    return get_data()["transactions_parkingmeters"]


def get_amount_parkingmeters() -> pd.DataFrame:
    return get_data()["amount_parkingmeters"]


def get_all_sensors() -> pd.DataFrame:
    return get_data()["all_sensors"]


def get_status_sensors() -> pd.DataFrame:
    return get_data()["status_sensors"]


def get_road_id(road_id: int) -> int | None:
    street_name_map = get_street_name_map()
    assert 0 not in street_name_map

    out = int(road_id)
    if out == 0:
        return None

    return int(road_id)


def get_legality_status_name(legality_status: str) -> LegalityStatus | None:
    if legality_status == "both":
        return None
    return "occupied_regularly" if legality_status == "legal" else "occupied_abusively"
