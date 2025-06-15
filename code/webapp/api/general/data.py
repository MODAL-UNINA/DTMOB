import datetime
from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Data structures
FloatArray = NDArray[np.float64]


class ZoneData(TypedDict):
    """
    TypedDict for the zone data structure.
    """

    code: str
    min_lat: float
    max_lat: float
    min_lng: float
    max_lng: float
    grid: list[FloatArray]


ZoneDataMapping = dict[str, ZoneData]


TimeSlotsMacroAreas = dict[str, list[list[list[int]]]]


class ZoneDictZoneData(TypedDict):
    parcometro: list[int]
    stalli: list[int]
    camera_ztl: list[str]
    strade: list[int]
    strade_name: list[str]


ZoneDictZoneDataMapping = dict[str, ZoneDictZoneData]


class CityMapData(TypedDict):
    area_id_zone_map: dict[int, str]
    area_name_map: dict[str, int]
    id_label_map: dict[int, str]
    area_parkingmeter_map: dict[int, list[int]]
    parkingmeter_area_map: dict[int, int]
    area_sensor_map: dict[int, list[int]]
    sensor_area_map: dict[int, list[int]]
    street_name_map: dict[int, str]
    area_street_map: dict[int, list[int]]
    street_area_map: dict[int, list[int]]


class HourSlotsData(TypedDict):
    range: list[int] | None
    label: str


class AvailableDatesData(TypedDict):
    min_date: datetime.date
    max_date: datetime.date


class MacroAreaMapData(TypedDict):
    macrozone_params: dict[str, list[str]]
    timeslots_macroareas: TimeSlotsMacroAreas
    hour_slots: dict[int, HourSlotsData]
    macroarea_id_map: dict[str, int]
    macroarea_name_map: dict[int, str]
    macroarea_label_map: dict[int, str]
    macroarea_timeslot_map: dict[str, dict[tuple[int, int], bool]]
    macroarea_hourslot_map: dict[str, dict[tuple[int, int], bool]]


class GeneralData(TypedDict):
    """
    TypedDict for the general data structure.
    """

    transactions_parkingmeters: pd.DataFrame
    amount_parkingmeters: pd.DataFrame
    all_sensors: pd.DataFrame
    status_sensors: pd.DataFrame
    zone: ZoneDataMapping
    hourslots: list[int]
    timeslots_macroareas: TimeSlotsMacroAreas
    zone_dict: ZoneDictZoneDataMapping
    macrozone_params: dict[str, list[str]]
    citymap: CityMapData
    hour_slots: dict[int, HourSlotsData]
    available_dates: AvailableDatesData
    macroarea_map: MacroAreaMapData
