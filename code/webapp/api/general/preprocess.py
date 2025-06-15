from typing import Any

import datetime
import pandas as pd

from .data import (  # isort: skip
    AvailableDatesData,
    CityMapData,
    GeneralData,
    HourSlotsData,
    MacroAreaMapData,
    ZoneDictZoneDataMapping,
    TimeSlotsMacroAreas,
)


def build_city_map(data: ZoneDictZoneDataMapping) -> CityMapData:
    """
    Build a city map from the provided JSON data.

    Args:
        data (dict): The JSON data containing the city map information.

    Returns:
        dict: The city map data structure.
    """

    def get_zone_label(zone: str) -> str:
        if zone == "all_map":
            return "All zones"
        return zone.replace("zone_", "Zone ").capitalize()

    def get_zone_id(zone: str) -> int:
        if zone == "all_map":
            return 0
        return int(zone.replace("zone_", "")) + 1

    zones = list(data.keys())

    assert "all_map" in zones

    zones = [zone for zone in zones if zone != "all_map"]

    zones = ["all_map"] + sorted(zones)

    assert all(zone.startswith("zone_") for zone in zones if zone != "all_map")

    id_zone_map = {get_zone_id(zone): zone for zone in zones}

    zone_name_map = {v: k for k, v in id_zone_map.items()}

    id_label_map = {
        zone_id: get_zone_label(zone) for zone_id, zone in id_zone_map.items()
    }

    zone_parkingmeter_map = {
        zone_id: data[zone]["parcometro"] for zone_id, zone in id_zone_map.items()
    }

    parkingmeter_zone_map = {
        parkingmeter: zone_id
        for zone_id, zone in id_zone_map.items()
        for parkingmeter in data[zone]["parcometro"]
    }
    parkingmeter_zone_map = {
        parkingmeter: parkingmeter_zone_map[parkingmeter]
        for parkingmeter in sorted(parkingmeter_zone_map)
    }

    slots = [s for zone in zones for s in data[zone]["stalli"]]

    assert set(slots) == set(data["all_map"]["stalli"])

    zone_slots_map = {
        zone_id: sorted(data[zone]["stalli"]) for zone_id, zone in id_zone_map.items()
    }

    slots_zone_map = {
        slot: [
            zone_id
            for zone_id, zone in id_zone_map.items()
            if slot in data[zone]["stalli"]
        ]
        for slot in slots
    }
    slots_zone_map = {
        slot: sorted(slots_zone_map[slot]) for slot in sorted(slots_zone_map)
    }

    roads = [s for zone in zones for s in data[zone]["strade"]]

    assert set(roads) == set(data["all_map"]["strade"])

    roads_names = [s for zone in zones for s in data[zone]["strade_name"]]

    n_roads = len(roads)
    n_roads_unique = len(set(roads))

    roads_names_pairs = list(zip(roads, roads_names, strict=False))
    assert len(roads_names_pairs) == n_roads

    # Check that the mapping is unique
    assert len(set(roads_names_pairs)) == n_roads_unique

    roads_name_map = dict(roads_names_pairs)

    zone_roads_map = {
        zone_id: sorted(data[zone]["strade"]) for zone_id, zone in id_zone_map.items()
    }

    road_zone_map = {
        road: [
            zone_id
            for zone_id, zone in id_zone_map.items()
            if road in data[zone]["strade"]
        ]
        for road in roads
    }
    road_zone_map = {
        road: sorted(road_zone_map[road]) for road in sorted(road_zone_map)
    }

    return CityMapData(
        area_id_zone_map=id_zone_map,
        area_name_map=zone_name_map,
        id_label_map=id_label_map,
        area_parkingmeter_map=zone_parkingmeter_map,
        parkingmeter_area_map=parkingmeter_zone_map,
        area_sensor_map=zone_slots_map,
        sensor_area_map=slots_zone_map,
        street_name_map=roads_name_map,
        area_street_map=zone_roads_map,
        street_area_map=road_zone_map,
    )


def build_hour_slots(hour_slots: list[int]) -> dict[int, HourSlotsData]:
    hour_slots_ranges = [
        [hour_slots[i], hour_slots[i + 1]] for i in range(len(hour_slots) - 1)
    ]

    hour_slots_s = [
        f"{start:02d}:00 - {end:02d}:00" for start, end in hour_slots_ranges
    ]

    pre_out = {
        i + 1: HourSlotsData(range=range, label=label)
        for i, (range, label) in enumerate(
            zip(hour_slots_ranges, hour_slots_s, strict=False)
        )
    }

    out = {0: HourSlotsData(range=None, label="All day"), **pre_out}

    return out


def build_available_dates(df_data: pd.DataFrame) -> AvailableDatesData:
    from typing import cast

    min_date = cast(
        datetime.date,
        df_data.index.min().date(),  # type: ignore
    )
    max_date = cast(
        datetime.date,
        df_data.index.max().date() - datetime.timedelta(days=6),  # type: ignore
    )

    return AvailableDatesData(min_date=min_date, max_date=max_date)


def build_macroarea_map(
    macrozone_data: dict[str, list[str]],
    timeslots_data: TimeSlotsMacroAreas,
    hour_slots_data: dict[int, HourSlotsData],
) -> MacroAreaMapData:
    macrozones = sorted(list(macrozone_data.keys()))
    assert sorted(list(timeslots_data.keys())) == macrozones
    macrozones_ids = list(range(len(macrozones)))

    macrozone_id_map = dict(zip(macrozones, macrozones_ids, strict=True))
    macrozone_name_map = dict(zip(macrozones_ids, macrozones, strict=True))

    macrozones_labels = ["Macroarea " + s.split("_")[1] for s in macrozones]
    macroarea_label_map = dict(zip(macrozones_ids, macrozones_labels, strict=True))

    assert all(len(timeslots_data[macrozone]) == 7 for macrozone in macrozones)

    timeslot_map: dict[str, dict[tuple[int, int], bool]] = {}
    for macrozone in macrozones:
        timezone_data = timeslots_data[macrozone]
        out_timeslots: dict[tuple[int, int], bool] = {}
        for weekday, timeslots in enumerate(timezone_data):
            payment_hours = [
                q for timeslot in timeslots for q in range(timeslot[0], timeslot[1])
            ]
            for hour in range(24):
                out_timeslots[weekday, hour] = hour in payment_hours
        timeslot_map[macrozone] = out_timeslots

    hourslot_map: dict[str, dict[tuple[int, int], bool]] = {}
    for macrozone, out_timeslots in timeslot_map.items():
        out_hourslots = {}
        for weekday in range(7):
            for i_hourslot, hourslot_data in hour_slots_data.items():
                hourslot_range = hourslot_data["range"]
                out_hourslots[weekday, i_hourslot] = hourslot_range is None or any(
                    out_timeslots[(weekday, hour)]
                    for hour in range(hourslot_range[0], hourslot_range[1])
                )
        hourslot_map[macrozone] = out_hourslots

    return MacroAreaMapData(
        macrozone_params=macrozone_data,
        timeslots_macroareas=timeslots_data,
        hour_slots=hour_slots_data,
        macroarea_id_map=macrozone_id_map,
        macroarea_name_map=macrozone_name_map,
        macroarea_label_map=macroarea_label_map,
        macroarea_timeslot_map=timeslot_map,
        macroarea_hourslot_map=hourslot_map,
    )


# Private function to postprocess the data
def preprocess(data: dict[str, Any]) -> GeneralData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    data["citymap"] = build_city_map(data["zone_dict"])
    data["hour_slots"] = build_hour_slots(data["hourslots"])
    data["available_dates"] = build_available_dates(data["transactions_parkingmeters"])
    data["macroarea_map"] = build_macroarea_map(
        data["macrozone_params"],
        data["timeslots_macroareas"],
        data["hour_slots"],
    )
    return GeneralData(**data)
