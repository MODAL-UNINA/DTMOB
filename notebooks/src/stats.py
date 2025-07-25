from typing import Any, Literal, NamedTuple, Sequence, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk  # type: ignore
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .general import ZoneDataMapping, ZoneDictZoneDataMapping

ZoneParamsMapping = dict[str, list[str]]


class StatsData(TypedDict):
    """
    TypedDict for the statistics data structure.
    """

    events_data: pd.DataFrame
    multe_data: pd.DataFrame
    poi_data: pd.DataFrame
    zone_params: ZoneParamsMapping


def preprocess(data: dict[str, Any]) -> StatsData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    return StatsData(**data)


class ParkingMeterAnalysis(TypedDict):
    number_of_parkingmeters: int
    most_used_parkingmeters: int | None
    least_used_parkingmeters: int | None


class ParkingSlotAnalysis(TypedDict):
    number_of_parkingslots: int
    most_used_parkingslots: int | None
    least_used_parkingslots: int | None
    average_daily_influx: float | None


class PoiCategoryCount(TypedDict):
    commercial: int
    cultural: int
    education: int
    finance: int
    food_and_drink: int
    healthcare: int
    services: int


class PoiAnalysis(TypedDict):
    number_of_poi: int
    number_of_poi_by_category: PoiCategoryCount
    average_distance_between_parkingmeters_and_poi: float


class EventCategoryCount(TypedDict):
    art_exhibitions: int
    concerts: int
    culture_other_events: int
    festivals_fairs_feasts: int
    clubs_pubs: int
    exhibitions_festivals_events: int
    theater: int


class EventAnalysis(TypedDict):
    number_of_events: int
    number_of_events_by_category: EventCategoryCount
    average_distance_between_parkingmeters_and_events: float
    average_daily_events: int


class OutStatsData(TypedDict):
    parkingmeter_analysis: ParkingMeterAnalysis
    parkingslot_analysis: ParkingSlotAnalysis
    poi_analysis: PoiAnalysis
    event_analysis: EventAnalysis


class Mann_Kendall_Test(NamedTuple):
    trend: Literal["increasing", "decreasing", "no trend"]
    h: np.bool_
    p: np.float64
    z: np.float64
    Tau: np.float64
    s: np.float64
    var_s: np.float64
    slope: np.float64
    intercept: np.float64


class AbusivismIndex(TypedDict):
    trend: str | None
    p_value: float | None
    sen_slope: float | None


class Fines(TypedDict):
    trend: str | None
    p_value: float | None
    sen_slope: float | None


class AbusivismData(TypedDict):
    abusivism_index: dict[str, AbusivismIndex]
    fines: dict[str, Fines]


class ColorMap(TypedDict):
    background_color: str
    border_color: str


def test_mann_kendall(
    data: pd.DataFrame, value_column: str, sort_column: str
) -> Mann_Kendall_Test | None:
    if data.empty or len(data) < 3:
        return None

    data = data.sort_values(sort_column)  # type: ignore
    result = mk.original_test(  # type: ignore
        data[value_column]
    )

    return Mann_Kendall_Test(
        trend=result.trend,  # type: ignore
        h=result.h,  # type: ignore
        p=result.p,  # type: ignore
        z=result.z,  # type: ignore
        Tau=result.Tau,  # type: ignore
        s=result.s,  # type: ignore
        var_s=result.var_s,  # type: ignore
        slope=result.slope,  # type: ignore
        intercept=result.intercept,  # type: ignore
    )


def sen_slope(data: pd.DataFrame, value_column: str, sort_column: str) -> float | None:
    if data.empty:
        return None

    n = len(data)
    data = data.sort_values(sort_column)  # type: ignore
    data_values = cast(NDArray[np.float64], data[value_column].values)  # type: ignore
    slopes: list[np.float64] = []
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((data_values[j] - data_values[i]) / (j - i))
    return float(np.median(slopes))


def find_shift(
    hour_range: Sequence[int], shift_hour_range: dict[str, list[tuple[int, int]]]
) -> str | None:
    for shift, ranges in shift_hour_range.items():
        for start, end in ranges:
            if hour_range[0] >= start and hour_range[1] <= end:
                return shift

    return None


def get_stats_info(
    zone_data: ZoneDataMapping,
    mapping_dict: ZoneDictZoneDataMapping,
    poi_data: pd.DataFrame,
    events_data: pd.DataFrame,
    transactions_parkingmeters: pd.DataFrame,
    all_parkingslots: pd.DataFrame,
    status_parkingslots: pd.DataFrame,
    fines_data: pd.DataFrame,
    zone_params: ZoneParamsMapping,
    zone_name: str = "all_map",
    date: pd.Timestamp | None = None,
    hour_range: Sequence[int] | None = None,
) -> tuple[
    OutStatsData,
    AbusivismData | None,
    tuple[Figure, Figure] | None,
    dict[str, str],
    dict[str, ColorMap],
]:
    import seaborn as sns

    results = OutStatsData(
        parkingmeter_analysis=ParkingMeterAnalysis(
            number_of_parkingmeters=0,
            most_used_parkingmeters=None,
            least_used_parkingmeters=None,
        ),
        parkingslot_analysis=ParkingSlotAnalysis(
            number_of_parkingslots=0,
            most_used_parkingslots=None,
            least_used_parkingslots=None,
            average_daily_influx=None,
        ),
        poi_analysis=PoiAnalysis(
            number_of_poi=0,
            number_of_poi_by_category=PoiCategoryCount(
                commercial=0,
                cultural=0,
                education=0,
                finance=0,
                food_and_drink=0,
                healthcare=0,
                services=0,
            ),
            average_distance_between_parkingmeters_and_poi=0.0,
        ),
        event_analysis=EventAnalysis(
            number_of_events=0,
            number_of_events_by_category=EventCategoryCount(
                art_exhibitions=0,
                concerts=0,
                culture_other_events=0,
                festivals_fairs_feasts=0,
                clubs_pubs=0,
                exhibitions_festivals_events=0,
                theater=0,
            ),
            average_distance_between_parkingmeters_and_events=0.0,
            average_daily_events=0,
        ),
    )

    results_abusivisms = None
    correlations: tuple[Figure, Figure] | None = None
    label_map = {
        "parkingmeter_analysis": "Parking meter Analysis",
        "parkingslot_analysis": "Parking slot Analysis",
        "poi_analysis": "POIs Analysis",
        "event_analysis": "Events Analysis",
        "abusivism_index": "Abusivism Index",
        "fines": "Fines",
        "number_of_parkingmeters": "Number of Parking meters",
        "most_used_parkingmeters": "Most Used Parking meters (ID)",
        "least_used_parkingmeters": "Least Used Parking meters  (ID)",
        "number_of_parkingslots": "Number of Parking slots",
        "most_used_parkingslots": "Most Used Parking slots (ID)",
        "least_used_parkingslots": "Least Used Parking slots (ID)",
        "average_daily_influx": "Average Daily Influx",
        "number_of_poi": "Number of POIs",
        "number_of_poi_by_category": "Number of POIs by Category",
        "average_distance_between_parkingmeters_and_poi": (
            "Average Distance between Parking meters and POI"
        ),
        "cultural": "Cultural",
        "commercial": "Commercial",
        "education": "Education",
        "finance": "Finance",
        "food_and_drink": "Food & Drink",
        "healthcare": "Healthcare",
        "services": "Services",
        "average_distance_between_parkingmeters_and_events": (
            "Average Distance between Parking meters and Events (meters)"
        ),
        "number_of_events": "Number of Events",
        "number_of_events_by_category": "Number of Events by Category",
        "art_exhibitions": "Art & Exhibitions",
        "concerts": "Concerts",
        "culture_other_events": "Culture & Other Events",
        "festivals_fairs_feasts": "Festivals, Fairs & Feasts",
        "clubs_pubs": "Clubs & Pubs",
        "exhibitions_festivals_events": "Exhibitions, Festivals & Events",
        "theater": "Theater",
        "average_daily_events": "Average Daily Events",
        "Morning": "Morning",
        "Afternoon": "Afternoon",
        "Evening": "Evening",
        "trend": "Trend",
        "p_value": "P-value",
        "sen_slope": "Sen's Slope",
    }

    color_map = {
        "parkingmeter_analysis": ColorMap(
            background_color="#c7e0c7ff", border_color="#108532ff"
        ),
        "parkingslot_analysis": ColorMap(
            background_color="#fae8c7ff", border_color="#fbc379ff"
        ),
        "poi_analysis": ColorMap(
            background_color="#499bd266", border_color="#499bd2ff"
        ),
        "event_analysis": ColorMap(
            background_color="#499bd266", border_color="#499bd2ff"
        ),
        "abusivism_index": ColorMap(
            background_color="#da572a66", border_color="#da5f2aff"
        ),
        "fines": ColorMap(background_color="#da572a66", border_color="#da5f2aff"),
    }

    parkingmeters = mapping_dict[zone_name]["parcometro"]
    transactions_parkingmeters = transactions_parkingmeters[parkingmeters]

    parkingslots = mapping_dict[zone_name]["stalli"]
    all_parkingslots = all_parkingslots[
        all_parkingslots["numeroStallo"].isin(parkingslots)  # type: ignore
    ]
    status_parkingslots = status_parkingslots[
        status_parkingslots["idStallo"].isin(parkingslots)  # type: ignore
    ]
    fines_data = fines_data[
        fines_data["num_stallo"].isin(parkingslots)  # type: ignore
    ]

    if zone_name != "all_map":
        zone_names = [zone_name]
    else:
        zone_names = list(mapping_dict.keys())
        zone_names.remove("all_map")

    status_parkingslots = status_parkingslots[
        status_parkingslots["zone_name"].isin(zone_names)  # type: ignore
    ]
    fines_data = fines_data[
        fines_data["zone_name"].isin(zone_names)  # type: ignore
    ]

    status_parkingslots1 = status_parkingslots.copy()
    fines_data1 = fines_data.copy()

    if date is not None:
        selection = date + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
        transactions_parkingmeters = transactions_parkingmeters.loc[date:selection]
        all_parkingslots = all_parkingslots[
            (all_parkingslots["datetime"] >= date)
            & (all_parkingslots["datetime"] <= selection)
        ]
        status_parkingslots = status_parkingslots[
            (status_parkingslots["datetime"] >= date)
            & (status_parkingslots["datetime"] <= selection)
        ]
        fines_data = fines_data[
            (fines_data["datetime"] >= date) & (fines_data["datetime"] <= selection)
        ]
        events_data = events_data[
            (events_data["days"] >= date) & (events_data["days"] <= selection)
        ]

        fines_data1 = fines_data1[fines_data1["datetime"] <= selection]
        status_parkingslots1 = status_parkingslots1[
            status_parkingslots1["datetime"] <= selection
        ]

    if hour_range is not None:
        hour_range1 = list(range(hour_range[0], hour_range[1]))
        transactions_parkingmeters = cast(
            pd.DataFrame,
            transactions_parkingmeters[
                transactions_parkingmeters.index.hour.isin(hour_range1)  # type: ignore
            ],
        )
        all_parkingslots = all_parkingslots[
            all_parkingslots["datetime"].dt.hour.isin(hour_range1)  # type: ignore
        ]
        status_parkingslots = status_parkingslots[
            status_parkingslots["datetime"].dt.hour.isin(hour_range1)  # type: ignore
        ]
        fines_data = fines_data[
            fines_data["datetime"].dt.hour.isin(hour_range1)  # type: ignore
        ]

    min_lat, min_lng, max_lat, max_lng = (
        zone_data[zone_name]["min_lat"],
        zone_data[zone_name]["min_lng"],
        zone_data[zone_name]["max_lat"],
        zone_data[zone_name]["max_lng"],
    )

    all_transactions = cast(
        "pd.Series[pd.Float64Dtype]",
        transactions_parkingmeters.sum(axis=0),  # type: ignore
    )

    results["parkingmeter_analysis"]["number_of_parkingmeters"] = len(parkingmeters)
    if not all_transactions.empty:
        results["parkingmeter_analysis"]["most_used_parkingmeters"] = int(
            all_transactions.idxmax()
        )
        results["parkingmeter_analysis"]["least_used_parkingmeters"] = int(
            all_transactions.idxmin()
        )
    else:
        results["parkingmeter_analysis"]["most_used_parkingmeters"] = None
        results["parkingmeter_analysis"]["least_used_parkingmeters"] = None

    all_parkingslots1 = all_parkingslots.drop_duplicates(
        ["numeroStallo", "datetime", "next_datetime"]
    )
    all_parkingslots_ = all_parkingslots1.groupby(  # type: ignore
        ["numeroStallo"]
    ).size()
    all_parkingslots_1 = all_parkingslots1.groupby(  # type: ignore
        ["days"]
    ).size()

    if not all_parkingslots_.empty:
        results["parkingslot_analysis"]["number_of_parkingslots"] = len(parkingslots)
        results["parkingslot_analysis"]["most_used_parkingslots"] = int(
            all_parkingslots_.idxmax()
        )
        results["parkingslot_analysis"]["least_used_parkingslots"] = int(
            all_parkingslots_.idxmin()
        )
        results["parkingslot_analysis"]["average_daily_influx"] = round(
            float(all_parkingslots_1.mean()), 2
        )

    else:
        results["parkingslot_analysis"]["number_of_parkingslots"] = 0
        results["parkingslot_analysis"]["most_used_parkingslots"] = None
        results["parkingslot_analysis"]["least_used_parkingslots"] = None
        results["parkingslot_analysis"]["average_daily_influx"] = None

    poi_data = poi_data[
        (poi_data["lat_poi"] >= min_lat)
        & (poi_data["lat_poi"] <= max_lat)
        & (poi_data["lng_poi"] >= min_lng)
        & (poi_data["lng_poi"] <= max_lng)
    ]

    n_poi = len(poi_data)
    n_poi_cat = poi_data.groupby(  # type: ignore
        "category"
    ).size()

    category_count = PoiCategoryCount(
        **n_poi_cat.to_dict()  # type: ignore
    )

    results["poi_analysis"]["number_of_poi"] = n_poi
    results["poi_analysis"]["number_of_poi_by_category"] = category_count

    if n_poi > 0:
        mean_distance = float(
            poi_data[parkingmeters].mean().mean()  # type: ignore
        )
        results["poi_analysis"]["average_distance_between_parkingmeters_and_poi"] = (
            round(mean_distance, 2)
        )
    else:
        results["poi_analysis"]["average_distance_between_parkingmeters_and_poi"] = 0

    events_data = events_data[
        (events_data["lat_event"] >= min_lat)
        & (events_data["lat_event"] <= max_lat)
        & (events_data["lng_event"] >= min_lng)
        & (events_data["lng_event"] <= max_lng)
    ]

    category_count = {
        "Arte e Mostre": 0,
        "Concerti": 0,
        "Cultura ed altri eventi": 0,
        "Feste, Fiere e Sagre": 0,
        "Locali e Pub": 0,
        "Rassegne, Festival, Manifestazioni": 0,
        "Teatro": 0,
    }
    n_events = len(events_data)
    n_events_cat = (
        events_data.groupby(  # type: ignore
            "Type"
        )
        .size()
        .to_dict()
    )
    category_count.update(n_events_cat)
    translation_map = {
        "Arte e Mostre": "art_exhibitions",
        "Concerti": "concerts",
        "Cultura ed altri eventi": "culture_other_events",
        "Feste, Fiere e Sagre": "festivals_fairs_feasts",
        "Locali e Pub": "clubs_pubs",
        "Rassegne, Festival, Manifestazioni": "exhibitions_festivals_events",
        "Teatro": "theater",
    }
    category_count = EventCategoryCount(
        **{
            translation_map[category]: count
            for category, count in category_count.items()
        }
    )

    results["event_analysis"]["number_of_events"] = n_events
    results["event_analysis"]["number_of_events_by_category"] = category_count

    if n_events > 0:
        mean_events = float(
            events_data.groupby("days").size().mean()  # type: ignore
        )
        results["event_analysis"]["average_daily_events"] = int(mean_events)
        mean_distance = float(
            events_data[parkingmeters].mean().mean()  # type: ignore
        )
        results["event_analysis"][
            "average_distance_between_parkingmeters_and_events"
        ] = round(mean_distance, 2)
    else:
        results["event_analysis"]["average_daily_events"] = 0
        results["event_analysis"][
            "average_distance_between_parkingmeters_and_events"
        ] = 0

    if all_parkingslots_.empty:
        return results, results_abusivisms, correlations, label_map, color_map

    slots_n = {zone_name: len(v["stalli"]) for zone_name, v in mapping_dict.items()}

    if zone_name == "all_map":
        status_parkingslots1["zone_name"] = "all_map"
        fines_data1["zone_name"] = "all_map"
        zone_names = ["all_map"]

    status_parkingslots_grouped = (
        status_parkingslots1.groupby(["zone_name", "datetime", "shift"])  # type: ignore
        .agg({"occupied_abusively": "sum", "occupied_regularly": "sum"})
        .reset_index()
    )

    status_parkingslots_grouped["indice_abusivismo"] = status_parkingslots_grouped[
        "occupied_abusively"
    ] / status_parkingslots_grouped["zone_name"].map(slots_n)  # type: ignore

    zone_status = (
        status_parkingslots_grouped.groupby(  # type: ignore
            ["zone_name", "datetime", "shift"]
        )
        .agg({"indice_abusivismo": "mean"})
        .reset_index()
    )

    results_abuse: dict[tuple[str, str], AbusivismIndex] = {}

    for zone in zone_names:
        for shift in ["morning", "afternoon", "evening"]:
            if (shift == "afternoon") and (zone in zone_params["zona_B"]):
                continue
            data = zone_status[
                (zone_status["zone_name"] == zone) & (zone_status["shift"] == shift)
            ]
            result = test_mann_kendall(data, "indice_abusivismo", "datetime")
            if result is None:
                results_abuse[(zone, shift)] = AbusivismIndex(
                    trend="No trend",
                    p_value=None,
                    sen_slope=None,
                )
                continue

            sen = sen_slope(data, "indice_abusivismo", "datetime")
            results_abuse[(zone, shift)] = AbusivismIndex(
                trend=result.trend,
                p_value=result.p.item(),
                sen_slope=sen,
            )

    fines_data_grouped = (
        fines_data1.groupby(["zone_name", "datetime", "shift"])  # type: ignore
        .agg({"id": "count"})
        .reset_index()
    )
    fines_data_grouped.rename(columns={"id": "num_fines"}, inplace=True)

    results_fines: dict[tuple[str, str], Fines] = {}
    for zone in zone_names:
        for shift in ["morning", "afternoon", "evening"]:
            if (shift == "afternoon") and (zone in zone_params["zona_B"]):
                results_fines[(zone, shift)] = Fines(
                    trend=None,
                    p_value=None,
                    sen_slope=None,
                )
                continue
            data = fines_data_grouped[
                (fines_data_grouped["zone_name"] == zone)
                & (fines_data_grouped["shift"] == shift)
            ]
            result = test_mann_kendall(data, "num_fines", "datetime")
            if result is None:
                results_fines[(zone, shift)] = Fines(
                    trend="No trend",
                    p_value=None,
                    sen_slope=None,
                )
                continue
            sen = sen_slope(data, "num_fines", "datetime")
            results_fines[(zone, shift)] = Fines(
                trend=result.trend,
                p_value=result.p.item(),
                sen_slope=sen,
            )

    new_df = zone_status.merge(  # type: ignore
        fines_data_grouped, on=["zone_name", "datetime", "shift"], how="outer"
    )
    new_df.fillna(0, inplace=True)  # type: ignore
    pivot_corr = (
        new_df.groupby(  # type: ignore
            ["zone_name", "shift"]
        )[["indice_abusivismo", "num_fines"]]
        .corr()
        .iloc[0::2, -1]
        .unstack()
        .reset_index()
        .pivot(index="zone_name", columns="shift", values="indice_abusivismo")
    )

    if zone_name in zone_params["zona_B"]:
        pivot_corr = pivot_corr.drop("afternoon", axis=1)
        results_abuse = {k: v for k, v in results_abuse.items() if k[1] != "afternoon"}
        results_fines = {k: v for k, v in results_fines.items() if k[1] != "afternoon"}

    shift_hour_range = {
        "morning": [(8, 12)],
        "afternoon": [(12, 16)],
        "evening": [(16, 24)],
    }

    if hour_range is not None:
        result_ = find_shift(hour_range, shift_hour_range)
        if result_ is None:
            return results, results_abusivisms, correlations, label_map, color_map
        if result_ == "afternoon" and zone_name in zone_params["zona_B"]:
            return results, results_abusivisms, correlations, label_map, color_map
        temp_abuse = {}
        temp_fines = {}
        for z in zone_names:
            temp_abuse[(z, result_)] = results_abuse[(z, result_)]
            temp_fines[(z, result_)] = results_fines[(z, result_)]
            pivot_corr = pd.DataFrame(
                pivot_corr.loc[z, result_], columns=[result_], index=[z]
            )

    pivot_corr = pivot_corr[
        [col for col in shift_hour_range.keys() if col in pivot_corr.columns]
    ]

    pivot_corr = pivot_corr.rename(
        index={
            zone_name: (
                f"Zone {zone_name.split('_')[1]}"
                if zone_name != "all_map"
                else "All zones"
            )
        }
    )

    ab_index = {
        shift.capitalize(): AbusivismIndex(
            trend=abuse_data["trend"],
            p_value=round(abuse_data["p_value"], 2)
            if abuse_data["p_value"] is not None
            else None,
            sen_slope=round(abuse_data["sen_slope"], 2)
            if abuse_data["sen_slope"] is not None
            else None,
        )
        for (_, shift), abuse_data in results_abuse.items()
    }

    fines_ = {
        shift.capitalize(): Fines(
            trend=fine_data["trend"],
            p_value=round(fine_data["p_value"], 2)
            if fine_data["p_value"] is not None
            else None,
            sen_slope=round(fine_data["sen_slope"], 2)
            if fine_data["sen_slope"] is not None
            else None,
        )
        for (_, shift), fine_data in results_fines.items()
    }

    results_abusivisms = AbusivismData(abusivism_index=ab_index, fines=fines_)

    fig_rel, ax_rel = plt.subplots(figsize=(5, 2))  # type: ignore
    sns.heatmap(  # type: ignore
        pivot_corr,
        annot=True,
        cmap="coolwarm",
        cbar=True,
        ax=ax_rel,
    )
    ax_rel.set_xlabel("")  # type: ignore
    ax_rel.set_ylabel("")  # type: ignore
    fig_rel.tight_layout()

    fig_abs, ax_abs = plt.subplots(figsize=(5, 2))  # type: ignore
    sns.heatmap(  # type: ignore
        pivot_corr,
        annot=True,
        cmap="coolwarm",
        cbar=True,
        ax=ax_abs,
        vmin=-1,
        vmax=1,
    )
    ax_abs.set_xlabel("")  # type: ignore
    ax_abs.set_ylabel("")  # type: ignore
    fig_abs.tight_layout()
    correlations = fig_rel, fig_abs

    return results, results_abusivisms, correlations, label_map, color_map
