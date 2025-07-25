import random
from dataclasses import dataclass
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from common.generation.models import Encoder, Generator, ModelArgs
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from .general import ZoneDictZoneDataMapping
from .utils.error_status import ErrorStatus

WhatIfScenarioType = Literal["1st", "2nd", "3rd"]
SCENARIOS = cast(list[WhatIfScenarioType], get_args(WhatIfScenarioType))

WhatIfDataKind = Literal["parkingmeter", "parkingslot"]
DATA_KINDS = cast(list[WhatIfDataKind], get_args(WhatIfDataKind))

SCENARIO_KIND_MAP: dict[WhatIfScenarioType, list[WhatIfDataKind]] = {
    "1st": ["parkingmeter", "parkingslot"],
    "2nd": ["parkingslot"],
    "3rd": ["parkingslot"],
}


class WhatIfFinalData(TypedDict):
    lat: float
    lng: float
    data: "pd.Series[pd.Float64Dtype]"
    id_strada: int


WhatIfFinalDataMapping = dict[float, WhatIfFinalData]

WhatIfRoadMapping = dict[int, "pd.Series[pd.Int64Dtype]"]


class WhatIfDataDict(TypedDict):
    cond: torch.Tensor
    data: torch.Tensor
    start_date: str
    end_date: str


WhatIfPCoordinatesMapping = dict[int, tuple[int, int]]
WhatIfSCoordinatesMapping = dict[int, tuple[int, int]]


class WhatIfScenarioData(TypedDict):
    final_parkingmeters: NotRequired[WhatIfFinalDataMapping]
    final_parkingslots: WhatIfFinalDataMapping
    p_coordinates: NotRequired[WhatIfPCoordinatesMapping]
    s_coordinates: WhatIfSCoordinatesMapping
    p_scaler: NotRequired[Pipeline]
    s_scaler: Pipeline
    road_dict: NotRequired[WhatIfRoadMapping]
    weather_data: NotRequired[pd.DataFrame]


WhatIfScenarioDataMapping = dict[WhatIfScenarioType, WhatIfScenarioData]

WhatIfVicinityData = dict[str, list[str]]


PreDistanceData = dict[str, dict[str, float]]
DistanceData = dict[int, dict[int, float]]


class WhatIfLoadedData(TypedDict):
    scenarios: WhatIfScenarioDataMapping
    dict_zone: WhatIfVicinityData
    distances_p: DistanceData
    distances_s: DistanceData
    start_date: str
    end_date: str


class WhatIfData(TypedDict):
    data: WhatIfLoadedData
    data_path: Path


FloatData = np.float32
FloatArray = NDArray[FloatData]


def check_dates(out_data_scenarios: WhatIfScenarioDataMapping) -> tuple[str, str]:
    mins_dates: list[pd.Timestamp] = []
    maxs_dates: list[pd.Timestamp] = []
    for scenario, scenario_data in out_data_scenarios.items():
        if "final_parkingmeters" in scenario_data:
            final_parkingmeters = scenario_data["final_parkingmeters"]
            mins_dates_: list[pd.Timestamp] = []
            maxs_dates_: list[pd.Timestamp] = []
            for _, inner_data in final_parkingmeters.items():
                mins_dates_.append(
                    inner_data["data"].index.min(  # type: ignore
                    )
                )
                maxs_dates_.append(
                    inner_data["data"].index.max(  # type: ignore
                    )
                )
            assert len(set(mins_dates_)) == 1, (
                f"Minimum dates for final parkingmeters in scenario {scenario} "
                f"are not equal: {set(mins_dates_)}"
            )
            assert len(set(maxs_dates_)) == 1, (
                f"Maximum dates for final parkingmeters in scenario {scenario} "
                f"are not equal: {set(maxs_dates_)}"
            )
            mins_dates.append(mins_dates_[0])
            maxs_dates.append(maxs_dates_[0])
        if "final_parkingslots" in scenario_data:
            final_parkingslots = scenario_data["final_parkingslots"]
            mins_dates_: list[pd.Timestamp] = []
            maxs_dates_: list[pd.Timestamp] = []
            for _, inner_data in final_parkingslots.items():
                mins_dates_.append(
                    inner_data["data"].index.min()  # type: ignore
                )
                maxs_dates_.append(
                    inner_data["data"].index.max()  # type: ignore
                )
            assert len(set(mins_dates_)) == 1, (
                f"Minimum dates for final parkingmeters in scenario {scenario} "
                f"are not equal: {set(mins_dates_)}"
            )
            assert len(set(maxs_dates_)) == 1, (
                f"Maximum dates for final parkingmeters in scenario {scenario} "
                f"are not equal: {set(maxs_dates_)}"
            )
            mins_dates.append(mins_dates_[0])
            maxs_dates.append(maxs_dates_[0])
        if "road_dict" in scenario_data:
            road_dict = scenario_data["road_dict"]
            mins_dates_: list[pd.Timestamp] = []
            maxs_dates_: list[pd.Timestamp] = []
            for _, inner_data in road_dict.items():
                mins_dates_.append(
                    inner_data.index.min()  # type: ignore
                )
                maxs_dates_.append(
                    inner_data.index.max()  # type: ignore
                )
            assert len(set(mins_dates_)) == 1, (
                f"Minimum dates for final parkingmeters in scenario {scenario} "
                f"are not equal: {set(mins_dates_)}"
            )
            assert len(set(maxs_dates_)) == 1, (
                f"Maximum dates for final parkingmeters in scenario {scenario} "
                f"are not equal: {set(maxs_dates_)}"
            )
            mins_dates.append(mins_dates_[0])
            maxs_dates.append(maxs_dates_[0])
        if "weather_data" in scenario_data:
            weather_data = scenario_data["weather_data"]
            mins_dates.append(
                weather_data.index.min()  # type: ignore
            )
            maxs_dates.append(
                weather_data.index.max()  # type: ignore
            )

    assert mins_dates, "No minimum dates found in scenarios"
    assert maxs_dates, "No maximum dates found in scenarios"
    assert len(set(mins_dates)) == 1, (
        f"Minimum dates across scenarios are not equal: {set(mins_dates)}"
    )
    assert len(set(maxs_dates)) == 1, (
        f"Maximum dates across scenarios are not equal: {set(maxs_dates)}"
    )
    start_date = mins_dates[0].strftime("%Y-%m-%d")
    end_date = (maxs_dates[0] - pd.Timedelta(days=6)).strftime("%Y-%m-%d")

    return start_date, end_date


def preprocess_distances_p(data: PreDistanceData) -> DistanceData:
    return {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in data.items()}


def preprocess_distances_s(data: PreDistanceData) -> DistanceData:
    return {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in data.items()}


def postprocess(data: dict[str, Any]) -> WhatIfLoadedData:
    from typing import cast

    weather_data = cast(pd.DataFrame, data.pop("weather_data__3rd"))
    weather_data.index = pd.to_datetime(  # type: ignore
        weather_data.index, format="%Y-%m-%d %H:%M:%S"
    )

    out_data_scenarios = WhatIfScenarioDataMapping(
        {
            "1st": WhatIfScenarioData(
                final_parkingmeters=WhatIfFinalDataMapping(
                    data.pop("final_parkingmeters__1st")
                ),
                final_parkingslots=WhatIfFinalDataMapping(
                    data.pop("final_parkingslots__1st")
                ),
                p_coordinates=WhatIfPCoordinatesMapping(data.pop("p_coordinates__1st")),
                s_coordinates=WhatIfSCoordinatesMapping(data.pop("s_coordinates__1st")),
                p_scaler=data.pop("p_scaler__1st"),
                s_scaler=data.pop("s_scaler__1st"),
            ),
            "2nd": WhatIfScenarioData(
                final_parkingslots=WhatIfFinalDataMapping(
                    data.pop("final_parkingslots__2nd")
                ),
                s_coordinates=WhatIfSCoordinatesMapping(data.pop("s_coordinates__2nd")),
                s_scaler=data.pop("s_scaler__2nd"),
                road_dict=WhatIfRoadMapping(data.pop("road_dict__2nd")),
            ),
            "3rd": WhatIfScenarioData(
                final_parkingmeters=WhatIfFinalDataMapping(
                    data.pop("final_parkingmeters__3rd")
                ),
                final_parkingslots=WhatIfFinalDataMapping(
                    data.pop("final_parkingslots__3rd")
                ),
                p_coordinates=WhatIfPCoordinatesMapping(data.pop("p_coordinates__3rd")),
                s_coordinates=WhatIfSCoordinatesMapping(data.pop("s_coordinates__3rd")),
                p_scaler=data.pop("p_scaler__3rd"),
                s_scaler=data.pop("s_scaler__3rd"),
                weather_data=weather_data,
            ),
        }
    )

    start_date, end_date = check_dates(out_data_scenarios)

    distances_p = preprocess_distances_p(cast(PreDistanceData, data["distances_p"]))
    distances_s = preprocess_distances_s(cast(PreDistanceData, data["distances_s"]))

    return WhatIfLoadedData(
        scenarios=out_data_scenarios,
        dict_zone=data["dict_zone"],
        distances_p=distances_p,
        distances_s=distances_s,
        start_date=start_date,
        end_date=end_date,
    )


def preprocess(data: dict[str, Any]) -> WhatIfData:
    data_path = data["data_path"]
    data_loaded = data["data"]

    data_sub_ = postprocess(data_loaded)

    # Create a WhatIfSubData instance and add it to the output data
    return WhatIfData(
        data=data_sub_,
        data_path=data_path,
    )


def get_week_range(date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the start and end dates of the week containing the given date.
    The week starts on Monday and ends on Sunday.
    """
    start_date = date - pd.Timedelta(days=date.weekday())
    end_date = start_date + pd.Timedelta(days=6)
    return start_date, end_date


def is_data_kind_valid(kind: WhatIfDataKind, scenario: WhatIfScenarioType) -> bool:
    return kind in SCENARIO_KIND_MAP[scenario]


@dataclass
class Scenario1Params:
    final_parkingmeters: WhatIfFinalDataMapping
    final_parkingslots: WhatIfFinalDataMapping
    p_coords: WhatIfPCoordinatesMapping
    s_coords: WhatIfSCoordinatesMapping


@dataclass
class Scenario2Params:
    final_parkingslots: WhatIfFinalDataMapping
    s_coords: WhatIfSCoordinatesMapping
    road_dict: WhatIfRoadMapping


@dataclass
class Scenario3Params:
    final_parkingmeters: WhatIfFinalDataMapping
    final_parkingslots: WhatIfFinalDataMapping
    p_coords: WhatIfPCoordinatesMapping
    s_coords: WhatIfSCoordinatesMapping
    weather_data: pd.DataFrame


def build_data_dict1(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    scnario_params: Scenario1Params,
) -> WhatIfDataDict:
    import numpy as np

    final_parkingmeters = scnario_params.final_parkingmeters
    final_parkingslots = scnario_params.final_parkingslots
    p_coords = scnario_params.p_coords
    s_coords = scnario_params.s_coords

    num_grid_points = 100
    n_entities = 2
    timestamp = pd.date_range(
        start=start_date.date(),
        end=end_date.date() + pd.Timedelta(days=1),
        freq="4H",
        inclusive="left",
    )
    matrix = np.zeros((num_grid_points, num_grid_points, n_entities, len(timestamp)))
    for key in s_coords.keys():
        (lat_index, lon_index) = s_coords[key]
        matrix[lat_index, lon_index, 0, :] = final_parkingslots[key]["data"].loc[
            timestamp
        ]
    for key in p_coords.keys():
        (lat_index, lon_index) = p_coords[key]
        matrix[lat_index, lon_index, 1, :] = final_parkingmeters[key]["data"].loc[
            timestamp
        ]
    data = torch.tensor(matrix, dtype=torch.float32).permute(3, 2, 0, 1).unsqueeze(0)
    cond = (data != 0).to(torch.float32)

    return WhatIfDataDict(
        cond=cond,
        data=data,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )


def build_data_dict2(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    scenario_params: Scenario2Params,
) -> WhatIfDataDict:
    import numpy as np

    final_parkingslots = scenario_params.final_parkingslots
    s_coords = scenario_params.s_coords
    road_dict = scenario_params.road_dict

    num_grid_points = 100
    n_entities = 1
    timestamp = pd.date_range(
        start=start_date.date(),
        end=end_date.date() + pd.Timedelta(days=1),
        freq="4H",
        inclusive="left",
    )
    matrix = np.zeros((num_grid_points, num_grid_points, n_entities, len(timestamp)))
    roads = np.zeros((num_grid_points, num_grid_points, n_entities, len(timestamp)))
    for key in s_coords.keys():
        (lat_index, lon_index) = s_coords[key]
        matrix[lat_index, lon_index, 0, :] = final_parkingslots[key]["data"].loc[
            timestamp
        ]
    for key in road_dict.keys():
        for key2 in s_coords.keys():
            if final_parkingslots[key2]["id_strada"] == key:
                (lat_index, lon_index) = s_coords[key2]
                roads[lat_index, lon_index, 0, :] = road_dict[key].loc[timestamp]

    data = torch.tensor(matrix, dtype=torch.float32).permute(3, 2, 0, 1).unsqueeze(0)
    roads = torch.tensor(roads, dtype=torch.float32).permute(3, 2, 0, 1).unsqueeze(0)
    cond = (data != 0).to(torch.float32)
    cond = torch.cat((cond, roads), dim=2)
    return WhatIfDataDict(
        cond=cond,
        data=data,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )


def build_data_dict3(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    scenario_params: Scenario3Params,
) -> WhatIfDataDict:
    import numpy as np

    final_parkingmeters = scenario_params.final_parkingmeters
    final_parkingslots = scenario_params.final_parkingslots
    p_coords = scenario_params.p_coords
    s_coords = scenario_params.s_coords
    weather_data = scenario_params.weather_data

    num_grid_points = 100
    n_entities = 2
    timestamp = pd.date_range(
        start=start_date.date(),
        end=end_date.date() + pd.Timedelta(days=1),
        freq="4H",
        inclusive="left",
    )
    matrix = np.zeros((num_grid_points, num_grid_points, n_entities, len(timestamp)))
    meteo = np.zeros((num_grid_points, num_grid_points, 1, len(timestamp)))
    for key in s_coords.keys():
        (lat_index, lon_index) = s_coords[key]
        matrix[lat_index, lon_index, 0, :] = final_parkingslots[key]["data"].loc[
            timestamp
        ]
        meteo[lat_index, lon_index, 0, :] = weather_data.loc[timestamp][
            "precipitation_mask"
        ].values  # type: ignore
    for key in p_coords.keys():
        (lat_index, lon_index) = p_coords[key]
        matrix[lat_index, lon_index, 1, :] = final_parkingmeters[key]["data"].loc[
            timestamp
        ]
        meteo[lat_index, lon_index, 0, :] = weather_data.loc[timestamp][
            "precipitation_mask"
        ].values  # type: ignore
    data = torch.tensor(matrix, dtype=torch.float32).permute(3, 2, 0, 1).unsqueeze(0)
    meteo = torch.tensor(meteo, dtype=torch.float32).permute(3, 2, 0, 1).unsqueeze(0)
    cond = (data != 0).to(torch.float32)
    cond = torch.cat((cond, meteo), dim=2)

    return WhatIfDataDict(
        cond=cond,
        data=data,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )


def build_data_dict(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    scenario: WhatIfScenarioType,
    scenario_params: Scenario1Params | Scenario2Params | Scenario3Params,
) -> WhatIfDataDict:
    """
    Build the data dictionary based on the scenario type and parameters.
    """
    if scenario == "1st":
        assert isinstance(scenario_params, Scenario1Params), (
            "Scenario 1 parameters must be of type Scenario1Params"
        )
        return build_data_dict1(start_date, end_date, scenario_params)
    elif scenario == "2nd":
        assert isinstance(scenario_params, Scenario2Params), (
            "Scenario 2 parameters must be of type Scenario2Params"
        )
        return build_data_dict2(start_date, end_date, scenario_params)
    elif scenario == "3rd":
        assert isinstance(scenario_params, Scenario3Params), (
            "Scenario 3 parameters must be of type Scenario3Params"
        )
        return build_data_dict3(start_date, end_date, scenario_params)
    else:
        raise ValueError(f"Unknown scenario type: {scenario}")


def do_get_generation(
    scenario: WhatIfScenarioType,
    date: pd.Timestamp,
    zone_name: str,
    quantity: int | None,
    data_path: Path,
    data: WhatIfLoadedData,
    zone_dict: ZoneDictZoneDataMapping,
    zones: list[str],
) -> tuple[str, FloatArray, WhatIfDataDict]:
    start_date, end_date = get_week_range(date)

    if quantity is None:
        if scenario == "2nd":
            quantity = 150

    distances_p = data["distances_p"]
    distances_s = data["distances_s"]
    scenario_data = data["scenarios"][scenario]
    final_parkingmeters = scenario_data.get("final_parkingmeters")
    final_parkingslots = scenario_data["final_parkingslots"]
    road_dict = scenario_data.get("road_dict")
    weather_data = scenario_data.get("weather_data")
    parkingmeters_coordinates = scenario_data.get("p_coordinates")
    parkingslots_coordinates = scenario_data.get("s_coordinates")

    if scenario == "1st":
        assert final_parkingmeters is not None, (
            "final_parkingmeters should not be None for the 1st scenario."
        )
        assert parkingmeters_coordinates is not None, (
            "p_coords should not be None for the 1st scenario."
        )
        scenario_params = Scenario1Params(
            final_parkingmeters=final_parkingmeters,
            final_parkingslots=final_parkingslots,
            p_coords=parkingmeters_coordinates,
            s_coords=parkingslots_coordinates,
        )
    elif scenario == "2nd":
        assert road_dict is not None, (
            "road_dict should not be None for the 2nd scenario."
        )
        scenario_params = Scenario2Params(
            final_parkingslots=final_parkingslots,
            s_coords=parkingslots_coordinates,
            road_dict=road_dict,
        )
    else:
        assert final_parkingmeters is not None, (
            "final_parkingmeters should not be None for the 3rd scenario."
        )
        assert parkingmeters_coordinates is not None, (
            "p_coords should not be None for the 3rd scenario."
        )
        assert weather_data is not None, (
            "weather_data should not be None for the 3rd scenario."
        )
        scenario_params = Scenario3Params(
            final_parkingmeters=final_parkingmeters,
            final_parkingslots=final_parkingslots,
            p_coords=parkingmeters_coordinates,
            s_coords=parkingslots_coordinates,
            weather_data=weather_data,
        )

    range_s = f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
    data_key = build_data_dict(
        start_date=start_date,
        end_date=end_date,
        scenario=scenario,
        scenario_params=scenario_params,
    )

    mask = data_key["cond"].clone()
    mask = mask.permute(0, 2, 1, 3, 4)

    p_keys_to_remove: list[int] = []
    s_keys_to_remove: list[int] = []

    if scenario == "1st":
        assert parkingmeters_coordinates is not None, (
            "p_coords should not be None for the 1st scenario."
        )
        assert parkingslots_coordinates is not None, (
            "s_coords should not be None for the 1st scenario."
        )
        adjacent_zones = [zone for zone in zones if zone != zone_name]

        p_keys_to_remove = zone_dict[zone_name]["parcometro"]
        p_keys_to_remove = [int(key) for key in p_keys_to_remove]
        s_keys_to_remove = zone_dict[zone_name]["stalli"]
        s_keys_to_remove = [int(key) for key in s_keys_to_remove]

        p_remaining = [zone_dict[z]["parcometro"] for z in adjacent_zones]
        p_remaining = [item for sublist in p_remaining for item in sublist]
        s_remaining = [zone_dict[z]["stalli"] for z in adjacent_zones]
        s_remaining = [item for sublist in s_remaining for item in sublist]
        delta_p = pd.Series(
            {
                key: sum([np.exp(-distances_p[key][key2]) for key2 in p_keys_to_remove])
                for key in p_remaining
            }
        )

        rho_p = delta_p / (delta_p.max() + 1e-6)

        delta_s = pd.Series(
            {
                key: sum([np.exp(-distances_s[key][key2]) for key2 in s_keys_to_remove])
                for key in s_remaining
            }
        )

        rho_s = delta_s / (delta_s.max() + 1e-6)

        for z in adjacent_zones:
            p_adjust = zone_dict[z]["parcometro"]
            s_adjust = zone_dict[z]["stalli"]

            for p in p_adjust:
                p_mask_zero = (
                    mask[
                        0,
                        1,
                        :,
                        parkingmeters_coordinates[p][0],
                        parkingmeters_coordinates[p][1],
                    ]
                    == 0
                )
                mask[
                    0,
                    1,
                    :,
                    parkingmeters_coordinates[p][0],
                    parkingmeters_coordinates[p][1],
                ] = torch.from_numpy(  # type: ignore
                    np.where(
                        p_mask_zero,
                        0,
                        mask[
                            0,
                            1,
                            :,
                            parkingmeters_coordinates[p][0],
                            parkingmeters_coordinates[p][1],
                        ]
                        + rho_p[p],
                    )
                ).float()

            for s in s_adjust:
                s_mask_zero = (
                    mask[
                        0,
                        0,
                        :,
                        parkingslots_coordinates[s][0],
                        parkingslots_coordinates[s][1],
                    ]
                    == 0
                )
                mask[
                    0,
                    0,
                    :,
                    parkingslots_coordinates[s][0],
                    parkingslots_coordinates[s][1],
                ] = torch.from_numpy(  # type: ignore
                    np.where(
                        s_mask_zero,
                        0,
                        mask[
                            0,
                            0,
                            :,
                            parkingslots_coordinates[s][0],
                            parkingslots_coordinates[s][1],
                        ]
                        + rho_s[s],
                    )
                ).float()

        for key in p_keys_to_remove:
            lat, lon = parkingmeters_coordinates[key]
            mask[0, 1, :, lat, lon] = 0

        for key in s_keys_to_remove:
            lat, lon = parkingslots_coordinates[key]
            mask[0, 0, :, lat, lon] = 0
    if scenario == "2nd":
        assert parkingslots_coordinates is not None, (
            "s_coords should not be None for the 2nd scenario."
        )
        assert quantity is not None, "quantity should not be None for the 2nd scenario."

        s_keys = zone_dict[zone_name]["stalli"]
        s_keys = [int(key) for key in s_keys]

        for key in s_keys:
            lat, lon = parkingslots_coordinates[key]
            mask[0, 1, :, lat, lon] += quantity

        lat_indices = [lat for lat, _ in parkingslots_coordinates.values()]
        lon_indices = [lon for _, lon in parkingslots_coordinates.values()]

        mask_zero = mask[:, 1, :, lat_indices, lon_indices] == 0
        mask[:, 1, :, lat_indices, lon_indices] = torch.from_numpy(  # type: ignore
            np.where(
                mask_zero,
                0,
                torch.log(1 / (mask[:, 1, :, lat_indices, lon_indices] + 1)),
            )
        ).float()
    if scenario == "3rd":
        assert parkingmeters_coordinates is not None, (
            "p_coords should not be None for the 3rd scenario."
        )
        assert parkingslots_coordinates is not None, (
            "s_coords should not be None for the 3rd scenario."
        )

        for slot in parkingslots_coordinates.keys():
            lat, lon = (
                parkingslots_coordinates[slot][0],
                parkingslots_coordinates[slot][1],
            )
            mask[0, 2, :18, lat, lon] = 1
            mask[0, 2, 18:, lat, lon] = 0
        for park in parkingmeters_coordinates.keys():
            lat, lon = (
                parkingmeters_coordinates[park][0],
                parkingmeters_coordinates[park][1],
            )
            mask[0, 2, :18, lat, lon] = 1
            mask[0, 2, 18:, lat, lon] = 0

    import torch.multiprocessing as mp

    # from datetime import datetime

    mp.set_start_method("spawn", force=True)

    manager = mp.Manager()
    return_dict: "DictProxy[str, FloatArray]" = manager.dict()

    p = mp.Process(
        target=get_generation,
        args=(
            return_dict,
            scenario,
            mask,
            parkingmeters_coordinates,
            parkingslots_coordinates,
            p_keys_to_remove,
            s_keys_to_remove,
            data_path,
        ),
    )
    p.start()
    p.join()

    return range_s, return_dict["generation"], data_key


def _load_models(
    data_path: Path, scenario: WhatIfScenarioType, device: torch.device
) -> tuple[Generator, Encoder]:
    if scenario not in SCENARIOS:
        raise ValueError("Invalid scenario")

    latent_dim = 100
    kernel_size = 3
    padding = 1
    input_dim = 2 if scenario != "2nd" else 1
    cond_dim = 2 if scenario != "3rd" else 3
    hidden_dim = 32 if scenario != "3rd" else 64
    horizon = 6 * 7
    grid_size = 100

    model_args = ModelArgs(
        input_dim=input_dim,
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        padding=padding,
        horizon=horizon,
        grid_size=grid_size,
        use_proximity=False,
    )

    generator = Generator(model_args).to(device)
    encoder = Encoder(model_args).to(device)

    models_path = data_path / f"{scenario}" / "models"
    generator.load_state_dict(
        torch.load(  # type: ignore
            models_path / "generator.pth",
            map_location=device,
            weights_only=True,
        )
    )
    encoder.load_state_dict(
        torch.load(  # type: ignore
            models_path / "encoder_cond.pth",
            map_location=device,
            weights_only=True,
        )
    )
    generator.eval()
    encoder.eval()

    return generator, encoder


def _generation(
    encoder: Encoder, generator: Generator, mask: torch.Tensor, device: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        z_gen, _, _, indices, map = encoder(mask.to(device))
        noise = torch.randn(z_gen.shape[0], 100).to(device)
        z_gen = torch.cat([noise, z_gen], dim=1)
        output = generator(z_gen, map, indices)
    return output


def get_generation(
    return_dict: "DictProxy[str, FloatArray]",
    scenario: WhatIfScenarioType,
    mask: torch.Tensor,
    parkingmeters_coordinates: WhatIfPCoordinatesMapping | None,
    parkingslots_coordinates: WhatIfSCoordinatesMapping | None,
    p_keys_to_remove: list[int],
    s_keys_to_remove: list[int],
    models_dir: Path,
) -> None:
    import os
    from typing import cast

    # Seed params
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator, encoder = _load_models(models_dir, scenario, device)
    output_t = _generation(encoder, generator, mask, device)

    if scenario == "1st":
        assert parkingmeters_coordinates is not None, (
            "p_coords should not be None for the 1st scenario."
        )
        assert parkingslots_coordinates is not None, (
            "s_coords should not be None for the 1st scenario."
        )

        for k in p_keys_to_remove:
            lat, lon = parkingmeters_coordinates[k]
            output_t[0, 1, :, lat, lon] = 0
        for k in s_keys_to_remove:
            lat, lon = parkingslots_coordinates[k]
            output_t[0, 0, :, lat, lon] = 0
    output = cast(
        FloatArray,
        output_t.numpy(  # type: ignore
            force=True
        ),
    )

    return_dict["generation"] = output


def _get_dfs_parkingmeter(
    data_real: FloatArray,
    output: FloatArray,
    coordinates_parkingmeters: WhatIfPCoordinatesMapping,
    scaler_parkingmeters: Pipeline,
) -> tuple[
    FloatArray, FloatArray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    from typing import cast

    p_data = cast(FloatArray, data_real[0, 1])
    p_gen = cast(FloatArray, output[0, 1])
    data_p: list[FloatArray] = []
    gen_p: list[FloatArray] = []
    for key in coordinates_parkingmeters.keys():
        lat, lon = coordinates_parkingmeters[key]
        data_p.append(p_data[:, lat, lon])
        gen_p.append(p_gen[:, lat, lon])
    final_data_p = np.stack(data_p, axis=1)
    final_gen_p = np.stack(gen_p, axis=1)
    p_data_df = pd.DataFrame(
        final_data_p, columns=list(coordinates_parkingmeters.keys())
    )
    p_gen_df = pd.DataFrame(final_gen_p, columns=list(coordinates_parkingmeters.keys()))
    p_data_df.columns = [int(i) for i in coordinates_parkingmeters.keys()]
    p_gen_df.columns = [int(i) for i in coordinates_parkingmeters.keys()]
    p_data_df = p_data_df.reindex(  # type: ignore
        sorted(p_data_df.columns), axis=1
    )
    p_gen_df = p_gen_df.reindex(  # type: ignore
        sorted(p_gen_df.columns), axis=1
    )

    p_data_df_old = p_data_df.copy(deep=True)
    p_gen_df_old = p_gen_df.copy(deep=True)
    p_data_df = pd.DataFrame(
        scaler_parkingmeters.inverse_transform(  # type: ignore
            p_data_df
        ),
        columns=p_data_df.columns,
    )
    p_data_df = p_data_df.round(  # type: ignore
        0
    )
    p_gen_df = pd.DataFrame(
        scaler_parkingmeters.inverse_transform(  # type: ignore
            p_gen_df
        ),
        columns=p_gen_df.columns,
    )

    return p_data, p_gen, p_data_df, p_gen_df, p_data_df_old, p_gen_df_old


def _get_dfs_parkingslot(
    data_real: FloatArray,
    output: FloatArray,
    coordinates_slots: WhatIfSCoordinatesMapping,
    scaler_slots: Pipeline,
) -> tuple[
    FloatArray, FloatArray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    from typing import cast

    s_data = cast(FloatArray, data_real[0, 0])
    s_gen = cast(FloatArray, output[0, 0])
    data_s: list[FloatArray] = []
    gen_s: list[FloatArray] = []
    for key in coordinates_slots.keys():
        lat, lon = coordinates_slots[key]
        data_s.append(s_data[:, lat, lon])
        gen_s.append(s_gen[:, lat, lon])
    final_data_s = np.stack(data_s, axis=1)
    final_gen_s = np.stack(gen_s, axis=1)
    s_data_df = pd.DataFrame(
        final_data_s, columns=[int(i) for i in coordinates_slots.keys()]
    )
    s_gen_df = pd.DataFrame(
        final_gen_s, columns=[int(i) for i in coordinates_slots.keys()]
    )
    s_data_df = s_data_df.reindex(  # type: ignore
        sorted(s_data_df.columns), axis=1
    )
    s_gen_df = s_gen_df.reindex(  # type: ignore
        sorted(s_gen_df.columns), axis=1
    )

    s_data_df_old = s_data_df.copy(deep=True)
    s_gen_df_old = s_gen_df.copy(deep=True)
    s_data_df = pd.DataFrame(
        scaler_slots.inverse_transform(  # type: ignore
            s_data_df
        ),
        columns=s_data_df.columns,
    )
    s_data_df = s_data_df.round(  # type: ignore
        0
    )
    s_gen_df = pd.DataFrame(
        scaler_slots.inverse_transform(  # type: ignore
            s_gen_df
        ),
        columns=s_gen_df.columns,
    )

    return s_data, s_gen, s_data_df, s_gen_df, s_data_df_old, s_gen_df_old


class GenerationData(TypedDict):
    selected_zones: list[str]
    p_data: FloatArray | None
    p_gen: FloatArray | None
    p_data_df: pd.DataFrame | None
    p_gen_df: pd.DataFrame | None
    p_data_df_old: pd.DataFrame | None
    p_gen_df_old: pd.DataFrame | None
    s_data: FloatArray | None
    s_gen: FloatArray | None
    s_data_df: pd.DataFrame | None
    s_gen_df: pd.DataFrame | None
    s_data_df_old: pd.DataFrame | None
    s_gen_df_old: pd.DataFrame | None


def prepare_generated_data(
    scenario: WhatIfScenarioType,
    zone_name: str,
    output: FloatArray,
    data_real: FloatArray,
    dictionary: list[str],
    data: WhatIfLoadedData,
) -> GenerationData:
    assert scenario in SCENARIOS

    p_data = None
    p_gen = None
    s_data = None
    s_gen = None
    p_data_df = None
    p_gen_df = None
    p_data_df_old = None
    p_gen_df_old = None
    s_data_df = None
    s_gen_df = None
    s_data_df_old = None
    s_gen_df_old = None

    scenario_data = data["scenarios"][scenario]
    p_coordinates = scenario_data.get("p_coordinates")
    s_coordinates = scenario_data["s_coordinates"]
    p_scaler = scenario_data.get("p_scaler")
    s_scaler = scenario_data["s_scaler"]

    if scenario == "1st":
        assert p_coordinates is not None, (
            "p_coordinates should not be None for the 1st scenario."
        )
        assert p_scaler is not None, "p_scaler should not be None for the 1st scenario."
        selected_zones = dictionary
        p_data, p_gen, p_data_df, p_gen_df, p_data_df_old, p_gen_df_old = (
            _get_dfs_parkingmeter(
                data_real,
                output,
                p_coordinates,
                p_scaler,
            )
        )

        s_data, s_gen, s_data_df, s_gen_df, s_data_df_old, s_gen_df_old = (
            _get_dfs_parkingslot(
                data_real,
                output,
                s_coordinates,
                s_scaler,
            )
        )

    elif scenario == "2nd":
        selected_zones = [zone_name] + dictionary
        s_data, s_gen, s_data_df, s_gen_df, s_data_df_old, s_gen_df_old = (
            _get_dfs_parkingslot(
                data_real,
                output,
                s_coordinates,
                s_scaler,
            )
        )

    elif scenario == "3rd":
        selected_zones = dictionary
        s_data, s_gen, s_data_df, s_gen_df, s_data_df_old, s_gen_df_old = (
            _get_dfs_parkingslot(
                data_real,
                output,
                s_coordinates,
                s_scaler,
            )
        )
    else:
        raise ValueError("Invalid scenario")

    return GenerationData(
        selected_zones=selected_zones,
        p_data=p_data,
        p_gen=p_gen,
        p_data_df=p_data_df,
        p_gen_df=p_gen_df,
        p_data_df_old=p_data_df_old,
        p_gen_df_old=p_gen_df_old,
        s_data=s_data,
        s_gen=s_gen,
        s_data_df=s_data_df,
        s_gen_df=s_gen_df,
        s_data_df_old=s_data_df_old,
        s_gen_df_old=s_gen_df_old,
    )


def create_heatmap(
    scenario: WhatIfScenarioType,
    start_date: str,
    out_data: GenerationData,
    hour_slots: dict[int, str],
    kind: WhatIfDataKind,
    which_data: str,
    selected_day: str,
) -> list[Figure] | ErrorStatus:
    """
    Creates heatmap plots for parking data with a yellow color intensity for high values and dark color for low values.
    Adds a color bar to each plot, and saves the images.
    """

    from scipy.ndimage import gaussian_filter  # type: ignore

    def apply_aura_effect(data: FloatArray, sigma: float = 1.5) -> FloatArray:
        """Apply Gaussian blur to simulate aura effect"""
        from typing import cast

        return cast(
            FloatArray,
            gaussian_filter(  # type: ignore
                data, sigma=sigma
            ),
        )

    from matplotlib.colors import LinearSegmentedColormap

    colors = [(0, "white"), (1, "darkblue")]
    custom_cmap = LinearSegmentedColormap.from_list("white_to_darkblue", colors, N=256)

    if scenario not in SCENARIOS:
        return ErrorStatus(error="Invalid scenario")

    assert is_data_kind_valid(kind, scenario)

    start_date_date = pd.Timestamp(start_date)
    selected_day_date = pd.Timestamp(selected_day)

    idx_start_data = (selected_day_date - start_date_date).days * 6
    idx_end_data = idx_start_data + 6

    data_plot_s = ""

    data_plot_s += "p_" if kind == "parkingmeter" else "s_"
    data_plot_s += "data" if which_data == "real" else "gen"

    data_plot = out_data[data_plot_s]
    assert data_plot is not None, "Data plot is None"

    figs: list[Figure] = []

    for it, t in enumerate(range(idx_start_data, idx_end_data)):
        fig, ax = plt.subplots()  # type: ignore
        aura_data = apply_aura_effect(data_plot[t], sigma=1.5)
        ax.imshow(  # type: ignore
            data_plot[t], cmap=custom_cmap, interpolation="nearest", vmin=0, vmax=1
        )
        ax.imshow(  # type: ignore
            aura_data, cmap="Blues", interpolation="nearest", alpha=0.4
        )
        ax.invert_yaxis()
        ax.set_xticks(  # type: ignore
            []
        )
        ax.set_yticks(  # type: ignore
            []
        )
        ax.set_title(  # type: ignore
            f"Hour slot: {hour_slots[it]}"
        )
        fig.set_size_inches(10, 6)

        figs.append(fig)

    return figs


def create_histograms_with_inset(
    scenario: WhatIfScenarioType, out_data: GenerationData, kind: WhatIfDataKind
) -> Figure | ErrorStatus:
    """
    Plot a histogram with an inset t-SNE visualization for parking meter data.
    """
    from typing import cast

    def flatten_data(df: pd.DataFrame) -> FloatArray:
        from typing import cast

        return cast(
            FloatArray,
            df.to_numpy().flatten(),  # type: ignore
        )

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # type: ignore
    from sklearn.manifold import TSNE

    assert is_data_kind_valid(kind, scenario)

    data_plot_s = "p_" if kind == "parkingmeter" else "s_"
    real_s = data_plot_s + "data_df_old"
    gen_s = data_plot_s + "gen_df_old"

    df_real_data = out_data[real_s]
    df_gen_data = out_data[gen_s]

    assert df_real_data is not None, "Real data DataFrame is None"
    assert df_gen_data is not None, "Generated data DataFrame is None"

    real_data_plot = flatten_data(df_real_data)
    gen_data_plot = flatten_data(df_gen_data)

    bin_edges = list(
        np.linspace(
            0,
            max(np.max(real_data_plot), np.max(gen_data_plot)),
            20,
        )
    )

    fig, ax_main = plt.subplots()  # type: ignore

    ax_main.hist(  # type: ignore
        real_data_plot,
        bins=bin_edges,
        alpha=0.5,
        color="orange",
        label="Real Data",
        density=True,
    )
    ax_main.hist(  # type: ignore
        gen_data_plot,
        bins=bin_edges,
        alpha=0.5,
        color="green",
        label="Generated Data",
        density=True,
    )
    ax_main.set_xlim(0.0, 1.1)

    tsne = TSNE(n_components=2, random_state=42)
    p_combined_data = cast(
        FloatArray,
        np.vstack(
            (
                df_real_data.values,  # type: ignore
                df_gen_data.values,  # type: ignore
            )
        ),
    )
    p_tsne = tsne.fit_transform(  # type: ignore
        p_combined_data
    )

    inset_ax = cast(
        Axes,
        inset_axes(  # type: ignore
            parent_axes=ax_main,
            width="45%",
            height="45%",
            borderpad=1,
        ),
    )
    inset_ax.scatter(  # type: ignore
        p_tsne[: len(df_real_data), 0],  # type: ignore
        p_tsne[: len(df_real_data), 1],  # type: ignore
        s=10,
        color="orange",
        label="Real Data",
        alpha=0.5,
    )
    inset_ax.scatter(  # type: ignore
        p_tsne[len(df_real_data) :, 0],  # type: ignore
        p_tsne[len(df_real_data) :, 1],  # type: ignore
        s=10,
        color="green",
        label="Generated Data",
        alpha=0.5,
    )

    inset_ax.text(  # type: ignore
        0.5,
        -0.04,
        "t-SNE",
        fontsize=10,
        ha="center",
        va="top",
        transform=inset_ax.transAxes,  # type: ignore
    )
    inset_ax.set_xticks(  # type: ignore
        []
    )
    inset_ax.set_yticks(  # type: ignore
        []
    )
    fig.set_size_inches(10, 6)
    return fig


def create_radar_chart_map(
    scenario: WhatIfScenarioType,
    out_data: GenerationData,
    kind: WhatIfDataKind,
    zone_dict: ZoneDictZoneDataMapping,
) -> Figure | ErrorStatus:
    """
    Plot a Radar Chart (Spider Plot) for real and generated data for each zone.
    Each element in real_data and gen_data is a time series of occupancy.
    """

    assert is_data_kind_valid(kind, scenario)

    data_plot_s = "p_" if kind == "parkingmeter" else "s_"

    zone_dict_kind = "parcometro" if kind == "parkingmeter" else "stalli"

    df_real_data = out_data[data_plot_s + "data_df"]
    df_gen_data = out_data[data_plot_s + "gen_df"]

    assert df_real_data is not None, "Real data DataFrame is None"
    assert df_gen_data is not None, "Generated data DataFrame is None"

    zones = [zone for zone in zone_dict.keys() if zone != "all_map"]

    real_data: list["pd.Series[pd.Float32Dtype]"] = []
    gen_data: list["pd.Series[pd.Float32Dtype]"] = []
    for key in zones:
        p_keys = zone_dict[key][zone_dict_kind]
        parc_data_df = df_real_data[p_keys]
        parc_gen_df = df_gen_data[p_keys]

        real_data.append(
            parc_data_df.mean(  # type: ignore
                axis=1
            )
        )
        gen_data.append(
            parc_gen_df.mean(  # type: ignore
                axis=1
            )
        )

    real_data_avg = [np.mean(series) for series in real_data]
    gen_data_avg = [np.mean(series) for series in gen_data]

    N = len(real_data_avg)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))  # type: ignore

    real_data_avg = np.concatenate((real_data_avg, [real_data_avg[0]]))
    gen_data_avg = np.concatenate((gen_data_avg, [gen_data_avg[0]]))

    max_value = float(max(np.max(real_data_avg), np.max(gen_data_avg)))

    ax.plot(  # type: ignore
        angles,
        real_data_avg,
        color="orange",
        linewidth=2,
        linestyle="solid",
        label="Real Data",
        alpha=0.9,
    )
    ax.fill(  # type: ignore
        angles, real_data_avg, color="orange", alpha=0.3
    )

    ax.plot(  # type: ignore
        angles,
        gen_data_avg,
        color="green",
        linewidth=2,
        linestyle="solid",
        label="Generated Data",
        alpha=0.9,
    )
    ax.fill(  # type: ignore
        angles, gen_data_avg, color="green", alpha=0.3
    )

    ax.legend(  # type: ignore
        loc="upper right",
        fontsize=7,
        bbox_to_anchor=(1.1, 1.1),
    )

    ax.set_xticks(  # type: ignore
        angles[:-1]
    )
    labels = [f"Zone {i}" for i in range(0, N)]

    ax.set_xticklabels(  # type: ignore
        []
    )
    for i, label in enumerate(labels):
        angle = angles[i]
        angle_deg = np.degrees(angle)

        rotation = angle_deg if angle_deg <= 90 or angle_deg >= 270 else angle_deg + 180
        ha = "center"

        ax.text(  # type: ignore
            angle,
            max_value * 1.1,
            label,
            horizontalalignment=ha,
            verticalalignment="center",
            fontsize=7,
            color="black",
            rotation=rotation,
            rotation_mode="anchor",
        )

    yticks = np.linspace(0, max_value, 10)
    ax.set_yticks(  # type: ignore
        yticks
    )
    ax.set_yticklabels(  # type: ignore
        []
    )
    ax.set_ylim(0, max_value)

    ax.set_facecolor("#F8F9F9")

    fig.tight_layout()
    fig.set_size_inches(10, 6)

    return fig


def create_cumulative_plot(
    scenario: WhatIfScenarioType,
    start_date: str,
    out_data: GenerationData,
    kind: WhatIfDataKind,
    selected_adjacent_zone: str,
    zone_dict: ZoneDictZoneDataMapping,
) -> Figure | ErrorStatus:
    timestamps = pd.date_range(  # type: ignore
        start=f"{start_date} 02:00:00", periods=42, freq="4H"
    )

    assert is_data_kind_valid(kind, scenario)

    data_plot_s = "p_" if kind == "parkingmeter" else "s_"

    zone_dict_kind = "parcometro" if kind == "parkingmeter" else "stalli"

    selected_zones = out_data["selected_zones"]

    df_real_data = out_data[data_plot_s + "data_df"]
    df_gen_data = out_data[data_plot_s + "gen_df"]

    assert df_real_data is not None, "Real data DataFrame is None"
    assert df_gen_data is not None, "Generated data DataFrame is None"

    s_data_plot = pd.DataFrame()
    s_gen_plot = pd.DataFrame()

    if selected_adjacent_zone == "all_map":
        adjacent_zones = selected_zones
        for selected_zone in adjacent_zones:
            if selected_zone == "all_map":
                continue
            selected_kinds = zone_dict[selected_zone][zone_dict_kind]

            s_data_plot = pd.concat([s_data_plot, df_real_data[selected_kinds]], axis=1)
            s_gen_plot = pd.concat([s_gen_plot, df_gen_data[selected_kinds]], axis=1)
    else:
        selected_kinds = zone_dict[selected_adjacent_zone][zone_dict_kind]
        s_data_plot = df_real_data[selected_kinds]
        s_gen_plot = df_gen_data[selected_kinds]

    s_data_plot.index = timestamps
    s_gen_plot.index = timestamps

    s_real_values = np.zeros(s_data_plot.shape[0])
    s_generated_values = np.zeros(s_gen_plot.shape[0])
    for key in s_data_plot.columns:
        s_real_values += s_data_plot[key]
        s_generated_values += s_gen_plot[key]

    fig, ax = plt.subplots(  # type: ignore
    )
    ax.plot(  # type: ignore
        timestamps, s_real_values, label="Real", color="#00509E", lw=2.5
    )
    ax.plot(  # type: ignore
        timestamps,
        s_generated_values,
        label="Generated",
        color="#FF3B3B",
        lw=2.5,
    )
    ax.legend(  # type: ignore
        loc="upper right", fontsize=10
    )
    ax.grid(  # type: ignore
        True, linestyle="--", alpha=0.5, color="gray"
    )
    fig.tight_layout()
    fig.set_size_inches(10, 6)

    return fig
