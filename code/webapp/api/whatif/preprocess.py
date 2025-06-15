from typing import Any

import pandas as pd

from .data import (
    DistanceData,
    PreDistanceData,
    WhatIfData,
    WhatIfFinalDataMapping,
    WhatIfLoadedData,
    WhatIfPCoordinatesMapping,
    WhatIfRoadMapping,
    WhatIfScenarioData,
    WhatIfScenarioDataMapping,
    WhatIfSCoordinatesMapping,
)


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
