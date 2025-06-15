from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast, get_args

import pandas as pd
import torch
from sklearn.pipeline import Pipeline

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
