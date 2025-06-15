from typing import TypedDict

import pandas as pd

ZoneParamsMapping = dict[str, list[str]]


class StatsData(TypedDict):
    """
    TypedDict for the statistics data structure.
    """

    events_data: pd.DataFrame
    multe_data: pd.DataFrame
    poi_data: pd.DataFrame
    zone_params: ZoneParamsMapping
