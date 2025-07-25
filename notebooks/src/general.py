import datetime
from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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


class ZoneDictZoneData(TypedDict):
    parcometro: list[int]
    stalli: list[int]
    camera_ztl: list[str]
    strade: list[int]
    strade_name: list[str]


ZoneDictZoneDataMapping = dict[str, ZoneDictZoneData]


class AvailableDatesData(TypedDict):
    min_date: datetime.date
    max_date: datetime.date


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
