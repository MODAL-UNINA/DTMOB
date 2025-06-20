from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast, get_args

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

ForecastDataType = Literal["transactions", "amount", "roads"]
FORECAST_DATA_TYPES = cast(list[ForecastDataType], get_args(ForecastDataType))


class ForecastModelArgs(TypedDict):
    num_nodes: int
    node_dim: int
    input_len: int
    input_dim: int
    embed_dim: int
    output_len: int
    num_layer: int
    temp_dim_tid: int
    temp_dim_diw: int
    time_of_day_size: int
    day_of_week_size: int
    if_T_i_D: bool
    if_D_i_W: bool
    if_node: bool
    if_poi: bool
    if_gps: bool
    num_poi_types: int
    exogenous_dim: NotRequired[int]


class WeatherData(TypedDict):
    prec: pd.DataFrame
    temp: pd.DataFrame
    wind: pd.DataFrame
    humidity: pd.DataFrame


class ForecastIndexMapData(TypedDict):
    parkimeters: dict[str, int]
    roads: dict[str, int]


class OriginalForecastData(TypedDict):
    weather: WeatherData
    events: pd.DataFrame
    hourly: dict[ForecastDataType, pd.DataFrame]
    poi_dists: dict[ForecastDataType, pd.DataFrame]
    poi_categories: dict[ForecastDataType, pd.DataFrame]
    data_scaler: dict[ForecastDataType, MinMaxScaler]
    exog_scaler: dict[ForecastDataType, MinMaxScaler]
    model_args: dict[ForecastDataType, ForecastModelArgs]
    index_map: ForecastIndexMapData


class PreprocessedForecastData(TypedDict):
    hourly_scaled: dict[ForecastDataType, pd.DataFrame]
    exog_scaled: dict[ForecastDataType, pd.DataFrame]
    poi_tensor: dict[ForecastDataType, FloatArray]
    mask: dict[ForecastDataType, BoolArray]


class ForecastData(TypedDict):
    forecast_data: OriginalForecastData
    preprocessed_data: PreprocessedForecastData
    data_path: Path
