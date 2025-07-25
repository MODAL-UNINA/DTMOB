import random
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from common.forecasting.models import ModelArgs, Modelcomplete
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL  # type: ignore

from .utils.error_status import ErrorStatus

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


class ZoneDictZoneData(TypedDict):
    parcometro: list[int]
    stalli: list[int]
    camera_ztl: list[str]
    strade: list[int]
    strade_name: list[str]


ZoneDictZoneDataMapping = dict[str, ZoneDictZoneData]


ForecastDataMapping = dict[str, FloatArray]
ForecastPlot1DataType = Literal["transactions", "amount"]
FORECAST_PLOT1_DATA_TYPES = cast(
    list[ForecastPlot1DataType], get_args(ForecastPlot1DataType)
)


class AvailableDatetimes(TypedDict):
    min_date: pd.Timestamp
    max_date: pd.Timestamp


class ForecastDecomposed1Data(TypedDict):
    data: pd.DataFrame
    seasonal: pd.DataFrame
    trend: pd.DataFrame
    residual: pd.DataFrame
    exog: pd.DataFrame
    index: pd.DatetimeIndex


class ForecastDecomposedWithFeaturesData(TypedDict):
    data: NotRequired[FloatArray]
    seasonal: NotRequired[FloatArray]
    trend: NotRequired[FloatArray]
    residual: NotRequired[FloatArray]
    exog: NotRequired[FloatArray]
    index: pd.DatetimeIndex


class ForecastDecomposedData(TypedDict):
    data: FloatArray
    seasonal: FloatArray
    trend: FloatArray
    residual: FloatArray
    exog: FloatArray
    index: pd.DatetimeIndex


def preprocess_data(data: OriginalForecastData) -> PreprocessedForecastData:
    from typing import cast

    import holidays
    from dateutil.easter import easter

    # Weather data
    for data_type in data["weather"]:
        data["weather"][data_type].index = pd.to_datetime(  # type: ignore
            data["weather"][data_type].index  # type: ignore
        )

    weather_data = pd.DataFrame(
        {
            "temperature": data["weather"]["temp"].mean(axis=1),  # type: ignore
            "precipitation": data["weather"]["prec"].mean(axis=1),  # type: ignore
            "wind": data["weather"]["wind"].mean(axis=1),  # type: ignore
            "humidity": data["weather"]["humidity"].mean(axis=1),  # type: ignore
        }
    )

    events_orig = data["events"]
    events_orig.index = pd.to_datetime(  # type: ignore
        events_orig.index
    )

    pois_categories = data["poi_categories"]
    pois_dists = data["poi_dists"]

    hourlies_scaled: dict[ForecastDataType, pd.DataFrame] = {}
    exogs_scaled: dict[ForecastDataType, pd.DataFrame] = {}
    poi_tensors: dict[ForecastDataType, FloatArray] = {}
    masks: dict[ForecastDataType, BoolArray] = {}

    for data_type, hourly in data["hourly"].items():
        hourly.index = pd.to_datetime(  # type: ignore
            hourly.index
        )
        hourly.columns = hourly.columns.astype(float)  # type: ignore

        all_index = cast(
            pd.DatetimeIndex,
            hourly.index,  # type: ignore
        )

        # Events and holidays
        events = events_orig.copy()

        events = events.reindex(  # type: ignore
            pd.date_range(  # type: ignore
                all_index.min(),  # type: ignore
                all_index.max()  # type: ignore
                + pd.Timedelta(days=1)
                - pd.Timedelta(hours=1),
                freq="H",
            )
        )
        events.fillna(  # type: ignore
            method="ffill", inplace=True
        )
        events.fillna(  # type: ignore
            0, inplace=True
        )
        events = events.loc[all_index]

        years = all_index.year.unique()
        it_holidays = pd.to_datetime(  # type: ignore
            [
                d
                for y in years
                for d in holidays.Italy(years=y).keys()  # type: ignore
            ]
        )
        is_holiday = pd.DataFrame(0, index=all_index, columns=["is_holiday"])
        is_holiday.loc[
            is_holiday.index.normalize(  # type: ignore
            ).isin(it_holidays),
            "is_holiday",
        ] = 1

        # Custom holidays
        easter_days = [
            easter(y) + pd.Timedelta(days=i) for y in years for i in range(-3, 2)
        ]
        christmas = pd.to_datetime(  # type: ignore
            np.concatenate(
                [
                    pd.date_range(  # type: ignore
                        f"{y}-12-23", f"{y + 1}-01-06"
                    )
                    for y in years
                ]
            )
        )
        august = pd.to_datetime(  # type: ignore
            np.concatenate(
                [
                    pd.date_range(  # type: ignore
                        f"{y}-08-01", f"{y}-08-31"
                    )
                    for y in years
                ]
            )
        )
        our_holidays = pd.DataFrame(0, index=all_index, columns=["our_holidays"])
        our_holidays.loc[
            our_holidays.index.normalize(  # type: ignore
            ).isin(easter_days + list(christmas) + list(august)),
            "our_holidays",
        ] = 1

        exog_data = pd.concat(
            [weather_data.loc[all_index], events, is_holiday, our_holidays], axis=1
        )

        poi_categories = np.expand_dims(
            cast(
                FloatArray,
                pois_categories[data_type].values,  # type: ignore
            ),
            axis=-1,
        )
        poi_dists = np.expand_dims(
            cast(
                FloatArray,
                pois_dists[data_type].values,  # type: ignore
            ),
            axis=-1,
        )

        # Mask to consider only POIs within 0.5 km from the parking meter
        mask = poi_dists <= 0.5

        poi_dist_masked = poi_dists * mask

        # Normalize distance matrix
        poi_dist_masked = (poi_dist_masked - poi_dist_masked.min()) / (
            poi_dist_masked.max() - poi_dist_masked.min()
        )

        poi_data_ = np.concatenate([poi_categories, poi_dist_masked], axis=-1)

        poi_tensor = np.expand_dims(poi_data_, axis=0)
        mask = np.expand_dims(mask, axis=0)

        data_scaler = data["data_scaler"][data_type]
        hourly_scaled = pd.DataFrame(
            data_scaler.transform(  # type: ignore
                hourly.values  # type: ignore
            ),
            index=hourly.index,
            columns=hourly.columns,  # type: ignore
        )

        exog_scaler = data["exog_scaler"][data_type]
        exog_scaled = pd.DataFrame(
            exog_scaler.transform(  # type: ignore
                exog_data.values  # type: ignore
            ),
            index=exog_data.index,
            columns=exog_data.columns,
        )
        hourlies_scaled[data_type] = hourly_scaled
        exogs_scaled[data_type] = exog_scaled
        poi_tensors[data_type] = poi_tensor
        masks[data_type] = mask

    return PreprocessedForecastData(
        hourly_scaled=hourlies_scaled,
        exog_scaled=exogs_scaled,
        poi_tensor=poi_tensors,
        mask=masks,
    )


def preprocess(all_data: dict[str, Any]) -> ForecastData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    from typing import cast

    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    data_path = all_data["data_path"]
    data = all_data["data"]

    weather = WeatherData(
        prec=cast(pd.DataFrame, data.pop("weather__prec")),
        temp=cast(pd.DataFrame, data.pop("weather__temp")),
        wind=cast(pd.DataFrame, data.pop("weather__wind")),
        humidity=cast(pd.DataFrame, data.pop("weather__humidity")),
    )

    hourlies: dict[ForecastDataType, pd.DataFrame] = {}
    hourlies["transactions"] = data.pop("hourlies__transactions")
    hourlies["amount"] = data.pop("hourlies__amount")
    hourly_roads = cast(pd.DataFrame, data.pop("hourlies__roads"))
    hourly_roads.columns = hourly_roads.columns.astype(int)  # type: ignore
    hourlies["roads"] = hourly_roads

    if not (hourlies["transactions"].index == hourlies["amount"].index).all():
        raise ValueError("Transactions and Amount data must have the same index.")

    poi_dists: dict[ForecastDataType, pd.DataFrame] = {}
    poi_dists["transactions"] = data.pop("poi_dists__parkingmeters")
    poi_dists["amount"] = poi_dists["transactions"].copy()
    poi_dists_roads = cast(pd.DataFrame, data.pop("poi_dists__roads"))
    poi_dists_roads = poi_dists_roads.loc[hourly_roads.columns]
    poi_dists["roads"] = poi_dists_roads

    poi_categories: dict[ForecastDataType, pd.DataFrame] = {}
    poi_categories["transactions"] = data.pop("poi_categories__parkingmeters")
    poi_categories["amount"] = poi_categories["transactions"].copy()
    poi_categories_roads = cast(pd.DataFrame, data.pop("poi_categories__roads"))
    poi_categories_roads = poi_categories_roads.loc[hourly_roads.columns]
    poi_categories["roads"] = poi_categories_roads

    data_scalers: dict[ForecastDataType, MinMaxScaler] = {}
    data_scalers["transactions"] = data.pop("data_scalers__transactions")
    data_scalers["amount"] = data.pop("data_scalers__amount")
    data_scalers["roads"] = data.pop("data_scalers__roads")

    exog_scalers: dict[ForecastDataType, MinMaxScaler] = {}
    exog_scalers["transactions"] = data.pop("exog_scalers__transactions")
    exog_scalers["amount"] = data.pop("exog_scalers__amount")
    exog_scalers["roads"] = data.pop("exog_scalers__roads")

    model_args: dict[ForecastDataType, ForecastModelArgs] = dict()
    model_args["transactions"] = ForecastModelArgs(
        num_nodes=97,
        node_dim=16,
        input_len=24 * 7 * 4,
        input_dim=1,
        embed_dim=512,
        output_len=24 * 7,
        num_layer=1,
        temp_dim_tid=8,
        temp_dim_diw=8,
        time_of_day_size=24,
        day_of_week_size=7,
        if_T_i_D=True,
        if_D_i_W=True,
        if_node=True,
        if_poi=True,
        if_gps=True,
        num_poi_types=7,
        exogenous_dim=13,
    )
    model_args["amount"] = ForecastModelArgs(
        num_nodes=97,
        node_dim=16,
        input_len=24 * 7 * 4,
        input_dim=1,
        embed_dim=512,
        output_len=24 * 7,
        num_layer=1,
        temp_dim_tid=8,
        temp_dim_diw=8,
        time_of_day_size=24,
        day_of_week_size=7,
        if_T_i_D=True,
        if_D_i_W=True,
        if_node=True,
        if_poi=True,
        if_gps=True,
        num_poi_types=7,
        exogenous_dim=13,
    )
    model_args["roads"] = ForecastModelArgs(
        num_nodes=56,
        node_dim=16,
        input_len=24 * 7 * 3,
        input_dim=1,
        embed_dim=256,
        output_len=24 * 7,
        num_layer=1,
        temp_dim_tid=8,
        temp_dim_diw=8,
        time_of_day_size=24,
        day_of_week_size=7,
        if_T_i_D=True,
        if_D_i_W=True,
        if_node=True,
        if_poi=True,
        if_gps=False,
        num_poi_types=7,
        exogenous_dim=13,
    )
    forecast_data = OriginalForecastData(
        weather=weather,
        events=data["events"],
        hourly=hourlies,
        poi_dists=poi_dists,
        poi_categories=poi_categories,
        data_scaler=data_scalers,
        exog_scaler=exog_scalers,
        model_args=model_args,
        index_map=data["index_map"],
    )
    preprocessed_data = preprocess_data(forecast_data)

    return ForecastData(
        forecast_data=forecast_data,
        preprocessed_data=preprocessed_data,
        data_path=data_path,
    )


def add_features(
    data: list[pd.DataFrame | pd.DatetimeIndex],
    add_time_of_day: bool = False,
    add_day_of_week: bool = False,
    steps_per_day: int = 24,
    steps_per_week: int = 7,
) -> FloatArray:
    """
    Add time-of-day and day-of-week features to the input data.

    Args:
    - data (list): List of input data arrays.
    - add_time_of_day (bool): Whether to add time-of-day features (default: None).
    - add_day_of_week (bool): Whether to add day-of-week features (default: None).
    - steps_per_day (int): Number of time steps per day (default: 24).
    - steps_per_week (int): Number of time steps per week (default: 7).

    Returns:
    - np.ndarray: Input data with added time-of-day and day-of-week features.

    """

    data1 = cast(
        FloatArray,
        np.expand_dims(  # type: ignore
            data[0].values,  # type: ignore
            axis=-1,
        ),
    )
    n = int(data[0].shape[1])  # type: ignore
    feature_list = [data1]

    if add_time_of_day:
        # Numerical time_of_day
        tod_index = cast(
            "pd.Index[pd.Float64Dtype]",
            data[0].index.hour / steps_per_day,  # type: ignore
        )
        tod = cast(
            FloatArray,
            np.array(
                tod_index,  # type: ignore
            ),
        )
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # Numerical day_of_week
        dow = cast(
            float,
            data[0].index.dayofweek / steps_per_week,  # type: ignore
        )
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data_with_features = np.concatenate(feature_list, axis=-1)

    return data_with_features


def decompose_data(
    input_len: int,
    data_scaled: pd.DataFrame,
    exog_scaled: pd.DataFrame,
    start_date: pd.Timestamp,
) -> ForecastDecomposedData:
    """
    Decompose the input data into seasonal, trend, and residual components.

    Args:
    - data (DataFrame): Input data to decompose.
    - exog (DataFrame): Exogenous data for the model.

    Returns:
    - dict: Decomposed data with added time-of-day and day-of-week features.

    """
    end_date1 = start_date - pd.Timedelta(hours=1)
    start_date1 = end_date1 - pd.Timedelta(hours=input_len - 1)

    data = data_scaled[start_date1:end_date1]
    exog = exog_scaled[start_date1:end_date1]

    trend_data = pd.DataFrame(columns=data.columns)
    seasonal_data = pd.DataFrame(columns=data.columns)
    residual_data = pd.DataFrame(columns=data.columns)

    for parkingmeters in data.columns:
        result = STL(data[parkingmeters], seasonal=23).fit()  # type: ignore
        (
            trend_data[parkingmeters],
            seasonal_data[parkingmeters],
            residual_data[parkingmeters],
        ) = (result.trend, result.seasonal, result.resid)  # type: ignore

    data_index = cast(pd.DatetimeIndex, data.index)

    data_1 = ForecastDecomposed1Data(
        data=data,
        seasonal=seasonal_data,
        trend=trend_data,
        residual=residual_data,
        exog=exog,
        index=data_index,
    )

    data_with_features: dict[str, FloatArray] = dict()

    data_type = list(data_1.keys())
    data_type.remove("index")
    for key in data_type:
        data_with_features[key] = add_features(
            [data_1[key]], add_time_of_day=True, add_day_of_week=True
        )

        data_with_features["exog"] = cast(
            FloatArray,
            exog.values,  # type: ignore
        )

    data_ = ForecastDecomposedData(
        data=np.expand_dims(data_with_features["data"], axis=0),
        seasonal=np.expand_dims(data_with_features["seasonal"], axis=0),
        trend=np.expand_dims(data_with_features["trend"], axis=0),
        residual=np.expand_dims(data_with_features["residual"], axis=0),
        exog=np.expand_dims(data_with_features["exog"], axis=0),
        index=data_1["index"],
    )

    return data_


def get_date_range(
    data_type: ForecastDataType,
    hourly_scaled_map: dict[ForecastDataType, pd.DataFrame],
    model_args: dict[ForecastDataType, ForecastModelArgs],
) -> AvailableDatetimes | ErrorStatus:
    if data_type not in hourly_scaled_map:
        return ErrorStatus(error=f"data_type {data_type} not supported")

    input_len = model_args[data_type]["input_len"]
    output_len = model_args[data_type]["output_len"]

    hourly_scaled = hourly_scaled_map[data_type]
    min_date = cast(
        pd.Timestamp,
        hourly_scaled.index.min() + pd.Timedelta(hours=input_len),  # type: ignore
    )
    max_date = cast(
        pd.Timestamp,
        hourly_scaled.index.max() - pd.Timedelta(hours=output_len - 1),  # type: ignore
    )

    return AvailableDatetimes(
        min_date=min_date,
        max_date=max_date,
    )


def predict(
    return_dict: "DictProxy[str, FloatArray]",
    model_args: ForecastModelArgs,
    model_path: Path,
    data: ForecastDecomposedData,
    poi_arr: FloatArray,
    mask_arr: BoolArray,
    data_scaler: MinMaxScaler,
    data_type: ForecastDataType,
) -> None:
    """
    Predict future values using the model

    Args:
    - model: trained model
    - data: dictionary with seasonal, residual, trend, and exogenous data
    - poi_arr: array with POI data
    - mask_arr: mask array for POI data
    - device: device for computation
    - data_scaler: scaler object for data
    - data_type: type of data (amount or transactions)

    Returns:
    - prediction: predicted values

    """
    import os
    from typing import cast

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

    seasonal = data["seasonal"]
    residual = data["residual"]
    trend = data["trend"]
    exog = data["exog"]

    poi_tensor = torch.tensor(poi_arr, dtype=torch.float32).to(device)
    mask = torch.tensor(mask_arr, dtype=torch.float32).to(device)

    seasonal = torch.tensor(seasonal, dtype=torch.float32).to(device)
    residual = torch.tensor(residual, dtype=torch.float32).to(device)
    trend = torch.tensor(trend, dtype=torch.float32).to(device)
    exog = torch.tensor(exog, dtype=torch.float32).to(device)

    model = Modelcomplete(ModelArgs(**model_args)).to(device)
    model.load_state_dict(
        torch.load(  # type: ignore
            model_path, map_location=device, weights_only=True
        )
    )
    model.eval()
    with torch.no_grad():
        prediction, _, _, _ = model(seasonal, residual, trend, exog, poi_tensor, mask)
    prediction = cast(
        FloatArray,
        cast(torch.Tensor, prediction).numpy(  # type: ignore
            force=True
        ),
    )

    prediction[:] = data_scaler.inverse_transform(  # type: ignore
        prediction.reshape(-1, prediction.shape[2])
    ).reshape(prediction.shape)
    prediction[:] = np.clip(prediction, 0, None)
    prediction[:] = np.floor(prediction)

    if data_type == "amount":
        prediction[(prediction < 30) & (prediction > 0)] = 0

    prediction = np.squeeze(prediction)

    return_dict["prediction"] = prediction


def plot1(
    pred_series: ForecastDataMapping,
    actual_series: ForecastDataMapping,
    zone_dict: ZoneDictZoneDataMapping,
    index_map: ForecastIndexMapData,
    date: str,
    zone_name: str | None = None,
    parkingmeter_id: int | None = None,
    data_type: ForecastPlot1DataType = "transactions",
) -> Figure:
    assert data_type in FORECAST_PLOT1_DATA_TYPES, (
        f"Invalid data_type: {data_type}. Expected one of {FORECAST_PLOT1_DATA_TYPES}."
    )

    if data_type == "amount":
        actual_series = {k: v / 100 for k, v in actual_series.items()}
        pred_series = {k: v / 100 for k, v in pred_series.items()}

    if parkingmeter_id is None:
        if zone_name is None:
            zone_name = "all_map"
        parkingmeters_zone = zone_dict[zone_name]["parcometro"]

        parkingmeters_zone = [
            index_map["parkimeters"][str(int(parkingmeter_id))]
            for parkingmeter_id in parkingmeters_zone
        ]

        actual = actual_series[date][:, parkingmeters_zone]
        pred = pred_series[date][:, parkingmeters_zone]

        data_sum_actual = actual.reshape(
            (actual.shape[0] // 4, 4, actual.shape[1])
        ).sum(axis=1)
        data_sum_pred = pred.reshape((pred.shape[0] // 4, 4, pred.shape[1])).sum(axis=1)

        real_avg = data_sum_actual.mean(axis=1)
        pred_avg = data_sum_pred.mean(axis=1)

        fig, ax = plt.subplots(1, figsize=(10, 6))  # type: ignore

        if data_type == "transactions":
            ax.plot(real_avg, label="Real", color="blue")  # type: ignore
            ax.plot(pred_avg, label="Predicted", color="red")  # type: ignore
            ax.fill_between(  # type: ignore
                range(actual.shape[0] // 4),
                data_sum_actual.min(axis=1),
                data_sum_actual.max(axis=1),
                color="blue",
                alpha=0.2,
                label="Real range",
            )

        elif data_type == "amount":
            ax.plot(real_avg, label="Real", color="green")  # type: ignore
            ax.plot(pred_avg, label="Predicted", color="red")  # type: ignore
            ax.fill_between(  # type: ignore
                range(actual.shape[0] // 4),
                data_sum_actual.min(axis=1),
                data_sum_actual.max(axis=1),
                color="green",
                alpha=0.2,
                label="Real range",
            )

        dates = pd.date_range(  # type: ignore
            start=date, periods=len(actual), freq="h"
        )
        formatted_dates = pd.Series(dates).dt.strftime("%Y-%m-%d")

        tick_indices = range(0, len(formatted_dates), len(actual) // 7)
        tick_indices_plot = range(0, actual.shape[0] // 4, actual.shape[0] // 4 // 7)
        ax.set_xticks(tick_indices_plot)  # type: ignore
        ax.set_xticklabels(  # type: ignore
            formatted_dates[tick_indices],  # type: ignore
            rotation=45,
        )

        yticks = ax.get_yticks()  # type: ignore
        yticklabels = [f"{int(tick)}" for tick in yticks]
        ax.set_yticklabels(yticklabels)  # type: ignore

        ax.legend(loc="upper right")  # type: ignore

        fig.tight_layout()

    else:
        idx_parkingmeter = index_map["parkimeters"][str(parkingmeter_id)]

        pred = pred_series[date][:, idx_parkingmeter]
        actual = actual_series[date][:, idx_parkingmeter]

        pred = pred.reshape((pred.shape[0] // 4, 4)).sum(axis=1)
        actual = actual.reshape((actual.shape[0] // 4, 4)).sum(axis=1)

        date_ = pd.to_datetime(date)  # type: ignore
        end = date_ + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
        dates_series = pd.date_range(  # type: ignore
            start=date_, end=end, freq="4h"
        )

        formatted_dates = pd.Series(dates_series).dt.strftime("%Y-%m-%d")

        fig, ax = plt.subplots(1, figsize=(10, 6))  # type: ignore

        if data_type == "transactions":
            ax.plot(actual, label="Real", color="blue")  # type: ignore
            ax.plot(pred, label="Predicted", color="red")  # type: ignore

        elif data_type == "amount":
            ax.plot(actual, label="Real", color="green")  # type: ignore
            ax.plot(pred, label="Predicted", color="red")  # type: ignore

        ax.legend()  # type: ignore
        tick_indices = range(0, len(formatted_dates), len(dates_series) // 7)
        ax.set_xticks(tick_indices)  # type: ignore
        ax.set_xticklabels(  # type: ignore
            formatted_dates[tick_indices],  # type: ignore
            rotation=45,
        )
        yticks = ax.get_yticks()  # type: ignore
        yticklabels = [f"{int(tick)}" for tick in yticks]
        ax.set_yticklabels(yticklabels)  # type: ignore

        fig.tight_layout()

    return fig


def plot2(
    pred_series: ForecastDataMapping,
    actual_series: ForecastDataMapping,
    zone_dict: ZoneDictZoneDataMapping,
    index_map: ForecastIndexMapData,
    date: str,
    zone_name: str | None = None,
    road_id: int | None = None,
) -> Figure:
    import pandas as pd

    if road_id is None:
        if zone_name is None:
            zone_name = "all_map"
        roads_zone = zone_dict[zone_name]["strade"]

        roads_zone = [
            index_map["roads"][str(road_id)]
            for road_id in roads_zone
            if str(road_id) in index_map["roads"]
        ]

        actual = actual_series[date][:, roads_zone]
        pred = pred_series[date][:, roads_zone]

        data_sum_actual = actual.reshape(
            (actual.shape[0] // 4, 4, actual.shape[1])
        ).sum(axis=1)
        data_sum_pred = pred.reshape((pred.shape[0] // 4, 4, pred.shape[1])).sum(axis=1)

        real_avg = data_sum_actual.mean(axis=1)
        pred_avg = data_sum_pred.mean(axis=1)

        fig, ax = plt.subplots(1, figsize=(10, 6))  # type: ignore

        ax.plot(real_avg, label="Real", color="darkorange")  # type: ignore
        ax.plot(pred_avg, label="Predicted", color="purple")  # type: ignore
        ax.fill_between(  # type: ignore
            range(actual.shape[0] // 4),
            data_sum_actual.min(axis=1),
            data_sum_actual.max(axis=1),
            color="orange",
            alpha=0.2,
            label="Real range",
        )

        dates = pd.date_range(  # type: ignore
            start=date, periods=len(actual), freq="h"
        )
        formatted_dates = pd.Series(dates).dt.strftime("%Y-%m-%d")

        tick_indices = range(0, len(formatted_dates), len(actual) // 7)
        tick_indices_plot = range(0, actual.shape[0] // 4, actual.shape[0] // 4 // 7)
        ax.set_xticks(tick_indices_plot)  # type: ignore
        ax.set_xticklabels(  # type: ignore
            formatted_dates[tick_indices],  # type: ignore
            rotation=45,
        )

        yticks = [v for v in ax.get_yticks() if v >= 0]  # type: ignore
        # Matplotlib sometimes returns negative yticks, which we filter out
        ax.set_yticks(yticks)  # type: ignore
        yticklabels = [f"{int(tick)}" for tick in yticks]
        ax.set_yticklabels(yticklabels)  # type: ignore

        ax.legend(loc="upper right")  # type: ignore

        fig.tight_layout()

    else:
        idx_road = index_map["roads"][str(road_id)]

        pred = pred_series[date][:, idx_road]
        actual = actual_series[date][:, idx_road]

        pred = pred.reshape((pred.shape[0] // 4, 4)).sum(axis=1)
        actual = actual.reshape((actual.shape[0] // 4, 4)).sum(axis=1)

        date_ = pd.to_datetime(date)  # type: ignore
        end = date_ + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
        dates_series = pd.date_range(  # type: ignore
            start=date_, end=end, freq="4h"
        )

        formatted_dates = pd.Series(dates_series).dt.strftime("%Y-%m-%d")

        fig, ax = plt.subplots(1, figsize=(10, 6))  # type: ignore

        ax.plot(actual, label="Real", color="darkorange")  # type: ignore
        ax.plot(pred, label="Predicted", color="purple")  # type: ignore

        ax.legend()  # type: ignore
        tick_indices = range(0, len(formatted_dates), len(dates_series) // 7)
        ax.set_xticks(tick_indices)  # type: ignore
        ax.set_xticklabels(  # type: ignore
            formatted_dates[tick_indices],  # type: ignore
            rotation=45,
        )
        yticks = ax.get_yticks()  # type: ignore
        yticklabels = [f"{int(tick)}" for tick in yticks]
        ax.set_yticklabels(yticklabels)  # type: ignore

        fig.tight_layout()

    return fig


def do_get_prediction(
    zone_name: str,
    date: pd.Timestamp,
    parkingmeter_id: int | None,
    road_id: int | None,
    data_type: ForecastDataType,
    data: ForecastData,
    zone_dict: ZoneDictZoneDataMapping,
) -> Figure | ErrorStatus:
    if parkingmeter_id is not None:
        if parkingmeter_id not in zone_dict[zone_name]["parcometro"]:
            return ErrorStatus(
                error=f"parking meter id {parkingmeter_id} not in {zone_name}"
            )

    if road_id is not None:
        if road_id not in zone_dict[zone_name]["strade"]:
            return ErrorStatus(error=f"road_id {road_id} not in {zone_name}")

    forecast_data = data["forecast_data"]
    preprocessed_data = data["preprocessed_data"]

    model_args = forecast_data["model_args"][data_type]

    hourly_scaled = preprocessed_data["hourly_scaled"][data_type]
    exog_scaled = preprocessed_data["exog_scaled"][data_type]

    date_range = get_date_range(
        data_type=data_type,
        hourly_scaled_map=preprocessed_data["hourly_scaled"],
        model_args=forecast_data["model_args"],
    )

    if "error" in date_range:
        return date_range

    min_date = date_range["min_date"]
    max_date = date_range["max_date"]

    if date < min_date or date > max_date:
        return ErrorStatus(
            error=f"Date {date} is out of range. Available range: {min_date.date()} to {max_date.date()}"
        )

    poi_arr = preprocessed_data["poi_tensor"][data_type]
    mask_arr = preprocessed_data["mask"][data_type]
    data_scaler = forecast_data["data_scaler"][data_type]

    model_path = data["data_path"] / data_type / "model_best.pth"

    index_map = forecast_data["index_map"]

    start_date = date

    data_decomposed = decompose_data(
        input_len=model_args["input_len"],
        data_scaled=hourly_scaled,
        exog_scaled=exog_scaled,
        start_date=start_date,
    )

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    manager = mp.Manager()
    return_dict: "DictProxy[str, FloatArray]" = manager.dict()

    p = mp.Process(
        target=predict,
        args=(
            return_dict,
            model_args,
            model_path,
            data_decomposed,
            poi_arr,
            mask_arr,
            data_scaler,
            data_type,
        ),
    )
    p.start()
    p.join()

    predictions: ForecastDataMapping = {
        date.date().strftime("%Y-%m-%d"): return_dict["prediction"]
    }
    hourly_data = forecast_data["hourly"][data_type]
    actuals: ForecastDataMapping = {
        date.date().strftime("%Y-%m-%d"): hourly_data.loc[  # type: ignore
            start_date : start_date + pd.Timedelta(hours=model_args["output_len"] - 1)
        ].values
    }

    if data_type == "transactions":
        # Plot Transactions
        return plot1(
            predictions,
            actuals,
            zone_dict=zone_dict,
            index_map=index_map,
            zone_name=zone_name,
            parkingmeter_id=parkingmeter_id,
            date=date.date().strftime("%Y-%m-%d"),
            data_type="transactions",
        )

    if data_type == "amount":
        # Plot Amount
        return plot1(
            predictions,
            actuals,
            zone_dict=zone_dict,
            index_map=index_map,
            zone_name=zone_name,
            parkingmeter_id=parkingmeter_id,
            date=date.date().strftime("%Y-%m-%d"),
            data_type="amount",
        )

    return plot2(
        predictions,
        actuals,
        zone_dict=zone_dict,
        index_map=index_map,
        zone_name=zone_name,
        road_id=road_id,
        date=date.date().strftime("%Y-%m-%d"),
    )
