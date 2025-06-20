import random
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast, get_args

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from common.forecasting.models import ModelArgs, Modelcomplete
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL  # type: ignore

from api.forecast.data import (
    BoolArray,
    FloatArray,
    ForecastData,
    ForecastDataType,
    ForecastIndexMapData,
    ForecastModelArgs,
)
from api.general.data import ZoneDictZoneDataMapping
from api.general.utils.error_status import ErrorStatus

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
    prediction = prediction.cpu().detach().numpy()

    prediction = data_scaler.inverse_transform(  # type: ignore
        prediction.reshape(-1, prediction.shape[2])
    ).reshape(prediction.shape)
    prediction = prediction.astype(int)
    prediction = np.maximum(prediction, 0)

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

        yticks = ax.get_yticks()  # type: ignore
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
            error=f"Date {date} is out of range. Available range: {min_date} to {max_date}"
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

    process_name = "__".join(
        ["predict", data_type, date.strftime("%Y-%m-%d"), zone_name]
    )
    p = mp.Process(
        target=predict,
        name=process_name,
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
        daemon=True,
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
