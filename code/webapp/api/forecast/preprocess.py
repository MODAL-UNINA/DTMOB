from typing import Any

import numpy as np
import pandas as pd

from .data import (
    BoolArray,
    FloatArray,
    ForecastData,
    ForecastDataType,
    ForecastModelArgs,
    OriginalForecastData,
    PreprocessedForecastData,
    WeatherData,
)


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
