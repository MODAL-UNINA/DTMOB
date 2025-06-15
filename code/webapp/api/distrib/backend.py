from typing import cast

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from api.general.data import ZoneDictZoneDataMapping


def plot1(
    data: pd.DataFrame,
    zone_dict: ZoneDictZoneDataMapping,
    zone_name: str = "all_map",
    parkingmeter_id: int | None = None,
    date: pd.Timestamp | None = None,
    hour_range: list[int] | None = None,
    data_type: str = "count",
) -> Figure:
    if parkingmeter_id is not None:
        if parkingmeter_id not in zone_dict[zone_name]["parcometro"]:
            raise ValueError(f"parkimeter_id {parkingmeter_id} not in {zone_name}")

    if parkingmeter_id is not None:
        parkingmeters_id = [parkingmeter_id]
    else:
        parkingmeters_id = zone_dict[zone_name]["parcometro"]

    data = data[parkingmeters_id]

    if date is not None:
        selection = date + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
        data = data.loc[date:selection]
    else:
        date = cast(
            pd.Timestamp,
            data.index.min(),  # type: ignore
        )
        selection = cast(
            pd.Timestamp,
            data.index.max(),  # type: ignore
        )

    all_time_ranges = pd.date_range(  # type: ignore
        start=date, end=selection, freq="h"
    )
    data = data.reindex(  # type: ignore
        all_time_ranges, fill_value=0
    )

    if hour_range is not None:
        hour_range1 = list(range(hour_range[0], hour_range[1]))
        data = cast(
            pd.DataFrame,
            data[
                data.index.hour.isin(hour_range1)  # type: ignore
            ],
        )
    else:
        data = data.resample(  # type: ignore
            "4h"
        ).sum()
        all_time_ranges = pd.date_range(  # type: ignore
            start=date, end=selection, freq="4H"
        )
        data = data.reindex(  # type: ignore
            all_time_ranges, fill_value=0
        )

    data_s = cast(
        "pd.Series[pd.Float64Dtype]",
        data.sum(  # type: ignore
            axis=1
        ),
    )
    if data_type == "amount":
        data_s = data_s / 100
        color = "green"
    else:
        color = "blue"
    dates = cast(
        list[str],
        list(
            data_s.index.strftime(  # type: ignore
                "%Y-%m-%d"
            )
        ),
    )

    tick_indices = range(0, len(dates), len(dates) // 7)

    fig, ax = plt.subplots()  # type: ignore
    ax.plot(  # type: ignore
        data_s.values,  # type: ignore
        color=color,
    )
    ax.set_xticks(  # type: ignore
        tick_indices, [dates[tick_index] for tick_index in tick_indices], rotation=45
    )
    fig.tight_layout()
    fig.set_size_inches(10, 6)

    return fig


def plot2(
    data: pd.DataFrame,
    zone_dict: ZoneDictZoneDataMapping,
    zone_name: str = "all_map",
    slot_id: int | None = None,
    date: pd.Timestamp | None = None,
    hour_range: list[int] | None = None,
) -> Figure:
    if slot_id is not None:
        if slot_id not in zone_dict[zone_name]["stalli"]:
            raise ValueError(f"slot_id {slot_id} not in {zone_name}")

    if slot_id is not None:
        slots = [slot_id]
    else:
        slots = zone_dict[zone_name]["stalli"]

    data = data[
        data["idStallo"].isin(slots)  # type: ignore
    ]

    if date is not None:
        selection = date + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
        data = data[(data["datetime"] >= date) & (data["datetime"] <= selection)]
    else:
        date = cast(pd.Timestamp, data["datetime"].min())
        selection = cast(pd.Timestamp, data["datetime"].max())
    data["total"] = data["occupied_regularly"] + data["occupied_abusively"]
    if hour_range is not None:
        data = data.groupby(  # type: ignore
            data["datetime"]  # type: ignore
        ).agg({"total": "sum"})
        all_time_ranges = pd.date_range(  # type: ignore
            start=date, end=selection, freq="H"
        )
        data = data.reindex(  # type: ignore
            all_time_ranges, fill_value=0
        )
        hour_range1 = list(range(hour_range[0], hour_range[1]))
        data = cast(
            pd.DataFrame,
            data[
                data.index.hour.isin(hour_range1)  # type: ignore
            ],
        )

    else:
        data = data.groupby(  # type: ignore
            data["datetime"]  # type: ignore
        ).agg({"total": "sum"})
        data = data.resample(  # type: ignore
            "4H"
        ).sum()
        all_time_ranges = pd.date_range(  # type: ignore
            start=date, end=selection, freq="4H"
        )
        data = data.reindex(  # type: ignore
            all_time_ranges, fill_value=0
        )

    dates = cast(
        list[str],
        list(
            data.index.strftime(  # type: ignore
                "%Y-%m-%d"
            )
        ),
    )
    tick_indices = range(0, len(dates), len(dates) // 7)

    fig, ax = plt.subplots()  # type: ignore
    ax.plot(  # type: ignore
        data.values,  # type: ignore
        color="orange",
    )
    ax.set_xticks(  # type: ignore
        tick_indices, [dates[tick_index] for tick_index in tick_indices], rotation=45
    )
    fig.tight_layout()
    fig.set_size_inches(10, 6)

    return fig


def plot3(
    data: pd.DataFrame,
    zone_dict: ZoneDictZoneDataMapping,
    zone_name: str = "all_map",
    slot_id: int | None = None,
    date: pd.Timestamp | None = None,
    hour_range: list[int] | None = None,
    data_type: str = "occupied_abusively",
) -> Figure:
    if slot_id is not None:
        if slot_id not in zone_dict[zone_name]["stalli"]:
            raise ValueError(f"slot_id {slot_id} not in {zone_name}")

    if slot_id is not None:
        slots = [slot_id]
    else:
        slots = zone_dict[zone_name]["stalli"]

    data = data[
        data["idStallo"].isin(slots)  # type: ignore
    ]

    if date is not None:
        selection = date + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
        data = data[(data["datetime"] >= date) & (data["datetime"] <= selection)]
    else:
        date = cast(pd.Timestamp, data["datetime"].min())
        selection = cast(pd.Timestamp, data["datetime"].max())

    if hour_range is not None:
        data = data.groupby(  # type: ignore
            data["datetime"].dt.floor("h")  # type: ignore
        )[data_type].sum()
        all_time_ranges = pd.date_range(  # type: ignore
            start=date, end=selection, freq="h"
        )
        data = data.reindex(  # type: ignore
            all_time_ranges, fill_value=0
        )
        data.rename_axis(  # type: ignore
            None, axis=0, inplace=True
        )
        hour_range1 = list(range(hour_range[0], hour_range[1]))
        data = cast(
            pd.DataFrame,
            data[
                data.index.hour.isin(hour_range1)  # type: ignore
            ],
        )
    else:
        data = data.groupby(  # type: ignore
            data["datetime"].dt.floor("h")  # type: ignore
        )[data_type].sum()
        all_time_ranges = pd.date_range(  # type: ignore
            start=date, end=selection, freq="h"
        )
        data = data.reindex(  # type: ignore
            all_time_ranges, fill_value=0
        )
        data.rename_axis(  # type: ignore
            None, axis=0, inplace=True
        )
        data = data.resample(  # type: ignore
            "4h"
        ).sum()

    data.index = pd.to_datetime(  # type: ignore
        data.index  # type: ignore
    )
    dates = cast(
        list[str],
        list(
            data.index.strftime(  # type: ignore
                "%Y-%m-%d"
            )
        ),
    )

    tick_indices = range(0, len(dates), len(dates) // 7)
    fig, ax = plt.subplots()  # type: ignore
    ax.plot(  # type: ignore
        data.values,  # type: ignore
        label=data_type,
        color="orange",
    )
    ax.set_xticks(  # type: ignore
        tick_indices, [dates[tick_index] for tick_index in tick_indices], rotation=45
    )
    fig.tight_layout()
    fig.set_size_inches(10, 6)
    return fig


def plot4(
    data: pd.DataFrame,
    zone_dict: ZoneDictZoneDataMapping,
    zone_name: str = "all_map",
    date: pd.Timestamp | None = None,
    hour_range: list[int] | None = None,
) -> Figure:
    if zone_name not in zone_dict:
        raise ValueError(f"zone_name {zone_name} not in zone_dict")

    streets = zone_dict[zone_name]["strade"]
    data = data[
        data["id_strada"].isin(streets)  # type: ignore
    ]

    if date is not None:
        selection = date + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
        data = data[(data["datetime"] >= date) & (data["datetime"] <= selection)]
    else:
        date = cast(pd.Timestamp, data["datetime"].min())
        selection = cast(pd.Timestamp, data["datetime"].max())

    if hour_range is not None:
        data = data.groupby(  # type: ignore
            data["datetime"].dt.floor("h")  # type: ignore
        )["num_tickets"].sum()
        all_time_ranges = pd.date_range(  # type: ignore
            start=date, end=selection, freq="h"
        )
        data = data.reindex(  # type: ignore
            all_time_ranges, fill_value=0
        )
        hour_range1 = list(range(hour_range[0], hour_range[1]))
        data = cast(
            pd.DataFrame,
            data[
                data.index.hour.isin(hour_range1)  # type: ignore
            ],
        )

    else:
        data = data.groupby(  # type: ignore
            data["datetime"].dt.floor("h")  # type: ignore
        )["num_tickets"].sum()
        all_time_ranges = pd.date_range(  # type: ignore
            start=date, end=selection, freq="h"
        )
        data = data.reindex(  # type: ignore
            all_time_ranges, fill_value=0
        )
        data = data.resample(  # type: ignore
            "4h"
        ).sum()
        data.index = pd.to_datetime(  # type: ignore
            data.index  # type: ignore
        )

    dates = cast(
        list[str],
        list(
            data.index.strftime("%Y-%m-%d")  # type: ignore
        ),
    )
    tick_indices = list(range(0, len(dates), len(dates) // 7))
    fig, ax = plt.subplots()  # type: ignore
    ax.plot(  # type: ignore
        data.values,  # type: ignore
        color="purple",
    )
    ax.set_xticks(  # type: ignore
        tick_indices, [dates[tick_index] for tick_index in tick_indices], rotation=45
    )
    fig.tight_layout()
    fig.set_size_inches(10, 6)
    return fig
