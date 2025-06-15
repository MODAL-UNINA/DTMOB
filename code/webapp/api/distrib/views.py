import matplotlib
import pandas as pd

from api.general.utils.image import get_base64_image

matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from api.general.views import (
    LegalityStatus,
    get_amount_parkingmeters,
    get_status_sensors,
    get_transactions_parkingmeters,
    get_zone_dict,
)

from .backend import plot1, plot2, plot3, plot4
from .startup import get_data


def get_transactions_count_plot_inner(
    zone_name: str,
    date: pd.Timestamp | None,
    hour_range: list[int] | None,
    parkingmeter_id: int | None,
) -> Figure:
    transactions_parkingmeters = get_transactions_parkingmeters()
    zone_dict = get_zone_dict()

    return plot1(
        transactions_parkingmeters,
        zone_dict=zone_dict,
        zone_name=zone_name,
        parkingmeter_id=parkingmeter_id,
        date=date,
        hour_range=hour_range,
        data_type="count",
    )


def get_transactions_amount_plot_inner(
    zone_name: str,
    date: pd.Timestamp | None,
    hour_range: list[int] | None,
    parkingmeter_id: int | None,
) -> Figure:
    amount_parkingmeters = get_amount_parkingmeters()
    zone_dict = get_zone_dict()

    return plot1(
        amount_parkingmeters,
        zone_dict=zone_dict,
        zone_name=zone_name,
        parkingmeter_id=parkingmeter_id,
        date=date,
        hour_range=hour_range,
        data_type="amount",
    )


def get_occupancy_plot_inner(
    zone_name: str,
    date: pd.Timestamp | None,
    hour_range: list[int] | None,
    parkingslot: int | None,
    legality_status: LegalityStatus | None,
) -> Figure:
    zone_dict = get_zone_dict()
    slot_id = parkingslot
    if legality_status is None:
        all_sensors = get_status_sensors()
        return plot2(
            all_sensors,
            zone_dict=zone_dict,
            zone_name=zone_name,
            slot_id=slot_id,
            date=date,
            hour_range=hour_range,
        )

    assert legality_status in ["occupied_regularly", "occupied_abusively"]

    status_sensors = get_status_sensors()
    data_type = legality_status

    return plot3(
        status_sensors,
        zone_dict=zone_dict,
        zone_name=zone_name,
        slot_id=slot_id,
        date=date,
        hour_range=hour_range,
        data_type=data_type,
    )


def get_fines_plot_inner(
    zone_name: str, date: pd.Timestamp | None, hour_range: list[int] | None
) -> Figure:
    fines_data = get_data()["multe_data"]
    zone_dict = get_zone_dict()
    return plot4(
        fines_data,
        zone_dict=zone_dict,
        zone_name=zone_name,
        date=date,
        hour_range=hour_range,
    )


def get_transactions_count_image(
    zone_name: str,
    date: pd.Timestamp | None,
    hour_range: list[int] | None,
    parkingmeter: int | None,
) -> str:
    fig = get_transactions_count_plot_inner(zone_name, date, hour_range, parkingmeter)

    fig, img_str = get_base64_image(fig)
    plt.close(fig)

    return img_str


def get_transactions_amount_image(
    zone_name: str,
    date: pd.Timestamp | None,
    hour_range: list[int] | None,
    parkingmeter: int | None,
) -> str:
    fig = get_transactions_amount_plot_inner(zone_name, date, hour_range, parkingmeter)

    fig, img_str = get_base64_image(fig)
    plt.close(fig)

    return img_str


def get_occupancy_image(
    zone_name: str,
    date: pd.Timestamp | None,
    hour_range: list[int] | None,
    parkingslot: int | None,
    legality_status: LegalityStatus | None,
) -> str:
    fig = get_occupancy_plot_inner(
        zone_name, date, hour_range, parkingslot, legality_status
    )

    fig, img_str = get_base64_image(fig)
    plt.close(fig)

    return img_str


def get_fines_image(
    zone_name: str, date: pd.Timestamp | None, hour_range: list[int] | None
) -> str:
    fig = get_fines_plot_inner(zone_name, date, hour_range)

    fig, img_str = get_base64_image(fig)
    plt.close(fig)

    return img_str
