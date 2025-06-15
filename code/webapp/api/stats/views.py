from typing import TypedDict

import matplotlib

from api.general.utils.image import get_base64_image

matplotlib.use("agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from api.general.views import (
    get_all_sensors,
    get_status_sensors,
    get_transactions_parkingmeters,
    get_zone_data,
    get_zone_dict,
)

from .backend import AbusivismData, ColorMap, StatsData, get_stats_info
from .startup import get_data


class StatsDescriptionInner(TypedDict):
    generalStats: StatsData
    abusivismStats: AbusivismData | None


class StatsDescription(TypedDict):
    generalStats: StatsData
    abusivismStats: AbusivismData | None
    statsImgRel: str | None
    statsImgAbs: str | None
    labelMap: dict[str, str]
    colorMap: dict[str, ColorMap]


def get_stats_describe_plot_inner(
    zone_name: str, date: pd.Timestamp | None, hour_range: list[int] | None
) -> tuple[
    StatsDescriptionInner,
    tuple[Figure, Figure] | None,
    dict[str, str],
    dict[str, ColorMap],
]:
    transactions_parkingmeters = get_transactions_parkingmeters()
    zone_data = get_zone_data()
    mapping_dict = get_zone_dict()
    stats_data = get_data()
    poi_data = stats_data["poi_data"]
    events_data = stats_data["events_data"]
    all_parkingslots = get_all_sensors()
    status_parkingslots = get_status_sensors()
    fines_data = stats_data["multe_data"]
    zone_params = stats_data["zone_params"]

    results, results_abusivism, correlations, label_map, color_map = get_stats_info(
        zone_data=zone_data,
        mapping_dict=mapping_dict,
        poi_data=poi_data,
        events_data=events_data,
        transactions_parkingmeters=transactions_parkingmeters,
        all_parkingslots=all_parkingslots,
        status_parkingslots=status_parkingslots,
        fines_data=fines_data,
        zone_params=zone_params,
        zone_name=zone_name,
        date=date,
        hour_range=hour_range,
    )

    return (
        StatsDescriptionInner(
            generalStats=results,
            abusivismStats=results_abusivism,
        ),
        correlations,
        label_map,
        color_map,
    )


def get_stats_describe_data(
    zone_name: str, date: pd.Timestamp | None, hour_range: list[int] | None
) -> StatsDescription:
    res, figs, label_map, color_map = get_stats_describe_plot_inner(
        zone_name, date, hour_range
    )

    if figs is not None:
        fig_rel, fig_abs = figs
        fig_rel, img_rel_str = get_base64_image(fig_rel)
        plt.close(fig_rel)
        fig_abs, img_abs_str = get_base64_image(fig_abs)
        plt.close(fig_abs)
    else:
        img_rel_str = None
        img_abs_str = None

    return StatsDescription(
        **res,
        statsImgRel=img_rel_str,
        statsImgAbs=img_abs_str,
        labelMap=label_map,
        colorMap=color_map,
    )
