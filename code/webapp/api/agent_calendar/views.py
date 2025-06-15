from typing import TypedDict

import pandas as pd
import plotly.graph_objects as go  # type: ignore

from api.general.utils.error_status import ErrorStatus
from api.general.utils.image import get_base64_image_from_plotly
from api.general.views import get_zone_data

from .backend import preprocess_calendar
from .startup import get_data


class AvailableDates(TypedDict):
    min_date: str
    max_date: str


def get_date(selected_date: str) -> str:
    if selected_date == "":
        calendar = get_data()["calendar"]
        selected_date = list(calendar.keys())[0]
    return selected_date


def get_available_calendar_dates() -> AvailableDates:
    calendar = get_data()["calendar"]
    dates = list(calendar.keys())
    return AvailableDates(min_date=min(dates), max_date=max(dates))


def get_calendar_image_inner(
    selected_date: str | pd.Timestamp | None,
) -> go.Figure | ErrorStatus:
    zone_dict = get_zone_data()
    calendar = get_data()["calendar"]
    avail_dates = get_available_calendar_dates()

    if selected_date is None:
        selected_date = avail_dates["max_date"]

    if pd.isnull(selected_date):  # type: ignore
        # An invalid date was selected
        return ErrorStatus(error="Invalid date selected")

    selected_date_ = pd.Timestamp(selected_date)

    min_date = pd.Timestamp(avail_dates["min_date"])
    max_date = pd.Timestamp(avail_dates["max_date"])

    if selected_date_ < min_date or selected_date_ > max_date:
        # The selected date is outside the available range
        return ErrorStatus(error="Date outside available range")

    day_of_week = selected_date_.weekday()

    if day_of_week != 0:
        selected_date_ = selected_date_ - pd.Timedelta(days=day_of_week)

    return preprocess_calendar(calendar, selected_date_, zone_dict)


def get_calendar(selected_date: str | pd.Timestamp | None) -> str | ErrorStatus:
    out = get_calendar_image_inner(selected_date)
    if isinstance(out, dict):
        return out

    fig = out
    img_str = get_base64_image_from_plotly(fig)
    return img_str
