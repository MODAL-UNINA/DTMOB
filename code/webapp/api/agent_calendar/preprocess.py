from typing import Any

from .data import AgentCalendarDataMapping, CalendarData


def _preprocess_calendar_data(
    calendar: AgentCalendarDataMapping,
) -> AgentCalendarDataMapping:
    import pandas as pd

    dates = list(calendar.keys())
    min_date = min(dates)
    max_date = max(dates)

    # assert that the available dates are in an interval
    if min_date >= max_date:
        raise ValueError("The minimum date must be less than the maximum date.")

    if (pd.Timestamp(max_date) - pd.Timestamp(min_date)).days + 1 != len(dates):
        raise ValueError(
            "The number of dates does not match the interval between min and max date."
        )

    return calendar


# Private function to postprocess the data
def preprocess(data: dict[str, Any]) -> CalendarData:
    return CalendarData(
        calendar=_preprocess_calendar_data(AgentCalendarDataMapping(data["calendar"]))
    )
