from typing import TypedDict


class AgentCalendarData(TypedDict):
    """
    TypedDict for the agent calendar data structure.
    """

    start: int
    end: int
    zone: list[str]


AgentCalendarDataMapping = dict[str, dict[str, AgentCalendarData]]


class CalendarData(TypedDict):
    """
    TypedDict for the calendar data structure.
    """

    calendar: AgentCalendarDataMapping
