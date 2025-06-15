from typing import Any

from .data import StatsData


def preprocess(data: dict[str, Any]) -> StatsData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    return StatsData(**data)
