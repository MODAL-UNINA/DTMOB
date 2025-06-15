from typing import Any
from .data import DistribData


def preprocess(data: dict[str, Any]) -> DistribData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    return DistribData(**data)
