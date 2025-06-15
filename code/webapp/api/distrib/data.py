from typing import TypedDict

import pandas as pd


# Data structures
class DistribData(TypedDict):
    """
    TypedDict for the distribution data structure.
    """

    multe_data: pd.DataFrame
