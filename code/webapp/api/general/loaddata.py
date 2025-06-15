from pathlib import Path

from api.general.utils.loading import load_startup_files

from .data import GeneralData
from .preprocess import preprocess


def load_data(data_path: Path) -> GeneralData:
    from .startup_data import STARTUP_DATA

    return preprocess(load_startup_files(data_path, STARTUP_DATA))
