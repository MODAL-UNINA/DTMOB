from pathlib import Path

from api.general.utils.loading import load_startup_files

from .data import CalendarData
from .preprocess import preprocess
from .startup_data import STARTUP_DATA


def load_data(data_path: Path) -> CalendarData:
    return preprocess(load_startup_files(data_path, STARTUP_DATA))
