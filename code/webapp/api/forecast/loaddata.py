from pathlib import Path
from typing import Any

from api.general.utils.loading import load_startup_files

from .data import ForecastData
from .preprocess import preprocess
from .startup_data import StartupData


def _load_forecast_files(data_path: Path, startup_data: StartupData) -> dict[str, Any]:
    data_folder = startup_data.get("data_folder")
    assert data_folder is not None, "Data folder must be specified in STARTUP_DATA"
    return dict(
        data_path=data_path / data_folder,
        data=load_startup_files(data_path, startup_data),
    )


def load_data(data_path: Path) -> ForecastData:
    from .startup_data import STARTUP_DATA

    return preprocess(_load_forecast_files(data_path, STARTUP_DATA))
