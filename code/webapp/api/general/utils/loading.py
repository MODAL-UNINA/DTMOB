from pathlib import Path
from typing import Any

from .startup_data import CSVMapping, JsonMapping, PklMapping, StartupData


def load_files(
    data_path: Path,
    module_name: str,
    pkl_files_dict: PklMapping,
    json_files_dict: JsonMapping,
    csv_files_dict: CSVMapping | None = None,
) -> dict[str, Any]:
    """
    Load all necessary JSON files into memory.
    """
    import json
    import pickle as pkl

    import pandas as pd

    out_data: dict[str, Any] = {}

    for key, value in pkl_files_dict.items():
        filepath = data_path / value
        with open(filepath, "rb") as f:
            data = pkl.load(f)
        out_data[key] = data
        print(f"{module_name}: loaded {key} from {filepath}")

    for key, value in json_files_dict.items():
        filepath = data_path / value
        with open(filepath, "r") as f:
            data = json.load(f)
        out_data[key] = data
        print(f"{module_name}: loaded {key} from {filepath}")

    if csv_files_dict:
        for key, options in csv_files_dict.items():
            filepath = data_path / options["filepath"]
            data = pd.read_csv(  # type: ignore
                filepath, **options["args"]
            )
            out_data[key] = data
            print(f"{module_name}: loaded {key} from {filepath}")

    return out_data


def load_startup_files(data_path: Path, startup_data: StartupData) -> dict[str, Any]:
    module_path = startup_data.get("data_folder")
    if module_path is not None:
        data_path = data_path / module_path
    return load_files(
        data_path,
        startup_data["module_name"],
        startup_data["pkl_files_data"],
        startup_data["json_files_data"],
        startup_data.get("csv_files_data"),
    )
