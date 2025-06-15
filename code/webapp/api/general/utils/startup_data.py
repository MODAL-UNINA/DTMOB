from typing import Any, NotRequired, TypedDict

PklMapping = dict[str, str]
JsonMapping = dict[str, str]


class CSVOptions(TypedDict):
    """
    TypedDict for CSV loading options.
    """

    filepath: str
    args: dict[str, Any]


CSVMapping = dict[str, CSVOptions]


class StartupData(TypedDict):
    """
    TypedDict for the startup data structure.
    This is used to define the structure of the data loaded at startup.
    """

    module_name: str
    data_folder: NotRequired[str]
    pkl_files_data: PklMapping
    json_files_data: JsonMapping
    csv_files_data: NotRequired[CSVMapping]
