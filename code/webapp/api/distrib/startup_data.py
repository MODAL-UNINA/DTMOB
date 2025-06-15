from api.general.utils.startup_data import StartupData, JsonMapping, PklMapping

STARTUP_DATA = StartupData(
    module_name="Distrib",
    data_folder="distrib",
    pkl_files_data=PklMapping(multe_data="multe_data.pkl"),
    json_files_data=JsonMapping(),
)
