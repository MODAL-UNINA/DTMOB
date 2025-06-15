from api.general.utils.startup_data import StartupData, JsonMapping, PklMapping

STARTUP_DATA = StartupData(
    module_name="Agent calendar",
    data_folder="agent_calendar",
    pkl_files_data=PklMapping(calendar="calendar.pkl"),
    json_files_data=JsonMapping(),
)
