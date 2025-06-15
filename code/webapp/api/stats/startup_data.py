from api.general.utils.startup_data import JsonMapping, PklMapping, StartupData

STARTUP_DATA = StartupData(
    module_name="Stats",
    data_folder="stats",
    pkl_files_data=PklMapping(
        events_data="event_data.pkl",
        multe_data="multe_data_tab3.pkl",
        poi_data="pois_data.pkl",
        zone_params="zone_params.pkl",
    ),
    json_files_data=JsonMapping(),
)
