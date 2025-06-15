from api.general.utils.startup_data import JsonMapping, PklMapping, StartupData

STARTUP_DATA = StartupData(
    module_name="Map",
    data_folder="map",
    pkl_files_data=PklMapping(
        parkingmeter_positions="posizioni_parcometri_new.pkl",
        roads_gdf="roads_gdf.pkl",
        sensors="stalli_selection_new.pkl",
    ),
    json_files_data=JsonMapping(),
)
