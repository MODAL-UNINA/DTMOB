from api.general.utils.startup_data import (
    CSVMapping,
    CSVOptions,
    JsonMapping,
    PklMapping,
    StartupData,
)


STARTUP_DATA = StartupData(
    module_name="WhatIf",
    data_folder="whatif",
    pkl_files_data=PklMapping(
        final_parkingmeters__1st="1st/input/final_parkingmeters.pkl",
        final_parkingslots__1st="1st/input/final_parkingslots.pkl",
        p_coordinates__1st="1st/input/p_coords.pkl",
        p_scaler__1st="1st/input/p_scaler.pkl",
        s_coordinates__1st="1st/input/s_coords.pkl",
        s_scaler__1st="1st/input/s_scaler.pkl",
        final_parkingslots__2nd="2nd/input/final_parkingslots.pkl",
        s_coordinates__2nd="2nd/input/s_coords.pkl",
        s_scaler__2nd="2nd/input/s_scaler.pkl",
        road_dict__2nd="2nd/input/road_dict.pkl",
        final_parkingmeters__3rd="3rd/input/final_parkingmeters.pkl",
        final_parkingslots__3rd="3rd/input/final_parkingslots.pkl",
        p_coordinates__3rd="3rd/input/p_coords.pkl",
        p_scaler__3rd="3rd/input/p_scaler.pkl",
        s_coordinates__3rd="3rd/input/s_coords.pkl",
        s_scaler__3rd="3rd/input/s_scaler.pkl",
    ),
    json_files_data=JsonMapping(
        distances_p="distances_p.json",
        distances_s="distances_s.json",
        dict_zone="vicinity_zone_map.json",
    ),
    csv_files_data=CSVMapping(
        weather_data__3rd=CSVOptions(
            filepath="3rd/input/data_meteo_4h.csv",
            args=dict(index_col="date"),
        )
    ),
)
