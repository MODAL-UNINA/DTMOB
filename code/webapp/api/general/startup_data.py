from api.general.utils.startup_data import (
    JsonMapping,
    PklMapping,
    StartupData,
)

STARTUP_DATA = StartupData(
    module_name="General",
    pkl_files_data=PklMapping(
        transactions_parkingmeters="transactions_number_parkimeters.pkl",
        amount_parkingmeters="transactions_amount_parkimeters.pkl",
        all_sensors="sensor_data_new.pkl",
        status_sensors="status_sensor_data_new_2.pkl",
        zone="zone.pkl",
    ),
    json_files_data=JsonMapping(
        hourslots="hour_slots.json",
        timeslots_macroareas="time_slots_macroareas.json",
        zone_dict="mapping_dict.json",
        macrozone_params="macrozone_params.json",
    ),
)
