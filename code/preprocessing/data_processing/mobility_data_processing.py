from typing import Any

import numpy as np
import pandas as pd


def get_registry_by_key(registry: list[dict[str, Any]], key: int) -> dict[str, Any]:
    """
    Retrieve the registry entry for a given key.
    """
    for v in registry:
        if v["id"] == key:
            return v

    raise KeyError(f"Key {key} not found in registry.")


def generate_time_ranges(
    row: "pd.Series[pd.Timestamp]", freq: str
) -> pd.DatetimeIndex | float:
    """
    Generates time intervals for each row of the dataframe based on the specified frequency.

    Parameters:
    - row: The row of the dataframe.
    - freq: The desired frequency for the intervals (e.g., '10T', 'H', '3H', etc.).

    Returns:
    - A pd.DatetimeIndex object representing the time interval, or NaN if conditions are not met.
    """

    start = row["datetime"].floor(freq)
    end = row["next_datetime"].floor(freq)

    if (start == end) and (start < row["datetime"]):
        return np.nan
    elif (start == end) and (start == row["datetime"]):
        return pd.date_range(  # type: ignore
            start=start, end=end, freq=freq
        )
    elif (start < end) and (start == row["datetime"]):
        return pd.date_range(  # type: ignore
            start=start, end=end, freq=freq
        )
    elif (start < end) and (end == row["next_datetime"]):
        start = start + pd.Timedelta(freq)
        return pd.date_range(  # type: ignore
            start=start, end=end, freq=freq
        )
    elif (start < end) and (start != row["datetime"]) and (end != row["next_datetime"]):
        start = start + pd.Timedelta(freq)
        return pd.date_range(  # type: ignore
            start=start, end=end, freq=freq
        )
    return np.nan


def remove_consecutive_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes consecutive duplicate rows based on specific conditions in the dataframe.

    Parameters:
    - df: A pandas DataFrame containing the data to process.

    Returns:
    - A new DataFrame with consecutive duplicate rows (matching the conditions) removed.
    """

    mask = (
        (df["status_change"] == 1)
        & (df["occupied"] == 1)
        & (
            df["numeroStallo"] == df["numeroStallo"].shift()  # type: ignore
        )
        & (
            df["status_change"] == df["status_change"].shift()  # type: ignore
        )
        & (
            df["occupied"] == df["occupied"].shift()  # type: ignore
        )
    )

    return df[~mask].reset_index(drop=True)


def preprocess_sensor_data(
    KPlace_signals: dict[str, Any], slots: pd.DataFrame, df_slots_history: pd.DataFrame
) -> pd.DataFrame:
    """
    Preprocesses sensor data to clean and organize it.

    Parameters:
    - KPlace_signals: Dictionary containing sensor data.
    - slots: DataFrame containing slot data.
    - df_slots_history: DataFrame with historical information on slots.

    Returns:
    - df_final: A cleaned and processed DataFrame.
    """
    from typing import cast

    # Convert KPlace_signals to a DataFrame and clean data
    df_KPlace_signals = pd.DataFrame(KPlace_signals)
    df_KPlace_signals["datetime"] = pd.to_datetime(  # type: ignore
        df_KPlace_signals["datetime"]
    )
    df_KPlace_signals = df_KPlace_signals.drop_duplicates()  # Remove duplicate rows

    # Ensure columns have the correct data types
    df_KPlace_signals["status_change"] = df_KPlace_signals["status_change"].astype(int)
    df_KPlace_signals["occupied"] = df_KPlace_signals["occupied"].astype(int)
    df_KPlace_signals["dev_id"] = df_KPlace_signals["dev_id"].astype(str)

    # Sort and reset index
    df_KPlace_signals.sort_values(  # type: ignore
        by=["dev_id", "datetime"], inplace=True
    )
    df_KPlace_signals.reset_index(drop=True, inplace=True)

    # Rename 'dev_id' to 'devID' for consistency
    df_KPlace_signals.rename(columns={"dev_id": "devID"}, inplace=True)

    # Split slots history into rows with single and multiple entries for 'devID'
    one_count = df_slots_history["devID"].value_counts() == 1
    more_count = df_slots_history["devID"].value_counts() > 1
    df_slots_history1 = df_slots_history[
        df_slots_history["devID"].isin(  # type: ignore
            one_count[one_count].index  # type: ignore
        )
    ]
    df_slots_history2 = df_slots_history[
        df_slots_history["devID"].isin(  # type: ignore
            more_count[more_count].index  # type: ignore
        )
    ]

    # Merge df_KPlace_signals with part 1 of slots history and filter by datetime range
    merged_df = pd.merge(  # type: ignore
        df_KPlace_signals, df_slots_history1, on="devID"
    )
    filtered_df = merged_df[
        (merged_df["start"] <= merged_df["datetime"])
        & ((merged_df["end"].isna()) | (merged_df["end"] >= merged_df["datetime"]))
    ]

    # Handle rows with multiple 'devID' entries
    diff_df = df_KPlace_signals[
        df_KPlace_signals["devID"].isin(  # type: ignore
            more_count[more_count].index  # type: ignore
        )
    ]
    diff_df.insert(1, "idStallo", np.nan)  # type: ignore
    diff_df.insert(1, "start", pd.NaT)  # type: ignore
    diff_df.insert(2, "end", pd.NaT)  # type: ignore

    for index, row in diff_df.iterrows():  # type: ignore
        slots_ = cast(
            pd.DataFrame, df_slots_history2[df_slots_history2["devID"] == row["devID"]]
        )
        for _, slot in slots_.iterrows():  # type: ignore
            if (slot["start"] <= row["datetime"]) and (
                pd.isna(slot["end"])  # type: ignore
                or slot["end"] >= row["datetime"]
            ):
                diff_df.loc[index, "idStallo"] = slot["idStallo"]  # type: ignore
                diff_df.loc[index, "start"] = slot["start"]  # type: ignore
                diff_df.loc[index, "end"] = slot["end"]  # type: ignore

    # Combine filtered data
    df_KPlace_signals1 = pd.concat([filtered_df, diff_df], ignore_index=True)
    df_KPlace_signals1.dropna(  # type: ignore
        subset=["idStallo"], inplace=True
    )
    df_KPlace_signals1["idStallo"] = df_KPlace_signals1["idStallo"].astype(int)

    # Merge with slot data
    df_KPlace_signals1 = pd.merge(  # type: ignore
        df_KPlace_signals1,
        slots[["numeroStallo", "id_strada"]],
        left_on="idStallo",
        right_on="numeroStallo",
        how="left",
    )

    # Filter and sort by slot number and datetime
    df = df_KPlace_signals1.copy()
    df.sort_values(  # type: ignore
        by=["numeroStallo", "datetime"], inplace=True
    )
    df.dropna(  # type: ignore
        subset=["numeroStallo"], inplace=True
    )

    # Filter rows of type 'info_evt'
    df = df[df["type"] == "info_evt"]

    # Remove consecutive duplicates
    df_cleaned = remove_consecutive_duplicates(df)
    df_cleaned.sort_values(  # type: ignore
        by=["numeroStallo", "datetime"], inplace=True
    )

    # Add hour column (floor to the nearest hour)
    df_cleaned["hour"] = df_cleaned["datetime"].dt.floor("H")

    # Filter rows with status_change = 1
    df_signals = df_cleaned[df_cleaned["status_change"] == 1]
    df_signals.drop(columns=["status_change"], inplace=True)

    # Identify and filter rows with meaningful changes
    df_signals["check"] = df_signals["occupied"].diff()
    df_signals = df_signals[df_signals["check"] != 0]
    df_signals.drop(columns=["check"], inplace=True)

    # Calculate time differences and filter by conditions
    df_signals["diff"] = df_signals.groupby(  # type: ignore
        "numeroStallo"
    )["datetime"].diff()
    df_signals["diff_shift"] = df_signals["diff"].shift(  # type: ignore
        -1
    )
    df_signals = df_signals[
        (df_signals["diff"] > pd.Timedelta(seconds=60))
        & (df_signals["diff_shift"] > pd.Timedelta(seconds=60))
    ]

    # Final processing
    df_signals["check"] = df_signals["occupied"].diff()
    df_signals = df_signals[df_signals["check"] != 0]

    # Prepare columns for output
    df_signals.drop(
        columns=["diff", "diff_shift", "check", "start", "end"], inplace=True
    )
    df_signals["diff"] = (
        df_signals.groupby(  # type: ignore
            "numeroStallo"
        )["datetime"]
        .diff()
        .shift(-1)
    )
    df_signals["next_datetime"] = df_signals["datetime"].shift(  # type: ignore
        -1
    )
    df_signals.dropna(  # type: ignore
        subset=["next_datetime"], inplace=True
    )

    # Final DataFrame with relevant columns
    df_final = df_signals.copy()
    df_final = df_final[
        ["numeroStallo", "id_strada", "datetime", "next_datetime", "occupied", "diff"]
    ]

    # Floor datetime and next_datetime to the nearest hour
    df_final["datetime"] = df_final["datetime"].dt.floor("h")
    df_final["next_datetime"] = df_final["next_datetime"].dt.floor("h")

    return df_final


def generate_slot_data(df_final: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """
    Generates aggregated occupancy data for slots based on frequency time intervals.

    Parameters:
    - df_final: DataFrame containing cleaned and preprocessed data.
    - freq: The desired frequency for the time intervals (default: 'h').

    Returns:
    - df_occupied_slots: A pandas DataFrame whose rows represent frequency time intervals,
    columns represent slots (`numeroStallo`),
    and values represent the sum of occupied slots for each slot at each time interval.
    """
    from typing import cast

    df = df_final.copy()
    df["time"] = df.apply(  # type: ignore
        generate_time_ranges,  # type: ignore
        freq=freq,
        axis=1,
    )
    df = df.explode("time")
    df["time"] = df["time"].fillna(  # type: ignore
        df["datetime"].dt.floor(freq)
    )

    df_occupied = df[df["occupied"] == 1]

    # Parking slots blocked for more than 3 days
    anomaly = df_occupied[df_occupied["diff"] > pd.Timedelta(days=3)]
    anomaly = anomaly[
        ["numeroStallo", "id_strada", "datetime", "next_datetime", "diff"]
    ]
    df_occupied_slots = df_occupied.copy()
    df_occupied_slots = (
        df_occupied_slots.groupby(  # type: ignore
            ["numeroStallo", "time"]
        )
        .agg({"occupied": "sum"})
        .reset_index()
    )

    # Adjusting the anomalies
    for id in anomaly["numeroStallo"].unique(  # type: ignore
    ):
        anomaly_subset = cast(pd.DataFrame, anomaly[anomaly["numeroStallo"] == id])
        occupied_subset_mask = cast(
            "pd.Index[pd.BooleanDtype]", df_occupied_slots["numeroStallo"] == id
        )
        for _, anomaly_row in anomaly_subset.iterrows():  # type: ignore
            mask = cast(
                "pd.Index[pd.BooleanDtype]",
                (occupied_subset_mask)
                & (df_occupied_slots["time"] > anomaly_row["datetime"])
                & (df_occupied_slots["time"] < anomaly_row["next_datetime"]),
            )
            df_occupied_slots.loc[mask, "occupied"] = 0

    # Building the finals slots
    resampled_data_list: list[pd.DataFrame] = []
    for slot_number, group in df_occupied_slots.groupby(  # type: ignore
        "numeroStallo"
    ):
        group.set_index(  # type: ignore
            "time", inplace=True
        )
        resampled_group = (
            group.resample(  # type: ignore
                "1h"
            )["occupied"]
            .sum()
            .reset_index(name="occupied")
        )
        resampled_group["numeroStallo"] = slot_number
        resampled_data_list.append(  # type: ignore
            resampled_group
        )

    df_occupied_slots = pd.concat(resampled_data_list)
    df_occupied_slots["numeroStallo"] = df_occupied_slots["numeroStallo"].astype(int)

    df_occupied_slots = df_occupied_slots.pivot(
        columns="numeroStallo", index="time", values="occupied"
    )

    return df_occupied_slots


def generate_road_data(df_final: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    """
    Generates aggregated occupancy data for roads based on hourly time intervals.

    Parameters:
    - df_final: DataFrame containing cleaned and preprocessed data.
    - slots: DataFrame containing slot data.

    Returns:
    - roads_data: A pandas DataFrame where rows represent hourly time intervals, columns represent roads (`id_strada`),
    and values represent the sum of occupied slots for each road at each time interval.
    """
    from typing import cast

    df = df_final.copy()
    freq = "1h"
    df["time"] = df.apply(  # type: ignore
        generate_time_ranges,  # type: ignore
        freq=freq,
        axis=1,
    )
    df = df.explode("time")

    # Group by slot and time, and calculate the sum of occupancy for each group
    df1 = (
        df.groupby(  # type: ignore
            ["numeroStallo", "time"]
        )
        .agg({"occupied": "sum"})
        .reset_index()
    )

    # Define the complete time range for the dataset
    start_time = cast(
        pd.Timestamp,
        df["datetime"].min(  # type: ignore
        ),
    ).floor(freq)  # Round down to the nearest hour
    end_time = cast(
        pd.Timestamp,
        df["next_datetime"].max(  # type: ignore
        ),
    ).ceil(freq)  # Round up to the nearest hour
    hour_intervals = pd.date_range(  # type: ignore
        start=start_time, end=end_time, freq=freq
    )

    # Create a multi-index of all slot-time combinations
    slot_intervals = pd.MultiIndex.from_product(  # type: ignore
        [
            df["numeroStallo"].unique(),  # type: ignore
            hour_intervals,
        ],
        names=["numeroStallo", "time"],
    )
    snapshot_df = pd.DataFrame(index=slot_intervals).reset_index()
    snapshot_df = pd.merge(  # type: ignore
        snapshot_df, df1, on=["numeroStallo", "time"], how="left"
    )
    snapshot_df = snapshot_df.fillna(0)  # type: ignore
    snapshot_df["numeroStallo"] = snapshot_df["numeroStallo"].astype(int)

    # Merge slot metadata (to associate each slot with a road)
    snapshot_df = snapshot_df.merge(  # type: ignore
        slots[["numeroStallo", "id_strada"]],
        left_on="numeroStallo",
        right_on="numeroStallo",
        how="left",
    )

    # Group by road and time, summing the occupancy of all slots for each road
    roads_data = (
        snapshot_df.groupby(["id_strada", "time"])  # type: ignore
        .agg({"occupied": "sum"})
        .reset_index()
    )

    # Pivot the data so that rows are time intervals, columns are roads, and values are occupancy
    roads_data = roads_data.pivot(index="time", columns="id_strada", values="occupied")
    roads_data = roads_data.reindex(  # type: ignore
        sorted(roads_data.columns), axis=1
    )
    roads_data.columns = roads_data.columns.astype(int)  # type: ignore
    roads_data.index = pd.to_datetime(roads_data.index)  # type: ignore

    return roads_data


def generate_hourly_transactions(
    transaction_data: pd.DataFrame, data_type: str, freq: str = "h"
) -> pd.DataFrame:
    """
    Preprocesses transaction data to aggregate hourly transaction counts for each 'id_parcometro'
    within a specified date range.

    Args:
    - transaction_data (DataFrame): Input transaction data.
    - data_type (str): The type of data to generate ('transactions' or 'amount').
    - freq (str): The frequency for the time intervals (default: 'h').

    Returns:
    - DataFrame: Hourly transaction counts for each 'id_parcometro' within the specified date range.

    """
    from typing import cast

    if data_type not in ["transactions", "amount"]:
        raise ValueError("data_type must be either 'transactions' or 'amount'.")

    # Clean data
    transaction_data = transaction_data.drop_duplicates()

    numeric_columns = [
        "id",
        "id_parcometro",
        "id_tipopagamento",
        "stallo",
        "amount",
        "numeroTransazione",
    ]
    transaction_data[numeric_columns] = transaction_data[numeric_columns].apply(  # type: ignore
        pd.to_numeric,  # type: ignore
        errors="coerce",
    )

    # Drop rows where 'end_park' is NaN or 'amount' is 0
    transaction_data.dropna(  # type: ignore
        subset=["end_park"], inplace=True
    )
    transaction_data = transaction_data[transaction_data["amount"] != 0]

    transaction_data = transaction_data[transaction_data["id_parcometro"].notna()]

    transaction_data["start_park"] = pd.to_datetime(  # type: ignore
        transaction_data["start_park"]
    )
    transaction_data["end_park"] = pd.to_datetime(  # type: ignore
        transaction_data["end_park"]
    )

    start = cast(
        pd.Timestamp,
        transaction_data["start_park"].min(  # type: ignore
        ),
    ).floor(freq)
    end = cast(
        pd.Timestamp,
        transaction_data["start_park"].max(  # type: ignore
        ),
    ).floor(freq)
    hour_range = pd.date_range(  # type: ignore
        start=start, end=end, freq=freq
    )

    # Initialize an empty DataFrame for hourly transactions
    parkingmeters_id = transaction_data["id_parcometro"].unique()  # type: ignore
    hourly_transactions = pd.DataFrame(
        index=hour_range, columns=parkingmeters_id
    ).fillna(  # type: ignore
        0
    )

    # Floor the start time to the nearest hour
    transaction_data["start_floor"] = transaction_data["start_park"].dt.floor(freq)

    if data_type == "transactions":
        start_group = (
            transaction_data.groupby(  # type: ignore
                ["start_floor", "id_parcometro"]
            )["start_floor"]
            .count()
            .unstack(fill_value=0)
        )
    else:
        start_group = (
            transaction_data.groupby(  # type: ignore
                ["start_floor", "id_parcometro"]
            )["amount"]
            .sum()
            .unstack(fill_value=0)
        )

    # Add the start contributions to hourly_transactions
    hourly_transactions = hourly_transactions.add(  # type: ignore
        start_group, fill_value=0
    )

    hourly_transactions.index = pd.to_datetime(  # type: ignore
        hourly_transactions.index
    )
    hourly_transactions.columns = hourly_transactions.columns.astype(  # type: ignore
        int
    )

    return hourly_transactions
