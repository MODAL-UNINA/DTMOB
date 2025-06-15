from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL  # type: ignore
from torch.utils.data import Dataset

FloatArray = NDArray[np.float64]
ObjectArray = NDArray[np.object_]


def create_sequences_multivariate(
    series: FloatArray, indices: "pd.Index[Any]", seq_length: int, horizon: int
) -> tuple[torch.Tensor, ObjectArray, torch.Tensor, ObjectArray]:
    """
    Creates sequences for multivariate time series forecasting.

    Parameters:
    - series: The multivariate time series data.
    - indices: The indices or time steps corresponding to each row in `series`.
    - seq_length: The length of the input sequence.
    - horizon: The number of future time steps to predict.

    Returns:
    - xs: Input sequences as torch tensors.
    - x_indices: Indices corresponding to the input sequences.
    - ys: Output sequences as torch tensors.
    - y_indices: Indices corresponding to the output sequences.
    """

    xs: list[FloatArray] = []
    ys: list[FloatArray] = []

    x_indices: list["pd.Index[Any]"] = []
    y_indices: list["pd.Index[Any]"] = []
    data = series
    for i in range(len(data) - seq_length - horizon + 1):
        x = data[i : i + seq_length, :]
        y = data[i + seq_length : i + seq_length + horizon, :]
        x_indx = indices[i : i + seq_length]
        y_indx = indices[i + seq_length : i + seq_length + horizon]
        x_indices.append(x_indx)
        y_indices.append(y_indx)
        xs.append(x)
        ys.append(y)

    return (
        torch.tensor(np.array(xs), dtype=torch.float32),
        np.array(x_indices),
        torch.tensor(np.array(ys), dtype=torch.float32),
        np.array(y_indices),
    )


def add_features(
    data: list[pd.DataFrame],
    add_time_of_day: bool = False,
    add_day_of_week: bool = False,
    steps_per_day: int = 24,
) -> FloatArray:
    """
    Adds additional time-related features (time of day and day of week) to a given time series dataset.

    Parameters:
    - data: The input time series data.
    - add_time_of_day: Boolean flag to indicate if the 'time of day' feature should be added.
    - add_day_of_week: Boolean flag to indicate if the 'day of the week' feature should be added.
    - steps_per_day: The number of steps per day (default is 24, assuming hourly data).

    Returns:
    - data_with_features: A NumPy array containing the original time series data plus any added time-related features.
    """
    from typing import cast

    data_0 = data[0]
    data1 = cast(
        FloatArray,
        np.expand_dims(
            data_0.values,  # type: ignore
            axis=-1,
        ),
    )
    _, n = data_0.shape
    feature_list = [data1]
    index = cast(pd.DatetimeIndex, data_0.index)

    if add_time_of_day:
        # Numerical time_of_day
        tod = index.hour / steps_per_day
        tod = cast(FloatArray, np.array(tod))
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # Numerical day_of_week
        dow = index.dayofweek / 7
        dow_tiled = cast(FloatArray, np.tile(dow, [1, n, 1]).transpose((2, 1, 0)))
        feature_list.append(dow_tiled)

    data_with_features = np.concatenate(feature_list, axis=-1)

    return data_with_features


class TimeSeriesDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor | None = None,
        seasonal: torch.Tensor | None = None,
        trend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        exogenous: torch.Tensor | None = None,
        poi_data: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> None:
        self.data = data
        self.targets = targets
        self.seasonal = seasonal
        self.trend = trend
        self.residual = residual
        self.exogenous = exogenous
        self.poi_data = poi_data
        self.mask = mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {}
        item["data"] = self.data[idx]
        if self.targets is not None:
            item["targets"] = self.targets[idx]
        if self.seasonal is not None:
            item["seasonal"] = self.seasonal[idx]
        if self.trend is not None:
            item["trend"] = self.trend[idx]
        if self.residual is not None:
            item["residual"] = self.residual[idx]
        if self.exogenous is not None:
            item["exogenous"] = self.exogenous[idx]
        if self.poi_data is not None:
            item["poi_data"] = self.poi_data
        if self.mask is not None:
            item["mask"] = self.mask
        return item


def haversine_matrix(gps_coordinates: torch.Tensor) -> torch.Tensor:
    """
    Compute a matrix of Haversine distances between all GPS points.

    Parameters:
    - gps_coordinates (tensor): A 2D tensor with shape (N, 2), where each row
    represents (latitude, longitude) in degrees.

    Returns:
    - distance_matrix (tensor): A 2D tensor of shape (N, N) containing the pairwise Haversine distances in kilometers.
    """
    latitudes = gps_coordinates[:, 0]
    longitudes = gps_coordinates[:, 1]

    latitudes = torch.deg2rad(latitudes)
    longitudes = torch.deg2rad(longitudes)

    lat1 = latitudes.unsqueeze(0)
    lat2 = latitudes.unsqueeze(1)
    lon1 = longitudes.unsqueeze(0)
    lon2 = longitudes.unsqueeze(1)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Haversine formula
    a = (
        torch.sin(delta_lat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(delta_lon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # Earth radius in km
    R = 6371.0

    # Compute distance matrix
    distance_matrix = R * c

    return distance_matrix


def normalize_distance_matrix(distance_matrix: torch.Tensor) -> torch.Tensor:
    """
    Normalize the distance matrix to a [0, 1] scale.

    Parameters:
    - distance_matrix (tensor): A 2D tensor containing pairwise distances.

    Returns:
    - normalized_matrix (tensor): A 2D tensor where values are scaled between 0 and 1.
    """

    min_dist = distance_matrix.min()
    max_dist = distance_matrix.max()

    denominator = max_dist - min_dist
    if denominator == 0:
        return distance_matrix
    else:
        normalized_matrix = (distance_matrix - min_dist) / denominator

    return normalized_matrix


def split(
    data: pd.DataFrame, exog_data: pd.DataFrame, train_percentage: float
) -> tuple[
    MinMaxScaler,
    dict[str, FloatArray | pd.DatetimeIndex],
    dict[str, FloatArray | pd.DatetimeIndex],
    dict[str, FloatArray | pd.DatetimeIndex],
]:
    """
    Splits the dataset into training, validation, and test sets.
    It scales the data, applies STL decomposition, and adds time-related features (time of day, day of week).

    Parameters:
    - data: DataFrame containing the main time series data.
    - exog_data: DataFrame containing the exogenous variables.
    - train_percentage: Percentage of training data.


    Returns:
    - data_scaler: The MinMaxScaler used for scaling the data.
    - train_data_with_features: Dictionary containing training data with features.
    - val_data_with_features: Dictionary containing validation data with features.
    - test_data_with_features: Dictionary containing test data with features.
    """
    from typing import cast

    # Define the end of the training period based on the train_percentage
    train_end = int(train_percentage * data.shape[0])

    train_data = data.iloc[:train_end]
    test_data = data.iloc[train_end:]

    # Scale the training and test data using MinMaxScaler
    data_scaler = MinMaxScaler()
    train_data_scaled = pd.DataFrame(
        data_scaler.fit_transform(  # type: ignore
            train_data
        ),
        columns=train_data.columns,
        index=train_data.index,
    )
    test_data_scaled = pd.DataFrame(
        data_scaler.transform(  # type: ignore
            test_data
        ),
        columns=test_data.columns,
        index=test_data.index,
    )

    # Scale the exogenous data
    train_exog = exog_data.iloc[:train_end]
    test_exog = exog_data.iloc[train_end:]
    exog_scaler = MinMaxScaler()
    train_exog_scaled = pd.DataFrame(
        exog_scaler.fit_transform(  # type: ignore
            train_exog
        ),
        columns=train_exog.columns,
        index=train_exog.index,
    )
    test_exog_scaled = pd.DataFrame(
        exog_scaler.transform(  # type: ignore
            test_exog
        ),
        columns=test_exog.columns,
        index=test_exog.index,
    )
    assert train_data.shape[0] == train_exog.shape[0]
    assert test_data.shape[0] == test_exog.shape[0]
    assert (train_data.index == train_exog.index).all()
    assert (test_data.index == test_exog.index).all()

    # Apply STL decomposition (Seasonal-Trend decomposition using LOESS) to the scaled data
    stl_train_data = pd.DataFrame(columns=data.columns)
    trend_train_data = pd.DataFrame(columns=data.columns)
    seasonal_train_data = pd.DataFrame(columns=data.columns)
    residual_train_data = pd.DataFrame(columns=data.columns)

    stl_test_data = pd.DataFrame(columns=data.columns)
    trend_test_data = pd.DataFrame(columns=data.columns)
    seasonal_test_data = pd.DataFrame(columns=data.columns)
    residual_test_data = pd.DataFrame(columns=data.columns)

    # Perform STL decomposition for each column in the training and test data
    for col in train_data.columns:
        result = STL(train_data_scaled[col], seasonal=23).fit()  # type: ignore
        stl_train_data[col] = result
        (
            trend_train_data[col],
            seasonal_train_data[col],
            residual_train_data[col],
        ) = (result.trend, result.seasonal, result.resid)  # type: ignore

        result = STL(test_data_scaled[col], seasonal=23).fit()  # type: ignore
        stl_test_data[col] = result
        (
            trend_test_data[col],
            seasonal_test_data[col],
            residual_test_data[col],
        ) = (result.trend, result.seasonal, result.resid)  # type: ignore

    # Split the training data into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * train_data.shape[0])

    train_index = cast(pd.DatetimeIndex, train_data.index[:train_size])
    val_index = cast(pd.DatetimeIndex, train_data.index[train_size:])
    test_index = cast(pd.DatetimeIndex, test_data.index)

    # Prepare data with features
    train_data_1: dict[str, "pd.DataFrame | pd.DatetimeIndex"] = {
        "data": train_data_scaled,
        "seasonal": seasonal_train_data,
        "trend": trend_train_data,
        "residual": residual_train_data,
        "index": train_index,
    }
    test_data_1: dict[str, "pd.DataFrame | pd.DatetimeIndex"] = {
        "data": test_data_scaled,
        "seasonal": seasonal_test_data,
        "trend": trend_test_data,
        "residual": residual_test_data,
        "index": test_index,
    }

    train_data_with_features: dict[str, FloatArray | pd.DatetimeIndex] = {
        "index": train_index,
    }
    val_data_with_features: dict[str, FloatArray | pd.DatetimeIndex] = {
        "index": val_index,
    }
    test_data_with_features: dict[str, FloatArray | pd.DatetimeIndex] = {
        "index": test_index,
    }

    data_type = list(train_data_1.keys())
    data_type.remove("index")

    # Add time-based features (e.g., time of day, day of week) to the data
    for key in data_type:
        train_val = train_data_1[key]
        assert isinstance(train_val, pd.DataFrame), "Expected DataFrame type"
        test_val = test_data_1[key]
        assert isinstance(test_val, pd.DataFrame), "Expected DataFrame type"
        train_data_with_features[key] = add_features(
            [train_val], add_time_of_day=True, add_day_of_week=True
        )
        test_data_with_features[key] = add_features(
            [test_val], add_time_of_day=True, add_day_of_week=True
        )

        val_data_with_features[key] = train_data_with_features[key][train_size:]
        train_data_with_features[key] = train_data_with_features[key][:train_size]

    # Add exogenous features
    train_data_with_features["exog"] = train_exog_scaled[:train_size].values  # type: ignore
    val_data_with_features["exog"] = train_exog_scaled[train_size:].values  # type: ignore
    test_data_with_features["exog"] = test_exog_scaled.values  # type: ignore

    return (
        data_scaler,
        train_data_with_features,
        val_data_with_features,
        test_data_with_features,
    )


def create_datasets(
    train_data_with_features: dict[str, FloatArray | pd.DatetimeIndex],
    val_data_with_features: dict[str, FloatArray | pd.DatetimeIndex],
    test_data_with_features: dict[str, FloatArray | pd.DatetimeIndex],
    poi_tensor: torch.Tensor,
    mask: torch.Tensor,
    input_length: int,
    horizon: int,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """
    Creates datasets for training, validation, and testing by generating sequences from multivariate time series data.

    Parameters:
    - train_data_with_features: Dictionary containing training data with features.
    - val_data_with_features: Dictionary containing validation data with features.
    - test_data_with_features: Dictionary containing test data with features.
    - poi_tensor: Tensor containing Points of Interest (POI) information.
    - mask: Mask tensor for POI data.
    - args: Argument object containing input_length (length of input sequences) and horizon (forecast horizon).

    Returns:
    - train_dataset: TimeSeriesDataset object for training.
    - val_dataset: TimeSeriesDataset object for validation.
    - test_dataset: TimeSeriesDataset object for testing.
    """

    # Initialize dictionaries to store input (X) and output (y) sequences for train, validation, and test sets
    X_train: dict[str, torch.Tensor] = {"data": torch.tensor([])}
    X_val: dict[str, torch.Tensor] = {"data": torch.tensor([])}
    X_test: dict[str, torch.Tensor] = {"data": torch.tensor([])}

    y_train: dict[str, torch.Tensor] = {"data": torch.tensor([])}
    y_val: dict[str, torch.Tensor] = {"data": torch.tensor([])}
    y_test: dict[str, torch.Tensor] = {"data": torch.tensor([])}

    # Define the data types that we will process (all keys except "indices")
    data_type = list(train_data_with_features.keys())
    data_type.remove("index")

    # Iterate over each data type and create sequences for train, validation, and test sets
    for key in data_type:
        train_features = train_data_with_features[key]
        assert isinstance(train_features, np.ndarray), (
            "Expected DataFrame type for training features"
        )
        val_features = val_data_with_features[key]
        assert isinstance(val_features, np.ndarray), (
            "Expected DataFrame type for validation features"
        )
        test_features = test_data_with_features[key]
        assert isinstance(test_features, np.ndarray), (
            "Expected DataFrame type for test features"
        )
        train_index = train_data_with_features["index"]
        assert isinstance(train_index, pd.DatetimeIndex), (
            "Expected DatetimeIndex type for training index"
        )
        val_index = val_data_with_features["index"]
        assert isinstance(val_index, pd.DatetimeIndex), (
            "Expected DatetimeIndex type for validation index"
        )
        test_index = test_data_with_features["index"]
        assert isinstance(test_index, pd.DatetimeIndex), (
            "Expected DatetimeIndex type for test index"
        )

        X_train[key], _, y_train[key], _ = create_sequences_multivariate(
            train_features,
            train_index,
            input_length,
            horizon,
        )

        X_val[key], _, y_val[key], _ = create_sequences_multivariate(
            val_features,
            val_index,
            input_length,
            horizon,
        )

        X_test[key], _, y_test[key], _ = create_sequences_multivariate(
            test_features,
            test_index,
            input_length,
            horizon,
        )

    # Create TimeSeriesDataset objects for train, validation, and test sets
    train_dataset = TimeSeriesDataset(
        data=X_train["data"],
        targets=y_train["data"],
        seasonal=X_train["seasonal"],
        trend=X_train["trend"],
        residual=X_train["residual"],
        exogenous=X_train["exog"],
        poi_data=poi_tensor,
        mask=mask,
    )
    val_dataset = TimeSeriesDataset(
        data=X_val["data"],
        targets=y_val["data"],
        seasonal=X_val["seasonal"],
        trend=X_val["trend"],
        residual=X_val["residual"],
        exogenous=X_val["exog"],
        poi_data=poi_tensor,
        mask=mask,
    )
    test_dataset = TimeSeriesDataset(
        data=X_test["data"],
        targets=y_test["data"],
        seasonal=X_test["seasonal"],
        trend=X_test["trend"],
        residual=X_test["residual"],
        exogenous=X_test["exog"],
        poi_data=poi_tensor,
        mask=mask,
    )

    return train_dataset, val_dataset, test_dataset
