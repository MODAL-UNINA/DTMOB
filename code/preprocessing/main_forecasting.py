import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import holidays
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common.forecasting.models import ModelArgs, Modelcomplete
from data_processing.generate_external_data import (
    DataType,
    download_weather,
    generate_events,
    generate_poi,
)
from data_processing.mobility_data_processing import (
    generate_hourly_transactions,
    generate_road_data,
    preprocess_sensor_data,
)
from forecasting.utils import (
    FloatArray,
    create_datasets,
    haversine_matrix,
    normalize_distance_matrix,
    split,
)
from sklearn.manifold import MDS
from torch.utils.data import DataLoader


@dataclass
class Arguments:
    data_type: DataType
    target_channel: int
    input_length: int
    horizon: int
    batch_size: int
    use_gps: bool
    num_nodes: int
    node_dim: int
    input_dim: int
    embed_dim: int
    num_layer: int
    temp_dim_tid: int
    temp_dim_diw: int
    time_of_day_size: int
    day_of_week_size: int
    if_T_i_D: bool
    if_D_i_W: bool
    use_poi: bool
    if_node: bool
    exogenous_dim: int
    num_poi_types: int
    num_epochs: int
    train_percentage: float


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--config", type=str)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # type: ignore
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_model(model: nn.Module, model_save_path: Path) -> None:
    torch.save(model.state_dict(), model_save_path)  # type: ignore


def train_model(
    train_dataloader: DataLoader[dict[str, torch.Tensor]],
    val_dataloader: DataLoader[dict[str, torch.Tensor]],
    criterion: nn.Module,
    device: int | str | torch.device,
    model_save_dir: Path,
    num_epochs: int,
    target_channel: int,
    model_args: ModelArgs,
    dist_matrix: torch.Tensor | None = None,
) -> tuple[list[float], list[float], int, Modelcomplete]:
    # Model parameters
    if model_args["if_gps"]:
        assert dist_matrix is not None, "dist_matrix must be provided if if_gps is True"
        mds = MDS(
            n_components=model_args["node_dim"],
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress=False,
        )
        gps_embeddings = mds.fit_transform(  # type: ignore
            dist_matrix.numpy()  # type: ignore
        )

        gps_embeddings = torch.tensor(gps_embeddings, dtype=torch.float32).to(
            device
        )  # [N, node_dim]
        model_args["gps_embedding"] = gps_embeddings

    model = Modelcomplete(model_args).to(device)

    patience = 50
    early_stop = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    decay_rate = 0.5

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=decay_rate
    )

    best_epoch = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    min_lr = 1e-5
    best_validate_loss = np.inf
    validate_score_non_decrease_count = 0
    model_best = model

    lr_update_interval = 5
    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()

                seasonal = batch["seasonal"].to(device)
                trend = batch["trend"].to(device)
                residual = batch["residual"].to(device)

                exogenous = batch["exogenous"].to(device)

                poi_tensor = batch["poi_data"].to(device)
                mask = batch["mask"].to(device)
                out, seasonal, residual, trend = model(
                    seasonal, residual, trend, exogenous, poi_tensor, mask
                )

                targets = batch["targets"].to(device)
                targets = targets[..., [target_channel]]

                loss = criterion(out, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()  # type: ignore
                train_loss += loss.item()

            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            val_pred: list[FloatArray] = []
            val_true: list[FloatArray] = []
            save_model(model, model_save_dir / "model_last.pth")
            with torch.no_grad():
                for batch in val_dataloader:
                    seasonal = batch["seasonal"].to(device)
                    trend = batch["trend"].to(device)
                    residual = batch["residual"].to(device)

                    exogenous = batch["exogenous"].to(device)

                    poi_tensor = batch["poi_data"].to(device)
                    mask = batch["mask"].to(device)
                    out, seasonal, residual, trend = model(
                        seasonal, residual, trend, exogenous, poi_tensor, mask
                    )

                    targets = batch["targets"].to(device)
                    targets = targets[..., [target_channel]]

                    loss = criterion(out, targets)

                    val_loss += loss.item()
                    val_pred.append(out.cpu().numpy())
                    val_true.append(targets.cpu().numpy())

            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            print(
                "Epoch: ",
                epoch,
                "Train Loss: ",
                round(train_loss, 5),
                "Val Loss: ",
                round(val_loss, 5),
            )

            if scheduler:
                is_best_for_now = False
                if best_validate_loss > val_loss + 1e-5:
                    best_validate_loss = val_loss
                    is_best_for_now = True
                    validate_score_non_decrease_count = 0
                    model_best = model
                    best_epoch = epoch
                else:
                    validate_score_non_decrease_count += 1

                if (validate_score_non_decrease_count + 1) % lr_update_interval == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    if current_lr > min_lr:
                        print(f"Current learning rate: {current_lr}")
                        model.load_state_dict(model_best.state_dict())
                        scheduler.step()
                if is_best_for_now:
                    model_save_path_best = model_save_dir / "model_best.pth"
                    save_model(model, model_save_path_best)
            # Early stop
            if early_stop and validate_score_non_decrease_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    except KeyboardInterrupt:
        print("Interrupted")
        save_model(model, model_save_dir / "model_last.pth")

    return train_losses, val_losses, best_epoch, model_best


def predict(
    model: Modelcomplete,
    test_dataloader: DataLoader[dict[str, torch.Tensor]],
    target_channel: int,
) -> tuple[FloatArray, FloatArray]:
    model.eval()
    predictions_l: list[FloatArray] = []
    actuals_l: list[FloatArray] = []

    with torch.no_grad():
        for batch in test_dataloader:
            targets = batch["targets"].to(device)
            seasonal = batch["seasonal"].to(device)
            trend = batch["trend"].to(device)
            residual = batch["residual"].to(device)
            exogenous = batch["exogenous"].to(device)

            poi_tensor = batch["poi_data"].to(device)
            mask = batch["mask"].to(device)
            out, _, _, _ = model(seasonal, residual, trend, exogenous, poi_tensor, mask)

            out = out.squeeze(-1)
            targets = targets[..., target_channel]
            predictions_l.append(out.cpu().numpy())
            actuals_l.append(targets.cpu().numpy())

    predictions = np.concatenate(predictions_l, axis=0)
    actuals = np.concatenate(actuals_l, axis=0)

    return predictions, actuals


if __name__ == "__main__":
    from typing import cast

    import yaml

    args, _ = parser.parse_known_args()

    print(
        f"Running forecasting using configuration: {args.config}. "
        f"Execution may take a while..."
    )

    with open(Path("configs") / "forecasting" / (args.config + ".yml"), "r") as f:
        config = yaml.safe_load(f)

    args = Arguments(**config)

    data_dir = Path(os.getenv("DATA_DIR", "../../data/preprocessing"))
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    results_dir = Path(os.getenv("RESULTS_DIR", "../../results/preprocessing"))
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

    parkingmeters_registry: list[dict[str, Any]] = []
    hourly_transactions = pd.DataFrame()
    roads_data = pd.DataFrame()
    # Generate data
    if args.data_type in ["transactions", "amount"]:
        transaction_data = pd.read_csv(  # type: ignore
            data_dir / "transaction_data.csv"
        )
        with open(data_dir / "anagraficaParcometro.json", "r") as f:
            parkingmeters_registry = json.load(f)

        hourly_transactions = generate_hourly_transactions(
            transaction_data, args.data_type
        )

        hourly_transactions = hourly_transactions[
            [reg["id"] for reg in parkingmeters_registry]
        ]
        all_data_index = cast(pd.DatetimeIndex, hourly_transactions.index)
    else:
        with open(data_dir / "AnagraficaStallo.json", "r") as f:
            slots = json.load(f)

        with open(data_dir / "KPlace_Signals.json") as f:
            KPlace_signals = json.load(f)

        with open(data_dir / "StoricoStallo.json", "r") as f:
            slots_history = json.load(f)

        slots_df = pd.DataFrame(slots)
        slots_df["numeroStallo"] = slots_df["numeroStallo"].astype(int)

        df_slots_history = pd.DataFrame(slots_history)
        df_slots_history["start"] = pd.to_datetime(  # type: ignore
            df_slots_history["start"]
        )
        df_slots_history["end"] = pd.to_datetime(  # type: ignore
            df_slots_history["end"]
        )

        # Generate roads data
        df_final = preprocess_sensor_data(KPlace_signals, slots_df, df_slots_history)

        assert len(df_final) > 0, "Dataframe df_final is empty."

        roads_data = generate_road_data(df_final, slots_df)

        all_data_index = cast(pd.DatetimeIndex, roads_data.index)

    start_date = all_data_index.min()  # type: ignore
    end_date = all_data_index.max()  # type: ignore

    # Generate weather data
    # Replace lat and lon with real values
    lat = 41.0726
    lon = 14.3323

    weather_data = download_weather(start_date, end_date, lat, lon)

    weather_data = weather_data.loc[all_data_index]

    # Generate events data
    events = pd.read_csv(  # type: ignore
        data_dir / "events.csv", index_col=0
    )

    north, south, east, west = 41.093810, 41.06036506, 14.358893, 14.324020

    events_data = generate_events(events, south, west, north, east)

    events_data.index = pd.to_datetime(  # type: ignore
        events_data.index
    )
    events_data = events_data.reindex(  # type: ignore
        pd.date_range(  # type: ignore
            all_data_index.min(),  # type: ignore
            all_data_index.max()  # type: ignore
            + pd.Timedelta(days=1)
            - pd.Timedelta(hours=1),
            freq="H",
        )
    )
    events_data = cast(
        pd.DataFrame,
        events_data.fillna(method="ffill"),  # type: ignore
    )
    events_data = cast(
        pd.DataFrame,
        events_data.fillna(0),  # type: ignore
    )

    events_data = events_data.loc[all_data_index]
    years = all_data_index.year.unique()  # type: ignore

    it_holidays = pd.to_datetime(  # type: ignore
        [date for year in years for date, _ in holidays.Italy(years=year).items()]  # type: ignore
    )

    is_holiday = pd.DataFrame(index=all_data_index, columns=["is_holiday"], data=0)

    is_holiday.loc[
        cast(pd.DatetimeIndex, is_holiday.index).normalize().isin(it_holidays),  # type: ignore
        "is_holiday",
    ] = 1

    easter = pd.date_range(  # type: ignore
        "2024-03-28", "2024-04-01", freq="D"
    )

    days_christmas = [
        "12-23",
        "12-24",
        "12-25",
        "12-26",
        "12-27",
        "12-28",
        "12-29",
        "12-30",
        "12-31",
        "01-01",
        "01-02",
        "01-03",
        "01-04",
        "01-05",
        "01-06",
    ]
    christmas = [
        pd.date_range(  # type: ignore
            start=f"{year}-12-23", end=f"{year + 1}-01-06", freq="D"
        )
        for year in years
    ]

    christmas = pd.DatetimeIndex(np.concatenate(christmas))

    august = pd.date_range(  # type: ignore
        "2024-08-01", "2024-08-31", freq="D"
    )

    days_our_holidays = easter.union(  # type: ignore
        christmas
    ).union(  # type: ignore
        august
    )

    our_holidays = pd.DataFrame(index=all_data_index, columns=["our_holidays"], data=0)

    our_holidays.loc[
        our_holidays.index.normalize().isin(days_our_holidays), "our_holidays"  # type: ignore
    ] = 1

    exog_data = pd.concat([weather_data, events_data, is_holiday, our_holidays], axis=1)

    # Generate POI data
    if args.data_type in ["transactions", "amount"]:
        poi_dist, poi_categories = generate_poi(
            parkingmeters_registry, south, west, north, east, args.data_type
        )

    else:
        with open(data_dir / "roads.json", "r") as f:
            roads_registry = json.load(f)

        unique_slots_roads = slots_df["id_strada"].unique()  # type: ignore
        roads_registry = [v for v in roads_registry if v["sqlID"] in unique_slots_roads]

        poi_dist, poi_categories = generate_poi(
            data=roads_registry,
            south=south,
            west=west,
            north=north,
            east=east,
            data_type=args.data_type,
        )

    poi_categories = np.expand_dims(
        poi_categories.values,  # type: ignore
        axis=-1,
    ).astype(np.float32)
    poi_dist = np.expand_dims(
        poi_dist.values,  # type: ignore
        axis=-1,
    ).astype(np.float32)

    # Mask to consider only POIs within 0.5 km from the parking meter
    mask = poi_dist <= 0.5

    poi_dist_masked = poi_dist * mask

    # Normalize distance matrix
    denominator = poi_dist_masked.max() - poi_dist_masked.min()
    if denominator == 0:
        pass
    else:
        poi_dist_masked = (poi_dist_masked - poi_dist_masked.min()) / denominator

    poi_data_ = np.concatenate([poi_categories, poi_dist_masked], axis=-1)

    poi_tensor = torch.tensor(poi_data_, dtype=torch.float32)

    mask = torch.tensor(mask, dtype=torch.float32)

    dist_matrix: torch.Tensor | None = None
    if args.use_gps:
        df_parkingmeters_registry = pd.DataFrame(parkingmeters_registry)
        df_parkingmeters_registry = df_parkingmeters_registry[["id", "lat", "lng"]]
        df_parkingmeters_registry["id"] = df_parkingmeters_registry["id"].astype(float)

        df_parkingmeters_registry = df_parkingmeters_registry.T.to_dict()  # type: ignore

        gps_coordinates = torch.tensor(
            [
                [
                    float(df_parkingmeters_registry[parkingmeter]["lat"]),
                    float(df_parkingmeters_registry[parkingmeter]["lng"]),
                ]
                for parkingmeter in df_parkingmeters_registry.keys()
            ]
        )

        dist_matrix = haversine_matrix(gps_coordinates)
        dist_matrix = normalize_distance_matrix(dist_matrix)

    if args.data_type in ["transactions", "amount"]:
        (
            data_scaler,
            train_data_with_features,
            val_data_with_features,
            test_data_with_features,
        ) = split(hourly_transactions, exog_data, args.train_percentage)
    else:
        (
            data_scaler,
            train_data_with_features,
            val_data_with_features,
            test_data_with_features,
        ) = split(roads_data, exog_data, args.train_percentage)

    train_dataset, val_dataset, test_dataset = create_datasets(
        train_data_with_features,
        val_data_with_features,
        test_data_with_features,
        poi_tensor,
        mask,
        args.input_length,
        args.horizon,
    )

    # Dataloader creation
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Training
    model_save_dir = results_dir / "forecasting"
    if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True)

    dir_name = "train"
    model_save_dir = model_save_dir / dir_name

    if not model_save_dir.exists():
        model_save_dir.mkdir()

    criterion = nn.HuberLoss(reduction="mean")

    print("Training model...")

    train_losses, val_losses, best_epoch, model_best = train_model(
        train_dataloader,
        val_dataloader,
        criterion,
        device,
        model_save_dir,
        num_epochs=args.num_epochs,
        target_channel=args.target_channel,
        model_args=ModelArgs(
            num_nodes=args.num_nodes,  # Number of nodes
            node_dim=args.node_dim,  # Spatial embedding dimension
            input_len=args.input_length,  # Input sequence length
            input_dim=args.input_dim,  # Input dimension
            embed_dim=args.embed_dim,  # Embedding dimension
            output_len=args.horizon,  # Output sequence length
            num_layer=args.num_layer,  # Number of MLP layers
            temp_dim_tid=args.temp_dim_tid,  # Daily temporal embedding dimension
            temp_dim_diw=args.temp_dim_diw,  # Weekly temporal embedding dimension
            time_of_day_size=args.time_of_day_size,  # Number of hours in a day
            day_of_week_size=args.day_of_week_size,  # Number of days in a week
            if_T_i_D=args.if_T_i_D,  # Use daily temporal embedding
            if_D_i_W=args.if_D_i_W,  # Use weekly temporal embedding
            if_node=args.if_node,  # Use spatial embedding
            if_gps=args.use_gps,  # Use GPS embedding
            if_poi=args.use_poi,  # Use POI embedding
            num_poi_types=args.num_poi_types,  # Number of POI types
            exogenous_dim=args.exogenous_dim,  # Exogenous dimension
        ),
        dist_matrix=dist_matrix,
    )

    # Prediction
    print("Predicting...")
    pred_series, actual_series = predict(
        model_best, test_dataloader, args.target_channel
    )

    pred_series = np.maximum(pred_series, 0)
    actual_series1 = data_scaler.inverse_transform(  # type: ignore
        actual_series.reshape(-1, actual_series.shape[2])
    ).reshape(actual_series.shape)
    pred_series1 = data_scaler.inverse_transform(  # type: ignore
        pred_series.reshape(-1, pred_series.shape[2])
    ).reshape(pred_series.shape)

    print("Prediction completed.")
