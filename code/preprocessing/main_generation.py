import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common.generation.models import Critic, Encoder, Generator, ModelArgs
from data_processing.generate_external_data import download_weather
from data_processing.mobility_data_processing import (
    generate_hourly_transactions,
    generate_slot_data,
    get_registry_by_key,
    preprocess_sensor_data,
)
from generation.utils import ScenarioType, add_conditions, grid_building
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Arguments:
    scenario: ScenarioType
    selected_zone: str
    cond_dim: int
    input_dim: int
    horizon: int
    batch_size: int
    lr_rate: float
    num_epochs: int
    train_percentage: float
    grid_size: int
    padding: int
    kernel_size: int
    hidden_dim: int
    latent_dim: int
    proximity: bool
    quantity: float | None = None


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
    torch.save(  # type: ignore
        model.state_dict(), model_save_path
    )


def train_model(
    train_dataloader: DataLoader[TensorDataset],
    val_dataloader: DataLoader[TensorDataset],
    device: int | str | torch.device,
    model_save_dir: Path,
    s_coords: dict[int, tuple[int, int]],
    scenario: ScenarioType,
    num_epochs: int,
    model_args: ModelArgs,
) -> tuple[list[list[float]], list[list[float]], Encoder, Generator]:
    # Model parameters

    encoder = Encoder(model_args).to(device)
    generator = Generator(model_args).to(device)
    critic = Critic(model_args).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=1e-4,
        betas=(0.5, 0.9),
    )
    opt_C = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # Training parameters
    vae_train_list: list[float] = []
    vae_val_list: list[float] = []
    g_train_list: list[float] = []
    g_val_list: list[float] = []
    c_train_list: list[float] = []
    c_val_list: list[float] = []
    one = torch.FloatTensor([1])[0]
    minusone = one * -1
    minusone = minusone.to(device)
    one = one.to(device)

    generator_best = generator
    encoder_best = encoder

    for epoch in range(num_epochs):
        g_loss_batch = 0
        c_loss_batch = 0
        vae_loss_batch = 0
        critic.train()
        generator.train()
        encoder.train()

        for _, (data, conditions) in enumerate(train_dataloader):
            data = data.permute(0, 2, 1, 3, 4)
            data = data.to(device)
            conditions = conditions.permute(0, 2, 1, 3, 4)
            if scenario == "2nd":
                lat_indices = [lat for lat, _ in s_coords.values()]
                lon_indices = [lon for _, lon in s_coords.values()]
                mask_zero = conditions[:, 1, :, lat_indices, lon_indices] == 0
                conditions[:, 1, :, lat_indices, lon_indices] = torch.from_numpy(  # type: ignore
                    np.where(
                        mask_zero,
                        0,
                        torch.log(
                            1 / (conditions[:, 1, :, lat_indices, lon_indices] + 1)
                        ),
                    )
                ).float()

            conditions = conditions.to(device)

            # Encoder conditions
            optimizer.zero_grad()
            z_cond, mu_cond, logvar_cond, indices, map = encoder(conditions)

            # Generator
            noise = torch.randn(data.shape[0], args.latent_dim).to(device)
            z_new = torch.cat([noise, z_cond], dim=1)
            fake_data = generator(z_new, map, indices)

            # Critic
            opt_C.zero_grad()
            real_output = critic(data, conditions.detach())
            c_loss_real = torch.mean(real_output)
            c_loss_real.backward(minusone, retain_graph=True)  # type: ignore

            fake_output = critic(fake_data.detach(), conditions.detach())
            c_loss_fake = torch.mean(fake_output)
            c_loss_fake.backward(one)  # type: ignore

            # Critic update
            c_loss = c_loss_fake - c_loss_real
            opt_C.step()  # type: ignore

            for p in critic.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Encoder and Generator update
            recon_loss = reconstruction_loss(fake_data, data)
            kull_loss = kl_loss(mu_cond, logvar_cond)
            loss_vae = recon_loss + 0.1 * kull_loss
            loss_vae.backward(one, retain_graph=True)  # type: ignore
            fake_output = critic(fake_data, conditions)
            g_loss = -torch.mean(fake_output)
            g_loss.backward(one)  # type: ignore
            optimizer.step()  # type: ignore

            g_loss_batch += g_loss.item()
            c_loss_batch += c_loss.item()
            vae_loss_batch += loss_vae.item()

        g_loss = g_loss_batch / len(train_dataloader)
        c_loss = c_loss_batch / len(train_dataloader)
        loss = vae_loss_batch / len(train_dataloader)

        c_train_list.append(c_loss)
        vae_train_list.append(loss)
        g_train_list.append(g_loss)

        # Validation
        g_loss_val_batch = 0
        c_loss_val_batch = 0
        vae_loss_val_batch = 0
        encoder.eval()
        critic.eval()
        generator.eval()
        with torch.no_grad():
            for _, (data, conditions) in enumerate(val_dataloader):
                data = data.permute(0, 2, 1, 3, 4)
                data = data.to(device)
                conditions = conditions.permute(0, 2, 1, 3, 4)
                if scenario == "2nd":
                    lat_indices = [lat for lat, _ in s_coords.values()]
                    lon_indices = [lon for _, lon in s_coords.values()]
                    mask_zero = conditions[:, 1, :, lat_indices, lon_indices] == 0
                    conditions[:, 1, :, lat_indices, lon_indices] = torch.from_numpy(  # type: ignore
                        np.where(
                            mask_zero,
                            0,
                            torch.log(
                                1 / (conditions[:, 1, :, lat_indices, lon_indices] + 1)
                            ),
                        )
                    ).float()

                conditions = conditions.to(device)

                # Encoder
                z_cond, mu_cond, logvar_cond, indices, map = encoder(conditions)

                # Generator
                noise = torch.randn(data.shape[0], args.latent_dim).to(device)
                z_new = torch.cat([noise, z_cond], dim=1)
                fake_data = generator(z_new, map, indices)

                # Critic
                real_output = critic(data, conditions)
                c_loss_real = torch.mean(real_output)
                fake_output = critic(fake_data, conditions)
                c_loss_fake = torch.mean(fake_output)

                # Losses
                c_val_loss = c_loss_fake - c_loss_real

                recon_loss_val = reconstruction_loss(fake_data, data)
                kull_loss_val = kl_loss(mu_cond, logvar_cond)
                loss_val_vae = recon_loss_val + 0.1 * kull_loss_val

                g_val_loss = -torch.mean(fake_output)

                g_loss_val_batch += g_val_loss.item()
                c_loss_val_batch += c_val_loss.item()
                vae_loss_val_batch += loss_val_vae.item()

        g_loss_val = g_loss_val_batch / len(val_dataloader)
        c_loss_val = c_loss_val_batch / len(val_dataloader)
        loss_val = vae_loss_val_batch / len(val_dataloader)

        if len(vae_val_list) != 0:
            if loss_val < min(vae_val_list):
                generator_best = generator
                encoder_best = encoder

        g_train_list.append(g_loss)
        c_train_list.append(c_loss)
        vae_val_list.append(loss_val)
        print(
            f"Epoch {epoch} - Generator Loss: {g_loss} - VAE Loss: {loss} - Critic Loss : {c_loss} - Generator Loss "
            f"Val: {g_loss_val} - VAE Loss Val: {loss_val} - Critic Loss Val: {c_loss_val}"
        )

    train_losses = [vae_train_list, g_train_list, c_train_list]
    val_losses = [vae_val_list, g_val_list, c_val_list]
    save_model(generator_best, model_save_dir / "generator.pth")
    save_model(encoder_best, model_save_dir / "encoder.pth")
    return train_losses, val_losses, encoder_best, generator_best


def reconstruction_loss(recon: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    loss_type = nn.MSELoss(reduction="sum")
    loss = loss_type(recon, data)

    return loss / (
        recon.shape[0]
        * recon.shape[1]
        * recon.shape[2]
        * recon.shape[3]
        * recon.shape[4]
    )


def kl_loss(z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
    return kl.mean()


def generate(
    encoder_best: Encoder,
    generator_best: Generator,
    conditions: torch.Tensor,
    device: int | str | torch.device,
    latent_dim: int,
) -> torch.Tensor:
    encoder_best.eval()
    generator_best.eval()
    with torch.no_grad():
        conditions = conditions.permute(0, 2, 1, 3, 4)
        conditions = conditions.to(device)
        z_cond, _, _, indices, map = encoder_best(conditions)

        noise = torch.randn(conditions.shape[0], latent_dim).to(device)
        z_new = torch.cat([noise, z_cond], dim=1)
        fake_data = generator_best(z_new, map, indices)

    return fake_data


# %%

if __name__ == "__main__":
    from typing import cast, get_args

    import yaml

    args, _ = parser.parse_known_args()

    print(
        f"Running generation using configuration: {args.config}. "
        f"Execution may take a while..."
    )

    with open(Path("configs") / "generation" / (args.config + ".yml"), "r") as f:
        config = yaml.safe_load(f)

    args = Arguments(**config)

    data_dir = Path(os.getenv("DATA_DIR", "../../data/preprocessing"))
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    results_dir = Path(os.getenv("RESULTS_DIR", "../../results/preprocessing"))
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

    scenario_ = str(args.scenario)

    if scenario_ not in get_args(ScenarioType):
        raise ValueError(
            f"Invalid scenario: {scenario_}. Expected one of {get_args(ScenarioType)}."
        )
    scenario = cast(ScenarioType, scenario_)

    # Generate parking meter data
    transaction_data = pd.read_csv(data_dir / "transaction_data.csv")  # type: ignore

    hourly_transactions = generate_hourly_transactions(
        transaction_data, "transactions", freq="h"
    )

    with open(data_dir / "anagraficaParcometro.json", "r") as f:
        parkingmeters_registry = json.load(f)

    # Extract only the relevant columns from hourly transactions
    hourly_transactions = hourly_transactions[
        [reg["id"] for reg in parkingmeters_registry]
    ]

    start_date = cast(pd.DatetimeIndex, hourly_transactions.index).min()  # type: ignore
    end_date = cast(pd.DatetimeIndex, hourly_transactions.index).max()  # type: ignore
    full_hour_range = pd.date_range(  # type: ignore
        start=start_date, end=end_date, freq="1h"
    )

    final_parkingmeters_registry: dict[int, dict[str, Any]] = {}
    keys = list([int(key) for key in hourly_transactions.columns])
    for key in keys:
        parkingmeter_key = get_registry_by_key(parkingmeters_registry, key)
        final_parkingmeters_registry[key] = {}
        final_parkingmeters_registry[key]["lat"] = parkingmeter_key["lat"]
        final_parkingmeters_registry[key]["lng"] = parkingmeter_key["lng"]
        final_parkingmeters_registry[key]["data"] = (
            hourly_transactions[key]
            .reindex(  # type: ignore
                full_hour_range, fill_value=np.nan
            )
            .fillna(0)
            .resample("4h")
            .sum()
        )

    # Generate slots data
    with open(data_dir / "AnagraficaStallo.json", "r") as f:
        slots = json.load(f)

    with open(data_dir / "KPlace_Signals.json") as f:
        KPlace_signals = json.load(f)

    with open(data_dir / "StoricoStallo.json", "r") as f:
        slots_history = json.load(f)

    df_slots_registry = pd.DataFrame(slots)
    df_slots_registry["numeroStallo"] = df_slots_registry["numeroStallo"].astype(int)

    df_slots_history = pd.DataFrame(slots_history)
    df_slots_history["start"] = pd.to_datetime(  # type: ignore
        df_slots_history["start"]
    )
    df_slots_history["end"] = pd.to_datetime(  # type: ignore
        df_slots_history["end"]
    )

    df_final = preprocess_sensor_data(
        KPlace_signals, df_slots_registry, df_slots_history
    )
    df_occupied_slots = generate_slot_data(df_final, freq="h")

    # Load mapping dictionary
    with open(data_dir / "mapping_dict.json", "r") as f:
        mapping_dict = json.load(f)

    all_available_slots = mapping_dict["all_map"]["stalli"]
    df_occupied_slots = df_occupied_slots.reindex(  # type: ignore
        all_available_slots, axis=1, fill_value=0
    )
    keys = list([int(key) for key in df_occupied_slots.columns])

    final_slots_registry: dict[int, dict[str, Any]] = {}
    for key in keys:
        slots_key = get_registry_by_key(slots, int(key))
        final_slots_registry[key] = {}
        final_slots_registry[key]["lat"] = float(slots_key["lat"])
        final_slots_registry[key]["lng"] = float(slots_key["lng"])
        final_slots_registry[key]["id_strada"] = int(slots_key["id_strada"])
        final_slots_registry[key]["data"] = (
            df_occupied_slots[key]
            .reindex(  # type: ignore
                full_hour_range, fill_value=np.nan
            )
            .fillna(0)
            .resample("4h")
            .sum()
        )

    # Common time range for all data
    time_start = max(
        min(
            cast(pd.DatetimeIndex, final_slots_registry[key]["data"].index)[0]
            for key in final_slots_registry.keys()
        ),
        min(
            cast(pd.DatetimeIndex, final_parkingmeters_registry[key]["data"].index)[0]
            for key in final_parkingmeters_registry.keys()
        ),
    )
    time_end = min(
        max(
            cast(pd.DatetimeIndex, final_slots_registry[key]["data"].index)[-1]
            for key in final_slots_registry.keys()
        ),
        max(
            cast(pd.DatetimeIndex, final_parkingmeters_registry[key]["data"].index)[-1]
            for key in final_parkingmeters_registry.keys()
        ),
    )

    indices_parkingmeter: dict[int, tuple[int, int]] = {}
    indices_slots: dict[int, tuple[int, int]] = {}
    # Grid matrix building
    if args.scenario in ["1st", "3rd"]:
        (
            matrix,
            indices_slots_,
            indices_parkingmeter,
            scaler_slots,
            scaler_parkingmeters,
        ) = grid_building(
            parkingmeters=final_parkingmeters_registry,
            slots=final_slots_registry,
            grid_size=args.grid_size,
            scenario=args.scenario,
            mapping_dict=mapping_dict,
            time_start=time_start,
            time_end=time_end,
        )
        assert isinstance(indices_slots_, dict), "Expected indices_slots to be a dict"
        indices_slots = indices_slots_
        assert scaler_parkingmeters is not None, (
            "Expected scaler_parking meters to be defined"
        )
        tensor_matrix = torch.tensor(matrix, dtype=torch.float32).permute(3, 2, 0, 1)
        mask = (tensor_matrix != 0).float()
        if args.scenario == "3rd":
            # Generate weather data
            lat = 41.0726
            lon = 14.3323

            weather_data = download_weather(time_start, time_end, lat, lon)

            weather_data.index = pd.to_datetime(  # type: ignore
                weather_data.index  # type: ignore
            )
            precipitation_data_4h = pd.DataFrame(
                weather_data["precipitation"]
                .resample(  # type: ignore
                    "4H"
                )
                .sum()
            )

            precipitation_data_4h["precipitation_mask"] = precipitation_data_4h[
                "precipitation"
            ].apply(  # type: ignore
                lambda x: 1 if x > 0 else 0  # type: ignore
            )

            mask_weather = torch.zeros((mask.shape[0], 1, mask.shape[2], mask.shape[3]))

            for key in final_slots_registry.keys():
                (lat_index, lon_index) = indices_slots[key]
                mask_weather[:, 0, lat_index, lon_index] = torch.tensor(
                    precipitation_data_4h["precipitation_mask"].values  # type: ignore
                )
            for key in final_parkingmeters_registry.keys():
                (lat_index, lon_index) = indices_parkingmeter[key]
                mask_weather[:, 0, lat_index, lon_index] = torch.tensor(
                    precipitation_data_4h["precipitation_mask"].values  # type: ignore
                )

            mask = torch.cat([mask, mask_weather], dim=1)

    else:
        matrix, cond_matrix, indices_slots, scaler_slots, _ = grid_building(
            parkingmeters=final_parkingmeters_registry,
            slots=final_slots_registry,
            grid_size=args.grid_size,
            scenario=args.scenario,
            mapping_dict=mapping_dict,
            time_start=time_start,
            time_end=time_end,
        )
        assert isinstance(cond_matrix, np.ndarray), (
            "Expected cond_matrix to be a numpy array"
        )
        tensor_matrix = torch.tensor(matrix, dtype=torch.float32).permute(3, 2, 0, 1)
        cond = torch.tensor(cond_matrix, dtype=torch.float32).permute(3, 2, 0, 1)
        mask = (tensor_matrix != 0).float()
        mask = torch.cat([mask, cond], dim=1)

    # Windowing
    X_l: list[torch.Tensor] = []
    mask_x_l: list[torch.Tensor] = []
    for i in range(
        0, (len(tensor_matrix) // args.horizon) * args.horizon, args.horizon
    ):
        X_l.append(tensor_matrix[i : i + args.horizon])
        mask_x_l.append(mask[i : i + args.horizon])

    X = torch.stack(X_l)
    mask_x = torch.stack(mask_x_l)

    num_samples = X.shape[0]
    train_size = int(args.train_percentage * num_samples)

    # Dataloader creation
    train_data, val_data, train_conditions, val_conditions = (
        X[:train_size],
        X[train_size:],
        mask_x[:train_size],
        mask_x[train_size:],
    )

    train_dataset = TensorDataset(train_data, train_conditions)
    val_dataset = TensorDataset(val_data, val_conditions)

    train_dataloader = cast(
        DataLoader[TensorDataset],
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
        ),
    )
    val_dataloader = cast(
        DataLoader[TensorDataset],
        DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
        ),
    )

    # Training
    model_save_dir = results_dir / "generation" / f"{args.scenario}"
    if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True)

    dir_name = "train"
    model_save_dir = model_save_dir / dir_name

    if not model_save_dir.exists():
        model_save_dir.mkdir()

    print("Training generative model...")

    train_losses, val_losses, encoder_best, generator_best = train_model(
        train_dataloader,
        val_dataloader,
        device,
        model_save_dir,
        indices_slots,
        scenario=args.scenario,
        num_epochs=args.num_epochs,
        model_args=ModelArgs(
            grid_size=args.grid_size,  # Size of the grid
            input_dim=args.input_dim,  # Input dimension
            cond_dim=args.cond_dim,  # Conditional dimension
            hidden_dim=args.hidden_dim,  # Starting hidden dimension
            latent_dim=args.latent_dim,  # Latent dimension
            horizon=args.horizon,  # Sequence length
            kernel_size=args.kernel_size,  # Kernel size
            padding=args.padding,  # Padding
            use_proximity=args.proximity,  # If proximity is used
        ),
    )

    # Generating
    print("Generating...")

    val_conditions_generation = add_conditions(
        val_conditions,
        indices_parkingmeter,
        indices_slots,
        final_slots_registry,
        final_parkingmeters_registry,
        mapping_dict,
        scenario=args.scenario,
        quantity=args.quantity if args.scenario == "2nd" else None,
        selected_zone=args.selected_zone if args.scenario in ["1st", "2nd"] else None,
    )

    fake_data = generate(
        encoder_best, generator_best, val_conditions_generation, device, args.latent_dim
    )

    print("Generation completed.")
