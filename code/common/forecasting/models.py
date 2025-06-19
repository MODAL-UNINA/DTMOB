from typing import NotRequired, TypedDict

import torch
import torch.nn.functional as F
from torch import nn


class ModelArgs(TypedDict):
    num_nodes: int
    node_dim: int
    input_len: int
    input_dim: int
    embed_dim: int
    output_len: int
    num_layer: int
    temp_dim_tid: int
    temp_dim_diw: int
    time_of_day_size: int
    day_of_week_size: int
    if_T_i_D: bool
    if_D_i_W: bool
    if_node: bool
    if_poi: bool
    if_gps: bool
    num_poi_types: int
    exogenous_dim: NotRequired[int]
    gps_embedding: NotRequired[torch.Tensor]


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()  # type: ignore
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


class MV_Forecasting(nn.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        super().__init__()  # type: ignore

        # Attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.if_poi = model_args["if_poi"]

        self.if_gps = model_args["if_gps"]

        # Spatial embeddings
        if self.if_spatial:
            if self.if_gps:
                if "gps_embedding" in model_args:
                    self.gps_embedding = model_args["gps_embedding"]
                    self.node_emb = nn.Parameter(
                        torch.empty(self.num_nodes, self.node_dim)
                    )
                    self.node_emb.data = self.gps_embedding.to(torch.float32)
                else:
                    # If no GPS embedding is provided, initialize node embeddings randomly
                    self.node_emb = nn.Parameter(
                        torch.empty(self.num_nodes, self.node_dim)
                    )
                    nn.init.xavier_uniform_(self.node_emb)
            else:
                self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
                nn.init.xavier_uniform_(self.node_emb)

        # Temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # Embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        # Encoding
        self.hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_day_in_week)
            + self.temp_dim_diw * int(self.if_time_in_day)
            + self.node_dim * int(self.if_poi)
        )

        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_layer)
            ]
        )

        # Regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

        # Attention for exogenous features
        self.exogenous_dim = model_args.get("exogenous_dim", 0)
        if self.exogenous_dim > 0:
            self.exogenous_encoder = nn.Linear(self.exogenous_dim, self.hidden_dim)

            self.attention_layer = nn.MultiheadAttention(
                embed_dim=self.hidden_dim, num_heads=1, batch_first=True
            )

        # POI embeddings
        if self.if_poi:
            self.num_poi_types = model_args["num_poi_types"]

            self.poi_type_embedding = nn.Embedding(self.num_poi_types, self.node_dim)
            self.distance_encoder = nn.Linear(1, self.node_dim)

            self.poi_encoder = nn.Sequential(
                nn.Linear(2 * self.node_dim, self.node_dim), nn.ReLU()
            )

            self.aggregator = AttentionAggregator(self.node_dim)

    def forward(
        self,
        history_data: torch.Tensor,
        exogenous_data: torch.Tensor | None,
        poi_data: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Prepare data
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).to(torch.long)
            ]
        else:
            time_in_day_emb = None

        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * self.day_of_week_size).to(torch.long)
            ]
        else:
            day_in_week_emb = None

        # Time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb: list[torch.Tensor] = []

        if self.if_spatial:
            # Expand node embeddings
            node_emb.append(
                self.node_emb.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .transpose(1, 2)
                .unsqueeze(-1)
            )

        # Temporal embeddings
        tem_emb: list[torch.Tensor] = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # POI embeddings
        if self.if_poi:
            assert poi_data is not None, "POI data must be provided if if_poi is True"
            poi_types = poi_data[..., 0].long()
            poi_distances = poi_data[..., 1].unsqueeze(-1)

            # Embedding category type and distance
            poi_type_emb = self.poi_type_embedding(
                poi_types
            )  # [B, N, num_POI, poi_embed_dim]
            poi_distance_emb = self.distance_encoder(
                poi_distances
            )  # [B, N, num_POI, poi_embed_dim]

            # Concatenate embeddings
            poi_combined = torch.cat(
                [poi_type_emb, poi_distance_emb], dim=-1
            )  # [B, N, num_POI, 2 * poi_embed_dim]
            poi_emb = self.poi_encoder(poi_combined)  # [B, N, num_POI, poi_embed_dim]

            # Mask to exclude invalid POI
            poi_emb = poi_emb * mask

            poi_emb = self.aggregator(poi_emb)  # [B, N, poi_embed_dim]

            poi_emb = poi_emb.transpose(1, 2).unsqueeze(-1)

            hidden = torch.cat([hidden] + [poi_emb], dim=1)

        # Attention for exogenous data
        if exogenous_data is not None:
            exogenous_encoded = F.relu(self.exogenous_encoder(exogenous_data))

            hidden_transposed = hidden.squeeze(-1).transpose(1, 2)  # [B, N, hidden_dim]

            attn_output, _ = self.attention_layer(
                hidden_transposed, exogenous_encoded, exogenous_encoded
            )
            hidden = hidden + attn_output.transpose(1, 2).unsqueeze(
                -1
            )  # Residual connection

        # Encoding
        hidden = self.encoder(hidden)

        # Regression
        prediction = self.regression_layer(hidden)

        return prediction


class AttentionAggregator(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()  # type: ignore
        self.query = nn.Linear(embed_dim, 1)

    def forward(
        self,
        poi_embeddings: torch.Tensor,  # [B, num_parkingmeters, num_poi, embed_dim]
    ) -> torch.Tensor:
        attention_weights = self.query(poi_embeddings).squeeze(
            -1
        )  # [B, num_parkingmeters, num_poi]
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Compute weighted sum of POI embeddings
        weighted_sum = torch.sum(
            poi_embeddings * attention_weights.unsqueeze(-1), dim=2
        )
        return weighted_sum  # [B, num_parkingmeters, embed_dim]


class Modelcomplete(nn.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        super().__init__()  # type: ignore

        self.residual = MV_Forecasting(model_args)
        self.trend = MV_Forecasting(model_args)
        self.seasonal = MV_Forecasting(model_args)

    def forward(
        self,
        seasonal: torch.Tensor,
        residual: torch.Tensor,
        trend: torch.Tensor,
        exogenous: torch.Tensor | None = None,
        poi_tensor: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seasonal = self.seasonal(seasonal, exogenous, poi_tensor, mask)
        residual = self.residual(residual, exogenous, poi_tensor, mask)
        trend = self.trend(trend, exogenous, poi_tensor, mask)

        out = seasonal + residual + trend

        return out, seasonal, residual, trend
