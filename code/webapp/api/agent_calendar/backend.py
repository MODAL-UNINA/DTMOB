from typing import cast

import pandas as pd
import plotly.graph_objects as go  # type: ignore
from numpy import object_
from numpy.typing import NDArray

from api.agent_calendar.data import AgentCalendarDataMapping
from api.general.data import ZoneDataMapping


def preprocess_calendar(
    calendar: AgentCalendarDataMapping,
    selected_date: pd.Timestamp,
    zone_dict: ZoneDataMapping,
) -> go.Figure:
    from datetime import time, timedelta

    name_zones = {zone_name: zone_name for zone_name in zone_dict.keys()}
    for zone_name in name_zones:
        if zone_name != "all_map":
            name_zones[zone_name] = f"Zone {zone_name.split('_')[1]}"

    zone_dict = {name_zones[zone_name]: value for zone_name, value in zone_dict.items()}

    zone_dict_: dict[str, str] = {}
    for zone_name, value in zone_dict.items():
        zone_dict_[value["code"]] = zone_name
    dates = pd.date_range(selected_date, periods=7)  # type: ignore
    dates = [date.strftime("%Y-%m-%d") for date in dates]
    calendar_week = AgentCalendarDataMapping(**{d: calendar[d] for d in dates})

    calendar_df = pd.DataFrame(columns=["agent_id", "date", "zone", "start", "finish"])
    for day, schedule in calendar_week.items():
        for agent, shift in schedule.items():
            for zone in shift["zone"]:
                calendar_df = pd.concat(
                    [
                        calendar_df,
                        pd.DataFrame(
                            {
                                "agent_id": agent,
                                "date": day,
                                "zone": zone_dict_[zone],
                                "start": shift["start"],
                                "finish": shift["end"],
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

    calendar_df["date"] = pd.to_datetime(calendar_df["date"])  # type: ignore
    calendar_df["start"] = calendar_df["start"].astype(str).str.zfill(2)
    finish1 = calendar_df["finish"].astype(str).str.zfill(2)
    finish1 = finish1.map(lambda x: "23:59" if x == "24" else str(x) + ":00")
    calendar_df["finish"] = finish1

    start_df: pd.Series = pd.to_datetime(  # type: ignore
        calendar_df["start"], format="%H"
    )
    end_df: pd.Series = pd.to_datetime(  # type: ignore
        calendar_df["finish"], format="%H:%M"
    )

    calendar_df["start"] = start_df.dt.time
    finish = end_df.dt.time
    finish2 = finish.apply(  # type: ignore
        lambda x: (  # type: ignore
            "24:00:00"
            if x
            == pd.to_datetime(  # type: ignore
                "23:59", format="%H:%M"
            ).time()
            else x
        )
    )
    calendar_df["finish"] = finish2

    min_date = cast(pd.Timestamp, calendar_df["date"].min())
    max_date = cast(pd.Timestamp, calendar_df["date"].max())
    start_of_week = min_date - timedelta(days=min_date.weekday())
    end_of_week = max_date + timedelta(days=(6 - max_date.weekday()))

    fig = go.Figure()

    unique_zones = cast(
        NDArray[object_],
        calendar_df["zone"].unique(),  # type: ignore
    )
    colors_new = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#1abc9c",
        "#e74c3c",
    ]
    zone_colors = {
        zone: k
        for zone, k in zip(
            sorted(unique_zones, key=lambda item: int(item.split(" ")[1])),
            colors_new,
            strict=False,
        )
    }

    agents = cast(
        NDArray[object_],
        calendar_df["agent_id"].unique(),  # type: ignore
    )

    hour_height = 1
    total_hours = 16

    total_days = (end_of_week - start_of_week).days + 1

    day_width = 150

    total_width = total_days * day_width

    current_date = start_of_week
    day_labels: list[str] = []
    x_tickvals: list[float] = []
    previous_day_offset = 0
    while current_date <= end_of_week:
        x_offset = previous_day_offset
        previous_day_offset += day_width

        for hour in range(8, 25):
            y_pos = (24 - hour) * hour_height
            fig.add_shape(  # type: ignore
                type="line",
                x0=x_offset,
                x1=x_offset + 1,
                y0=y_pos,
                y1=y_pos,
                line=dict(color="rgba(150, 150, 150, 0.6)", width=1),
            )

        spacing = 1
        padding = 2
        day_margin = 2
        agent_width = (day_width - (len(agents) - 1) * spacing) / len(agents)
        for agent_idx, agent in enumerate(agents):
            day_schedule = cast(
                pd.DataFrame,
                calendar_df[
                    (calendar_df["agent_id"] == agent)
                    & (calendar_df["date"].dt.date == current_date.date())
                ],
            )

            x_start = (
                x_offset + agent_idx * agent_width + spacing * agent_idx + day_margin
            )
            x_end = x_start + agent_width - 2 * day_margin

            day_schedule.reset_index(drop=True, inplace=True)
            if len(day_schedule) == 1:
                schedule = day_schedule.iloc[0]  # type: ignore
                start_hour = cast(
                    int,
                    schedule["start"].hour,  # type: ignore
                )
                end_hour = cast(
                    int,
                    schedule["finish"].hour  # type: ignore
                    if isinstance(schedule["finish"], time)
                    else 24,
                )

                y_start = (24 - start_hour) * hour_height
                y_end = (24 - end_hour) * hour_height

                fill_color = zone_colors[schedule["zone"]]
                x_start = x_start - padding
                x_end = x_end + padding

                fig.add_shape(  # type: ignore
                    type="rect",
                    x0=x_start,
                    x1=x_end,
                    y0=y_start,
                    y1=y_end,
                    line=dict(color="rgba(150, 150, 150, 0.6)", width=1),
                    fillcolor=fill_color,
                    opacity=0.7,
                    legendgroup=schedule["zone"],  # type: ignore
                )

                fig.add_annotation(  # type: ignore
                    x=(x_start + x_end) / 2,
                    y=(y_start + y_end) / 2,
                    text=f"AGENT {agent}",
                    showarrow=False,
                    font=dict(size=10),
                    align="center",
                    textangle=90,
                )
            elif len(day_schedule) > 1:
                x_start = x_start - padding
                x_end = x_end + padding

                y_start: int | None = None
                y_end: int | None = None

                for i, schedule in day_schedule.iterrows():  # type: ignore
                    i = cast(int, i)
                    start_hour = cast(
                        int,
                        schedule["start"].hour,  # type: ignore
                    )
                    end_hour = cast(
                        int,
                        schedule["finish"].hour  # type: ignore
                        if isinstance(schedule["finish"], time)
                        else 24,
                    )

                    y_start = (24 - start_hour) * hour_height
                    y_end = (24 - end_hour) * hour_height

                    x_start_adjusted = x_start + i * agent_width / len(day_schedule)
                    x_end_adjusted = x_end - (
                        len(day_schedule) - i - 1
                    ) * agent_width / len(day_schedule)

                    fill_color = zone_colors[schedule["zone"]]
                    fig.add_shape(  # type: ignore
                        type="rect",
                        x0=x_start_adjusted,
                        x1=x_end_adjusted,
                        y0=y_start,
                        y1=y_end,
                        line=dict(color="rgba(150, 150, 150, 0.6)", width=1),
                        fillcolor=fill_color,
                        opacity=0.7,
                        legendgroup=schedule["zone"],  # type: ignore
                    )

                if y_start is not None and y_end is not None:
                    fig.add_annotation(  # type: ignore
                        x=(x_start + x_end) / 2,
                        y=(y_start + y_end) / 2,
                        text=f"AGENT {agent}",
                        showarrow=False,
                        font=dict(size=10),
                        align="center",
                        textangle=90,
                    )

        day_labels.append(
            f"{current_date.strftime('%A')}<br>{current_date.strftime('%Y-%m-%d')}"
        )
        x_tickvals.append(x_offset + 0.5 * day_width)

        fig.add_shape(  # type: ignore
            type="line",
            x0=x_offset + day_width,
            x1=x_offset + day_width,
            y0=0,
            y1=total_hours * hour_height,
            line=dict(color="rgba(80, 80, 80, 0.8)", width=1),
        )

        current_date += timedelta(days=1)

    final_x_offset = total_days * day_width

    for hour in range(9, 24):
        y_pos = (24 - hour) * hour_height
        fig.add_shape(  # type: ignore
            type="line",
            x0=0,
            x1=final_x_offset,
            y0=y_pos,
            y1=y_pos,
            line=dict(color="rgba(150, 150, 150, 0.6)", width=1),
        )

    start_offset = 0
    fig.add_shape(  # type: ignore
        type="line",
        x0=start_offset,
        x1=start_offset,
        y0=0,
        y1=total_hours * hour_height,
        line=dict(color="rgba(80, 80, 80, 0.8)", width=1),
    )

    legend_box_height = 0.2
    legend_box_width = 6
    legend_padding = 40

    legend_total_width = (
        len(zone_colors) * (legend_box_width + legend_padding) - legend_padding
    )

    legend_start_x = (total_width - legend_total_width) / 2

    legend_start_y = -0.5

    zone_colors = {
        k: v
        for k, v in sorted(
            zone_colors.items(), key=lambda item: int(item[0].split(" ")[1])
        )
    }

    for i, zone in enumerate(zone_colors):
        x_position = legend_start_x + i * (legend_box_width + legend_padding)

        fig.add_shape(  # type: ignore
            type="rect",
            x0=x_position,
            x1=x_position + legend_box_width,
            y0=legend_start_y - legend_box_height,
            y1=legend_start_y,
            line=dict(color="rgba(150, 150, 150, 0.6)", width=1),
            fillcolor=zone_colors[zone],
            opacity=0.7,
        )

        fig.add_annotation(  # type: ignore
            x=x_position + legend_box_width + 1,
            y=legend_start_y - legend_box_height / 2,
            text=zone,
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="middle",
        )

    fig.update_layout(  # type: ignore
        title="",
        showlegend=False,
        xaxis=dict(
            title="",
            tickvals=x_tickvals,
            ticktext=day_labels,
            calendar="gregorian",
            range=[
                -1,
                total_width + 10,
            ],
            tickangle=0,
            tickmode="array",
            tickprefix=" ",
            ticks="inside",
            showgrid=False,
            ticklen=6,
            side="top",
            showline=False,
            linewidth=2,
        ),
        yaxis=dict(
            range=[-1.5, total_hours + 0.5],
            tickmode="array",
            ticktext=[f"{i:02d}:00" for i in range(8, 25)],
            tickvals=[24 - i for i in range(8, 25)],
            showgrid=False,
        ),
        height=800,
        width=2400,
        plot_bgcolor="white",
        margin=dict(l=50, r=0, t=0, b=0),
    )

    return fig
