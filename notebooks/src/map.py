from typing import Any, Sequence, TypedDict

import contextily as cx  # type: ignore
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from .general import ZoneDataMapping, ZoneDictZoneDataMapping


class MapData(TypedDict):
    """
    TypedDict for the map data structure.
    """

    parkingmeter_positions: pd.DataFrame
    roads_gdf: gpd.GeoDataFrame
    sensors: pd.DataFrame


class ZoneBoundary(TypedDict):
    """Data structure for the boundaries of a zone."""

    bounds: list[list[float]]
    label: str


class ParkingmeterPosition(TypedDict):
    """Data structure for parking meter positions."""

    lat: float
    lng: float
    road_id: int
    id: int
    zone_id: int


class ParkingslotPosition(TypedDict):
    """Data structure for parking meter positions."""

    lat: float
    lng: float
    road_id: int
    id: int
    zone_id: int


class PolygonData(TypedDict):
    """Data structure for polygon data."""

    type: str
    coordinates: tuple[tuple[tuple[float, float], ...], ...]


class LineStringData(TypedDict):
    """Data structure for line string data."""

    type: str
    coordinates: tuple[tuple[float, float], ...]


ShapeData = PolygonData | LineStringData


class RoadData(TypedDict):
    """Data structure for road data."""

    road_id: int
    zone_id: int
    geometry: ShapeData


class InnerOutMapData(TypedDict):
    """Data structure for the output of the map data."""

    zones: dict[str, ZoneBoundary]
    center: list[float]
    parking_meters: list[ParkingmeterPosition]
    parking_slots: list[ParkingslotPosition]
    roads: list[RoadData]


MAP_ZONE_COLORS = [
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


# Private function to postprocess the data
def _postprocess_parkingmeter_positions(
    data: pd.DataFrame, zone_dict: ZoneDictZoneDataMapping, zone_names: list[str]
) -> pd.DataFrame:
    data["id_parcometro"] = data["id_parcometro"].astype(int)
    data = data.rename(columns={"id_parcometro": "id", "id_strada": "road_id"})
    data["zone_id"] = data["id"].apply(  # type: ignore
        lambda x: int(  # type: ignore
            [
                zone_name
                for zone_name in zone_names
                if x in zone_dict[zone_name]["parcometro"]
            ][0].split("_")[1]
        )
        + 1
    )
    return data


def _postprocess_roads_gdf(
    data: gpd.GeoDataFrame, zone_dict: ZoneDictZoneDataMapping, zone_names: list[str]
) -> gpd.GeoDataFrame:
    data.insert(  # type: ignore
        1,
        "zone_id",
        data["road_id"].apply(  # type: ignore
            lambda x: int(  # type: ignore
                [
                    zone_name
                    for zone_name in zone_names
                    if x in zone_dict[zone_name]["strade"]
                ][0].split("_")[1]
            )
            + 1
        ),
    )
    return data


def _postprocess_sensors(
    data: pd.DataFrame, zone_dict: ZoneDictZoneDataMapping, zone_names: list[str]
) -> pd.DataFrame:
    data = data.rename(columns={"id_strada": "road_id"})

    sensors_mapping = {
        zone_name: zone_dict[zone_name]["stalli"] for zone_name in zone_names
    }
    sensors_sets = [set(v) for v in sensors_mapping.values()]
    assert len(set.intersection(*sensors_sets)) == 0, (  # type: ignore
        "There are sensors that are present in multiple zones. "
        "This is not allowed and will cause issues in the application."
    )
    sensors_mapping_inv = {
        x: zone_name for zone_name, v in sensors_mapping.items() for x in v
    }
    data["zone_id"] = data["id"].apply(  # type: ignore
        lambda x:  # type: ignore
        int(sensors_mapping_inv[x].split("_")[1]) + 1
        if x in sensors_mapping_inv
        else -1
    )
    data = data[data["zone_id"] != -1]  # Filter out sensors not in any zone
    return data


def preprocess(data: dict[str, Any], zone_dict: ZoneDictZoneDataMapping) -> MapData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    zone_names = sorted(
        [zone_name for zone_name in zone_dict.keys() if zone_name != "all_map"],
        key=lambda v: int(v.split("_")[1]),
    )

    return MapData(
        parkingmeter_positions=_postprocess_parkingmeter_positions(
            data["parkingmeter_positions"], zone_dict, zone_names
        ),
        roads_gdf=_postprocess_roads_gdf(data["roads_gdf"], zone_dict, zone_names),
        sensors=_postprocess_sensors(data["sensors"], zone_dict, zone_names),
    )


def get_map_data(
    meshgrid: ZoneDataMapping,
    parkingmeters_selection: pd.DataFrame,
    slots_selection: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> InnerOutMapData:
    import numpy as np

    zone_names = sorted(
        [zone_name for zone_name in meshgrid.keys() if zone_name != "all_map"],
        key=lambda v: int(v.split("_")[1]),
    )

    mapping_labels = {
        zone_name: zone_name.replace("_", " ").capitalize() for zone_name in zone_names
    }

    grid = meshgrid["all_map"]["grid"]

    lat_grid = grid[0]
    lon_grid = grid[1]

    center_lat = float(np.mean(lat_grid))
    center_lon = float(np.mean(lon_grid))

    zones: dict[str, ZoneBoundary] = {}

    for zone_name in zone_names:
        if zone_name == "all_map":
            continue

        zone_data = meshgrid[zone_name]

        lat_grid, lng_grid = zone_data["grid"]
        min_lat, max_lat = np.min(lat_grid).item(), np.max(lat_grid).item()
        min_lng, max_lng = np.min(lng_grid).item(), np.max(lng_grid).item()
        zone_label = mapping_labels[zone_name]

        zones[zone_name] = ZoneBoundary(
            bounds=[[min_lat, min_lng], [max_lat, max_lng]],
            label=zone_label,
        )

    zones = {k: zones[k] for k in sorted(zones, key=lambda x: int(x.split("_")[1]))}
    parking_meters: list[ParkingmeterPosition] = list(
        parkingmeters_selection.to_dict("index").values()  # type: ignore
    )
    parking_slots: list[ParkingslotPosition] = list(
        slots_selection.to_dict("index").values()  # type: ignore
    )

    roads = [
        RoadData(
            road_id=row.road_id,  # type: ignore
            zone_id=row.zone_id,  # type: ignore
            geometry=row.geometry.__geo_interface__,  # type: ignore
        )
        for _, row in roads_gdf.iterrows()  # type: ignore
    ]

    return InnerOutMapData(
        zones=zones,
        center=[center_lat, center_lon],
        parking_meters=parking_meters,
        parking_slots=parking_slots,
        roads=roads,
    )


# Rounded‑corner radius in degrees (~30 metres).
_ROUNDED_RADIUS_DEG: float = 3e-4  # 0.0003 deg ≈ 33 m @ the Equator


def _latlng_bounds(
    bounds: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    """Return *(min_lat, max_lat, min_lng, max_lng)* for a 2-point bounding box."""
    (lat1, lng1), (lat2, lng2) = bounds  # type: ignore[arg-type]
    return min(lat1, lat2), max(lat1, lat2), min(lng1, lng2), max(lng1, lng2)


def _add_rounded_rect(
    ax: Axes,
    bounds: Sequence[Sequence[float]],
    colour: str,
    radius: float = _ROUNDED_RADIUS_DEG,
) -> None:
    """Draw a rounded rectangle matching *bounds* onto *ax*."""
    min_lat, max_lat, min_lng, max_lng = _latlng_bounds(bounds)
    width, height = max_lng - min_lng, max_lat - min_lat

    # Matplotlib expects (x, y) == (lng, lat).
    rect = FancyBboxPatch(
        (min_lng, min_lat),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        edgecolor=colour,
        facecolor=colour,
        linewidth=2,
        alpha=0.15,
    )
    ax.add_patch(rect)


def _within_bounds(lat: float, lng: float, bounds: Sequence[Sequence[float]]) -> bool:
    """Return *True* iff *(lat, lng)* lies within *bounds*."""
    min_lat, max_lat, min_lng, max_lng = _latlng_bounds(bounds)
    return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng


def plot_zones(
    # grid: Sequence[NDArray[np.float64]],
    zone_data: InnerOutMapData,
    zone_name: str,
) -> Figure:
    from typing import cast

    zones = zone_data["zones"]
    parking_meters = zone_data["parking_meters"]
    parking_slots = zone_data["parking_slots"]
    roads = zone_data["roads"]

    if zone_name != "all_map" and zone_name not in zones:
        raise ValueError(f"Zone '{zone_name}' not found in the provided zone data.")

    global_lat: list[float] = []
    global_lng: list[float] = []

    fig, ax = plt.subplots(  # type: ignore
        figsize=(15, 10), dpi=100
    )

    # Zones (rounded rectangles)
    for idx, (zone_name_, zdata) in enumerate(zones.items()):
        if zone_name != "all_map" and zone_name_ != zone_name:
            continue

        color = MAP_ZONE_COLORS[idx % len(MAP_ZONE_COLORS)]
        bounds = zdata["bounds"]
        _add_rounded_rect(ax, bounds, color)

        # Add label roughly in the centre of the rectangle.
        min_lat, max_lat, min_lng, max_lng = _latlng_bounds(bounds)
        # ax.text(  # type: ignore
        #     (min_lng + max_lng) / 2,
        #     (min_lat + max_lat) / 2,
        #     zdata.get("label", zone_name),
        #     ha="center",
        #     va="center",
        #     fontsize=9,
        #     weight="bold",
        #     color="black",
        #     zorder=6,
        # )
        global_lat.extend([min_lat, max_lat])
        global_lng.extend([min_lng, max_lng])

    def _in_any_zone(obj: ParkingmeterPosition | ParkingslotPosition) -> bool:
        return any(
            _within_bounds(obj["lat"], obj["lng"], z["bounds"])
            for zone_name_, z in zones.items()
            if zone_name == "all_map" or zone_name_ == zone_name
        )

    parking_meters = [m for m in parking_meters if _in_any_zone(m)]
    parking_slots = [s for s in parking_slots if _in_any_zone(s)]

    # ax.scatter(  # type: ignore
    #     lng_grid, lat_grid, s=0, c="gray"
    # )

    if parking_slots:
        ax.scatter(  # type: ignore
            [s["lng"] for s in parking_slots],
            [s["lat"] for s in parking_slots],
            s=8,
            c="steelblue",
            label="Parking Slots",
            zorder=6,
        )
        global_lat.extend(s["lat"] for s in parking_slots)
        global_lng.extend(s["lng"] for s in parking_slots)

    if parking_meters:
        ax.scatter(  # type: ignore
            [m["lng"] for m in parking_meters],
            [m["lat"] for m in parking_meters],
            s=20,
            c="crimson",
            marker="o",
            edgecolors="none",
            label="Parking Meters",
            zorder=7,
        )
        global_lat.extend(m["lat"] for m in parking_meters)
        global_lng.extend(m["lng"] for m in parking_meters)

    # Roads
    for road in roads:
        if zone_name != "all_map" and road["zone_id"] - 1 != int(
            zone_name.split("_")[1]
        ):
            continue
        geom = road.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = cast(Sequence[tuple[float, float]], geom["coordinates"])
            ax.plot(  # type: ignore
                [lng for lng, _ in coords],
                [lat for _, lat in coords],
                color="yellow",
                linewidth=2,
                alpha=0.7,
                label=f"{road.get('road_name', 'Road')}"
                if road["road_id"] == 0
                else None,  # avoid legend spam
                zorder=4,
            )
            global_lat.extend(lat for _, lat in coords)
            global_lng.extend(lng for lng, _ in coords)
        elif geom.get("type") == "Polygon":
            coords = cast(Sequence[tuple[float, float]], geom["coordinates"][0])
            ax.fill(  # type: ignore
                [lng for lng, _ in coords],
                [lat for _, lat in coords],
                color="yellow",
                alpha=0.3,
                zorder=3,
            )
            global_lat.extend(lat for _, lat in coords)
            global_lng.extend(lng for lng, _ in coords)

    for _, (zone_name_, zdata) in enumerate(zones.items()):
        if zone_name != "all_map" and zone_name_ != zone_name:
            continue

        bounds = zdata["bounds"]

        min_lat, max_lat, min_lng, max_lng = _latlng_bounds(bounds)
        ax.text(  # type: ignore
            (min_lng + max_lng) / 2,
            (min_lat + max_lat) / 2,
            zdata.get("label", zone_name),
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color="black",
            zorder=8,
        )

    if global_lat and global_lng:
        margin = 0.001  # add small padding around combined extent
        ax.set_xlim(min(global_lng) - margin, max(global_lng) + margin)
        ax.set_ylim(min(global_lat) - margin, max(global_lat) + margin)

    ax.set_aspect(  # type: ignore
        "equal", adjustable="box", anchor="C"
    )
    # Add the basemap
    cx.add_basemap(  # type: ignore
        ax,
        crs="EPSG:4326",
        source=cx.providers.CartoDB.Voyager,  # type: ignore
        attribution_size=4,
    )

    ax.set_title(  # type: ignore
        "Selected Zone", fontsize=14
    )
    ax.set_xticks(  # type: ignore
        []
    )
    ax.set_yticks(  # type: ignore
        []
    )
    ax.set_xlabel(  # type: ignore
        "Longitude", fontsize=12
    )
    ax.set_ylabel(  # type: ignore
        "Latitude", fontsize=12
    )
    ax.legend(  # type: ignore
        loc="lower right", frameon=True, fontsize=12, framealpha=0.8
    )
    ax.grid(  # type: ignore
        True, linestyle="--", alpha=0.5
    )
    fig.tight_layout()

    return fig
