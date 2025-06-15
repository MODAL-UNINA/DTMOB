import time
from typing import Any, Literal

import geopandas as gpd
import openmeteo_requests  # type: ignore
import osmnx as ox  # type: ignore
import pandas as pd
import requests
import requests_cache
from geopy.distance import geodesic  # type: ignore
from openmeteo_requests.Client import OpenMeteoRequestsError  # type: ignore
from retry_requests import retry  # type: ignore

DataType = Literal["transactions", "amount", "roads"]


def generate_poi(
    data: list[dict[str, Any]],
    south: float,
    west: float,
    north: float,
    east: float,
    data_type: DataType,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts and processes Points of Interest (POI) within a specified bounding box.
    Computes distances from POIs to given locations (e.g., parking meters or roads) and
    categorizes POIs.

    Parameters:
    - data: List of dictionaries containing data (parking meters or roads) with coordinates.
    - south, west, north, east: Floats specifying the bounding box coordinates for POI extraction.
    - data_type: String specifying if data represents "transactions"/"amount" (parking meters) or "roads".

    Returns:
    - poi_distances: DataFrame with distances between each location and each POI.
    - poi_categories: DataFrame mapping each location to the category of each POI.
    """
    from typing import cast

    poi_data = pd.DataFrame()

    print("Extracting and processing Points of Interest (POI) data...")

    # POI's extraction
    poi_data = cast(
        gpd.GeoDataFrame,
        ox.geometries_from_bbox(  # type: ignore
            north, south, east, west, tags={"amenity": True}
        ),
    )
    poi_data.reset_index(drop=True, inplace=True)

    # List of amenities to include
    amenities_good = [
        "police",
        "cafe",
        "school",
        "pharmacy",
        "post_office",
        "kindergarten",
        "bar",
        "atm",
        "restaurant",
        "bank",
        "pub",
        "fast_food",
        "hospital",
        "university",
        "courthouse",
        "dentist",
        "doctors",
        "prep_school",
        "clinic",
        "library",
        "public_building",
        "place_of_worship",
        "townhall",
        "theatre",
        "marketplace",
        "veterinary",
        "arts_centre",
        "social_facility",
    ]

    poi_data = poi_data[
        poi_data["amenity"].isin(  # type: ignore
            amenities_good
        )
    ]

    poi_data["geometry"] = poi_data["geometry"].apply(  # type: ignore
        lambda geom: geom.centroid  # type: ignore
        if geom.geom_type == "Polygon" or geom.geom_type == "MultiPolygon"  # type: ignore
        else geom  # type: ignore
    )

    # Mapping of POI categories
    categories_mapping = {
        "police": "services",
        "cafe": "food_and_drink",
        "school": "education",
        "pharmacy": "healthcare",
        "post_office": "services",
        "kindergarten": "education",
        "bar": "food_and_drink",
        "atm": "finance",
        "restaurant": "food_and_drink",
        "bank": "finance",
        "pub": "food_and_drink",
        "fast_food": "food_and_drink",
        "hospital": "healthcare",
        "university": "education",
        "courthouse": "services",
        "dentist": "healthcare",
        "doctors": "healthcare",
        "prep_school": "education",
        "clinic": "healthcare",
        "library": "education",
        "public_building": "services",
        "place_of_worship": "cultural",
        "townhall": "services",
        "theatre": "cultural",
        "marketplace": "commercial",
        "veterinary": "healthcare",
        "arts_centre": "cultural",
        "social_facility": "healthcare",
    }

    poi_data["category"] = poi_data["amenity"].map(  # type: ignore
        categories_mapping
    )

    categories = list(poi_data["category"].unique())  # type: ignore

    # Mapping of categories to integers
    category_dict = {category: i for i, category in enumerate(categories)}

    if data_type in ["transactions", "amount"]:
        # Create a dictionary with the parking meters and their coordinates
        parkingmeters: dict[int, dict[str, float]] = {}
        for parkingmeter in data:
            parkingmeters[parkingmeter["id"]] = {
                "lat": parkingmeter["lat"],
                "lng": parkingmeter["lng"],
            }

        # Compute the distance between each parking meter and each POI
        poi_distances = pd.DataFrame(
            columns=poi_data.index, index=list(parkingmeters.keys())
        )

        for _, parc in enumerate(parkingmeters.keys()):
            for idx in poi_data.index:
                coords1 = (  # type: ignore
                    poi_data.loc[idx, "geometry"].y,  # type: ignore
                    poi_data.loc[idx, "geometry"].x,  # type: ignore
                )
                coords2 = (
                    float(parkingmeters[parc]["lat"]),
                    float(parkingmeters[parc]["lng"]),
                )
                poi_distances.loc[parc, idx] = geodesic(coords1, coords2).kilometers  # type: ignore

        poi_distances = poi_distances.sort_index()  # type: ignore

        # Create a dataframe with the categories of each POI for each parking meter
        poi_categories = pd.DataFrame(
            columns=poi_data.index, index=list(parkingmeters.keys())
        )

        for parc in parkingmeters.keys():
            for idx in poi_data.index:
                poi_categories.loc[parc, idx] = category_dict[
                    poi_data.loc[idx, "category"]
                ]

        poi_categories = poi_categories.sort_index()  # type: ignore

    else:
        # Create a dictionary with the roads and their coordinates
        roads: dict[int, list[dict[str, Any]]] = {}
        for road in data:
            if road["sqlID"] not in roads:
                roads[road["sqlID"]] = []

            center = road["geofences"][0]["center"]
            if not isinstance(center, list):
                center = [center]
            roads[road["sqlID"]] += road["geofences"][0]["path"] + center

        # Compute the distance between each road and each POI
        poi_distances = pd.DataFrame(columns=poi_data.index, index=list(roads.keys()))

        for road in roads.keys():
            for idx in poi_data.index:
                poi_distances.loc[road, idx] = min(
                    [
                        geodesic(  # type: ignore
                            (roads[road][i]["lat"], roads[road][i]["lng"]),
                            (
                                poi_data.loc[idx, "geometry"].y,  # type: ignore
                                poi_data.loc[idx, "geometry"].x,  # type: ignore
                            ),
                        ).kilometers
                        for i in range(len(roads[road]))
                    ]
                )

        poi_distances = poi_distances.sort_index()  # type: ignore

        # Create a dataframe with the categories of each POI for each road
        poi_categories = pd.DataFrame(columns=poi_data.index, index=list(roads.keys()))

        for road in roads.keys():
            for idx in poi_data.index:
                poi_categories.loc[road, idx] = category_dict[
                    poi_data.loc[idx, "category"]
                ]

        poi_categories = poi_categories.sort_index()  # type: ignore

    return poi_distances, poi_categories


def generate_events(
    events: pd.DataFrame, south: float, west: float, north: float, east: float
) -> pd.DataFrame:
    """
    Process events data.

    Parameters:
    - events: DataFrame with events data.
    - south, west, north, east: Floats specifying the bounding box coordinates for filtering events.

    Returns:
    - events_final: DataFrame with counts of events grouped by date and type.
    """

    print("Processing events data...")

    # Remove rows with missing latitude or longitude values
    events = events.dropna(  # type: ignore
        subset=["Latitude", "Longitude"]
    )

    # Convert latitude and longitude to float type for consistency
    events["Latitude"] = events["Latitude"].astype(float)
    events["Longitude"] = events["Longitude"].astype(float)

    # Filter events based on geographic bounding box
    events = events[
        (events["Latitude"] >= south)
        & (events["Latitude"] <= north)
        & (events["Longitude"] >= west)
        & (events["Longitude"] <= east)
    ]

    # Convert 'Date' column to datetime format, extracting only the date portion
    events["Date"] = pd.to_datetime(  # type: ignore
        events["Date"]
        .str.split(" ")  # type: ignore
        .str[0]
    )

    # Identify and remove duplicate event entries, keeping the first occurrence
    duplicates = events.duplicated()  # type: ignore
    if duplicates.any():  # type: ignore
        events.drop_duplicates(inplace=True, ignore_index=True, keep="first")

    # Group events by 'Date' and 'Type', then count occurrences
    events_final = (
        events.groupby(  # type: ignore
            ["Date", "Type"]
        )
        .size()
        .unstack(fill_value=0)
    )

    # Sort the final DataFrame by date in ascending order
    events_final.sort_values(  # type: ignore
        by="Date", inplace=True
    )

    events_types = [
        "Arte e Mostre",
        "Concerti",
        "Cultura ed altri eventi",
        "Feste, Fiere e Sagre",
        "Locali e Pub",
        "Rassegne, Festival, Manifestazioni",
        "Teatro",
    ]
    events_final = events_final.reindex(  # type: ignore
        events_types, axis=1, fill_value=0
    )

    return events_final


def download_weather(
    start_date: pd.Timestamp, end_date: pd.Timestamp, lat: float, lon: float
) -> pd.DataFrame:
    """
    Downloads meteorological data from the remote OpenMeteo API within a specified date range.

    Args:
    - start_date (Timestamp): Start date for downloading meteorological data.
    - end_date (Timestamp): End date for downloading meteorological data.
    - lat (float): Latitude of the location for which to download meteorological data.
    - lon (float): Longitude of the location for which to download meteorological data.

    Returns:
    - DataFrame: A DataFrame containing the downloaded meteorological data.

    """

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession(
        "openmeteo_cache", use_cache_dir=True, expire_after=-1
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"

    all_hourly_data = pd.DataFrame()
    start_date_download = start_date - pd.Timedelta(days=1)
    start_date_download = start_date_download.strftime("%Y-%m-%d")
    end_date_download = end_date + pd.Timedelta(days=1)
    end_date_download = end_date_download.strftime("%Y-%m-%d")

    print("Downloading weather data from OpenMeteo API...")

    params = {  # type: ignore
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date_download,
        "end_date": end_date_download,
        "hourly": [
            "precipitation",
            "temperature_2m",
            "wind_speed_10m",
            "relative_humidity_2m",
        ],
        "timezone": "Europe/Berlin",
    }

    try:
        # Request weather data from Open-Meteo API
        responses = openmeteo.weather_api(  # type: ignore
            url, params=params
        )

        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        assert hourly is not None, "Hourly data is missing in the response."
        hourly_precipitation = hourly.Variables(0).ValuesAsNumpy()  # type: ignore
        hourly_temperature_2m = hourly.Variables(1).ValuesAsNumpy()  # type: ignore
        hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()  # type: ignore
        hourly_relative_humidity_2m = hourly.Variables(3).ValuesAsNumpy()  # type: ignore

        # Construct DataFrame for hourly weather data
        hourly_data = {
            "date": pd.date_range(  # type: ignore
                start=pd.to_datetime(  # type: ignore
                    hourly.Time(),  # type: ignore
                    unit="s",
                    utc=True,
                ),
                end=pd.to_datetime(  # type: ignore
                    hourly.TimeEnd(),  # type: ignore
                    unit="s",
                    utc=True,
                ),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }
        hourly_data["temperature_2m"] = hourly_temperature_2m  # type: ignore
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m  # type: ignore
        hourly_data["precipitation"] = hourly_precipitation  # type: ignore
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m  # type: ignore

        all_hourly_data = pd.DataFrame(data=hourly_data)

        # Pause to avoid exceeding API request limits
        time.sleep(5)

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")

    except OpenMeteoRequestsError as om_err:
        print(f"API error occurred: {om_err}")
        if "Minutely API request limit exceeded" in str(om_err):
            print("Waiting 60 seconds before retrying...")
            time.sleep(60)

    # Initialize an empty DataFrame for processed weather data
    weather_data = pd.DataFrame()

    all_hourly_data["date"] = pd.to_datetime(  # type: ignore
        all_hourly_data["date"]
    ).dt.tz_localize(None)

    # Process and filter weather variables within the requested date range
    hourly_precipitation = all_hourly_data[["date", "precipitation"]].set_index(  # type: ignore
        "date"
    )
    hourly_precipitation.index = hourly_precipitation.index - pd.Timedelta(hours=1)
    weather_data["precipitation"] = hourly_precipitation.loc[start_date:end_date]

    hourly_temperature_2m = all_hourly_data[["date", "temperature_2m"]].set_index(  # type: ignore
        "date"
    )
    weather_data["temperature"] = hourly_temperature_2m.loc[start_date:end_date]

    hourly_wind_speed_10m = all_hourly_data[["date", "wind_speed_10m"]].set_index(  # type: ignore
        "date"
    )
    weather_data["wind"] = hourly_wind_speed_10m.loc[start_date:end_date]

    hourly_relative_humidity_2m = all_hourly_data[
        ["date", "relative_humidity_2m"]
    ].set_index(  # type: ignore
        "date"
    )
    weather_data["humidity"] = hourly_relative_humidity_2m.loc[start_date:end_date]

    return weather_data
