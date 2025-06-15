import json
import traceback
from datetime import datetime

import pandas as pd
from api.agent_calendar.views import get_available_calendar_dates, get_calendar
from api.distrib.views import (
    get_fines_image,
    get_occupancy_image,
    get_transactions_amount_image,
    get_transactions_count_image,
)
from api.forecast.views import (
    get_available_forecasting_dates,
    get_available_forecasting_parkingmeters,
    get_available_forecasting_roads,
    get_plot_forecast_amount,
    get_plot_forecast_roads,
    get_plot_forecast_transactions,
)
from api.general.views import (
    get_area_id_label_mapping,
    get_available_dates,
    get_date,
    get_hour_slot_range,
    get_hour_slots_items,
    get_legality_status_name,
    get_parkingmeter_name,
    get_parkingmeters,
    get_parkingslot_name,
    get_parkingslots,
    get_road_id,
    get_zone_name,
)
from api.map.views import do_get_map_data
from api.stats.views import get_stats_describe_data
from api.whatif.views import (
    WhatIfScenarioType,
    WhatIfDataKind,
    get_scenario,
    validate_data_kind,
    get_available_whatif_scenario_dates,
    get_quantity,
    get_whatif_cumulative_plot,
    get_whatif_distributions,
    get_whatif_heatmaps,
    run_generation,
)
from django.conf import settings

from django.contrib.auth.views import LoginView
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

logfile = settings.LOGS_DIR / "django.log"


def get_ip_addr(request: HttpRequest) -> str:
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        ip = x_forwarded_for.split(",")[0]
    else:
        ip = str(request.META.get("REMOTE_ADDR"))
    return ip


class CustomLoginView(LoginView):
    template_name = "login.html"


def get_post_reply(request: HttpRequest) -> JsonResponse:
    try:
        # Parse JSON body for data
        data = json.loads(request.body)

        action = data["action"]
        args = data.get("args", {})

        req_time = f"{datetime.now()}"
        with open(logfile, "a") as f:
            f.write(
                f"[{req_time}] Called action '{action}' with args: {args} "
                f"from user IP Address: {get_ip_addr(request)}\n"
            )
        print(
            f"[{req_time}] Called action '{action}' with args: {args} "
            f"from user IP Address: {get_ip_addr(request)}"
        )

        success_out = (
            f"Action '{action}' with args: {args} "
            f"from user IP Address: {get_ip_addr(request)} successful."
        )

        zone_name = "all_map"
        if "zone_id" in args:
            zone_id = args["zone_id"]
            if zone_id == "":
                zone_id = "0"
            zone_id = int(zone_id)
            try:
                zone_name = get_zone_name(zone_id)
            except Exception as e:
                message = traceback.format_exc()
                print(message)
                print(e)
                return JsonResponse(
                    {"error": "Invalid 'zone_id' provided."},
                    status=202,
                )

        date: pd.Timestamp | None = None
        if "selected_date" in args:
            try:
                date = get_date(args["selected_date"])
            except Exception as e:
                message = traceback.format_exc()
                print(message)
                print(e)
                return JsonResponse(
                    {"error": "Invalid 'selected_date' format provided."},
                    status=202,
                )

        scenario: WhatIfScenarioType | None = None
        if "scenario" in args:
            out = get_scenario(args["scenario"])
            if isinstance(out, dict):
                return JsonResponse(out, status=202)
            scenario = out

        quantity: int | None = None
        if "quantity" in args:
            try:
                quantity = get_quantity(args["quantity"])
            except Exception as e:
                message = traceback.format_exc()
                print(message)
                print(e)
                return JsonResponse(
                    {"error": "Invalid 'quantity' format provided."},
                    status=202,
                )

        kind: WhatIfDataKind | None = None
        if "kind" in args:
            out = validate_data_kind(args["kind"], scenario)
            if isinstance(out, dict):
                return JsonResponse(out, status=202)
            kind = out

        if action == "check_server_status":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse({"status": "running"})
        if action == "get_zone_names":
            zone_id_label_map = get_area_id_label_mapping()
            if len(zone_id_label_map) == 0:
                return JsonResponse({"error": "No zone names found."}, status=500)
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(zone_id_label_map)
        if action == "get_available_dates":
            available_dates = get_available_dates()
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(available_dates)
        if action == "get_map_data":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(do_get_map_data())
        if action == "get_available_distrib_dates":
            available_dates = get_available_dates()
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(available_dates)
        if action == "get_distrib_parkingmeters":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(get_parkingmeters(int(args["zone_id"])))
        if action == "get_available_distrib_parkingmeters":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(
                {
                    zone_id: get_parkingmeters(zone_id)
                    for zone_id in get_area_id_label_mapping().keys()
                }
            )
        if action == "get_distrib_parkingslots":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(get_parkingslots(int(args["zone_id"])))
        if action == "get_available_distrib_parkingslots":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(
                {
                    zone_id: get_parkingslots(zone_id)
                    for zone_id in get_area_id_label_mapping().keys()
                }
            )
        if action == "get_distr_transactions_count":
            hour_range = get_hour_slot_range(int(args["hourslot"]))
            parkingmeter = get_parkingmeter_name(int(args["parkingmeter"]))
            img_str = get_transactions_count_image(
                zone_name, date, hour_range, parkingmeter
            )
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse({"transactions_count": img_str})
        if action == "get_distr_transactions_amount":
            hour_range = get_hour_slot_range(int(args["hourslot"]))
            parkingmeter = get_parkingmeter_name(int(args["parkingmeter"]))
            img_str = get_transactions_amount_image(
                zone_name, date, hour_range, parkingmeter
            )
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse({"transactions_amount": img_str})
        if action == "get_distr_occupancy":
            hour_range = get_hour_slot_range(int(args["hourslot"]))
            parkingslot = get_parkingslot_name(int(args["parkingslot"]))
            legality_status = get_legality_status_name(args["legalitystatus"])
            img_str = get_occupancy_image(
                zone_name, date, hour_range, parkingslot, legality_status
            )
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse({"occupancy": img_str})
        if action == "get_distr_fines":
            hour_range = get_hour_slot_range(int(args["hourslot"]))
            img_str = get_fines_image(zone_name, date, hour_range)
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse({"fines": img_str})
        if action == "get_available_stats_dates":
            available_dates = get_available_dates()
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(available_dates)
        if action == "get_stats_info":
            hour_range = get_hour_slot_range(int(args["hourslot"]))
            out = get_stats_describe_data(zone_name, date, hour_range)
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse({"stats_data": out})
        if action == "get_hour_slots":
            hour_slots = get_hour_slots_items()
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse({"slots": hour_slots})
        if action == "get_available_calendar_dates":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(get_available_calendar_dates())
        if action == "get_calendar":
            out = get_calendar(date)
            if isinstance(out, dict):
                error = out["error"]
                avail_dates = get_available_calendar_dates()
                min_date = avail_dates["min_date"]
                max_date = avail_dates["max_date"]

                return JsonResponse(
                    {
                        "error": f"{error}. Please set the date between {min_date} and {max_date}."
                    },
                    status=202,
                )
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse({"calendar": out})
        if action == "get_available_forecast_dates":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(get_available_forecasting_dates())
        if action == "get_available_forecast_parkingmeters":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(
                {
                    zone_id: get_available_forecasting_parkingmeters(
                        get_zone_name(zone_id)
                    )
                    for zone_id in get_area_id_label_mapping().keys()
                }
            )
        if action == "get_available_forecast_roads":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(
                {
                    zone_id: get_available_forecasting_roads(get_zone_name(zone_id))
                    for zone_id in get_area_id_label_mapping().keys()
                }
            )
        if action == "get_forecast_transactions":
            if date is None:
                return JsonResponse(
                    {"error": "Please provide a valid 'selected_date'."}, status=202
                )

            parkingmeter = get_parkingmeter_name(int(args["parkingmeter"]))
            out = get_plot_forecast_transactions(zone_name, date, parkingmeter)
            if "error" in out:
                return JsonResponse({}, status=202)
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(out)
        if action == "get_forecast_amount":
            if date is None:
                return JsonResponse(
                    {"error": "Please provide a valid 'selected_date'."}, status=202
                )

            parkingmeter = get_parkingmeter_name(int(args["parkingmeter"]))
            out = get_plot_forecast_amount(zone_name, date, parkingmeter)
            if "error" in out:
                return JsonResponse(out, status=202)
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(out)
        if action == "get_forecast_road":
            if date is None:
                return JsonResponse(
                    {"error": "Please provide a valid 'selected_date'."}, status=202
                )

            road = get_road_id(args["road"])
            out = get_plot_forecast_roads(zone_name, date, road)
            if "error" in out:
                return JsonResponse(out, status=202)
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(out)
        if action == "get_available_whatif_1stscenario_dates":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(get_available_whatif_scenario_dates())
        if action == "get_available_whatif_2ndscenario_dates":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(get_available_whatif_scenario_dates())
        if action == "get_available_whatif_3rdscenario_dates":
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(get_available_whatif_scenario_dates())
        if action == "run_whatif_generation":
            if date is None:
                return JsonResponse(
                    {"error": "Please provide a valid 'selected_date'."}, status=202
                )
            if scenario is None:
                return JsonResponse("Please provide a valid 'scenario'.")
            if scenario != "3rd" and zone_name == "all_map":
                return JsonResponse(
                    {"error": "Please provide a valid 'zone_name'."},
                    status=202,
                )

            res = run_generation(request, scenario, zone_name, date, quantity)
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(res)
        if action == "get_whatif_heatmaps":
            if date is None:
                return JsonResponse(
                    {"error": "Please provide a valid 'selected_date'."}, status=202
                )
            if scenario is None:
                return JsonResponse("Please provide a valid 'scenario'.")
            if kind is None:
                return JsonResponse(
                    {"error": "Please provide a valid 'kind'."}, status=202
                )

            selected_weekday = get_date(args.get("selected_weekday"))
            if selected_weekday is None:
                return JsonResponse(
                    {"error": "Please provide a valid 'selected_weekday'."}, status=202
                )
            selected_day = selected_weekday.date().isoformat()
            out = get_whatif_heatmaps(
                request,
                scenario,
                zone_name,
                date,
                quantity,
                kind,
                selected_day,
            )
            if "error" in out:
                return JsonResponse(out, status=202)
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(out)
        if action == "get_whatif_distributions":
            if date is None:
                return JsonResponse(
                    {"error": "Please provide a valid 'selected_date'."}, status=202
                )
            if scenario is None:
                return JsonResponse(
                    "Scenario must be provided for getting distributions."
                )
            if kind is None:
                return JsonResponse(
                    {"error": "Please provide a valid 'kind'."}, status=202
                )

            out = get_whatif_distributions(
                request,
                scenario,
                zone_name,
                date,
                quantity,
                kind,
            )
            if "error" in out:
                return JsonResponse(out, status=202)
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(out)
        if action == "get_whatif_cumulative_plot":
            if date is None:
                return JsonResponse(
                    {"error": "Please provide a valid 'selected_date'."}, status=202
                )
            if scenario is None:
                return JsonResponse(
                    "Scenario must be provided for getting cumulative plot."
                )
            if kind is None:
                return JsonResponse(
                    {"error": "Please provide a valid 'kind'."}, status=202
                )

            adjacent_zone_id = args["adjacent_zone_id"]
            if adjacent_zone_id == "":
                adjacent_zone_id = "0"
            try:
                selected_adjacent_zone = get_zone_name(int(adjacent_zone_id))
            except Exception as e:
                message = traceback.format_exc()
                print(message)
                print(e)
                return JsonResponse(
                    {"error": "Invalid 'adjacent_zone_id' provided."},
                    status=202,
                )
            out = get_whatif_cumulative_plot(
                request,
                scenario,
                zone_name,
                date,
                quantity,
                kind,
                selected_adjacent_zone,
            )
            if "error" in out:
                return JsonResponse(out, status=202)
            ans_time = f"{datetime.now()}"
            with open(logfile, "a") as f:
                f.write(f"[{ans_time}] {success_out}\n")
            print(f"[{ans_time}] {success_out}")
            return JsonResponse(out)
        message = traceback.format_exc()
        print(message)
        return JsonResponse({}, status=400)
    except Exception as e:
        message = traceback.format_exc()
        print(message)
        return JsonResponse({"error": str(e)}, status=500)


@ensure_csrf_cookie
def main(request: HttpRequest) -> HttpResponse | JsonResponse:
    if (
        request.method == "POST"
        and request.headers.get("x-requested-with") == "XMLHttpRequest"
    ):
        return get_post_reply(request)

    out_msg = (
        f"[{datetime.now()}] Got request of type {request.method} "
        f"from user IP Address: {get_ip_addr(request)}\n"
    )

    with open(logfile, "a") as f:
        f.write(out_msg)
    print(out_msg)

    # Render the template for GET requests
    return render(request, "base.html")
