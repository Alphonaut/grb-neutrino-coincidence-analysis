"""
GRB / IceCube neutrino coincidence check
========================================

This script reads one IceCube neutrino alert from a JSON file, queries
Astro-COLIBRI for gamma-ray bursts (GRBs) in a selected time interval,
compares the GRB and neutrino positions, and plots every GRB-neutrino pair.

How to run the script
---------------------

Mode 1: Fixed time interval before the neutrino trigger

    python grb_neutrino_check.py --t 10

This searches from:

    neutrino trigger time - 10 minutes

up to:

    neutrino trigger time



Mode 2: Time interval around the neutrino trigger

    python grb_neutrino_check.py --t1 10 --t2 5

This searches from:

    neutrino trigger time - 10 minutes

up to:

    neutrino trigger time + 5 minutes





Selecting the neutrino JSON file
--------------------------------
A JSON file can be supplied explicitly:

    python grb_neutrino_check.py "Path_to_json_file.json" --t 10

If no JSON path is supplied, the script searches JSON_FOLDER and selects
the JSON file with the most recent file-modification timestamp.

Starting code in Spyder example
--------------
Fixed interval before the neutrino:

    runfile(
        "C:/path/to/grb_neutrino_check.py",
        args="--t 10",
        wdir="C:/path/to"
    )

Interval around the neutrino:

    runfile(
        "C:/path/to/grb_neutrino_check.py",
        args="--t1 10 --t2 5",
        wdir="C:/path/to"
    )
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u


# ============================================================
# Settings
# ============================================================

# Personal Astro-COLIBRI API user ID.
# Replace this value before sharing or running the script elsewhere.
ASTROCOLIBRI_UID = "PASTE_YOUR_UID_HERE"

# Astro-COLIBRI API endpoint used to request transient events.
LATEST_TRANSIENTS_URL = "https://astro-colibri.science/latest_transients"

# Maximum waiting time for the Astro-COLIBRI request.
REQUEST_TIMEOUT_SECONDS = 30

# If True, print the complete filtered GRB event dictionaries returned
# by Astro-COLIBRI.
PRINT_ASTROCOLIBRI_API_RESPONSE = True

# If True, save the complete unfiltered Astro-COLIBRI API response as a
# human-readable TXT file.
SAVE_ASTROCOLIBRI_API_RESPONSE = True

# Folder used for saved Astro-COLIBRI API responses.
# Relative paths are resolved from the current working directory.
ASTROCOLIBRI_API_RESPONSE_OUTPUT_DIR = Path("PATH/TO/YOUR/DIRECTORY")

# Folder searched when no JSON file is supplied on the command line.
# In that case, the most recently modified *.json file is selected.
JSON_FOLDER = Path("PATH/TO/JSON/FOLDER")

# Plot output settings.
SAVE_PLOTS = False
PLOT_OUTPUT_DIR = Path("PATH/TO/PLOT/FOLDER")

# Used if no valid GRB positional error is available.
# np.nan means that no GRB error circle is plotted and no spatial-overlap
# decision can be made from the error radii.
FALLBACK_GRB_ERROR_DEG = np.nan

# GRB time-field priority.
# "time" is treated as the accurate event time.
# "last_modified" and "ivorn_time" are fallback times and are marked as
# inaccurate in the plot.
# No need to change anything here.
GRB_TIME_FIELDS = (
    "time",
    "last_modified",
    "ivorn_time",
)

# Plot appearance you could change.
PLOT_FIGSIZE = (11, 6.5)
PLOT_DPI = 200
PLOT_LEGEND_FONTSIZE = 11
PLOT_INFO_FONTSIZE = 11
PLOT_TITLE_FONTSIZE = 12


# ============================================================
# Generic helpers
# ============================================================

def to_float(value):
    """Convert a value to float and return NaN if conversion fails."""
    try:
        if value is None:
            return np.nan

        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "n/a", "null"}:
            return np.nan

        return float(
            text.replace("<", "")
            .replace(">", "")
            .replace("~", "")
            .strip()
        )

    except (TypeError, ValueError):
        return np.nan


def first_existing(mapping, keys, default=None):
    """
    Return the first present, non-empty value from a mapping.

    The order of keys defines the priority.
    """
    for key in keys:
        if key not in mapping:
            continue

        value = mapping[key]

        if value is None:
            continue

        if isinstance(value, str) and value.strip().lower() in {
            "", "nan", "none", "n/a", "null"
        }:
            continue

        return value

    return default


def parse_utc_time(value):
    """Parse a time value as a timezone-aware UTC pandas Timestamp."""
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(timestamp):
        raise ValueError(f"Could not parse UTC time: {value!r}")

    return timestamp


def scalar_text(value, default="Unknown"):
    """Convert scalar or one-element list values to display text."""
    if isinstance(value, list):
        return str(value[0]) if value else default

    if value is None:
        return default

    return str(value)


# ============================================================
# JSON neutrino alert
# ============================================================

def find_json_file(command_line_path=None):
    """
    Select the neutrino JSON file.

    If a path is supplied on the command line, that exact file is used.
    Otherwise, the most recently modified JSON file in JSON_FOLDER is used.
    """
    if command_line_path is not None:
        path = Path(command_line_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found: {path}")

        return path

    folder = JSON_FOLDER.expanduser().resolve()

    if not folder.is_dir():
        raise FileNotFoundError(f"JSON directory not found: {folder}")

    candidates = [
        path
        for path in folder.glob("*.json")
        if path.is_file()
    ]

    if not candidates:
        raise FileNotFoundError(f"No JSON files were found in: {folder}")

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    ).resolve()


def load_neutrino_alert(json_path):
    """Read and validate the IceCube neutrino alert JSON file."""
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    required_fields = ["ra", "dec", "ra_dec_error"]
    missing = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing:
        raise ValueError(
            f"Missing required JSON fields: {', '.join(missing)}"
        )

    trigger_value = first_existing(
        data,
        ["trigger_time", "alert_datetime"],
        default=None,
    )

    if trigger_value is None:
        raise ValueError(
            "The JSON must contain 'trigger_time' or 'alert_datetime'."
        )

    ra_deg = to_float(data["ra"])
    dec_deg = to_float(data["dec"])
    error_deg = to_float(data["ra_dec_error"])

    if not np.isfinite(ra_deg) or not 0.0 <= ra_deg <= 360.0:
        raise ValueError(f"Invalid neutrino RA: {data['ra']!r}")

    if not np.isfinite(dec_deg) or not -90.0 <= dec_deg <= 90.0:
        raise ValueError(f"Invalid neutrino DEC: {data['dec']!r}")

    if not np.isfinite(error_deg) or error_deg < 0.0:
        raise ValueError(
            f"Invalid neutrino error radius: {data['ra_dec_error']!r}"
        )

    return {
        "name": scalar_text(
            first_existing(data, ["event_name", "id"])
        ),
        "time_utc": parse_utc_time(trigger_value),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "err_deg": error_deg,
    }


# ============================================================
# Astro-COLIBRI GRB handling
# ============================================================

def flatten_response(data):
    """
    Recursively collect event-like dictionaries from the API response.

    The Astro-COLIBRI response can contain nested dictionaries and lists.
    This function extracts dictionaries that appear to represent events and
    removes duplicates using a multi-field signature.
    """
    events = []

    def walk(obj):
        if isinstance(obj, dict):
            has_identity = any(
                key in obj
                for key in [
                    "source_name",
                    "name",
                    "trigger_id",
                    "identifier",
                    "astro_colibri_id",
                ]
            )

            has_position = (
                any(key in obj for key in ["ra", "RA", "ra_deg"])
                and
                any(key in obj for key in ["dec", "DEC", "dec_deg"])
            )

            has_classification = any(
                key in obj
                for key in [
                    "type",
                    "event_type",
                    "class",
                    "classification",
                    "source_type",
                ]
            )

            if has_identity or (has_position and has_classification):
                events.append(obj)

            for value in obj.values():
                walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    unique_events = []
    seen = set()

    for event in events:
        signature = (
            str(event.get("source_name", "")),
            str(event.get("trigger_id", "")),
            str(event.get("timestamp", "")),
            str(event.get("time", "")),
            str(event.get("ra", "")),
            str(event.get("dec", "")),
        )

        if signature not in seen:
            seen.add(signature)
            unique_events.append(event)

    return unique_events


def get_grb_time(event):
    """
    Return the best available GRB time and information about its reliability.

    Priority:
        1. time          -> treated as accurate
        2. last_modified -> fallback, treated as inaccurate
        3. ivorn_time    -> fallback, treated as inaccurate
    """
    for field_name in GRB_TIME_FIELDS:
        value = first_existing(
            event,
            [field_name],
            default=None,
        )

        if value is None:
            continue

        event_time = pd.to_datetime(
            value,
            utc=True,
            errors="coerce",
        )

        if pd.notna(event_time):
            time_is_accurate = field_name == "time"
            return event_time, field_name, time_is_accurate

    return pd.NaT, None, False


def grb_name(event):
    """
    Return the Astro-COLIBRI source_name.

    If source_name is missing, create a fallback label from the best available
    GRB date.
    """
    source_name = first_existing(
        event,
        ["source_name"],
        default=None,
    )

    if source_name is not None:
        return scalar_text(source_name)

    event_time, _, _ = get_grb_time(event)

    if pd.notna(event_time):
        date_text = pd.Timestamp(event_time).strftime("%d.%m.%y")
        return f"GRB Name unknown ({date_text})"

    return "GRB Name unknown (Date unknown)"


def event_type_text(event):
    """
    Combine the documented event type and the GRB display name for filtering.
    """
    parts = []

    if event.get("type") is not None:
        parts.append(str(event["type"]))

    parts.append(grb_name(event))

    return " ".join(parts).lower()


def is_grb_event(event):
    """Return True if the event type or display name identifies a GRB."""
    return "grb" in event_type_text(event)


def is_retracted_event(event):
    """Detect retracted or withdrawn events from common text fields."""
    text_fields = [
        event.get("source_name"),
        event.get("name"),
        event.get("alert_type"),
        event.get("type"),
        event.get("classification"),
        event.get("event_type"),
    ]

    text = " ".join(
        str(value).lower()
        for value in text_fields
        if value is not None
    )

    return any(
        keyword in text
        for keyword in (
            "retracted",
            "retraction",
            "withdrawn",
        )
    )


def get_grb_coordinates(event):
    """Read GRB right ascension and declination in degrees."""
    ra_deg = to_float(
        first_existing(
            event,
            ["ra", "RA", "ra_deg", "RA_deg"],
        )
    )

    dec_deg = to_float(
        first_existing(
            event,
            ["dec", "Dec", "DEC", "dec_deg", "DEC_deg"],
        )
    )

    return ra_deg, dec_deg


def get_grb_error(event):
    """Read the GRB positional error radius in degrees."""
    error_deg = to_float(
        first_existing(
            event,
            [
                "err",
                "error",
                "pos_error",
                "positional_error",
                "error_radius",
                "error90",
                "err90",
                "uncertainty",
                "localization_error",
            ],
            default=np.nan,
        )
    )

    if not np.isfinite(error_deg):
        return FALLBACK_GRB_ERROR_DEG

    return error_deg


def get_grb_observatory_and_instrument(event):
    """Read the observatory and instrument names from the GRB event."""
    observatory = scalar_text(
        first_existing(
            event,
            ["observatory"],
            default="Unknown",
        )
    )

    instrument = scalar_text(
        first_existing(
            event,
            ["instrument"],
            default="Unknown",
        )
    )

    return observatory, instrument


def save_astrocolibri_api_response(data, start_time, end_time):
    """
    Save the complete unfiltered Astro-COLIBRI API response as a TXT file.

    A timestamp in the filename prevents previous responses from being
    overwritten. The requested time interval is written above the JSON data.
    """
    ASTROCOLIBRI_API_RESPONSE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f_UTC"
    )

    output_path = ASTROCOLIBRI_API_RESPONSE_OUTPUT_DIR / (
        f"astro_colibri_api_response_{timestamp}.txt"
    )

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Astro-COLIBRI API response\n")
        file.write("=" * 100 + "\n")
        file.write(
            f"Requested start time: "
            f"{pd.Timestamp(start_time).isoformat()}\n"
        )
        file.write(
            f"Requested end time:   "
            f"{pd.Timestamp(end_time).isoformat()}\n"
        )
        file.write(
            f"Saved at UTC:         "
            f"{datetime.now(timezone.utc).isoformat()}\n"
        )
        file.write("=" * 100 + "\n\n")
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        file.write("\n")

    print(f"Saved Astro-COLIBRI API response: {output_path.resolve()}")


def query_recent_grbs(start_time, end_time):
    """
    Query Astro-COLIBRI and return valid GRBs inside the selected time range.
    """
    if (
        not ASTROCOLIBRI_UID
        or ASTROCOLIBRI_UID == "PASTE_YOUR_UID_HERE"
    ):
        raise ValueError(
            "Insert your Astro-COLIBRI UID in ASTROCOLIBRI_UID."
        )

    payload = {
        "time_range": {
            "min": pd.Timestamp(start_time).isoformat(),
            "max": pd.Timestamp(end_time).isoformat(),
        },
        "uid": ASTROCOLIBRI_UID,
        "return_format": "json",
    }

    response = requests.post(
        LATEST_TRANSIENTS_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()

    if SAVE_ASTROCOLIBRI_API_RESPONSE:
        save_astrocolibri_api_response(
            data,
            start_time,
            end_time,
        )

    all_events = flatten_response(data)

    grb_events = [
        event
        for event in all_events
        if is_grb_event(event)
        and not is_retracted_event(event)
    ]

    if PRINT_ASTROCOLIBRI_API_RESPONSE:
        print()
        print("=" * 100)
        print("GRB EVENTS FROM ASTRO-COLIBRI")
        print("=" * 100)

        if not grb_events:
            print(
                "No GRB events were found in the Astro-COLIBRI response."
            )

        else:
            for index, event in enumerate(grb_events, start=1):
                print()
                print(f"GRB event {index}")
                print("-" * 100)
                print(
                    json.dumps(
                        event,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )

    rows = []

    for event in grb_events:
        event_time, time_source, time_is_accurate = get_grb_time(event)
        ra_deg, dec_deg = get_grb_coordinates(event)
    
        observatory, instrument = get_grb_observatory_and_instrument(
            event
        )

        # A GRB without any valid time cannot be assigned to the search window.
        if pd.isna(event_time):
            continue

        if not (start_time <= event_time <= end_time):
            continue

        if not (
            np.isfinite(ra_deg)
            and np.isfinite(dec_deg)
            and 0.0 <= ra_deg <= 360.0
            and -90.0 <= dec_deg <= 90.0
        ):
            continue

        rows.append(
            {
                "name": grb_name(event),
                "observatory": observatory,
                "instrument": instrument,
                "time_utc": event_time,
                "time_source": time_source,
                "time_is_accurate": time_is_accurate,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "err_deg": get_grb_error(event),
            }
        )

    columns = [
        "name",
        "observatory",
        "instrument",
        "time_utc",
        "time_source",
        "time_is_accurate",
        "ra_deg",
        "dec_deg",
        "err_deg",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        rows,
        columns=columns,
    ).sort_values(
        "time_utc",
        ignore_index=True,
    )


# ============================================================
# Pair comparison
# ============================================================

def compare_pair(grb, neutrino):
    """Calculate temporal and angular separation for one GRB-neutrino pair."""
    grb_coord = SkyCoord(
        ra=float(grb["ra_deg"]) * u.deg,
        dec=float(grb["dec_deg"]) * u.deg,
        frame="icrs",
    )

    neutrino_coord = SkyCoord(
        ra=neutrino["ra_deg"] * u.deg,
        dec=neutrino["dec_deg"] * u.deg,
        frame="icrs",
    )

    angular_separation_deg = grb_coord.separation(
        neutrino_coord
    ).deg

    # Positive values mean that the GRB occurred before the neutrino.
    # Negative values mean that the GRB occurred after the neutrino.
    signed_time_difference_min = (
        neutrino["time_utc"] - pd.Timestamp(grb["time_utc"])
    ).total_seconds() / 60.0

    grb_error_deg = float(grb["err_deg"])
    neutrino_error_deg = neutrino["err_deg"]

    spatial_overlap = None

    if (
        np.isfinite(grb_error_deg)
        and np.isfinite(neutrino_error_deg)
    ):
        spatial_overlap = (
            angular_separation_deg
            <= grb_error_deg + neutrino_error_deg
        )

    return {
        "GRB": grb["name"],
        "GRB_time_UTC": grb["time_utc"],
        "GRB_time_source": grb["time_source"],
        "GRB_time_is_accurate": bool(grb["time_is_accurate"]),
        "GRB_RA_deg": grb["ra_deg"],
        "GRB_DEC_deg": grb["dec_deg"],
        "GRB_error_deg": grb_error_deg,
        "Neutrino": neutrino["name"],
        "Neutrino_time_UTC": neutrino["time_utc"],
        "Neutrino_RA_deg": neutrino["ra_deg"],
        "Neutrino_DEC_deg": neutrino["dec_deg"],
        "Neutrino_error_deg": neutrino_error_deg,
        "time_difference_min": signed_time_difference_min,
        "angular_separation_deg": angular_separation_deg,
        "spatial_overlap": spatial_overlap,
    }


# ============================================================
# Mollweide plotting
# ============================================================

def radec_to_mollweide(ra_deg, dec_deg):
    """Convert RA/DEC in degrees to Matplotlib Mollweide coordinates."""
    wrapped_ra = (
        (np.asarray(ra_deg) + 180.0) % 360.0
    ) - 180.0

    # The minus sign follows the astronomical convention that RA increases
    # towards the left.
    longitude = -np.deg2rad(wrapped_ra)
    latitude = np.deg2rad(dec_deg)

    return longitude, latitude


def split_wrapped_curve(longitude, latitude, threshold=np.pi):
    """Split curves that cross the Mollweide longitude wrap boundary."""
    longitude = np.asarray(longitude)
    latitude = np.asarray(latitude)

    jumps = np.where(
        np.abs(np.diff(longitude)) > threshold
    )[0] + 1

    return (
        np.split(longitude, jumps),
        np.split(latitude, jumps),
    )


def make_error_circle(ra_deg, dec_deg, radius_deg, points=720):
    """
    Create a spherical error circle around a sky position.

    Astropy's directional_offset_by keeps the circle geometrically correct
    near the celestial poles and across the RA wrap boundary.
    """
    if not np.isfinite(radius_deg):
        return None

    center = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        frame="icrs",
    )

    position_angles = np.linspace(
        0.0,
        2.0 * np.pi,
        points,
    ) * u.rad

    circle = center.directional_offset_by(
        position_angle=position_angles,
        separation=radius_deg * u.deg,
    )

    return np.column_stack(
        (
            circle.ra.deg,
            circle.dec.deg,
        )
    )


def plot_curve(ax, ra_deg, dec_deg, label, linestyle="-"):
    """Plot a wrapped sky curve on a Mollweide axis."""
    longitude, latitude = radec_to_mollweide(
        ra_deg,
        dec_deg,
    )

    longitude_parts, latitude_parts = split_wrapped_curve(
        longitude,
        latitude,
    )

    for index, (x_values, y_values) in enumerate(
        zip(longitude_parts, latitude_parts)
    ):
        ax.plot(
            x_values,
            y_values,
            linestyle=linestyle,
            linewidth=1.5,
            label=label if index == 0 else None,
        )


def plot_pair(grb, neutrino, comparison, pair_index):
    """Create one Mollweide plot for a GRB-neutrino pair."""
    grb_lon, grb_lat = radec_to_mollweide(
        grb["ra_deg"],
        grb["dec_deg"],
    )

    nu_lon, nu_lat = radec_to_mollweide(
        neutrino["ra_deg"],
        neutrino["dec_deg"],
    )

    figure = plt.figure(
        figsize=PLOT_FIGSIZE,
        dpi=PLOT_DPI,
    )

    axis = figure.add_subplot(
        111,
        projection="mollweide",
    )

    axis.scatter(
        grb_lon,
        grb_lat,
        marker="*",
        s=85,
        alpha=0.8,
        label=(
            f"{grb['name']} best fit\n"
            f"Observatory: {grb['observatory']}\n"
            f"Instrument: {grb['instrument']}"
        ),
    )

    axis.scatter(
        nu_lon,
        nu_lat,
        marker="x",
        s=65,
        alpha=0.8,
        label=f"{neutrino['name']} best fit",
    )

    grb_circle = make_error_circle(
        grb["ra_deg"],
        grb["dec_deg"],
        float(grb["err_deg"]),
    )

    if grb_circle is not None:
        plot_curve(
            axis,
            grb_circle[:, 0],
            grb_circle[:, 1],
            label=(
                "GRB 90% error radius: "
                f"{float(grb['err_deg']):.3f} deg"
            ),
            linestyle="-",
        )

    neutrino_circle = make_error_circle(
        neutrino["ra_deg"],
        neutrino["dec_deg"],
        neutrino["err_deg"],
    )

    if neutrino_circle is not None:
        plot_curve(
            axis,
            neutrino_circle[:, 0],
            neutrino_circle[:, 1],
            label=(
                "Neutrino error radius: "
                f"{neutrino['err_deg']:.3f} deg"
            ),
            linestyle="--",
        )

    axis.grid(True, alpha=0.35)

    axis.set_title(
        f"{grb['name']} vs {neutrino['name']}",
        fontsize=PLOT_TITLE_FONTSIZE,
    )

    if comparison["GRB_time_is_accurate"]:
        time_warning = ""
    else:
        time_warning = "  INACCURATE TIME!"

    info_text = (
        f"GRB:\n"
        f"RA = {float(grb['ra_deg']):.2f} deg, "
        f"DEC = {float(grb['dec_deg']):.2f} deg, "
        f"90% Error radius = {float(grb['err_deg']):.2f} deg\n"
        f"Neutrino:\n"
        f"RA = {neutrino['ra_deg']:.2f} deg, "
        f"DEC = {neutrino['dec_deg']:.2f} deg, "
        f"Error radius = {neutrino['err_deg']:.2f} deg\n\n"
        f"Time difference (t_neutrino - t_0_grb) = "
        f"{comparison['time_difference_min']:.3f} min"
        f"{time_warning}\n"
        f"Angular distance = "
        f"{comparison['angular_separation_deg']:.3f} deg\n"
        f"Spatial overlap = {comparison['spatial_overlap']}"
        f"\n\nWe acknowledge Astro-COLIBRI!"
    )

    # Symbol legend below the plot on the left.
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.00, -0.08),
        fontsize=PLOT_LEGEND_FONTSIZE,
        ncol=1,
        framealpha=0.78,
    )

    # Coordinate and comparison information below the plot on the right.
    axis.text(
        0.45,
        -0.05,
        info_text,
        transform=axis.transAxes,
        fontsize=PLOT_INFO_FONTSIZE,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.78,
            "edgecolor": "gray",
        },
    )

    plt.tight_layout(
        rect=[0.0, 0.22, 1.0, 1.0]
    )

    figure.subplots_adjust(bottom=0.30)

    if SAVE_PLOTS:
        PLOT_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_grb_name = (
            str(grb["name"])
            .replace("/", "_")
            .replace(" ", "_")
        )

        safe_neutrino_name = (
            neutrino["name"]
            .replace("/", "_")
            .replace(" ", "_")
        )

        output_path = PLOT_OUTPUT_DIR / (
            f"{pair_index:03d}_"
            f"{safe_grb_name}__{safe_neutrino_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Saved plot: {output_path}")

    plt.show()


# ============================================================
# Time-window selection
# ============================================================

def get_grb_search_window(
    neutrino,
    t_minutes=None,
    t1_minutes=None,
    t2_minutes=None,
):
    """
    Build the Astro-COLIBRI search interval.

    Mode 1:
        --t X

        Search from neutrino time - X minutes up to the neutrino time.

    Mode 2:
        --t1 X --t2 Y

        Search from neutrino time - X minutes up to
        neutrino time + Y minutes.
    """
    neutrino_time = pd.Timestamp(neutrino["time_utc"])

    if t_minutes is not None:
        start_time = neutrino_time - timedelta(
            minutes=t_minutes
        )

        end_time = neutrino_time

        window_description = (
            f"{t_minutes:g} minutes before the neutrino trigger "
            "until the neutrino trigger"
        )

        window_mode = "fixed window before neutrino"

    else:
        start_time = neutrino_time - timedelta(
            minutes=t1_minutes
        )

        end_time = neutrino_time + timedelta(
            minutes=t2_minutes
        )

        window_description = (
            f"{t1_minutes:g} minutes before to "
            f"{t2_minutes:g} minutes after the neutrino trigger"
        )

        window_mode = "window around neutrino"

    return (
        start_time,
        end_time,
        window_description,
        window_mode,
    )


# ============================================================
# Main check
# ============================================================

def run_check(
    json_path,
    t_minutes=None,
    t1_minutes=None,
    t2_minutes=None,
):
    """Run the complete GRB-neutrino comparison."""
    neutrino = load_neutrino_alert(json_path)

    (
        start_time,
        end_time,
        window_description,
        window_mode,
    ) = get_grb_search_window(
        neutrino,
        t_minutes=t_minutes,
        t1_minutes=t1_minutes,
        t2_minutes=t2_minutes,
    )

    print("=" * 100)
    print("GRB / single-neutrino alert check")
    print("=" * 100)
    print(f"JSON file: {json_path}")
    print(f"Neutrino event: {neutrino['name']}")
    print(
        "Neutrino trigger time: "
        f"{neutrino['time_utc'].isoformat()}"
    )
    print(f"Time-window mode: {window_mode}")
    print(
        "GRB search window: "
        f"{start_time.isoformat()} to {end_time.isoformat()}"
    )
    print(f"Window definition: {window_description}")

    grb_table = query_recent_grbs(
        start_time,
        end_time,
    )

    print()
    print(
        f"GRBs found in the time window: {len(grb_table)}"
    )

    if grb_table.empty:
        print(
            "No GRB was detected in the selected time window."
        )
        return pd.DataFrame()

    print(
    grb_table[
        [
            "name",
            "observatory",
            "instrument",
            "time_utc",
            "time_source",
            "time_is_accurate",
            "ra_deg",
            "dec_deg",
            "err_deg",
        ]

        ].to_string(index=False)
    )

    comparisons = []

    for pair_index, (_, grb) in enumerate(
        grb_table.iterrows(),
        start=1,
    ):
        comparison = compare_pair(
            grb,
            neutrino,
        )

        comparisons.append(comparison)

        plot_pair(
            grb,
            neutrino,
            comparison,
            pair_index,
        )

    comparison_table = pd.DataFrame(comparisons)

    print()
    print("Pair comparison:")
    print(
        comparison_table.to_string(index=False)
    )

    return comparison_table


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    """
    Parse and validate the JSON path and time-window arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Read one IceCube single-neutrino JSON alert, "
            "query Astro-COLIBRI for GRBs around the neutrino "
            "trigger time, and plot every pair."
        )
    )

    parser.add_argument(
        "json_file",
        nargs="?",
        help=(
            "Optional path to an IceCube JSON alert. "
            "If omitted, the most recently modified JSON file "
            "in JSON_FOLDER is used."
        ),
    )

    parser.add_argument(
        "--t",
        type=float,
        default=None,
        help=(
            "Fixed time window in minutes before the neutrino trigger. "
            "The interval is [neutrino time - t, neutrino time]."
        ),
    )

    parser.add_argument(
        "--t1",
        type=float,
        default=None,
        help=(
            "Time in minutes before the neutrino trigger when using "
            "the time window around the neutrino."
        ),
    )

    parser.add_argument(
        "--t2",
        type=float,
        default=None,
        help=(
            "Time in minutes after the neutrino trigger when using "
            "the time window around the neutrino."
        ),
    )

    arguments = parser.parse_args()

    # The fixed-window mode and around-trigger mode are mutually exclusive.
    if arguments.t is not None and (
        arguments.t1 is not None
        or arguments.t2 is not None
    ):
        parser.error(
            "Use either --t or the combination --t1 and --t2, not both."
        )

    # Require one complete time-window definition.
    if arguments.t is None and (
        arguments.t1 is None
        or arguments.t2 is None
    ):
        parser.error(
            "Specify either --t X or both --t1 X and --t2 Y."
        )

    # Negative intervals are not valid.
    if arguments.t is not None and arguments.t < 0:
        parser.error(
            "--t must be greater than or equal to 0."
        )

    if arguments.t1 is not None and arguments.t1 < 0:
        parser.error(
            "--t1 must be greater than or equal to 0."
        )

    if arguments.t2 is not None and arguments.t2 < 0:
        parser.error(
            "--t2 must be greater than or equal to 0."
        )

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()

    try:
        selected_json_path = find_json_file(
            arguments.json_file
        )

        run_check(
            selected_json_path,
            t_minutes=arguments.t,
            t1_minutes=arguments.t1,
            t2_minutes=arguments.t2,
        )

    except Exception as error:
        print(f"ERROR: {error}")
        raise