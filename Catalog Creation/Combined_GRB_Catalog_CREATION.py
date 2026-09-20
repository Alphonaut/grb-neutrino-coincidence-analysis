# ============================================================
#  Imports
# ============================================================
from pathlib import Path
import pandas as pd
import re
import json
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


# ============================================================
#  Control
# ============================================================
CREATE_CATALOG = True
WRITE_CATALOG = True


# ============================================================
#  New Perera et al. 2026 GBM coherent-search catalog settings
# ============================================================
USE_NEW_ANALYSIS_CATALOG = True

# Switch for the T90-like duration from the new Perera catalog.
# Allowed values:
#   "duration"
#   "pe_duration"
#
# Recommendation:
#   "pe_duration", because it is the refined duration from parameter estimation.
NEW_ANALYSIS_T90_SOURCE = "duration"

ALLOWED_NEW_ANALYSIS_GRB_CLASSES = [
    "GRB1",
    "GRB2",
    "GRB3",
    "GRB4",
    "GRB5",
    "GRB6",
    "GRB7",
]

NEW_ANALYSIS_MIN_PASTRO = 0.9


# ============================================================
#  Input directory (preserved for file1-file5)
# ============================================================
base_dir = Path("X:\\Uni+Arbeit\\Uni\\BACHELOR THESIS Linux\\Combined Catalog")


# ============================================================
#  File paths
# ============================================================
file1 = base_dir / "1FERMI GBM (COMBINED) with t90_start.tsv"
file2 = base_dir / "2FERMI LAT (COMBINED).txt"
file3 = base_dir / "3COMBINED SWIFT GRB TABLE (COMBINED).txt"
file4 = base_dir / "4SWIFT BAT ENHANCED CAT with T90_start.txt"

# New shortened Perera catalog
file5 = base_dir / "perera et al 2026 gbm grb catalog v1.csv"

# Outputs are written to the current working directory.
output_dir = Path.cwd()

output_txt = output_dir / "final_combined_catalog_v19.txt"
output_tsv = output_dir / "final_combined_catalog_v19.tsv"
output_csv = output_dir / "final_combined_catalog_v19.csv"
duplicate_pairs_path = output_dir / "duplicate_pairs_v19.tsv"


# ============================================================
#  File writing checks
# ============================================================
def check_file_is_writable(path: Path):
    """
    Check whether a file is writable.
    If the file is open in Excel or locked, the code stops immediately.
    """
    try:
        if path.exists():
            with open(path, "a", encoding="utf-8"):
                pass
        else:
            with open(path, "w", encoding="utf-8"):
                pass
            path.unlink()

    except PermissionError:
        raise PermissionError(
            f"\nThe file is currently open or locked:\n{path}\n\n"
            "Please close the file and run the code again."
        )


def check_output_files_are_writable():
    check_file_is_writable(output_txt)
    check_file_is_writable(output_tsv)
    check_file_is_writable(output_csv)
    check_file_is_writable(duplicate_pairs_path)


# ============================================================
#  General helpers
# ============================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("##", "", regex=False)
        .str.strip()
    )
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = []

    for col in df.columns:
        values = df[col].astype(str).str.strip()

        if not ((values == "").all() or values.isna().all()):
            keep_cols.append(col)

    return df[keep_cols]


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}{col}" for col in df.columns]
    return df


def format_value(x):
    if pd.isna(x):
        return ""

    if isinstance(x, float):
        return f"{x:16.10f}"

    return str(x)


def is_missing(value) -> bool:
    if pd.isna(value):
        return True

    value_str = str(value).strip().lower()

    return value_str in ["", "nan", "n/a", "none"]


def to_float(value):
    try:
        value = str(value).strip()

        if value.lower() in ["", "nan", "n/a", "none"]:
            return float("nan")

        return float(value)

    except Exception:
        return float("nan")


def parse_redshift_value(value):
    if pd.isna(value):
        return ""

    value_str = str(value).strip()

    if value_str.lower() in ["", "nan", "n/a", "none"]:
        return ""

    inside_parentheses = 0
    current_outside = []

    for char in value_str:
        if char == "(":
            inside_parentheses += 1

        elif char == ")":
            if inside_parentheses > 0:
                inside_parentheses -= 1

        elif inside_parentheses == 0:
            current_outside.append(char)

    outside_text = "".join(current_outside)
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", outside_text)

    if len(numbers) == 0:
        return ""

    return "; ".join(numbers)


# ============================================================
#  Readers
# ============================================================
def read_pipe_table(path: Path, skiprows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", engine="python", skiprows=skiprows)

    df = clean_columns(df)
    df = df.loc[:, ~df.columns.str.fullmatch(r"Unnamed: \d+")]
    df = drop_empty_columns(df)

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def read_tab_table(path: Path, skiprows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", engine="python", skiprows=skiprows)

    df = clean_columns(df)

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def read_pipe_table_with_header_fix(path: Path) -> pd.DataFrame:
    """
    Reads catalog 4.

    Supports both formats:
        1. Old pipe-separated format with "|"
        2. New fixed-width format with dashed separator line
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header_index = None

    for i, line in enumerate(lines):
        if "GRBname" in line:
            header_index = i
            break

    if header_index is None:
        raise ValueError("Header not found!")

    header_line = lines[header_index]

    # ------------------------------------------------------------
    # Old format: pipe-separated table
    # ------------------------------------------------------------
    if "|" in header_line:
        df = pd.read_csv(
            path,
            sep="|",
            engine="python",
            skiprows=header_index
        )

        df = clean_columns(df)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = drop_empty_columns(df)

        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

        return df

    # ------------------------------------------------------------
    # New format: fixed-width table
    # Header line followed by dashed separator line
    # ------------------------------------------------------------
    separator_index = header_index + 1

    if separator_index >= len(lines):
        raise ValueError("Separator line after header not found!")

    separator_line = lines[separator_index]

    dash_matches = list(re.finditer(r"-+", separator_line))

    if len(dash_matches) == 0:
        raise ValueError("Fixed-width column structure not recognized!")

    colspecs = [match.span() for match in dash_matches]

    column_names = []

    for start, end in colspecs:
        column_name = header_line[start:end].strip()
        column_names.append(column_name)

    df = pd.read_fwf(
        path,
        colspecs=colspecs,
        names=column_names,
        skiprows=separator_index + 1
    )

    df = clean_columns(df)
    df = drop_empty_columns(df)

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def read_new_analysis_catalog(path: Path) -> pd.DataFrame:
    """
    Reads the shortened Perera et al. 2026 catalog.

    Required columns:
        grb_name or trigtime
        trigtime
        classification
        pastro
        gbm_catalog
        ra_median
        dec_median
        ra_err_plus
        ra_err_minus
        dec_err_plus
        dec_err_minus

    If grb_name is missing, it is generated from trigtime.
    """

    df = pd.read_csv(path)
    df = clean_columns(df)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    required_columns = [
        "trigtime",
        "classification",
        "pastro",
        "gbm_catalog",
        "ra_median",
        "dec_median",
        "ra_err_plus",
        "ra_err_minus",
        "dec_err_plus",
        "dec_err_minus",
    ]

    missing_required_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_required_columns:
        raise ValueError(
            "Missing required columns in new analysis catalog:\n"
            + "\n".join(missing_required_columns)
        )

    if NEW_ANALYSIS_T90_SOURCE not in ["duration", "pe_duration"]:
        raise ValueError(
            "NEW_ANALYSIS_T90_SOURCE must be either 'duration' or 'pe_duration'."
        )

    if NEW_ANALYSIS_T90_SOURCE not in df.columns:
        raise ValueError(
            f"Selected NEW_ANALYSIS_T90_SOURCE column not found: "
            f"{NEW_ANALYSIS_T90_SOURCE}"
        )

    df["pastro"] = pd.to_numeric(df["pastro"], errors="coerce")

    df = df[
        df["classification"].isin(ALLOWED_NEW_ANALYSIS_GRB_CLASSES)
        & (df["pastro"] >= NEW_ANALYSIS_MIN_PASTRO)
    ].copy()

    if "grb_name" not in df.columns:
        df.insert(
            0,
            "grb_name",
            df["trigtime"].apply(trigtime_to_grb_name_rounded)
        )

    df["pos_error_90_arcsec"] = df.apply(
        calculate_perera_pos_error_90_arcsec,
        axis=1
    )
    

    df["trigtime"] = pd.to_datetime(
        df["trigtime"],
        utc=True,
        errors="coerce"
    )

    return df


# ============================================================
#  Time helpers
# ============================================================
DAY_NS = 86_400_000_000_000  # 86400 s in nanoseconds


def has_valid_ut_time(value) -> bool:
    s = str(value).strip()

    if pd.isna(value):
        return False

    if s.lower() in ["", "nan", "n/a", "none"]:
        return False

    return bool(re.fullmatch(r"\d{2}:\d{2}:\d{2}(\.\d+)?", s))


def ut_to_day_fraction(time_str: str) -> float:
    time_str = str(time_str).strip()

    if time_str.lower() in ["nan", "n/a", ""]:
        return float("nan")

    parts = time_str.split(":")

    if len(parts) != 3:
        return float("nan")

    hour = float(parts[0])
    minute = float(parts[1])
    second = float(parts[2])

    total_seconds = hour * 3600 + minute * 60 + second

    return total_seconds / 86400.0


def iso_to_day_fraction(datetime_str: str) -> float:
    datetime_str = str(datetime_str).strip()

    if datetime_str.lower() in ["nan", "n/a", ""]:
        return float("nan")

    timestamp = pd.to_datetime(datetime_str, errors="coerce")

    if pd.isna(timestamp):
        timestamp = pd.to_datetime(
            datetime_str,
            format="%Y%m%dT%H:%M:%S.%f",
            errors="coerce"
        )

    if pd.isna(timestamp):
        timestamp = pd.to_datetime(
            datetime_str,
            format="%Y%m%dT%H:%M:%S",
            errors="coerce"
        )

    if pd.isna(timestamp):
        return float("nan")

    total_seconds = (
        timestamp.hour * 3600
        + timestamp.minute * 60
        + timestamp.second
        + timestamp.microsecond / 1_000_000
    )

    return total_seconds / 86400.0


def trigtime_to_grb_name_rounded(trigtime, prefix: str = "GRB") -> str:
    """
    Builds a Fermi-style GRB name:

        GRBYYMMDDfff

    fff is the rounded fraction of the UTC day in thousandths.
    If rounding gives 1000, the date is advanced by one day
    and fff is set to 000.
    """

    ts = pd.to_datetime(trigtime, utc=True, errors="coerce")

    if pd.isna(ts):
        return ""

    midnight = ts.normalize()
    total_ns = int((ts - midnight).value)

    fraction_code = (total_ns * 1000 + DAY_NS // 2) // DAY_NS

    if fraction_code == 1000:
        ts = ts + pd.Timedelta(days=1)
        fraction_code = 0

    date_part = ts.strftime("%y%m%d")

    return f"{prefix}{date_part}{fraction_code:03d}"


def build_grb_yymmddfff(name_value, time_value) -> str:
    name_str = str(name_value).strip()
    time_str = str(time_value).strip()

    if len(name_str) < 6:
        return ""

    yymmdd = name_str[:6]
    day_fraction = ut_to_day_fraction(time_str)

    if pd.isna(day_fraction):
        return ""

    rounded_fraction = round(day_fraction, 3)
    fff = f"{rounded_fraction:.3f}".split(".")[1]

    return f"GRB{yymmdd}{fff}"


def build_grb_from_table4(name_value, time_value) -> str:
    name_str = str(name_value).strip()

    if name_str.lower() in ["nan", "n/a", ""]:
        return ""

    base_name = name_str[:-1] if len(name_str) > 0 else ""
    day_fraction = iso_to_day_fraction(time_value)

    if pd.isna(day_fraction):
        return base_name

    rounded_fraction = round(day_fraction, 3)
    fff = f"{rounded_fraction:.3f}".split(".")[1]

    return f"{base_name}{fff}"


def grb_num_difference_to_minutes(diff):
    return diff / 1000.0 * 24.0 * 60.0


def normalize_swift_name(value) -> str:
    """
    Converts Swift names to the format GRBYYMMDDX.

    Examples:
        201104B     -> GRB201104B
        GRB201104B  -> GRB201104B
    """
    name = str(value).strip()

    if name.lower() in ["", "nan", "n/a", "none"]:
        return ""

    if not name.startswith("GRB"):
        name = "GRB" + name

    return name


def build_fermi_grb_from_swift_merge(
    swift_name, trig_time_utc_4, time_ut_3
) -> str:
    """Build a rounded Fermi-style name, including date rollover."""
    timestamp = pd.to_datetime(trig_time_utc_4, utc=True, errors="coerce")
    if pd.notna(timestamp):
        return trigtime_to_grb_name_rounded(timestamp)

    name = normalize_swift_name(swift_name)
    match = re.fullmatch(r"GRB(\d{6})[A-Za-z]*", name)
    if match is None or not has_valid_ut_time(time_ut_3):
        return ""

    date = pd.to_datetime(match.group(1), format="%y%m%d", errors="coerce")
    if pd.isna(date):
        return ""

    timestamp = pd.to_datetime(
        f"{date:%Y-%m-%d} {str(time_ut_3).strip()}",
        utc=True,
        errors="coerce"
    )
    return trigtime_to_grb_name_rounded(timestamp)



def datetime_to_mjd(timestamp) -> float:
    if pd.isna(timestamp):
        return float("nan")

    return timestamp.to_julian_date() - 2400000.5


def catalog_time_to_mjd(row: pd.Series, best_key: str) -> float:
    timestamp = pd.NaT

    if best_key == "1":
        time_str = str(row.get("1trigger_time")).strip()
        time_str = re.sub(r"\s+", " ", time_str)

        timestamp = pd.to_datetime(
            time_str,
            errors="coerce"
        )

    elif best_key == "2":
        time_str = str(row.get("2time")).strip()
        time_str = re.sub(r"\s+", " ", time_str)

        timestamp = pd.to_datetime(
            time_str,
            errors="coerce"
        )

    elif best_key in ["3BAT", "3XRT", "3UVOT"]:
        # The rounded Fermi-style name may refer to the following day.
        # Reconstruct the physical trigger time from the original Swift name.
        time_ut = str(row.get("3Time [UT]")).strip()
        swift_name = normalize_swift_name(row.get("3name"))
        match = re.fullmatch(r"GRB(\d{6})[A-Za-z]*", swift_name)

        if match is not None and has_valid_ut_time(time_ut):
            date = pd.to_datetime(
                match.group(1), format="%y%m%d", errors="coerce"
            )
            if pd.notna(date):
                timestamp = pd.to_datetime(
                    f"{date:%Y-%m-%d} {time_ut}",
                    utc=True,
                    errors="coerce"
                )

    elif best_key == "4":
        timestamp = pd.to_datetime(
            row.get("4Trig_time_UTC"),
            errors="coerce"
        )

    elif best_key == "5":
        timestamp = pd.to_datetime(
            row.get("5trigtime"),
            utc=True,
            errors="coerce"
        )

    if pd.isna(timestamp):
        return float("nan")

    return round(datetime_to_mjd(timestamp), 10)


# ============================================================
#  Coordinates
# ============================================================
def sexagesimal_to_deg(value, is_ra=True):
    try:
        value = str(value).strip()

        if value.lower() in ["", "nan", "n/a", "none"]:
            return float("nan")

        parts = value.split(":")

        if len(parts) != 3:
            return float("nan")

        h = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])

        if is_ra:
            return (h + m / 60 + s / 3600) * 15.0

        sign = -1 if h < 0 else 1

        return sign * (abs(h) + m / 60 + s / 3600)

    except Exception:
        return float("nan")


def galactic_to_j2000(l_deg, b_deg):
    l_deg = to_float(l_deg)
    b_deg = to_float(b_deg)

    if pd.isna(l_deg) or pd.isna(b_deg):
        return float("nan"), float("nan")

    try:
        c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
        return c.icrs.ra.deg, c.icrs.dec.deg

    except Exception:
        return float("nan"), float("nan")


def standardize_coordinate_columns_to_j2000_deg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all original catalog coordinate columns that remain in the final
    catalog to J2000/ICRS equatorial coordinates in degrees.

    After this function:
        1ra, 1dec       -> RA/DEC J2000 [deg]
        2ra, 2dec       -> RA/DEC J2000 [deg]
        3BAT RA/Dec     -> RA/DEC J2000 [deg]
        3XRT RA/Dec     -> RA/DEC J2000 [deg]
        3UVOT RA/Dec    -> RA/DEC J2000 [deg]
        4RA_ground/DEC  -> RA/DEC J2000 [deg]
        5ra_median/dec  -> already J2000/ICRS [deg], only numeric
    """
    df = df.copy()

    # ------------------------------------------------------------
    # Fermi GBM:
    # Original 1ra / 1dec are treated as Galactic l / b [deg].
    # Convert them to equatorial J2000/ICRS RA / DEC [deg].
    # ------------------------------------------------------------
    if {"1ra", "1dec"}.issubset(df.columns):
        converted = df.apply(
            lambda row: galactic_to_j2000(row["1ra"], row["1dec"]),
            axis=1
        )

        df["1ra"] = [x[0] for x in converted]
        df["1dec"] = [x[1] for x in converted]

    # ------------------------------------------------------------
    # Fermi LAT:
    # Original 2ra / 2dec are treated as Galactic l / b [deg].
    # Convert them to equatorial J2000/ICRS RA / DEC [deg].
    # ------------------------------------------------------------
    if {"2ra", "2dec"}.issubset(df.columns):
        converted = df.apply(
            lambda row: galactic_to_j2000(row["2ra"], row["2dec"]),
            axis=1
        )

        df["2ra"] = [x[0] for x in converted]
        df["2dec"] = [x[1] for x in converted]

    # ------------------------------------------------------------
    # Swift BAT and Swift BAT Enhanced:
    # Already numeric equatorial J2000 coordinates in degrees.
    # ------------------------------------------------------------
    for col in [
        "3BAT RA",
        "3BAT Dec",
        "4RA_ground",
        "4DEC_ground",
        "5ra_median",
        "5dec_median",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------
    # Swift XRT:
    # Original RA/DEC are sexagesimal.
    # Convert to equatorial J2000 degrees.
    # ------------------------------------------------------------
    if "3XRT RA" in df.columns:
        df["3XRT RA"] = df["3XRT RA"].apply(
            lambda value: sexagesimal_to_deg(value, is_ra=True)
        )

    if "3XRT Dec" in df.columns:
        df["3XRT Dec"] = df["3XRT Dec"].apply(
            lambda value: sexagesimal_to_deg(value, is_ra=False)
        )

    # ------------------------------------------------------------
    # Swift UVOT:
    # Original RA/DEC are sexagesimal.
    # Convert to equatorial J2000 degrees.
    # ------------------------------------------------------------
    if "3UVOT RA" in df.columns:
        df["3UVOT RA"] = df["3UVOT RA"].apply(
            lambda value: sexagesimal_to_deg(value, is_ra=True)
        )

    if "3UVOT Dec" in df.columns:
        df["3UVOT Dec"] = df["3UVOT Dec"].apply(
            lambda value: sexagesimal_to_deg(value, is_ra=False)
        )

    return df


def standardize_error_radius_columns_to_90_arcsec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts original catalog error-radius columns to a common standard:
    90% confidence error radius in arcseconds.

    After this function:
        1error_radius              -> 90% error radius [arcsec]
        2error_radius              -> 90% error radius [arcsec]
        3BAT 90% Error [arcmin]    -> 90% error radius [arcsec]
        3XRT 90% Error [arcsec]    -> 90% error radius [arcsec]
        3UVOT 90% Error [arcsec]   -> 90% error radius [arcsec]
        4Image_position_err        -> 90% error radius [arcsec]
        5pos_error_90_arcsec -> already 90% upper limit [arcsec]
    """
    df = df.copy()

    # Fermi GBM:
    # Original value is treated as 1-sigma error radius in degrees.
    # Convert degree -> arcsec and 1-sigma -> 90%.
    if "1error_radius" in df.columns:
        df["1error_radius"] = (
            pd.to_numeric(df["1error_radius"], errors="coerce")
            * 3600.0
            * 2.146
        )

    # Fermi LAT:
    # Original value is treated as 90% error radius in degrees.
    # Convert degree -> arcsec.
    if "2error_radius" in df.columns:
        df["2error_radius"] = (
            pd.to_numeric(df["2error_radius"], errors="coerce")
            * 3600.0
        )

    # Swift BAT:
    # Original value is already 90%, but in arcmin.
    # Convert arcmin -> arcsec.
    if "3BAT 90% Error [arcmin]" in df.columns:
        df["3BAT 90% Error [arcmin]"] = (
            pd.to_numeric(df["3BAT 90% Error [arcmin]"], errors="coerce")
            * 60.0
        )

    # Swift XRT:
    # Already 90% and arcsec.
    if "3XRT 90% Error [arcsec]" in df.columns:
        df["3XRT 90% Error [arcsec]"] = pd.to_numeric(
            df["3XRT 90% Error [arcsec]"],
            errors="coerce"
        )

    # Swift UVOT:
    # Already 90% and arcsec.
    if "3UVOT 90% Error [arcsec]" in df.columns:
        df["3UVOT 90% Error [arcsec]"] = pd.to_numeric(
            df["3UVOT 90% Error [arcsec]"],
            errors="coerce"
        )

    # Swift BAT Enhanced:
    # Original value is treated as 90% error radius in arcmin.
    # Convert arcmin -> arcsec.
    if "4Image_position_err" in df.columns:
        df["4Image_position_err"] = (
            pd.to_numeric(df["4Image_position_err"], errors="coerce")
            * 60.0
        )

    # Perera catalog:
    # Already computed as 90% upper limit in arcsec.
    if "5pos_error_90_arcsec" in df.columns:
        df["5pos_error_90_arcsec"] = pd.to_numeric(
            df["5pos_error_90_arcsec"],
            errors="coerce"
        )

    df = df.rename(
        columns={
            "1error_radius": "1error_radius_90_arcsec",
            "2error_radius": "2error_radius_90_arcsec",
            "3BAT 90% Error [arcmin]": "3BAT 90% Error [arcsec]",
            "4Image_position_err": "4Image_position_err_90_arcsec",
        }
    )

    return df


# ============================================================
#  Perera catalog helpers
# ============================================================
def calculate_perera_pos_error_90_arcsec(row: pd.Series) -> float:
    """
    Calculates the Perera positional error as an equivalent-area circular
    90% error radius.

    Procedure:
        1. Symmetrize the asymmetric 1-sigma RA errors:
           ra_err_mean = (abs(ra_err_plus) + abs(ra_err_minus)) / 2
        2. Symmetrize the asymmetric 1-sigma DEC errors:
           dec_err_mean = (abs(dec_err_plus) + abs(dec_err_minus)) / 2
        3. Convert RA coordinate uncertainty into true angular distance:
           ra_err_mean * cos(dec)
        4. Compute the radius of a circle with the same area as the
           1-sigma error ellipse:
           r = sqrt(ra_err_mean_projected * dec_err_mean)
        5. Convert 1-sigma to 90% using factor 2.146
        6. Convert degree to arcsec

    Output:
        Equivalent-area 90% positional error radius [arcsec]
    """

    ra_err_plus = to_float(row.get("ra_err_plus"))
    ra_err_minus = to_float(row.get("ra_err_minus"))
    dec_err_plus = to_float(row.get("dec_err_plus"))
    dec_err_minus = to_float(row.get("dec_err_minus"))
    dec = to_float(row.get("dec_median"))

    required_values = [
        ra_err_plus,
        ra_err_minus,
        dec_err_plus,
        dec_err_minus,
        dec,
    ]

    if any(pd.isna(value) for value in required_values):
        return float("nan")

    ra_err_mean_deg = (
        abs(ra_err_plus)
        + abs(ra_err_minus)
    ) / 2.0

    dec_err_mean_deg = (
        abs(dec_err_plus)
        + abs(dec_err_minus)
    ) / 2.0

    ra_err_projected_deg = (
        ra_err_mean_deg
        * abs(np.cos(np.deg2rad(dec)))
    )

    equivalent_radius_deg_1sigma = np.sqrt(
        ra_err_projected_deg
        * dec_err_mean_deg
    )

    equivalent_radius_arcsec_90 = (
        equivalent_radius_deg_1sigma
        * 2.146
        * 3600.0
    )

    return equivalent_radius_arcsec_90


def is_new_perera_grb(row: pd.Series) -> bool:
    """
    A Perera trigger is treated as new if 5gbm_catalog is empty.
    """

    if "5gbm_catalog" not in row.index:
        return False

    return is_missing(row.get("5gbm_catalog"))


def get_new_analysis_t90(row: pd.Series) -> float:
    """
    Returns the selected duration-like value from the new Perera catalog.

    Controlled by:
        NEW_ANALYSIS_T90_SOURCE = "duration"
        NEW_ANALYSIS_T90_SOURCE = "pe_duration"
    """

    if NEW_ANALYSIS_T90_SOURCE not in ["duration", "pe_duration"]:
        raise ValueError(
            "NEW_ANALYSIS_T90_SOURCE must be either 'duration' or 'pe_duration'."
        )

    if NEW_ANALYSIS_T90_SOURCE == "duration":
        return to_float(row.get("5duration"))

    if NEW_ANALYSIS_T90_SOURCE == "pe_duration":
        return to_float(row.get("5pe_duration"))

    return float("nan")


# ============================================================
#  Catalog position info
# ============================================================
def get_catalog_position_info(row: pd.Series, catalog_key: str) -> dict:
    table3_time_valid = has_valid_ut_time(row.get("3Time [UT]"))

    if catalog_key == "1":
        ra = to_float(row.get("1ra"))
        dec = to_float(row.get("1dec"))
        err90 = to_float(row.get("1error_radius_90_arcsec"))

        return {
            "catalog_name": "Fermi/GBM",
            "ra": ra,
            "dec": dec,
            "err90_arcsec": err90
        }

    if catalog_key == "2":
        ra = to_float(row.get("2ra"))
        dec = to_float(row.get("2dec"))
        err90 = to_float(row.get("2error_radius_90_arcsec"))

        return {
            "catalog_name": "Fermi/LAT",
            "ra": ra,
            "dec": dec,
            "err90_arcsec": err90
        }

    if catalog_key == "3BAT":
        if not table3_time_valid:
            return {
                "catalog_name": "Swift Burst Table BAT",
                "ra": float("nan"),
                "dec": float("nan"),
                "err90_arcsec": float("nan")
            }

        ra = to_float(row.get("3BAT RA"))
        dec = to_float(row.get("3BAT Dec"))
        err90 = to_float(row.get("3BAT 90% Error [arcsec]"))

        return {
            "catalog_name": "Swift Burst Table BAT",
            "ra": ra,
            "dec": dec,
            "err90_arcsec": err90
        }

    if catalog_key == "3XRT":
        if not table3_time_valid:
            return {
                "catalog_name": "Swift Burst Table XRT",
                "ra": float("nan"),
                "dec": float("nan"),
                "err90_arcsec": float("nan")
            }

        ra = to_float(row.get("3XRT RA"))
        dec = to_float(row.get("3XRT Dec"))
        err90 = to_float(row.get("3XRT 90% Error [arcsec]"))

        return {
            "catalog_name": "Swift Burst Table XRT",
            "ra": ra,
            "dec": dec,
            "err90_arcsec": err90
        }

    if catalog_key == "3UVOT":
        if not table3_time_valid:
            return {
                "catalog_name": "Swift Burst Table UVOT",
                "ra": float("nan"),
                "dec": float("nan"),
                "err90_arcsec": float("nan")
            }

        ra = to_float(row.get("3UVOT RA"))
        dec = to_float(row.get("3UVOT Dec"))
        err90 = to_float(row.get("3UVOT 90% Error [arcsec]"))

        return {
            "catalog_name": "Swift Burst Table UVOT",
            "ra": ra,
            "dec": dec,
            "err90_arcsec": err90
        }

    if catalog_key == "4":
        ra = to_float(row.get("4RA_ground"))
        dec = to_float(row.get("4DEC_ground"))
        err90 = to_float(row.get("4Image_position_err_90_arcsec"))

        return {
            "catalog_name": "Swift/BAT",
            "ra": ra,
            "dec": dec,
            "err90_arcsec": err90
        }

    if catalog_key == "5":
        # Catalog 5 is only allowed to become best catalog
        # for newly discovered Perera GRBs.
        if not is_new_perera_grb(row):
            return {
                "catalog_name": "Perera et al.",
                "ra": float("nan"),
                "dec": float("nan"),
                "err90_arcsec": float("nan")
            }

        ra = to_float(row.get("5ra_median"))
        dec = to_float(row.get("5dec_median"))
        err90 = to_float(row.get("5pos_error_90_arcsec"))

        return {
            "catalog_name": "Perera et al.",
            "ra": ra,
            "dec": dec,
            "err90_arcsec": err90
        }

    return {
        "catalog_name": "",
        "ra": float("nan"),
        "dec": float("nan"),
        "err90_arcsec": float("nan")
    }


def determine_best_catalog(row: pd.Series):
    infos = {
        "1": get_catalog_position_info(row, "1"),
        "2": get_catalog_position_info(row, "2"),
        "3BAT": get_catalog_position_info(row, "3BAT"),
        "3XRT": get_catalog_position_info(row, "3XRT"),
        "3UVOT": get_catalog_position_info(row, "3UVOT"),
        "4": get_catalog_position_info(row, "4"),
        "5": get_catalog_position_info(row, "5"),
    }

    valid = []
    
    for key, info in infos.items():
        ra = info["ra"]
        dec = info["dec"]
        err90 = info["err90_arcsec"]
    
        if (
            pd.notna(ra)
            and pd.notna(dec)
            and pd.notna(err90)
            and np.isfinite(ra)
            and np.isfinite(dec)
            and np.isfinite(err90)
            and 0 <= ra < 360
            and -90 <= dec <= 90
            and 0 <= err90 <= 648000
        ):
            valid.append((key, err90))
    
    if len(valid) == 0:
        return "", infos
    
    # Positive error radii are ranked by their size.
    # A zero error radius is always ranked after positive values.
    best_key = min(
        valid,
        key=lambda item: (
            item[1] == 0,
            item[1]
        )
    )[0]
    
    return best_key, infos


TIME_FIELDS = [
    "T90 [s]", "T90 Error [s]", "T90 Start [s]",
    "T90 Trigger MJD", "T90 Trigger UTC", "T90 catalog", "T90 source GRB",
]


def get_t90_package(row):
    """Maximum duration with its OWN start, trigger and provenance.
    Ties retain catalog order 1,2,3,4,5, as before.
    Missing starts are never borrowed from another measurement.
    """
    specs = [
        ("1", "1t90", "1t90_error", "1t90_start", "1trigger_time"),
        ("2", "2t90", "2t90_error", None, "2time"),
        ("3", "3BAT T90 [sec]", None, None, "3Time [UT]"),
        ("4", "4T90", "4T90_err", "4T90_start", "4Trig_time_UTC"),
    ]
    if is_new_perera_grb(row):
        specs.append(("5", "5" + NEW_ANALYSIS_T90_SOURCE, None, None, "5trigtime"))
    candidates = []
    for key, duration_col, error_col, start_col, trigger_col in specs:
        duration = to_float(row.get(duration_col))
        if pd.isna(duration):
            continue
        trigger = row.get(trigger_col)
        if key == "3":
            name = normalize_swift_name(row.get("3name"))
            date = pd.to_datetime(name[3:9], format="%y%m%d", errors="coerce")
            if pd.notna(date) and has_valid_ut_time(trigger):
                trigger = f"{date:%Y-%m-%d} {trigger}"
            else:
                trigger = None
        timestamp = pd.to_datetime(trigger, utc=True, errors="coerce")
        candidates.append({
            "T90 [s]": duration,
            "T90 Error [s]": to_float(row.get(error_col)) if error_col else float("nan"),
            "T90 Start [s]": to_float(row.get(start_col)) if start_col else float("nan"),
            "T90 Trigger MJD": round(datetime_to_mjd(timestamp), 6) if pd.notna(timestamp) else float("nan"),
            "T90 Trigger UTC": timestamp.isoformat() if pd.notna(timestamp) else "",
            "T90 catalog": key,
            "T90 source GRB": str(row.get("GRB", "")),
        })
    if not candidates:
        return {key: ("" if key in ["T90 Trigger UTC", "T90 catalog", "T90 source GRB"]
                      else float("nan")) for key in TIME_FIELDS}
    return max(candidates, key=lambda candidate: candidate["T90 [s]"])


def original_record_json(row):
    """Lossless provenance of all retained input columns before duplicate merging."""
    record = {}
    for key, value in row.items():
        if key != "4Trig_time_met" and (key == "GRB" or re.match(r"^[1-5]", str(key))):
            record[key] = None if is_missing(value) else str(value)
    return json.dumps([record], ensure_ascii=False)


# ============================================================
#  Build best-of row
# ============================================================
def build_best_of_row(row: pd.Series) -> dict:
    catalog_name_map = {
        "1": "Fermi/GBM",
        "2": "Fermi/LAT",
        "3BAT": "Swift Burst Table BAT",
        "3XRT": "Swift Burst Table XRT",
        "3UVOT": "Swift Burst Table UVOT",
        "4": "Swift/BAT",
        "5": "Perera et al."
    }

    grb = row["GRB"]
    best_key, infos = determine_best_catalog(row)

    time_package = get_t90_package(row)
    original_records = original_record_json(row)

    # Prefer redshift from Catalog 3.
    # Use Catalog 2 only if Catalog 3 contains no valid redshift.
    redshift_3 = parse_redshift_value(row.get("3Redshift"))
    redshift_2 = parse_redshift_value(row.get("2redshift"))
    
    if redshift_3 != "":
        redshift = redshift_3
    
    elif redshift_2 != "":
        redshift = redshift_2
    
    else:
        redshift = float("nan")

    mjd = catalog_time_to_mjd(row, best_key)

    # ------------------------------------------------------------
    # Error radius for duplicate detection:
    # Use the Best Data error radius by default.
    # Only for positions selected from the Swift Burst Table
    # (Catalog 3), use its BAT error radius when available.
    # ------------------------------------------------------------
    duplicate_error_radius_arcsec = float("nan")

    if best_key in ("3BAT", "3XRT", "3UVOT"):
        bat_error_arcsec = to_float(
            row.get("3BAT 90% Error [arcsec]")
        )

        if pd.notna(bat_error_arcsec):
            duplicate_error_radius_arcsec = bat_error_arcsec

    if best_key == "":
        return {
            "GRB": grb,
            "RA": float("nan"),
            "DEC": float("nan"),
            "Error Radius 90% [arcsec]": float("nan"),
            "Duplicate Error Radius 90% [arcsec]": duplicate_error_radius_arcsec,
            **time_package,
            "Original entries JSON": original_records,
            "Fluence [erg/cm^2]": float("nan"),
            "Fluence Error 1 Sigma [erg/cm^2]": float("nan"),
            "Redshift": redshift,
            "MJD": mjd,
            "Position catalog": ""
        }

    ra = infos[best_key]["ra"]
    dec = infos[best_key]["dec"]
    err90 = infos[best_key]["err90_arcsec"]
    best_catalog_name = catalog_name_map[best_key]

    if pd.isna(duplicate_error_radius_arcsec):
        duplicate_error_radius_arcsec = err90

    fluence = float("nan")
    fluence_err = float("nan")

    if best_key == "1":
        if not is_missing(row.get("1fluence")):
            fluence = row.get("1fluence")

        if not is_missing(row.get("1fluence_error")):
            fluence_err = row.get("1fluence_error")

    elif best_key == "2":
        if not is_missing(row.get("2fluence")):
            fluence = row.get("2fluence")

        if not is_missing(row.get("2fluence_error")):
            fluence_err = row.get("2fluence_error")

    elif best_key == "3BAT":
        if not is_missing(row.get("3BAT Fluence")):
            fluence = row.get("3BAT Fluence")

        if not is_missing(row.get("3BAT Fluence 90% Error")):
            fluence_err = to_float(row.get("3BAT Fluence 90% Error")) / 1.645

    # Perera catalog does not provide fluence in the shortened catalog.
    elif best_key == "5":
        fluence = float("nan")
        fluence_err = float("nan")

    return {
        "GRB": grb,
        "RA": ra,
        "DEC": dec,
        "Error Radius 90% [arcsec]": err90,
        "Duplicate Error Radius 90% [arcsec]": duplicate_error_radius_arcsec,
        **time_package,
        "Original entries JSON": original_records,
        "Fluence [erg/cm^2]": fluence,
        "Fluence Error 1 Sigma [erg/cm^2]": fluence_err,
        "Redshift": redshift,
        "MJD": mjd,
        "Position catalog": best_catalog_name
    }


# ============================================================
#  Duplicate handling
# ============================================================
def extract_grb_number(grb_name):
    s = str(grb_name).strip()
    match = re.search(r"GRB(\d+)", s)

    if match:
        try:
            return int(match.group(1))

        except Exception:
            return float("nan")

    return float("nan")


def t90_windows_overlap(row_a: pd.Series, row_b: pd.Series) -> bool:
    """
    Checks whether the later GRB lies within the T90 time window
    of the earlier GRB.

    If both GRBs come from different best catalogs, a tolerance of
    10% of the earlier T90 is allowed.

    Same catalog:
        time_difference <= T90_earlier

    Different catalogs:
        time_difference <= 1.1 * T90_earlier
    """
    mjd_a = pd.to_numeric(row_a["MJD"], errors="coerce")
    mjd_b = pd.to_numeric(row_b["MJD"], errors="coerce")

    t90_a = pd.to_numeric(row_a["T90 [s]"], errors="coerce")
    t90_b = pd.to_numeric(row_b["T90 [s]"], errors="coerce")

    if pd.isna(mjd_a) or pd.isna(mjd_b):
        return False

    if mjd_a <= mjd_b:
        mjd_earlier = mjd_a
        mjd_later = mjd_b
        t90_earlier = t90_a

    else:
        mjd_earlier = mjd_b
        mjd_later = mjd_a
        t90_earlier = t90_b

    if pd.isna(t90_earlier):
        return False

    if t90_earlier < 0:
        return False

    time_difference_seconds = (mjd_later - mjd_earlier) * 86400.0

    allowed_time_difference_seconds = 1.1 * t90_earlier

    return time_difference_seconds <= allowed_time_difference_seconds


def rows_are_near_duplicate(row_a: pd.Series, row_b: pd.Series) -> bool:
    """
    Pairwise duplicate check using Astropy SkyCoord.

    Kept for debugging or manual checks.
    The faster merge_bestof_duplicates() function performs the same logic
    vectorized for candidate groups.
    """
    mjd_a = pd.to_numeric(row_a["MJD"], errors="coerce")
    mjd_b = pd.to_numeric(row_b["MJD"], errors="coerce")

    ra_a = pd.to_numeric(row_a["RA"], errors="coerce")
    ra_b = pd.to_numeric(row_b["RA"], errors="coerce")

    dec_a = pd.to_numeric(row_a["DEC"], errors="coerce")
    dec_b = pd.to_numeric(row_b["DEC"], errors="coerce")

    err_a_arcsec = pd.to_numeric(
        row_a.get(
            "Duplicate Error Radius 90% [arcsec]",
            row_a["Error Radius 90% [arcsec]"]
        ),
        errors="coerce"
    )

    err_b_arcsec = pd.to_numeric(
        row_b.get(
            "Duplicate Error Radius 90% [arcsec]",
            row_b["Error Radius 90% [arcsec]"]
        ),
        errors="coerce"
    )

    if pd.isna(mjd_a) or pd.isna(mjd_b):
        return False

    if pd.isna(ra_a) or pd.isna(ra_b):
        return False

    if pd.isna(dec_a) or pd.isna(dec_b):
        return False

    if pd.isna(err_a_arcsec) or pd.isna(err_b_arcsec):
        return False

    coord_a = SkyCoord(
        ra=ra_a * u.deg,
        dec=dec_a * u.deg,
        frame="icrs"
    )

    coord_b = SkyCoord(
        ra=ra_b * u.deg,
        dec=dec_b * u.deg,
        frame="icrs"
    )

    distance_deg = coord_a.separation(coord_b).deg
    err_sum_deg = (err_a_arcsec + err_b_arcsec) / 3600.0

    time_condition = abs(mjd_a - mjd_b) <= 0.0208333333
    position_condition = distance_deg <= err_sum_deg
    t90_condition = t90_windows_overlap(row_a, row_b)

    return time_condition and position_condition and t90_condition


def merge_group_rows(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values(
        by="Error Radius 90% [arcsec]",
        ascending=True,
        na_position="last"
    ).reset_index(drop=True)

    merged = group.loc[0].copy()

    protected_cols = {
        "GRB",
        "RA",
        "DEC",
        "Error Radius 90% [arcsec]",
        "Duplicate Error Radius 90% [arcsec]",
    }

    for i in range(1, len(group)):
        candidate = group.loc[i]

        for col in group.columns:
            if col in protected_cols:
                continue

            if col in set(TIME_FIELDS) | {"Original entries JSON"}:
                continue

            if is_missing(merged[col]) and not is_missing(candidate[col]):
                merged[col] = candidate[col]

    candidates = [candidate for _, candidate in group.iterrows()
                  if pd.notna(to_float(candidate.get("T90 [s]")))]
    if candidates:
        winner = max(candidates, key=lambda candidate: to_float(candidate["T90 [s]"]))
        for column in TIME_FIELDS:
            merged[column] = winner[column]
    records = []
    for value in group["Original entries JSON"]:
        records.extend(json.loads(value))
    merged["Original entries JSON"] = json.dumps(records, ensure_ascii=False)

    return merged


def angular_separation_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """
    Vectorized angular separation in degrees.
    Equivalent purpose to SkyCoord.separation, but much faster for many pairs.
    """
    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)

    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)

    cos_sep = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )

    cos_sep = np.clip(cos_sep, -1.0, 1.0)

    return np.rad2deg(np.arccos(cos_sep))


def merge_bestof_duplicates(best_of_df: pd.DataFrame):
    """
    Faster duplicate merging with exact Astropy SkyCoord separation.

    Improvements:
    - Numeric columns are converted once.
    - The dataframe is sorted by MJD.
    - Only rows inside the 30-minute MJD window are checked.
    - SkyCoord.separation is used vectorized for candidate groups.
    """
    time_window_days = 0.0208333333
    seconds_per_day = 86400.0
    minutes_per_day = 24.0 * 60.0

    df = best_of_df.copy()

    df = df.sort_values(
        by=["MJD", "Error Radius 90% [arcsec]"],
        ascending=[True, True],
        na_position="last"
    ).reset_index(drop=True)

    n = len(df)

    if n == 0:
        return df, pd.DataFrame()

    df["_MJD_NUM"] = pd.to_numeric(df["MJD"], errors="coerce")
    df["_RA_NUM"] = pd.to_numeric(df["RA"], errors="coerce")
    df["_DEC_NUM"] = pd.to_numeric(df["DEC"], errors="coerce")
    df["_T90_NUM"] = pd.to_numeric(df["T90 [s]"], errors="coerce")

    if "Duplicate Error Radius 90% [arcsec]" in df.columns:
        df["_DUP_ERR_NUM"] = pd.to_numeric(
            df["Duplicate Error Radius 90% [arcsec]"],
            errors="coerce"
        )

    else:
        df["_DUP_ERR_NUM"] = pd.to_numeric(
            df["Error Radius 90% [arcsec]"],
            errors="coerce"
        )


    mjd_values = df["_MJD_NUM"].to_numpy()
    ra_values = df["_RA_NUM"].to_numpy()
    dec_values = df["_DEC_NUM"].to_numpy()
    err_values = df["_DUP_ERR_NUM"].to_numpy()
    t90_values = df["_T90_NUM"].to_numpy()

    visited = np.zeros(n, dtype=bool)
    merged_rows = []
    duplicate_pairs = []

    duplicate_input_count = 0
    duplicate_output_count = 0

    for i in range(n):
        if visited[i]:
            continue

        queue = [i]
        component = []
        visited[i] = True

        while queue:
            current = queue.pop(0)
            component.append(current)

            mjd_current = mjd_values[current]

            if not np.isfinite(mjd_current):
                continue

            left = np.searchsorted(
                mjd_values,
                mjd_current - time_window_days,
                side="left"
            )

            right = np.searchsorted(
                mjd_values,
                mjd_current + time_window_days,
                side="right"
            )

            candidate_indices = np.arange(left, right)

            candidate_indices = candidate_indices[
                ~visited[candidate_indices]
            ]

            if len(candidate_indices) == 0:
                continue

            valid = (
                np.isfinite(ra_values[current])
                & np.isfinite(dec_values[current])
                & np.isfinite(err_values[current])
                & np.isfinite(mjd_values[candidate_indices])
                & np.isfinite(ra_values[candidate_indices])
                & np.isfinite(dec_values[candidate_indices])
                & np.isfinite(err_values[candidate_indices])
            )

            candidate_indices = candidate_indices[valid]

            if len(candidate_indices) == 0:
                continue

            coord_current = SkyCoord(
                ra=ra_values[current] * u.deg,
                dec=dec_values[current] * u.deg,
                frame="icrs"
            )

            coord_candidates = SkyCoord(
                ra=ra_values[candidate_indices] * u.deg,
                dec=dec_values[candidate_indices] * u.deg,
                frame="icrs"
            )

            distances_deg = coord_current.separation(coord_candidates).deg

            err_sum_deg = (
                err_values[current]
                + err_values[candidate_indices]
            ) / 3600.0

            position_condition = distances_deg <= err_sum_deg

            candidate_indices = candidate_indices[position_condition]

            if len(candidate_indices) == 0:
                continue

            mjd_candidates = mjd_values[candidate_indices]

            current_is_earlier = mjd_current <= mjd_candidates

            t90_earlier = np.where(
                current_is_earlier,
                t90_values[current],
                t90_values[candidate_indices]
            )

            time_difference_seconds = (
                np.abs(mjd_current - mjd_candidates)
                * seconds_per_day
            )

            allowed_time_difference_seconds = 1.1 * t90_earlier

            t90_condition = (
                np.isfinite(t90_earlier)
                & (t90_earlier >= 0)
                & (time_difference_seconds <= allowed_time_difference_seconds)
            )

            duplicate_indices = candidate_indices[t90_condition]

            for j in duplicate_indices:
                time_diff_min = (
                    abs(mjd_values[current] - mjd_values[j])
                    * minutes_per_day
                )

                t90_current = t90_values[current]
                t90_j = t90_values[j]

                valid_t90 = [
                    value for value in [t90_current, t90_j]
                    if np.isfinite(value)
                ]

                t90_pair = (
                    max(valid_t90)
                    if len(valid_t90) > 0
                    else float("nan")
                )

                duplicate_pairs.append({
                    "GRB_1": df.loc[current, "GRB"],
                    "GRB_2": df.loc[j, "GRB"],
                    "MJD_1": mjd_values[current],
                    "MJD_2": mjd_values[j],
                    "time_diff_min": time_diff_min,
                    "T90 [s]": t90_pair
                })

                visited[j] = True
                queue.append(j)

        group = df.loc[component].copy().reset_index(drop=True)

        if len(group) > 1:
            duplicate_input_count += len(group)
            duplicate_output_count += 1

            group_print = group.copy()
            group_print["Error Radius 90% [deg]"] = (
                group_print["Error Radius 90% [arcsec]"] / 3600.0
            )

            if len(group) > 2:
                print()
                print("GROUP WITH MORE THAN 2 GRBs:")
                print(
                    group_print[
                        [
                            "GRB",
                            "MJD",
                            "T90 [s]",
                            "RA",
                            "DEC",
                            "Error Radius 90% [deg]"
                        ]
                    ].to_string(index=False)
                )
                print("-" * 80)

        merged_row = merge_group_rows(group)
        merged_rows.append(merged_row)

    merged_df = pd.DataFrame(merged_rows).reset_index(drop=True)

    work_columns = [
        "_MJD_NUM",
        "_RA_NUM",
        "_DEC_NUM",
        "_T90_NUM",
        "_DUP_ERR_NUM",
    ]

    merged_df = merged_df.drop(columns=work_columns, errors="ignore")

    if "GRB_num" in merged_df.columns:
        merged_df = merged_df.drop(columns=["GRB_num"])

    duplicate_pairs_df = pd.DataFrame(duplicate_pairs)

    print()
    print("Duplicate statistics:")
    print(f"Number of GRBs in duplicate groups: {duplicate_input_count}")
    print(f"Number of GRBs remaining after merging: {duplicate_output_count}")
    print(f"Number of detected duplicate pairs: {len(duplicate_pairs_df)}")

    return merged_df, duplicate_pairs_df


# ============================================================
#  Catalog creation
# ============================================================
def create_catalog():
    df1 = read_tab_table(file1)
    df2 = read_pipe_table(file2)
    df3 = read_tab_table(file3)
    df4 = read_pipe_table_with_header_fix(file4)

    if USE_NEW_ANALYSIS_CATALOG:
        df5 = read_new_analysis_catalog(file5)
    else:
        df5 = pd.DataFrame()

    for col in ["BAT Fluence", "BAT Fluence 90% Error"]:
        if col in df3.columns:
            df3[col] = pd.to_numeric(df3[col], errors="coerce") / 1e7

    df1_prefixed = prefix_columns(df1, "1")
    df2_prefixed = prefix_columns(df2, "2")
    df3_prefixed = prefix_columns(df3, "3")
    df4_prefixed = prefix_columns(df4, "4")

    if USE_NEW_ANALYSIS_CATALOG:
        df5_prefixed = prefix_columns(df5, "5")
    else:
        df5_prefixed = pd.DataFrame()

    # ============================================================
    #  Build native Swift names for catalog 3 and catalog 4
    # ============================================================
    df3_prefixed["Swift_Name"] = df3_prefixed["3name"].apply(normalize_swift_name)
    df4_prefixed["Swift_Name"] = df4_prefixed["4GRBname"].apply(normalize_swift_name)

    df3_prefixed = df3_prefixed[
        df3_prefixed["Swift_Name"] != ""
    ].reset_index(drop=True)

    df4_prefixed = df4_prefixed[
        df4_prefixed["Swift_Name"] != ""
    ].reset_index(drop=True)

    df3_prefixed = df3_prefixed[
        df3_prefixed["Swift_Name"].str[3:9] >= "110500"
    ].reset_index(drop=True)

    df4_prefixed = df4_prefixed[
        df4_prefixed["Swift_Name"].str[3:9] >= "110500"
    ].reset_index(drop=True)

    names_3 = set(df3_prefixed["Swift_Name"])
    names_4 = set(df4_prefixed["Swift_Name"])

    names_in_both = names_3 & names_4
    names_only_3 = names_3 - names_4
    names_only_4 = names_4 - names_3

    print()
    print("=" * 80)
    print("Swift catalog merge statistics")
    print("=" * 80)
    print(f"Catalog 3 entries: {len(df3_prefixed)}")
    print(f"Catalog 4 entries: {len(df4_prefixed)}")
    print(f"Swift names in both catalog 3 and 4: {len(names_in_both)}")
    print(f"Swift names only in catalog 3: {len(names_only_3)}")
    print(f"Swift names only in catalog 4: {len(names_only_4)}")

    swift_combined_df = df3_prefixed.merge(
        df4_prefixed,
        on="Swift_Name",
        how="outer"
    )

    print(f"Combined Swift catalog entries: {len(swift_combined_df)}")

    swift_combined_df["GRB"] = [
        build_fermi_grb_from_swift_merge(
            swift_name=swift_name,
            trig_time_utc_4=trig_time_utc_4,
            time_ut_3=time_ut_3
        )
        for swift_name, trig_time_utc_4, time_ut_3 in zip(
            swift_combined_df["Swift_Name"],
            swift_combined_df.get(
                "4Trig_time_UTC",
                pd.Series([""] * len(swift_combined_df))
            ),
            swift_combined_df.get(
                "3Time [UT]",
                pd.Series([""] * len(swift_combined_df))
            )
        )
    ]

    fermi_from_4_count = 0
    fermi_from_3_count = 0
    fermi_missing_count = 0

    for _, row in swift_combined_df.iterrows():
        grb_name = str(row.get("GRB", "")).strip()

        if grb_name == "":
            fermi_missing_count += 1
            continue

        day_fraction_4 = iso_to_day_fraction(row.get("4Trig_time_UTC"))
        day_fraction_3 = ut_to_day_fraction(row.get("3Time [UT]"))

        if pd.notna(day_fraction_4):
            fermi_from_4_count += 1

        elif pd.notna(day_fraction_3):
            fermi_from_3_count += 1

    print(f"Fermi-style names generated from catalog 4 Trig_time_UTC: {fermi_from_4_count}")
    print(f"Fermi-style names generated from catalog 3 Time [UT] fallback: {fermi_from_3_count}")
    print(f"Swift entries without valid Fermi-style name: {fermi_missing_count}")

    swift_combined_df = swift_combined_df[
        swift_combined_df["GRB"] != ""
    ].reset_index(drop=True)

    # ============================================================
    #  Prepare Fermi catalogs
    # ============================================================
    df1_prefixed["GRB"] = df1_prefixed["1name"]
    df2_prefixed["GRB"] = df2_prefixed["2name"]

    df1_prefixed = df1_prefixed.drop(columns=["1name"])
    df2_prefixed = df2_prefixed.drop(columns=["2name"])

    swift_combined_df = swift_combined_df.drop(
        columns=["Swift_Name"],
        errors="ignore"
    )

    # ============================================================
    #  Prepare new Perera catalog
    # ============================================================
    if USE_NEW_ANALYSIS_CATALOG:
        if "5grb_name" not in df5_prefixed.columns:
            raise ValueError("Column '5grb_name' not found in new Perera catalog.")
    
        df5_prefixed["GRB"] = df5_prefixed["5grb_name"]
    
        new_perera_count = df5_prefixed["5gbm_catalog"].apply(is_missing).sum()
        known_perera_count = len(df5_prefixed) - new_perera_count
    
        old_catalog_grb_names = (
            set(df1_prefixed["GRB"])
            | set(df2_prefixed["GRB"])
            | set(swift_combined_df["GRB"])
        )
    
        perera_grb_names = set(df5_prefixed["GRB"])
    
        perera_matched_by_grb_name = perera_grb_names & old_catalog_grb_names
        perera_not_matched_by_grb_name = perera_grb_names - old_catalog_grb_names
    
        print()
        print("=" * 80)
        print("Perera et al. 2026 catalog statistics")
        print("=" * 80)
        print(f"Perera GRB-class entries after filtering: {len(df5_prefixed)}")
        print(f"New Perera GRB candidates with empty gbm_catalog: {new_perera_count}")
        print(f"Known Perera GRB candidates with gbm_catalog entry: {known_perera_count}")
        print(f"Perera GRBs matched to old combined catalog by GRB name: {len(perera_matched_by_grb_name)}")
        print(f"Perera GRBs not matched to old combined catalog by GRB name: {len(perera_not_matched_by_grb_name)}")
    else:
        old_catalog_grb_names = set()
        perera_grb_names = set()


        old_catalog_grb_names = set(df1_prefixed["GRB"]) | set(df2_prefixed["GRB"]) | set(swift_combined_df["GRB"])
        perera_grb_names = set(df5_prefixed["GRB"])
    
        perera_matched_by_grb_name = perera_grb_names & old_catalog_grb_names
        perera_not_matched_by_grb_name = perera_grb_names - old_catalog_grb_names
    
        print(f"Perera GRBs matched to old combined catalog by GRB name: {len(perera_matched_by_grb_name)}")
        print(f"Perera GRBs not matched to old combined catalog by GRB name: {len(perera_not_matched_by_grb_name)}")
    # ============================================================
    #  Sort before merging
    # ============================================================
    df1_prefixed = df1_prefixed.sort_values(by="GRB").reset_index(drop=True)
    df2_prefixed = df2_prefixed.sort_values(by="GRB").reset_index(drop=True)
    swift_combined_df = swift_combined_df.sort_values(by="GRB").reset_index(drop=True)

    if USE_NEW_ANALYSIS_CATALOG:
        df5_prefixed = df5_prefixed.sort_values(by="GRB").reset_index(drop=True)

    # ============================================================
    #  Final merge:
    #  Fermi GBM + Fermi LAT + combined Swift catalog + Perera catalog
    # ============================================================
    combined_df = df1_prefixed.merge(df2_prefixed, on="GRB", how="outer")
    combined_df = combined_df.merge(swift_combined_df, on="GRB", how="outer")

    if USE_NEW_ANALYSIS_CATALOG:
        combined_df = combined_df.merge(df5_prefixed, on="GRB", how="outer")

    combined_df = combined_df.sort_values(by="GRB").reset_index(drop=True)

    combined_df = standardize_coordinate_columns_to_j2000_deg(combined_df)
    combined_df = standardize_error_radius_columns_to_90_arcsec(combined_df)

    best_of_rows = combined_df.apply(build_best_of_row, axis=1)
    best_of_df = pd.DataFrame(best_of_rows.tolist())

    numeric_cols = [
        "RA",
        "DEC",
        "Error Radius 90% [arcsec]",
        "Duplicate Error Radius 90% [arcsec]",
        "T90 [s]",
        "T90 Error [s]",
        "T90 Start [s]",
        "T90 Trigger MJD",
        "Fluence [erg/cm^2]",
        "Fluence Error 1 Sigma [erg/cm^2]",
        "MJD"
    ]

    for col in numeric_cols:
        best_of_df[col] = pd.to_numeric(best_of_df[col], errors="coerce")

    best_of_df = best_of_df[
        [
            "GRB",
            "RA",
            "DEC",
            "Error Radius 90% [arcsec]",
            "Duplicate Error Radius 90% [arcsec]",
            "T90 [s]",
            "T90 Error [s]",
            "T90 Start [s]",
            "Fluence [erg/cm^2]",
            "Fluence Error 1 Sigma [erg/cm^2]",
            "Redshift",
            "MJD",
            "T90 Trigger MJD",
            "T90 Trigger UTC",
            "T90 catalog",
            "T90 source GRB",
            "Original entries JSON",
            "Position catalog"
        ]
    ].sort_values(by="GRB", ascending=False).reset_index(drop=True)

    print()
    print("Best-of before duplicate merging:", best_of_df.shape)
    
    # ============================================================
    #  Duplicate check without Perera catalog
    # ============================================================
    if USE_NEW_ANALYSIS_CATALOG:
        perera_bestof_mask = best_of_df["Position catalog"].astype(str).str.strip() == "Perera et al."
    
        best_of_df_without_perera = best_of_df[
            ~perera_bestof_mask
        ].copy().reset_index(drop=True)
    
        best_of_df_perera = best_of_df[
            perera_bestof_mask
        ].copy().reset_index(drop=True)
    
        print()
        print("Duplicate check excluding the Perera catalog:")
        print(f"Best-of entries used for duplicate check: {len(best_of_df_without_perera)}")
    
        best_of_df_without_perera, duplicate_pairs_df = merge_bestof_duplicates(
            best_of_df_without_perera
        )
    
        best_of_df = pd.concat(
            [best_of_df_without_perera, best_of_df_perera],
            ignore_index=True
        )
    
    else:
        best_of_df, duplicate_pairs_df = merge_bestof_duplicates(best_of_df)
    
    best_of_df = best_of_df.sort_values(
        by="GRB",
        ascending=False
    ).reset_index(drop=True)
    

    best_of_df = best_of_df.drop(
        columns=["Duplicate Error Radius 90% [arcsec]"],
        errors="ignore"
    )

    combined_df_sorted = combined_df.sort_values(
        by="GRB",
        ascending=False
    ).reset_index(drop=True)

    columns_to_drop = [
        "4Trig_time_met",
        "2gcn_name",
        "3Trigger Number",
        "4Trig_ID",
        "4Image_SNR",
        "4T50",
        "4T50_err",
        "4Evt_start_sincetrig",
        "4Evt_stop_sincetrig",
        "4pcode",
        "4Trigger_method",
        "4comment",
        "4XRT_detection"
    ]

    combined_df_reduced = combined_df_sorted.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    combined_df_reduced = combined_df_reduced.drop_duplicates(
        subset="GRB",
        keep="first"
    )

    final_df = best_of_df.merge(
        combined_df_reduced,
        on="GRB",
        how="left"
    )

    duration_columns = {"1": "1t90", "2": "2t90", "3": "3BAT T90 [sec]",
                        "4": "4T90", "5": "5" + NEW_ANALYSIS_T90_SOURCE}
    for index, row in final_df.iterrows():
        records = json.loads(row["Original entries JSON"])
        for key, duration_column in duration_columns.items():
            columns = [c for c in final_df.columns if str(c).startswith(key)]
            candidates = [r for r in records if any(not is_missing(r.get(c)) for c in columns)]
            if not candidates:
                continue
            winners = [r for r in candidates if key == str(row["T90 catalog"])
                       and str(r.get("GRB")) == str(row["T90 source GRB"])
                       and to_float(r.get(duration_column)) == to_float(row["T90 [s]"])]
            def rank(record):
                value = to_float(record.get(duration_column))
                return value if pd.notna(value) else -np.inf
            selected = winners[0] if winners else max(candidates, key=rank)
            for column in columns:
                value = selected.get(column)
                if pd.api.types.is_numeric_dtype(final_df[column]):
                    value = to_float(value)
                final_df.at[index, column] = value

    # Rename only after duplicate detection, preserving its existing behavior.
    t90_mjd = pd.to_numeric(
        final_df["T90 Trigger MJD"],
        errors="coerce"
    )
    
    position_mjd = pd.to_numeric(
        final_df["MJD"],
        errors="coerce"
    )
    
    # Prefer the trigger time associated with the selected T90.
    # If no T90 trigger time is available, retain the previously calculated MJD.
    final_df["MJD"] = t90_mjd.fillna(position_mjd)
    names = {"1": "Fermi/GBM", "2": "Fermi/LAT", "3": "Swift Burst Table BAT",
             "4": "Swift/BAT", "5": "Perera et al."}
    final_df["T90 catalog"] = final_df["T90 catalog"].map(names).fillna("")
    final_df = final_df.drop(columns=[
        "T90 Trigger MJD", "T90 Trigger UTC", "T90 source GRB", "Original entries JSON"
    ])

    # ============================================================
    #  Reorder final catalog columns
    # ============================================================
    preferred_column_order = [
        "GRB",
        "RA",
        "DEC",
        "Error Radius 90% [arcsec]",
        "T90 [s]",
        "T90 Error [s]",
        "T90 Start [s]",
        "Fluence [erg/cm^2]",
        "Fluence Error 1 Sigma [erg/cm^2]",
        "Redshift",
        "MJD",
        "T90 Trigger MJD",
        "T90 Trigger UTC",
        "T90 catalog",
        "T90 source GRB",
        "Original entries JSON",
        "Position catalog",

        # Original Fermi GBM columns
        "1ra",
        "1dec",
        "1error_radius_90_arcsec",
        "1t90",
        "1t90_error",
        "1t90_start",
        "1fluence",
        "1fluence_error",
        "1trigger_time",

        # Original Fermi LAT columns
        "2ra",
        "2dec",
        "2error_radius_90_arcsec",
        "2t90",
        "2t90_error",
        "2fluence",
        "2fluence_error",
        "2time",
        "2redshift",

        # Original Swift catalog 3 columns
        "3name",
        "3Time [UT]",
        "3BAT RA",
        "3BAT Dec",
        "3BAT 90% Error [arcsec]",
        "3BAT T90 [sec]",
        "3BAT Fluence",
        "3BAT Fluence 90% Error",
        "3XRT RA",
        "3XRT Dec",
        "3XRT 90% Error [arcsec]",
        "3UVOT RA",
        "3UVOT Dec",
        "3UVOT 90% Error [arcsec]",
        "3Redshift",

        # Original Swift catalog 4 columns
        "4GRBname",
        "4Trig_time_UTC",
        "4RA_ground",
        "4DEC_ground",
        "4Image_position_err_90_arcsec",
        "4T90",
        "4T90_err",
        "4T90_start",

        # New Perera et al. 2026 catalog columns
        "5grb_name",
        "5trigtime",
        "5duration",
        "5pe_duration",
        "5pastro",
        "5pbat",
        "5classification",
        "5gbm_catalog",
        "5ra_median",
        "5dec_median",
        "5ra_err_plus",
        "5ra_err_minus",
        "5dec_err_plus",
        "5dec_err_minus",
        "5pos_error_90_arcsec",
    ]

    existing_preferred_columns = [
        col for col in preferred_column_order
        if col in final_df.columns
    ]

    remaining_columns = [
        col for col in final_df.columns
        if col not in existing_preferred_columns
    ]

    final_df = final_df[
        existing_preferred_columns + remaining_columns
    ]

    return final_df, best_of_df, combined_df, duplicate_pairs_df


# ============================================================
#  Saving
# ============================================================
def save_catalog(final_df):
    final_df = final_df.copy()
    final_df["MJD"] = final_df["MJD"].apply(
        lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
    )
    col_widths = {}

    for col in final_df.columns:
        max_len_col = len(str(col))
        max_len_data = final_df[col].apply(lambda x: len(format_value(x))).max()
        col_widths[col] = max(max_len_col, max_len_data) + 2

    with open(output_txt, "w", encoding="utf-8") as f:
        header_line = ""

        for col in final_df.columns:
            header_line += str(col).ljust(col_widths[col])

        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")

        for _, row in final_df.iterrows():
            line = ""

            for col in final_df.columns:
                value = format_value(row[col])
                line += value.ljust(col_widths[col])

            f.write(line + "\n")

    final_df.to_csv(output_tsv, sep="\t", index=False)
    final_df.to_csv(output_csv, sep=",", index=False)

    print()
    print(f"TXT saved to: {output_txt}")
    print(f"TSV saved to: {output_tsv}")
    print(f"CSV saved to: {output_csv}")


# ============================================================
#  Main
# ============================================================
def main():
    if CREATE_CATALOG:
        check_file_is_writable(duplicate_pairs_path)

        if WRITE_CATALOG:
            check_output_files_are_writable()

    if CREATE_CATALOG:
        final_df, best_of_df, combined_df, duplicate_pairs_df = create_catalog()

        duplicate_pairs_df.to_csv(
            duplicate_pairs_path,
            sep="\t",
            index=False
        )

        print(f"Duplicate pairs saved to: {duplicate_pairs_path}")

        print()
        print("Best-of catalog shape:", best_of_df.shape)
        print("Final catalog shape:", final_df.shape)
        

        if WRITE_CATALOG:
            save_catalog(final_df)

        else:
            print()
            print("WRITE_CATALOG = False -> Catalog was not written.")


    else:
        print()
        print("CREATE_CATALOG = False -> Catalog will not be recreated.")

    if not CREATE_CATALOG:
        if output_tsv.exists():
            final_df = pd.read_csv(output_tsv, sep="\t")
            print(f"Existing catalog loaded: {output_tsv}")
            print("Shape:", final_df.shape)

        else:
            print(f"No existing TSV file found: {output_tsv}")

        if duplicate_pairs_path.exists():
            duplicate_pairs_df = pd.read_csv(duplicate_pairs_path, sep="\t")
            print(f"Existing duplicate pairs loaded: {duplicate_pairs_path}")
            print("Duplicate pairs shape:", duplicate_pairs_df.shape)


        else:
            print(f"No saved duplicate-pair file found: {duplicate_pairs_path}")


if __name__ == "__main__":
    main()
