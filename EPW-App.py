# Streamlit EPW Viewer
# -------------------------------------------------------------
# A Streamlit app that fetches an EPW file from a URL, parses it,
# shows metadata, quick QA/QC, plots, and allows CSV download.
# -------------------------------------------------------------
# How to run locally:
#   1) pip install streamlit pandas numpy requests
#   2) streamlit run app.py
# -------------------------------------------------------------

from __future__ import annotations
import io
import math
import textwrap
from datetime import datetime, timedelta, timezone
import hashlib
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
import plotly.graph_objects as go
try:
    from streamlit_javascript import st_javascript
except Exception:
    st_javascript = None

# ---------------------------
# Constants
# ---------------------------
EPW_COLS = [
    "Year", "Month", "Day", "Hour", "Minute",
    "Data Source and Uncertainty Flags",
    "Dry Bulb Temperature (C)",
    "Dew Point Temperature (C)",
    "Relative Humidity (%)",
    "Atmospheric Station Pressure (Pa)",
    "Extraterrestrial Horizontal Radiation (Wh/m2)",
    "Extraterrestrial Direct Normal Radiation (Wh/m2)",
    "Horizontal Infrared Radiation Intensity (Wh/m2)",
    "Global Horizontal Radiation (Wh/m2)",
    "Direct Normal Radiation (Wh/m2)",
    "Diffuse Horizontal Radiation (Wh/m2)",
    "Global Horizontal Illuminance (lux)",
    "Direct Normal Illuminance (lux)",
    "Diffuse Horizontal Illuminance (lux)",
    "Zenith Luminance (Cd/m2)",
    "Wind Direction (deg)",
    "Wind Speed (m/s)",
    "Total Sky Cover (tenths)",
    "Opaque Sky Cover (tenths)",
    "Visibility (km)",
    "Ceiling Height (m)",
    "Present Weather Observation",
    "Present Weather Codes",
    "Precipitable Water (mm)",
    "Aerosol Optical Depth (dimensionless)",
    "Snow Depth (cm)",
    "Days Since Last Snowfall",
    "Albedo",
    "Liquid Precipitation Depth (mm)",
    "Liquid Precipitation Quantity (hr)",
]

NULL_SENTINELS = {"", "NA", "N/A", "null"}
# Typical EPW missings: 99, 99.9, 999, 9999, 99999 etc. We'll coerce any absolute value >= 9_000 to NaN after parsing.

# ---------------------------
# Helpers
# ---------------------------

def _safe_float(x):
    try:
        if isinstance(x, str) and x.strip() in NULL_SENTINELS:
            return np.nan
        v = float(x)
        if abs(v) >= 9000:
            return np.nan
        # EPW often uses 99 or 99.9 as missing for temps/humidity
        if v in (99.0, 99.9):
            return np.nan
        return v
    except Exception:
        return np.nan


def _decode_epw(epw_bytes: bytes) -> str:
    """Decode EPW bytes robustly, trying common encodings.
    Tries UTF-8 (with BOM), then cp1252, then latin-1; finally falls back to replacement.
    """
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return epw_bytes.decode(enc)
        except UnicodeDecodeError:
            pass
    for enc in ("cp1252", "latin-1"):
        try:
            return epw_bytes.decode(enc)
        except UnicodeDecodeError:
            pass
    return epw_bytes.decode("utf-8", errors="replace")


@st.cache_data(show_spinner=False, ttl=24*3600)
def parse_epw_bytes(epw_bytes: bytes) -> tuple[dict, pd.DataFrame, str]:
    """
    Parse EPW from raw bytes.

    Returns: (metadata dict, dataframe with DateTime index in local time (naive), original_text)
    """
    text = _decode_epw(epw_bytes)
    lines = text.splitlines()
    if not lines:
        raise ValueError("Empty EPW file")

    # ---- Parse the LOCATION line (first line)
    # EPW LOCATION format:
    # 0: LOCATION, 1: City, 2: State/Province, 3: Country, 4: Source,
    # 5: WMO, 6: Latitude, 7: Longitude, 8: Time Zone (hours from GMT), 9: Elevation (m)
    header = lines[0].split(',')
    if not header or header[0].strip().upper() != "LOCATION":
        raise ValueError("EPW must start with a LOCATION line")

    md = {
        "city": header[1].strip() if len(header) > 1 else "",
        "region": header[2].strip() if len(header) > 2 else "",
        "country": header[3].strip() if len(header) > 3 else "",
        "source": header[4].strip() if len(header) > 4 else "",
        "wmo": header[5].strip() if len(header) > 5 else "",
        "latitude": _safe_float(header[6]) if len(header) > 6 else np.nan,
        "longitude": _safe_float(header[7]) if len(header) > 7 else np.nan,
        "timezone": _safe_float(header[8]) if len(header) > 8 else 0.0,
        "elevation_m": _safe_float(header[9]) if len(header) > 9 else np.nan,
    }

    # ---- Find the start of data rows: after the line starting with 'DATA PERIODS'
    data_start = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("DATA PERIODS"):
            data_start = i + 1  # Next line is the first data row
            break
    if data_start is None:
        # Some EPWs have 'DATA PERIODS," followed by one line describing periods, then data
        for i, ln in enumerate(lines):
            if ln.upper().startswith("DATA") and "PERIOD" in ln.upper():
                data_start = i + 1
                break
    if data_start is None:
        # Fallback: some EPWs start data on line 9 (1-based) after fixed header. We'll attempt after 8 lines
        data_start = 8

    data_lines = lines[data_start:]
    # Read using pandas from the joined string buffer
    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=EPW_COLS,
        # Read all columns; select after to avoid pyarrow include_columns issues
        engine="pyarrow",                     # <-- faster if available (pandas >=2.0)
        # pyarrow engine requires na_values to be strings
        na_values=list(NULL_SENTINELS) + ["999", "9999", "99999"],
    )

    # Keep only the columns we actually use
    desired_cols = [
        "Year","Month","Day","Hour","Minute",
        "Dry Bulb Temperature (C)",
        "Relative Humidity (%)",
        "Global Horizontal Radiation (Wh/m2)",
        "Direct Normal Radiation (Wh/m2)",
        "Diffuse Horizontal Radiation (Wh/m2)",
        "Wind Speed (m/s)",
        "Wind Direction (deg)",
    ]
    df = df[[c for c in desired_cols if c in df.columns]]

    # Numeric coercion, sentinel cleanup, rounding, and downcast to float32
    radiation_cols = {
        "Global Horizontal Radiation (Wh/m2)",
        "Direct Normal Radiation (Wh/m2)",
        "Diffuse Horizontal Radiation (Wh/m2)",
    }
    for c in df.columns:
        if c not in ("Year","Month","Day","Hour","Minute"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Mask extreme EPW sentinels like 9999 etc.
            df.loc[df[c].abs() >= 9000, c] = np.nan
            # Radiation to 0 decimals; others to 1 decimal
            if c in radiation_cols:
                df[c] = df[c].round(0)
            else:
                df[c] = df[c].round(1)
            df[c] = df[c].astype("float32")

    # Columns are now numeric, rounded, and downcast; no additional coercion here

    # If the first data row contains 99 or 99.9 for specific fields,
    # treat the entire column as missing (NaN)
    if not df.empty:
        sentinel_cols = [
            "Dry Bulb Temperature (C)",
            "Relative Humidity (%)",
            "Dew Point Temperature (C)",
        ]
        for col in sentinel_cols:
            if col in df.columns:
                first_val = df[col].iloc[0]
                if pd.notna(first_val) and float(first_val) in (99.0, 99.9):
                    df[col] = np.nan

    # Build a proper datetime. EPW uses 1–24 for Hour as the END of the hour.
    # We'll convert to a timestamp at the END of hour, then shift by -1 hour to get start-of-hour timestamps.
    # Many EPWs set Minute=60; we will normalize to 0 minutes after the shift.
    dt = pd.to_datetime(
        {
            "year": df["Year"].ffill().astype(int),
            "month": df["Month"].astype(int),
            "day": df["Day"].astype(int),
        },
        errors="coerce",
    )
    # Hour handling
    hrs = df["Hour"].fillna(1).astype(int).clip(1, 24)
    # end-of-hour -> subtract 1 hour
    dt = dt + pd.to_timedelta(hrs, unit="h") - pd.to_timedelta(1, unit="h")

    # Apply timezone info if useful (we keep naive for plotting, but store tz offset)
    tz_offset = md.get("timezone") or 0.0
    md["timezone_hours"] = tz_offset

    df.index = dt
    df.index.name = "Datetime (local)"

    return md, df, text


@st.cache_data(show_spinner=False, ttl=24*3600)
def fetch_epw(url: str) -> tuple[dict, pd.DataFrame, str]:
    """Download and parse EPW from URL."""
    if not url:
        raise ValueError("Please provide an EPW URL")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return parse_epw_bytes(r.content)


def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    # Monthly groups with Month indexed 0..11 (0=Jan, 11=Dec)
    month0 = (df.index.month - 1).rename("Month")
    g = df.groupby([df.index.year.rename("Year"), month0])
    out = pd.DataFrame({
        "T_mean (C)": g["Dry Bulb Temperature (C)"].mean(),
        "GHI sum (kWh/m2)": g["Global Horizontal Radiation (Wh/m2)"].sum() / 1000.0,
        "DNI sum (kWh/m2)": g["Direct Normal Radiation (Wh/m2)"].sum() / 1000.0,
        "DHI sum (kWh/m2)": g["Diffuse Horizontal Radiation (Wh/m2)"].sum() / 1000.0,
        "Wind mean (m/s)": g["Wind Speed (m/s)"].mean(),
        "RH mean (%)": g["Relative Humidity (%)"].mean(),
    }).reset_index()

    # Yearly summary (13th row): Month = 12
    # Means are over all hours; radiation is the sum of monthly sums
    rad_cols = [
        "GHI sum (kWh/m2)",
        "DNI sum (kWh/m2)",
        "DHI sum (kWh/m2)",
    ]
    rad_sums = out[rad_cols].sum(numeric_only=True)
    yearly = pd.DataFrame({
        "Year": [int(df.index.year.min()) if df.index.year.notna().any() else 0],
        "Month": [12],
        "T_mean (C)": [df["Dry Bulb Temperature (C)"].mean()],
        "GHI sum (kWh/m2)": [rad_sums.get("GHI sum (kWh/m2)", 0.0)],
        "DNI sum (kWh/m2)": [rad_sums.get("DNI sum (kWh/m2)", 0.0)],
        "DHI sum (kWh/m2)": [rad_sums.get("DHI sum (kWh/m2)", 0.0)],
        "Wind mean (m/s)": [df["Wind Speed (m/s)"].mean()],
        "RH mean (%)": [df["Relative Humidity (%)"].mean()],
    })

    full = pd.concat([out, yearly], ignore_index=True)
    # Round all physical values to 1 decimal
    val_cols = [c for c in full.columns if c not in ("Year", "Month")]
    full[val_cols] = full[val_cols].astype(float).round(1)

    # Add friendly month names with 0..11 = Jan..Dec and 12 = Year
    month_names = {i: name for i, name in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]) }
    month_names[12] = "Year"
    full["Month Name"] = full["Month"].map(month_names)

    # Sort by Year then Month 0..12
    full = full.sort_values(["Year", "Month"], ascending=[True, True]).reset_index(drop=True)
    return full


@st.cache_data(show_spinner=False)
def compute_monthly_agg(source_key: str, df: pd.DataFrame) -> pd.DataFrame:
    # source_key ensures cache invalidates when source changes
    return monthly_agg(df)


def quick_metrics(df: pd.DataFrame) -> dict:
    t = df["Dry Bulb Temperature (C)"]
    ghi = df["Global Horizontal Radiation (Wh/m2)"]
    wind = df["Wind Speed (m/s)"]

    return {
        "T_min (C)": float(np.nanmin(t)),
        "T_max (C)": float(np.nanmax(t)),
        "T_mean (C)": float(np.nanmean(t)),
        "Total GHI (kWh/m2)": float(np.nansum(ghi) / 1000.0),
        "Mean wind (m/s)": float(np.nanmean(wind)),
        "Records": int(len(df)),
        "Missing (%)": float(100.0 * df.isna().mean().mean()),
    }


@st.cache_data(show_spinner=False)
def compute_quick_metrics(source_key: str, df: pd.DataFrame) -> dict:
    # source_key ensures cache invalidates when source changes
    return quick_metrics(df)


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="NMBU - EPW Viewer", layout="wide")

left, right = st.columns([7, 1])
with left:
    st.title("NMBU - EPW Viewer")
    st.caption("Drop a .epw file, or paste a direct .epw URL.")
with right:
    try:
        st.image("Logos/NMBU_Logo_Bokmaal_RGB.png", caption="", width=600)
    except Exception:
        pass

# Measure available container width to avoid first-draw resizing
def _get_container_width(default: int = 1000) -> int:
    try:
        if st_javascript is None:
            return default
        js = """
        (() => {
          const el = parent.document.querySelector('.block-container');
          return el ? el.clientWidth : window.innerWidth;
        })()
        """
        val = st_javascript(js)
        w = int(float(val)) if val is not None else default
        return max(320, min(w, 2000))
    except Exception:
        return default

PLOT_WIDTH = _get_container_width()

# Cached chart builders to avoid redraw cost on reruns/tab switches
@st.cache_data(show_spinner=False)
def build_time_series_fig(source_key: str, cols: tuple, res: str, width: int) -> dict:
    df = st.session_state.df
    plot_df = df[list(cols)]
    fig = go.Figure()
    if res == "Hourly (raw)":
        plot_df = plot_df.round(1)
        for col in plot_df.columns:
            fig.add_trace(go.Scattergl(x=plot_df.index, y=plot_df[col], name=col, mode="lines"))
    else:
        rule = {"Daily mean": "D", "Weekly mean": "W", "Monthly mean": "M"}[res]
        agg_df = plot_df.resample(rule).mean().round(1)
        for col in agg_df.columns:
            fig.add_trace(go.Scatter(x=agg_df.index, y=agg_df[col], name=col, mode="lines"))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420, width=width)
    # Show months on x-axis (no year)
    fig.update_xaxes(dtick="M1", tickformat="%b")
    return fig.to_dict()

@st.cache_data(show_spinner=False)
def build_radiation_fig(source_key: str, width: int) -> dict:
    df = st.session_state.df
    cols = [
        "Global Horizontal Radiation (Wh/m2)",
        "Direct Normal Radiation (Wh/m2)",
        "Diffuse Horizontal Radiation (Wh/m2)",
    ]
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[col].round(1), name=col, mode="lines"))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=300, width=width)
    fig.update_xaxes(dtick="M1", tickformat="%b")
    return fig.to_dict()

@st.cache_data(show_spinner=False)
def build_wind_speed_fig(source_key: str, width: int) -> dict:
    df = st.session_state.df
    fig = go.Figure(go.Scatter(x=df.index, y=df["Wind Speed (m/s)"].round(1), name="Wind Speed (m/s)", mode="lines"))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=300, width=width)
    fig.update_xaxes(dtick="M1", tickformat="%b")
    return fig.to_dict()

@st.cache_data(show_spinner=False)
def build_windrose_fig_cached(source_key: str, width: int) -> dict:
    df = st.session_state.df
    needed = {"Wind Speed (m/s)", "Wind Direction (deg)"}
    if not needed.issubset(df.columns) or df.empty:
        return go.Figure().to_dict()
    w = df[list(needed)].dropna()
    if len(w) == 0:
        return go.Figure().to_dict()
    speeds = np.clip(w["Wind Speed (m/s)"].to_numpy(dtype=float), a_min=0, a_max=None)
    dirs = (w["Wind Direction (deg)"].to_numpy(dtype=float) % 360.0)
    calm_mask = speeds < 0.2
    nb = 16
    sector_width = 360.0 / nb
    dirs_nc = dirs[~calm_mask]
    speeds_nc = speeds[~calm_mask]
    sectors = np.floor(dirs_nc / sector_width).astype(int) % nb
    cat_edges = np.array([0.2, 1.6, 3.4, 5.5, 8.0, 10.8])
    idx = np.digitize(speeds_nc, bins=cat_edges, right=False)
    cat_idx = np.clip(idx - 1, 0, 5)
    cat_labels = ["0.2–1.5", "1.6–3.3", "3.4–5.4", "5.5–7.9", "8.0–10.7", "10.8+"]
    cat_colors = ["#deebf7", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
    counts = np.zeros((nb, len(cat_labels)), dtype=int)
    if len(sectors) > 0:
        np.add.at(counts, (sectors, cat_idx), 1)
    total = counts.sum()
    perc = (counts / total * 100.0) if total > 0 else counts.astype(float)
    angle_centers = (np.arange(nb) * sector_width + sector_width / 2.0)
    widths = [sector_width] * nb
    fig = go.Figure()
    for j, label in enumerate(cat_labels):
        fig.add_trace(go.Barpolar(theta=angle_centers, r=perc[:, j], width=widths, name=label, marker_color=cat_colors[j], marker_line_color="white", marker_line_width=0.5, opacity=0.9))
    fig.update_layout(
        title=dict(text="Windrose (frequency by direction)", font=dict(size=10)),
        polar=dict(
            angularaxis=dict(direction="clockwise", rotation=90, tickfont=dict(size=9)),
            radialaxis=dict(tickfont=dict(size=9), ticksuffix="%", angle=90),
        ),
        legend=dict(orientation="v", x=1.2, xanchor="left", y=0.5, yanchor="middle", font=dict(size=9)),
        margin=dict(l=20, r=200, t=40, b=20),
        height=640,
        width=width,
    )
    return fig.to_dict()

@st.cache_data(show_spinner=False)
def compute_heatmap_grid(source_key: str, var: str) -> pd.DataFrame:
    df = st.session_state.df
    if "Hour" in df.columns:
        hours = pd.to_numeric(df["Hour"], errors="coerce").fillna(1).astype(int).clip(1, 24)
    else:
        hours = (df.index.hour + 1).astype(int)
    grid_src = pd.DataFrame({
        "DOY": df.index.dayofyear,
        "Hour": hours.values,
        "Value": df[var].values,
    }, index=df.index)
    return grid_src.drop_duplicates(subset=["DOY", "Hour"])[["DOY", "Hour", "Value"]]

@st.cache_data(show_spinner=False)
def compute_xy_tidy(source_key: str, x_var: str, y_vars: tuple) -> pd.DataFrame:
    df = st.session_state.df
    plot_df = df[[x_var] + list(y_vars)].dropna()
    return plot_df.melt(id_vars=[x_var], var_name="Variable", value_name="Value")

example_url = (
    "https://klimadataforbygninger.no/FATMY/epw/TMY-NO-1991-2020/TMYNO_Oslo_CERRA_1991-2020.epw"
)

# Allow overriding the default via URL query params, e.g. ?epw=HTTPS_URL or ?url=HTTPS_URL
def _query_param_url() -> str | None:
    try:
        qp = st.query_params  # modern API
        v = qp.get("epw") or qp.get("url")
        if isinstance(v, (list, tuple)):
            return v[0] if v else None
        return v
    except Exception:
        try:
            qp = st.experimental_get_query_params()  # legacy API returns lists
            v = qp.get("epw") or qp.get("url")
            if isinstance(v, list) and v:
                return v[0]
            return v
        except Exception:
            return None

param_url_override = _query_param_url()
if param_url_override:
    example_url = param_url_override

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader(
        "Upload EPW",
        type=["epw"],
        help="Drag and drop a .epw file here or browse",
    )
    url = st.text_input("EPW URL", value=example_url, help="Direct URL to a .epw file")
    st.markdown("---")
    st.write("Tip: Works with any reachable URL. For private files, host behind a temporary link.")

# --- Session state for auto-load and dedup ---
if "last_url" not in st.session_state:
    st.session_state.last_url = ""
if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = ""
if "meta" not in st.session_state:
    st.session_state.meta = None
    st.session_state.df = None
    st.session_state.original_text = None
    st.session_state.source_name = "weather.epw"
if "source_key" not in st.session_state:
    st.session_state.source_key = ""


def _looks_like_epw_url(u: str) -> bool:
    try:
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            return False
        # Heuristic: path ends with .epw or contains a filename-like segment
        return p.path.lower().endswith(".epw") or ".epw" in p.path.lower()
    except Exception:
        return False


# --- Auto-load on file drop or URL change ---
try:
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if file_hash != st.session_state.last_file_hash:
            with st.spinner("Parsing uploaded EPW..."):
                meta, df, original_text = parse_epw_bytes(file_bytes)
            st.session_state.meta = meta
            st.session_state.df = df
            st.session_state.original_text = original_text
            st.session_state.source_name = uploaded.name or "weather.epw"
            st.session_state.last_file_hash = file_hash
            st.session_state.source_key = f"upload:{file_hash}"
            st.session_state.last_url = ""
    elif url and _looks_like_epw_url(url) and url != st.session_state.last_url:
        with st.spinner("Fetching and parsing EPW from URL..."):
            meta, df, original_text = fetch_epw(url)
        st.session_state.meta = meta
        st.session_state.df = df
        st.session_state.original_text = original_text
        st.session_state.source_name = url.split("/")[-1] or "weather.epw"
        st.session_state.last_url = url
        st.session_state.last_file_hash = ""
        try:
            _content_hash = hashlib.md5(original_text.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            _content_hash = hashlib.md5(url.encode("utf-8", errors="ignore")).hexdigest()
        st.session_state.source_key = f"url:{_content_hash}"
except Exception as e:
    st.error(f"Failed to load EPW: {e}")

if st.session_state.df is not None and st.session_state.meta is not None:
    meta = st.session_state.meta
    df = st.session_state.df
    original_text = st.session_state.original_text

    # ---- Sidebar metadata
    with st.sidebar:
        st.subheader("Location")
        city = (meta.get("city") or "").strip()
        region = (meta.get("region") or "").strip()
        country = (meta.get("country") or "").strip()
        # Hide 'None' or empty regions and avoid odd punctuation
        show_region = region and region.lower() != "none"
        if show_region:
            loc_line = f"**{city}**, {region} {country}".strip()
        else:
            loc_line = f"**{city}** {country}".strip()
        st.write(loc_line)
        colA, colB = st.columns(2)
        with colA:
            st.metric("Latitude", f"{meta.get('latitude', np.nan):.4f}")
            st.metric("Timezone (h)", f"{meta.get('timezone_hours', 0):.1f}")
        with colB:
            st.metric("Longitude", f"{meta.get('longitude', np.nan):.4f}")
            st.metric("Elevation (m)", f"{meta.get('elevation_m', np.nan):.0f}")

    # ---- Overview
    st.success("EPW loaded successfully.")

    m = compute_quick_metrics(st.session_state.source_key, df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("T_min (C)", f"{m['T_min (C)']:.1f}")
    c2.metric("T_max (C)", f"{m['T_max (C)']:.1f}")
    c3.metric("T_mean (C)", f"{m['T_mean (C)']:.1f}")
    c4.metric("GHI total (kWh/m2)", f"{m['Total GHI (kWh/m2)']:.1f}")
    c5.metric("Wind mean (m/s)", f"{m['Mean wind (m/s)']:.1f}")
    c6.metric("Rows", f"{m['Records']}")

    st.markdown("---")

    # Tabs
    tab_map, tab_ts, tab_xy, tab_heat, tab_rad, tab_windrose, tab_month, tab_table = st.tabs([
        "Map", "Time Series", "XY Scatter", "Heatmap", "Radiation", "Windrose", "Monthly", "Table",
    ])
    # ---- Map
    with tab_map:
        st.subheader("Location map")
        lat = meta.get("latitude", np.nan)
        lon = meta.get("longitude", np.nan)
        if pd.notna(lat) and pd.notna(lon):
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
            st.caption(f"Pin at latitude {lat:.4f}, longitude {lon:.4f}.")
        else:
            st.info("This EPW lacks valid latitude/longitude metadata.")

    # ---- Time Series
    with tab_ts:
        st.subheader("Time Series: Temp, Radiation, Wind, RH")
        cols = [
            "Dry Bulb Temperature (C)",
            "Global Horizontal Radiation (Wh/m2)",
            "Direct Normal Radiation (Wh/m2)",
            "Diffuse Horizontal Radiation (Wh/m2)",
            "Wind Speed (m/s)",
            "Wind Direction (deg)",
            "Relative Humidity (%)",
        ]
        sel = st.multiselect("Variables", cols, default=[cols[0], cols[1]])
        res = st.selectbox("Resolution", ["Hourly (raw)", "Daily mean", "Weekly mean", "Monthly mean"], index=0)
        if sel:
            fig_dict = build_time_series_fig(st.session_state.source_key, tuple(sel), res, PLOT_WIDTH)
            st.plotly_chart(go.Figure(fig_dict), theme="streamlit")
        st.caption("Choose resolution to reduce points; timestamps normalized to start-of-hour.")

    # ---- XY Scatter
    with tab_xy:
        st.subheader("XY Scatter: choose X and Y variables")
        numeric_cols = [
            "Dry Bulb Temperature (C)",
            "Relative Humidity (%)",
            "Atmospheric Station Pressure (Pa)",
            "Global Horizontal Radiation (Wh/m2)",
            "Direct Normal Radiation (Wh/m2)",
            "Diffuse Horizontal Radiation (Wh/m2)",
            "Wind Speed (m/s)",
            "Wind Direction (deg)",
        ]
        x_var = st.selectbox("X axis", numeric_cols, index=0)
        y_var = st.selectbox("Y axis", numeric_cols, index=1)
        if x_var and y_var:
            plot_df = df[[x_var, y_var]].dropna().copy()
            plot_df[x_var] = plot_df[x_var].round(1)
            plot_df[y_var] = plot_df[y_var].round(1)
            chart = (
                alt.Chart(plot_df)
                .mark_circle(size=28, opacity=0.4)
                .encode(
                    x=alt.X(f"{x_var}:Q", title=x_var),
                    y=alt.Y(f"{y_var}:Q", title=y_var),
                    tooltip=[
                        alt.Tooltip(f"{x_var}:Q", title=x_var, format=".1f"),
                        alt.Tooltip(f"{y_var}:Q", title=y_var, format=".1f"),
                    ],
                )
                .properties(height=360, width=PLOT_WIDTH-40)
                .interactive()
            )
            st.altair_chart(chart)

    # ---- Radiation
    with tab_rad:
        st.subheader("Irradiance components (hourly)")
        rad_fig = build_radiation_fig(st.session_state.source_key, PLOT_WIDTH)
        st.plotly_chart(go.Figure(rad_fig), theme="streamlit")
        st.caption("Global (GHI), Direct Normal (DNI), and Diffuse (DHI) radiation.")

    # ---- Windrose
    with tab_windrose:
        st.subheader("Wind time series and windrose")

        # Show wind speed time series for full period
        if "Wind Speed (m/s)" in df.columns:
            w_fig = build_wind_speed_fig(st.session_state.source_key, PLOT_WIDTH)
            st.plotly_chart(go.Figure(w_fig), theme="streamlit")

        # Compute and render windrose for full period with 16 sectors and speed categories
        needed = {"Wind Speed (m/s)", "Wind Direction (deg)"}
        if needed.issubset(df.columns):
            w = df[list(needed)].dropna()
            if len(w) == 0:
                st.info("No wind data available.")
            else:
                speeds = np.clip(w["Wind Speed (m/s)"].to_numpy(dtype=float), a_min=0, a_max=None)
                dirs = (w["Wind Direction (deg)"].to_numpy(dtype=float) % 360.0)

                # Calm definition: < 0.2 m/s
                calm_mask = speeds < 0.2
                calm_pct = float(calm_mask.mean() * 100.0)

                # Direction sectors: fixed 16 sectors for non-calm samples
                nb = 16
                sector_width = 360.0 / nb  # 22.5 deg
                dirs_nc = dirs[~calm_mask]
                speeds_nc = speeds[~calm_mask]
                sectors = np.floor(dirs_nc / sector_width).astype(int) % nb

                # Speed categories (m/s) with edges:
                # [0.2,1.6), [1.6,3.4), [3.4,5.5), [5.5,7.9), [8.0,10.8), [10.8, inf)
                cat_edges = np.array([0.2, 1.6, 3.4, 5.5, 8.0, 10.8])
                idx = np.digitize(speeds_nc, bins=cat_edges, right=False)  # 1..6 for these bins, 6+ for >=10.8
                cat_idx = np.clip(idx - 1, 0, 5)  # 0..5
                cat_labels = ["0.2–1.5", "1.6–3.3", "3.4–5.4", "5.5–7.9", "8.0–10.7", "10.8+"]
                cat_colors = ["#deebf7", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]

                counts = np.zeros((nb, len(cat_labels)), dtype=int)
                if len(sectors) > 0:
                    np.add.at(counts, (sectors, cat_idx), 1)
                total = counts.sum()
                perc = (counts / total * 100.0) if total > 0 else counts.astype(float)

                # Angles (centers) in degrees for Plotly barpolar
                angle_centers = (np.arange(nb) * sector_width + sector_width / 2.0)
                widths = [sector_width] * nb

                fig = go.Figure()
                for j, label in enumerate(cat_labels):
                    fig.add_trace(
                        go.Barpolar(
                            theta=angle_centers,
                            r=perc[:, j],
                            width=widths,
                            name=label,
                            marker_color=cat_colors[j],
                            marker_line_color="white",
                            marker_line_width=0.5,
                            opacity=0.9,
                        )
                    )

                fig.update_layout(
                    title=dict(text="Windrose (frequency by direction)", font=dict(size=10)),
                    polar=dict(
                        angularaxis=dict(
                            direction="clockwise",
                            rotation=90,  # 0° at North
                            tickfont=dict(size=9),
                        ),
                        radialaxis=dict(
                            tickfont=dict(size=9),
                            ticksuffix="%",
                            angle=90,
                        ),
                    ),
                    legend=dict(
                        orientation="v",
                        x=1.2,
                        xanchor="left",
                        y=0.5,
                        yanchor="middle",
                        font=dict(size=9),
                    ),
                    margin=dict(l=20, r=200, t=40, b=20),
                    height=640,
                )

                fig.update_layout(width=PLOT_WIDTH)
                st.plotly_chart(fig)

                # Calm hours percentage (<0.2 m/s)
                st.caption(f"Calm (<0.2 m/s): {calm_pct:.1f}% of hours")
        else:
            st.info("Wind speed/direction columns not found in data.")

    # ---- Heatmap
    with tab_heat:
        st.subheader("Heatmap of a time series (Day-of-year × Hour)")
        heat_cols = [
            "Dry Bulb Temperature (C)",
            "Relative Humidity (%)",
            "Global Horizontal Radiation (Wh/m2)",
            "Direct Normal Radiation (Wh/m2)",
            "Diffuse Horizontal Radiation (Wh/m2)",
            "Wind Speed (m/s)",
            "Wind Direction (deg)",
        ]
        var = st.selectbox("Variable", heat_cols, index=0)

        # Build Day-of-year (1..366) × Hour (1..24 from EPW) grid
        # Prefer the EPW Hour column (end-of-hour 1..24) if available
        if "Hour" in df.columns:
            hours = pd.to_numeric(df["Hour"], errors="coerce").fillna(1).astype(int).clip(1, 24)
        else:
            hours = (df.index.hour + 1).astype(int)

        grid_src = pd.DataFrame({
            "DOY": df.index.dayofyear,
            "Hour": hours.values,
            var: df[var].values,
        }, index=df.index)
        # Keep a single hourly value per DOY×Hour; drop duplicates if any (no aggregation)
        grid = grid_src.drop_duplicates(subset=["DOY", "Hour"]).rename(columns={var: "Value"})[["DOY", "Hour", "Value"]]

        # round values for display
        grid["Value"] = pd.to_numeric(grid["Value"], errors="coerce").round(1)
        heat = (
            alt.Chart(grid)
            .mark_rect()
            .encode(
                x=alt.X("DOY:O", title="Day of year (1-366)"),
                y=alt.Y("Hour:O", title="Hour of day (1-24)"),
                color=alt.Color("Value:Q", title=f"{var}", scale=alt.Scale(scheme="viridis")),
                tooltip=["DOY", "Hour", alt.Tooltip("Value:Q", format=".1f")],
            )
            .properties(height=360, width=PLOT_WIDTH-40)
        )
        st.altair_chart(heat)

    # ---- Monthly aggregates
    with tab_month:
        st.subheader("Monthly summary by Year")
        monthly = compute_monthly_agg(st.session_state.source_key, df)
        # Backward-compatible: add Month Name if missing (e.g., cached older result)
        if "Month Name" not in monthly.columns:
            month_names = {i: name for i, name in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]) }
            month_names[12] = "Year"
            if "Month" in monthly.columns:
                monthly["Month Name"] = monthly["Month"].map(month_names)
            else:
                monthly["Month Name"] = ""
        # Reorder to show Month Name first, and hide numeric Year/Month columns for a cleaner view
        display_cols = ["Month Name"] + [c for c in monthly.columns if c not in ("Month Name", "Year", "Month")]
        st.dataframe(monthly[display_cols], width='stretch')

    # ---- Table
    with tab_table:
        st.subheader("Raw (parsed) data table")
        if "show_table" not in st.session_state:
            st.session_state.show_table = False
        colb1, colb2 = st.columns([1,1])
        with colb1:
            if not st.session_state.show_table and st.button("Show table"):
                st.session_state.show_table = True
        with colb2:
            if st.session_state.show_table and st.button("Hide table"):
                st.session_state.show_table = False
        if st.session_state.show_table:
            st.dataframe(df, width='stretch')
        else:
            st.info("Press 'Show table' to render the full dataset.")

    # Downloads removed per request
else:
    st.info("Drop a .epw file or paste a direct .epw URL in the sidebar to load.")
