"""
Rule-based (deterministic, no LLM) anomaly detection over AIS position reports.

Expected input dataframe columns:
    mmsi        : str or int, vessel MMSI
    name        : str, vessel name
    lat, lon    : float, position
    speed       : float, knots (SOG)
    timestamp   : pandas.Timestamp, UTC

These functions are intentionally simple and explainable -- every flag has a
plain-language reason attached, so the LLM layer on top never has to justify
a black-box score.
"""

from itertools import combinations

import numpy as np
import pandas as pd


def detect_ais_gaps(df: pd.DataFrame, gap_hours: float = 4.0) -> pd.DataFrame:
    """Flag vessels with a silent gap longer than `gap_hours` between consecutive AIS pings."""
    flags = []
    for mmsi, group in df.sort_values("timestamp").groupby("mmsi"):
        group = group.sort_values("timestamp")
        gaps = group["timestamp"].diff().dt.total_seconds() / 3600
        for idx, gap in gaps.items():
            if gap is not None and gap >= gap_hours:
                row = group.loc[idx]
                prev_row = group.loc[group.index[group.index.get_loc(idx) - 1]]
                flags.append({
                    "mmsi": mmsi,
                    "name": row["name"],
                    "flag_type": "ais_gap",
                    "reason": f"went dark for {gap:.1f}h (last seen {prev_row['lat']:.2f},{prev_row['lon']:.2f})",
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "timestamp": row["timestamp"],
                })
    return pd.DataFrame(flags)


def detect_loitering(df: pd.DataFrame, speed_kn: float = 1.5,
                      min_duration_hours: float = 3.0, max_radius_km: float = 3.0) -> pd.DataFrame:
    """Flag vessels moving under `speed_kn` and staying within `max_radius_km`
    for at least `min_duration_hours`, in open water (caller should pre-filter out port zones)."""
    flags = []
    for mmsi, group in df.sort_values("timestamp").groupby("mmsi"):
        group = group.sort_values("timestamp")
        slow = group[group["speed"] <= speed_kn]
        if slow.empty:
            continue
        # find contiguous slow-speed runs
        slow_idx = slow.index.to_series()
        run_id = (slow_idx.diff() != 1).cumsum()
        for _, run in slow.groupby(run_id):
            if len(run) < 2:
                continue
            duration = (run["timestamp"].max() - run["timestamp"].min()).total_seconds() / 3600
            if duration < min_duration_hours:
                continue
            lat0, lon0 = run.iloc[0][["lat", "lon"]]
            radius_km = _haversine_km(lat0, lon0, run["lat"].values, run["lon"].values).max()
            if radius_km <= max_radius_km:
                flags.append({
                    "mmsi": mmsi,
                    "name": run.iloc[0]["name"],
                    "flag_type": "loitering",
                    "reason": f"loitered {duration:.1f}h within {radius_km:.1f}km, speed <= {speed_kn}kn",
                    "lat": run["lat"].mean(),
                    "lon": run["lon"].mean(),
                    "timestamp": run["timestamp"].max(),
                })
    return pd.DataFrame(flags)


def detect_rendezvous(df: pd.DataFrame, proximity_km: float = 0.5,
                       speed_kn: float = 3.0, window_minutes: int = 30) -> pd.DataFrame:
    """Flag pairs of vessels that come within `proximity_km` of each other while
    both are moving under `speed_kn`, within the same time window -- a possible
    ship-to-ship transfer signature."""
    flags = []
    df = df.sort_values("timestamp")
    for ts, snapshot in df.groupby(pd.Grouper(key="timestamp", freq=f"{window_minutes}min")):
        slow = snapshot[snapshot["speed"] <= speed_kn]
        vessels = slow["mmsi"].unique()
        for m1, m2 in combinations(vessels, 2):
            r1 = slow[slow["mmsi"] == m1].iloc[-1]
            r2 = slow[slow["mmsi"] == m2].iloc[-1]
            dist = _haversine_km(r1["lat"], r1["lon"], np.array([r2["lat"]]), np.array([r2["lon"]]))[0]
            if dist <= proximity_km:
                flags.append({
                    "mmsi": f"{m1}+{m2}",
                    "name": f"{r1['name']} & {r2['name']}",
                    "flag_type": "rendezvous",
                    "reason": f"came within {dist*1000:.0f}m of each other, both under {speed_kn}kn",
                    "lat": (r1["lat"] + r2["lat"]) / 2,
                    "lon": (r1["lon"] + r2["lon"]) / 2,
                    "timestamp": ts,
                })
    return pd.DataFrame(flags)


def run_all_detectors(df: pd.DataFrame) -> pd.DataFrame:
    """Run all three detectors and return a single combined flagged-vessel dataframe."""
    parts = [detect_ais_gaps(df), detect_loitering(df), detect_rendezvous(df)]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["mmsi", "name", "flag_type", "reason", "lat", "lon", "timestamp"])
    return pd.concat(parts, ignore_index=True)


def _haversine_km(lat0, lon0, lats, lons):
    """Vectorized haversine distance in km from a single point to arrays of points."""
    r = 6371.0
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    lats, lons = np.radians(lats), np.radians(lons)
    dlat = lats - lat0
    dlon = lons - lon0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lats) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
