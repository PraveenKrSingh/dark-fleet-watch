"""
The three tools exposed to the agent. Each is a plain Python function --
deterministic, testable on its own, no LLM involved. The agent's only job
is deciding *when* to call which one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from anomaly_detection import run_all_detectors
from mid_lookup import mid_to_iso

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "hormuz_synthetic.csv"


def _load_positions() -> pd.DataFrame:
    return pd.read_csv(_DATA_PATH, parse_dates=["timestamp"])


def get_flagged_vessels() -> list[dict]:
    """Return every vessel currently flagged by the anomaly detectors, with reason and flag state."""
    df = _load_positions()
    flagged = run_all_detectors(df)
    out = []
    for _, row in flagged.iterrows():
        mmsi = str(row["mmsi"]).split("+")[0]
        out.append({
            "mmsi": row["mmsi"],
            "name": row["name"],
            "flag_type": row["flag_type"],
            "flag_state": mid_to_iso(mmsi),
            "reason": row["reason"],
        })
    return out


def get_vessel_history(mmsi: str) -> list[dict]:
    """Return the full position history for a given MMSI, sorted by time."""
    df = _load_positions()
    history = df[df["mmsi"].astype(str) == str(mmsi)].sort_values("timestamp")
    if history.empty:
        return [{"error": f"no positions found for MMSI {mmsi}"}]
    return history[["timestamp", "lat", "lon", "speed"]].astype(str).to_dict(orient="records")


def explain_flag(mmsi: str) -> dict:
    """Return the specific anomaly rule(s) that flagged a given MMSI, if any."""
    df = _load_positions()
    known_mmsis = set(df["mmsi"].astype(str))
    if str(mmsi) not in known_mmsis:
        return {"mmsi": mmsi, "error": f"MMSI {mmsi} does not exist in this dataset -- "
                                        "if you only know the vessel's name, call find_vessel_by_name first"}
    flagged = get_flagged_vessels()
    matches = [f for f in flagged if str(f["mmsi"]).split("+")[0] == str(mmsi) or str(mmsi) in str(f["mmsi"])]
    if not matches:
        return {"mmsi": mmsi, "flagged": False, "reason": "this vessel is not currently flagged"}
    return {"mmsi": mmsi, "flagged": True, "matches": matches}


def find_vessel_by_name(name: str) -> dict:
    """Look up a vessel's MMSI and flag state by its name (case-insensitive, partial match ok).
    Use this whenever you know a vessel's name but not its MMSI."""
    df = _load_positions()
    matches = df[df["name"].str.contains(name, case=False, na=False, regex=False)]
    if matches.empty:
        return {"found": False, "reason": f"no vessel matching '{name}' in the dataset"}
    results = []
    for mmsi, group in matches.groupby("mmsi"):
        results.append({
            "mmsi": mmsi,
            "name": group.iloc[0]["name"],
            "flag_state": mid_to_iso(str(mmsi)),
        })
    return {"found": True, "vessels": results}


if __name__ == "__main__":
    # quick manual check
    flagged = get_flagged_vessels()
    print(f"{len(flagged)} vessels currently flagged:")
    for f in flagged:
        print(f"  {f['name']} ({f['flag_state']}) - {f['flag_type']}: {f['reason']}")
