import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from anomaly_detection import run_all_detectors
from mid_lookup import mid_to_iso

df = pd.read_csv("data/hormuz_synthetic.csv", parse_dates=["timestamp"])
flagged = run_all_detectors(df)

print(f"total positions: {len(df)}, vessels: {df['mmsi'].nunique()}, flagged events: {len(flagged)}\n")
for _, row in flagged.iterrows():
    iso = mid_to_iso(str(row["mmsi"]).split("+")[0])
    print(f"[{row['flag_type']:10s}] {row['name']:25s} flag={iso}  -> {row['reason']}")
