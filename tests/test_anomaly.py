import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anomaly_detection import run_all_detectors  # noqa: E402
from mid_lookup import flag_label, mid_to_iso  # noqa: E402


def make_synthetic_ais() -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-07-20 00:00:00")

    # Vessel A: normal transit, steady speed, no anomalies
    for i in range(6):
        rows.append(dict(mmsi="470123456", name="MV Ocean Pearl",
                          lat=26.55 + i * 0.02, lon=56.25 + i * 0.03,
                          speed=12.0, timestamp=base + pd.Timedelta(hours=i)))

    # Vessel B: goes dark for 6 hours near the strait, reappears displaced
    rows.append(dict(mmsi="422987654", name="MV Al-Noor",
                      lat=26.60, lon=56.30, speed=10.0, timestamp=base))
    rows.append(dict(mmsi="422987654", name="MV Al-Noor",
                      lat=26.70, lon=56.45, speed=9.0, timestamp=base + pd.Timedelta(hours=6, minutes=30)))

    # Vessel C: loiters in open water for 4 hours at near-zero speed
    for i in range(5):
        rows.append(dict(mmsi="372555111", name="MV Sagar Kanya",
                          lat=26.30 + i * 0.001, lon=56.10 + i * 0.001,
                          speed=0.4, timestamp=base + pd.Timedelta(hours=i)))

    # Vessels D & E: rendezvous -- converge and both slow down in the same window
    rows.append(dict(mmsi="636111222", name="Nordic Star",
                      lat=26.40, lon=56.00, speed=1.0, timestamp=base + pd.Timedelta(hours=2)))
    rows.append(dict(mmsi="563222333", name="Pacific Trader",
                      lat=26.4003, lon=56.0002, speed=1.2, timestamp=base + pd.Timedelta(hours=2)))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_synthetic_ais()
    flagged = run_all_detectors(df)

    print(f"input positions: {len(df)}")
    print(f"flagged events : {len(flagged)}\n")

    for _, row in flagged.iterrows():
        iso = mid_to_iso(str(row["mmsi"]).split("+")[0])
        print(f"[{row['flag_type']:10s}] {row['name']:30s} flag={iso}  -> {row['reason']}")

    print("\nsample tooltip html:")
    print(flag_label("470123456", "MV Ocean Pearl", "normal"))
