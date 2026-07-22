"""
Builds a realistic-but-synthetic AIS dataset for the Strait of Hormuz:
a handful of vessels transiting the corridor normally, plus a few
deliberately anomalous ones (AIS gap, loitering, rendezvous), so the rest
of the pipeline (detectors, agent, map) can be built and demoed without
depending on AISStream's live feed being available.

Usage:
    python src/generate_hormuz_dataset.py --out data/hormuz_synthetic.csv
"""

import argparse

import numpy as np
import pandas as pd

# Approximate transit corridor through the strait (Persian Gulf <-> Gulf of Oman)
CORRIDOR_START = (26.75, 56.05)   # Gulf of Oman side (east)
CORRIDOR_END = (26.35, 56.55)     # Persian Gulf side (west)

# Vessels transiting the corridor normally, with real flag-state MIDs
NORMAL_VESSELS = [
    ("372100111", "MV Gulf Voyager", "eastbound"),   # Panama
    ("636200222", "MT Liberty Star", "westbound"),   # Liberia
    ("537300333", "MV Marshall Pride", "eastbound"), # Marshall Islands
    ("477400444", "MT Hong Kong Trader", "westbound"),  # Hong Kong
    ("563500555", "MV Singapore Breeze", "eastbound"),  # Singapore
    ("470600666", "MT Emirates Falcon", "westbound"),   # UAE
    ("419700777", "MV Konkan Express", "eastbound"),    # India
    ("256800888", "MT Malta Horizon", "westbound"),     # Malta
    ("209900999", "MV Cyprus Wind", "eastbound"),       # Cyprus
    ("372101112", "MT Panama Crest", "westbound"),      # Panama
]

# Deliberately anomalous vessels
GAP_VESSEL = ("422111222", "MV Al-Noor")           # Iran-flagged, AIS gap
LOITER_VESSEL = ("636222333", "MT Silent Reach")   # Liberia-flagged, loitering
RENDEZVOUS_PAIR = [
    ("537333444", "Nordic Star"),      # Marshall Islands
    ("372444555", "Pacific Trader"),   # Panama
]


def _interp(p0, p1, frac):
    return p0[0] + (p1[0] - p0[0]) * frac, p0[1] + (p1[1] - p0[1]) * frac


def build_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2026-07-22 00:00:00", tz="UTC")

    # Normal transiting traffic: 10 pings each, 40 min apart, steady speed
    for mmsi, name, direction in NORMAL_VESSELS:
        start_frac = rng.uniform(0.0, 0.3)
        src, dst = (CORRIDOR_START, CORRIDOR_END) if direction == "eastbound" else (CORRIDOR_END, CORRIDOR_START)
        speed = rng.uniform(10, 14)
        for i in range(10):
            frac = min(start_frac + i * 0.07, 1.0)
            lat, lon = _interp(src, dst, frac)
            lat += rng.normal(0, 0.005)
            lon += rng.normal(0, 0.005)
            rows.append(dict(mmsi=mmsi, name=name, lat=lat, lon=lon,
                              speed=round(speed + rng.normal(0, 0.4), 1),
                              timestamp=base + pd.Timedelta(minutes=40 * i)))

    # Gap vessel: normal for a while, then vanishes for 7 hours, reappears displaced
    lat, lon = 26.55, 56.30
    for i in range(3):
        rows.append(dict(mmsi=GAP_VESSEL[0], name=GAP_VESSEL[1],
                          lat=lat + i * 0.01, lon=lon + i * 0.015, speed=9.5,
                          timestamp=base + pd.Timedelta(minutes=40 * i)))
    rows.append(dict(mmsi=GAP_VESSEL[0], name=GAP_VESSEL[1],
                      lat=lat + 0.25, lon=lon + 0.35, speed=8.0,
                      timestamp=base + pd.Timedelta(minutes=40 * 3) + pd.Timedelta(hours=7)))

    # Loitering vessel: sits within a small radius at near-zero speed for 5 hours
    lat0, lon0 = 26.30, 56.15
    for i in range(8):
        rows.append(dict(mmsi=LOITER_VESSEL[0], name=LOITER_VESSEL[1],
                          lat=lat0 + rng.normal(0, 0.003), lon=lon0 + rng.normal(0, 0.003),
                          speed=round(rng.uniform(0.1, 0.8), 1),
                          timestamp=base + pd.Timedelta(hours=1) + pd.Timedelta(minutes=40 * i)))

    # Rendezvous pair: converge to near-zero separation, both slow, same window
    (m1, n1), (m2, n2) = RENDEZVOUS_PAIR
    conv_lat, conv_lon = 26.42, 56.08
    rows.append(dict(mmsi=m1, name=n1, lat=conv_lat, lon=conv_lon, speed=1.1,
                      timestamp=base + pd.Timedelta(hours=3)))
    rows.append(dict(mmsi=m2, name=n2, lat=conv_lat + 0.0004, lon=conv_lon + 0.0003, speed=1.3,
                      timestamp=base + pd.Timedelta(hours=3)))

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/hormuz_synthetic.csv")
    args = parser.parse_args()

    df = build_dataset()
    df.to_csv(args.out, index=False)
    print(f"generated {len(df)} position reports across {df['mmsi'].nunique()} vessels")
    print(f"saved to {args.out}")
