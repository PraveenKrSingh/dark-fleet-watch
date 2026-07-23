"""
Builds a realistic-but-synthetic AIS dataset for the Strait of Malacca:
vessels transiting the corridor from the Andaman Sea (northwest) down to
the Singapore Strait (southeast), plus a few deliberately anomalous ones
(AIS gap, loitering, rendezvous) so the pipeline can be tested and demoed
consistently, independent of live feed availability.

Usage:
    python src/generate_malacca_dataset.py --out data/malacca_synthetic.csv
"""

import argparse

import numpy as np
import pandas as pd

# Approximate transit corridor: Andaman Sea entrance (NW) -> Singapore Strait (SE)
CORRIDOR_START = (5.60, 98.20)   # Andaman Sea / northern approach
CORRIDOR_END = (1.25, 103.90)    # Singapore Strait exit

NORMAL_VESSELS = [
    ("563100111", "MV Singapore Breeze", "southeast"),   # Singapore
    ("525200222", "MT Jakarta Trader", "northwest"),     # Indonesia
    ("533300333", "MV Straits Pioneer", "southeast"),    # Malaysia
    ("636400444", "MT Liberty Star", "northwest"),       # Liberia
    ("372500555", "MV Panama Crest", "southeast"),       # Panama
    ("477600666", "MT Hong Kong Trader", "northwest"),   # Hong Kong
    ("413700777", "MV Shanghai Wind", "southeast"),      # China
    ("567800888", "MT Andaman Star", "northwest"),       # Thailand
    ("537900999", "MV Marshall Pride", "southeast"),     # Marshall Islands
    ("209101112", "MT Cyprus Horizon", "northwest"),     # Cyprus
    ("419202223", "MV Konkan Express", "southeast"),     # India
    ("256303334", "MT Malta Wind", "northwest"),         # Malta
]

GAP_VESSEL = ("525111222", "MV Sumatra Ghost")          # Indonesia-flagged, AIS gap
LOITER_VESSEL = ("636222333", "MT Silent Reach")        # Liberia-flagged, loitering
RENDEZVOUS_PAIR = [
    ("537333444", "Nordic Star"),      # Marshall Islands
    ("372444555", "Pacific Trader"),   # Panama
]


def _interp(p0, p1, frac):
    return p0[0] + (p1[0] - p0[0]) * frac, p0[1] + (p1[1] - p0[1]) * frac


def build_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2026-07-23 00:00:00", tz="UTC")

    for mmsi, name, direction in NORMAL_VESSELS:
        start_frac = rng.uniform(0.0, 0.3)
        src, dst = (CORRIDOR_START, CORRIDOR_END) if direction == "southeast" else (CORRIDOR_END, CORRIDOR_START)
        speed = rng.uniform(11, 15)
        for i in range(12):
            frac = min(start_frac + i * 0.06, 1.0)
            lat, lon = _interp(src, dst, frac)
            lat += rng.normal(0, 0.03)
            lon += rng.normal(0, 0.03)
            rows.append(dict(mmsi=mmsi, name=name, lat=lat, lon=lon,
                              speed=round(speed + rng.normal(0, 0.4), 1),
                              timestamp=base + pd.Timedelta(minutes=40 * i)))

    # Gap vessel: normal for a while near the Sumatra coast, then vanishes for 8 hours
    lat, lon = 3.60, 100.60
    for i in range(3):
        rows.append(dict(mmsi=GAP_VESSEL[0], name=GAP_VESSEL[1],
                          lat=lat + i * 0.05, lon=lon + i * 0.04, speed=10.0,
                          timestamp=base + pd.Timedelta(minutes=40 * i)))
    rows.append(dict(mmsi=GAP_VESSEL[0], name=GAP_VESSEL[1],
                      lat=lat + 0.6, lon=lon + 0.5, speed=8.5,
                      timestamp=base + pd.Timedelta(minutes=40 * 3) + pd.Timedelta(hours=8)))

    # Loitering vessel: sits within a small radius near a known piracy-watch area
    lat0, lon0 = 2.90, 101.20
    for i in range(8):
        rows.append(dict(mmsi=LOITER_VESSEL[0], name=LOITER_VESSEL[1],
                          lat=lat0 + rng.normal(0, 0.003), lon=lon0 + rng.normal(0, 0.003),
                          speed=round(rng.uniform(0.1, 0.8), 1),
                          timestamp=base + pd.Timedelta(hours=1) + pd.Timedelta(minutes=40 * i)))

    # Rendezvous pair: converge, both slow, same window
    (m1, n1), (m2, n2) = RENDEZVOUS_PAIR
    conv_lat, conv_lon = 3.20, 100.80
    rows.append(dict(mmsi=m1, name=n1, lat=conv_lat, lon=conv_lon, speed=1.1,
                      timestamp=base + pd.Timedelta(hours=3)))
    rows.append(dict(mmsi=m2, name=n2, lat=conv_lat + 0.0004, lon=conv_lon + 0.0003, speed=1.3,
                      timestamp=base + pd.Timedelta(hours=3)))

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/malacca_synthetic.csv")
    args = parser.parse_args()

    df = build_dataset()
    df.to_csv(args.out, index=False)
    print(f"generated {len(df)} position reports across {df['mmsi'].nunique()} vessels")
    print(f"saved to {args.out}")
