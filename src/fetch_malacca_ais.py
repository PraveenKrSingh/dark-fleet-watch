"""
Connects to AISStream.io, subscribes to a bounding box around the Strait of
Hormuz, and collects incoming position reports for a fixed duration into a
pandas dataframe shaped for our anomaly detectors (mmsi, name, lat, lon,
speed, timestamp).

Requires AISSTREAM_API_KEY in a local .env file (see .env.example).
Usage:
    python src/fetch_hormuz_ais.py --minutes 5 --out data/hormuz_live.csv
"""

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import websocket
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("AISSTREAM_API_KEY")

# Bounding box: [[lat_min, lon_min], [lat_max, lon_max]]
# Primary target: the Strait of Malacca, from the Andaman Sea approach down
# through the Singapore Strait -- a genuinely busy, verified-live corridor.
MALACCA_BBOX = [[1.0, 98.0], [6.5, 104.5]]

# Kept as an optional comparison region -- this is where we discovered
# AISStream has a real coverage gap, likely due to sparse volunteer AIS
# receiver infrastructure on the Iranian coastline.
HORMUZ_BBOX = [[24.0, 54.0], [28.0, 59.0]]


def collect(minutes: float, out_path: str, bbox=None):
    if not API_KEY:
        raise SystemExit("AISSTREAM_API_KEY not found -- check your .env file")
    bbox = bbox or MALACCA_BBOX

    rows = []
    had_error = [False]
    deadline = datetime.now(timezone.utc).timestamp() + minutes * 60

    def on_open(ws):
        print(f"connected -- subscribing to bounding box {bbox}")
        ws.send(json.dumps({
            "APIKey": API_KEY,
            "BoundingBoxes": [bbox],
            "FilterMessageTypes": ["PositionReport"],
        }))

    def on_message(ws, message):
        msg = json.loads(message)
        if msg.get("MessageType") != "PositionReport":
            return
        report = msg["Message"]["PositionReport"]
        meta = msg.get("MetaData", {})
        rows.append({
            "mmsi": meta.get("MMSI") or report.get("UserID"),
            "name": (meta.get("ShipName") or "").strip() or f"MMSI {meta.get('MMSI')}",
            "lat": report.get("Latitude"),
            "lon": report.get("Longitude"),
            "speed": report.get("Sog", 0.0),
            "timestamp": pd.Timestamp.now(tz="UTC"),
        })
        print(f"[{len(rows)}] {rows[-1]['name']} @ ({rows[-1]['lat']:.3f}, {rows[-1]['lon']:.3f})")

        if datetime.now(timezone.utc).timestamp() >= deadline:
            ws.close()

    def on_error(ws, error):
        had_error[0] = True
        print("error:", error)

    def on_close(ws, *_):
        print("connection closed")

    max_retries = 4
    for attempt in range(1, max_retries + 1):
        rows.clear()
        had_error[0] = False
        ws = websocket.WebSocketApp(
            "wss://stream.aisstream.io/v0/stream",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        timeout_timer = threading.Timer(minutes * 60, ws.close)
        timeout_timer.daemon = True
        timeout_timer.start()
        ws.run_forever()
        timeout_timer.cancel()
        if not had_error[0]:
            # clean connection -- whether or not vessels were seen, this was a real read
            break
        if attempt < max_retries:
            wait = 15 * attempt
            print(f"attempt {attempt} failed to connect (server overload) -- retrying in {wait}s")
            time.sleep(wait)
        else:
            print("all retries exhausted -- AISStream may be overloaded right now, try again later")

    df = pd.DataFrame(rows)

    out_file = Path(out_path)
    if out_file.exists() and not df.empty:
        existing = pd.read_csv(out_file, parse_dates=["timestamp"])
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["mmsi", "timestamp"]).sort_values("timestamp")
        combined.to_csv(out_path, index=False)
        print(f"\nappended {len(df)} new reports -- {len(combined)} total across "
              f"{combined['mmsi'].nunique()} vessels now in {out_path}")
    elif not df.empty:
        df.to_csv(out_path, index=False)
        print(f"\nsaved {len(df)} position reports to {out_path}")
    else:
        print(f"\nno data captured this run -- {out_path} unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=5.0, help="how long to listen")
    parser.add_argument("--out", type=str, default="data/malacca_live.csv")
    parser.add_argument("--region", type=str, default="malacca", choices=["malacca", "hormuz"],
                         help="hormuz is kept as an optional comparison region -- known to have "
                              "sparse AISStream coverage, useful context but not the main target")
    args = parser.parse_args()
    chosen_bbox = HORMUZ_BBOX if args.region == "hormuz" else MALACCA_BBOX
    collect(args.minutes, args.out, bbox=chosen_bbox)
