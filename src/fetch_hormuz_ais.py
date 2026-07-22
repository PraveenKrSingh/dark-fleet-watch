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
import time
from datetime import datetime, timezone

import pandas as pd
import websocket
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("AISSTREAM_API_KEY")

# Bounding box: [[lat_min, lon_min], [lat_max, lon_max]]
# Covers the strait itself plus the approach waters on both sides.
HORMUZ_BBOX = [[25.5, 55.0], [27.5, 57.5]]


def collect(minutes: float, out_path: str):
    if not API_KEY:
        raise SystemExit("AISSTREAM_API_KEY not found -- check your .env file")

    rows = []
    had_error = [False]
    deadline = datetime.now(timezone.utc).timestamp() + minutes * 60

    def on_open(ws):
        print("connected -- subscribing to Hormuz bounding box")
        ws.send(json.dumps({
            "APIKey": API_KEY,
            "BoundingBoxes": [HORMUZ_BBOX],
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
        ws.run_forever()
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
    df.to_csv(out_path, index=False)
    print(f"\nsaved {len(df)} position reports to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=5.0, help="how long to listen")
    parser.add_argument("--out", type=str, default="data/hormuz_live.csv")
    args = parser.parse_args()
    collect(args.minutes, args.out)
