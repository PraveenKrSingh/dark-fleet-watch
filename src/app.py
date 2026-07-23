"""
Dark Fleet Watch -- Streamlit UI.

Map: folium, one marker per vessel (latest known position), colored by
anomaly status, with flag-icons in the tooltip/popup so status is visible
without hovering or clicking.

Chat: reuses the exact same agent.ask() function we already built and
tested on the command line -- the UI is just a thin layer on top of it.

Run:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from agent import ask
from agent_tools import get_flagged_vessels, _load_positions, data_source_status, live_traffic_summary
from mid_lookup import mid_to_iso

st.set_page_config(page_title="Dark Fleet Watch", layout="wide")

FLAG_ICONS_CSS = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/flag-icons@7/css/flag-icons.min.css">'
st.markdown(FLAG_ICONS_CSS, unsafe_allow_html=True)

STATUS_COLOR = {"ais_gap": "red", "loitering": "orange", "rendezvous": "orange", None: "gray"}

st.title("Dark Fleet Watch")

_src = data_source_status()
if _src["source"] == "live":
    st.caption(
        f"Live AISStream data -- {_src['span_hours']}h span, {_src['vessels']} vessels. "
        "For demonstration purposes only -- not operational intelligence. Scope: Strait of Malacca."
    )
else:
    st.caption(
        f"Synthetic demo data ({_src['reason']}). "
        "For demonstration purposes only -- not operational intelligence. Scope: Strait of Malacca."
    )

_live = live_traffic_summary()
if _live["captured"] and _live.get("positions", 0) is not None:
    with st.container():
        if _live["vessels"] <= 3:
            st.warning(
                f"**Live AIS monitor:** only **{_live['vessels']}** vessel(s) / **{_live['positions']}** "
                f"position reports observed in the Malacca box over {_live['span_hours']}h of live listening. "
                "Malacca is normally one of the busiest straits on Earth, so a count this low is unexpected -- "
                "worth checking the connection or bounding box rather than treating this as a real finding."
            )
        else:
            st.success(
                f"**Live AIS monitor:** {_live['vessels']} vessels, {_live['positions']} position reports "
                f"observed over {_live['span_hours']}h."
            )


@st.cache_data(ttl=60)
def load_map_data():
    df = _load_positions()
    latest = df.sort_values("timestamp").groupby("mmsi").tail(1)
    flagged = get_flagged_vessels()
    flag_by_mmsi = {}
    for f in flagged:
        mmsi_key = str(f["mmsi"]).split("+")[0]
        flag_by_mmsi[mmsi_key] = f
    return latest, flagged, flag_by_mmsi


latest, flagged, flag_by_mmsi = load_map_data()

col_map, col_side = st.columns([1.6, 1])

with col_map:
    m = folium.Map(location=[3.0, 101.5], zoom_start=7, tiles="cartodbpositron")

    for _, row in latest.iterrows():
        mmsi_str = str(row["mmsi"])
        iso = mid_to_iso(mmsi_str)
        status = flag_by_mmsi.get(mmsi_str)
        color = STATUS_COLOR["ais_gap"] if status and status["flag_type"] == "ais_gap" \
            else STATUS_COLOR["loitering"] if status else STATUS_COLOR[None]

        label_html = (
            f'<span class="fi fi-{iso}"></span> <b>{row["name"]}</b><br>'
            f'MMSI: {mmsi_str}<br>'
            + (f'<span style="color:{color}">{status["flag_type"]}: {status["reason"]}</span>'
               if status else '<span style="color:gray">no anomalies</span>')
        )

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9 if status else 6,
            color=color,
            fill=True,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(label_html),
            popup=folium.Popup(label_html, max_width=250),
        ).add_to(m)

    st_folium(m, width=None, height=480, returned_objects=[])

    st.markdown(
        f"**{len(latest)}** vessels tracked &middot; **{len(flagged)}** currently flagged"
    )

with col_side:
    st.subheader("Flagged vessels")
    if not flagged:
        st.write("No vessels currently flagged.")
    for f in flagged:
        iso = f["flag_state"]
        st.markdown(
            f'<div style="border:1px solid #444; border-radius:8px; padding:8px; margin-bottom:6px;">'
            f'<span class="fi fi-{iso}"></span> <b>{f["name"]}</b><br>'
            f'<span style="font-size:0.85em; color:#aaa;">{f["flag_type"]} &middot; {f["reason"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()
st.subheader("Ask the analyst")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

question = st.chat_input("e.g. why is MV Al-Noor flagged?")
if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("checking..."):
            answer = ask(question)
        st.write(answer)
    st.session_state.chat_history.append(("assistant", answer))
