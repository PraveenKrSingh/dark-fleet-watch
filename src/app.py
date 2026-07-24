"""
Sentinel Strait -- Dark Fleet Watch (Strait of Malacca demo).

Layout: hero section (map / flagged vessels / analyst chat) fits in one
viewport on load, each panel independently scrollable. Below the fold:
usage instructions, architecture explainer, and data-source details.

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
from agent_tools import _load_positions, data_source_status, live_traffic_summary
from anomaly_detection import run_all_detectors
from mid_lookup import mid_to_iso

st.set_page_config(page_title="Sentinel Strait -- Dark Fleet Watch", layout="wide")

FLAG_ICONS_CSS = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/flag-icons@7/css/flag-icons.min.css">'
st.markdown(FLAG_ICONS_CSS, unsafe_allow_html=True)

# --- Defence Intelligence theme ---------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0b1220; color: #dfe4ee; }
    .block-container { padding-top: 3rem; padding-bottom: 0.5rem; max-width: 100%; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #101a30; border: 1px solid #2a3a55; border-radius: 6px; padding: 0.5rem;
    }
    h1, h2, h3 { color: #e8ecf4 !important; letter-spacing: 0.5px; }
    .sentinel-title { font-family: Georgia, 'Times New Roman', serif; font-weight: 700;
        font-size: 2.1rem; color: #f0d896; letter-spacing: 1px; margin-bottom: 0; }
    .sentinel-subtitle { font-family: 'Trebuchet MS', sans-serif; font-size: 0.8rem;
        color: #8ea3c7; letter-spacing: 2px; text-transform: uppercase; margin-top: -6px; }
    .section-label { font-family: 'Trebuchet MS', sans-serif; font-size: 0.75rem;
        color: #c9a24d; letter-spacing: 2px; text-transform: uppercase; font-weight: 600; }
    .analyst-tagline { color: #8ea3c7; font-size: 0.82rem; margin-top: -4px; margin-bottom: 6px; }
    .stCaption, [data-testid="stCaptionContainer"] { color: #8ea3c7 !important; }
    .flag-card { border: 1px solid #2a3a55; background-color: #131e35; border-radius: 6px;
        padding: 8px; margin-bottom: 6px; }
    .flag-card .vname { color: #e8ecf4; font-weight: 600; }
    .flag-card .vreason { color: #8ea3c7; font-size: 0.85em; }
    .flag-card.gap { border-left: 3px solid #b5322c; }
    .flag-card.loiter { border-left: 3px solid #c9a24d; }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS_COLOR = {"ais_gap": "#e0453c", "loitering": "#e0a83c", "rendezvous": "#e0a83c", None: "#7d8aa0"}
PANEL_HEIGHT = 560

# --- Header ------------------------------------------------------------------
_src = data_source_status()
_live = live_traffic_summary()

header_col1, header_col2 = st.columns([1.4, 3])
with header_col1:
    st.markdown('<div class="sentinel-title">SENTINEL STRAIT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sentinel-subtitle">Dark Fleet Watch &middot; Strait of Malacca</div>',
                unsafe_allow_html=True)
with header_col2:
    if _src["source"] in ("live", "live_thin"):
        status_line = f"Live AISStream data -- {_src['span_hours']}h span, {_src['vessels']} vessels."
    else:
        status_line = "Live capture not yet available."
    st.caption(f"{status_line} Not operational intelligence &middot; open-source demonstration only.")

if _src["source"] in ("live_thin",):
    st.caption(_src["reason"])

if _src["source"] == "not_ready":
    st.error(
        f"{_src['reason']}. Run this in a terminal, then reload this page once it's collecting data:\n\n"
        "`python src/fetch_malacca_ais.py --minutes 240 --out data/malacca_live.csv`"
    )
    st.stop()

# --- Timeline slider -----------------------------------------------------------
LOOKBACK_OPTIONS = {
    "Last 6 Hours": 6, "Last 12 Hours": 12, "Last 1 Day": 24,
    "Last 3 Days": 72, "Last 7 Days": 168, "All Available Data": None,
}
lookback_label = st.select_slider(
    "Data timeline", options=list(LOOKBACK_OPTIONS.keys()), value="All Available Data",
    label_visibility="collapsed",
)
lookback_hours = LOOKBACK_OPTIONS[lookback_label]


@st.cache_data(ttl=60)
def load_map_data(hours):
    df = _load_positions()
    if hours is not None and not df.empty:
        cutoff = df["timestamp"].max() - pd.Timedelta(hours=hours)
        df = df[df["timestamp"] >= cutoff]
    latest = df.sort_values("timestamp").groupby("mmsi").tail(1) if not df.empty else df
    flagged_df = run_all_detectors(df) if not df.empty else df
    flagged = []
    flag_by_mmsi = {}
    for _, row in flagged_df.iterrows():
        mmsi_key = str(row["mmsi"]).split("+")[0]
        entry = {"mmsi": row["mmsi"], "name": row["name"], "flag_type": row["flag_type"],
                  "flag_state": mid_to_iso(mmsi_key), "reason": row["reason"]}
        flagged.append(entry)
        flag_by_mmsi[mmsi_key] = entry
    return latest, flagged, flag_by_mmsi


latest, flagged, flag_by_mmsi = load_map_data(lookback_hours)

# --- Three columns: map / flagged vessels / analyst chat ----------------------
col_map, col_vessels, col_chat = st.columns([2.2, 1.1, 1.3])

with col_map:
    st.markdown(f'<span class="section-label">TACTICAL PLOT</span> &middot; '
                f'{len(latest)} vessels tracked &middot; {len(flagged)} flagged ({lookback_label.lower()})',
                unsafe_allow_html=True)
    m = folium.Map(location=[3.0, 101.5], zoom_start=7, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, GEBCO, NOAA, National Geographic, DeLorme, HERE, Garmin",
        name="Ocean Physical",
    ).add_to(m)

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
            fill_opacity=0.9,
            tooltip=folium.Tooltip(label_html),
            popup=folium.Popup(label_html, max_width=250),
        ).add_to(m)

    st_folium(m, width=None, height=PANEL_HEIGHT, returned_objects=[])

with col_vessels:
    st.markdown('<span class="section-label">FLAGGED VESSELS</span>', unsafe_allow_html=True)
    with st.container(height=PANEL_HEIGHT):
        if not flagged:
            st.write("No vessels currently flagged in this window.")
        for f in flagged:
            iso = f["flag_state"]
            css_class = "gap" if f["flag_type"] == "ais_gap" else "loiter"
            st.markdown(
                f'<div class="flag-card {css_class}">'
                f'<span class="fi fi-{iso}"></span> <span class="vname">{f["name"]}</span><br>'
                f'<span class="vreason">{f["flag_type"]} &middot; {f["reason"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

with col_chat:
    st.markdown('<span class="section-label">INTERACTIONS WITH THE NAVAL AGENTIC ANALYST</span>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="analyst-tagline">Ask about any vessel\'s flag, history, or anomaly -- '
        'grounded, tool-verified answers, not guesses.</div>',
        unsafe_allow_html=True,
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.container(height=PANEL_HEIGHT - 40):
        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(content)

question = st.chat_input("Ask the Naval Agentic Analyst about a ship, e.g. why is MV Sumatra Ghost flagged?")
if question:
    st.session_state.chat_history.append(("user", question))
    with st.spinner("checking..."):
        answer = ask(question)
    st.session_state.chat_history.append(("assistant", answer))
    st.rerun()

# ==============================================================================
# Below the fold: usage guide, architecture explainer, data sources
# ==============================================================================
st.divider()

st.markdown('<span class="section-label">HOW TO USE SENTINEL STRAIT</span>', unsafe_allow_html=True)
st.markdown(
    """
    1. **Scan the tactical plot** -- red markers indicate an AIS gap (vessel went dark), amber
       indicates loitering or a possible rendezvous, gray is normal traffic. Hover or click any
       marker for details.
    2. **Check the flagged vessels panel** for a plain-language reason behind every flag --
       every flag is generated by a deterministic rule, never a black-box score.
    3. **Ask the Naval Agentic Analyst anything** -- e.g. *"why is MV Sumatra Ghost flagged?"*,
       *"which flagged vessel has the longest AIS gap?"*, or *"tell me about MT Silent Reach."*
       The analyst calls the same underlying tools shown above to answer, live.
    4. **Adjust the timeline slider** to widen or narrow the observation window.
    """
)

st.caption("Demo walkthrough -- add a short screen recording (GIF or MP4) here to show the "
           "map, flagging, and chat in action. Use st.image('demo.gif') or st.video('demo.mp4').")

st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown('<span class="section-label">WHAT MAKES THIS AGENTIC</span>', unsafe_allow_html=True)

arch_col1, arch_col2, arch_col3 = st.columns(3)
with arch_col1:
    st.markdown("**Deterministic detection layer**")
    st.markdown(
        "Three rule-based detectors run over raw AIS position reports with pandas -- "
        "no LLM involved: AIS gaps (silence beyond a threshold), loitering (low speed, "
        "small radius, sustained duration), and rendezvous (two vessels converging and "
        "both slowing down). Every flag carries an explicit, auditable reason."
    )
with arch_col2:
    st.markdown("**Agentic reasoning layer**")
    st.markdown(
        "An LLM (Groq-hosted Llama 3.3 70B) sits above the detectors with four callable "
        "tools -- `get_flagged_vessels`, `get_vessel_history`, `explain_flag`, and "
        "`find_vessel_by_name`. The model decides which tool(s) to call, in what order, "
        "chaining multiple calls when a question needs it, before answering."
    )
with arch_col3:
    st.markdown("**Data sources**")
    st.markdown(
        "Live AIS positions via [AISStream.io](https://aisstream.io) (free, community-run "
        "terrestrial receivers), with a synthetic dataset as fallback when live coverage is "
        "thin. Flag states resolved from each vessel's MMSI via the MID country-code lookup."
    )

st.caption(
    "Sentinel Strait is a personal portfolio project built on public, open-source AIS data. "
    "It is a demonstration of agentic AI architecture, not an operational intelligence system."
)
