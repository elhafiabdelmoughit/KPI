# KPI.py
# Streamlit PV KPI calculator (Basic / Intermediate / Advanced)
# Run:  pip install -U streamlit pandas numpy
# Then: streamlit run KPI.py

import streamlit as st
from basic import page_basic
from intermediate import page_intermediate
from advanced import page_advanced

st.set_page_config(
    page_title="PV KPI Calculator (PR at STC)",
    page_icon="⚡",
    layout="wide",
)

# ---- Defaults (used as initial values in sidebar) ----
DEFAULTS = {
    "ref_irr": 1000.0,   # W/m²
    "gamma": -0.0035,    # 1/°C
    "ref_temp": 25.0,    # °C
    "king_a": -3.56,     # King coefficients (used in Intermediate)
    "king_b": -0.075,
    # NEW for Potential Production
    "proxy_yield_h": 0.90,       # Proxy Specific Yield (h)
    "ac_max_project_kw": 1800.0, # AC Max Power (project) [kW]
    "pr_nominal": 0.80,          # PR nominal for Met-based
}

if "_reset_token" not in st.session_state:
    st.session_state["_reset_token"] = 0

# -----------------------------
# Sidebar (Static Data + Expected Parameters)
# -----------------------------
with st.sidebar:
    st.title("Parameters")

    # --- Static Data ---
    st.subheader("Static Data")

    ref_irr = st.number_input(
        "Ref. Irradiance (W/m²)",
        min_value=0.0,
        value=float(DEFAULTS["ref_irr"]),
        step=1.0,
        key=f"ref_irr_{st.session_state['_reset_token']}",
    )
    ref_temp = st.number_input(
        "Ref. Temp (°C)",
        value=float(DEFAULTS["ref_temp"]),
        step=0.1,
        key=f"ref_temp_{st.session_state['_reset_token']}",
    )
    gamma = st.number_input(
        "Gamma (1/°C)",
        value=float(DEFAULTS["gamma"]),
        step=0.0001,
        format="%.5f",
        key=f"gamma_{st.session_state['_reset_token']}",
    )

    # REMOVED: dc_cap → now handled inside Basic page

    # NEW: Proxy yield & AC Max & PR_nominal (static project-wide)
    st.markdown("---")
    proxy_yield_h = st.number_input(
        "Proxy Specific Yield (h)",
        min_value=0.0,
        value=float(DEFAULTS["proxy_yield_h"]),
        step=0.01,
        key=f"proxy_yield_{st.session_state['_reset_token']}",
        help="Reference specific yield used in Proxy-based potential."
    )
    ac_max_project_kw = st.number_input(
        "AC Max Power (project) [kW]",
        min_value=0.0,
        value=float(DEFAULTS["ac_max_project_kw"]),
        step=10.0,
        key=f"acmax_{st.session_state['_reset_token']}",
        help="POI AC limit; converted to kWh per 10-min window as AC/6."
    )
    pr_nominal = st.number_input(
        "PR_nominal (0–1)",
        min_value=0.0,
        max_value=1.2,
        value=float(DEFAULTS["pr_nominal"]),
        step=0.01,
        key=f"prnom_{st.session_state['_reset_token']}",
        help="Nominal PR used in Met-based potential."
    )

    # store in session_state so all tabs can read
    st.session_state["ref_irr"] = ref_irr
    st.session_state["ref_temp"] = ref_temp
    st.session_state["gamma"] = gamma
    st.session_state["proxy_yield_h"] = proxy_yield_h
    st.session_state["ac_max_project_kw"] = ac_max_project_kw
    st.session_state["pr_nominal"] = pr_nominal

    st.markdown("---")

    # --- King coefficients (used in Intermediate) ---
    st.subheader("King Coefficients")
    col1, col2 = st.columns(2)
    with col1:
        king_a = st.number_input(
            "King a",
            value=float(st.session_state.get("king_a", DEFAULTS["king_a"])),
            step=0.01, format="%.3f",
            key=f"king_a_{st.session_state['_reset_token']}",
        )
    with col2:
        king_b = st.number_input(
            "King b",
            value=float(st.session_state.get("king_b", DEFAULTS["king_b"])),
            step=0.005, format="%.3f",
            key=f"king_b_{st.session_state['_reset_token']}",
        )
    st.session_state["king_a"] = king_a
    st.session_state["king_b"] = king_b

# -----------------------------
# Navigation
# -----------------------------
st.markdown("## PV KPI Workspace")
tab_basic, tab_intermediate, tab_advanced = st.tabs(["Basic", "Intermediate", "Advanced"])

with tab_basic:
    # dc_cap removed from sidebar → pass dummy (0)
    page_basic(ref_irr, ref_temp, gamma, 0, e_pot_mode="PR measured", pr_target=0.90)

with tab_intermediate:
    page_intermediate()

with tab_advanced:
    page_advanced()
