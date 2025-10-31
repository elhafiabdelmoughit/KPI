# intermediate.py
import math
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Global constants
# ---------------------------
N_INV = 3               # number of inverters on site
DT_HOURS = 1.0 / 6.0    # 10 minutes in hours

# ---------------------------
# Column header tooltips
# ---------------------------
HOVER = {
    # Common / time
    "Time": "Time label of the 10-min interval (hh:mm).",

    # MET / SAT inputs & flags
    "MET_POA_Wm2": "Plane-of-array irradiance from MET sensor (W/m²).",
    "MET_POA_available": "True if MET POA is available & finite at this row.",
    "SAT_POA_Wm2": "Fallback POA from satellite or secondary source (W/m²).",

    "MET_Amb_C": "Ambient air temperature from MET (°C).",
    "MET_amb_available": "True if MET ambient temperature is available.",
    "MET_Back_Module_C": "Back-of-module temperature (°C).",
    "MET_back_available": "True if back-of-module temperature is available.",
    "MET_WindSpeed": "Wind speed from MET (m/s).",
    "Wind_available": "True if wind speed is available.",

    "SAT_T_mod_C": "Module temperature fallback from SAT/estimate (°C).",

    # Energy & status
    "AC_kWh": "Measured AC energy per 10-min (kWh).",
    "Availability": "Reference availability factor applied to losses (for more details move to advance).",
    "Event_Weight": "Event weighting factor applied to losses (for more details move to advance).",
    "FullCap": "If True, this row is eligible for Median(Y_ac) proxy.",

    # Resolved inputs (per-inverter)
    "POA_used (W/m²)": "POA actually used for KPI (MET if available, else SAT).",
    "POA_Source": "Which POA source was used (MET_POA or SAT_POA).",
    "T_mod_used (°C)": "Module temperature used (Back / King(MET[,WS]) / SAT).",
    "T_mod_Source": "Which source/method produced T_mod_used.",
    "AC_KWh": "AC energy per 10-min (kWh).",  # kept to mirror existing column_config key

    # KPI (per-inverter)
    "Reference Yield (h at STC)": "Y_ref at STC: POA / (RefIrr*(1+γ*(Tmod−Tref))) * (1/6).",
    "Specific Yield (h = kWh/kWp)": "Y_ac = E_ac / DC_inv (per 10-min).",
    "Performance Ratio (% at STC)": "PR = (Y_ac / Y_ref) × 100.",
    "Site Median Specific Yield (h)": "Row-wise median of other inverters’ Y_ac (FullCap + threshold).",
    "Proxy — Reference Potential (kWh)": "min( Median(Y_ac)*DC_inv , AC_max_project/N_inv * (1/6) ).",
    "Lost Energy (kWh)": "max(0, (1−Availability) * gap * Event_Weight), gap=max(0, E_potential−E_ac).",

    # Totals (per-inverter)
    "Σ E_ac (kWh)": "Sum of AC energy over the day (kWh).",
    "Σ Potential (kWh)": "Sum of proxy potential over the day (kWh).",
    "Σ Lost (kWh)": "Sum of lost energy over the day (kWh).",
    "Reference Yield — day (h)": "Sum of Y_ref over the day (h).",
    "Specific Yield — day (h)": "ΣE_ac / DC_inv (h).",
    "Performance Ratio — day (%)": "PR_day = (Y_ac_day / Y_ref_day) × 100.",
    "All PBA (%)": "Performance Before Availability globale.",

    # Global summary (per-inverter)
    "Gross inverter production (kWh)": "ΣE_ac for this inverter.",
    "All losses (kWh)": "ΣLost for this inverter.",
    "All PR STC (%)": "PR_day × 100 for this inverter.",

    # Site resolved inputs
    "POA_site (W/m², DC-weighted)": "DC-weighted POA across inverters.",
    "T_mod_site (°C, DC-weighted)": "DC-weighted T_mod across inverters.",
    "DC_total (kWp)": "Sum of inverter DC capacities.",
    "AC_revenue_kWh": "AC energy from revenue meter per 10-min (kWh).",
    "AC_revenue_available": "Availability flag for revenue meter.",
    "AC_check_kWh": "AC energy from check meter per 10-min (kWh).",
    "AC_check_available": "Availability flag for check meter.",
    "Σ Inverter AC (kWh)": "Sum of per-inverter AC (kWh).",
    "AC_site_resolved (kWh)": "Resolved AC used for site KPI (fallback chain).",
    "AC_source": "Which AC source was selected (revenue/check/sum).",

    # KPI (site)
    "Reference Yield (h)": "Site Y_ref using POA_site & T_mod_site.",
    "Specific Yield (h)": "Site Y_ac = E_ac_site_resolved / DC_total.",
    "Performance Ratio (%)": "Site PR = (Y_ac_site / Y_ref_site) × 100.",
    "Median(Y_ac) (h)": "Site row-wise median of per-inverter Y_ac (FullCap + threshold).",

    # Totals (site)
    "Performance Ratio — day (%)": "Site PR_day in percent.",
    "Reference POI Production (kWh)": "daily AC Revenue Meter (kWh).",
}

# ---------------------------
# About this section
# ---------------------------
def _about():
    with st.expander("ℹ️ About this section (INTERMEDIATE)", expanded=False):
        st.markdown("""
### Overview
This section analyzes **inverter-level performance** at a **10-minute resolution** and consolidates the results at the **site level**.  
It allows precise tracking of energy production, yields, and losses across multiple inverters operating under the same conditions.

### Page structure
The INTERMEDIATE section includes **four pages**:
1. **Inverter 1** — Detailed 10-min performance of the first inverter.  
2. **Inverter 2** — Same structure, showing KPIs and losses for the second inverter.  
3. **Inverter 3** — Performance of the third inverter, synchronized with the others.  
4. **Site** — Combines data from all inverters to provide site-wide KPIs and daily totals.

Each inverter page follows the same layout, enabling easy comparison between units.  
The **Site** page aggregates inverter outputs to produce overall site performance indicators.

### Main metrics
- **Reference Yield (h)** — Expected energy yield at reference conditions.  
- **Specific Yield (h)** — Energy produced per kWp of DC capacity.  
- **Performance Ratio (%)** — Efficiency ratio between actual and expected yields.  
- **Proxy Reference Potential (kWh)** — Estimated potential production using median-based peer comparison.  
- **Lost Energy (kWh)** — Energy unavailable due to downtime or reduced capacity.  
- **All PBA (%)** — Overall performance before availability adjustments.

### Calculation scope
- Each inverter is evaluated separately using its DC capacity, availability, and event weighting.
- Median-based filtering ensures that only valid inverters contribute to the reference potential.
- The site-level aggregation merges synchronized 10-minute data from all inverters.

### Displayed elements
1. **Resolved Inputs** — Shows which sensors were used and their availability status.  
2. **KPI Data tables** — 10-min interval results for all performance metrics.  
3. **Numeric details** — Expanders with numeric substitution for each KPI.  
4. **Daily Totals & Aggregated KPIs** — Summarized daily performance indicators.  
5. **Global Daily Energy Summary** — Consolidated overview of production, potential, and losses.

### Purpose
The INTERMEDIATE section provides a detailed yet accessible view of system performance.  
It helps you:
- Compare all inverters side by side with synchronized data.  
- Verify data consistency, sensor quality, and availability.  
- Understand the daily and global energy balance at both inverter and site levels.
        """)

# ---------------------------
# Session site params (with safe defaults)
# ---------------------------
def _get_site_params():
    ref_irr   = float(st.session_state.get("ref_irr", 1000.0))
    ref_temp  = float(st.session_state.get("ref_temp", 25.0))
    gamma     = float(st.session_state.get("gamma", -0.0035))
    a         = float(st.session_state.get("king_a", -3.56))
    b         = float(st.session_state.get("king_b", -0.075))
    ac_max_prj= float(st.session_state.get("ac_max_project_kw", 1800.0))
    pr_nom    = float(st.session_state.get("pr_nominal", 0.80))
    return ref_irr, ref_temp, gamma, a, b, ac_max_prj, pr_nom

# ---------------------------
# Small helpers
# ---------------------------
def _fmt(x, n=6):
    try:
        x = float(x)
        return f"{np.round(x, n)}" if math.isfinite(x) else "NaN"
    except Exception:
        return "NaN"

def _latex(template: str, **vals):
    s = template
    for k, v in vals.items():
        s = s.replace(f"<{k}>", str(v))
    s = s.replace("[[", "{").replace("]]", "}")
    st.latex(s)

def _get_dc_scalar(sfx: str, default: float = 700.0) -> float:
    if f"dc_scalar_{sfx}" in st.session_state:
        return float(st.session_state[f"dc_scalar_{sfx}"])
    return float(st.session_state.get(f"dc_{sfx}_kwp", default))

# ---------------------------
# Auto-sync flags with values
# ---------------------------
def _col(df: pd.DataFrame, name: str, default=None):
    if name in df.columns:
        return df[name]
    n = len(df) if len(df.columns) else 0
    if isinstance(default, bool):
        return pd.Series([default]*n, index=df.index)
    return pd.Series([default]*n, index=df.index)

def _auto_sync_availability_flags(ed: pd.DataFrame) -> pd.DataFrame:
    ed = ed.copy()

    for cname in ["MET_POA_available","MET_amb_available","MET_back_available","Wind_available","FullCap"]:
        if cname not in ed.columns:
            ed[cname] = False
        ed[cname] = ed[cname].fillna(False).astype(bool)

    if "Availability" not in ed.columns:
        ed["Availability"] = 1.0
    if "Event_Weight" not in ed.columns:
        ed["Event_Weight"] = 1.0

    for cname in [
        "MET_POA_Wm2","SAT_POA_Wm2","MET_Amb_C","MET_Back_Module_C",
        "MET_WindSpeed","SAT_T_mod_C","AC_kWh","Availability","Event_Weight"
    ]:
        if cname not in ed.columns:
            ed[cname] = np.nan
        ed[cname] = pd.to_numeric(ed[cname], errors="coerce").astype(float)

    ed["Availability"] = ed["Availability"].clip(0.0, 1.0)
    ed["Event_Weight"] = ed["Event_Weight"].clip(0.0, 1.0)

    poa = ed["MET_POA_Wm2"]
    ed.loc[~np.isfinite(poa), "MET_POA_available"] = False

    amb = ed["MET_Amb_C"]
    ed.loc[~np.isfinite(amb), "MET_amb_available"] = False

    back = ed["MET_Back_Module_C"]
    ed.loc[~np.isfinite(back), "MET_back_available"] = False

    ws = ed["MET_WindSpeed"]
    ed.loc[~np.isfinite(ws), "Wind_available"] = False

    return ed

# ---------------------------
# Temperature solver
# ---------------------------
def _solve_tmod(df: pd.DataFrame, a: float, b: float):
    back_av = _col(df, "MET_back_available", False).fillna(False).astype(bool).to_numpy()
    amb_av  = _col(df, "MET_amb_available", False).fillna(False).astype(bool).to_numpy()
    t_back  = pd.to_numeric(_col(df, "MET_Back_Module_C", np.nan), errors="coerce").astype(float).to_numpy()
    t_amb   = pd.to_numeric(_col(df, "MET_Amb_C", np.nan), errors="coerce").astype(float).to_numpy()

    poa_met = pd.to_numeric(_col(df, "MET_POA_Wm2", np.nan), errors="coerce").astype(float).to_numpy()
    met_poa_av = _col(df, "MET_POA_available", False).fillna(False).astype(bool).to_numpy()
    poa_ok  = met_poa_av & np.isfinite(poa_met)

    ws      = pd.to_numeric(_col(df, "MET_WindSpeed", np.nan), errors="coerce").astype(float).to_numpy()
    wind_av = _col(df, "Wind_available", False).fillna(False).astype(bool).to_numpy()
    ws_ok   = np.isfinite(ws)

    t_sat   = pd.to_numeric(_col(df, "SAT_T_mod_C", np.nan), errors="coerce").astype(float).to_numpy()

    n = len(df)
    t_mod = np.full(n, np.nan, dtype=float)
    src   = np.array([""]*n, dtype=object)

    idx1 = back_av
    if np.any(idx1):
        t_mod[idx1] = t_back[idx1]
        src[idx1]   = "MET_back"

    idx2 = (~back_av) & poa_ok & amb_av
    if np.any(idx2):
        t_k = t_amb + poa_met * np.exp(a)
        use_ws = idx2 & wind_av & ws_ok
        if np.any(use_ws):
            t_k[use_ws] = t_amb[use_ws] + poa_met[use_ws] * np.exp(a + b*ws[use_ws])
        t_mod[idx2] = t_k[idx2]
        src[idx2]   = np.where(use_ws[idx2], "King(MET,WS)", "King(MET)")

    idx3 = (~back_av) & (~(poa_ok & amb_av))
    if np.any(idx3):
        t_mod[idx3] = t_sat[idx3]
        src[idx3]   = "SAT"

    return t_mod, src

# ---------------------------
# Yields & PR (scalar DC per inverter)
# ---------------------------
def _yields_pr_scalar_dc(poa_wm2, t_mod_c, ref_irr, ref_temp, gamma, e_ac_kwh, dc_cap_kwp_scalar):
    temp_factor = 1.0 + gamma * (t_mod_c - ref_temp)
    denom = ref_irr * temp_factor
    y_ref_ratio = np.where(np.isclose(denom, 0.0), np.nan, poa_wm2 / denom)
    y_ref_h = y_ref_ratio * DT_HOURS
    dc = float(dc_cap_kwp_scalar)
    y_ac_h = np.where(dc > 0, e_ac_kwh / dc, np.nan)
    pr = np.where((y_ref_h == 0) | np.isnan(y_ref_h), np.nan, y_ac_h / y_ref_h)
    return y_ref_h, y_ac_h, pr

# ---------------------------
# Real-time refresh helpers
# ---------------------------
def _recompute_for_one_inverter_from_src(inv_name: str, key_suffix: str, inv_dc_kwp: float = None):
    src_key = f"src_{key_suffix}"
    if src_key not in st.session_state:
        return
    ref_irr, ref_temp, gamma, a, b, ac_max_project_kw, pr_nominal = _get_site_params()
    df = st.session_state[src_key].copy()
    if df.empty:
        return
    df = _auto_sync_availability_flags(df)

    poa_met = pd.to_numeric(df["MET_POA_Wm2"], errors="coerce").astype(float).to_numpy()
    poa_sat = pd.to_numeric(df["SAT_POA_Wm2"], errors="coerce").astype(float).to_numpy()
    met_poa_av = df["MET_POA_available"].fillna(False).astype(bool).to_numpy()
    poa_for_kpi = np.where(met_poa_av & np.isfinite(poa_met), poa_met, poa_sat)

    t_mod, _ = _solve_tmod(df, a, b)

    eac = pd.to_numeric(df["AC_kWh"], errors="coerce").astype(float).to_numpy()

    if inv_dc_kwp is None:
        inv_dc_kwp = _get_dc_scalar(key_suffix, np.nan)

    y_ref, y_ac, _ = _yields_pr_scalar_dc(poa_for_kpi, t_mod, ref_irr, ref_temp, gamma, eac, inv_dc_kwp)

    fullcap = df["FullCap"].fillna(False).astype(bool).to_numpy()

    st.session_state.setdefault("mi_yac", {})
    st.session_state.setdefault("mi_fullcap", {})
    st.session_state["mi_yac"][inv_name] = y_ac
    st.session_state["mi_fullcap"][inv_name] = fullcap

def _refresh_site_median_cache():
    inv_map = {"INV1": "inv1", "INV2": "inv2", "INV3": "inv3"}
    st.session_state["mi_yac"] = {}
    st.session_state["mi_fullcap"] = {}
    for name, sfx in inv_map.items():
        dc_scalar = _get_dc_scalar(sfx, np.nan)
        _recompute_for_one_inverter_from_src(name, sfx, inv_dc_kwp=dc_scalar)

# ---------------------------
# Realistic presets
# ---------------------------
def _realistic_presets(inv_name: str) -> pd.DataFrame:
    times = ["10:00","10:10","10:20","10:30","10:40"]
    if inv_name == "INV1":
        return pd.DataFrame({
            "Time": times,
            "MET_POA_Wm2":[880, 920, 940, 900, 840],
            "MET_POA_available":[True, True, True, True, True],
            "SAT_POA_Wm2":[885, 925, 945, 905, 845],
            "MET_Amb_C":[26.0, 27.0, 27.5, 27.2, 26.8],
            "MET_amb_available":[True, True, True, True, True],
            "MET_Back_Module_C":[np.nan, 49.0, np.nan, 50.0, np.nan],
            "MET_back_available":[False, True, False, True, False],
            "MET_WindSpeed":[2.0, 2.5, 3.2, 2.8, 2.2],
            "Wind_available":[True, True, True, True, True],
            "SAT_T_mod_C":[47.0, 48.5, 49.5, 48.8, 47.9],
            "AC_kWh":[50.0, 50.0, 46.0, 48.0, 47.0],
            "Availability":[1.0,1.0,1.0,1.0,1.0],
            "Event_Weight":[1.0,1.0,1.0,1.0,1.0],
            "FullCap":[True, True, True, True, True],
        })
    if inv_name == "INV2":
        return pd.DataFrame({
            "Time": times,
            "MET_POA_Wm2":[820, 860, 880, 850, 800],
            "MET_POA_available":[True, True, True, True, False],
            "SAT_POA_Wm2":[830, 870, 890, 860, 810],
            "MET_Amb_C":[26.1, 27.1, 27.6, 27.3, 26.9],
            "MET_amb_available":[True, True, True, True, True],
            "MET_Back_Module_C":[np.nan, np.nan, 48.5, np.nan, np.nan],
            "MET_back_available":[False, False, True, False, False],
            "MET_WindSpeed":[2.3, 2.8, 3.0, 2.6, 2.1],
            "Wind_available":[True, True, True, True, True],
            "SAT_T_mod_C":[46.8, 47.9, 48.6, 48.2, 47.5],
            "AC_kWh":[46.0, 49.0, 51.0, 49.0, 48.0],
            "Availability":[1.0,1.0,1.0,1.0,1.0],
            "Event_Weight":[1.0,1.0,1.0,1.0,1.0],
            "FullCap":[True, True, True, True, True],
        })
    return pd.DataFrame({
        "Time": times,
        "MET_POA_Wm2":[760, 790, 810, 780, 740],
        "MET_POA_available":[True, True, True, True, True],
        "SAT_POA_Wm2":[765, 795, 815, 785, 745],
        "MET_Amb_C":[25.8, 26.7, 27.2, 27.0, 26.5],
        "MET_amb_available":[True, True, True, True, True],
        "MET_Back_Module_C":[np.nan, np.nan, np.nan, 47.0, np.nan],
        "MET_back_available":[False, False, False, True, False],
        "MET_WindSpeed":[2.0, 2.4, 2.9, 2.5, 2.0],
        "Wind_available":[True, True, True, True, True],
        "SAT_T_mod_C":[46.0, 47.4, 48.0, 47.7, 46.8],
        "AC_kWh":[50.0, 49.0, 48.0, 49.0, 47.0],
        "Availability":[1.0,1.0,1.0,1.0,1.0],
        "Event_Weight":[1.0,1.0,1.0,1.0,1.0],
        "FullCap":[True, True, False, True, True],
    })

# ---------------------------
# Proxy median builder
# ---------------------------
def _proxy_median_rowwise(others_yac: dict, others_fc: dict, idx: int, threshold: float):
    vals = []
    for name, ya_vec in others_yac.items():
        v = ya_vec[idx] if idx < len(ya_vec) else np.nan
        fc_vec = others_fc.get(name, np.full_like(ya_vec, True, dtype=bool))
        ok = (idx < len(fc_vec) and bool(fc_vec[idx])) and np.isfinite(v) and (v > threshold)
        if ok:
            vals.append(float(v))
    if len(vals) == 0:
        return [], float("nan")
    return vals, float(np.median(vals))

# ---------------------------
# Inverter page
# ---------------------------
def _inverter_page(inv_name: str, key_suffix: str):
    ref_irr, ref_temp, gamma, a, b, ac_max_project_kw, pr_nominal = _get_site_params()

    st.subheader(f"{inv_name} — 10-min Concept Data")

    # Inverter Settings
    st.markdown("#### Inverter Settings")
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.markdown("🔌 **Inverter DC capacity (kWp)**")
        dc_inv_kwp = st.number_input(
            f"{inv_name} — DC capacity (kWp)",
            min_value=0.0,
            value=float(st.session_state.get(f"dc_scalar_{key_suffix}",
                                             st.session_state.get(f"dc_{key_suffix}_kwp", 700.0))),
            step=10.0,
            key=f"dc_scalar_{key_suffix}",
            help="Single DC capacity used for all 10-min rows (kWp).",
        )
    with c2:
        st.markdown("🏷️ **FullCap threshold for Median(Y_ac)**")
        thr = st.number_input(
            "Min Specific Yield (h) to include in Median",
            min_value=0.0, value=float(st.session_state.get("site_median_min_yac", 0.05)),
            step=0.01, key=f"thr_{key_suffix}",
            help="Rows with Y_ac <= threshold are excluded from the median.",
        )
        st.session_state["site_median_min_yac"] = float(thr)

    # init source DF
    src_key = f"src_{key_suffix}"
    if src_key not in st.session_state:
        st.session_state[src_key] = _realistic_presets(inv_name).copy()

    # editor
    st.markdown("### Inputs")
    ed = st.data_editor(
        st.session_state[src_key],
        num_rows="dynamic",
        key=f"ed_{key_suffix}",
        use_container_width=True,
        column_config={
            "Time": st.column_config.TextColumn("Time (hh:mm)", help=HOVER["Time"]),

            "MET_POA_Wm2": st.column_config.NumberColumn("MET-POA (W/m²)", min_value=0.0, step=1.0, help=HOVER["MET_POA_Wm2"]),
            "MET_POA_available": st.column_config.CheckboxColumn("MET POA available", help=HOVER["MET_POA_available"]),

            "SAT_POA_Wm2": st.column_config.NumberColumn("SAT-POA (W/m²)", min_value=0.0, step=1.0, help=HOVER["SAT_POA_Wm2"]),

            "MET_Amb_C": st.column_config.NumberColumn("MET Amb (°C)", step=0.1, help=HOVER["MET_Amb_C"]),
            "MET_amb_available": st.column_config.CheckboxColumn("MET Amb available", help=HOVER["MET_amb_available"]),

            "MET_Back_Module_C": st.column_config.NumberColumn("MET Back Module (°C)", step=0.1, help=HOVER["MET_Back_Module_C"]),
            "MET_back_available": st.column_config.CheckboxColumn("MET Back available", help=HOVER["MET_back_available"]),

            "MET_WindSpeed": st.column_config.NumberColumn("Wind speed (m/s)", step=0.1, help=HOVER["MET_WindSpeed"]),
            "Wind_available": st.column_config.CheckboxColumn("Wind available", help=HOVER["Wind_available"]),

            "SAT_T_mod_C": st.column_config.NumberColumn("SAT T_mod (°C)", step=0.1, help=HOVER["SAT_T_mod_C"]),

            "AC_kWh": st.column_config.NumberColumn("AC_kWh (per 10-min)", min_value=0.0, step=0.001, help=HOVER["AC_kWh"]),

            "Availability": st.column_config.NumberColumn("Availability (0–1)", min_value=0.0, max_value=1.0, step=0.01, help=HOVER["Availability"]),
            "Event_Weight": st.column_config.NumberColumn("Event Weight (0–1)", min_value=0.0, max_value=1.0, step=0.01, help=HOVER["Event_Weight"]),

            "FullCap": st.column_config.CheckboxColumn("Full Capacity (include in Median)", help=HOVER["FullCap"]),
        },
    )

    if ed.empty:
        st.info("No rows to compute.")
        return

    # auto-sync + persist + refresh caches
    synced = _auto_sync_availability_flags(ed)

    editor_changed = not synced.reset_index(drop=True).equals(
        st.session_state[src_key].reset_index(drop=True)
    )
    if editor_changed:
        st.session_state[src_key] = synced
        _refresh_site_median_cache()
        st.rerun()

    ed = st.session_state[src_key].copy()
    ed = _auto_sync_availability_flags(ed)

    # Resolve inputs
    poa_met = pd.to_numeric(ed["MET_POA_Wm2"], errors="coerce").astype(float).to_numpy()
    poa_sat = pd.to_numeric(ed["SAT_POA_Wm2"], errors="coerce").astype(float).to_numpy()
    met_poa_av = ed["MET_POA_available"].fillna(False).astype(bool).to_numpy()
    poa_for_kpi = np.where(met_poa_av & np.isfinite(poa_met), poa_met, poa_sat)
    poa_src = np.where(met_poa_av & np.isfinite(poa_met), "MET_POA", "SAT_POA")

    t_mod, t_src = _solve_tmod(ed, a, b)

    eac = pd.to_numeric(ed["AC_kWh"], errors="coerce").astype(float).to_numpy()
    fullcap = ed["FullCap"].fillna(False).astype(bool).to_numpy()

    # KPI yields
    y_ref, y_ac, pr = _yields_pr_scalar_dc(poa_for_kpi, t_mod, ref_irr, ref_temp, gamma, eac, dc_inv_kwp)

    # Save for site-median usage
    st.session_state.setdefault("mi_yac", {})
    st.session_state.setdefault("mi_fullcap", {})
    st.session_state["mi_yac"][inv_name] = y_ac
    st.session_state["mi_fullcap"][inv_name] = fullcap

    # Site-median (others only)
    others_yac = {k:v for k,v in st.session_state["mi_yac"].items() if k != inv_name}
    others_fc  = {k:v for k,v in st.session_state["mi_fullcap"].items() if k != inv_name}
    n_rows = len(eac)
    site_median_yac = np.full(n_rows, np.nan, dtype=float)
    for i in range(n_rows):
        used, med = _proxy_median_rowwise(others_yac, others_fc, i, float(st.session_state.get("site_median_min_yac", 0.05)))
        site_median_yac[i] = med if np.isfinite(med) else np.nan

    # Proxy potential (per inverter row)
    cap_10min = (float(st.session_state.get("ac_max_project_kw", 1800.0)) / max(N_INV,1)) * DT_HOURS
    e_proxy_raw = site_median_yac * float(dc_inv_kwp)
    e_potential = np.minimum(e_proxy_raw, cap_10min)

    # Lost Energy components
    availability = pd.to_numeric(ed["Availability"], errors="coerce").astype(float).to_numpy()
    event_weight = pd.to_numeric(ed["Event_Weight"], errors="coerce").astype(float).to_numpy()
    availability = np.nan_to_num(availability, nan=1.0).clip(0.0, 1.0)
    event_weight = np.nan_to_num(event_weight, nan=1.0).clip(0.0, 1.0)

    eac_used = np.where(np.isfinite(eac), eac, 0.0)
    gap_vec = np.maximum(0.0, e_potential - eac_used)
    lost_vec = np.maximum(0.0, (1.0 - availability) * gap_vec * event_weight)

    # Resolved Inputs
    st.markdown("### Resolved Inputs")
    resolved = pd.DataFrame({
        "Time": ed["Time"],
        "POA_used (W/m²)": np.round(poa_for_kpi, 3),
        "POA_Source": poa_src,
        "T_mod_used (°C)": np.round(t_mod, 3),
        "T_mod_Source": t_src,
        "AC_kWh": np.round(eac, 6),
        "Availability": np.round(availability, 3),
        "Event_Weight": np.round(event_weight, 3),
        "FullCap": fullcap,
    })
    st.data_editor(
        resolved, key=f"resolved_{key_suffix}", use_container_width=True, disabled=True,
        column_config={
            "Time": st.column_config.TextColumn("Time (hh:mm)", help=HOVER["Time"]),
            "POA_used (W/m²)": st.column_config.NumberColumn("POA_used (W/m²)", help=HOVER["POA_used (W/m²)"]),
            "POA_Source": st.column_config.TextColumn("POA_Source", help=HOVER["POA_Source"]),
            "T_mod_used (°C)": st.column_config.NumberColumn("T_mod_used (°C)", help=HOVER["T_mod_used (°C)"]),
            "T_mod_Source": st.column_config.TextColumn("T_mod_Source", help=HOVER["T_mod_Source"]),
            # keep original key case mapping; only add help
            "AC_KWh": st.column_config.NumberColumn("AC_kWh", help=HOVER["AC_KWh"]),
            "Availability": st.column_config.NumberColumn("Availability (0–1)", help=HOVER["Availability"]),
            "Event_Weight": st.column_config.NumberColumn("Event Weight (0–1)", help=HOVER["Event_Weight"]),
            "FullCap": st.column_config.CheckboxColumn("FullCap", help=HOVER["FullCap"]),
        }
    )

    # Resolved Inputs — parameter details
    with st.expander("Resolved Inputs — parameter details (selected row)", expanded=False):
        times_list = ed["Time"].astype(str).tolist() if "Time" in ed.columns else [f"Row {i}" for i in range(len(ed))]
        labels_details = [f"{i:02d} — {t}" for i, t in enumerate(times_list)]
        idx_detail = st.selectbox(
            f"{inv_name} — interval (details)",
            options=list(range(len(labels_details))),
            format_func=lambda i: labels_details[i],
            key=f"selrow_details_{key_suffix}"
        )

        def _num(col):
            if col in ed.columns:
                arr = pd.to_numeric(ed[col], errors="coerce").astype(float).to_numpy()
                return arr[idx_detail] if idx_detail < len(arr) and np.isfinite(arr[idx_detail]) else np.nan
            return np.nan

        def _flag(col):
            if col in ed.columns:
                arr = ed[col].fillna(False).astype(bool).to_numpy()
                return bool(arr[idx_detail]) if idx_detail < len(arr) else False
            return False

        poa_met_i  = _num("MET_POA_Wm2")
        poa_sat_i  = _num("SAT_POA_Wm2")
        poa_used = poa_met_i if (_flag("MET_POA_available") and np.isfinite(poa_met_i)) else poa_sat_i
        src_poa  = "MET_POA" if (_flag("MET_POA_available") and np.isfinite(poa_met_i)) else "SAT_POA"

        t_back = _num("MET_Back_Module_C")
        t_amb  = _num("MET_Amb_C")
        ws     = _num("MET_WindSpeed")
        t_sat  = _num("SAT_T_mod_C")

        if _flag("MET_back_available") and np.isfinite(t_back):
            src_tmod = "MET_back"
        elif (_flag("MET_POA_available") and np.isfinite(poa_met_i)) and _flag("MET_amb_available") and np.isfinite(t_amb):
            src_tmod = "King(MET,WS)" if _flag("Wind_available") and np.isfinite(ws) else "King(MET)"
        else:
            src_tmod = "SAT"

        poa_val  = float(poa_for_kpi[idx_detail]) if np.isfinite(poa_for_kpi[idx_detail]) else (poa_used if np.isfinite(poa_used) else np.nan)
        tmod_val = float(t_mod[idx_detail]) if np.isfinite(t_mod[idx_detail]) else np.nan
        eac_val  = float(eac[idx_detail]) if np.isfinite(eac[idx_detail]) else np.nan

        param_rows = [
            {"Parameter": "POA_used (W/m²)",     "Value": _fmt(poa_val, 3),  "Description": f"Using {src_poa}."},
            {"Parameter": "T_mod_used (°C)",     "Value": _fmt(tmod_val, 3), "Description": f"Derived from {src_tmod}."},
            {"Parameter": "AC_kWh (per 10-min)", "Value": _fmt(eac_val, 6),  "Description": "Measured AC energy."},
        ]
        details_df = pd.DataFrame(param_rows)
        st.data_editor(
            details_df,
            key=f"details_params_inputs_only_{key_suffix}",
            use_container_width=True,
            disabled=True,
            hide_index=True,
        )

        # T_mod_used equation display
        st.markdown("**T_mod_used (°C) — formula and numeric application**")
        a_local = float(st.session_state.get("king_a", -3.56))
        b_local = float(st.session_state.get("king_b", -0.075))

        if src_tmod == "MET_back":
            st.latex(r"T_{\mathrm{mod}} = T_{\mathrm{back}}")
            st.latex(fr"T_{{\mathrm{{mod}}}} = {t_back:.2f}\ \mathrm{{^\circ C}}")

        elif src_tmod == "King(MET,WS)":
            st.latex(r"T_{\mathrm{mod}} = T_{\mathrm{amb}} + \mathrm{POA}\,\exp\!\left(a + b\,\mathrm{WS}\right)")
            exp_inner = fr"{a_local:.3f} {'-' if b_local < 0 else '+'} {abs(b_local):.3f}\cdot {ws:.2f}"
            st.latex(
                fr"T_{{\mathrm{{mod}}}} = {t_amb:.2f} + {poa_met_i:.2f}\,\exp\!\left({exp_inner}\right)"
                fr" = {tmod_val:.2f}\ \mathrm{{^\circ C}}"
            )

        elif src_tmod == "King(MET)":
            st.latex(r"T_{\mathrm{mod}} = T_{\mathrm{amb}} + \mathrm{POA}\,\exp\!\left(a\right)")
            st.latex(
                fr"T_{{\mathrm{{mod}}}} = {t_amb:.2f} + {poa_met_i:.2f}\,\exp\!\left({a_local:.3f}\right)"
                fr" = {tmod_val:.2f}\ \mathrm{{^\circ C}}"
            )

        else:
            st.latex(r"T_{\mathrm{mod}} = T_{\mathrm{sat}}")
            st.latex(fr"T_{{\mathrm{{mod}}}} = {t_sat:.2f}\ \mathrm{{^\circ C}}")

    # KPI table (per 10-min)
    st.markdown("### KPI Data (per 10-min)")
    kpi = pd.DataFrame({
        "Time": ed["Time"],
        "Reference Yield (h at STC)": np.round(y_ref, 6),
        "Specific Yield (h = kWh/kWp)": np.round(y_ac, 6),
        "Performance Ratio (% at STC)": np.round((y_ac / y_ref) * 100.0, 3),
        "Site Median Specific Yield (h)": np.round(site_median_yac, 6),
        "Proxy — Reference Potential (kWh)": np.round(e_potential, 6),
        "Lost Energy (kWh)": np.round(lost_vec, 3),
    })
    st.data_editor(
        kpi, key=f"kpi_{key_suffix}", use_container_width=True, disabled=True,
        column_config={
            "Time": st.column_config.TextColumn("Time (hh:mm)", help=HOVER["Time"]),
            "Reference Yield (h at STC)": st.column_config.NumberColumn("Reference Yield (h at STC)", format="%.3f", help=HOVER["Reference Yield (h at STC)"]),
            "Specific Yield (h = kWh/kWp)": st.column_config.NumberColumn("Specific Yield (h = kWh/kWp)", format="%.3f", help=HOVER["Specific Yield (h = kWh/kWp)"]),
            "Performance Ratio (% at STC)": st.column_config.NumberColumn("Performance Ratio (% at STC)", format="%.2f", help=HOVER["Performance Ratio (% at STC)"]),
            "Site Median Specific Yield (h)": st.column_config.NumberColumn("Site Median Specific Yield (h)", format="%.3f", help=HOVER["Site Median Specific Yield (h)"]),
            "Proxy — Reference Potential (kWh)": st.column_config.NumberColumn("Proxy — Reference Potential (kWh)", format="%.3f", help=HOVER["Proxy — Reference Potential (kWh)"]),
            "Lost Energy (kWh)": st.column_config.NumberColumn("Lost Energy (kWh)", format="%.3f", help=HOVER["Lost Energy (kWh)"]),
        }
    )
# ------------- KPI numeric details (per 10-min) -------------
    with st.expander("KPI numeric details — one-line formulas per 10-min", expanded=False):
        st.caption("One-line numeric substitutions for Y_ref, SpecificYield, PR, Proxy Potential, gap and Lost.")

        labels = [f"{i:02d} — {t}" for i, t in enumerate(ed["Time"].tolist())]
        idx = st.selectbox(f"{inv_name} — interval", options=list(range(len(labels))), format_func=lambda i: labels[i], key=f"selbox_{key_suffix}")

        poa_i   = float(poa_for_kpi[idx]) if np.isfinite(poa_for_kpi[idx]) else float("nan")
        tmod_i  = float(t_mod[idx]) if np.isfinite(t_mod[idx]) else float("nan")
        eac_i   = float(eac[idx]) if np.isfinite(eac[idx]) else float("nan")
        yref_i  = float(y_ref[idx]) if np.isfinite(y_ref[idx]) else float("nan")
        yac_i   = float(y_ac[idx]) if np.isfinite(y_ac[idx]) else float("nan")
        pr_i    = (yac_i / yref_i * 100.0) if (np.isfinite(yref_i) and yref_i != 0 and np.isfinite(yac_i)) else float("nan")

        ep_i    = float(e_potential[idx]) if np.isfinite(e_potential[idx]) else float("nan")
        eac_used_i = eac_i if np.isfinite(eac_i) else 0.0
        gap_i   = max(0.0, (ep_i - eac_used_i)) if (np.isfinite(ep_i) and np.isfinite(eac_used_i)) else float("nan")
        av_i    = float(availability[idx]) if np.isfinite(availability[idx]) else 1.0
        w_i     = float(event_weight[idx]) if np.isfinite(event_weight[idx]) else 1.0
        lost_i  = max(0.0, (1.0 - av_i) * gap_i * w_i) if np.isfinite(gap_i) else float("nan")

        poa_s   = _fmt(poa_i, 3)
        refirr  = _fmt(ref_irr, 3)
        gamma_s = _fmt(gamma, 6)
        tmod_s  = _fmt(tmod_i, 3)
        tref_s  = _fmt(ref_temp, 3)
        yref_s  = _fmt(yref_i, 6)
        eac_s   = _fmt(eac_i, 6)
        dc_s    = _fmt(dc_inv_kwp, 3)
        yac_s   = _fmt(yac_i, 6)
        pr_s    = _fmt(pr_i, 3)

        ep_s    = _fmt(ep_i, 3)
        eac_gap_s = _fmt(eac_used_i, 3)
        gap_s   = _fmt(gap_i, 3)
        av_s    = _fmt(av_i, 3)
        w_s     = _fmt(w_i, 3)
        lost_s  = _fmt(lost_i, 3)

        acmax = float(st.session_state.get("ac_max_project_kw", 1800.0))  # project AC cap
        mini_inputs_flipped = pd.DataFrame([{
            "POA_used (W/m²)":              _fmt(poa_i, 3),
            "POA_Source":                   str(poa_src[idx]),
            "T_mod_used (°C)":              _fmt(tmod_i, 3),
            "T_mod_Source":                 str(t_src[idx]),
            "AC_kWh (per 10-min)":          _fmt(eac_i, 6),
            f"{inv_name} DC capacity (kWp)": _fmt(dc_inv_kwp, 3),
            "AC Max Power (project) [kW]":  _fmt(acmax, 3),
            "Availability (0–1)":           av_s,
            "Event Weight (0–1)":           w_s,
            "Count of inverters (N_inv)":   str(N_INV),
        }])

        st.data_editor(
            mini_inputs_flipped,
            key=f"kpi_inputs_flipped_{key_suffix}",
            use_container_width=True,
            disabled=True,
            hide_index=True,
        )

        # 1) Reference Yield (h)
        st.markdown("#### Reference Yield (h)")
        _latex(
            r"Y_{[[\mathrm{ref}]]}(h)=\frac{[[\mathrm{POA}]]}{[[\mathrm{RefIrr}]]\cdot\left(1+\gamma\cdot(T_{[[\mathrm{mod}]]}-T_{[[\mathrm{ref}]]})\right)}\cdot\frac{1}{6}"
        )
        _latex(
            r"=\frac{<poa>}{<refirr>\cdot\left(1+<gamma>\cdot(<tmod>-<tref>)\right)}\cdot\frac{1}{6}=<yref>\ \mathrm{h}",
            poa=poa_s, refirr=refirr, gamma=gamma_s, tmod=tmod_s, tref=tref_s, yref=yref_s
        )

        # 2) Specific Yield (h)
        st.markdown("#### Specific Yield (h)")
        _latex(r"\mathrm{SpecificYield}(h)=\frac{E_{[[\mathrm{ac}]]}(\mathrm{kWh})}{DC_{[[\mathrm{inv}]]}(\mathrm{kWp})}")
        _latex(r"=\frac{<eac>}{<dc>}=<yac>\ \mathrm{h}", eac=eac_s, dc=dc_s, yac=yac_s)

        # 3) Performance Ratio (%)
        st.markdown("#### Performance Ratio (%)")
        _latex(r"\mathrm{PR}(\%)=\frac{\mathrm{SpecificYield}(h)}{Y_{[[\mathrm{ref}]]}(h)}\times 100")
        _latex(r"=\left(\frac{<yac>}{<yref>}\right)\times 100=<pr>\ \%", yac=yac_s, yref=yref_s, pr=pr_s)
        # --- Median(Y_ac) — explicit candidates + derivation (others only) ---
        st.markdown("#### Median(Y_ac) — candidates and numeric derivation")

        # threshold used for inclusion
        min_yac_th = float(st.session_state.get("site_median_min_yac", 0.05))

        # build candidate list from OTHER inverters only (FullCap + threshold)
        candidates = []
        used_vals = []
        mi_yac = st.session_state.get("mi_yac", {})
        mi_fc  = st.session_state.get("mi_fullcap", {})
        for name, ya_vec in mi_yac.items():
            if name == inv_name:
                continue
            val = ya_vec[idx] if idx < len(ya_vec) else np.nan
            fc_vec = mi_fc.get(name, np.full_like(ya_vec, True, dtype=bool))
            fc_ok  = (idx < len(fc_vec) and bool(fc_vec[idx]))
            val_ok = np.isfinite(val) and (val > min_yac_th)
            used   = fc_ok and val_ok
            candidates.append({
                "Inverter": name,
                "Y_ac (h)": _fmt(val, 6),
                "FullCap": "True" if fc_ok else "False",
                f"> {min_yac_th:.2f} h": "True" if val_ok else "False",
                "Used in median": "Yes" if used else "No",
            })
            if used:
                used_vals.append(float(val))

        st.data_editor(
            pd.DataFrame(candidates) if candidates else pd.DataFrame(
                columns=["Inverter","Y_ac (h)","FullCap",f"> {min_yac_th:.2f} h","Used in median"]
            ),
            key=f"proxy_median_candidates_details_{key_suffix}",
            use_container_width=True, disabled=True, hide_index=True,
        )

        # numeric derivation of the row-wise median
        if len(used_vals) == 0:
            med_show = float("nan")
            _latex(r"\mathrm{Median}(Y_{[[\mathrm{ac}]]})(h)=\text{No valid inputs}\ \Rightarrow\ \text{NaN}")
        else:
            sorted_vals = sorted(used_vals)
            n_used = len(sorted_vals)
            med_calc = float(np.median(sorted_vals))
            lst = ", ".join(_fmt(v, 6) for v in sorted_vals)
            if n_used % 2 == 1:
                vmid = sorted_vals[n_used // 2]
                _latex(
                    r"\mathrm{Median}(Y_{[[\mathrm{ac}]]})(h)=\mathrm{median}\!\Big([<lst>]\Big)=<vmid>\ \mathrm{h}",
                    lst=lst, vmid=_fmt(vmid, 6)
                )
            else:
                k1 = n_used // 2 - 1
                k2 = n_used // 2
                v1, v2 = sorted_vals[k1], sorted_vals[k2]
                avg = (v1 + v2)/2.0
                _latex(
                    r"\mathrm{Median}(Y_{[[\mathrm{ac}]]})(h)=\mathrm{median}\!\Big([<lst>]\Big)=\frac{<v1>+<v2>}{2}=<avg>\ \mathrm{h}",
                    lst=lst, v1=_fmt(v1,6), v2=_fmt(v2,6), avg=_fmt(avg,6)
                )
            med_show = med_calc

        st.caption(f"Median used here: {_fmt(med_show,6)} h — value used by E_potential at this row: {_fmt(site_median_yac[idx],6)} h")

        # 4) Proxy Potential (kWh)
        st.markdown("#### Proxy — Reference Potential (kWh)")
        _latex(
            r"E_{[[\mathrm{potential}]]}(\mathrm{kWh})"
            r"=\min\!\Big(\mathrm{Median}(Y_{[[\mathrm{ac}]]})(h)\times DC_{[[\mathrm{inv}]]},\ "
            r"\frac{AC_{\max,[[\mathrm{project}]]}}{N_{[[\mathrm{inv}]]}}\cdot\frac{1}{6}\Big)"
        )
        _latex(
            r"=\min\!\Big(<med>\times <dc>,\ \frac{<acmax>}{<ninv>}\cdot\frac{1}{6}\Big)=<ep>\ \mathrm{kWh}",
            med=_fmt(site_median_yac[idx],6), dc=_fmt(dc_inv_kwp,3),
            acmax=_fmt(acmax,3), ninv=str(N_INV), ep=_fmt(ep_i,3)
        )

        # 5) Lost Energy (kWh)
        st.markdown("#### Lost Energy (kWh)")
        _latex(
            r"\textbf{gap}(\mathrm{kWh})=\max\!\big(0,\ E_{\mathrm{potential}}-E_{\mathrm{ac}}\big)"
            r"=\max\!\big(0,\ <ep>-<eacgap>\big)=<gap>\ \mathrm{kWh}",
            ep=_fmt(ep_i,3), eacgap=_fmt(eac_used_i,3), gap=_fmt(gap_i,3)
        )
        _latex(
            r"\textbf{Lost}(\mathrm{kWh})=\max\!\big(0,\ (1-\mathrm{Availability})\times \text{gap}\times \mathrm{Event\_Weight}\big)"
            r"=\max\!\big(0,\ (1-<av>)\times <gap>\times <w>\big)=<lost>\ \mathrm{kWh}",
            av=_fmt(av_i,3), gap=_fmt(gap_i,3), w=_fmt(w_i,3), lost=_fmt(lost_i,3)
        )

    # KPI numeric details (expander) — unchanged logic

    # Daily totals (inverter)
    st.markdown("### Daily Totals & Aggregated KPIs")
    total_eac = float(np.nansum(eac))
    total_pot = float(np.nansum(e_potential))
    total_lost = float(np.nansum(lost_vec))

    y_ac_day  = (total_eac / float(dc_inv_kwp)) if float(dc_inv_kwp) > 0 else np.nan
    y_ref_day = float(np.nansum(y_ref)) if np.isfinite(np.nansum(y_ref)) else np.nan
    pr_day    = (y_ac_day / y_ref_day) if (np.isfinite(y_ref_day) and y_ref_day != 0) else np.nan

    all_pba = ((total_pot - total_lost) / total_pot * 100.0) if (np.isfinite(total_pot) and total_pot > 0) else np.nan

    totals = pd.DataFrame([{
        "Σ E_ac (kWh)": total_eac if np.isfinite(total_eac) else np.nan,
        "Σ Potential (kWh)": total_pot if np.isfinite(total_pot) else np.nan,
        "Σ Lost (kWh)": total_lost if np.isfinite(total_lost) else np.nan,
        "Reference Yield — day (h)": y_ref_day if np.isfinite(y_ref_day) else np.nan,
        "Specific Yield — day (h)": y_ac_day if np.isfinite(y_ac_day) else np.nan,
        "Performance Ratio — day (%)": pr_day*100 if np.isfinite(pr_day) else np.nan,
        "All PBA (%)": all_pba if np.isfinite(all_pba) else np.nan,
    }])
    st.data_editor(
        totals, key=f"totals_{key_suffix}", use_container_width=True, disabled=True,
        column_config={
            "Σ E_ac (kWh)": st.column_config.NumberColumn("Σ E_ac (kWh)", format="%.3f", help=HOVER["Σ E_ac (kWh)"]),
            "Σ Potential (kWh)": st.column_config.NumberColumn("Σ Potential (kWh)", format="%.3f", help=HOVER["Σ Potential (kWh)"]),
            "Σ Lost (kWh)": st.column_config.NumberColumn("Σ Lost (kWh)", format="%.3f", help=HOVER["Σ Lost (kWh)"]),
            "Reference Yield — day (h)": st.column_config.NumberColumn("Reference Yield — day (h)", format="%.3f", help=HOVER["Reference Yield — day (h)"]),
            "Specific Yield — day (h)": st.column_config.NumberColumn("Specific Yield — day (h)", format="%.3f", help=HOVER["Specific Yield — day (h)"]),
            "Performance Ratio — day (%)": st.column_config.NumberColumn("Performance Ratio — day (%)", format="%.2f", help=HOVER["Performance Ratio — day (%)"]),
            "All PBA (%)": st.column_config.NumberColumn("All PBA (%)", format="%.2f", help=HOVER["All PBA (%)"]),
        }
    )
 # --- Daily totals — numeric details (one-line formulas) ---
    with st.expander("Daily Totals — numeric details (one-line formulas)", expanded=False):
        eac_terms = [v for v in eac if np.isfinite(v)]
        yref_terms = [v for v in y_ref if np.isfinite(v)]
        epot_terms = [v for v in e_potential if np.isfinite(v)]
        lost_terms = [v for v in lost_vec if np.isfinite(v)]

        def _join(vals, n=3):
            return " + ".join(_fmt(v, n) for v in vals)

        total_eac_s = _fmt(np.nansum(eac), 3)
        total_pot_s = _fmt(np.nansum(e_potential), 3)
        total_lost_s= _fmt(np.nansum(lost_vec), 3)
        y_ref_day_s = _fmt(np.nansum(y_ref), 6)
        y_ac_day_s  = _fmt((np.nansum(eac) / float(dc_inv_kwp)) if float(dc_inv_kwp) > 0 else np.nan, 3)
        pr_day_pct  = (float(y_ac_day_s) / float(y_ref_day_s) * 100.0) if (y_ref_day_s not in ["NaN","nan"] and y_ac_day_s not in ["NaN","nan"] and float(y_ref_day_s) != 0) else float("nan")
        pr_day_s    = _fmt(pr_day_pct, 3)
        dc_inv_s    = _fmt(dc_inv_kwp, 1)
        all_pba_s   = _fmt(((float(total_pot_s) - float(total_lost_s)) / float(total_pot_s) * 100.0) if (total_pot_s not in ["NaN","nan"] and float(total_pot_s)>0) else np.nan, 3)

        st.markdown("#### 1) ΣE_ac (kWh)")
        _latex(r"\sum E_{[[\mathrm{ac}]]}(\mathrm{kWh})")
        _latex(r"= <terms> = <total>\ \mathrm{kWh}", terms=_join(eac_terms, n=1), total=total_eac_s)

        st.markdown("#### 2) DC capacity (kWp)")
        _latex(r"DC_{[[\mathrm{inv}]]}(\mathrm{kWp}) = <dc>", dc=dc_inv_s)

        st.markdown("#### 3) SpecificYield_day (h)")
        _latex(r"Y_{[[\mathrm{ac}},\ \mathrm{day]]}(h) = \frac{\sum E_{[[\mathrm{ac}]]}}{DC_{[[\mathrm{inv}]]}}")
        _latex(r"= <sum_eac> \div <dc> \ \to\ <yac_day>\ \mathrm{h}", sum_eac=total_eac_s, dc=dc_inv_s, yac_day=y_ac_day_s)

        st.markdown("#### 4) ΣReferenceYield (h)")
        _latex(r"\sum Y_{[[\mathrm{ref}]]}(h)")
        _latex(r"= <terms> = <total>\ \mathrm{h}", terms=_join(yref_terms, n=6), total=y_ref_day_s)

        st.markdown("#### 5) PR_day (%)")
        _latex(r"\mathrm{PR}_{[[\mathrm{day}]]}(\%) = \frac{Y_{[[\mathrm{ac}},\ \mathrm{day]]}}{Y_{[[\mathrm{ref}},\ \mathrm{day]]}}\times 100")
        _latex(r"= \left(<yac_day> \div <yref_day>\right) \times 100 \ \to\ <pr_day>\ \%", yac_day=y_ac_day_s, yref_day=y_ref_day_s, pr_day=pr_day_s)

        st.markdown("#### 6) ΣE_potential (kWh)")
        _latex(r"\sum E_{[[\mathrm{potential}]]} = <terms> = <total>\ \mathrm{kWh}", terms=_join(epot_terms, n=1), total=total_pot_s)

        st.markdown("#### 7) ΣLost Energy (kWh)")
        _latex(r"\sum \mathrm{Lost} = <terms> = <total>\ \mathrm{kWh}", terms=_join(lost_terms, n=1), total=total_lost_s)

        st.markdown("#### 8) All PBA (%)")
        _latex(
            r"\mathrm{All\ PBA}(\%)=\frac{\sum E_{[[\mathrm{potential}]]}-\sum \mathrm{Lost}}{\sum E_{[[\mathrm{potential}]]}}\times 100"
        )
        _latex(
            r"=\frac{<sum_pot>-<sum_lost>}{<sum_pot>}\times 100 \ \to\ <pba>\ \%",
            sum_pot=total_pot_s, sum_lost=total_lost_s, pba=all_pba_s
        )

    # Global Daily Energy Summary (inverter)
    st.markdown("### Global Daily Energy Summary")
    gross_prod = total_eac if np.isfinite(total_eac) else np.nan
    poi_ref    = gross_prod * 0.97 if np.isfinite(gross_prod) else np.nan
    all_losses = total_lost if np.isfinite(total_lost) else np.nan
    all_pba_val= all_pba if np.isfinite(all_pba) else np.nan
    all_pr_stc = (pr_day * 100.0) if np.isfinite(pr_day) else np.nan

    global_summary = pd.DataFrame([{
        "Gross inverter production (kWh)": np.round(gross_prod, 3),
        # "Reference POI Production (kWh)": np.round(poi_ref, 3),
        "All losses (kWh)": np.round(all_losses, 3),
        "All PBA (%)": np.round(all_pba_val, 2),
        "All PR STC (%)": np.round(all_pr_stc, 2),
    }])

    st.data_editor(
        global_summary,
        key=f"global_summary_{key_suffix}",
        use_container_width=True,
        disabled=True,
        hide_index=True,
        column_config={
            "Gross inverter production (kWh)": st.column_config.NumberColumn("Gross inverter production (kWh)", format="%.3f", help=HOVER["Gross inverter production (kWh)"]),
            # "Reference POI Production (kWh)": st.column_config.NumberColumn("Reference POI Production (kWh)", format="%.3f", help=HOVER["Reference POI Production (kWh)"]),
            "All losses (kWh)": st.column_config.NumberColumn("All losses (kWh)", format="%.3f", help=HOVER["All losses (kWh)"]),
            "All PBA (%)": st.column_config.NumberColumn("All PBA (%)", format="%.2f", help=HOVER["All PBA (%)"]),
            "All PR STC (%)": st.column_config.NumberColumn("All PR STC (%)", format="%.2f", help=HOVER["All PR STC (%)"]),
        }
    )

    st.session_state.setdefault("mi_outputs", {})
    st.session_state["mi_outputs"][inv_name] = {
        "time": ed["Time"].tolist(),
        "E_ac": eac,
        "E_potential": e_potential,
        "Lost": lost_vec,
        "Y_ref": y_ref,
        "DC_inv": float(dc_inv_kwp),
    }

# ---------------------------
# Site aggregation helpers
# ---------------------------
def _get_or_init_src(inv_name: str, key_suffix: str) -> pd.DataFrame:
    src_key = f"src_{key_suffix}"
    if src_key not in st.session_state:
        st.session_state[src_key] = _realistic_presets(inv_name).copy()
    return _auto_sync_availability_flags(st.session_state[src_key].copy())

def _build_site_default_meters(times, ac_sum_inv):
    n = len(times)
    base = np.asarray(ac_sum_inv, dtype=float)
    base = np.where(np.isfinite(base), base, 0.0)
    return pd.DataFrame({
        "Time": list(times),
        "AC_revenue_kWh": base,
        "AC_revenue_available": [True] * n,
        "AC_check_kWh": base,
        "AC_check_available": [True] * n,
    })

def _ensure_site_meters_df(times, default_df: pd.DataFrame | None = None):
    key = "site_meters_df"
    need_new = (key not in st.session_state)
    if not need_new:
        df = st.session_state[key]
        need_new = ("Time" not in df.columns) or (len(df) != len(times)) or (list(df["Time"]) != list(times))
    if need_new:
        if default_df is not None:
            st.session_state[key] = default_df.copy()
        else:
            n = len(times)
            st.session_state[key] = pd.DataFrame({
                "Time": list(times),
                "AC_revenue_kWh": np.zeros(n, dtype=float),
                "AC_revenue_available": [True] * n,
                "AC_check_kWh": np.zeros(n, dtype=float),
                "AC_check_available": [True] * n,
            })
    return st.session_state[key]

# ---------------------------
# Site page
# ---------------------------
def _site_page():
    st.subheader("Site — Inputs")

    ref_irr, ref_temp, gamma, a, b, ac_max_project_kw, pr_nominal = _get_site_params()

    invs = [("INV1", "inv1"), ("INV2", "inv2"), ("INV3", "inv3")]
    per_inv = []

    for name, sfx in invs:
        df = _get_or_init_src(name, sfx)
        dc = _get_dc_scalar(sfx, 700.0)

        poa_met = pd.to_numeric(df["MET_POA_Wm2"], errors="coerce").astype(float).to_numpy()
        poa_sat = pd.to_numeric(df["SAT_POA_Wm2"], errors="coerce").astype(float).to_numpy()
        met_av  = df["MET_POA_available"].fillna(False).astype(bool).to_numpy()
        poa_used = np.where(met_av & np.isfinite(poa_met), poa_met, poa_sat)

        t_mod_used, _t_src = _solve_tmod(df, a, b)

        eac = pd.to_numeric(df["AC_KWh"] if "AC_KWh" in df.columns else df["AC_kWh"], errors="coerce").astype(float).to_numpy()

        per_inv.append({
            "name": name,
            "dc": dc,
            "poa_used": poa_used,
            "t_mod_used": t_mod_used,
            "ac_kwh": eac,
            "time": df["Time"].astype(str).tolist(),
        })

    times = per_inv[0]["time"]

    n = len(times)
    dc_arr    = np.array([p["dc"] for p in per_inv], dtype=float)
    poa_mat   = np.vstack([p["poa_used"]    for p in per_inv])
    tmod_mat  = np.vstack([p["t_mod_used"]  for p in per_inv])
    ac_mat    = np.vstack([p["ac_kwh"]      for p in per_inv])
    ac_sum_inv = np.nansum(ac_mat, axis=0)

    if "site_meters_df" in st.session_state:
        meters_df = _ensure_site_meters_df(times)
    else:
        defaults_df = _build_site_default_meters(times, ac_sum_inv)
        meters_df = _ensure_site_meters_df(times, default_df=defaults_df)

    meters_df = st.data_editor(
        meters_df,
        key="site_meters_editor",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Time": st.column_config.TextColumn("Time (hh:mm)", help=HOVER["Time"]),
            "AC_revenue_kWh": st.column_config.NumberColumn("AC Revenue Meter (kWh)", min_value=0.0, step=0.001, help=HOVER["AC_revenue_kWh"]),
            "AC_revenue_available": st.column_config.CheckboxColumn("AC Revenue available", help=HOVER["AC_revenue_available"]),
            "AC_check_kWh": st.column_config.NumberColumn("AC Check Meter (kWh)", min_value=0.0, step=0.001, help=HOVER["AC_check_kWh"]),
            "AC_check_available": st.column_config.CheckboxColumn("AC Check available", help=HOVER["AC_check_available"]),
        },
    )
    st.session_state["site_meters_df"] = meters_df.copy()

    ac_rev = pd.to_numeric(meters_df["AC_revenue_kWh"], errors="coerce").astype(float).to_numpy()
    av_rev = meters_df["AC_revenue_available"].fillna(False).astype(bool).to_numpy()
    ac_chk = pd.to_numeric(meters_df["AC_check_kWh"], errors="coerce").astype(float).to_numpy()
    av_chk = meters_df["AC_check_available"].fillna(False).astype(bool).to_numpy()

    poa_site        = np.full(n, np.nan, dtype=float)
    t_mod_site      = np.full(n, np.nan, dtype=float)
    ac_site_resolved = np.full(n, np.nan, dtype=float)
    ac_source       = np.array([""]*n, dtype=object)

    for t in range(n):
        mask_poa = np.isfinite(poa_mat[:, t])
        denom_poa = np.sum(dc_arr[mask_poa]) if np.any(mask_poa) else 0.0
        poa_site[t] = (np.sum(poa_mat[:, t][mask_poa] * dc_arr[mask_poa]) / denom_poa) if denom_poa > 0 else np.nan

        mask_tmod = np.isfinite(tmod_mat[:, t])
        denom_t   = np.sum(dc_arr[mask_tmod]) if np.any(mask_tmod) else 0.0
        t_mod_site[t] = (np.sum(tmod_mat[:, t][mask_tmod] * dc_arr[mask_tmod]) / denom_t) if denom_t > 0 else np.nan

        if av_rev[t] and np.isfinite(ac_rev[t]):
            ac_site_resolved[t] = float(ac_rev[t]); ac_source[t] = "AC_revenue"
        elif av_chk[t] and np.isfinite(ac_chk[t]):
            ac_site_resolved[t] = float(ac_chk[t]); ac_source[t] = "AC_check"
        else:
            ac_site_resolved[t] = float(ac_sum_inv[t]) if np.isfinite(ac_sum_inv[t]) else np.nan
            ac_source[t] = "Σ inverter AC"

    st.markdown("### Resolved Inputs (Site)")
    dc_total = float(np.nansum(dc_arr)) if np.isfinite(np.nansum(dc_arr)) else np.nan
    resolved_site = pd.DataFrame({
        "Time": times,
        "POA_site (W/m², DC-weighted)": np.round(poa_site, 3),
        "T_mod_site (°C, DC-weighted)": np.round(t_mod_site, 3),
        "DC_total (kWp)": np.round(dc_total, 3),
        "AC_revenue (kWh)": np.round(ac_rev, 6),
        "AC_revenue_available": av_rev,
        "AC_check (kWh)": np.round(ac_chk, 6),
        "AC_check_available": av_chk,
        "Σ Inverter AC (kWh)": np.round(ac_sum_inv, 6),
        "AC_site_resolved (kWh)": np.round(ac_site_resolved, 6),
        "AC_source": ac_source,
    })
    st.data_editor(
        resolved_site,
        key="site_resolved_table",
        use_container_width=True,
        disabled=True,
        column_config={
            "Time": st.column_config.TextColumn("Time (hh:mm)", help=HOVER["Time"]),
            "POA_site (W/m², DC-weighted)": st.column_config.NumberColumn("POA_site (W/m², DC-weighted)", format="%.3f", help=HOVER["POA_site (W/m², DC-weighted)"]),
            "T_mod_site (°C, DC-weighted)": st.column_config.NumberColumn("T_mod_site (°C, DC-weighted)", format="%.3f", help=HOVER["T_mod_site (°C, DC-weighted)"]),
            "DC_total (kWp)": st.column_config.NumberColumn("DC_total (kWp)", format="%.3f", help=HOVER["DC_total (kWp)"]),
            "AC_revenue (kWh)": st.column_config.NumberColumn("AC_revenue (kWh)", format="%.6f", help=HOVER["AC_revenue_kWh"]),
            "AC_revenue_available": st.column_config.CheckboxColumn("AC_revenue available", help=HOVER["AC_revenue_available"]),
            "AC_check (kWh)": st.column_config.NumberColumn("AC_check (kWh)", format="%.6f", help=HOVER["AC_check_kWh"]),
            "AC_check_available": st.column_config.CheckboxColumn("AC_check available", help=HOVER["AC_check_available"]),
            "Σ Inverter AC (kWh)": st.column_config.NumberColumn("Σ Inverter AC (kWh)", format="%.6f", help=HOVER["Σ Inverter AC (kWh)"]),
            "AC_site_resolved (kWh)": st.column_config.NumberColumn("AC_site_resolved (kWh)", format="%.6f", help=HOVER["AC_site_resolved (kWh)"]),
            "AC_source": st.column_config.TextColumn("AC_source", help=HOVER["AC_source"]),
        }
    )
    #expander Resolved Inputs (site) 
    with st.expander("Resolved Inputs — parameter details (selected row)", expanded=False):
        labels_site = [f"{i:02d} — {t}" for i, t in enumerate(times)]
        idx_site = st.selectbox(
            "Site — interval (details)",
            options=list(range(len(labels_site))),
            format_func=lambda i: labels_site[i],
            key="selrow_details_site_all"
        )

        def _num_safe(arr, i):
            if 0 <= i < len(arr) and np.isfinite(arr[i]):
                return float(arr[i])
            return float("nan")

        def _bool_safe(arr, i):
            if 0 <= i < len(arr):
                return bool(arr[i])
            return False

        poa_site_i   = _num_safe(poa_site, idx_site)
        t_mod_site_i = _num_safe(t_mod_site, idx_site)
        dc_total_val = float(np.nansum(dc_arr)) if np.isfinite(np.nansum(dc_arr)) else float("nan")

        ac_rev_i  = _num_safe(ac_rev, idx_site)
        av_rev_i  = _bool_safe(av_rev, idx_site)
        ac_chk_i  = _num_safe(ac_chk, idx_site)
        av_chk_i  = _bool_safe(av_chk, idx_site)
        ac_sum_i  = _num_safe(ac_sum_inv, idx_site)
        ac_res_i  = _num_safe(ac_site_resolved, idx_site)
        ac_src_i  = str(ac_source[idx_site]) if 0 <= idx_site < len(ac_source) else ""

        site_param_rows = [
            {"Parameter": "POA_site (W/m², DC-weighted)", "Value": _fmt(poa_site_i, 3),  "Description": "DC-weighted average of inverter POA_used"},
            {"Parameter": "T_mod_site (°C, DC-weighted)", "Value": _fmt(t_mod_site_i, 3), "Description": "DC-weighted average of per-inverter T_mod_used"},
            {"Parameter": "DC_total (kWp)",               "Value": _fmt(dc_total_val, 3), "Description": "Sum of inverter DC capacities (weights)"},
            {"Parameter": "AC_site_resolved (kWh)",       "Value": _fmt(ac_res_i, 6),     "Description": f"Resolved by fallback → {ac_src_i}"},
        ]
        st.data_editor(
            pd.DataFrame(site_param_rows),
            key="details_params_inputs_site_all",
            use_container_width=True,
            disabled=True,
            hide_index=True,
        )

        st.markdown("#### POA_site (W/m²) — DC-weighted average")
        terms_p = []
        num_p_val = 0.0
        den_p_val = 0.0
        if 'poa_mat' in locals() and 0 <= idx_site < poa_mat.shape[1]:
            for i_inv in range(poa_mat.shape[0]):
                v_p  = poa_mat[i_inv, idx_site]
                v_dc = dc_arr[i_inv]
                if np.isfinite(v_p) and np.isfinite(v_dc):
                    terms_p.append(f"({_fmt(v_p,3)}\\times{_fmt(v_dc,3)})")
                    num_p_val += float(v_p) * float(v_dc)
                    den_p_val += float(v_dc)
        _latex(r"\mathrm{POA}_{[[\mathrm{site}]]}=\frac{\sum_i \mathrm{POA}_i\cdot DC_i}{\sum_i DC_i}")
        _latex(
            r"=\frac{<num_expand>}{<den>}=\boxed{<poa>\ \mathrm{W/m^2}}",
            num_expand=" + ".join(terms_p) if terms_p else "—",
            den=_fmt(den_p_val, 3),
            poa=_fmt(poa_site_i, 3)
        )

        st.markdown("#### T_mod_site (°C) — DC-weighted average")
        terms_t = []
        num_t_val = 0.0
        den_t_val = 0.0
        if 'tmod_mat' in locals() and 0 <= idx_site < tmod_mat.shape[1]:
            for i_inv in range(tmod_mat.shape[0]):
                v_t  = tmod_mat[i_inv, idx_site]
                v_dc = dc_arr[i_inv]
                if np.isfinite(v_t) and np.isfinite(v_dc):
                    terms_t.append(f"({_fmt(v_t,3)}\\times{_fmt(v_dc,3)})")
                    num_t_val += float(v_t) * float(v_dc)
                    den_t_val += float(v_dc)
        _latex(r"T_{[[\mathrm{mod,site}]]}=\frac{\sum_i T_{[[\mathrm{mod},i]]}\cdot DC_i}{\sum_i DC_i}")
        _latex(
            r"=\frac{<num_expand>}{<den>}=\boxed{<tmod>\ \mathrm{^\circ C}}",
            num_expand=" + ".join(terms_t) if terms_t else "—",
            den=_fmt(den_t_val, 3),
            tmod=_fmt(t_mod_site_i, 3)
        )

        st.markdown("#### AC_site_resolved (kWh) — fallback logic")
        _latex(
            r"\mathrm{AC}_{[[\mathrm{site}]]}^{[[\mathrm{resolved}]]}="
            r"\begin{cases}"
            r"AC_{[[\mathrm{revenue}]]} & \text{if available and finite} \\"
            r"AC_{[[\mathrm{check}]]} & \text{elif available and finite} \\"
            r"\sum_i AC_i & \text{otherwise}"
            r"\end{cases}"
        )

        ac_rev_s = _fmt(ac_rev_i, 6)
        ac_chk_s = _fmt(ac_chk_i, 6)
        ac_sum_s = _fmt(ac_sum_i, 6)
        ac_res_s = _fmt(ac_res_i, 6)

        if ac_src_i == "AC_revenue":
            _latex(
                r"\Rightarrow\ AC_{[[\mathrm{revenue}]]} = <rev>\ \mathrm{kWh}\ "
                r"(\mathrm{avail}=<a1>)\ \Rightarrow\ \boxed{<res>\ \mathrm{kWh}}",
                rev=ac_rev_s, a1=str(av_rev_i), res=ac_res_s
            )
        elif ac_src_i == "AC_check":
            _latex(
                r"\Rightarrow\ AC_{[[\mathrm{check}]]} = <chk>\ \mathrm{kWh}\ "
                r"(\mathrm{avail}=<a2>)\ \Rightarrow\ \boxed{<res>\ \mathrm{kWh}}",
                chk=ac_chk_s, a2=str(av_chk_i), res=ac_res_s
            )
        else:
            sum_terms = []
            sum_val = 0.0
            if 'ac_mat' in locals() and 0 <= idx_site < ac_mat.shape[1]:
                for i_inv in range(ac_mat.shape[0]):
                    v_ac = ac_mat[i_inv, idx_site]
                    if np.isfinite(v_ac):
                        sum_terms.append(_fmt(v_ac, 6))
                        sum_val += float(v_ac)
            _latex(
                r"\Rightarrow\ \sum_i AC_i\ =\ <terms>\ =\ <sum>\ \mathrm{kWh}\ "
                r"\Rightarrow\ \boxed{<res>\ \mathrm{kWh}}",
                terms=" + ".join(sum_terms) if sum_terms else "—",
                sum=_fmt(sum_val, 6),
                res=ac_res_s
            )

    # KPI Data (Site)
    st.markdown("### KPI Data (Site, per 10-min)")
    temp_factor_site = 1.0 + gamma * (t_mod_site - ref_temp)
    denom_site = ref_irr * temp_factor_site
    y_ref_site = np.where(np.isclose(denom_site, 0.0), np.nan, poa_site / denom_site) * DT_HOURS
    y_ac_site = np.where(dc_total > 0, ac_site_resolved / dc_total, np.nan)
    pr_site_pct = np.where((y_ref_site == 0) | np.isnan(y_ref_site), np.nan, (y_ac_site / y_ref_site) * 100.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        yac_mat = np.vstack([
            np.where(p["dc"] > 0, p["ac_kwh"] / float(p["dc"]), np.nan)
            for p in per_inv
        ])
    median_yac = np.nanmedian(yac_mat, axis=0)

    cap_10min_site = float(st.session_state.get("ac_max_project_kw", ac_max_project_kw)) * DT_HOURS
    e_potential_site = np.minimum(median_yac * dc_total, cap_10min_site)

    lost_site = np.full(len(times), np.nan, dtype=float)
    if "mi_outputs" in st.session_state and isinstance(st.session_state["mi_outputs"], dict):
        rows = []
        for inv_name in ["INV1", "INV2", "INV3"]:
            v = st.session_state["mi_outputs"].get(inv_name, {}).get("Lost", None)
            if isinstance(v, (list, np.ndarray)):
                rows.append(np.asarray(v, dtype=float))
        if rows:
            L = min(map(len, rows))
            arr = np.vstack([r[:L] for r in rows])
            lost_site[:L] = np.nansum(arr, axis=0)

    kpi_site_df = pd.DataFrame({
        "Time": times,
        "Reference Yield (h)": np.round(y_ref_site, 6),
        "Specific Yield (h)": np.round(y_ac_site, 6),
        "Performance Ratio (%)": np.round(pr_site_pct, 3),
        "Median(Y_ac) (h)": np.round(median_yac, 6),
        "Proxy — Reference Potential (kWh)": np.round(e_potential_site, 6),
        "Lost Energy (kWh)": np.round(lost_site, 3),
    })

    st.data_editor(
        kpi_site_df,
        key="kpi_site_table",
        use_container_width=True,
        disabled=True,
        column_config={
            "Time": st.column_config.TextColumn("Time (hh:mm)", help=HOVER["Time"]),
            "Reference Yield (h)": st.column_config.NumberColumn("Reference Yield (h)", format="%.3f", help=HOVER["Reference Yield (h)"]),
            "Specific Yield (h)": st.column_config.NumberColumn("Specific Yield (h)", format="%.3f", help=HOVER["Specific Yield (h)"]),
            "Performance Ratio (%)": st.column_config.NumberColumn("Performance Ratio (%)", format="%.2f", help=HOVER["Performance Ratio (%)"]),
            "Median(Y_ac) (h)": st.column_config.NumberColumn("Median(Y_ac) (h)", format="%.3f", help=HOVER["Median(Y_ac) (h)"]),
            "Proxy — Reference Potential (kWh)": st.column_config.NumberColumn("Proxy — Reference Potential (kWh)", format="%.3f", help=HOVER["Proxy — Reference Potential (kWh)"]),
            "Lost Energy (kWh)": st.column_config.NumberColumn("Lost Energy (kWh)", format="%.3f", help=HOVER["Lost Energy (kWh)"]),
        }
    )
# --- KPI numeric details (Site) — one-line formulas per 10-min (self-contained) ---
    with st.expander("KPI numeric details — one-line formulas per 10-min (Site)", expanded=False):
        st.caption("One-line numeric substitutions for Y_ref_site, SpecificYield_site, PR_site, Site Median(Y_ac), Proxy Potential, gap and Lost (site).")

        # Threshold for site median filtering
        site_thr = float(st.session_state.get("site_median_min_yac", 0.05))

        # Ensure dc_total exists
        try:
            dc_tot_val = float(dc_total)
        except NameError:
            try:
                dc_tot_val = float(np.nansum(dc_arr))
            except Exception:
                dc_tot_val = float("nan")

        # Ensure AC max (project) exists
        try:
            acmax_proj_kw = float(st.session_state.get("ac_max_project_kw", ac_max_project_kw))
        except NameError:
            acmax_proj_kw = float(st.session_state.get("ac_max_project_kw", 1800.0))

        # Build per-inverter Y_ac and FullCap matrices from per_inv/invs
        try:
            _per_inv_local = list(per_inv)
            _invs_local = list(invs)
        except NameError:
            _invs_local = [("INV1","inv1"),("INV2","inv2"),("INV3","inv3")]
            _per_inv_local = []
            for name, sfx in _invs_local:
                df_src = _get_or_init_src(name, sfx)
                dc_tmp = _get_dc_scalar(sfx, 700.0)
                poa_met = pd.to_numeric(df_src.get("MET_POA_Wm2", np.nan), errors="coerce").astype(float).to_numpy()
                poa_sat = pd.to_numeric(df_src.get("SAT_POA_Wm2", np.nan), errors="coerce").astype(float).to_numpy()
                met_av  = df_src.get("MET_POA_available", False).fillna(False).astype(bool).to_numpy()
                poa_used = np.where(met_av & np.isfinite(poa_met), poa_met, poa_sat)
                a_local = float(st.session_state.get("king_a", -3.56))
                b_local = float(st.session_state.get("king_b", -0.075))
                t_mod_used, _ = _solve_tmod(df_src, a_local, b_local)
                eac = pd.to_numeric(df_src.get("AC_kWh", np.nan), errors="coerce").astype(float).to_numpy()
                _per_inv_local.append({
                    "name": name, "dc": dc_tmp, "poa_used": poa_used, "t_mod_used": t_mod_used,
                    "ac_kwh": eac, "time": df_src.get("Time", pd.Series([f"{i:02d}:00" for i in range(len(eac))])).astype(str).tolist()
                })
            if 'times' not in locals():
                times = _per_inv_local[0]["time"]

        try:
            times_vec = list(times)
        except NameError:
            times_vec = _per_inv_local[0]["time"]

        yac_list = []
        fc_list  = []
        names    = []
        for (name, sfx), p in zip(_invs_local, _per_inv_local):
            names.append(name)
            dc_i = float(p["dc"])
            ac_i = np.asarray(p["ac_kwh"], dtype=float)
            if dc_i > 0:
                yac_vec = ac_i / dc_i
            else:
                yac_vec = np.full_like(ac_i, np.nan, dtype=float)
            df_src = _get_or_init_src(name, sfx)
            fc_vec = df_src.get("FullCap", False)
            if not isinstance(fc_vec, pd.Series):
                fc_vec = pd.Series([False]*len(ac_i))
            fc_vec = fc_vec.fillna(False).astype(bool).to_numpy()

            yac_list.append(np.asarray(yac_vec, dtype=float))
            fc_list.append(np.asarray(fc_vec, dtype=bool))

        yac_mat_all = np.vstack(yac_list) if yac_list else np.zeros((0,0))
        fc_mat_all  = np.vstack(fc_list)  if fc_list  else np.zeros((0,0), dtype=bool)

        # Compute site-filtered median (FullCap + threshold)
        n_rows = yac_mat_all.shape[1] if yac_mat_all.size else 0
        median_yac_site_local = np.full(n_rows, np.nan, dtype=float)
        for t in range(n_rows):
            cands = []
            for i in range(yac_mat_all.shape[0]):
                v = float(yac_mat_all[i, t])
                ok = bool(fc_mat_all[i, t]) and np.isfinite(v) and (v > site_thr)
                if ok: cands.append(v)
            if cands:
                median_yac_site_local[t] = float(np.median(cands))

        # Fallbacks to existing variables if available (fixed: no self-reference)
        if 'median_yac' in locals():
            median_yac_site = median_yac
        elif 'median_yac_site_local' in locals() and np.size(median_yac_site_local):
            median_yac_site = median_yac_site_local
        else:
            median_yac_site = np.asarray(times_vec, dtype=float) * np.nan

        # Ensure e_potential_site vector locally
        try:
            e_potential_site_vec = e_potential_site
        except NameError:
            e_potential_site_vec = np.minimum(median_yac_site * dc_tot_val, float(acmax_proj_kw) * DT_HOURS)

        # Ensure lost_site vector locally (sum per-inverter Lost if available)
        try:
            lost_site_vec = lost_site
        except NameError:
            lost_site_vec = np.full(len(times_vec), np.nan, dtype=float)
            if "mi_outputs" in st.session_state and isinstance(st.session_state["mi_outputs"], dict):
                rows = []
                for nm in [n for n,_ in _invs_local]:
                    v = st.session_state["mi_outputs"].get(nm, {}).get("Lost", None)
                    if isinstance(v, (list, np.ndarray)):
                        rows.append(np.asarray(v, dtype=float))
                if rows:
                    L = min(map(len, rows))
                    arr = np.vstack([r[:L] for r in rows])
                    lost_site_vec[:L] = np.nansum(arr, axis=0)

        labels_site_details = [f"{i:02d} — {t}" for i, t in enumerate(times_vec)]
        idx_site_detail = st.selectbox(
            "Site — interval (numeric details)",
            options=list(range(len(labels_site_details))),
            format_func=lambda i: labels_site_details[i],
            key="selbox_site_numeric_details"
        )

        def _pick(arr):
            return float(arr[idx_site_detail]) if (isinstance(arr, (list, np.ndarray)) and 0 <= idx_site_detail < len(arr) and np.isfinite(arr[idx_site_detail])) else float("nan")

        poa_i    = _pick(poa_site)            if 'poa_site' in locals()            else float("nan")
        tmod_i   = _pick(t_mod_site)          if 't_mod_site' in locals()          else float("nan")
        yref_i   = _pick(y_ref_site)          if 'y_ref_site' in locals()          else float("nan")
        yac_i    = _pick(y_ac_site)           if 'y_ac_site' in locals()           else float("nan")
        pr_i     = _pick(pr_site_pct)         if 'pr_site_pct' in locals()         else float("nan")
        med_i    = _pick(median_yac_site)
        ep_i     = _pick(e_potential_site_vec)
        ac_res_i = _pick(ac_site_resolved)    if 'ac_site_resolved' in locals()    else float("nan")
        gap_i    = (max(0.0, ep_i - (ac_res_i if np.isfinite(ac_res_i) else 0.0))
                    if (np.isfinite(ep_i)) else float("nan"))
        lost_i   = _pick(lost_site_vec)

        def _safe_fmt(x, n):
            try: return _fmt(x, n)
            except Exception: return "NaN"

        try:
            ref_irr_s, ref_temp_s, gamma_sci, *_ = _get_site_params()
        except Exception:
            ref_irr_s, ref_temp_s, gamma_sci = 1000.0, 25.0, -0.0035

        poa_s   = _safe_fmt(poa_i, 3)
        refirr  = _safe_fmt(ref_irr_s, 3)
        gamma_s = _safe_fmt(gamma_sci, 6)
        tmod_s  = _safe_fmt(tmod_i, 3)
        tref_s  = _safe_fmt(ref_temp_s, 3)
        yref_s  = _safe_fmt(yref_i, 6)
        yac_s   = _safe_fmt(yac_i, 6)
        pr_s    = _safe_fmt(pr_i, 3)
        med_s   = _safe_fmt(med_i, 6)
        dc_tot_s= _safe_fmt(dc_tot_val, 3)
        acmax_s = _safe_fmt(acmax_proj_kw, 3)
        ep_s    = _safe_fmt(ep_i, 3)
        acres_s = _safe_fmt(ac_res_i, 3)
        gap_s   = _safe_fmt(gap_i, 3)
        lost_s  = _safe_fmt(lost_i, 3)

        recap_df = pd.DataFrame([{
            "POA_site (W/m²)": poa_s,
            "T_mod_site (°C)": tmod_s,
            "DC_total (kWp)": dc_tot_s,
            "AC_site_resolved (kWh)": acres_s,
            "Median(Y_ac) (h)": med_s,
            "AC Max Power (project) [kW]": acmax_s,
        }])
        st.data_editor(
            recap_df, key="site_numeric_inputs_recap",
            use_container_width=True, disabled=True, hide_index=True
        )

        # 1) Reference Yield (h) — Site
        st.markdown("#### Reference Yield (h) — Site")
        _latex(r"Y_{[[\mathrm{ref,site}]]}(h)=\frac{[[\mathrm{POA_{site}}]]}{[[\mathrm{RefIrr}]]\cdot\left(1+\gamma\cdot(T_{[[\mathrm{mod,site}]]}-T_{[[\mathrm{ref}]]})\right)}\cdot\frac{1}{6}")
        _latex(
            r"=\frac{<poa>}{<refirr>\cdot\left(1+<gamma>\cdot(<tmod>-<tref>)\right)}\cdot\frac{1}{6}=<yref>\ \mathrm{h}",
            poa=poa_s, refirr=refirr, gamma=gamma_s, tmod=tmod_s, tref=tref_s, yref=yref_s
        )

        # 2) Specific Yield (h) — Site
        st.markdown("#### Specific Yield (h) — Site")
        _latex(r"Y_{[[\mathrm{ac,site}]]}(h)=\frac{E_{[[\mathrm{ac,site}}]]^{[[\mathrm{resolved}]]}(\mathrm{kWh})}{DC_{[[\mathrm{total}]]}(\mathrm{kWp})}")
        _latex(r"=\frac{<eac>}{<dc>}= <yac>\ \mathrm{h}", eac=acres_s, dc=dc_tot_s, yac=yac_s)

        # 3) Performance Ratio (%) — Site
        st.markdown("#### Performance Ratio (%) — Site")
        _latex(r"\mathrm{PR}_{[[\mathrm{site}]]}(\%)=\frac{Y_{[[\mathrm{ac,site}]]}}{Y_{[[\mathrm{ref,site}]]}}\times 100")
        _latex(r"=\left(\frac{<yac>}{<yref>}\right)\times 100=<pr>\ \%", yac=yac_s, yref=yref_s, pr=pr_s)

        # 4) Median(Y_ac) — candidates and numeric derivation (Site)
        st.markdown("#### Median(Y_ac) — candidates and numeric derivation (Site)")
        rows = []
        used_vals = []
        for i, name in enumerate([p["name"] for p in per_inv]):
            v = float(yac_mat[i, idx_site_detail]) if (yac_mat.size and 0 <= idx_site_detail < yac_mat.shape[1] and np.isfinite(yac_mat[i, idx_site_detail])) else float("nan")
            df_src = _get_or_init_src(name, name.lower())
            fc_col = df_src.get("FullCap", False)
            if not isinstance(fc_col, pd.Series):
                fc_col = pd.Series([False]*yac_mat.shape[1])
            fc_ok = bool(fc_col.fillna(False).astype(bool).to_numpy()[idx_site_detail]) if yac_mat.size else False
            v_ok  = np.isfinite(v) and (v > site_thr)
            used  = fc_ok and v_ok
            if used: used_vals.append(v)
            rows.append({
                "Inverter": name,
                "Y_ac (h)": _fmt(v, 6),
                "FullCap": "True" if fc_ok else "False",
                f"> {site_thr:.2f} h": "True" if v_ok else "False",
                "Used in median": "Yes" if used else "No",
            })
        st.data_editor(pd.DataFrame(rows), key="site_median_candidates_table",
                       use_container_width=True, disabled=True, hide_index=True)

        if len(used_vals) == 0:
            _latex(r"\mathrm{Median}(Y_{[[\mathrm{ac}]]})(h)=\text{No valid inputs}\ \Rightarrow\ \text{NaN}")
        else:
            sorted_vals = sorted(used_vals)
            n_used = len(sorted_vals)
            lst = ", ".join(_fmt(v, 6) for v in sorted_vals)
            if n_used % 2 == 1:
                vmid = sorted_vals[n_used // 2]
                _latex(r"\mathrm{Median}(Y_{[[\mathrm{ac}]]})(h)=\mathrm{median}\!\Big([<lst>]\Big)=<vmid>\ \mathrm{h}",
                       lst=lst, vmid=_fmt(vmid, 6))
            else:
                k1 = n_used // 2 - 1
                k2 = n_used // 2
                v1, v2 = sorted_vals[k1], sorted_vals[k2]
                avg = (v1 + v2) / 2.0
                _latex(r"\mathrm{Median}(Y_{[[\mathrm{ac}]]})(h)=\mathrm{median}\!\Big([<lst>]\Big)=\frac{<v1>+<v2>}{2}=<avg>\ \mathrm{h}",
                       lst=lst, v1=_fmt(v1,6), v2=_fmt(v2,6), avg=_fmt(avg,6))

        st.caption(f"Site Median(Y_ac) used at this row: {_fmt(med_i,6)} h")

        # 5) Proxy — Reference Potential (kWh) — Site
        st.markdown("#### Proxy — Reference Potential (kWh) — Site")
        _latex(r"E_{[[\mathrm{potential,site}]]}(\mathrm{kWh})=\min\!\Big(\mathrm{Median}(Y_{[[\mathrm{ac}}]])(h)\times DC_{[[\mathrm{total}]]},\ AC_{\max,[[\mathrm{project}]]}\cdot\frac{1}{6}\Big)")
        _latex(r"=\min\!\Big(<med>\times <dc>,\ <acmax>\cdot\frac{1}{6}\Big)=<ep>\ \mathrm{kWh}",
               med=med_s, dc=dc_tot_s, acmax=acmax_s, ep=ep_s)


        # 6) Lost Energy (kWh) — Site (Σ inverter Lost)
        st.markdown("#### Lost Energy (kWh) ")
        _latex(r"\mathrm{Lost}_{[[\mathrm{site}]]}=\sum_i \mathrm{Lost}_i \quad\text{(when per-inverter Lost is available)}")
        _latex(r"=\ <lost>\ \mathrm{kWh}", lost=lost_s)

    # Daily Totals & Aggregated KPIs (Site)
    st.markdown("### Daily Totals & Aggregated KPIs (Site)")

    total_eac_site   = float(np.nansum(ac_site_resolved)) if 'ac_site_resolved' in locals() else np.nan
    total_pot_site   = float(np.nansum(e_potential_site)) if 'e_potential_site' in locals() else np.nan
    total_lost_site  = float(np.nansum(lost_site))        if 'lost_site' in locals() else np.nan

    y_ref_day_site   = float(np.nansum(y_ref_site)) if 'y_ref_site' in locals() else np.nan
    dc_total_site    = float(dc_total) if 'dc_total' in locals() else np.nan
    y_ac_day_site    = (total_eac_site / dc_total_site) if (np.isfinite(dc_total_site) and dc_total_site > 0) else np.nan
    pr_day_site      = (y_ac_day_site / y_ref_day_site) if (np.isfinite(y_ref_day_site) and y_ref_day_site != 0) else np.nan

    all_pba_site     = ((total_pot_site - total_lost_site) / total_pot_site * 100.0) if (np.isfinite(total_pot_site) and total_pot_site > 0) else np.nan

    totals_site = pd.DataFrame([{
        "Σ E_ac (kWh)": total_eac_site if np.isfinite(total_eac_site) else np.nan,
        "Σ Potential (kWh)": total_pot_site if np.isfinite(total_pot_site) else np.nan,
        "Σ Lost (kWh)": total_lost_site if np.isfinite(total_lost_site) else np.nan,
        "Reference Yield — day (h)": y_ref_day_site if np.isfinite(y_ref_day_site) else np.nan,
        "Specific Yield — day (h)": y_ac_day_site if np.isfinite(y_ac_day_site) else np.nan,
        "Performance Ratio — day (%)": (pr_day_site*100) if np.isfinite(pr_day_site) else np.nan,
        "All PBA (%)": all_pba_site if np.isfinite(all_pba_site) else np.nan,
    }])

    st.data_editor(
        totals_site, key="totals_site",
        use_container_width=True, disabled=True,
        column_config={
            "Σ E_ac (kWh)": st.column_config.NumberColumn("Σ E_ac (kWh)", format="%.3f", help=HOVER["Σ E_ac (kWh)"]),
            "Σ Potential (kWh)": st.column_config.NumberColumn("Σ Potential (kWh)", format="%.3f", help=HOVER["Σ Potential (kWh)"]),
            "Σ Lost (kWh)": st.column_config.NumberColumn("Σ Lost (kWh)", format="%.3f", help=HOVER["Σ Lost (kWh)"]),
            "Reference Yield — day (h)": st.column_config.NumberColumn("Reference Yield — day (h)", format="%.3f", help=HOVER["Reference Yield — day (h)"]),
            "Specific Yield — day (h)": st.column_config.NumberColumn("Specific Yield — day (h)", format="%.3f", help=HOVER["Specific Yield — day (h)"]),
            "Performance Ratio — day (%)": st.column_config.NumberColumn("Performance Ratio — day (%)", format="%.2f", help=HOVER["Performance Ratio — day (%)"]),
            "All PBA (%)": st.column_config.NumberColumn("All PBA (%)", format="%.2f", help=HOVER["All PBA (%)"]),
        }
    )
# --- Daily Totals — numeric details (one-line formulas), same style as inverter ---
    with st.expander("Daily Totals — numeric details (one-line formulas) — Site", expanded=False):
        eac_terms   = [v for v in (ac_site_resolved if 'ac_site_resolved' in locals() else []) if np.isfinite(v)]
        yref_terms  = [v for v in (y_ref_site if 'y_ref_site' in locals() else []) if np.isfinite(v)]
        epot_terms  = [v for v in (e_potential_site if 'e_potential_site' in locals() else []) if np.isfinite(v)]
        lost_terms  = [v for v in (lost_site if 'lost_site' in locals() else []) if np.isfinite(v)]

        def _join(vals, n=3):
            return " + ".join(_fmt(v, n) for v in vals) if len(vals) else "—"

        total_eac_s   = _fmt(total_eac_site, 3)
        total_pot_s   = _fmt(total_pot_site, 3)
        total_lost_s  = _fmt(total_lost_site, 3)
        y_ref_day_s   = _fmt(y_ref_day_site, 6)
        y_ac_day_s    = _fmt(y_ac_day_site, 3)
        pr_day_pct    = (float(y_ac_day_s) / float(y_ref_day_s) * 100.0) if (y_ref_day_s not in ["NaN","nan"] and y_ac_day_s not in ["NaN","nan"] and float(y_ref_day_s) != 0) else float("nan")
        pr_day_s      = _fmt(pr_day_pct, 3)
        dc_tot_s      = _fmt(dc_total_site, 1)
        all_pba_s     = _fmt(((float(total_pot_s) - float(total_lost_s)) / float(total_pot_s) * 100.0) if (total_pot_s not in ["NaN","nan"] and float(total_pot_s)>0) else np.nan, 3)

        st.markdown("#### 1) ΣE_ac (kWh)")
        _latex(r"\sum E_{[[\mathrm{ac,site}]]}(\mathrm{kWh})")
        _latex(r"= <terms> = <total>\ \mathrm{kWh}", terms=_join(eac_terms, n=1), total=total_eac_s)

        st.markdown("#### 2) DC total (kWp)")
        _latex(r"DC_{[[\mathrm{total,site}]]}(\mathrm{kWp}) = <dc>", dc=dc_tot_s)

        st.markdown("#### 3) SpecificYield_day (h)")
        _latex(r"Y_{[[\mathrm{ac,day}}]}(h) = \frac{\sum E_{[[\mathrm{ac,site}]]}}{DC_{[[\mathrm{total,site}]]}}")
        _latex(r"= <sum_eac> \div <dc> \ \to\ <yac_day>\ \mathrm{h}", sum_eac=total_eac_s, dc=dc_tot_s, yac_day=y_ac_day_s)

        st.markdown("#### 4) ΣReferenceYield (h)")
        _latex(r"\sum Y_{[[\mathrm{ref,site}]]}(h)")
        _latex(r"= <terms> = <total>\ \mathrm{h}", terms=_join(yref_terms, n=6), total=y_ref_day_s)

        st.markdown("#### 5) PR_day (%)")
        _latex(r"\mathrm{PR}_{[[\mathrm{day,site}]]}(\%) = \frac{Y_{[[\mathrm{ac,day}}]}}{Y_{[[\mathrm{ref,day}}]}}\times 100")
        _latex(r"= \left(<yac_day> \div <yref_day>\right) \times 100 \ \to\ <pr_day>\ \%", yac_day=y_ac_day_s, yref_day=y_ref_day_s, pr_day=pr_day_s)

        st.markdown("#### 6) ΣE_potential (kWh)")
        _latex(r"\sum E_{[[\mathrm{potential,site}]]} = <terms> = <total>\ \mathrm{kWh}", terms=_join(epot_terms, n=1), total=total_pot_s)

        st.markdown("#### 7) ΣLost Energy (kWh)")
        _latex(r"\sum \mathrm{Lost}_{[[\mathrm{site}]]} = <terms> = <total>\ \mathrm{kWh}", terms=_join(lost_terms, n=1), total=total_lost_s)

        st.markdown("#### 8) All PBA (%)")
        _latex(r"\mathrm{All\ PBA}(\%)=\frac{\sum E_{[[\mathrm{potential}]]}-\sum \mathrm{Lost}}{\sum E_{[[\mathrm{potential}]]}}\times 100")
        _latex(r"=\frac{<sum_pot>-<sum_lost>}{<sum_pot>}\times 100 \ \to\ <pba>\ \%", sum_pot=total_pot_s, sum_lost=total_lost_s, pba=all_pba_s)

    # Global Daily Energy Summary (Site)
    st.markdown("### Global Daily Energy Summary (Site)")
    gross_prod_site = total_eac_site if np.isfinite(total_eac_site) else np.nan
    poi_ref_site    = gross_prod_site * 0.97 if np.isfinite(gross_prod_site) else np.nan
    all_losses_site = total_lost_site if np.isfinite(total_lost_site) else np.nan
    all_pba_val_site= all_pba_site if np.isfinite(all_pba_site) else np.nan
    all_pr_stc_site = (pr_day_site * 100.0) if np.isfinite(pr_day_site) else np.nan

    global_summary_site = pd.DataFrame([{
        "Gross inverter production (kWh)": np.round(gross_prod_site, 3),
        "Reference POI Production (kWh)": np.round(poi_ref_site, 3),
        "All losses (kWh)": np.round(all_losses_site, 3),
        "All PBA (%)": np.round(all_pba_val_site, 2),
        "All PR STC (%)": np.round(all_pr_stc_site, 2),
    }])

    st.data_editor(
        global_summary_site,
        key="global_summary_site",
        use_container_width=True,
        disabled=True,
        hide_index=True,
        column_config={
            "Gross inverter production (kWh)": st.column_config.NumberColumn("Gross inverter production (kWh)", format="%.3f", help=HOVER["Gross inverter production (kWh)"]),
            "Reference POI Production (kWh)": st.column_config.NumberColumn("Reference POI Production (kWh)", format="%.3f", help=HOVER["Reference POI Production (kWh)"]),
            "All losses (kWh)": st.column_config.NumberColumn("All losses (kWh)", format="%.3f", help=HOVER["All losses (kWh)"]),
            "All PBA (%)": st.column_config.NumberColumn("All PBA (%))", format="%.2f", help=HOVER["All PBA (%)"]),
            "All PR STC (%)": st.column_config.NumberColumn("All PR STC (%)", format="%.2f", help=HOVER["All PR STC (%)"]),
        }
    )

# ---------------------------
# Public entry
# ---------------------------
def page_intermediate():
    st.title("INTERMEDIATE")

    _about()

    # Project Settings (global)
    st.markdown("### Project Settings")
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        ac_max_kw = st.number_input(
            "AC Max Power (project) [kW]",
            min_value=0.0, value=float(st.session_state.get("ac_max_project_kw", 1800.0)),
            step=10.0, help="Project-level AC capacity used to cap potential per inverter.",
            key="acmax_project_input"
        )
        st.session_state["ac_max_project_kw"] = float(ac_max_kw)
    with c2:
        dc1 = _get_dc_scalar("inv1", 700.0)
        dc2 = _get_dc_scalar("inv2", 700.0)
        dc3 = _get_dc_scalar("inv3", 700.0)
        dc_total_site = float(np.round(dc1 + dc2 + dc3, 3))

        st.number_input(
            "DC Total (site) [kWp]",
            min_value=0.0,
            value=dc_total_site,
            step=0.0,
            disabled=True,
            help="Sum of inverter DC capacities; auto-updates when inverter DC values change."
        )
        st.session_state["dc_total_site_kwp"] = dc_total_site

    _refresh_site_median_cache()

    t1, t2, t3, t4 = st.tabs(["Inverter 1", "Inverter 2", "Inverter 3", "Site"])
    with t1:
        _inverter_page("INV1","inv1")
    with t2:
        _inverter_page("INV2","inv2")
    with t3:
        _inverter_page("INV3","inv3")
    with t4:
        _site_page()
