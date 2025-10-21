# intermediate.py
# --------------------------------------------------------------------------------------
# PV KPI - INTERMEDIATE (no AC fallback) — Per-inverter single DC capacity
# - Project Settings (site DC total, AC Max, Proxy Yield) at the top
# - Each inverter tab has ONE "Inverter DC capacity (kWp)" numeric control (no DC column)
# - KPI computations use this scalar DC (broadcast per row)
# - Resolved Inputs include the scalar DC; parameter-details table explains each parameter
# - Numeric one-line formulas and daily sums included
# - Site page uses site DC total (from Project Settings)
# - Robust to missing columns (no KeyError for Wind_available, etc.)
# --------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st

N_INV = 3  # number of inverters


# ---------- Formatting helpers ----------
def _fmt(x, n=6):
    try:
        x = float(x)
        return f"{np.round(x, n)}" if np.isfinite(x) else "NaN"
    except Exception:
        return "NaN"


def _list_vals(arr, n=3):
    arr = np.asarray(arr, dtype=float)
    return [f"{np.round(v, n)}" for v in arr.tolist() if np.isfinite(v)]


def _latex(template: str, **vals):
    """
    Safe LaTeX builder:
    - Use [[...]] instead of {...} in the template for LaTeX braces.
    - Use <name> placeholders for numeric/text values to inject.
    """
    s = template
    for k, v in vals.items():
        s = s.replace(f"<{k}>", str(v))
    s = s.replace("[[", "{").replace("]]", "}")
    st.latex(s)


# ---------- Robust column access ----------
def _get_bool_col(df: pd.DataFrame, col: str, default=False) -> np.ndarray:
    if col in df.columns:
        return df[col].fillna(default).astype(bool).to_numpy()
    return np.full(len(df), bool(default), dtype=bool)


def _get_num_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()
    return np.full(len(df), np.nan, dtype=float)


# ---------- Shared site parameters ----------
def _get_site_params():
    """Read shared site parameters from session_state with safe defaults."""
    ref_irr = float(st.session_state.get("ref_irr", 1000.0))
    ref_temp = float(st.session_state.get("ref_temp", 25.0))
    gamma = float(st.session_state.get("gamma", -0.0035))
    a = float(st.session_state.get("king_a", -3.56))
    b = float(st.session_state.get("king_b", -0.075))
    ac_max_prj = float(st.session_state.get("ac_max_project_kw", 1800.0))
    pr_nom = float(st.session_state.get("pr_nominal", 0.80))
    return ref_irr, ref_temp, gamma, a, b, ac_max_prj, pr_nom


# ---------- Yields & PR ----------
def _yields_pr(poa_for_kpi, t_mod, ref_irr, ref_temp, gamma, eac, dc_cap_inv_scalar):
    """
    Compute Reference Yield (h), Specific Yield (h), PR for a 10-min interval.
      - poa_for_kpi: 10-min average POA in W/m²
      - eac: kWh per 10-min
      - dc_cap_inv_scalar: kWp (single scalar per inverter)
    """
    DT_HOURS = 1.0 / 6.0  # 10 minutes = 1/6 hour
    temp_factor = 1.0 + gamma * (t_mod - ref_temp)
    denom = ref_irr * temp_factor
    y_ref_ratio = np.where(np.isclose(denom, 0.0), np.nan, poa_for_kpi / denom)
    y_ref = y_ref_ratio * DT_HOURS
    dc = float(dc_cap_inv_scalar)
    y_ac = np.where(dc > 0, eac / dc, np.nan)
    pr = np.where((y_ref == 0) | np.isnan(y_ref), np.nan, y_ac / y_ref)
    return y_ref, y_ac, pr


# ---------- Availability helpers (robust) ----------
def _auto_sync_availability_flags(ed: pd.DataFrame) -> pd.DataFrame:
    """
    Copy of ed where availability flags are forced to False when their value columns are missing or NaN.
    If a flag column is missing, it is created (False).
    """
    ed = ed.copy()

    def _sync(value_col: str, flag_col: str):
        vals = _get_num_col(ed, value_col)
        if flag_col not in ed.columns:
            ed[flag_col] = False
        ed[flag_col] = ed[flag_col].fillna(False).astype(bool)
        ed.loc[~np.isfinite(vals), flag_col] = False

    _sync("MET_POA_Wm2", "MET_POA_available")
    _sync("MET_Amb_C", "MET_amb_available")
    _sync("MET_Back_Module_C", "MET_back_available")
    _sync("MET_WindSpeed", "Wind_available")
    _sync("PVLIB_project_kWh", "PVLIB_available")
    return ed


# ---------- T_mod resolution (robust to missing columns) ----------
def _solve_tmod(df: pd.DataFrame, a: float, b: float):
    """
    T_mod fallback order:
      (1) MET_back
      (2) King(MET,[WS]) if MET_POA_available & MET_amb_available
      (3) SAT
    Robust to missing columns.
    """
    back_av = _get_bool_col(df, "MET_back_available")
    amb_av = _get_bool_col(df, "MET_amb_available")
    met_poa_av = _get_bool_col(df, "MET_POA_available")
    wind_av = _get_bool_col(df, "Wind_available")

    t_back = _get_num_col(df, "MET_Back_Module_C")
    t_amb = _get_num_col(df, "MET_Amb_C")
    t_sat = _get_num_col(df, "SAT_T_mod_C")
    poa_met = _get_num_col(df, "MET_POA_Wm2")
    poa_sat = _get_num_col(df, "SAT_POA_Wm2")
    ws = _get_num_col(df, "MET_WindSpeed")

    poa_ok = met_poa_av & np.isfinite(poa_met)
    ws_ok = np.isfinite(ws)

    n = len(df)
    t_mod = np.full(n, np.nan, dtype=float)
    src = np.array([""] * n, dtype=object)

    # (1) measured back temp
    idx1 = back_av
    if np.any(idx1):
        t_mod[idx1] = t_back[idx1]
        src[idx1] = "MET_back"

    # (2) King(MET,[WS])
    idx2 = (~back_av) & poa_ok & amb_av
    if np.any(idx2):
        # base without wind
        t_k = t_amb + poa_met * np.exp(a)
        use_ws = idx2 & wind_av & ws_ok
        if np.any(use_ws):
            t_k[use_ws] = t_amb[use_ws] + poa_met[use_ws] * np.exp(a + b * ws[use_ws])
        t_mod[idx2] = t_k[idx2]
        src[idx2] = np.where(use_ws[idx2], "King(MET,WS)", "King(MET)")

    # (3) SAT fallback
    idx3 = (~back_av) & (~(poa_ok & amb_av))
    if np.any(idx3):
        t_mod[idx3] = t_sat[idx3]
        src[idx3] = "SAT"

    return t_mod, src


# ---------- Proxy & PVLIB helpers ----------
def _proxy_potential(y_ac_all: dict, fullcap_all: dict, _unused, ac_max_project_kw: float):
    """
    Row-wise Site Median Specific Yield across inverters with FullCap and Y_ac > threshold.
    Returns (h) per row.
    """
    inv_names = list(y_ac_all.keys())
    if not inv_names:
        return np.array([])
    n = len(next(iter(y_ac_all.values())))
    mat = np.full((n, len(inv_names)), np.nan, dtype=float)

    min_yac = float(st.session_state.get("site_median_min_yac", 0.05))  # threshold in h/10-min

    for j, name in enumerate(inv_names):
        ya = y_ac_all[name]
        fc = fullcap_all[name]
        ya_fc = np.where(fc & np.isfinite(ya) & (ya > min_yac), ya, np.nan)
        mat[:, j] = ya_fc

    site_median = np.nanmedian(mat, axis=1)
    # carry-forward last valid
    last = np.nan
    for i in range(n):
        if np.isfinite(site_median[i]):
            last = site_median[i]
        else:
            site_median[i] = last
    # default from sidebar if still NaN
    default_proxy = float(st.session_state.get("proxy_yield_h", np.nan))
    if np.isfinite(default_proxy):
        site_median = np.where(np.isfinite(site_median), site_median, default_proxy)
    return site_median


def _pvlib_potential(project_pvlib_kwh, ac_max_inv_kw, sum_ac_max_kw):
    """PVLIB per inverter share = project_pvlib_kwh × (AC_max_inv / sum_AC_max_inv)."""
    share = np.divide(ac_max_inv_kw, sum_ac_max_kw) if sum_ac_max_kw > 0 else 0.0
    return project_pvlib_kwh * share


def _cap_proxy_met_per_inv(energy_kwh, ac_max_project_kw: float):
    """Cap per inverter for Proxy/Met: AC_max_project/N_inv per 10-min → /6 (kW→kWh)."""
    cap_kwh = (ac_max_project_kw / N_INV) / 6.0
    return np.minimum(energy_kwh, cap_kwh)


def _cap_pvlib_per_inv(energy_kwh, ac_max_inv_kw):
    """Optional per-inverter 10-min cap for PVLIB: AC_max_inv/6."""
    cap_kwh = ac_max_inv_kw / 6.0
    return np.minimum(energy_kwh, cap_kwh)


# ---------- Realistic presets (single AC_kWh, no DC column) ----------
def _realistic_presets(inv_name: str) -> pd.DataFrame:
    """Return a realistic 10-min dataset per inverter so KPIs look real."""
    times = ["10:00", "10:10", "10:20", "10:30", "10:40"]

    if inv_name == "INV1":
        return pd.DataFrame({
            "Time": times,
            "MET_POA_Wm2": [880, 920, 940, 900, 840],
            "MET_POA_available": [True, True, True, True, True],
            "SAT_POA_Wm2": [885, 925, 945, 905, 845],
            "MET_Amb_C": [26.0, 27.0, 27.5, 27.2, 26.8],
            "MET_amb_available": [True, True, True, True, True],
            "MET_Back_Module_C": [np.nan, 49.0, np.nan, 50.0, np.nan],
            "MET_back_available": [False, True, False, True, False],
            "MET_WindSpeed": [2.0, 2.5, 3.2, 2.8, 2.2],
            "Wind_available": [True, True, True, True, True],
            "SAT_T_mod_C": [47.0, 48.5, 49.5, 48.8, 47.9],
            "AC_kWh": [78.0, 82.0, 85.0, 80.0, 75.0],
            "FullCap": [True, True, True, True, True],
            "AC_max_inv_kW": [650, 650, 650, 650, 650],
            "PVLIB_project_kWh": [np.nan] * 5,
            "PVLIB_available": [False] * 5,
            "Proxy_available": [True] * 5,
            "Met_available": [True] * 5,
        })

    if inv_name == "INV2":
        return pd.DataFrame({
            "Time": times,
            "MET_POA_Wm2": [820, 860, 880, 850, 800],
            "MET_POA_available": [True, True, True, True, False],  # last row uses SAT
            "SAT_POA_Wm2": [830, 870, 890, 860, 810],
            "MET_Amb_C": [26.1, 27.1, 27.6, 27.3, 26.9],
            "MET_amb_available": [True, True, True, True, True],
            "MET_Back_Module_C": [np.nan, np.nan, 48.5, np.nan, np.nan],
            "MET_back_available": [False, False, True, False, False],
            "MET_WindSpeed": [2.3, 2.8, 3.0, 2.6, 2.1],
            "Wind_available": [True, True, True, True, True],
            "SAT_T_mod_C": [46.8, 47.9, 48.6, 48.2, 47.5],
            "AC_kWh": [65.0, 70.0, 74.0, 72.0, 68.0],
            "FullCap": [True, True, True, True, True],
            "AC_max_inv_kW": [620, 620, 620, 620, 620],
            "PVLIB_project_kWh": [np.nan] * 5,
            "PVLIB_available": [False] * 5,
            "Proxy_available": [True] * 5,
            "Met_available": [True] * 5,
        })

    return pd.DataFrame({  # INV3
        "Time": times,
        "MET_POA_Wm2": [760, 790, 810, 780, 740],
        "MET_POA_available": [True, True, True, True, True],
        "SAT_POA_Wm2": [765, 795, 815, 785, 745],
        "MET_Amb_C": [25.8, 26.7, 27.2, 27.0, 26.5],
        "MET_amb_available": [True, True, True, True, True],
        "MET_Back_Module_C": [np.nan, np.nan, np.nan, 47.0, np.nan],
        "MET_back_available": [False, False, False, True, False],
        "MET_WindSpeed": [2.0, 2.4, 2.9, 2.5, 2.0],
        "Wind_available": [True, True, True, True, True],
        "SAT_T_mod_C": [46.0, 47.4, 48.0, 47.7, 46.8],
        "AC_kWh": [55.0, 60.0, 63.0, 61.0, 58.0],
        "FullCap": [True, True, False, True, True],
        "AC_max_inv_kW": [600, 600, 600, 600, 600],
        "PVLIB_project_kWh": [np.nan] * 5,
        "PVLIB_available": [False] * 5,
        "Proxy_available": [True] * 5,
        "Met_available": [True] * 5,
    })


# ---------- Inverter page ----------
def _inverter_page(inv_name: str, key_suffix: str):
    ref_irr, ref_temp, gamma, a, b, ac_max_project_kw, pr_nominal = _get_site_params()

    st.subheader(f"{inv_name} — 10-min Concept Data")

    # Inverter-level DC (single scalar control)
    default_dc_map = {"INV1": 700.0, "INV2": 650.0, "INV3": 600.0}
    dc_key = f"dc_cap_inv_kwp_{key_suffix}"
    dc_cap_inv_scalar = float(st.session_state.get(dc_key, default_dc_map.get(inv_name, 650.0)))

    c_dc, = st.columns([1])
    with c_dc:
        dc_cap_inv_scalar = st.number_input(
            "Inverter DC capacity (kWp) — global",
            min_value=0.0,
            value=dc_cap_inv_scalar,
            step=10.0,
            help="Single DC capacity for this inverter, used in Specific Yield and potentials.",
            key=f"{dc_key}_input",
        )
    st.session_state[dc_key] = dc_cap_inv_scalar

    # Source DF in session
    src_key = f"src_{key_suffix}"
    if src_key not in st.session_state:
        st.session_state[src_key] = _realistic_presets(inv_name).copy()

    # Editor (no DC column)
    ed = st.data_editor(
        st.session_state[src_key],
        num_rows="dynamic",
        key=f"ed_{key_suffix}",
        use_container_width=True,
        column_config={
            "Time": st.column_config.TextColumn("Time (hh:mm)"),

            "MET_POA_Wm2": st.column_config.NumberColumn("MET-POA (W/m²)", min_value=0.0, step=1.0),
            "MET_POA_available": st.column_config.CheckboxColumn("MET POA available"),
            "SAT_POA_Wm2": st.column_config.NumberColumn("SAT-POA (W/m²)", min_value=0.0, step=1.0),

            "MET_Amb_C": st.column_config.NumberColumn("MET Amb (°C)", step=0.1),
            "MET_amb_available": st.column_config.CheckboxColumn("MET Amb available"),
            "MET_Back_Module_C": st.column_config.NumberColumn("MET Back Module (°C)", step=0.1),
            "MET_back_available": st.column_config.CheckboxColumn("MET Back available"),
            "MET_WindSpeed": st.column_config.NumberColumn("Wind speed (m/s)", step=0.1),
            "Wind_available": st.column_config.CheckboxColumn("Wind available"),
            "SAT_T_mod_C": st.column_config.NumberColumn("SAT T_mod (°C)", step=0.1),

            "AC_kWh": st.column_config.NumberColumn("AC_kWh (kWh/10-min)", min_value=0.0, step=0.001),

            "FullCap": st.column_config.CheckboxColumn("Full Capacity (for Site Median Specific Yield)"),
            "AC_max_inv_kW": st.column_config.NumberColumn("AC max (kW)", min_value=0.0, step=1.0),

            "PVLIB_project_kWh": st.column_config.NumberColumn("PVLIB project (kWh/10-min)", min_value=0.0, step=0.001),
            "PVLIB_available": st.column_config.CheckboxColumn("PVLIB available"),

            "Proxy_available": st.column_config.CheckboxColumn("Proxy available"),
            "Met_available": st.column_config.CheckboxColumn("Met available"),
        },
    )

    if ed.empty:
        st.info("No rows to compute.")
        return

    # Sync flags with value presence
    synced = _auto_sync_availability_flags(ed)

    flag_cols = [
        "MET_POA_available", "MET_amb_available", "MET_back_available",
        "Wind_available", "PVLIB_available"
    ]
    flag_cols = [c for c in flag_cols if c in synced.columns and c in st.session_state[src_key].columns]

    flags_changed = not synced[flag_cols].reset_index(drop=True).equals(
        st.session_state[src_key][flag_cols].reset_index(drop=True)
    )
    if flags_changed:
        st.session_state[src_key] = synced
        st.rerun()

    ed = synced  # use synced DF

    # Resolve inputs (no AC fallback; single AC_kWh)
    eac = _get_num_col(ed, "AC_kWh")
    t_mod, t_src = _solve_tmod(ed, a, b)

    poa_met = _get_num_col(ed, "MET_POA_Wm2")
    poa_sat = _get_num_col(ed, "SAT_POA_Wm2")
    met_poa_av = _get_bool_col(ed, "MET_POA_available")
    poa_met_ok = met_poa_av & np.isfinite(poa_met)
    poa_for_kpi = np.where(poa_met_ok, poa_met, poa_sat)
    poa_source = np.where(poa_met_ok, "MET_POA", "SAT_POA")

    ac_max_inv = _get_num_col(ed, "AC_max_inv_kW")
    fullcap = _get_bool_col(ed, "FullCap")

    # KPIs (scalar DC)
    y_ref, y_ac_inv, _ = _yields_pr(poa_for_kpi, t_mod, ref_irr, ref_temp, gamma, eac, dc_cap_inv_scalar)
    pr_pct = np.where((y_ref == 0) | np.isnan(y_ref), np.nan, (y_ac_inv / y_ref) * 100.0)

    # share Y_ac/fullcap for site-median proxy
    st.session_state.setdefault("mi_yac", {})
    st.session_state.setdefault("mi_fullcap", {})
    st.session_state["mi_yac"][inv_name] = y_ac_inv
    st.session_state["mi_fullcap"][inv_name] = fullcap

    yac_all = st.session_state["mi_yac"]
    fc_all = st.session_state["mi_fullcap"]

    # site median specific yield excluding current inverter
    others_yac = {k: v for k, v in yac_all.items() if k != inv_name}
    others_fc = {k: v for k, v in fc_all.items() if k != inv_name}
    if len(others_yac) >= 1:
        site_median_yac = _proxy_potential(others_yac, others_fc, None, ac_max_project_kw)
    else:
        n_rows = len(y_ac_inv)
        default_proxy = float(st.session_state.get("proxy_yield_h", np.nan))
        site_median_yac = np.full(n_rows, default_proxy, dtype=float)

    # potentials (scalar DC)
    e_proxy_raw = site_median_yac * dc_cap_inv_scalar
    e_proxy = _cap_proxy_met_per_inv(e_proxy_raw, ac_max_project_kw)

    e_met_raw = y_ref * pr_nominal * dc_cap_inv_scalar
    e_met = _cap_proxy_met_per_inv(e_met_raw, ac_max_project_kw)

    pvlib_prj = _get_num_col(ed, "PVLIB_project_kWh")
    pvlib_av = _get_bool_col(ed, "PVLIB_available")
    sum_acmax = float(np.nansum(ac_max_inv)) if np.isfinite(np.nansum(ac_max_inv)) else 0.0
    e_pvlib_raw = _pvlib_potential(pvlib_prj, ac_max_inv, sum_acmax)
    e_pvlib = _cap_pvlib_per_inv(e_pvlib_raw, ac_max_inv)

    proxy_av = _get_bool_col(ed, "Proxy_available")
    met_av = _get_bool_col(ed, "Met_available")

    potential = np.where(
        proxy_av, e_proxy,
        np.where(pvlib_av, e_pvlib,
                 np.where(met_av, e_met, np.nan))
    )
    pot_src = np.where(
        proxy_av, "Proxy",
        np.where(pvlib_av, "PVLIB",
                 np.where(met_av, "Met", "None"))
    )

    # Resolved Inputs
    st.markdown("### Resolved Inputs")
    resolved = pd.DataFrame({
        "Time": ed["Time"],
        "POA_used (W/m²)": np.round(poa_for_kpi, 1),
        "POA_Source": poa_source,
        "T_mod_used (°C)": np.round(t_mod, 3),
        "T_mod_Source": t_src,
        "AC_kWh": np.round(eac, 6),
        "Inverter DC_cap (kWp) — global": np.full(len(ed), dc_cap_inv_scalar, dtype=float),
        "FullCap": fullcap,
        "AC_max_inv (kW)": ac_max_inv,
    })
    st.data_editor(
        resolved, key=f"resolved_{key_suffix}", use_container_width=True, disabled=True,
        column_config={
            "POA_used (W/m²)": st.column_config.NumberColumn("POA_used (W/m²)", help="MET_POA if available and valid; otherwise SAT_POA."),
            "POA_Source": st.column_config.TextColumn("POA_Source", help="Which POA stream was used."),
            "T_mod_used (°C)": st.column_config.NumberColumn("T_mod_used (°C)", help="Module temperature used for KPIs."),
            "T_mod_Source": st.column_config.TextColumn("T_mod_Source", help="Source: MET_back / King(MET,WS) / King(MET) / SAT."),
            "AC_kWh": st.column_config.NumberColumn("AC_kWh", help="AC energy per 10-min (single input)."),
            "Inverter DC_cap (kWp) — global": st.column_config.NumberColumn("Inverter DC_cap (kWp) — global", help="Single DC capacity used across all rows."),
            "FullCap": st.column_config.CheckboxColumn("FullCap", help="If True, row contributes to Site Median for Proxy potential."),
            "AC_max_inv (kW)": st.column_config.NumberColumn("AC_max_inv (kW)", help="Inverter AC nameplate; used in PVLIB capping."),
        }
    )

    # Parameter details for one row
    with st.expander("Resolved Inputs — parameter details", expanded=False):
        times = ed["Time"].tolist()
        labels = [f"{i:02d} — {t}" for i, t in enumerate(times)]
        idx_detail = st.selectbox(
            "Select the 10-min row for details",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            key=f"resolved_detail_row_{key_suffix}"
        )

        poa_val = float(poa_for_kpi[idx_detail]) if np.isfinite(poa_for_kpi[idx_detail]) else np.nan
        poa_src = str(poa_source[idx_detail])
        tmod_val = float(t_mod[idx_detail]) if np.isfinite(t_mod[idx_detail]) else np.nan
        tmod_src = str(t_src[idx_detail])
        eac_val = float(eac[idx_detail]) if np.isfinite(eac[idx_detail]) else np.nan
        fullcap_v = bool(fullcap[idx_detail])
        acmax_val = float(ac_max_inv[idx_detail]) if np.isfinite(ac_max_inv[idx_detail]) else np.nan

        poa_desc = (
            "We use MET station data because it is available and valid."
            if poa_src == "MET_POA"
            else "We use SAT POA fallback because MET POA is unavailable or invalid."
        )
        tmod_desc_map = {
            "MET_back": "Module back-sensor temperature is available; it is used directly.",
            "King(MET,WS)": "Temperature estimated with King correlation using MET POA, ambient and wind speed.",
            "King(MET)": "Temperature estimated with King correlation using MET POA and ambient (no wind).",
            "SAT": "SAT temperature is used as a fallback because measured/correlated inputs are unavailable."
        }
        tmod_desc = tmod_desc_map.get(tmod_src, "Temperature source could not be resolved.")

        details_df = pd.DataFrame([
            {"Parameter": "POA_used (W/m²)", "Value": _fmt(poa_val, 3),
             "Description": f"{poa_desc} (Source: {poa_src})."},
            {"Parameter": "T_mod_used (°C)", "Value": _fmt(tmod_val, 3),
             "Description": f"{tmod_desc} (Source: {tmod_src})."},
            {"Parameter": "AC_kWh", "Value": _fmt(eac_val, 6),
             "Description": "Single AC energy input for this interval (no revenue/check fallback)."},
            {"Parameter": "Inverter DC_cap (kWp) — global", "Value": _fmt(dc_cap_inv_scalar, 3),
             "Description": "Single DC capacity for this inverter; used in Specific Yield = E_ac / DC_cap."},
            {"Parameter": "FullCap", "Value": str(fullcap_v),
             "Description": ("This row contributes to the site median specific yield used by the Proxy potential."
                             if fullcap_v else
                             "This row is excluded from the site median specific yield (Proxy potential).")},
            {"Parameter": "AC_max_inv (kW)", "Value": _fmt(acmax_val, 3),
             "Description": "Inverter AC nameplate (kW); used for PVLIB capping at AC_max_inv/6 per 10-min."},
        ])

        st.data_editor(
            details_df,
            key=f"resolved_param_details_{key_suffix}",
            use_container_width=True,
            disabled=True,
            hide_index=True,
            column_config={
                "Parameter": st.column_config.TextColumn("Parameter", help="Name of the resolved parameter."),
                "Value": st.column_config.TextColumn("Value", help="Numeric value used for this 10-min interval."),
                "Description": st.column_config.TextColumn("Description", help="Why/how this value was selected."),
            },
        )

    # KPI table
    st.markdown("### KPI Data (per 10-min)")
    kpi = pd.DataFrame({
        "Time": ed["Time"],
        "Reference Yield (h at STC)": np.round(y_ref, 6),
        "Specific Yield (h = kWh/kWp)": np.round(y_ac_inv, 6),
        "Performance Ratio (% at STC)": np.round(pr_pct, 3),
        "Site Median Specific Yield (h)": np.round(site_median_yac, 6),
        "Proxy (kWh)": np.round(e_proxy, 6),
        "PVLIB (kWh)": np.round(e_pvlib, 6),
        "Met (kWh)": np.round(e_met, 6),
        "Reference Potential (kWh)": np.round(potential, 6),
        "Potential Source": pot_src,
    })
    st.data_editor(
        kpi, key=f"kpi_{key_suffix}", use_container_width=True, disabled=True,
        column_config={
            "Reference Yield (h at STC)": st.column_config.NumberColumn(
                "Reference Yield (h at STC)", format="%.3f",
                help="POA / [RefIrr × (1 + Gamma × (T_mod − T_ref))] × (1/6 h)."
            ),
            "Specific Yield (h = kWh/kWp)": st.column_config.NumberColumn(
                "Specific Yield (h = kWh/kWp)", format="%.3f",
                help="E_ac / DC_cap (inverter scalar)."
            ),
            "Performance Ratio (% at STC)": st.column_config.NumberColumn(
                "Performance Ratio (% at STC)", format="%.2f",
                help="(Y_ac / Y_ref) × 100."
            ),
            "Site Median Specific Yield (h)": st.column_config.NumberColumn(
                "Site Median Specific Yield (h)", format="%.3f",
                help="Median of Y_ac across other inverters with FullCap (row-wise)."
            ),
            "Proxy (kWh)": st.column_config.NumberColumn(
                "Proxy (kWh)", format="%.3f",
                help="Site Median Y_ac × DC_cap (inverter scalar), capped at (AC_max_project/N_inv)/6."
            ),
            "PVLIB (kWh)": st.column_config.NumberColumn(
                "PVLIB (kWh)", format="%.3f",
                help="Project PVLIB × (AC_max_inv / Σ AC_max_inv), capped by AC_max_inv/6."
            ),
            "Met (kWh)": st.column_config.NumberColumn(
                "Met (kWh)", format="%.3f",
                help="Y_ref × PR_nominal × DC_cap (inverter scalar), capped at (AC_max_project/N_inv)/6."
            ),
            "Reference Potential (kWh)": st.column_config.NumberColumn(
                "Reference Potential (kWh)", format="%.3f",
                help="Fallback: Proxy → PVLIB → Met."
            ),
            "Potential Source": st.column_config.TextColumn("Potential Source"),
        }
    )

    # Per-row numeric details (one-line formulas)
    with st.expander("KPI numeric details — one-line formulas per 10-min", expanded=False):
        times = ed["Time"].tolist()
        labels = [f"{i:02d} — {t}" for i, t in enumerate(times)]
        idx = st.selectbox("Select 10-min row", options=list(range(len(labels))), format_func=lambda i: labels[i], key=f"sel_row_{key_suffix}")

        poa_i = float(poa_for_kpi[idx]) if np.isfinite(poa_for_kpi[idx]) else np.nan
        tmod_i = float(t_mod[idx]) if np.isfinite(t_mod[idx]) else np.nan
        eac_i = float(eac[idx]) if np.isfinite(eac[idx]) else np.nan
        dc_i = float(dc_cap_inv_scalar)

        ref_irr_loc, ref_temp_loc, gamma_loc, _, _, ac_max_project_kw_loc, pr_nominal_loc = _get_site_params()
        cap_10min = (ac_max_project_kw_loc / N_INV) / 6.0

        DT_HOURS = 1.0 / 6.0
        denom = ref_irr_loc * (1.0 + gamma_loc * (tmod_i - ref_temp_loc)) if np.isfinite(tmod_i) else np.nan
        yref_i = (poa_i / denom) * DT_HOURS if (np.isfinite(denom) and denom != 0) else np.nan
        yac_inv_i = (eac_i / dc_i) if (np.isfinite(dc_i) and dc_i > 0) else np.nan
        pr_i_pct = (yac_inv_i / yref_i) * 100.0 if (np.isfinite(yref_i) and yref_i != 0) else np.nan

        median_i = float(site_median_yac[idx]) if np.isfinite(site_median_yac[idx]) else np.nan
        e_proxy_i_raw = median_i * dc_i if np.isfinite(median_i) and np.isfinite(dc_i) else np.nan
        e_proxy_i = min(e_proxy_i_raw, cap_10min) if np.isfinite(e_proxy_i_raw) else np.nan

        e_met_i_raw = (yref_i * pr_nominal_loc * dc_i) if (np.isfinite(yref_i) and np.isfinite(dc_i)) else np.nan
        e_met_i = min(e_met_i_raw, cap_10min) if np.isfinite(e_met_i_raw) else np.nan

        pvlib_prj_i = float(_get_num_col(ed, "PVLIB_project_kWh")[idx])
        ac_max_inv_i = float(ac_max_inv[idx]) if np.isfinite(ac_max_inv[idx]) else np.nan
        sum_acmax = float(np.nansum(ac_max_inv)) if np.isfinite(np.nansum(ac_max_inv)) else 0.0
        share_i = (ac_max_inv_i / sum_acmax) if (sum_acmax > 0 and np.isfinite(ac_max_inv_i)) else np.nan
        e_pvlib_i_raw = (pvlib_prj_i * share_i) if (np.isfinite(pvlib_prj_i) and np.isfinite(share_i)) else np.nan
        cap_pvlib_i = (ac_max_inv_i / 6.0) if np.isfinite(ac_max_inv_i) else np.nan
        e_pvlib_i = min(e_pvlib_i_raw, cap_pvlib_i) if (np.isfinite(e_pvlib_i_raw) and np.isfinite(cap_pvlib_i)) else np.nan

        psrc_i = "Proxy" if _get_bool_col(ed, "Proxy_available")[idx] else (
            "PVLIB" if _get_bool_col(ed, "PVLIB_available")[idx] else (
                "Met" if _get_bool_col(ed, "Met_available")[idx] else "None"
            )
        )
        epot_i = e_proxy_i if psrc_i == "Proxy" else (e_pvlib_i if psrc_i == "PVLIB" else (e_met_i if psrc_i == "Met" else np.nan))

        st.markdown("**Inputs (selected row)**")
        st.dataframe(pd.DataFrame([{
            "POA (W/m²)": poa_i, "T_mod (°C)": tmod_i, "E_ac (kWh)": eac_i, "DC_cap_inv (kWp) — global": dc_i,
            "RefIrr (W/m²)": ref_irr_loc, "T_ref (°C)": ref_temp_loc, "Gamma (1/°C)": gamma_loc,
            "AC_max_project/Inv/6 (kWh)": (ac_max_project_kw_loc / N_INV) / 6.0,
            "PVLIB_project (kWh)": pvlib_prj_i, "AC_max_inv (kW)": ac_max_inv_i,
            "Potential Source": psrc_i
        }]), use_container_width=True, hide_index=True)

        poa_n = _fmt(poa_i, 3); refirr = _fmt(ref_irr_loc, 3); gamma_n = _fmt(gamma_loc, 6)
        tmod_n = _fmt(tmod_i, 3); tref_n = _fmt(ref_temp_loc, 3)
        yref_n = _fmt(yref_i, 6); yac_n = _fmt(yac_inv_i, 6); pr_n = _fmt(pr_i_pct, 3)
        dc_n = _fmt(dc_i, 3); eac_n = _fmt(eac_i, 3)
        med_n = _fmt(median_i, 6); cap6_n = _fmt((ac_max_project_kw_loc / N_INV) / 6.0, 3)
        eprx_n = _fmt(e_proxy_i, 6); emet_n = _fmt(e_met_i, 6)
        share_n = _fmt(share_i, 6); pvprj_n = _fmt(pvlib_prj_i, 3); cappv_n = _fmt(cap_pvlib_i, 3); epvl_n = _fmt(e_pvlib_i, 6)

        st.markdown("**One-line numeric application**")
        _latex(
            r"Y_{[[\mathrm{ref}]]}(h)=\frac{[[\mathrm{POA}]]}{[[\mathrm{RefIrr}]]\cdot\left(1+\gamma\cdot(T_{[[\mathrm{mod}]]}-T_{[[\mathrm{ref}]]})\right)}\cdot\frac{1}{6}"
            r"=\frac{<poa>}{<refirr>\cdot\left(1+<gamma>\cdot(<tmod>-<tref>)\right)}\cdot\frac{1}{6}"
            r"=<yref>\ \mathrm{h}",
            poa=poa_n, refirr=refirr, gamma=gamma_n, tmod=tmod_n, tref=tref_n, yref=yref_n
        )
        _latex(
            r"\mathrm{SpecificYield}(h)=\frac{E_{[[\mathrm{ac}]]}(\mathrm{kWh})}{[[\mathrm{DC\_cap\_inv}]](\mathrm{kWp})}"
            r"=\frac{<eac>}{<dc>}"
            r"=<yac>\ \mathrm{h}",
            eac=eac_n, dc=dc_n, yac=yac_n
        )
        _latex(
            r"\mathrm{PR}(\%)=\frac{\mathrm{SpecificYield}(h)}{Y_{[[\mathrm{ref}]]}(h)}\times 100"
            r"=\left(\frac{<yac>}{<yref>}\right)\times 100"
            r"=<pr>\ \%",
            yac=yac_n, yref=yref_n, pr=pr_n
        )
        if psrc_i == "Proxy":
            _latex(
                r"E_{[[\mathrm{potential}]]}=\min\!\Big(\mathrm{Median}(Y_{[[\mathrm{ac}]]})\times [[\mathrm{DC\_cap\_inv}]],\ \frac{[[\mathrm{AC\_max\_project}]]}{[[N\_{{\mathrm{inv}}}]]}\cdot\frac{1}{6}\Big)"
                r"=\min\!\Big(<med>\times <dc>,\ <cap6>\Big)"
                r"=<ep>\ \mathrm{kWh}",
                med=med_n, dc=dc_n, cap6=cap6_n, ep=eprx_n
            )
        elif psrc_i == "PVLIB":
            _latex(
                r"E_{[[\mathrm{potential}]]}=\min\!\Big(\mathrm{PVLIB\_project}\times \frac{[[\mathrm{AC\_max\_inv}]]}{\sum [[\mathrm{AC\_max\_inv}]]},\ \frac{[[\mathrm{AC\_max\_inv}]]}{6}\Big)"
                r"=\min\!\Big(<pv>\times <share>,\ <cap>\Big)"
                r"=<ep>\ \mathrm{kWh}",
                pv=pvprj_n, share=share_n, cap=cappv_n, ep=epvl_n
            )
        elif psrc_i == "Met":
            _latex(
                r"E_{[[\mathrm{potential}]]}=\min\!\Big(Y_{[[\mathrm{ref}]]}\cdot PR_{\mathrm{nom}}\cdot [[\mathrm{DC\_cap\_inv}]],\ \frac{[[\mathrm{AC\_max\_project}]]}{[[N\_{{\mathrm{inv}}}]]}\cdot\frac{1}{6}\Big)"
                r"=\min\!\Big(<yref>\cdot PR_{\mathrm{nom}}\cdot <dc>,\ <cap6>\Big)"
                r"=<ep>\ \mathrm{kWh}",
                yref=yref_n, dc=dc_n, cap6=cap6_n, ep=emet_n
            )
        else:
            st.info("No potential source available for this row.")

    # Daily totals & Aggregated KPIs (scalar DC)
    st.markdown("### Daily Totals & Aggregated KPIs")
    total_eac = float(np.nansum(eac))
    total_pot = float(np.nansum(potential))
    y_ac_day = (total_eac / dc_cap_inv_scalar) if dc_cap_inv_scalar > 0 else np.nan
    y_ref_day = float(np.nansum(y_ref)) if np.isfinite(np.nansum(y_ref)) else np.nan
    pr_day = (y_ac_day / y_ref_day) if (np.isfinite(y_ref_day) and y_ref_day != 0) else np.nan
    site_median_day = float(np.nanmean(site_median_yac)) if np.isfinite(np.nanmean(site_median_yac)) else np.nan

    totals = pd.DataFrame([{
        "Σ E_ac (kWh)": total_eac if np.isfinite(total_eac) else np.nan,
        "Σ Potential (kWh)": total_pot if np.isfinite(total_pot) else np.nan,
        "Reference Yield — day (h)": y_ref_day if np.isfinite(y_ref_day) else np.nan,
        "Specific Yield — day (h)": y_ac_day if np.isfinite(y_ac_day) else np.nan,
        "Performance Ratio — day (%)": pr_day * 100 if np.isfinite(pr_day) else np.nan,
        "Site Median Specific Yield — day (h)": site_median_day if np.isfinite(site_median_day) else np.nan,
    }])
    st.data_editor(
        totals, key=f"totals_{key_suffix}", use_container_width=True, disabled=True,
        column_config={
            "Σ E_ac (kWh)": st.column_config.NumberColumn("Σ E_ac (kWh)", format="%.3f",
                                                          help="Sum of measured AC energy over the day."),
            "Σ Potential (kWh)": st.column_config.NumberColumn("Σ Potential (kWh)", format="%.3f",
                                                               help="Sum of Reference Potential over the day."),
            "Reference Yield — day (h)": st.column_config.NumberColumn("Reference Yield — day (h)", format="%.3f",
                                                                       help="SUM of per-interval Y_ref (hours)."),
            "Specific Yield — day (h)": st.column_config.NumberColumn("Specific Yield — day (h)", format="%.3f",
                                                                      help="Σ E_ac / inverter DC_cap (scalar)."),
            "Performance Ratio — day (%)": st.column_config.NumberColumn("Performance Ratio — day (%)", format="%.2f",
                                                                         help="Y_ac_day / Y_ref_day × 100."),
            "Site Median Specific Yield — day (h)": st.column_config.NumberColumn("Site Median Specific Yield — day (h)", format="%.3f"),
        }
    )

    # Aggregated numeric details (day)
    with st.expander("Day sums — numeric details (a + b + c = total)", expanded=False):
        eac_terms = _list_vals(eac, 3)
        pot_terms = _list_vals(potential, 3)
        yref_terms = _list_vals(y_ref, 6)

        eac_sum_expr = " + ".join(eac_terms) if eac_terms else "NaN"
        pot_sum_expr = " + ".join(pot_terms) if pot_terms else "NaN"
        yref_sum_expr = " + ".join(yref_terms) if yref_terms else "NaN"

        st.latex(r"\sum E_{\mathrm{ac}}(\mathrm{kWh}) = \sum_{i=1}^{N} E_{\mathrm{ac},i}")
        st.write(f"= {eac_sum_expr} = **{_fmt(total_eac, 3)} kWh**")

        st.latex(r"\sum E_{\mathrm{potential}}(\mathrm{kWh}) = \sum_{i=1}^{N} E_{\mathrm{potential},i}")
        st.write(f"= {pot_sum_expr} = **{_fmt(total_pot, 3)} kWh**")

        st.latex(r"\sum Y_{[[\mathrm{ref}]]}(h)")
        st.write(f"= {yref_sum_expr} = **{_fmt(y_ref_day, 6)} h**")

        st.latex(r"\mathrm{SpecificYield}_{\mathrm{day}}(h) = \frac{\sum E_{\mathrm{ac}}}{\mathrm{DC\_cap\_inv\ (scalar)}}")
        st.write(f"= {_fmt(total_eac, 3)} ÷ {_fmt(dc_cap_inv_scalar, 6)} → **{_fmt(y_ac_day, 6)} h**")

        st.latex(r"\mathrm{PR}_{\mathrm{day}}(\%) = \frac{\mathrm{SpecificYield}_{\mathrm{day}}}{Y_{[[\mathrm{ref}]],\mathrm{day}}} \times 100")
        st.write(f"= ({_fmt(y_ac_day, 6)} ÷ {_fmt(y_ref_day, 6)}) × 100 → **{_fmt(pr_day * 100, 3)} %**")

    # store for site page
    st.session_state.setdefault("mi_outputs", {})
    st.session_state["mi_outputs"][inv_name] = {
        "time": ed["Time"].tolist(),
        "E_ac": eac,
        "E_potential": potential,
        "Y_ref": y_ref,
        # keep an array (replicated scalar) just for consistency
        "DC_cap_inv": np.full(len(ed), dc_cap_inv_scalar, dtype=float),
    }


# ---------- Site aggregation page ----------
def _site_page():
    st.subheader("Site — Aggregation (10-min)")
    mi = st.session_state.get("mi_outputs", {})
    if len(mi) < N_INV:
        st.info("Open the three inverter pages first to populate site aggregation.")
        return

    # DC site total
    dc_site_total = float(st.session_state.get("dc_cap_site_kwp", np.nan))

    names = list(mi.keys())
    n = len(mi[names[0]]["E_ac"])

    e_ac_site = np.zeros(n, dtype=float)
    e_pot_site = np.zeros(n, dtype=float)
    y_ref_stack = []

    for name in names:
        e_ac_site += np.nan_to_num(mi[name]["E_ac"], nan=0.0)
        e_pot_site += np.nan_to_num(mi[name]["E_potential"], nan=0.0)
        y_ref_stack.append(mi[name]["Y_ref"])

    y_ref_stack = np.vstack(y_ref_stack)
    y_ref_site = np.nanmean(y_ref_stack, axis=0)

    # Specific Yield (site) using DC_site_total
    y_ac_site = np.where(dc_site_total > 0, e_ac_site / dc_site_total, np.nan)
    pr_site = np.where((y_ref_site == 0) | np.isnan(y_ref_site), np.nan, y_ac_site / y_ref_site)

    df = pd.DataFrame({
        "Time": mi[names[0]]["time"],
        "Σ E_ac (kWh)": np.round(e_ac_site, 6),
        "Σ Potential (kWh)": np.round(e_pot_site, 6),
        "Reference Yield (h at STC) — site mean": np.round(y_ref_site, 6),
        "Specific Yield (h = kWh/kWp) — site (using Site DC)": np.round(y_ac_site, 6),
        "Performance Ratio (% at STC) — site": np.round(pr_site * 100.0, 3),
    })
    st.data_editor(
        df, key="site_10min_table", use_container_width=True, disabled=True,
        column_config={
            "Σ E_ac (kWh)": st.column_config.NumberColumn("Σ E_ac (kWh)", format="%.3f"),
            "Σ Potential (kWh)": st.column_config.NumberColumn("Σ Potential (kWh)", format="%.3f"),
            "Reference Yield (h at STC) — site mean": st.column_config.NumberColumn("Reference Yield (h at STC) — site mean", format="%.3f"),
            "Specific Yield (h = kWh/kWp) — site (using Site DC)": st.column_config.NumberColumn(
                "Specific Yield (h = kWh/kWp) — site (using Site DC)", format="%.3f",
                help="Σ E_ac_site / DC_cap_site_total."
            ),
            "Performance Ratio (% at STC) — site": st.column_config.NumberColumn("Performance Ratio (% at STC) — site", format="%.2f"),
        }
    )

    # Site 10-min numeric details
    with st.expander("Site numeric details — per 10-min", expanded=False):
        labels = [f"{i:02d} — {t}" for i, t in enumerate(mi[names[0]]["time"])]
        idx = st.selectbox("Select 10-min row (site)", options=list(range(len(labels))), format_func=lambda i: labels[i], key="site_sel_row")

        eac_i = float(e_ac_site[idx]) if np.isfinite(e_ac_site[idx]) else np.nan
        yref_i = float(y_ref_site[idx]) if np.isfinite(y_ref_site[idx]) else np.nan
        dcsite = float(st.session_state.get("dc_cap_site_kwp", np.nan))

        yac_site_i = (eac_i / dcsite) if (np.isfinite(dcsite) and dcsite > 0) else np.nan
        pr_site_i = (yac_site_i / yref_i) * 100.0 if (np.isfinite(yref_i) and yref_i != 0) else np.nan

        _latex(r"Y_{[[\mathrm{ref}]],\ \mathrm{site}}(h)=\mathrm{mean\ across\ inverters}")
        _latex(
            r"\mathrm{SpecificYield}_{\mathrm{site}}(h)=\frac{\sum E_{\mathrm{ac,site}}}{\mathrm{DC\_cap\_site\_total}}"
            r"=\frac{<eac>}{<dcsite>}"
            r"=<yac>\ \mathrm{h}",
            eac=_fmt(eac_i, 3), dcsite=_fmt(dcsite, 3), yac=_fmt(yac_site_i, 6)
        )
        _latex(
            r"\mathrm{PR}_{\mathrm{site}}(\%)=\frac{\mathrm{SpecificYield}_{\mathrm{site}}}{Y_{[[\mathrm{ref}]],\mathrm{site}}}\times 100"
            r"=\left(\frac{<yac>}{<yref>}\right)\times 100"
            r"=<pr>\ \%",
            yac=_fmt(yac_site_i, 6), yref=_fmt(yref_i, 6), pr=_fmt(pr_site_i, 3)
        )

    # Daily totals (site)
    total_eac_site = float(np.nansum(e_ac_site))
    total_pot_site = float(np.nansum(e_pot_site))
    y_ref_day_site = float(np.nansum(y_ref_site)) if np.isfinite(np.nansum(y_ref_site)) else np.nan
    y_ac_day_site = (total_eac_site / dc_site_total) if dc_site_total > 0 else np.nan
    pr_day_site = (y_ac_day_site / y_ref_day_site) if (np.isfinite(y_ref_day_site) and y_ref_day_site != 0) else np.nan

    totals = pd.DataFrame([{
        "Σ E_ac (kWh)": total_eac_site if np.isfinite(total_eac_site) else np.nan,
        "Σ Potential (kWh)": total_pot_site if np.isfinite(total_pot_site) else np.nan,
        "Reference Yield — day (h)": y_ref_day_site if np.isfinite(y_ref_day_site) else np.nan,
        "Specific Yield — day (h) — site (using Site DC)": y_ac_day_site if np.isfinite(y_ac_day_site) else np.nan,
        "Performance Ratio — day (%)": pr_day_site * 100 if np.isfinite(pr_day_site) else np.nan,
    }])
    st.markdown("### Daily Totals (Site)")
    st.data_editor(
        totals, key="site_totals_table", use_container_width=True, disabled=True,
        column_config={
            "Σ E_ac (kWh)": st.column_config.NumberColumn("Σ E_ac (kWh)", format="%.3f",
                                                          help="Sum of site measured AC energy across all inverters."),
            "Σ Potential (kWh)": st.column_config.NumberColumn("Σ Potential (kWh)", format="%.3f",
                                                               help="Sum of site Reference Potential across all inverters."),
            "Reference Yield — day (h)": st.column_config.NumberColumn("Reference Yield — day (h)", format="%.3f",
                                                                       help="SUM of site-level Y_ref hours."),
            "Specific Yield — day (h) — site (using Site DC)": st.column_config.NumberColumn(
                "Specific Yield — day (h) — site (using Site DC)", format="%.3f",
                help="Σ E_ac_site / DC_cap_site_total."
            ),
            "Performance Ratio — day (%)": st.column_config.NumberColumn("Performance Ratio — day (%)", format="%.2f",
                                                                         help="Site Y_ac_day / Site Y_ref_day × 100."),
        }
    )

    # Site daily aggregated sums
    with st.expander("Day sums — numeric details (Site)", expanded=False):
        eac_terms = _list_vals(e_ac_site, 3)
        pot_terms = _list_vals(e_pot_site, 3)
        yref_terms = _list_vals(y_ref_site, 6)

        eac_sum_expr = " + ".join(eac_terms) if eac_terms else "NaN"
        pot_sum_expr = " + ".join(pot_terms) if pot_terms else "NaN"
        yref_sum_expr = " + ".join(yref_terms) if yref_terms else "NaN"

        st.latex(r"\sum E_{\mathrm{ac,site}}(\mathrm{kWh})")
        st.write(f"= {eac_sum_expr} = **{_fmt(total_eac_site, 3)} kWh**")

        st.latex(r"\sum E_{\mathrm{potential,site}}(\mathrm{kWh})")
        st.write(f"= {pot_sum_expr} = **{_fmt(total_pot_site, 3)} kWh**")

        st.latex(r"\sum Y_{[[\mathrm{ref}]],\mathrm{site}}(h)")
        st.write(f"= {yref_sum_expr} = **{_fmt(y_ref_day_site, 6)} h**")

        st.latex(r"\mathrm{SpecificYield}_{\mathrm{day,site}}(h) = \frac{\sum E_{\mathrm{ac,site}}}{\mathrm{DC\_cap\_site\_total}}")
        st.write(f"= {_fmt(total_eac_site, 3)} ÷ {_fmt(dc_site_total, 6)} → **{_fmt(y_ac_day_site, 6)} h**")

        st.latex(r"\mathrm{PR}_{\mathrm{day,site}}(\%) = \frac{\mathrm{SpecificYield}_{\mathrm{day,site}}}{Y_{[[\mathrm{ref}]],\mathrm{day,site}}} \times 100")
        st.write(f"= ({_fmt(y_ac_day_site, 6)} ÷ {_fmt(y_ref_day_site, 6)}) × 100 → **{_fmt(pr_day_site * 100, 3)} %**")


# ---------- Project Settings (site-level UI) + About this section ----------
def _project_settings_ui():
    # About section (INTERMEDIATE)
    with st.expander("ℹ️ About this section (INTERMEDIATE)", expanded=False):
        st.markdown("""
**Objective**  
Compare each inverter’s measured 10-minute production with multiple reference potentials (Proxy, PVLIB share, MET-based model).  
Compute **Reference Yield**, **Specific Yield**, **PR**, and aggregate to the site using a single **site DC capacity**.

**Notes**  
- AC energy is a single input **AC_kWh** per 10-min (no revenue/check fallback).  
- Each inverter has **one DC capacity (kWp)** input used everywhere in that inverter’s calculations.  
- Proxy potential uses the **site median specific yield** (other inverters, FullCap rows only) × inverter DC, capped at **AC_max_project/N_inv/6**.
        """)

    st.markdown("### ⚙️ Project Settings (Site-level)")
    c1, c2, c3 = st.columns([0.34, 0.33, 0.33])

    with c1:
        dc_site = st.number_input(
            "DC capacity (kWp) — site total",
            min_value=0.0,
            value=float(st.session_state.get("dc_cap_site_kwp", 1950.0)),
            step=10.0,
            help="Global site DC capacity used for Site-level Specific Yield.",
            key="dc_cap_site_kwp_input",
        )
        st.session_state["dc_cap_site_kwp"] = dc_site

    with c2:
        ac_max = st.number_input(
            "AC Max Power (project) [kW]",
            min_value=0.0,
            value=float(st.session_state.get("ac_max_project_kw", 1800.0)),
            step=10.0,
            help="Project AC limit used for 10-min caps (AC_max/6).",
            key="ac_max_project_kw_input",
        )
        st.session_state["ac_max_project_kw"] = ac_max

    with c3:
        proxy_yield = st.number_input(
            "Proxy Yield (h per 10-min)",
            min_value=0.0,
            value=float(st.session_state.get("proxy_yield_h", 0.90)),
            step=0.01,
            help="Default proxy specific yield per 10-min used in potential estimation.",
            key="proxy_yield_h_input",
        )
        st.session_state["proxy_yield_h"] = proxy_yield


# ---------- Public entry ----------
def page_intermediate():
    # Site-level settings on top (+ About)
    _project_settings_ui()

    # Tabs
    t1, t2, t3, t4 = st.tabs(["Inverter 1", "Inverter 2", "Inverter 3", "Site"])
    with t1:
        _inverter_page("INV1", "inv1")
    with t2:
        _inverter_page("INV2", "inv2")
    with t3:
        _inverter_page("INV3", "inv3")
    with t4:
        _site_page()
