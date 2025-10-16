# intermediate.py
import numpy as np
import pandas as pd
import streamlit as st

N_INV = 3  # number of inverters

# ---------------------------
# Shared site parameters
# ---------------------------
def _get_site_params():
    """Read shared site parameters from session_state with safe defaults."""
    ref_irr   = float(st.session_state.get("ref_irr", 1000.0))
    ref_temp  = float(st.session_state.get("ref_temp", 25.0))
    gamma     = float(st.session_state.get("gamma", -0.0035))
    a         = float(st.session_state.get("king_a", -3.56))
    b         = float(st.session_state.get("king_b", -0.075))
    ac_max_prj= float(st.session_state.get("ac_max_project_kw", 1800.0))
    pr_nom    = float(st.session_state.get("pr_nominal", 0.80))
    return ref_irr, ref_temp, gamma, a, b, ac_max_prj, pr_nom

# ---------------------------
# Yields & PR (10-min window → hours)
# ---------------------------
def _yields_pr(poa_for_kpi, t_mod, ref_irr, ref_temp, gamma, eac, dc_cap_inv):
    """
    Compute Reference Yield (hours), Specific Yield (hours), and PR for a 10-min interval.
      - poa_for_kpi: 10-min average POA in W/m² (not Wh/m²)
      - eac: kWh per 10-min
      - dc_cap_inv: kWp
    """
    DT_HOURS = 1.0 / 6.0  # 10 minutes = 1/6 hour
    temp_factor = 1.0 + gamma * (t_mod - ref_temp)
    denom = ref_irr * temp_factor
    y_ref_ratio = np.where(np.isclose(denom, 0.0), np.nan, poa_for_kpi / denom)
    y_ref = y_ref_ratio * DT_HOURS
    y_ac  = np.where(dc_cap_inv > 0, eac / dc_cap_inv, np.nan)
    pr    = np.where((y_ref == 0) | np.isnan(y_ref), np.nan, y_ac / y_ref)
    return y_ref, y_ac, pr

# ---------------------------
# Availability auto-sync helpers (for sensor flags, not site availability)
# ---------------------------
def _force_flag_false_if_value_missing(ed: pd.DataFrame, value_col: str, flag_col: str):
    """If the numeric value is NaN/inf, force the availability flag to False in the DataFrame copy."""
    if value_col in ed.columns and flag_col in ed.columns:
        ed[flag_col] = ed[flag_col].fillna(False).astype(bool)
        vals = pd.to_numeric(ed[value_col], errors="coerce")
        ed.loc[~np.isfinite(vals), flag_col] = False

def _auto_sync_availability_flags(ed: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of ed where availability flags are set to False when their values are missing.
    Keeps UI checkboxes but ensures calculations never treat empty values as available.
    """
    ed = ed.copy()
    _force_flag_false_if_value_missing(ed, "MET_POA_Wm2", "MET_POA_available")
    _force_flag_false_if_value_missing(ed, "MET_Amb_C", "MET_amb_available")
    _force_flag_false_if_value_missing(ed, "MET_Back_Module_C", "MET_back_available")
    _force_flag_false_if_value_missing(ed, "MET_WindSpeed", "Wind_available")
    _force_flag_false_if_value_missing(ed, "AC_rev_kWh", "Rev_available")
    _force_flag_false_if_value_missing(ed, "AC_check_kWh", "Check_available")
    _force_flag_false_if_value_missing(ed, "PVLIB_project_kWh", "PVLIB_available")
    return ed

# ---------------------------
# AC & T_mod fallbacks
# ---------------------------
def _solve_ac_energy(df: pd.DataFrame):
    """AC fallback: revenue → check → NaN."""
    rev_av = df["Rev_available"].fillna(False).astype(bool).to_numpy()
    chk_av = df["Check_available"].fillna(False).astype(bool).to_numpy()
    ac_rev = pd.to_numeric(df["AC_rev_kWh"], errors="coerce").astype(float).to_numpy()
    ac_chk = pd.to_numeric(df["AC_check_kWh"], errors="coerce").astype(float).to_numpy()
    eac    = np.where(rev_av, ac_rev, np.where(chk_av, ac_chk, np.nan))
    src    = np.where(rev_av, "revenue", np.where(chk_av, "check", "none"))
    return eac, src

def _solve_tmod(df: pd.DataFrame, a: float, b: float):
    """
    T_mod fallback order:
      (1) MET_back
      (2) King(MET,[WS]) if MET_POA_available & MET_amb_available
      (3) SAT
    """
    back_av = df["MET_back_available"].fillna(False).astype(bool).to_numpy()
    amb_av  = df["MET_amb_available"].fillna(False).astype(bool).to_numpy()
    t_back  = pd.to_numeric(df["MET_Back_Module_C"], errors="coerce").astype(float).to_numpy()
    t_amb   = pd.to_numeric(df["MET_Amb_C"], errors="coerce").astype(float).to_numpy()
    t_sat   = pd.to_numeric(df["SAT_T_mod_C"], errors="coerce").astype(float).to_numpy()

    poa_met = pd.to_numeric(df["MET_POA_Wm2"], errors="coerce").astype(float).to_numpy()
    met_poa_av = df["MET_POA_available"].fillna(False).astype(bool).to_numpy()
    poa_ok  = met_poa_av & np.isfinite(poa_met)

    ws      = pd.to_numeric(df["MET_WindSpeed"], errors="coerce").astype(float).to_numpy()
    wind_av = df["Wind_available"].fillna(False).astype(bool).to_numpy()
    ws_ok   = np.isfinite(ws)

    n = len(df)
    t_mod = np.full(n, np.nan, dtype=float)
    src   = np.array([""]*n, dtype=object)

    # (1) measured back temp
    idx1 = back_av
    if np.any(idx1):
        t_mod[idx1] = t_back[idx1]
        src[idx1]   = "MET_back"

    # (2) King(MET,[WS])
    idx2 = (~back_av) & poa_ok & amb_av
    if np.any(idx2):
        t_k = t_amb + poa_met * np.exp(a)  # base without wind
        use_ws = idx2 & wind_av & ws_ok
        if np.any(use_ws):
            t_k[use_ws] = t_amb[use_ws] + poa_met[use_ws] * np.exp(a + b*ws[use_ws])
        t_mod[idx2] = t_k[idx2]
        src[idx2]   = np.where(use_ws[idx2], "King(MET,WS)", "King(MET)")

    # (3) SAT fallback
    idx3 = (~back_av) & (~(poa_ok & amb_av))
    if np.any(idx3):
        t_mod[idx3] = t_sat[idx3]
        src[idx3]   = "SAT"

    return t_mod, src

# ---------------------------
# Proxy & PVLIB helpers
# ---------------------------
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

# ---------------------------
# Realistic presets per inverter
# ---------------------------
def _realistic_presets(inv_name: str) -> pd.DataFrame:
    """Return a realistic 10-min dataset per inverter so KPIs look real."""
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
            "AC_rev_kWh":[78.0, 82.0, 85.0, 80.0, 75.0],
            "Rev_available":[True, True, True, True, True],
            "AC_check_kWh":[77.0, 81.0, 84.0, 79.5, 74.0],
            "Check_available":[True, True, True, True, True],
            "FullCap":[True, True, True, True, True],
            "DC_cap_inv_kWp":[700, 700, 700, 700, 700],
            "AC_max_inv_kW":[650, 650, 650, 650, 650],
            "PVLIB_project_kWh":[np.nan]*5,
            "PVLIB_available":[False]*5,
            "Proxy_available":[True]*5,
            "Met_available":[True]*5,
        })

    if inv_name == "INV2":
        return pd.DataFrame({
            "Time": times,
            "MET_POA_Wm2":[820, 860, 880, 850, 800],
            "MET_POA_available":[True, True, True, True, False],  # last row uses SAT
            "SAT_POA_Wm2":[830, 870, 890, 860, 810],
            "MET_Amb_C":[26.1, 27.1, 27.6, 27.3, 26.9],
            "MET_amb_available":[True, True, True, True, True],
            "MET_Back_Module_C":[np.nan, np.nan, 48.5, np.nan, np.nan],
            "MET_back_available":[False, False, True, False, False],
            "MET_WindSpeed":[2.3, 2.8, 3.0, 2.6, 2.1],
            "Wind_available":[True, True, True, True, True],
            "SAT_T_mod_C":[46.8, 47.9, 48.6, 48.2, 47.5],
            "AC_rev_kWh":[65.0, 70.0, 74.0, 72.0, 68.0],
            "Rev_available":[True, True, True, True, True],
            "AC_check_kWh":[64.0, 69.0, 73.0, 71.0, 67.0],
            "Check_available":[True, True, True, True, True],
            "FullCap":[True, True, True, True, True],
            "DC_cap_inv_kWp":[650, 650, 650, 650, 650],
            "AC_max_inv_kW":[620, 620, 620, 620, 620],
            "PVLIB_project_kWh":[np.nan]*5,
            "PVLIB_available":[False]*5,
            "Proxy_available":[True]*5,
            "Met_available":[True]*5,
        })

    return pd.DataFrame({  # INV3
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
        "AC_rev_kWh":[55.0, 60.0, 63.0, 61.0, 58.0],
        "Rev_available":[True, True, True, True, True],
        "AC_check_kWh":[54.0, 59.0, 62.0, 60.0, 57.0],
        "Check_available":[True, True, True, True, True],
        "FullCap":[True, True, False, True, True],
        "DC_cap_inv_kWp":[600, 600, 600, 600, 600],
        "AC_max_inv_kW":[600, 600, 600, 600, 600],
        "PVLIB_project_kWh":[np.nan]*5,
        "PVLIB_available":[False]*5,
        "Proxy_available":[True]*5,
        "Met_available":[True]*5,
    })

# ---------------------------
# Inverter page
# ---------------------------
def _inverter_page(inv_name: str, key_suffix: str):
    ref_irr, ref_temp, gamma, a, b, ac_max_project_kw, pr_nominal = _get_site_params()

    st.subheader(f"{inv_name} — 10-min Concept Data")

    # 1) keep the source DF in session to allow programmatic updates & re-render
    src_key = f"src_{key_suffix}"
    if src_key not in st.session_state:
        st.session_state[src_key] = _realistic_presets(inv_name).copy()

    # 2) render editor from the session source
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

            "AC_rev_kWh": st.column_config.NumberColumn("AC revenue (kWh/10-min)", min_value=0.0, step=0.001),
            "Rev_available": st.column_config.CheckboxColumn("Revenue available"),
            "AC_check_kWh": st.column_config.NumberColumn("AC check (kWh/10-min)", min_value=0.0, step=0.001),
            "Check_available": st.column_config.CheckboxColumn("Check available"),

            "FullCap": st.column_config.CheckboxColumn("Full Capacity (for Site Median Specific Yield)"),
            "DC_cap_inv_kWp": st.column_config.NumberColumn("DC capacity (kWp)", min_value=0.0, step=1.0),
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

    # 3) auto-sync flags with value presence
    synced = _auto_sync_availability_flags(ed)

    # 4) if any availability flag changed, push back to source DF and rerun (to reflect in UI)
    flag_cols = [
        "MET_POA_available", "MET_amb_available", "MET_back_available",
        "Wind_available", "Rev_available", "Check_available", "PVLIB_available"
    ]
    flag_cols = [c for c in flag_cols if c in synced.columns and c in st.session_state[src_key].columns]

    flags_changed = not synced[flag_cols].reset_index(drop=True).equals(
        st.session_state[src_key][flag_cols].reset_index(drop=True)
    )
    if flags_changed:
        st.session_state[src_key] = synced
        st.rerun()

    # 5) use the synced dataframe from here on
    ed = synced

    # Resolve inputs
    eac, ac_src = _solve_ac_energy(ed)
    t_mod, t_src = _solve_tmod(ed, a, b)

    poa_met = pd.to_numeric(ed["MET_POA_Wm2"], errors="coerce").astype(float).to_numpy()
    poa_sat = pd.to_numeric(ed["SAT_POA_Wm2"], errors="coerce").astype(float).to_numpy()
    met_poa_av = ed["MET_POA_available"].fillna(False).astype(bool).to_numpy()
    poa_met_ok = met_poa_av & np.isfinite(poa_met)
    poa_for_kpi = np.where(poa_met_ok, poa_met, poa_sat)
    poa_source  = np.where(poa_met_ok, "MET_POA", "SAT_POA")

    dc_cap_inv = pd.to_numeric(ed["DC_cap_inv_kWp"], errors="coerce").astype(float).to_numpy()
    ac_max_inv = pd.to_numeric(ed["AC_max_inv_kW"], errors="coerce").astype(float).to_numpy()
    fullcap    = ed["FullCap"].fillna(False).astype(bool).to_numpy()

    # KPIs
    y_ref, y_ac, _ = _yields_pr(poa_for_kpi, t_mod, ref_irr, ref_temp, gamma, eac, dc_cap_inv)

    # store per-inverter Y_ac & FullCap for site median
    st.session_state.setdefault("mi_yac", {})
    st.session_state.setdefault("mi_fullcap", {})
    st.session_state["mi_yac"][inv_name] = y_ac
    st.session_state["mi_fullcap"][inv_name] = fullcap

    yac_all = st.session_state["mi_yac"]
    fc_all  = st.session_state["mi_fullcap"]

    # site median specific yield excluding current inverter (avoid self-reference)
    others_yac = {k: v for k, v in yac_all.items() if k != inv_name}
    others_fc  = {k: v for k, v in fc_all.items() if k != inv_name}
    if len(others_yac) >= 1:
        site_median_yac = _proxy_potential(others_yac, others_fc, None, ac_max_project_kw)
    else:
        n_rows = len(y_ac)
        default_proxy = float(st.session_state.get("proxy_yield_h", np.nan))
        site_median_yac = np.full(n_rows, default_proxy, dtype=float)

    # potentials
    e_proxy_raw = site_median_yac * dc_cap_inv
    e_proxy     = _cap_proxy_met_per_inv(e_proxy_raw, ac_max_project_kw)

    e_met_raw   = y_ref * pr_nominal * dc_cap_inv
    e_met       = _cap_proxy_met_per_inv(e_met_raw, ac_max_project_kw)

    pvlib_prj   = pd.to_numeric(ed["PVLIB_project_kWh"], errors="coerce").astype(float).to_numpy()
    pvlib_av    = ed["PVLIB_available"].fillna(False).astype(bool).to_numpy()
    sum_acmax   = float(np.nansum(ac_max_inv)) if np.isfinite(np.nansum(ac_max_inv)) else 0.0
    e_pvlib_raw = _pvlib_potential(pvlib_prj, ac_max_inv, sum_acmax)
    e_pvlib     = _cap_pvlib_per_inv(e_pvlib_raw, ac_max_inv)

    proxy_av = ed["Proxy_available"].fillna(False).astype(bool).to_numpy()
    met_av   = ed["Met_available"].fillna(False).astype(bool).to_numpy()

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

    # resolved inputs (no site availability / event weight columns)
    st.markdown("### Resolved Inputs")
    resolved = pd.DataFrame({
        "Time": ed["Time"],
        "POA_used (W/m²)": np.round(poa_for_kpi, 1),
        "POA_Source": poa_source,
        "T_mod_used (°C)": np.round(t_mod, 3),
        "T_mod_Source": t_src,
        "AC_kWh": np.round(eac, 6),
        "AC_Source": ac_src,
        "FullCap": fullcap,
        "DC_cap_inv (kWp)": dc_cap_inv,
        "AC_max_inv (kW)": ac_max_inv,
    })
    st.data_editor(
        resolved, key=f"resolved_{key_suffix}", use_container_width=True, disabled=True,
        column_config={
            "POA_used (W/m²)": st.column_config.NumberColumn("POA_used (W/m²)", help="MET_POA if available, else SAT_POA."),
            "POA_Source": st.column_config.TextColumn("POA_Source", help="MET_POA or SAT_POA."),
            "T_mod_used (°C)": st.column_config.NumberColumn("T_mod_used (°C)", help="MET_back or King(MET,[WS]) or SAT."),
            "T_mod_Source": st.column_config.TextColumn("T_mod_Source", help="MET_back / King(MET,WS) / King(MET) / SAT."),
            "AC_kWh": st.column_config.NumberColumn("AC_kWh", help="Resolved via revenue→check fallback."),
            "AC_Source": st.column_config.TextColumn("AC_Source", help="revenue / check / none."),
            "FullCap": st.column_config.CheckboxColumn("FullCap", help="If True, contributes to Site Median."),
            "DC_cap_inv (kWp)": st.column_config.NumberColumn("DC_cap_inv (kWp)", help="Per-row DC capacity."),
            "AC_max_inv (kW)": st.column_config.NumberColumn("AC_max_inv (kW)", help="Inverter AC nameplate."),
        }
    )

    # KPI table
    st.markdown("### KPI Data (per 10-min)")
    kpi = pd.DataFrame({
        "Time": ed["Time"],
        "Reference Yield (h at STC)": np.round(y_ref, 6),
        "Specific Yield (h = kWh/kWp)": np.round(y_ac, 6),
        "Performance Ratio (% at STC)": np.round((y_ac / y_ref) * 100.0, 3),
        "Site Median Specific Yield (h)": np.round(site_median_yac, 6),
        "Proxy (kWh)": np.round(e_proxy, 6),
        "PVLIB (kWh)": np.round(e_pvlib, 6),
        "Met (kWh)":   np.round(e_met, 6),
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
                help="E_ac / DC_cap_inv."
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
                help="Site Median Y_ac × DC_cap_inv, capped at (AC_max_project/N_inv)/6."
            ),
            "PVLIB (kWh)": st.column_config.NumberColumn(
                "PVLIB (kWh)", format="%.3f",
                help="Project PVLIB × (AC_max_inv / Σ AC_max_inv), capped by AC_max_inv/6."
            ),
            "Met (kWh)": st.column_config.NumberColumn(
                "Met (kWh)", format="%.3f",
                help="Y_ref × PR_nominal × DC_cap_inv, capped at (AC_max_project/N_inv)/6."
            ),
            "Reference Potential (kWh)": st.column_config.NumberColumn(
                "Reference Potential (kWh)", format="%.3f",
                help="Fallback: Proxy → PVLIB → Met."
            ),
            "Potential Source": st.column_config.TextColumn("Potential Source"),
        }
    )

    # Daily totals (no site availability / event weight / lost metrics)
    st.markdown("### Daily Totals & Aggregated KPIs")
    total_eac = float(np.nansum(eac))
    total_pot = float(np.nansum(potential))

    y_ac_day  = (total_eac / np.nanmean(dc_cap_inv)) if np.nanmean(dc_cap_inv) > 0 else np.nan
    y_ref_day = float(np.nansum(y_ref)) if np.isfinite(np.nansum(y_ref)) else np.nan
    pr_day    = (y_ac_day / y_ref_day) if (np.isfinite(y_ref_day) and y_ref_day != 0) else np.nan
    site_median_day = float(np.nanmean(site_median_yac)) if np.isfinite(np.nanmean(site_median_yac)) else np.nan

    totals = pd.DataFrame([{
        "Σ E_ac (kWh)": total_eac if np.isfinite(total_eac) else np.nan,
        "Σ Potential (kWh)": total_pot if np.isfinite(total_pot) else np.nan,
        "Reference Yield — day (h)": y_ref_day if np.isfinite(y_ref_day) else np.nan,
        "Specific Yield — day (h)": y_ac_day if np.isfinite(y_ac_day) else np.nan,
        "Performance Ratio — day (%)": pr_day*100 if np.isfinite(pr_day) else np.nan,
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
                help="Σ E_ac / mean(DC_cap_inv)."),
            "Performance Ratio — day (%)": st.column_config.NumberColumn("Performance Ratio — day (%)", format="%.2f",
                help="Y_ac_day / Y_ref_day × 100."),
            "Site Median Specific Yield — day (h)": st.column_config.NumberColumn("Site Median Specific Yield — day (h)", format="%.3f"),
        }
    )

    # store for site page
    st.session_state.setdefault("mi_outputs", {})
    st.session_state["mi_outputs"][inv_name] = {
        "time": ed["Time"].tolist(),
        "E_ac": eac,
        "E_potential": potential,
        "Y_ref": y_ref,
        "DC_cap_inv": dc_cap_inv,
    }

# ---------------------------
# Site aggregation page
# ---------------------------
def _site_page():
    st.subheader("Site — Aggregation (10-min)")
    mi = st.session_state.get("mi_outputs", {})
    if len(mi) < N_INV:
        st.info("Open the three inverter pages first to populate site aggregation.")
        return

    names = list(mi.keys())
    n = len(mi[names[0]]["E_ac"])

    e_ac_site    = np.zeros(n, dtype=float)
    e_pot_site   = np.zeros(n, dtype=float)
    y_ref_stack  = []

    for name in names:
        e_ac_site    += np.nan_to_num(mi[name]["E_ac"], nan=0.0)
        e_pot_site   += np.nan_to_num(mi[name]["E_potential"], nan=0.0)
        y_ref_stack.append(mi[name]["Y_ref"])

    y_ref_stack = np.vstack(y_ref_stack)
    y_ref_site = np.nanmean(y_ref_stack, axis=0)

    dc_cap_total = 0.0
    for name in names:
        dc_cap_total += float(np.nanmean(mi[name]["DC_cap_inv"]))

    y_ac_site = np.where(dc_cap_total > 0, e_ac_site / dc_cap_total, np.nan)
    pr_site   = np.where((y_ref_site == 0) | np.isnan(y_ref_site), np.nan, y_ac_site / y_ref_site)

    df = pd.DataFrame({
        "Time": mi[names[0]]["time"],
        "Σ E_ac (kWh)": np.round(e_ac_site, 6),
        "Σ Potential (kWh)": np.round(e_pot_site, 6),
        "Reference Yield (h at STC) — site mean": np.round(y_ref_site, 6),
        "Specific Yield (h = kWh/kWp) — site": np.round(y_ac_site, 6),
        "Performance Ratio (% at STC) — site": np.round(pr_site*100.0, 3),
    })
    st.data_editor(
        df, key="site_10min_table", use_container_width=True, disabled=True,
        column_config={
            "Σ E_ac (kWh)": st.column_config.NumberColumn("Σ E_ac (kWh)", format="%.3f"),
            "Σ Potential (kWh)": st.column_config.NumberColumn("Σ Potential (kWh)", format="%.3f"),
            "Reference Yield (h at STC) — site mean": st.column_config.NumberColumn("Reference Yield (h at STC) — site mean", format="%.3f"),
            "Specific Yield (h = kWh/kWp) — site": st.column_config.NumberColumn("Specific Yield (h = kWh/kWp) — site", format="%.3f"),
            "Performance Ratio (% at STC) — site": st.column_config.NumberColumn("Performance Ratio (% at STC) — site", format="%.2f"),
        }
    )

    # Daily totals (site)
    total_eac_site  = float(np.nansum(e_ac_site))
    total_pot_site  = float(np.nansum(e_pot_site))
    y_ref_day_site  = float(np.nansum(y_ref_site)) if np.isfinite(np.nansum(y_ref_site)) else np.nan
    y_ac_day_site   = (total_eac_site / dc_cap_total) if dc_cap_total > 0 else np.nan
    pr_day_site     = (y_ac_day_site / y_ref_day_site) if (np.isfinite(y_ref_day_site) and y_ref_day_site != 0) else np.nan

    totals = pd.DataFrame([{
        "Σ E_ac (kWh)": total_eac_site if np.isfinite(total_eac_site) else np.nan,
        "Σ Potential (kWh)": total_pot_site if np.isfinite(total_pot_site) else np.nan,
        "Reference Yield — day (h)": y_ref_day_site if np.isfinite(y_ref_day_site) else np.nan,
        "Specific Yield — day (h)": y_ac_day_site if np.isfinite(y_ac_day_site) else np.nan,
        "Performance Ratio — day (%)": pr_day_site*100 if np.isfinite(pr_day_site) else np.nan,
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
            "Specific Yield — day (h)": st.column_config.NumberColumn("Specific Yield — day (h)", format="%.3f",
                help="Σ E_ac_site / Σ DC_cap (approx via mean of inverter DCs)."),
            "Performance Ratio — day (%)": st.column_config.NumberColumn("Performance Ratio — day (%)", format="%.2f",
                help="Site Y_ac_day / Site Y_ref_day × 100."),
        }
    )

# ---------------------------
# Public entry
# ---------------------------
def page_intermediate():
    t1, t2, t3, t4 = st.tabs(["Inverter 1", "Inverter 2", "Inverter 3", "Site"])
    with t1:
        _inverter_page("INV1", "inv1")
    with t2:
        _inverter_page("INV2", "inv2")
    with t3:
        _inverter_page("INV3", "inv3")
    with t4:
        _site_page()
