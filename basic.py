# basic.py
import math
import numpy as np
import pandas as pd
import streamlit as st

def _yields_pr(poa_wm2, t_mod_c, ref_irr, ref_temp, gamma, e_ac_kwh, dc_cap_kwp):
    """
    Compute per-interval Reference Yield (hours), Specific Yield (hours), and PR.
    Assumptions:
      - poa_wm2: 10-min average POA in W/m².
      - t_mod_c: module temperature in °C.
      - ref_irr: reference irradiance in W/m² (e.g., 1000).
      - e_ac_kwh: AC energy per 10 minutes in kWh.
      - dc_cap_kwp: DC capacity per row (kWp).
    """
    DT_HOURS = 1.0 / 6.0  # 10 minutes

    temp_factor = 1.0 + gamma * (t_mod_c - ref_temp)
    denom = ref_irr * temp_factor
    y_ref_ratio = np.where(np.isclose(denom, 0.0), np.nan, poa_wm2 / denom)
    y_ref_h = y_ref_ratio * DT_HOURS  # convert ratio to hours over the 10-min window

    y_ac_h = np.where(dc_cap_kwp > 0, e_ac_kwh / dc_cap_kwp, np.nan)  # kWh/kWp = hours

    pr = np.where((y_ref_h == 0) | np.isnan(y_ref_h), np.nan, y_ac_h / y_ref_h)
    return y_ref_h, y_ac_h, pr

def page_basic(ref_irr, ref_temp, gamma, _dc_cap_unused, e_pot_mode, pr_target):
    """
    BASIC page:
      - DC capacity is per-row in the input table (DC_cap_kWp), not from sidebar.
      - Potential (Proxy only): min(proxy_yield_h * DC_cap_row, AC_Max_Project/6).
      - Daily Y_ref uses SUM over 10-min windows; PR_day becomes consistent.
    """
    st.title("BASIC")

    # Project-level settings from sidebar (already stored in session_state by KPI.py)
    proxy_yield_h = float(st.session_state.get("proxy_yield_h", 0.90))        # h
    ac_max_kw     = float(st.session_state.get("ac_max_project_kw", 1800.0))  # kW
    ac_cap_kwh    = ac_max_kw / 6.0  # kWh per 10-min window

    # --------------------------
    # Inputs (editable table)
    # --------------------------
    st.markdown("### Inputs DATA (per 10-min window)")
    default_df = pd.DataFrame({
        "Time": ["10:00", "10:10", "10:20", "10:30", "10:40"],
        "POA_Wm2": [700.0, 820.0, 910.0, 860.0, 760.0],
        "T_module_C": [42.0, 44.5, 46.5, 45.0, 43.0],
        "AC_kWh": [180.0, 210.0, 230.0, 220.0, 190.0],
        "DC_cap_kWp": [2000.0, 2000.0, 2000.0, 2000.0, 2000.0],  # per-row DC capacity
    })

    edited = st.data_editor(
        default_df,
        num_rows="dynamic",
        key="basic_editor_inputs_proxy_final",
        use_container_width=True,
        column_config={
            "Time": st.column_config.TextColumn(
                "Time (hh:mm)",
                help="Timestamp (optional) for the 10-min window."
            ),
            "POA_Wm2": st.column_config.NumberColumn(
                "POA (W/m²)",
                min_value=0.0, step=1.0,
                help="Plane of Array irradiance (10-min average, W/m²)."
            ),
            "T_module_C": st.column_config.NumberColumn(
                "Module Temp (°C)",
                step=0.1,
                help="Module temperature (back or estimated) in °C."
            ),
            "AC_kWh": st.column_config.NumberColumn(
                "AC Energy (kWh / 10 min)",
                min_value=0.0, step=0.001,
                help="Measured AC energy during the 10-min window (kWh)."
            ),
            "DC_cap_kWp": st.column_config.NumberColumn(
                "DC capacity (kWp)",
                min_value=0.0, step=1.0,
                help="Per-row DC capacity used for Y_ac and potential."
            ),
        },
    )

    if edited.empty:
        st.info("No rows to compute.")
        return

    # --------------------------
    # Validations
    # --------------------------
    if ref_irr <= 0:
        st.error("Ref. Irradiance must be > 0.")
        return

    dc_cap_arr = np.array(edited["DC_cap_kWp"], dtype=float)
    if np.any(~np.isfinite(dc_cap_arr)) or np.all(dc_cap_arr <= 0):
        st.error("DC capacity column (DC_cap_kWp) must have positive numeric values.")
        return
    if np.nanstd(dc_cap_arr) > 1e-6:
        st.warning("DC_cap_kWp varies across rows. If unintended, set it constant.")

    # --------------------------
    # KPI calculations
    # --------------------------
    poa  = np.array(edited["POA_Wm2"], dtype=float)
    tmod = np.array(edited["T_module_C"], dtype=float)
    eac  = np.array(edited["AC_kWh"], dtype=float)

    y_ref, y_ac, pr_arr = _yields_pr(poa, tmod, ref_irr, ref_temp, gamma, eac, dc_cap_arr)

    # Potential energy (Proxy only, per 10-min)
    e_proxy_raw = proxy_yield_h * dc_cap_arr
    e_potential = np.minimum(e_proxy_raw, ac_cap_kwh)

    # --------------------------
    # KPI table (per 10-min)
    # --------------------------
    st.markdown("### KPI Data (per 10-min window)")
    kpi_df = pd.DataFrame({
        "Reference Yield (h at STC)": np.round(y_ref, 6),
        "Specific Yield (h = kWh/kWp)": np.round(y_ac, 6),
        "Performance Ratio (% at STC)": np.round(pr_arr * 100.0, 3),
        "Potential energy (kWh)": np.round(e_potential, 6),
    })

    st.data_editor(
        kpi_df,
        key="basic_kpi_table_potential_proxy_final",
        use_container_width=True,
        disabled=True,
        column_config={
            "Reference Yield (h at STC)": st.column_config.NumberColumn(
                "Reference Yield (h at STC)",
                help="POA / [RefIrr × (1 + Gamma × (T_mod − T_ref))] × (10 min = 1/6 h)."
            ),
            "Specific Yield (h = kWh/kWp)": st.column_config.NumberColumn(
                "Specific Yield (h = kWh/kWp)",
                help="E_ac / DC_cap_kWp(row)."
            ),
            "Performance Ratio (% at STC)": st.column_config.NumberColumn(
                "Performance Ratio (% at STC)",
                help="(Specific Yield) / (Reference Yield) × 100."
            ),
            "Potential energy (kWh)": st.column_config.NumberColumn(
                "Potential energy (kWh)",
                help="min(Proxy_Specific_Yield × DC_cap_kWp(row), AC_Max_Power/6)."
            ),
        },
    )

    # --------------------------
    # Daily totals & aggregated KPIs
    # --------------------------
    st.markdown("### Daily Totals & Aggregated KPIs")

    total_eac   = float(np.nansum(eac))
    total_pot   = float(np.nansum(e_potential))

    # Effective DC is mean of per-row DC (OK when DC is constant; acceptable approximation otherwise)
    dc_cap_eff = float(np.nanmean(dc_cap_arr)) if np.isfinite(np.nanmean(dc_cap_arr)) else np.nan
    y_ac_day   = (total_eac / dc_cap_eff) if (isinstance(dc_cap_eff, float) and dc_cap_eff > 0) else float("nan")

    # IMPORTANT: daily Reference Yield is SUM over all 10-min intervals (already in hours)
    y_ref_day  = float(np.nansum(y_ref)) if np.isfinite(np.nansum(y_ref)) else float("nan")

    pr_day     = (y_ac_day / y_ref_day) if (math.isfinite(y_ref_day) and y_ref_day != 0) else float("nan")

    totals_df = pd.DataFrame([{
        "Σ E_ac (kWh)": total_eac if math.isfinite(total_eac) else np.nan,
        "Σ Potential energy (kWh)": total_pot if math.isfinite(total_pot) else np.nan,
        "Reference Yield — day (h)": y_ref_day if math.isfinite(y_ref_day) else np.nan,
        "Specific Yield — day (h)": y_ac_day if math.isfinite(y_ac_day) else np.nan,
        "Performance Ratio — day (%)": pr_day*100 if math.isfinite(pr_day) else np.nan,
    }])

    st.data_editor(
        totals_df,
        key="basic_totals_hover_only_final",
        use_container_width=True,
        disabled=True,
        column_config={
            "Σ E_ac (kWh)": st.column_config.NumberColumn(
                "Σ E_ac (kWh)",
                help="Total measured AC energy over the day: Σ E_ac."
            ),
            "Σ Potential energy (kWh)": st.column_config.NumberColumn(
                "Σ Potential energy (kWh)",
                help="Sum of per-interval min(Proxy_Yield × DC_cap_row, AC_Max/6)."
            ),
            "Reference Yield — day (h)": st.column_config.NumberColumn(
                "Reference Yield — day (h)",
                help="SUM of Reference Yield (hours) over all 10-min intervals."
            ),
            "Specific Yield — day (h)": st.column_config.NumberColumn(
                "Specific Yield — day (h)",
                help="Σ E_ac / mean(DC_cap_kWp)."
            ),
            "Performance Ratio — day (%)": st.column_config.NumberColumn(
                "Performance Ratio — day (%)",
                help="Y_ac_day / Y_ref_day × 100."
            ),
        },
    )
