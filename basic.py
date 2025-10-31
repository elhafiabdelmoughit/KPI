# basic.py
# --------------------------------------------------------------------------------------
# PV KPI - BASIC (per 10-min window) — single global DC capacity + Proxy Yield beside it
# - Removes per-row DC_cap_kWp column from the inputs table.
# - Adds TWO project-level controls side-by-side: DC capacity (kWp) and Proxy Yield (h/10min).
# - Uses safe LaTeX templates (no f-strings) for numeric "one-line" formulas.
# - Shows daily sums as "a + b + c = total".
# --------------------------------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import streamlit as st

DT_HOURS = 1.0 / 6.0  # 10 minutes in hours


def _yields_pr_scalar_dc(poa_wm2, t_mod_c, ref_irr, ref_temp, gamma, e_ac_kwh, dc_cap_kwp_scalar):
    """
    Compute per-interval KPIs using a single (global) DC capacity:
      - ReferenceYield_h (hours)
      - SpecificYield_h (hours = kWh/kWp)
      - PR (ratio, not %)
    Inputs are numpy arrays for poa, t_mod_c, e_ac_kwh; dc_cap_kwp_scalar is a float.
    """
    temp_factor = 1.0 + gamma * (t_mod_c - ref_temp)
    denom = ref_irr * temp_factor  # W/m²
    y_ref_ratio = np.where(np.isclose(denom, 0.0), np.nan, poa_wm2 / denom)
    reference_yield_h = y_ref_ratio * DT_HOURS  # hours per 10-min window

    dc = float(dc_cap_kwp_scalar)
    specific_yield_h = np.where(dc > 0, e_ac_kwh / dc, np.nan)

    pr_ratio = np.where(
        (reference_yield_h == 0) | np.isnan(reference_yield_h),
        np.nan,
        specific_yield_h / reference_yield_h
    )
    return reference_yield_h, specific_yield_h, pr_ratio


def _calc_details_one_row_scalar_dc(poa, tmod, eac, ref_irr, ref_temp, gamma, proxy_yield_h, ac_cap_kwh, dc_cap_kwp_scalar):
    """
    Full numeric breakdown for one 10-min step with single (global) DC capacity.
    """
    # (1) Temperature factor
    temp_factor = 1.0 + gamma * (tmod - ref_temp)

    # (2) Denominator
    denom = ref_irr * temp_factor

    # (3) Reference Yield (h)
    if math.isclose(denom, 0.0):
        reference_yield_h = float("nan")
    else:
        reference_yield_h = (poa / denom) * DT_HOURS

    # (4) Specific Yield (h = kWh/kWp) using single DC capacity
    dc = float(dc_cap_kwp_scalar)
    specific_yield_h = (eac / dc) if dc > 0 else float("nan")

    # (5) PR (%)
    pr_ratio = (specific_yield_h / reference_yield_h) if (
        math.isfinite(reference_yield_h) and reference_yield_h != 0
    ) else float("nan")
    pr_percent = pr_ratio * 100.0 if math.isfinite(pr_ratio) else float("nan")

    # (6) Potential energy (kWh) via proxy, capped by AC/6
    e_proxy_raw = proxy_yield_h * dc
    e_potential = min(e_proxy_raw, ac_cap_kwh)

    return {
        # Inputs
        "POA_Wm2": poa,
        "T_module_C": tmod,
        "AC_kWh": eac,
        "DC_cap_kWp_global": dc,
        "Ref_Irr_Wm2": ref_irr,
        "Ref_Temp_C": ref_temp,
        "Gamma_per_C": gamma,
        "Proxy_Yield_h": proxy_yield_h,
        "AC_cap_kWh_per_10min": ac_cap_kwh,
        # Intermediates
        "Temp_Factor": temp_factor,
        "Denominator_Wm2": denom,
        "ReferenceYield_h": reference_yield_h,
        "SpecificYield_h": specific_yield_h,
        # Outputs
        "PR_percent": pr_percent,
        "E_proxy_raw_kWh": e_proxy_raw,
        "E_potential_kWh": e_potential,
    }


def page_basic(ref_irr, ref_temp, gamma, _dc_cap_unused, e_pot_mode=None, pr_target=None, **kwargs):
    """
    BASIC page with single DC capacity + Proxy Yield (side-by-side controls):
      - Removes per-row DC capacity column; use one global DC capacity (kWp) for all rows.
      - Proxy Yield (hours per 10-min) is set beside DC capacity and used in potential-energy.
      - Daily ReferenceYield is SUM over intervals (already in hours); PR_day consistent.
    """
    st.title("BASIC")
           # ------------------------------------------------------------------
    # Short description (info icon)
    # ------------------------------------------------------------------
    with st.expander("ℹ️ About this section (BASIC)", expanded=False):
        st.markdown("""
        **Objective:**  
        This section provides a simplified analysis of PV plant performance using 10-minute data intervals.  
        It calculates key performance indicators (KPIs) such as **Reference Yield**, **Specific Yield**,  
        **Performance Ratio (PR)**, and **Potential Energy**, based on a single DC capacity and Proxy Yield.  

        **Goal:**  
        To give a quick overview of the plant’s operational efficiency by comparing measured AC energy  
        with the expected (potential) energy under reference conditions.
        """)


    # Project-level settings (defaults if not set by KPI.py)
    default_proxy = float(st.session_state.get("proxy_yield_h", 0.90))           # h per 10-min
    ac_max_kw     = float(st.session_state.get("ac_max_project_kw", 1800.0))     # kW
    ac_cap_kwh    = ac_max_kw / 6.0  # kWh per 10-min window

        # ------------------------------------------------------------------
    # Global project settings — DC capacity, AC Max Power & Proxy Yield (side-by-side)
    # ------------------------------------------------------------------
    st.markdown("### Project Settings")
    col_dc, col_ac, col_proxy = st.columns([0.33, 0.33, 0.34])

    with col_dc:
        st.markdown("####  DC Capacity")
        dc_cap_global_kwp = st.number_input(
            "DC capacity (kWp) — global",
            min_value=0.0,
            value=float(st.session_state.get("dc_cap_site_kwp", 2000.0)),
            step=10.0,
            help="Single site DC capacity used for all calculations (kWp).",
            key="dc_cap_global_input",
        )
        st.session_state["dc_cap_site_kwp"] = dc_cap_global_kwp

    with col_ac:
        st.markdown("####  AC Max Power ")
        ac_max_kw = st.number_input(
            "AC Max Power (project) [kW]",
            min_value=0.0,
            value=float(st.session_state.get("ac_max_project_kw", 1800.0)),
            step=10.0,
            help="Maximum AC power limit of the project (kW). Used to cap potential energy per 10-min.",
            key="ac_max_input",
        )
        st.session_state["ac_max_project_kw"] = ac_max_kw
        # convert to 10-min energy equivalent (kWh)
        ac_cap_kwh = ac_max_kw / 6.0

    with col_proxy:
        st.markdown("####  Proxy Yield ")
        proxy_yield_h = st.number_input(
            "Proxy Yield (h per 10-min)",
            min_value=0.0,
            value=float(st.session_state.get("proxy_yield_h", 0.90)),
            step=0.01,
            help="Expected specific yield (hours) per 10-min window used for potential energy.",
            key="proxy_yield_input",
        )
        st.session_state["proxy_yield_h"] = proxy_yield_h

    # ------------------------------------------------------------------
    # Inputs (editable table) — without DC_cap column
    # ------------------------------------------------------------------
    st.markdown("### Inputs DATA (per 10-min window)")
    default_df = pd.DataFrame({
        "Time": ["10:00", "10:10", "10:20", "10:30", "10:40"],
        "POA_Wm2": [700.0, 820.0, 910.0, 860.0, 760.0],
        "T_module_C": [42.0, 44.5, 46.5, 45.0, 43.0],
        "AC_kWh": [180.0, 210.0, 230.0, 220.0, 190.0],
    })

    edited = st.data_editor(
        default_df,
        num_rows="dynamic",
        key="basic_editor_inputs_proxy_globaldc",
        use_container_width=True,
        column_config={
            "Time": st.column_config.TextColumn(
                "Time (hh:mm)",
                help="Optional timestamp for the 10-min window."
            ),
            "POA_Wm2": st.column_config.NumberColumn(
                "POA (W/m²)",
                min_value=0.0, step=1.0,
                help="Plane of Array irradiance (10-min average, W/m²)."
            ),
            "T_module_C": st.column_config.NumberColumn(
                "Module Temp (°C)",
                step=0.1,
                help="Module temperature in °C."
            ),
            "AC_kWh": st.column_config.NumberColumn(
                "AC Energy (kWh / 10 min)",
                min_value=0.0, step=0.001,
                help="Measured AC energy during the 10-min window (kWh)."
            ),
        },
    )

    if edited.empty:
        st.info("No rows to compute.")
        return

    # ------------------------------------------------------------------
    # KPI calculations (per 10-min) — with single DC capacity
    # ------------------------------------------------------------------
    poa  = np.array(edited["POA_Wm2"], dtype=float)
    tmod = np.array(edited["T_module_C"], dtype=float)
    eac  = np.array(edited["AC_kWh"], dtype=float)

    y_ref_h, y_ac_h, pr_ratio = _yields_pr_scalar_dc(
        poa, tmod, ref_irr, ref_temp, gamma, eac, dc_cap_global_kwp
    )

    # Potential (proxy), capped by AC/6 (same per-row since DC is global)
    e_proxy_raw = proxy_yield_h * dc_cap_global_kwp
    e_potential = np.full_like(eac, np.minimum(e_proxy_raw, ac_cap_kwh), dtype=float)

    # ------------------------------------------------------------------
    # KPI table (per 10-min)
    # ------------------------------------------------------------------
    st.markdown("### KPI Data (per 10-min window)")
    kpi_df = pd.DataFrame({
        "Reference Yield (h at STC)": np.round(y_ref_h, 6),
        "Specific Yield (h = kWh/kWp)": np.round(y_ac_h, 6),
        "Performance Ratio (% at STC)": np.round(pr_ratio * 100.0, 3),
        "Potential energy (kWh)": np.round(e_potential, 6),
    })
    st.data_editor(
        kpi_df,
        key="basic_kpi_table_potential_proxy_globaldc",
        use_container_width=True,
        disabled=True,
        column_config={
            "Reference Yield (h at STC)": st.column_config.NumberColumn(
                "Reference Yield (h at STC)",
                help="(POA / [RefIrr × (1 + Gamma × (T_mod − T_ref))]) × (10 min = 1/6 h)."
            ),
            "Specific Yield (h = kWh/kWp)": st.column_config.NumberColumn(
                "Specific Yield (h = kWh/kWp)",
                help="E_ac / DC_cap_global (kWp)."
            ),
            "Performance Ratio (% at STC)": st.column_config.NumberColumn(
                "Performance Ratio (% at STC)",
                help="(Specific Yield / Reference Yield) × 100."
            ),
            "Potential energy (kWh)": st.column_config.NumberColumn(
                "Potential energy (kWh)",
                help="min(Proxy_Yield_h × DC_cap_global, AC_max/6)."
            ),
        },
    )

    # ------------------------------------------------------------------
    # LaTeX helper: safe templates (avoid f-strings with braces)
    # ------------------------------------------------------------------
    def render_latex_template(template: str, **vals):
        """
        Safe LaTeX builder:
        - Use [[ ... ]] instead of { ... } for LaTeX braces
        - Use <name> placeholders for numeric values
        """
        s = template
        for k, v in vals.items():
            s = s.replace(f"<{k}>", str(v))
        s = s.replace("[[", "{").replace("]]", "}")
        st.latex(s)

    # ------------------------------------------------------------------
    # Details (per 10-min) — application numérique in ONE line per KPI
    # ------------------------------------------------------------------
    with st.expander("Détails de calcul ", expanded=False):

        # Which interval?
        times = edited.get("Time", pd.Series([f"Row {i}" for i in range(len(edited))])).tolist()
        labels = [f"{i:02d} — {t}" for i, t in enumerate(times)]
        idx = st.selectbox("Choisir le pas de 10 min", options=list(range(len(labels))), format_func=lambda i: labels[i])

        # Row values
        row = edited.iloc[idx]
        details = _calc_details_one_row_scalar_dc(
            float(row["POA_Wm2"]),
            float(row["T_module_C"]),
            float(row["AC_kWh"]),
            ref_irr, ref_temp, gamma,
            proxy_yield_h, ac_cap_kwh,
            dc_cap_global_kwp,
        )

        # --- Entrées (pas sélectionné) ---
        st.markdown("**Entrées (pas sélectionné)**")
        def _safe_float(v):
            try:
                f = float(v)
                return f if math.isfinite(f) else np.nan
            except Exception:
                return np.nan

        tbl_inputs = pd.DataFrame([{
            "POA (W/m²)":             _safe_float(details["POA_Wm2"]),
            "T_mod (°C)":             _safe_float(details["T_module_C"]),
            "E_ac (kWh)":             _safe_float(details["AC_kWh"]),
            "DC_cap (kWp) — global":  _safe_float(details["DC_cap_kWp_global"]),
            "RefIrr (W/m²)":          _safe_float(details["Ref_Irr_Wm2"]),
            "T_ref (°C)":             _safe_float(details["Ref_Temp_C"]),
            "Gamma (1/°C)":           _safe_float(details["Gamma_per_C"]),
            "Proxy_Yield_h (h)":      _safe_float(details["Proxy_Yield_h"]),
            "AC_cap/6 (kWh)":         _safe_float(details["AC_cap_kWh_per_10min"]),
        }])
        st.dataframe(tbl_inputs, use_container_width=True, hide_index=True)

        # values for LaTeX lines (rounded strings)
        def _r(x, n=6):
            try:
                x = float(x)
                return f"{np.round(x, n)}" if math.isfinite(x) else "NaN"
            except Exception:
                return "NaN"

        poa_n   = _r(details["POA_Wm2"], 3)
        refirr  = _r(details["Ref_Irr_Wm2"], 3)
        gamma_n = _r(details["Gamma_per_C"], 6)
        tmod_n  = _r(details["T_module_C"], 3)
        tref_n  = _r(details["Ref_Temp_C"], 3)
        yref_n  = _r(details["ReferenceYield_h"], 6)
        yac_n   = _r(details["SpecificYield_h"], 6)
        prpct_n = _r(details["PR_percent"], 3)
        dc_n    = _r(details["DC_cap_kWp_global"], 3)
        eac_n   = _r(details["AC_kWh"], 3)
        py_n    = _r(details["Proxy_Yield_h"], 6)
        accap6  = _r(details["AC_cap_kWh_per_10min"], 3)
        epot_n  = _r(details["E_potential_kWh"], 6)

        st.markdown("**Application numérique**")

        # 1) Reference Yield (h)
        render_latex_template(
            r"Y_{[[\mathrm{ref}]]}(h)=\frac{[[\mathrm{POA}]]}{[[\mathrm{RefIrr}]]\cdot\left(1+\gamma\cdot(T_{[[\mathrm{mod}]]}-T_{[[\mathrm{ref}]]})\right)}\cdot\frac{1}{6}"
            r"=\frac{<poa>}{<refirr>\cdot\left(1+<gamma>\cdot(<tmod>-<tref>)\right)}\cdot\frac{1}{6}"
            r"=<yref>\ \mathrm{h}",
            poa=poa_n, refirr=refirr, gamma=gamma_n, tmod=tmod_n, tref=tref_n, yref=yref_n
        )

        # 2) Specific Yield (h = kWh/kWp)
        render_latex_template(
            r"\mathrm{SpecificYield}(h)=\frac{E_{[[\mathrm{ac}]]}(\mathrm{kWh})}{[[\mathrm{DC\_cap}]](\mathrm{kWp})}"
            r"=\frac{<eac>}{<dc>}"
            r"=<yac>\ \mathrm{h}",
            eac=eac_n, dc=dc_n, yac=yac_n
        )

        # 3) Performance Ratio (%)
        render_latex_template(
            r"\mathrm{PR}(\%)=\frac{\mathrm{SpecificYield}(h)}{Y_{[[\mathrm{ref}]]}(h)}\times 100"
            r"=\left(\frac{<yac>}{<yref>}\right)\times 100"
            r"=<pr>\ \%",
            yac=yac_n, yref=yref_n, pr=prpct_n
        )

        # 4) Potential energy (kWh) — proxy, cap AC/6
        render_latex_template(
            r"E_{[[\mathrm{potential}]]}(\mathrm{kWh})=\min\!\Big(\mathrm{Proxy\_Yield}(h)\times [[\mathrm{DC\_cap}]](\mathrm{kWp}),\ \frac{[[\mathrm{AC\_max}]]}{6}\Big)"
            r"=\min\!\Big(<py>\times <dc>,\ <cap6>\Big)"
            r"=<epot>\ \mathrm{kWh}",
            py=py_n, dc=dc_n, cap6=accap6, epot=epot_n
        )

    # ------------------------------------------------------------------
    # Daily totals & aggregated KPIs
    # ------------------------------------------------------------------
    st.markdown("### Daily Totals & Aggregated KPIs")

    total_eac   = float(np.nansum(eac))
    total_pot   = float(np.nansum(e_potential))
    # SpecificYield_day uses the same global DC
    y_ac_day    = (total_eac / dc_cap_global_kwp) if dc_cap_global_kwp > 0 else float("nan")
    # ReferenceYield_day is SUM of per-interval Y_ref hours
    y_ref_day   = float(np.nansum(y_ref_h)) if np.isfinite(np.nansum(y_ref_h)) else float("nan")
    pr_day      = (y_ac_day / y_ref_day) if (math.isfinite(y_ref_day) and y_ref_day != 0) else float("nan")

    totals_df = pd.DataFrame([{
        "Σ E_ac (kWh)": total_eac if math.isfinite(total_eac) else np.nan,
        "Σ Potential energy (kWh)": total_pot if math.isfinite(total_pot) else np.nan,
        "Reference Yield — day (h)": y_ref_day if math.isfinite(y_ref_day) else np.nan,
        "Specific Yield — day (h)": y_ac_day if math.isfinite(y_ac_day) else np.nan,
        "Performance Ratio — day (%)": pr_day*100 if math.isfinite(pr_day) else np.nan,
    }])
    st.data_editor(
        totals_df,
        key="basic_totals_hover_only_globaldc",
        use_container_width=True,
        disabled=True,
        column_config={
            "Σ E_ac (kWh)": st.column_config.NumberColumn(
                "Σ E_ac (kWh)",
                help="Total measured AC energy over the day: Σ E_ac."
            ),
            "Σ Potential energy (kWh)": st.column_config.NumberColumn(
                "Σ Potential energy (kWh)",
                help="Sum of per-interval min(Proxy_Yield × DC_cap_global, AC_Max/6)."
            ),
            "Reference Yield — day (h)": st.column_config.NumberColumn(
                "Reference Yield — day (h)",
                help="SUM of Reference Yield (hours) over all 10-min intervals."
            ),
            "Specific Yield — day (h)": st.column_config.NumberColumn(
                "Specific Yield — day (h)",
                help="Σ E_ac / DC_cap_global."
            ),
            "Performance Ratio — day (%)": st.column_config.NumberColumn(
                "Performance Ratio — day (%)",
                help="SpecificYield_day / ReferenceYield_day × 100."
            ),
        },
    )

    # ------------------------------------------------------------------
    # Aggregated details — sums shown as "a + b + c = total"
    # ------------------------------------------------------------------
    with st.expander("Détails de calcul ", expanded=False):

        def _fmt(x, n=6):
            try:
                x = float(x)
                return f"{np.round(x, n)}" if math.isfinite(x) else "NaN"
            except Exception:
                return "NaN"

        def _list_vals(arr, n=3):
            return [f"{np.round(v, n)}" for v in np.asarray(arr).tolist() if math.isfinite(v)]

        eac_terms  = _list_vals(eac, 3)
        pot_terms  = _list_vals(e_potential, 3)
        yref_terms = _list_vals(y_ref_h, 6)

        eac_sum_expr  = " + ".join(eac_terms)  if eac_terms  else "NaN"
        pot_sum_expr  = " + ".join(pot_terms)  if pot_terms  else "NaN"
        yref_sum_expr = " + ".join(yref_terms) if yref_terms else "NaN"

        # Σ E_ac
        st.latex(r"\sum E_{\mathrm{ac}}(\mathrm{kWh}) = \sum_{i=1}^{N} E_{\mathrm{ac},i}")
        st.write(f"= {eac_sum_expr} = **{_fmt(total_eac, 3)} kWh**")

        # DC capacity shown as constant
        st.latex(r"\mathrm{DC\_cap\_global}(\mathrm{kWp}) = \text{constante du site}")
        st.write(f"= **{_fmt(dc_cap_global_kwp, 3)} kWp**")

        # SpecificYield_day
        st.latex(r"\text{SpecificYield}_{\text{day}}(h) = \frac{\sum E_{\mathrm{ac}}(\mathrm{kWh})}{\mathrm{DC\_cap\_global}(\mathrm{kWp})}")
        st.write(f"= {_fmt(total_eac, 3)} ÷ {_fmt(dc_cap_global_kwp, 6)} → **{_fmt(y_ac_day, 6)} h**")

        # Σ ReferenceYield(h)
        st.latex(r"\sum \text{ReferenceYield}_{(h)}")
        st.write(f"= {yref_sum_expr} = **{_fmt(y_ref_day, 6)} h**")

        # PR_day
        st.latex(r"\mathrm{PR}_{\text{day}}(\%) = \frac{\text{SpecificYield}_{\text{day}}(h)}{\text{ReferenceYield}_{\text{day}}(h)} \times 100")
        st.write(
            f"= ({_fmt(y_ac_day, 6)} ÷ {_fmt(y_ref_day, 6)}) × 100 "
            f"→ **{_fmt(pr_day * 100.0, 3)} %**"
        )

        # Σ E_potential
        st.latex(r"\sum E_{\text{potential}}(\mathrm{kWh}) = \sum \min\!\Big(\,\text{Proxy\_Yield}_{(h)} \times \mathrm{DC\_cap\_global},\; \frac{\mathrm{AC\_max}}{6}\Big)")
        st.write(f"= {pot_sum_expr} = **{_fmt(total_pot, 3)} kWh**")
       # ------------------------------------------------------------------
    # Final Summary Table — Global Daily Results
    # ------------------------------------------------------------------
    st.markdown("###  Global Daily Energy Summary")

    # --- Calculs principaux ---
    gross_inverter_prod = total_eac  # Σ E_ac (kWh)
    ref_poi_prod = total_eac * 0.97  # 3% loss to POI
    all_losses = total_pot - ref_poi_prod  # Σ Potential − Reference POI
    all_pr_stc = pr_day * 100 if math.isfinite(pr_day) else np.nan

    # ✅ Relation correcte:
    # All PBA = (Σ E_potential − All losses) / Σ E_potential
    all_pba = ((total_pot - all_losses) / total_pot) if total_pot > 0 else np.nan

    # --- Tableau récapitulatif ---
    summary_df = pd.DataFrame([{
        "Gross inverter production (kWh)": gross_inverter_prod,
        "Reference POI Production (kWh)": ref_poi_prod,
        "All losses (kWh)": all_losses,
        "All PR STC (%)": all_pr_stc,
        "All PBA": all_pba,
    }])

    st.data_editor(
        summary_df,
        key="basic_summary_final_global",
        use_container_width=True,
        disabled=True,
        column_config={
            "Gross inverter production (kWh)": st.column_config.NumberColumn(
                "Gross inverter production (kWh)",
                help="Sum of all measured AC energy: Σ E_ac (kWh)."
            ),
            "Reference POI Production (kWh)": st.column_config.NumberColumn(
                "Reference POI Production (kWh)",
                help="Σ E_ac × 0.97 (accounts for 3% POI loss)."
            ),
            "All losses (kWh)": st.column_config.NumberColumn(
                "All losses (kWh)",
                help="Σ Potential energy − Reference POI Production."
            ),
            "All PR STC (%)": st.column_config.NumberColumn(
                "All PR STC (%)",
                help="Daily performance ratio at STC."
            ),
            "All PBA": st.column_config.NumberColumn(
                "All PBA",
                help="(Σ E_potential − All losses) / Σ E_potential."
            ),
        },
    )
