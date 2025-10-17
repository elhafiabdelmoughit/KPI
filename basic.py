# basic.py
# --------------------------------------------------------------------------------------
# PV KPI - BASIC page (per 10-min window) — version complète avec substitutions visibles
# et détails de sommes "a + b + c = total" pour tous les agrégés.
# --------------------------------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import streamlit as st

DT_HOURS = 1.0 / 6.0  # 10 minutes en heures


def _yields_pr(poa_wm2, t_mod_c, ref_irr, ref_temp, gamma, e_ac_kwh, dc_cap_kwp):
    """
    Calcule par pas:
      - ReferenceYield_h (heures)
      - SpecificYield_h (heures = kWh/kWp)
      - PR (ratio, pas en %)
    """
    temp_factor = 1.0 + gamma * (t_mod_c - ref_temp)
    denom = ref_irr * temp_factor  # W/m²
    y_ref_ratio = np.where(np.isclose(denom, 0.0), np.nan, poa_wm2 / denom)
    reference_yield_h = y_ref_ratio * DT_HOURS  # heures sur la fenêtre

    specific_yield_h = np.where(dc_cap_kwp > 0, e_ac_kwh / dc_cap_kwp, np.nan)

    pr_ratio = np.where(
        (reference_yield_h == 0) | np.isnan(reference_yield_h),
        np.nan,
        specific_yield_h / reference_yield_h
    )
    return reference_yield_h, specific_yield_h, pr_ratio


def _calc_details_one_row(row, ref_irr, ref_temp, gamma, proxy_yield_h, ac_cap_kwh):
    """
    Détail numérique complet pour un pas de 10 min (dict).
    """
    poa = float(row["POA_Wm2"])
    tmod = float(row["T_module_C"])
    eac = float(row["AC_kWh"])
    dc_cap = float(row["DC_cap_kWp"])

    # (1) Temperature factor
    temp_factor = 1.0 + gamma * (tmod - ref_temp)

    # (2) Denominator
    denom = ref_irr * temp_factor

    # (3) Reference Yield (heures)
    if math.isclose(denom, 0.0):
        reference_yield_h = float("nan")
    else:
        reference_yield_h = (poa / denom) * DT_HOURS

    # (4) Specific Yield (heures = kWh/kWp)
    specific_yield_h = (eac / dc_cap) if dc_cap > 0 else float("nan")

    # (5) PR (%) à STC
    pr_ratio = (specific_yield_h / reference_yield_h) if (
        math.isfinite(reference_yield_h) and reference_yield_h != 0
    ) else float("nan")
    pr_percent = pr_ratio * 100.0 if math.isfinite(pr_ratio) else float("nan")

    # (6) Potential energy (kWh), borne AC/6
    e_proxy_raw = proxy_yield_h * dc_cap
    e_potential = min(e_proxy_raw, ac_cap_kwh)

    return {
        # Entrées
        "POA_Wm2": poa,
        "T_module_C": tmod,
        "AC_kWh": eac,
        "DC_cap_kWp": dc_cap,
        "Ref_Irr_Wm2": ref_irr,
        "Ref_Temp_C": ref_temp,
        "Gamma_per_C": gamma,
        "Proxy_Yield_h": proxy_yield_h,
        "AC_cap_kWh_per_10min": ac_cap_kwh,
        # Intermédiaires
        "Temp_Factor": temp_factor,
        "Denominator_Wm2": denom,
        "ReferenceYield_h": reference_yield_h,
        "SpecificYield_h": specific_yield_h,
        # Sorties
        "PR_percent": pr_percent,
        "E_proxy_raw_kWh": e_proxy_raw,
        "E_potential_kWh": e_potential,
    }


def page_basic(ref_irr, ref_temp, gamma, _dc_cap_unused, e_pot_mode=None, pr_target=None, **kwargs):
    """
    BASIC page :
      - DC_cap_kWp est par ligne (table), pas depuis la sidebar.
      - Potential energy (proxy) : min(Proxy_Yield_h * DC_cap_kWp, AC_max/6).
      - ReferenceYield_day_h = somme des pas (déjà en heures) ⇒ PR_day cohérent.
      - e_pot_mode / pr_target acceptés (optionnels) pour compatibilité avec KPI.py.
    """
    st.title("BASIC")

    # Réglages projet (stockés par KPI.py dans session_state)
    proxy_yield_h = float(st.session_state.get("proxy_yield_h", 0.90))           # h (par pas)
    ac_max_kw     = float(st.session_state.get("ac_max_project_kw", 1800.0))     # kW
    ac_cap_kwh    = ac_max_kw / 6.0  # kWh par pas de 10 min

    # --------------------------
    # Inputs (editable table)
    # --------------------------
    st.markdown("### Inputs DATA (per 10-min window)")
    default_df = pd.DataFrame({
        "Time": ["10:00", "10:10", "10:20", "10:30", "10:40"],
        "POA_Wm2": [700.0, 820.0, 910.0, 860.0, 760.0],
        "T_module_C": [42.0, 44.5, 46.5, 45.0, 43.0],
        "AC_kWh": [180.0, 210.0, 230.0, 220.0, 190.0],
        "DC_cap_kWp": [2000.0, 2000.0, 2000.0, 2000.0, 2000.0],  # kWp
    })

    edited = st.data_editor(
        default_df,
        num_rows="dynamic",
        key="basic_editor_inputs_proxy_final",
        use_container_width=True,
        column_config={
            "Time": st.column_config.TextColumn(
                "Time (hh:mm)",
                help="Horodatage (optionnel) du pas de 10 min."
            ),
            "POA_Wm2": st.column_config.NumberColumn(
                "POA (W/m²)",
                min_value=0.0, step=1.0,
                help="Plane of Array irradiance (moyenne 10 min)."
            ),
            "T_module_C": st.column_config.NumberColumn(
                "Module Temp (°C)",
                step=0.1,
                help="Température module (arrière ou estimée)."
            ),
            "AC_kWh": st.column_config.NumberColumn(
                "AC Energy (kWh / 10 min)",
                min_value=0.0, step=0.001,
                help="Énergie AC mesurée durant les 10 min."
            ),
            "DC_cap_kWp": st.column_config.NumberColumn(
                "DC capacity (kWp)",
                min_value=0.0, step=1.0,
                help="Capacité DC utilisée pour Specific Yield et potentiel."
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
    # KPI calculations (per 10-min)
    # --------------------------
    poa  = np.array(edited["POA_Wm2"], dtype=float)
    tmod = np.array(edited["T_module_C"], dtype=float)
    eac  = np.array(edited["AC_kWh"], dtype=float)

    y_ref_h, y_ac_h, pr_ratio = _yields_pr(poa, tmod, ref_irr, ref_temp, gamma, eac, dc_cap_arr)

    # Potential (proxy), borné par AC/6
    e_proxy_raw = proxy_yield_h * dc_cap_arr
    e_potential = np.minimum(e_proxy_raw, ac_cap_kwh)

    # --------------------------
    # KPI table (per 10-min)
    # --------------------------
    st.markdown("### KPI Data (per 10-min window)")
    kpi_df = pd.DataFrame({
        "Reference Yield (h at STC)": np.round(y_ref_h, 6),
        "Specific Yield (h = kWh/kWp)": np.round(y_ac_h, 6),
        "Performance Ratio (% at STC)": np.round(pr_ratio * 100.0, 3),
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
                help="(POA / [RefIrr × (1 + Gamma × (T_mod − T_ref))]) × (10 min = 1/6 h)."
            ),
            "Specific Yield (h = kWh/kWp)": st.column_config.NumberColumn(
                "Specific Yield (h = kWh/kWp)",
                help="AC_kWh / DC_cap_kWp."
            ),
            "Performance Ratio (% at STC)": st.column_config.NumberColumn(
                "Performance Ratio (% at STC)",
                help="(Specific Yield / Reference Yield) × 100."
            ),
            "Potential energy (kWh)": st.column_config.NumberColumn(
                "Potential energy (kWh)",
                help="min(Proxy_Yield_h × DC_cap_kWp, AC_max/6)."
            ),
        },
    )

    # --------------------------
    # Détails numériques (par pas) — avec substitutions visibles
    # --------------------------
    with st.expander("Détails de calcul (application numérique) — par pas de 10 min", expanded=False):
        st.caption("Sélectionne un pas de temps pour voir les formules et la substitution numérique.")
        times = edited.get("Time", pd.Series([f"Row {i}" for i in range(len(edited))])).tolist()
        labels = [f"{i:02d} — {t}" for i, t in enumerate(times)]
        idx = st.selectbox("Choisir le pas de 10 min", options=list(range(len(labels))), format_func=lambda i: labels[i])

        row = edited.iloc[idx]
        details = _calc_details_one_row(row, ref_irr, ref_temp, gamma, proxy_yield_h, ac_cap_kwh)

        # Entrées
        st.markdown("**Entrées (pas sélectionné)**")
        tbl_inputs = pd.DataFrame([{
            "POA (W/m²)": np.round(details["POA_Wm2"], 6),
            "T_mod (°C)": np.round(details["T_module_C"], 6),
            "E_ac (kWh)": np.round(details["AC_KWh"] if "AC_KWh" in details else details["AC_kWh"], 6),
            "DC_cap (kWp)": np.round(details["DC_cap_kWp"], 6),
            "RefIrr (W/m²)": np.round(details["Ref_Irr_Wm2"], 6),
            "T_ref (°C)": np.round(details["Ref_Temp_C"], 6),
            "Gamma (1/°C)": np.round(details["Gamma_per_C"], 6),
            "Proxy_Yield_h (h)": np.round(details["Proxy_Yield_h"], 6),
            "AC_cap/6 (kWh)": np.round(details["AC_cap_kWh_per_10min"], 6),
        }])
        st.dataframe(tbl_inputs, use_container_width=True, hide_index=True)

        # Formules LaTeX + substitution (avec opérations visibles)
        st.markdown("**Formules et substitution numérique (opérations montrées)**")

        # (1) Temperature factor
        st.latex(r"T_{\text{factor}} = 1 + \gamma \cdot (T_{\mathrm{mod}} - T_{\mathrm{ref}})")
        st.write(
            f"= 1 + ({details['Gamma_per_C']}) × ({details['T_module_C']} − {details['Ref_Temp_C']}) "
            f"= **{np.round(details['Temp_Factor'], 6)}**"
        )

        # (2) Denominator
        st.latex(r"\text{denom} = \text{RefIrr} \times T_{\text{factor}}")
        st.write(
            f"= {details['Ref_Irr_Wm2']} × {np.round(details['Temp_Factor'], 6)} "
            f"= **{np.round(details['Denominator_Wm2'], 6)} W/m²**"
        )

        # (3) Reference Yield (heures)
        st.latex(r"\text{ReferenceYield}_{(h)} = \frac{\mathrm{POA}}{\text{denom}} \times \frac{1}{6}")
        if math.isfinite(details["ReferenceYield_h"]):
            st.write(
                f"= ({np.round(details['POA_Wm2'], 6)} ÷ {np.round(details['Denominator_Wm2'], 6)}) × (1/6) "
                f"→ **{np.round(details['ReferenceYield_h'], 6)} h**"
            )
        else:
            st.write("= NaN (denominator ≈ 0)")

        # (4) Specific Yield (heures = kWh/kWp)
        st.latex(r"\text{SpecificYield}_{(h)} = \frac{E_{\mathrm{ac}}(\mathrm{kWh})}{\mathrm{DC\_cap}(\mathrm{kWp})}")
        if math.isfinite(details["SpecificYield_h"]):
            st.write(
                f"= {np.round(details['AC_kWh'], 6)} ÷ {np.round(details['DC_cap_kWp'], 6)} "
                f"→ **{np.round(details['SpecificYield_h'], 6)} h (kWh/kWp)**"
            )
        else:
            st.write("= NaN (DC_cap ≤ 0)")

        # (5) PR (%)
        st.latex(r"\mathrm{PR}(\%) = \frac{\text{SpecificYield}_{(h)}}{\text{ReferenceYield}_{(h)}} \times 100")
        if math.isfinite(details["PR_percent"]):
            st.write(
                f"= ({np.round(details['SpecificYield_h'], 6)} ÷ {np.round(details['ReferenceYield_h'], 6)}) × 100 "
                f"→ **{np.round(details['PR_percent'], 3)} %**"
            )
        else:
            st.write("= NaN (ReferenceYield ≈ 0 ou invalide)")

        # (6) Potential energy (kWh)
        st.latex(r"E_{\text{potential}}(\mathrm{kWh}) = \min\!\Big(\,\text{Proxy\_Yield}_{(h)} \times \mathrm{DC\_cap},\; \frac{\mathrm{AC\_max}}{6}\Big)")
        st.write(
            f"= min({np.round(details['Proxy_Yield_h'], 6)} × {np.round(details['DC_cap_kWp'], 6)}, "
            f"{np.round(details['AC_cap_kWh_per_10min'], 6)}) "
            f"= min({np.round(details['E_proxy_raw_kWh'], 6)}, {np.round(details['AC_cap_kWh_per_10min'], 6)}) "
            f"→ **{np.round(details['E_potential_kWh'], 6)} kWh**"
        )

    # --------------------------
    # Daily totals & aggregated KPIs
    # --------------------------
    st.markdown("### Daily Totals & Aggregated KPIs")

    total_eac   = float(np.nansum(eac))
    total_pot   = float(np.nansum(e_potential))

    # DC_cap_eff_kWp : moyenne (OK si DC_cap_kWp constant)
    dc_cap_eff = float(np.nanmean(dc_cap_arr)) if np.isfinite(np.nanmean(dc_cap_arr)) else np.nan
    y_ac_day   = (total_eac / dc_cap_eff) if (isinstance(dc_cap_eff, float) and dc_cap_eff > 0) else float("nan")

    # Somme des ReferenceYield_h (déjà en heures)
    y_ref_day  = float(np.nansum(y_ref_h)) if np.isfinite(np.nansum(y_ref_h)) else float("nan")

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
                help="Somme des énergies AC mesurées : Σ E_ac."
            ),
            "Σ Potential energy (kWh)": st.column_config.NumberColumn(
                "Σ Potential energy (kWh)",
                help="Somme des min(Proxy_Yield_h × DC_cap_kWp, AC_max/6)."
            ),
            "Reference Yield — day (h)": st.column_config.NumberColumn(
                "Reference Yield — day (h)",
                help="Somme des ReferenceYield_h sur tous les pas."
            ),
            "Specific Yield — day (h)": st.column_config.NumberColumn(
                "Specific Yield — day (h)",
                help="Σ E_ac / DC_cap_eff_kWp."
            ),
            "Performance Ratio — day (%)": st.column_config.NumberColumn(
                "Performance Ratio — day (%)",
                help="(SpecificYield_day_h / ReferenceYield_day_h) × 100."
            ),
        },
    )

    # --------------------------
    # Détails numériques (agrégés jour) — toutes les sommes affichées "a + b + c = total"
    # --------------------------
    with st.expander("Détails de calcul (application numérique) — agrégés (jour)", expanded=False):
        st.caption("Sommes détaillées et substitutions visibles pour tous les agrégés.")

        def _fmt(x, n=6):
            try:
                x = float(x)
                return f"{np.round(x, n)}" if math.isfinite(x) else "NaN"
            except Exception:
                return "NaN"

        def _list_vals(arr, n=3):
            return [f"{np.round(v, n)}" for v in np.asarray(arr).tolist() if math.isfinite(v)]

        # Construire chaînes "a + b + c"
        eac_terms = _list_vals(eac, 3)
        pot_terms = _list_vals(e_potential, 3)
        yref_terms = _list_vals(y_ref_h, 6)  # plus de précision pour les heures
        dccap_terms = _list_vals(dc_cap_arr, 3)

        eac_sum_expr = " + ".join(eac_terms) if eac_terms else "NaN"
        pot_sum_expr = " + ".join(pot_terms) if pot_terms else "NaN"
        yref_sum_expr = " + ".join(yref_terms) if yref_terms else "NaN"
        dccap_sum_expr = " + ".join(dccap_terms) if dccap_terms else "NaN"

        n_rows = len(eac_terms)

        # ---- Σ E_ac (kWh)
        st.latex(r"\sum E_{\mathrm{ac}}(\mathrm{kWh}) = \sum_{i=1}^{N} E_{\mathrm{ac},i}")
        st.write(f"= {eac_sum_expr} = **{_fmt(total_eac, 3)} kWh**")

        # ---- DC_cap_eff (kWp) : valeur constante
        st.latex(r"\mathrm{DC\_cap\_eff}(\mathrm{kWp}) = \text{constante du site}")
        st.write(f"= **{_fmt(dc_cap_eff, 3)} kWp (statique)**")


        # ---- SpecificYield_day(h) = ΣE_ac / DC_cap_eff
        st.latex(r"\text{SpecificYield}_{\text{day}}(h) = \frac{\sum E_{\mathrm{ac}}(\mathrm{kWh})}{\mathrm{DC\_cap\_eff}(\mathrm{kWp})}")
        st.write(
            f"= {_fmt(total_eac, 3)} ÷ {_fmt(dc_cap_eff, 6)} "
            f"→ **{_fmt((total_eac / dc_cap_eff) if dc_cap_eff and math.isfinite(dc_cap_eff) else float('nan'), 6)} h**"
        )

        # ---- Σ ReferenceYield(h)
        st.latex(r"\sum \text{ReferenceYield}_{(h)}")
        st.write(f"= {yref_sum_expr} = **{_fmt(y_ref_day, 6)} h**")

        # ---- PR_day(%)
        st.latex(r"\mathrm{PR}_{\text{day}}(\%) = \frac{\text{SpecificYield}_{\text{day}}(h)}{\text{ReferenceYield}_{\text{day}}(h)} \times 100")
        pr_txt = (
            (total_eac / dc_cap_eff) / y_ref_day * 100.0
            if all(map(lambda v: isinstance(v, float) and math.isfinite(v) and v != 0, [total_eac, dc_cap_eff, y_ref_day]))
            else float("nan")
        )
        st.write(
            f"= ({_fmt(total_eac / dc_cap_eff, 6)} ÷ {_fmt(y_ref_day, 6)}) × 100 "
            f"→ **{_fmt(pr_txt, 3)} %**"
        )

        # ---- Σ E_potential (kWh)
        st.latex(r"\sum E_{\text{potential}}(\mathrm{kWh}) = \sum \min\!\Big(\,\text{Proxy\_Yield}_{(h)} \times \mathrm{DC\_cap},\; \frac{\mathrm{AC\_max}}{6}\Big)")
        st.write(f"= {pot_sum_expr} = **{_fmt(total_pot, 3)} kWh**")
