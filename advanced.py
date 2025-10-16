

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from typing import List

# Import Intermediate (original UI)
try:
    import intermediate as mi_mod
except Exception:
    mi_mod = None

INV_NAMES = ["INV1", "INV2", "INV3"]
SLOT_MINUTES = 10

# ---------------------------
# Helpers
# ---------------------------

def _require_mi():
    mi = st.session_state.get("mi_outputs")
    if not mi or not all(k in mi for k in INV_NAMES):
        st.warning("Open Intermediate → Inverter 1/2/3 first so E_potential & E_ac are available.")
        return {}
    return mi


def _timeline(mi: dict) -> List[str]:
    return list(mi[INV_NAMES[0]].get("time", []))


def _time_to_minutes(t: str) -> int:
    try:
        hh, mm = t.split(":")
        return int(hh)*60 + int(mm)
    except Exception:
        return 0


def _minutes_to_time(total_min: int) -> str:
    hh = (total_min // 60) % 24
    mm = total_min % 60
    return f"{hh:02d}:{mm:02d}"


def _overlap_minutes(slot_start_min: int, slot_end_min: int, ev_start: str|None, ev_end: str|None) -> int:
    if not ev_start or not ev_end:
        return 0
    try:
        s = _time_to_minutes(ev_start)
        e = _time_to_minutes(ev_end)
        if e <= s:
            return 0
        a = max(slot_start_min, s)
        b = min(slot_end_min, e)
        return max(0, b - a)
    except Exception:
        return 0

# ---------------------------
# Events grid (defaults with examples)
# ---------------------------

def _events_table_state(scope: str, times: List[str]) -> pd.DataFrame:
    key = f"adv_events_grid_{scope}"
    if key not in st.session_state:
        df = pd.DataFrame({
            "Time": times,
            "Event 1": [None]*len(times),
            "Event1 Start": [None]*len(times),
            "Event1 End": [None]*len(times),
            "Event 2": [None]*len(times),
            "Event2 Start": [None]*len(times),
            "Event2 End": [None]*len(times),
        })
        starts_min = [_time_to_minutes(t) for t in times]
        for i, smin in enumerate(starts_min):
            emin = smin + SLOT_MINUTES
            if i % 5 == 0:
                df.loc[i, "Event 1"] = "Curtailment"
                df.loc[i, "Event1 Start"] = _minutes_to_time(smin)
                df.loc[i, "Event1 End"] = _minutes_to_time(emin)
            elif i % 5 == 1:
                df.loc[i, "Event 1"] = "OMC"
                df.loc[i, "Event1 Start"] = _minutes_to_time(smin)
                df.loc[i, "Event1 End"] = _minutes_to_time(smin+5)
            elif i % 5 == 2:
                df.loc[i, "Event 1"] = "OMC"
                df.loc[i, "Event1 Start"] = _minutes_to_time(smin)
                df.loc[i, "Event1 End"] = _minutes_to_time(smin+6)
                df.loc[i, "Event 2"] = "UMC"
                df.loc[i, "Event2 Start"] = _minutes_to_time(smin+6)
                df.loc[i, "Event2 End"] = _minutes_to_time(emin)
        st.session_state[key] = df
    return st.session_state[key]


def _compute_weights_availability(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    slot_starts = out["Time"].apply(_time_to_minutes)
    slot_ends = slot_starts + SLOT_MINUTES
    ov1 = [_overlap_minutes(s,e,out.loc[i,"Event1 Start"],out.loc[i,"Event1 End"]) for i,(s,e) in enumerate(zip(slot_starts,slot_ends))]
    ov2 = [_overlap_minutes(s,e,out.loc[i,"Event2 Start"],out.loc[i,"Event2 End"]) for i,(s,e) in enumerate(zip(slot_starts,slot_ends))]
    ov1, ov2 = np.array(ov1,float), np.array(ov2,float)
    sum_ov = ov1 + ov2
    with np.errstate(divide='ignore', invalid='ignore'):
        wf1 = np.where(sum_ov>0, ov1/sum_ov, 0.0)
        wf2 = np.where(sum_ov>0, ov2/sum_ov, 0.0)
    av1 = np.where(ov1>0, 0, 1)
    av2 = np.where(ov2>0, 0, 1)
    ew_global = np.clip(wf1+wf2, 0.0, 1.0)
    av_global = np.where(sum_ov>0, 0, 1)
    out["Weighting Factor 1"], out["Availability 1"] = np.round(wf1,3), av1
    out["Weighting Factor 2"], out["Availability 2"] = np.round(wf2,3), av2
    out["Event Weight (Global)"] = np.round(ew_global,3)
    out["Availability (Global)"] = av_global
    return out

# ---------------------------
# Losses & PBA (DC‑AC only)
# ---------------------------

def _losses_pba_per10(potential: np.ndarray, measured: np.ndarray, ew_global: np.ndarray):
    potential = np.asarray(potential, float)
    measured = np.asarray(measured, float)
    ew_global = np.asarray(ew_global, float)
    loss_base = np.maximum(0.0, potential - measured)
    all_lost = loss_base * np.clip(ew_global, 0.0, 1.0)
    pba = np.where(potential>0, (potential - all_lost)/potential * 100.0, np.nan)
    return all_lost, pba

# ---------------------------
# Inverter tab (renders Intermediate but suppresses its own Daily table)
# ---------------------------

def _inv_tab(inv: str, key_sfx: str):
    

    mi_all = st.session_state.get("mi_outputs", {})
    # Temporarily clear Intermediate daily table to avoid duplicate daily tables
    orig_daily = None
    if mi_mod and hasattr(mi_mod, "_inverter_page"):
        try:
            if inv in mi_all and isinstance(mi_all[inv].get("daily_kpi", None), pd.DataFrame):
                orig_daily = mi_all[inv]["daily_kpi"].copy()
                mi_all[inv]["daily_kpi"] = pd.DataFrame()
        except Exception:
            pass
        mi_mod._inverter_page(inv, f"adv_{key_sfx}")
        # Restore for our own merge
        try:
            if orig_daily is not None:
                mi_all[inv]["daily_kpi"] = orig_daily
        except Exception:
            pass

    mi = _require_mi()
    if not mi:
        return

    times = _timeline(mi)
    pot = np.asarray(mi[inv].get("E_potential", [np.nan]*len(times)), float)
    ac  = np.asarray(mi[inv].get("E_ac", [np.nan]*len(times)), float)

    # Table 1: Events & Weights & Availability (per 10‑min)
    st.markdown("### Events & Weights & Availability (per 10‑min)")
    grid = _events_table_state(inv, times)
    grid = st.data_editor(grid, key=f"adv_grid_{key_sfx}", use_container_width=True)
    comp = _compute_weights_availability(grid)
    st.data_editor(
        comp[["Time","Event 1","Event1 Start","Event1 End","Weighting Factor 1","Availability 1","Event 2","Event2 Start","Event2 End","Weighting Factor 2","Availability 2","Event Weight (Global)","Availability (Global)"]],
        key=f"adv_comp_{key_sfx}", disabled=True, use_container_width=True,
    )

    # Table 2: Losses & PBA (per 10‑min)
    st.markdown("### Losses & PBA (per 10‑min)")
    ew = comp["Event Weight (Global)"].to_numpy(float)
    all_lost, pba = _losses_pba_per10(pot, ac, ew)
    per10 = pd.DataFrame({
        "Time": times,
        "AllLost_kWh_10m": np.round(all_lost, 4),
        "PBA_10m_%": np.round(pba, 2),
    })
    st.data_editor(per10, key=f"adv_loss_{key_sfx}", disabled=True, use_container_width=True)

    # Final ONE merged Daily table
    total_lost = float(np.nansum(all_lost))
    pba_den = float(np.nansum(pot))
    pba_day = (1 - total_lost/pba_den)*100 if pba_den>0 else np.nan

    inter_daily = mi[inv].get("daily_kpi", pd.DataFrame())
    add = pd.DataFrame([{ "All Lost (kWh)": round(total_lost,4), "All PBA (%)": round(pba_day,2)}])
    merged = pd.concat([inter_daily.reset_index(drop=True), add], axis=1) if isinstance(inter_daily, pd.DataFrame) and not inter_daily.empty else add

    st.markdown("### Daily Totals & Aggregated KPIs")
    st.data_editor(merged, key=f"adv_daily_{key_sfx}", disabled=True, use_container_width=True)

# ---------------------------
# Site tab (same idea)
# ---------------------------

def _site_tab():
    st.markdown("## Site — Advanced")

    mi_all = st.session_state.get("mi_outputs", {})
    # Hide Intermediate site daily table
    orig_site_daily = None
    if mi_mod and hasattr(mi_mod, "_site_page"):
        try:
            if isinstance(mi_all.get("site_daily_kpi", None), pd.DataFrame):
                orig_site_daily = mi_all["site_daily_kpi"].copy()
                mi_all["site_daily_kpi"] = pd.DataFrame()
        except Exception:
            pass
        try:
            mi_mod._site_page()
        except Exception:
            st.warning("Skipped Intermediate Site UI to avoid duplicate keys.")
        # restore
        try:
            if orig_site_daily is not None:
                mi_all["site_daily_kpi"] = orig_site_daily
        except Exception:
            pass

    mi = _require_mi()
    if not mi:
        return

    times = _timeline(mi)
    # aggregate site arrays
    pot_site = np.zeros(len(times))
    ac_site  = np.zeros(len(times))
    for inv in INV_NAMES:
        pot_site += np.nan_to_num(np.asarray(mi[inv].get("E_potential", [np.nan]*len(times)), float), nan=0.0)
        ac_site  += np.nan_to_num(np.asarray(mi[inv].get("E_ac", [np.nan]*len(times)), float), nan=0.0)

    # Table 1 (Site)
    st.markdown("### Events & Weights & Availability (per 10‑min) — Site")
    grid = _events_table_state("site", times)
    grid = st.data_editor(grid, key="adv_grid_site", use_container_width=True)
    comp = _compute_weights_availability(grid)
    st.data_editor(
        comp[["Time","Event 1","Event1 Start","Event1 End","Weighting Factor 1","Availability 1","Event 2","Event2 Start","Event2 End","Weighting Factor 2","Availability 2","Event Weight (Global)","Availability (Global)"]],
        key="adv_comp_site", disabled=True, use_container_width=True,
    )

    # Table 2 (Site)
    st.markdown("### Losses & PBA (per 10‑min) — Site")
    ew = comp["Event Weight (Global)"].to_numpy(float)
    all_lost, pba = _losses_pba_per10(pot_site, ac_site, ew)
    per10 = pd.DataFrame({
        "Time": times,
        "AllLost_kWh_10m": np.round(all_lost, 4),
        "PBA_10m_%": np.round(pba, 2),
    })
    st.data_editor(per10, key="adv_loss_site", disabled=True, use_container_width=True)

    # Final ONE merged Daily table (Site)
    total_lost = float(np.nansum(all_lost))
    pba_den = float(np.nansum(pot_site))
    pba_day = (1 - total_lost/pba_den)*100 if pba_den>0 else np.nan

    inter_daily = mi_all.get("site_daily_kpi", pd.DataFrame())
    add = pd.DataFrame([{ "All Lost (kWh)": round(total_lost,4), "All PBA (%)": round(pba_day,2)}])
    merged = pd.concat([inter_daily.reset_index(drop=True), add], axis=1) if isinstance(inter_daily, pd.DataFrame) and not inter_daily.empty else add

    st.markdown("### Daily Totals & Aggregated KPIs")
    st.data_editor(merged, key="adv_daily_site", disabled=True, use_container_width=True)

# ---------------------------
# Entry point
# ---------------------------

def page_advanced():
    
    t1,t2,t3,t4 = st.tabs(["Inverter 1","Inverter 2","Inverter 3","Site"])
    with t1: _inv_tab("INV1","inv1")
    with t2: _inv_tab("INV2","inv2")
    with t3: _inv_tab("INV3","inv3")
    with t4: _site_tab()

if __name__ == "__main__":
    page_advanced()