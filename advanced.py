from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Optional

SLOT_MINUTES = 10
INV_CHOICES = ["INV1", "INV2", "INV3"]

# ---------------------------
# Time helpers
# ---------------------------
def _time_to_minutes(t: Optional[str]) -> int:
    if not t or not isinstance(t, str):
        return 0
    try:
        hh, mm = t.split(":")
        return int(hh) * 60 + int(mm)
    except Exception:
        return 0

def _minutes_to_time(total_min: int) -> str:
    hh = (total_min // 60) % 24
    mm = total_min % 60
    return f"{hh:02d}:{mm:02d}"

def _overlap_minutes(slot_start_min: int, slot_end_min: int, ev_start: Optional[str], ev_end: Optional[str]) -> int:
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
# Defaults
# ---------------------------
def _default_times(n: int = 5, start: str = "10:00") -> List[str]:
    h, m = map(int, start.split(":"))
    base = h * 60 + m
    return [_minutes_to_time(base + i * SLOT_MINUTES) for i in range(n)]

def _default_inputs() -> pd.DataFrame:
    times = _default_times(5, "10:00")
    return pd.DataFrame({
        "Time": times,
        "Potential_kWh": [220.0, 210.0, 200.0, 195.0, 185.0],
        "AC_kWh_Site":  [200.0, 205.0, 198.0, 190.0, 180.0],
    })

# ---------- Global two-events defaults ----------
def _default_global_two_events() -> pd.DataFrame:
    # Two editable events that can last minutes → hours
    return pd.DataFrame([
        {"EventID": "E1", "Inv": "INV1", "IEC": "00110", "Event": "OMC",         "Start": "10:05", "End": "11:20"},
        {"EventID": "E2", "Inv": "INV2", "IEC": "0002x", "Event": "Curtailment", "Start": "10:15", "End": "10:55"},
    ])

def _sanitize_two_events(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["EventID","Inv","IEC","Event","Start","End"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    if len(df) == 0:
        return _default_global_two_events()
    return df.iloc[:2][cols].reset_index(drop=True)

# ---------------------------
# IEC → reference availability (simplified mapping)
# ---------------------------
def _iec_ref_availability(iec: Optional[str]) -> int:
    """
    Return 1 (available) or 0 (downtime) based on simplified IEC rules.
    - startswith '0002' (partial curtailment) → 0
    - equals '00101' (test OK) → 1
    - numeric >= 100 (i.e., ≥00100) → 0
    - else (<00100) → 1
    Non-parsable / None → treat as available (1).
    """
    if not iec or not isinstance(iec, str):
        return 1
    s = iec.strip().lower()
    if s.startswith("0002"):
        return 0
    if s == "00101":
        return 1
    digits = ''.join(ch for ch in s if ch.isdigit())
    if len(digits) >= 3:
        try:
            val = int(digits[:5]) if len(digits) >= 5 else int(digits)
            return 0 if val >= 100 else 1
        except Exception:
            return 1
    return 1

def _ref_av_reason(iec: Optional[str]) -> str:
    if not iec or not isinstance(iec, str):
        return "No IEC provided → default available (1)"
    s = iec.strip().lower()
    if s.startswith("0002"):
        return "0002x (Partial Curtailment) ⇒ ref_av = 0"
    if s == "00101":
        return "00101 (Test OK) ⇒ ref_av = 1"
    digits = ''.join(ch for ch in s if ch.isdigit())
    if len(digits) >= 3:
        try:
            val = int(digits[:5]) if len(digits) >= 5 else int(digits)
            return (f"{val:05d} ≥ 00100 (Downtime) ⇒ ref_av = 0"
                    if val >= 100 else
                    f"{val:05d} < 00100 (Normal/Monitoring) ⇒ ref_av = 1")
        except Exception:
            return "Non-parsable digits → default available (1)"
    return "Non-parsable IEC → default available (1)"

# ---------------------------
# 10-min slot grid from events
# ---------------------------
def _round_down_to_slot_minutes(mins: int) -> int:
    return (mins // SLOT_MINUTES) * SLOT_MINUTES

def _round_up_to_slot_minutes(mins: int) -> int:
    return mins if mins % SLOT_MINUTES == 0 else ((mins // SLOT_MINUTES) + 1) * SLOT_MINUTES

def _slots_from_events(ev2_df: pd.DataFrame) -> List[str]:
    """Build a 10-min slot list that spans from min(Start) to max(End) across the two global events."""
    if ev2_df is None or ev2_df.empty:
        return []
    starts = []
    ends = []
    for _, r in ev2_df.iterrows():
        s = r.get("Start"); e = r.get("End")
        if isinstance(s, str) and isinstance(e, str):
            smin = _time_to_minutes(s)
            emin = _time_to_minutes(e)
            if emin > smin:
                starts.append(smin)
                ends.append(emin)
    if not starts:
        return []
    g_start = min(starts)
    g_end = max(ends)

    grid_start = _round_down_to_slot_minutes(g_start)
    grid_end = _round_up_to_slot_minutes(g_end)

    slots = []
    cur = grid_start
    while cur < grid_end:
        slots.append(_minutes_to_time(cur))
        cur += SLOT_MINUTES
    return slots

def _ensure_inputs_cover_events(inputs_df: pd.DataFrame, ev2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure inputs_df has a row for every 10-min slot spanned by the two events.
    If missing, append with default Potential/AC = 0.0 (user can edit later).
    """
    needed = _slots_from_events(ev2_df)
    if not needed:
        return inputs_df.copy()

    base = inputs_df.copy()
    base["Time"] = base["Time"].astype(str)

    have = set(base["Time"].tolist())
    add_times = [t for t in needed if t not in have]

    if add_times:
        extra = pd.DataFrame({
            "Time": add_times,
            "Potential_kWh": [0.0] * len(add_times),
            "AC_kWh_Site":  [0.0] * len(add_times),
        })
        base = pd.concat([base, extra], ignore_index=True)

    base = base.sort_values(by="Time").reset_index(drop=True)
    return base

# ---------------------------
# Computations (two global events → per-slot projection)
# ---------------------------
def _compute_weights_availability_from_global_two_slots(slot_times: List[str], ev2_df: pd.DataFrame) -> pd.DataFrame:
    """Project exactly two global events over the provided 10-min slot starts."""
    e1 = ev2_df.iloc[0] if len(ev2_df) >= 1 else pd.Series()
    e2 = ev2_df.iloc[1] if len(ev2_df) >= 2 else pd.Series()

    inv1 = str(e1.get("Inv")) if "Inv" in e1 else None
    inv2 = str(e2.get("Inv")) if "Inv" in e2 else None
    iec1 = e1.get("IEC") if "IEC" in e1 else None
    iec2 = e2.get("IEC") if "IEC" in e2 else None
    rav1 = _iec_ref_availability(iec1)
    rav2 = _iec_ref_availability(iec2)

    out = []
    for t in slot_times:
        smin = _time_to_minutes(t)
        emin = smin + SLOT_MINUTES

        ov1 = _overlap_minutes(smin, emin, e1.get("Start"), e1.get("End")) if len(ev2_df) >= 1 else 0
        ov2 = _overlap_minutes(smin, emin, e2.get("Start"), e2.get("End")) if len(ev2_df) >= 2 else 0
        denom = (ov1 + ov2)
        wf1 = (ov1 / denom) if denom > 0 else 0.0
        wf2 = (ov2 / denom) if denom > 0 else 0.0

        any_overlap_downtime = (ov1 > 0 and rav1 == 0) or (ov2 > 0 and rav2 == 0)
        av_global = 0 if any_overlap_downtime else 1

        out.append({
            "Time": t,
            "Inv_1": inv1, "IEC_1": iec1, "Overlap_1_min": ov1, "wf1 (distribution)": round(wf1, 3),
            "Inv_2": inv2, "IEC_2": iec2, "Overlap_2_min": ov2, "wf2 (distribution)": round(wf2, 3),
            "ref_av_1 (computed)": rav1,
            "ref_av_2 (computed)": rav2,
            "Event Weight (Global)": round(min(1.0, wf1 + wf2), 3),
            "Availability (Global)": av_global,
        })
    return pd.DataFrame(out)

def _compute_losses(inputs_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    # Join by Time
    base = pd.merge(
        inputs_df[["Time", "Potential_kWh", "AC_kWh_Site"]].copy(),
        weights_df[[
            "Time",
            "Inv_1","Inv_2",
            "wf1 (distribution)","wf2 (distribution)",
            "ref_av_1 (computed)","ref_av_2 (computed)"
        ]].copy(),
        on="Time", how="left"
    ).copy()

    # Replace NaNs early
    base = base.fillna(0)

    # Gap = max(0, Potential - AC)
    pot = pd.to_numeric(base["Potential_kWh"], errors="coerce").fillna(0.0)
    ac  = pd.to_numeric(base["AC_kWh_Site"],   errors="coerce").fillna(0.0)
    gap = np.maximum(0.0, pot - ac)

    # Weights
    w1 = pd.to_numeric(base["wf1 (distribution)"], errors="coerce").fillna(0.0)
    w2 = pd.to_numeric(base["wf2 (distribution)"], errors="coerce").fillna(0.0)

    # Cast ref_av to {0,1} strictly
    def _to01(x):
        try:
            return 1 if float(x) >= 0.5 else 0
        except Exception:
            return 1  # default available

    r1 = base["ref_av_1 (computed)"].apply(_to01).astype(int)
    r2 = base["ref_av_2 (computed)"].apply(_to01).astype(int)

    # Apply the relation exactly: Lost_i = (1 - ref_av_i) * wf_i * Gap
    lost1 = (1.0 - r1) * w1 * gap
    lost2 = (1.0 - r2) * w2 * gap

    base["Gap_kWh"] = np.round(gap, 6)
    base["Lost_kWh_event1"] = np.round(lost1, 6)
    base["Lost_kWh_event2"] = np.round(lost2, 6)
    base["Lost_kWh_slot"]   = np.round(lost1 + lost2, 6)

    # Keep inverter tags as strings
    base["Inv_1"] = base["Inv_1"].astype("string")
    base["Inv_2"] = base["Inv_2"].astype("string")

    return base[[
        "Time","Gap_kWh",
        "Inv_1","Lost_kWh_event1",
        "Inv_2","Lost_kWh_event2",
        "Lost_kWh_slot"
    ]]

# ---------------------------
# UI
# ---------------------------
def _about():
    with st.expander("ℹ️ About this section (ADVANCED )", expanded=False):
        st.markdown("""
### **Purpose**
This Advanced section computes **DC–AC-only losses** at the **site level** .  
You provide:
- Site-level energy data (`Potential_kWh`, `AC_kWh_Site`),  
- Exactly two global events (each with an `IEC` code, inverter tag, and start/end time).

The program automatically:
- Extends the time grid to cover all 10-minute intervals crossed by the events,  
- Computes **event overlaps** and **distribution weights** (`wf₁`, `wf₂`),  
- Determines **reference availability** (`ref_av`) from IEC codes,  
- Calculates the energy **Gap** and allocates losses only for events where `ref_av = 0`,  
- Provides a **global summary**: Total Lost Energy, Total Potential, and All PBA ratio.

---

### **Key Concepts**

| Symbol | Description |
|:-------|:-------------|
| `Gap_kWh` | Energy difference between potential and measured AC |
| `wf₁`, `wf₂` | Event distribution weights (proportional to overlap) |
| `ref_av` | Reference availability (0 = downtime, 1 = available) |
| `Lost_kWh_event1/2` | Lost energy attributed to each event |
| `Lost_kWh_slot` | Total lost energy per 10-minute slot |
| `All PBA` | Global performance ratio after applying losses |

---

### **IEC Codes — Simplified Mapping**

| IEC Code | Description | ref_av | Typical Meaning |
|:----------|:-------------|:-------:|:----------------|
| `00101` | Test OK / Normal Operation | 1 | No fault detected — available |
| `00100–00119` | Downtime / Maintenance | 0 | Full shutdown or corrective maintenance |
| `00110` | Corrective Maintenance (OMC) | 0 | Planned or unplanned downtime |
| `0002x` | Curtailment (partial) | 0 | External or grid curtailment |
| `0001x` | Normal Monitoring | 1 | Observation or data validation only |
| _None or Unknown_ | Default treated as available | 1 | No IEC provided, assumed available |

**Rule summary:**
- IEC ≥ 00100 → Downtime (`ref_av = 0`)  
- IEC < 00100 → Available (`ref_av = 1`)  
- Codes starting with `0002` → Curtailment (`ref_av = 0`)
""")


def page_advanced():
    st.title("ADVANCED — Site (Standalone, Two Global Events)")
    _about()

    # ----- Inputs (per 10-min) -----
    st.markdown("### Inputs — Site (per 10-min)")
    if "adv_inputs_site" not in st.session_state:
        st.session_state["adv_inputs_site"] = _default_inputs()
    inputs_df = st.data_editor(
        st.session_state["adv_inputs_site"],
        key="adv_inputs_site_editor",
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Time": st.column_config.TextColumn(
                "Time (hh:mm)",
                help="Start timestamp of the 10-min interval (hh:mm)."
            ),
            "Potential_kWh": st.column_config.NumberColumn(
                "Potential production (kWh)",
                help="Estimated site potential production during the interval, derived from irradiance & temperature."
            ),
            "AC_kWh_Site": st.column_config.NumberColumn(
                "AC energy (kWh)",
                help="Measured AC energy exported to the grid during the interval."
            ),
        },
    )
    st.session_state["adv_inputs_site"] = inputs_df.copy()

    # ----- Events (two global events) -----
    st.markdown("### Events — Global IEC-coded events (editable)")
    if "adv_events_global2" not in st.session_state:
        st.session_state["adv_events_global2"] = _default_global_two_events()
    events2_df = st.data_editor(
        st.session_state["adv_events_global2"],
        key="adv_events_global2_editor",
        use_container_width=True,
        num_rows=2,
        column_config={
            "EventID": st.column_config.TextColumn(
                "Event ID",
                help="Unique event identifier (e.g., E1, E2)."
            ),
            "Inv": st.column_config.SelectboxColumn(
                "Inverter",
                options=INV_CHOICES,
                help="Inverter concerned by this event."
            ),
            "IEC": st.column_config.TextColumn(
                "IEC code",
                help="IEC event code (e.g., 00101=OK, 00110=OMC, 0002x=Curtailment)."
            ),
            "Event": st.column_config.TextColumn(
                "Event name",
                help="Descriptive label for the event (OMC, Curtailment, Test OK, etc.)."
            ),
            "Start": st.column_config.TextColumn(
                "Start (hh:mm)",
                help="Start time of the event (hh:mm)."
            ),
            "End": st.column_config.TextColumn(
                "End (hh:mm)",
                help="End time of the event (hh:mm)."
            ),
        },
    )
    events2_df = _sanitize_two_events(events2_df)
    st.session_state["adv_events_global2"] = events2_df.copy()

    # Ensure Inputs cover all event 10-min slots
    inputs_df = _ensure_inputs_cover_events(inputs_df, events2_df)
    st.session_state["adv_inputs_site"] = inputs_df.copy()

    # ----- Computed Weights & Availability -----
    st.markdown("### Event Weights & Availability (computed from two global events)")
    slot_times = inputs_df["Time"].astype(str).tolist()
    comp = _compute_weights_availability_from_global_two_slots(slot_times, events2_df)
    comp_view = comp[[
        "Time",
        "Inv_1","IEC_1","Overlap_1_min","wf1 (distribution)","ref_av_1 (computed)",
        "Inv_2","IEC_2","Overlap_2_min","wf2 (distribution)","ref_av_2 (computed)",
        "Event Weight (Global)","Availability (Global)"
    ]].copy()
    st.data_editor(
        comp_view,
        key="adv_comp_view",
        use_container_width=True,
        disabled=True,
        column_config={
            "Time": st.column_config.TextColumn("Time (hh:mm)", help="10-min slot timestamp."),
            "Inv_1": st.column_config.TextColumn("Inverter 1", help="Inverter associated with Event 1."),
            "IEC_1": st.column_config.TextColumn("IEC₁", help="IEC code for Event 1."),
            "Overlap_1_min": st.column_config.NumberColumn("Overlap₁ [min]", help="Duration of overlap between Event 1 and this 10-min slot."),
            "wf1 (distribution)": st.column_config.NumberColumn("wf₁ (distribution)", help="Weight of Event 1: ov₁ / (ov₁ + ov₂)."),
            "ref_av_1 (computed)": st.column_config.NumberColumn("ref_av₁", help="Reference availability derived from IEC₁ (0=downtime, 1=available)."),
            "Inv_2": st.column_config.TextColumn("Inverter 2", help="Inverter associated with Event 2."),
            "IEC_2": st.column_config.TextColumn("IEC₂", help="IEC code for Event 2."),
            "Overlap_2_min": st.column_config.NumberColumn("Overlap₂ [min]", help="Duration of overlap between Event 2 and this 10-min slot."),
            "wf2 (distribution)": st.column_config.NumberColumn("wf₂ (distribution)", help="Weight of Event 2: ov₂ / (ov₁ + ov₂)."),
            "ref_av_2 (computed)": st.column_config.NumberColumn("ref_av₂", help="Reference availability derived from IEC₂ (0=downtime, 1=available)."),
            "Event Weight (Global)": st.column_config.NumberColumn("Event Weight (Global)", help="Total event weighting applied to this 10-min slot (≤1)."),
            "Availability (Global)": st.column_config.NumberColumn("Availability (Global)", help="Global availability: 0 if any overlapping event has ref_av=0."),
        }
    )

    # ----- Details — Weights & Availability (per slot) -----
    with st.expander("Details — Weights & Availability (per slot)", expanded=False):
        if not comp.empty:
            labels = [f"{i:02d} — {t}" for i, t in enumerate(comp["Time"].tolist())]
            idx = st.selectbox(
                "Select interval",
                options=list(range(len(labels))),
                format_func=lambda i: labels[i],
                key="adv_weights_slot_select"  # unique key
            )
            row = comp.iloc[idx]
            t = row["Time"]
            ts_min = _time_to_minutes(t)
            te_min = ts_min + SLOT_MINUTES
            slot_s = _minutes_to_time(ts_min)
            slot_e = _minutes_to_time(te_min)

            # Raw two global events (for transparency)
            e1 = events2_df.iloc[0] if len(events2_df) >= 1 else pd.Series()
            e2 = events2_df.iloc[1] if len(events2_df) >= 2 else pd.Series()

            inv1 = e1.get("Inv"); iec1 = e1.get("IEC")
            s1 = e1.get("Start"); e1e = e1.get("End")
            inv2 = e2.get("Inv"); iec2 = e2.get("IEC")
            s2 = e2.get("Start"); e2e = e2.get("End")

            ov1 = _overlap_minutes(ts_min, te_min, s1, e1e)
            ov2 = _overlap_minutes(ts_min, te_min, s2, e2e)
            denom = ov1 + ov2
            wf1 = (ov1/denom) if denom > 0 else 0.0
            wf2 = (ov2/denom) if denom > 0 else 0.0

            rav1 = _iec_ref_availability(iec1)
            rav2 = _iec_ref_availability(iec2)
            why1 = _ref_av_reason(iec1)
            why2 = _ref_av_reason(iec2)
            avail_global = 0 if ((ov1 > 0 and rav1 == 0) or (ov2 > 0 and rav2 == 0)) else 1
            afrac = (wf1 if rav1 == 1 else 0.0) + (wf2 if rav2 == 1 else 0.0)

            st.markdown(f"**Slot:** `{slot_s} → {slot_e}` (10 minutes)")
            st.latex(r"ov_i = \max\left(0, \min(t_e, e_i) - \max(t_s, s_i)\right)")
            st.markdown(
                "- Where: "
                f"`t_s = {slot_s}`, `t_e = {slot_e}`, "
                f"`Event1=[{s1 or '-'} → {e1e or '-'}]`, `Event2=[{s2 or '-'} → {e2e or '-'}]`."
            )

            det = pd.DataFrame([
                {"Event": "Event 1", "Inv": inv1, "IEC": iec1, "Start": s1, "End": e1e,
                 "Overlap[min]": ov1, "wf": round(wf1, 3), "ref_av": rav1, "Reason": why1},
                {"Event": "Event 2", "Inv": inv2, "IEC": iec2, "Start": s2, "End": e2e,
                 "Overlap[min]": ov2, "wf": round(wf2, 3), "ref_av": rav2, "Reason": why2},
            ])
            st.dataframe(det, use_container_width=True)

            st.latex(r"wf_1=\frac{ov_1}{ov_1+ov_2},\quad wf_2=\frac{ov_2}{ov_1+ov_2}")
            if denom > 0:
                st.markdown(f"- Numerically: wf1 = **{ov1} / {denom} = {wf1:.3f}**, wf2 = **{ov2} / {denom} = {wf2:.3f}**")
            else:
                st.markdown("- No overlaps in this slot ⇒ `wf1 = wf2 = 0`.")

            st.markdown(f"**Availability (Global)** = **{avail_global}** (= 0 if any overlapping event has ref_av=0; = 1 otherwise).")
            st.markdown(f"**Optional UI-only** `availability_fraction_10min` = **{afrac:.3f}** (sum of `wf` over events with `ref_av=1`).")

            if denom == 0:
                st.info("No event overlaps this slot. Weights are zero and no unavailability is attributed here.")
            elif (rav1 == 1 and rav2 == 1):
                st.info("Both events considered available (ref_av=1). Availability(Global)=1 and no downtime attribution.")

    # ----- Losses — DC–AC-only -----
    st.markdown("### Losses (computed)")
    losses = _compute_losses(inputs_df, comp)
    st.data_editor(
        losses,
        key="adv_losses_view",
        use_container_width=True,
        disabled=True,
        column_config={
            "Time": st.column_config.TextColumn("Time (hh:mm)", help="10-min interval timestamp."),
            "Gap_kWh": st.column_config.NumberColumn("Gap (kWh)", help="max(0, Potential − AC)."),
            "Inv_1": st.column_config.TextColumn("Inv (Event 1)", help="Inverter linked to Event 1."),
            "Lost_kWh_event1": st.column_config.NumberColumn("Lost (Event 1) [kWh]", help="(1−ref_av₁)×wf₁×Gap."),
            "Inv_2": st.column_config.TextColumn("Inv (Event 2)", help="Inverter linked to Event 2."),
            "Lost_kWh_event2": st.column_config.NumberColumn("Lost (Event 2) [kWh]", help="(1−ref_av₂)×wf₂×Gap."),
            "Lost_kWh_slot": st.column_config.NumberColumn("Lost (slot) [kWh]", help="Lost(Event1)+Lost(Event2) in the slot."),
        }
    )

    # --- Details — Losses (numeric application) ---
    def _fmt(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "—"

    with st.expander("Details — Losses (numeric application)", expanded=False):
        if not losses.empty and not comp.empty:
            times = losses["Time"].astype(str).tolist()
            idx2 = st.selectbox(
                "Select interval (losses)",
                options=list(range(len(times))),
                format_func=lambda i: f"{i:02d} — {times[i]}",
                key="adv_losses_slot_select"  # unique key
            )
            r_loss = losses.iloc[idx2]
            t_sel = str(r_loss["Time"])

            r_comp = comp.loc[comp["Time"].astype(str) == t_sel]
            if r_comp.empty:
                st.warning("No weights/availability found for this slot.")
            else:
                r_comp = r_comp.iloc[0]

                gap = float(pd.to_numeric(r_loss.get("Gap_kWh"), errors="coerce") or 0.0)
                wf1 = float(pd.to_numeric(r_comp.get("wf1 (distribution)"), errors="coerce") or 0.0)
                wf2 = float(pd.to_numeric(r_comp.get("wf2 (distribution)"), errors="coerce") or 0.0)

                def _to01(x):
                    try:
                        return 1 if float(x) >= 0.5 else 0
                    except Exception:
                        return 1

                rav1 = _to01(r_comp.get("ref_av_1 (computed)"))
                rav2 = _to01(r_comp.get("ref_av_2 (computed)"))

                lost1_tbl = float(pd.to_numeric(r_loss.get("Lost_kWh_event1"), errors="coerce") or 0.0)
                lost2_tbl = float(pd.to_numeric(r_loss.get("Lost_kWh_event2"), errors="coerce") or 0.0)

                lost1_calc = (1 - rav1) * wf1 * gap
                lost2_calc = (1 - rav2) * wf2 * gap

                st.markdown(f"**Time:** {t_sel} — Gap = **{_fmt(gap)} kWh**")
                st.latex(r"\mathrm{Lost}_i = (1-\mathrm{ref\_av}_i)\times wf_i \times \mathrm{Gap}")

                st.markdown(
                    f"- **Event 1**: (1 − {rav1}) × {_fmt(wf1)} × {_fmt(gap)} "
                    f"= **{_fmt(lost1_calc)} kWh**  (table = **{_fmt(lost1_tbl)} kWh**)"
                )
                st.markdown(
                    f"- **Event 2**: (1 − {rav2}) × {_fmt(wf2)} × {_fmt(gap)} "
                    f"= **{_fmt(lost2_calc)} kWh**  (table = **{_fmt(lost2_tbl)} kWh**)"
                )

                eps = 1e-6
                if (abs(lost1_calc - lost1_tbl) > eps) or (abs(lost2_calc - lost2_tbl) > eps):
                    st.error("Mismatch detected: computed vs table. Check IEC→ref_av mapping and weights/inputs for this slot.")

    # ===== Global Summary — All Losts & All PBA =====
    st.markdown("### Global Summary — All Losts & All PBA")
    potential_sum = float(pd.to_numeric(inputs_df["Potential_kWh"], errors="coerce").fillna(0.0).sum()) if not inputs_df.empty else 0.0
    losts_sum     = float(pd.to_numeric(losses["Lost_kWh_slot"], errors="coerce").fillna(0.0).sum()) if not losses.empty else 0.0
    all_pba = (potential_sum - losts_sum) / potential_sum if potential_sum > 0 else 0.0

    summary_df = pd.DataFrame([{
        "Sum Potential (kWh)": round(potential_sum, 6),
        "All Losts (kWh)": round(losts_sum, 6),
        "All PBA (%)": round(all_pba * 100.0, 4),
    }])

    st.data_editor(
        summary_df,
        key="adv_global_summary_all_pba",
        use_container_width=True,
        disabled=True,
        column_config={
            "Sum Potential (kWh)": st.column_config.NumberColumn(
                "Sum Potential (kWh)",
                help="Sum of all potential production over the selected period."
            ),
            "All Losts (kWh)": st.column_config.NumberColumn(
                "All Losts (kWh)",
                help="Sum of all computed energy losses."
            ),
            "All PBA (%)": st.column_config.NumberColumn(
                "All PBA (%)",
                help="Global Performance-Based Availability ratio: (Potential−Losts)/Potential ×100."
            ),
        }
    )

    # ---- Details — All PBA (global calculation) ----
    with st.expander("Details — All PBA (global calculation)", expanded=False):
        # local fmt (independent from previous)
        def _fmt2(x, nd=3):
            try:
                return f"{float(x):.{nd}f}"
            except Exception:
                return "—"

        st.markdown("**Definitions**")
        st.latex(r"\text{All Losts}=\sum \text{Lost\_kWh\_slot}")
        st.latex(r"\text{Sum Potential}=\sum \text{Potential\_kWh}")
        st.latex(r"\text{All PBA}=\frac{\text{Sum Potential}-\text{All Losts}}{\text{Sum Potential}}")

        st.markdown("**Numeric application**")
        st.markdown(
            f"- All Losts = **{_fmt2(losts_sum)} kWh**  \n"
            f"- Sum Potential = **{_fmt2(potential_sum)} kWh**  \n"
            f"- All PBA = ({_fmt2(potential_sum)} − {_fmt2(losts_sum)}) / {_fmt2(potential_sum)} "
            f"= **{_fmt2(all_pba)}** → **{_fmt2(all_pba*100, 2)} %**"
        )

# Optional: run the page by itself
if __name__ == "__main__":
    page_advanced()
