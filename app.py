# pages/1_Target_Input_Dashboard.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from typing import Optional

st.set_page_config(page_title="Target Input Dashboard", page_icon="🧮", layout="wide")
st.title("🧮 Target Input Dashboard")

# --------------------------- Load data ---------------------------
DATA_CANDIDATES = ["data/EDA_final_24-08.csv", "data/EDA_final_24-08.csv".replace("_","")]
MODEL_CANDIDATES = ["models/best_model.pkl"]
FEATS_CANDIDATES = ["models/feature_columns.pkl"]

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

data_path  = first_existing(DATA_CANDIDATES)
model_path = first_existing(MODEL_CANDIDATES)
feats_path = first_existing(FEATS_CANDIDATES)

if not data_path:
    st.error("Couldn't find base data: `data/EDA_final_24-08.csv`.")
    st.stop()

df = pd.read_csv(data_path)

# column soft-fixes
if "NetSalesValue" not in df.columns:
    for altname in ["Net Sales Value", "Net_Sales_Value", "NetSales"]:
        if altname in df.columns:
            df = df.rename(columns={altname: "NetSalesValue"})
            break
if "Year" not in df.columns:
    st.error("CSV must include 'Year'.")
    st.stop()

if "MonthNum" not in df.columns:
    if "Month" in df.columns:
        month_map = {
            "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
            "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,
            "september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
        }
        try:
            df["MonthNum"] = df["Month"].astype(int)
        except Exception:
            df["MonthNum"] = df["Month"].astype(str).str.lower().map(month_map).astype(int)
    else:
        df["MonthNum"] = 1

# --------------------------- Defaults ---------------------------
years = sorted(df["Year"].dropna().astype(int).unique())
base_year = int(years[-1])           # latest year in your data
next_year = base_year + 1
use_model_prefill = True

# Guardrails
ABS_MIN_TARGET = 50_000              # hard absolute floor (existing reps)
ABS_MIN_FOR_NEW_REPS = False         # set True if you also want the hard floor for brand-new names

floor_5m,  cap_5m  = 1.00, 1.15
floor_1_5, cap_1_5 = 0.95, 1.20
floor_100, cap_100 = 0.90, 1.25

# --------------------------- Helpers ---------------------------
# Treat CSV "Salesperson" as the *code*
ly = (df.loc[df["Year"] == base_year]
        .groupby("Salesperson", dropna=True)["NetSalesValue"]
        .sum()
        .rename(f"Total Net Sales {base_year}")
        .astype(float))

def get_picker_options():
    """Codes in CSV ∪ codes already added to the plan."""
    base_codes = set(ly.index.astype(str))
    plan_codes = set(st.session_state.get("plan", {}).keys())
    return sorted(base_codes | plan_codes)

def bounds_from_last_year(last_year_total: float, use_abs_min: bool = True):
    v = float(last_year_total)
    if   v >= 5_000_000: floor_pct, cap_pct = floor_5m,  cap_5m
    elif v >= 1_000_000: floor_pct, cap_pct = floor_1_5,  cap_1_5
    elif v >=   100_000: floor_pct, cap_pct = floor_100, cap_100
    else:                floor_pct, cap_pct = 0.00,     1.50
    floor_abs = v * floor_pct
    if use_abs_min:
        floor_abs = max(ABS_MIN_TARGET, floor_abs)
    cap_abs = max(floor_abs, v * cap_pct)
    return floor_abs, cap_abs

FEATURE_COLS = joblib.load(feats_path) if feats_path and os.path.exists(feats_path) else []
model = joblib.load(model_path) if model_path and os.path.exists(model_path) else None

def _build_features_row(sp_code, year_val, month_val, net_sales, qty, ninv, feature_cols):
    row = pd.DataFrame([{
        "Salesperson": str(sp_code), "ProductClass": None, "CustomerClass": None,
        "Year": int(year_val), "Month": int(month_val),
        "NetSalesValue": float(net_sales), "QtyInvoiced": int(qty), "NumInvoices": int(ninv),
    }])
    one_hot = ["Salesperson","ProductClass","CustomerClass"]
    if any(isinstance(c,str) and c.startswith("Year_") for c in feature_cols):
        one_hot.append("Year")
    return (pd.get_dummies(row, columns=one_hot, drop_first=False)
              .reindex(columns=feature_cols, fill_value=0))

def _monthly_base(sp_code):
    """Monthly totals for an existing code in base_year; zeroes if missing."""
    cols = ["MonthNum","NetSalesValue"]
    if "QtyInvoiced" in df.columns: cols.append("QtyInvoiced")
    if "NumInvoices" in df.columns: cols.append("NumInvoices")

    if sp_code in (df["Salesperson"].astype(str).unique() if "Salesperson" in df.columns else []):
        g = (df.loc[(df["Salesperson"]==sp_code) & (df["Year"]==base_year), cols]
               .groupby("MonthNum").sum().reindex(range(1,13)).fillna(0))
    else:
        g = pd.DataFrame({"NetSalesValue":[0.0]*12}, index=range(1,13))
    if "QtyInvoiced" not in g.columns: g["QtyInvoiced"] = 1
    if "NumInvoices" not in g.columns: g["NumInvoices"] = 1
    return g

def model_suggested(sp_code: str, ly_hint: Optional[float] = None):
    """Predict next-year sum for `sp_code`, then apply guardrails."""
    if model is None or not len(FEATURE_COLS):
        return np.nan, "n/a"

    is_existing = sp_code in ly.index
    if ly_hint is not None and not np.isnan(ly_hint):
        ly_total = float(ly_hint)
    elif is_existing:
        ly_total = float(ly.loc[sp_code])
    else:
        ly_total = 0.0

    base = _monthly_base(sp_code)
    preds = []
    for m in range(1, 13):
        ns = float(base.loc[m, "NetSalesValue"])
        q  = int(base.loc[m, "QtyInvoiced"])
        ni = int(base.loc[m, "NumInvoices"])
        raw = float(model.predict(_build_features_row(sp_code, next_year, m, ns, q, ni, FEATURE_COLS))[0])
        preds.append(max(0.0, raw))
    raw_sum = float(np.sum(preds))

    use_abs_min = ABS_MIN_FOR_NEW_REPS if not is_existing else True
    floor_abs, cap_abs = bounds_from_last_year(ly_total, use_abs_min=use_abs_min)
    bounded = min(max(raw_sum, floor_abs), cap_abs)
    reason = "floor" if np.isclose(bounded, floor_abs) else ("cap" if np.isclose(bounded, cap_abs) else "ok")
    return bounded, reason

def company_achievability(plan_dict):
    """
    Achievability % = min(1, sum(reachable_i) / sum(target_i)) * 100
    Uses LY from CSV for existing codes, else LY from user input for new codes.
    """
    if not plan_dict:
        return 100.0, 0.0, 0.0
    entered_total = float(sum(v.get("target", 0.0) for v in plan_dict.values()))
    if entered_total <= 0:
        return 100.0, 0.0, 0.0
    if model is None or not len(FEATURE_COLS):
        return 0.0, 0.0, entered_total

    reachable_total = 0.0
    for code, rec in plan_dict.items():
        ly_hint = rec.get("ly", None)
        suggested, _ = model_suggested(code, ly_hint=ly_hint)
        if not np.isnan(suggested):
            reachable_total += float(suggested)
    pct = min(1.0, reachable_total / entered_total) * 100.0
    return pct, reachable_total, entered_total

def render_achievability(ach_pct: float, reachable_amt: float, entered_total: float):
    """Show '% of target reachable' only."""
    try:
        pct_reachable = float(ach_pct)
    except Exception:
        pct_reachable = 0.0
    pct_reachable = max(0.0, min(100.0, pct_reachable))
    st.metric("Achievability %", f"{pct_reachable:.1f}%", f"{reachable_amt:,.0f} reachable")
    st.progress(int(round(pct_reachable)))

# --------------------------- Session state ---------------------------
# plan is keyed by CODE; value: {"name": str, "ly": float, "target": float}
if "plan" not in st.session_state:
    st.session_state.plan = {}
# code -> latest name string (reference)
if "name_map" not in st.session_state:
    st.session_state.name_map = {}
if "last_added_code" not in st.session_state:
    st.session_state.last_added_code = None
if "last_paste_codes" not in st.session_state:
    st.session_state.last_paste_codes = []
# delete selection clear flag + value store (for safe clearing)
if "clear_del_sel" not in st.session_state:
    st.session_state.clear_del_sel = False
if "del_sel_multiselect" not in st.session_state:
    st.session_state.del_sel_multiselect = []

# Helper for label display in pickers
def display_code(code: str) -> str:
    nm = st.session_state.name_map.get(code, "")
    return f"{code} — {nm}" if nm else str(code)

# --------------------------- Two-pane dashboard ---------------------------
left, right = st.columns([1, 1.2])

with left:
    st.subheader("Enter / Update a Salesperson Target")

    tab_existing, tab_new = st.tabs(["Pick from data", "Add new salesperson"])

    # ----- TAB 1: Pick from data -----
    with tab_existing:
        picker_options = get_picker_options()

        # Remember last-added code (if any) for the very first render
        default_code = st.session_state.get("last_added_code")

        # Seed a default BEFORE creating the widget (only once)
        if "rep_select" not in st.session_state:
            if picker_options:
                st.session_state.rep_select = (
                    default_code if default_code in picker_options else picker_options[0]
                )
            else:
                st.session_state.rep_select = ""  # no options yet

        # If options changed and current selection disappeared, reset gracefully
        if picker_options and st.session_state.rep_select not in picker_options:
            st.session_state.rep_select = picker_options[0]
        if not picker_options:
            st.session_state.rep_select = ""

        # Picker (or fallback input when list is empty)
        if picker_options:
            sp_code = st.selectbox(
                "Salesperson (code)",
                picker_options,
                key="rep_select",
                format_func=display_code,
            )
        else:
            sp_code = st.text_input(
                "Salesperson (code)",
                value=st.session_state.get("rep_select_fallback", ""),
                key="rep_select_fallback",
            ).strip()

        # Only continue if we have a value
        if sp_code:
            # is this an existing code in CSV?
            is_in_data = sp_code in ly.index
            current_plan = st.session_state.plan.get(sp_code, {})

            # Name (reference) — allow editing even for existing codes
            name_prefill = current_plan.get("name", st.session_state.name_map.get(sp_code, ""))
            edited_name = st.text_input("Salesperson name (reference)", value=name_prefill, key="edit_name")

            # LY: for existing codes we use CSV; for new codes allow edit here
            ly_prefill = float(ly.loc[sp_code]) if is_in_data else float(current_plan.get("ly", 0.0))
            if is_in_data:
                st.caption(f"Detected last-year total: **{ly_prefill:,.0f}**")
                ly_for_calc = ly_prefill
            else:
                ly_for_calc = st.number_input(
                    f"Total Net Sales {base_year} (edit for new rep)",
                    min_value=0, step=1000, value=int(ly_prefill), key="edit_ly_for_new"
                )

            # Suggestion uses CSV LY for existing codes; for new codes uses the entered LY
            sug_val, reason = (
                model_suggested(sp_code, ly_hint=None if is_in_data else ly_for_calc)
                if use_model_prefill and sp_code else (np.nan, "n/a")
            )

            # Default target: current plan value or suggestion
            default_target = current_plan.get("target", 0 if np.isnan(sug_val) else int(round(sug_val)))

            with st.form("entry_form_existing", clear_on_submit=False):
                colA, colB = st.columns(2)
                with colA:
                    st.number_input(
                        "Suggested target (info)",
                        value=0 if np.isnan(sug_val) else int(round(sug_val)),
                        step=1, disabled=True,
                    )
                with colB:
                    st.text_input("Bound reason", value=reason, disabled=True)

                target_existing = st.number_input(
                    f"Your Target {next_year}", min_value=0, step=1000,
                    value=int(default_target), key="target_existing"
                )

                c1, c2 = st.columns(2)
                add_upd = c1.form_submit_button("Add / Update")
                rm_one = c2.form_submit_button("Remove this salesperson")

                if add_upd:
                    st.session_state.plan[sp_code] = {
                        "name": edited_name.strip(),
                        "ly": float(ly_for_calc),
                        "target": float(target_existing),
                    }
                    if edited_name.strip():
                        st.session_state.name_map[sp_code] = edited_name.strip()
                    st.session_state.last_added_code = sp_code
                    st.success(f"Saved target for **{display_code(sp_code)}**.")

                if rm_one:
                    if sp_code in st.session_state.plan:
                        st.session_state.plan.pop(sp_code, None)
                        st.warning(f"Removed **{display_code(sp_code)}** from Target Sales Plan.")
                    else:
                        st.info("This salesperson isn't in the Target Sales Plan yet.")
        else:
            st.info("No salespeople yet. Add a new salesperson first.")

    # ----- TAB 2: Add NEW salesperson -----
    with tab_new:
        with st.form("entry_form_new", clear_on_submit=False):
            new_code  = st.text_input("Salesperson code (required)", key="new_code").strip()
            new_name  = st.text_input("Salesperson name (reference)", key="new_name").strip()
            new_ly    = st.number_input(f"Total Net Sales {base_year}", min_value=0, step=1000, key="new_ly")
            new_target= st.number_input(f"Your Target {next_year}",   min_value=0, step=1000, key="new_target")

            c1, c2 = st.columns(2)
            add_new = c1.form_submit_button("Add / Update (new)")
            rm_new  = c2.form_submit_button("Remove this salesperson")

            if add_new:
                if not new_code:
                    st.error("Please enter a salesperson code.")
                else:
                    st.session_state.plan[new_code] = {"name": new_name, "ly": float(new_ly), "target": float(new_target)}
                    if new_name:
                        st.session_state.name_map[new_code] = new_name
                    st.session_state.last_added_code = new_code
                    st.success(f"Saved target for **{display_code(new_code)}**.")
                    st.rerun()

            if rm_new:
                if new_code in st.session_state.plan:
                    st.session_state.plan.pop(new_code, None)
                    st.warning(f"Removed **{display_code(new_code)}** from Target Sales Plan.")
                else:
                    st.info("This salesperson isn't in the Target Sales Plan yet.")

# --- QUICK TOOLS (Code + Target only) ---
st.markdown("#### Quick Tools")

qt_left, qt_right = st.columns([1, 2])

# Clear all
with qt_left:
    if st.button("Clear ALL", key="qt_clear_all"):
        st.session_state.plan = {}
        st.session_state.name_map = {}
        st.session_state.last_added_code = None
        st.session_state.last_paste_codes = []
        st.warning("Cleared all entries.")

with qt_right:
    with st.expander("Paste by CODE (format: CODE, Target)", expanded=False):
        st.caption(
            "Enter one line per salesperson. **Codes only** (no names). "
            "Examples:\n"
            "`SM-JW, 1,200,000`\n"
            "`ECOM-HK, 50,000`\n"
            "Unknown codes are ignored."
        )
        txt = st.text_area(
            "One line per salesperson",
            height=80,
            placeholder="SM-JW, 1,200,000\nECOM-HK, 50,000",
            key="qt_paste_text",
        )

        # Build lookup of canonical codes we allow (CSV ∪ current plan)
        code_lookup = {c.lower(): c for c in get_picker_options()}

        c1, c2 = st.columns(2)

        # Add pasted
        with c1:
            if st.button("Add pasted", key="qt_btn_add_pasted"):
                lines = [r.strip() for r in txt.splitlines() if r.strip()]
                added_codes, skipped = [], []
                last_ok_code = None

                for line in lines:
                    try:
                        # Must be "CODE, TARGET"
                        parts = line.split(",", 1)
                        if len(parts) != 2:
                            skipped.append(line)
                            continue

                        raw_code, tgt_str = parts[0].strip(), parts[1].strip()
                        low = raw_code.lower()

                        # Only accept known codes
                        if low not in code_lookup:
                            skipped.append(raw_code)
                            continue

                        code = code_lookup[low]
                        # Parse number with optional commas
                        tgt_val = float(tgt_str.replace(",", ""))

                        # Keep existing name if we have it; else blank
                        name_val = st.session_state.name_map.get(
                            code,
                            st.session_state.plan.get(code, {}).get("name", "")
                        )
                        # LY from CSV if known; else keep whatever we already have (or 0)
                        ly_val = float(ly.loc[code]) if code in ly.index \
                                 else float(st.session_state.plan.get(code, {}).get("ly", 0.0))

                        st.session_state.plan[code] = {"name": name_val, "ly": ly_val, "target": tgt_val}
                        if name_val:
                            st.session_state.name_map[code] = name_val
                        added_codes.append(code)
                        last_ok_code = code

                    except Exception:
                        skipped.append(line)

                st.session_state["last_paste_codes"] = added_codes
                if last_ok_code:
                    st.session_state["last_added_code"] = last_ok_code

                if added_codes:
                    st.success(f"Added/updated {len(added_codes)} entr{'y' if len(added_codes)==1 else 'ies'}.")
                if skipped:
                    st.warning(
                        "Skipped unknown/invalid lines: "
                        + ", ".join(skipped[:5])
                        + (" ..." if len(skipped) > 5 else "")
                    )

        # Undo last paste
        with c2:
            can_undo = len(st.session_state.get("last_paste_codes", [])) > 0
            if st.button("Undo last paste", disabled=not can_undo, key="qt_btn_undo_paste"):
                removed = 0
                for code in st.session_state.get("last_paste_codes", []):
                    if code in st.session_state.plan:
                        st.session_state.plan.pop(code, None)
                        removed += 1
                st.session_state["last_paste_codes"] = []
                st.warning(f"Removed {removed} entr{'y' if removed == 1 else 'ies'} from last paste.")

    st.markdown("---")
    st.subheader("Delete from Target Sales Plan")

    if st.session_state.plan:
        # IMPORTANT: clear selection BEFORE rendering the multiselect (next run after delete)
        if st.session_state.get("clear_del_sel"):
            st.session_state.del_sel_multiselect = []
            st.session_state.clear_del_sel = False

        all_codes = sorted(st.session_state.plan.keys())
        del_sel = st.multiselect(
            "Select salesperson code(s) to delete",
            options=all_codes,
            format_func=lambda c: f"{c} — {st.session_state.name_map.get(c, '')}",
            key="del_sel_multiselect"
        )

        if st.button("Delete selected", disabled=len(del_sel) == 0, key="btn_delete_selected"):
            removed = 0
            for code in del_sel:
                if code in st.session_state.plan:
                    st.session_state.plan.pop(code, None)
                    removed += 1
            st.success(f"Deleted {removed} entr{'y' if removed==1 else 'ies'}.")
            # ask the NEXT run to clear the multiselect value, then rerun
            st.session_state.clear_del_sel = True
            st.rerun()
    else:
        st.info("No entries in the list yet.")

with right:
    st.subheader("Target Sales Plan (Live)")
    if st.session_state.plan:
        # Build DF that includes Code + Name
        records = []
        for code, rec in st.session_state.plan.items():
            records.append({
                "Code": code,
                "Name": rec.get("name", st.session_state.name_map.get(code, "")),
                f"Total Net Sales {base_year}": rec.get("ly", 0.0),
                f"Your Target {next_year}": rec.get("target", 0.0)
            })
        plan_df = pd.DataFrame(records)

        st.dataframe(
            plan_df.sort_values(f"Your Target {next_year}", ascending=False)
                   .style.format({f"Total Net Sales {base_year}":"{:,.0f}",
                                  f"Your Target {next_year}":"{:,.0f}"}),
            use_container_width=True, height=440
        )

        # ---- Summary (BOTTOM ONLY) ----
        t_ly = float(plan_df[f"Total Net Sales {base_year}"].sum())
        t_tgt = float(plan_df[f"Your Target {next_year}"].sum())
        headcount = len(plan_df)
        pct_vs_ly = ((t_tgt - t_ly)/t_ly*100) if t_ly else 0.0

        k1, k2, k3 = st.columns(3)
        k1.metric(f"Total Net Sales {base_year}", f"{t_ly:,.0f}")
        k2.metric(f"Target Sales {next_year}", f"{t_tgt:,.0f}", f"{pct_vs_ly:+.1f}% vs LY")
        k3.metric("Headcount (entered)", f"{headcount}")

        ach_pct, reachable_amt, entered_total = company_achievability(st.session_state.plan)
        render_achievability(ach_pct, reachable_amt, entered_total)

        st.download_button(
            "Download plan (CSV)",
            data=plan_df.sort_values(f"Your Target {next_year}", ascending=False).to_csv(index=False).encode("utf-8"),
            file_name=f"entered_targets_{next_year}.csv",
            mime="text/csv"
        )
    else:
        st.info("No entries yet. Use the forms on the left to add targets.")
