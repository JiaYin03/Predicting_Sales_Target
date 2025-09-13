# app.py — Home / Overview
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import joblib

# ---------------------------------------------------------
# Page chrome
# ---------------------------------------------------------
st.set_page_config(page_title="Sales Target Overview", page_icon="📊", layout="wide")
st.title("📊 Sales Target Plan — Overview")

# ---------------------------------------------------------
# Load base data
# ---------------------------------------------------------
DATA_CANDIDATES = [
    "data/EDA_final_24-08.csv",
    "data/EDA_final_24-08.csv".replace("_",""),
]

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

data_path = first_existing(DATA_CANDIDATES)
if not data_path:
    st.error("Couldn't find `data/EDA_final_24-08.csv` in ./data.")
    st.stop()

df = pd.read_csv(data_path)

# soft column fixes
if "NetSalesValue" not in df.columns:
    for altname in ["Net Sales Value","Net_Sales_Value","NetSales"]:
        if altname in df.columns:
            df = df.rename(columns={altname:"NetSalesValue"})
            break
if "Month" not in df.columns:
    st.error("CSV must include 'Month'.")
    st.stop()

# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------
MODEL_CANDIDATES = ["models/best_model.pkl"]
FEATS_CANDIDATES = ["models/feature_columns.pkl"]

def first_ok(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

model_path = first_ok(MODEL_CANDIDATES)
feats_path = first_ok(FEATS_CANDIDATES)

if not model_path or not feats_path:
    st.info("Model files not found in ./models — add XGBoost.pkl/best_model.pkl and feature_columns to enable predictions.")
    st.stop()

model = joblib.load(model_path)
FEATURE_COLS = joblib.load(feats_path)

# ---------------------------------------------------------
# Sidebar controls (NO guardrails UI)
# ---------------------------------------------------------
years = sorted(df["Year"].unique()) if "Year" in df.columns else [2023]
year = st.sidebar.selectbox("Base year", years, index=len(years)-1)
next_year = int(year) + 1

st.sidebar.markdown("—")
ABS_MIN_TARGET = st.sidebar.number_input("Absolute min target", value=50_000, step=5_000)

st.sidebar.markdown("—")
growth_goal_pct = st.sidebar.number_input(
    "Company growth goal vs base year (%)",
    min_value=0.0, max_value=50.0, value=5.0, step=0.5
)

# ---------------------------------------------------------
# Fixed guardrails (hard-coded defaults; no sidebar)
# ---------------------------------------------------------
FLOOR_5M  = 1.00
CAP_5M    = 1.15
FLOOR_1_5 = 0.95
CAP_1_5   = 1.20
FLOOR_100 = 0.90
CAP_100   = 1.25

def bounds_from_last_year(last_year_total: float) -> tuple[float, float]:
    """Return (floor_abs, cap_abs) for a rep given last-year total using fixed defaults."""
    ly = float(last_year_total)
    if   ly >= 5_000_000:
        floor_pct, cap_pct = FLOOR_5M,  CAP_5M
    elif ly >= 1_000_000:
        floor_pct, cap_pct = FLOOR_1_5,  CAP_1_5
    elif ly >=   100_000:
        floor_pct, cap_pct = FLOOR_100,  CAP_100
    else:
        floor_pct, cap_pct = 0.00, 1.50

    floor_abs = max(ABS_MIN_TARGET, ly * floor_pct)
    cap_abs   = max(floor_abs, ly * cap_pct)
    return floor_abs, cap_abs

def apply_bounds(val: float, floor_abs: float, cap_abs: float) -> tuple[float, str]:
    if val < floor_abs:
        return floor_abs, "floor"
    if val > cap_abs:
        return cap_abs, "cap"
    return val, "ok"

# ---------------------------------------------------------
# Feature builder for per-month prediction
# ---------------------------------------------------------
def build_features_row(sp, year_val, month_val, net_sales, qty, ninv, feature_cols):
    row = pd.DataFrame([{
        "Salesperson": str(sp),
        "ProductClass": None,
        "CustomerClass": None,
        "Year": int(year_val),
        "Month": int(month_val),
        "NetSalesValue": float(net_sales),
        "QtyInvoiced": int(qty) if "QtyInvoiced" in df.columns else 1,
        "NumInvoices": int(ninv) if "NumInvoices" in df.columns else 1,
    }])

    one_hot = ["Salesperson","ProductClass","CustomerClass"]
    if any(isinstance(c,str) and c.startswith("Year_") for c in feature_cols):
        one_hot.append("Year")

    X = (pd.get_dummies(row, columns=one_hot, drop_first=False)
           .reindex(columns=feature_cols, fill_value=0))
    return X

def predict_annual_for_rep(rep_name, base_year_monthly_df):
    preds = []
    has_qty = "QtyInvoiced" in base_year_monthly_df.columns
    has_inv = "NumInvoices" in base_year_monthly_df.columns
    for m in range(1, 12+1):
        ns   = float(base_year_monthly_df.loc[m, "NetSalesValue"]) if m in base_year_monthly_df.index else 0.0
        qty  = int(base_year_monthly_df.loc[m, "QtyInvoiced"]) if (has_qty and m in base_year_monthly_df.index) else 1
        ninv = int(base_year_monthly_df.loc[m, "NumInvoices"]) if (has_inv and m in base_year_monthly_df.index) else 1
        Xrow = build_features_row(rep_name, next_year, m, ns, qty, ninv, FEATURE_COLS)
        raw  = float(model.predict(Xrow)[0])
        preds.append(max(0.0, raw))   # no negatives
    return float(sum(preds))

# ---------------------------------------------------------
# Build base-year aggregates for ALL reps
# ---------------------------------------------------------
mask_year = (df["Year"] == year) if "Year" in df.columns else np.full(len(df), True)
base_cols = ["MonthNum","NetSalesValue"]
if "MonthNum" not in df.columns:
    # make MonthNum from Month
    month_map = {"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
                 "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,
                 "september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12}
    try:
        df["MonthNum"] = df["Month"].astype(int)
    except Exception:
        df["MonthNum"] = df["Month"].astype(str).str.lower().map(month_map).astype(int)

if "QtyInvoiced" in df.columns: base_cols.append("QtyInvoiced")
if "NumInvoices" in df.columns: base_cols.append("NumInvoices")

rows = []
for rep in sorted(df.loc[mask_year, "Salesperson"].dropna().unique()):
    by_rep = df.loc[(df["Salesperson"]==rep) & mask_year, base_cols].copy()
    monthly = by_rep.groupby("MonthNum").sum().reindex(range(1,13)).fillna(0)

    last_year_total = float(monthly["NetSalesValue"].sum())
    raw = predict_annual_for_rep(rep, monthly)

    floor_abs, cap_abs = bounds_from_last_year(last_year_total)
    bounded, reason = apply_bounds(raw, floor_abs, cap_abs)

    rows.append({
        "Salesperson": rep,
        f"Total Net Sales {year}": last_year_total,
        f"Raw model sum {next_year}": raw,
        f"Predicted Target {next_year}": bounded,
        "Bound": reason,
        "Floor used": floor_abs,
        "Cap used": cap_abs,
        "Seen in training": any(
            str(rep) == str(c).replace("Salesperson_","")
            for c in FEATURE_COLS if isinstance(c,str) and c.startswith("Salesperson_")
        )
    })

plan = pd.DataFrame(rows)

# ---------------------------------------------------------
# Optional company growth goal reallocation
# ---------------------------------------------------------
total_ly     = plan[f"Total Net Sales {year}"].sum()
target_col   = f"Predicted Target {next_year}"
base_plan    = plan[target_col].sum()
desired_total= total_ly * (1.0 + growth_goal_pct/100.0)

gap = desired_total - base_plan
if gap > 0:
    headroom = (plan["Cap used"] - plan[target_col]).clip(lower=0)
    total_headroom = headroom.sum()
    if total_headroom > 0:
        scale = min(1.0, gap / total_headroom)
        increment = headroom * scale
        plan[target_col] = plan[target_col] + increment

        # update Bound where we hit cap due to scaling
        plan["Bound"] = np.where(
            np.isclose(plan[target_col], plan["Cap used"], rtol=0, atol=1e-6),
            "cap",
            plan["Bound"]
        )
    else:
        st.warning("Not enough cap headroom to reach the growth goal. Raise caps or floors.")

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------
total_plan = plan[target_col].sum()
delta      = total_plan - total_ly
pct        = (delta/total_ly*100) if total_ly else 0

# Header “how to” + method
with st.expander("How to use this page", expanded=True):
    st.markdown(
        """
1. **Choose the base year** on the left (plan is base_year + 1).  
2. **Review KPIs**: company total, planned total (Δ vs LY), and reps at floor/cap.  
3. See the **Top N** chart and the **Full Plan** table below.  

**Method**: We predict each rep’s **12 monthly values** with the model, sum them, then apply **guardrails**  
(≥5M: floor/cap; 1–5M: floor/cap; 100k–1M: floor/cap; <100k: absolute min) and an optional **company growth goal** that distributes extra only to reps with cap headroom.
"""
    )

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Net Sales (base year)", f"{total_ly:,.0f}")
c2.metric(f"Planned Target {next_year}", f"{total_plan:,.0f}", f"{pct:+.1f}%")
c3.metric("Reps at floor", f"{(plan['Bound']=='floor').sum()} / {len(plan)}")
c4.metric("Reps at cap",   f"{(plan['Bound']=='cap').sum()} / {len(plan)}")

st.caption(
    f"Guardrails applied after model prediction: "
    f"≥5M {int(FLOOR_5M*100)}–{int(CAP_5M*100)}%, "
    f"1–5M {int(FLOOR_1_5*100)}–{int(CAP_1_5*100)}%, "
    f"100k–1M {int(FLOOR_100*100)}–{int(CAP_100*100)}%, "
    f"<100k min {ABS_MIN_TARGET:,.0f}.  "
    f"Company growth goal: {growth_goal_pct:.1f}%"
)

# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------
top_n = st.slider("Show top N by target", min_value=5, max_value=min(15, len(plan)), value=min(10, len(plan)), step=1)

top_df = plan.sort_values(target_col, ascending=False).head(top_n)
bar = alt.Chart(top_df).mark_bar().encode(
    x=alt.X("Salesperson:N", sort="-y", title="Salesperson"),
    y=alt.Y(f"{target_col}:Q", title="Target"),
    tooltip=[
        "Salesperson",
        alt.Tooltip(f"{target_col}:Q", title="Target", format=",.0f"),
        alt.Tooltip(f"Total Net Sales {year}:Q", title="Last year", format=",.0f"),
        "Bound"
    ]
)
st.altair_chart(bar, use_container_width=True)

# ---------------------------------------------------------
# Full table + download
# ---------------------------------------------------------
show_cols = [
    "Salesperson",
    f"Total Net Sales {year}",
    target_col,
    "Bound",
    "Seen in training"
]

st.subheader("Full Plan")
st.dataframe(
    plan[show_cols].sort_values(target_col, ascending=False)
        .style.format({f"Total Net Sales {year}":"{:,.0f}", target_col:"{:,.2f}"}),
    use_container_width=True
)

st.download_button(
    "Download plan (CSV)",
    data=plan.sort_values(target_col, ascending=False).to_csv(index=False).encode("utf-8"),
    file_name=f"target_plan_{next_year}.csv",
    mime="text/csv"
)

st.info("Tip: Use the sidebar to adjust the absolute minimum target or set a growth goal if you need the company plan to be ≥ last year.")
