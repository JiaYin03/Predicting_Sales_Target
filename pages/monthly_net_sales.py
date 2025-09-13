# pages/monthly_net_sales.py
import os
import pandas as pd
import streamlit as st
import altair as alt
import joblib

st.set_page_config(page_title="Monthly Net Sales by Salesperson", page_icon="📅", layout="wide")
st.title("📅 Net Sales by Month (Pick a Salesperson)")

# ---------- Load data ----------
DATA_CANDIDATES = [
    "data/EDA_final_24-08.csv",
    "data/EDA_final_24-08.csv".replace("_", ""),  # loose fallback
]

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

data_path = first_existing(DATA_CANDIDATES)
if not data_path:
    st.error("Couldn't find your CSV in ./data. Expected `EDA_final_24-08.csv`.")
    st.stop()

df = pd.read_csv(data_path)

# Column name fallbacks
if "NetSalesValue" not in df.columns:
    for alt_name in ["Net Sales Value", "Net_Sales_Value", "NetSales"]:
        if alt_name in df.columns:
            df = df.rename(columns={alt_name: "NetSalesValue"})
            break
if "Month" not in df.columns:
    st.error("CSV must have a 'Month' column (1–12 or month names).")
    st.stop()
if "Salesperson" not in df.columns:
    st.error("CSV must have a 'Salesperson' column.")
    st.stop()

# ---------- Clean month to 1..12 ----------
month_map = {
    "jan":1,"january":1, "feb":2,"february":2, "mar":3,"march":3, "apr":4,"april":4,
    "may":5, "jun":6,"june":6, "jul":7,"july":7, "aug":8,"august":8,
    "sep":9,"sept":9,"september":9, "oct":10,"october":10, "nov":11,"november":11, "dec":12,"december":12
}
def to_month_num(x):
    try:
        n = int(x)
        return n if 1 <= n <= 12 else None
    except Exception:
        if pd.isna(x): return None
        return month_map.get(str(x).strip().lower(), None)

df["MonthNum"] = df["Month"].apply(to_month_num)
if df["MonthNum"].isna().any():
    bad = df.loc[df["MonthNum"].isna(), "Month"].unique()[:5]
    st.warning(f"Some Month values could not be parsed: {bad}")

# Keep only needed columns (+ qty/invoices if present)
cols_needed = ["Salesperson", "MonthNum", "NetSalesValue"] + (["Year"] if "Year" in df.columns else [])
if "QtyInvoiced" in df.columns: cols_needed.append("QtyInvoiced")
if "NumInvoices" in df.columns: cols_needed.append("NumInvoices")

df = df[cols_needed].dropna(subset=["Salesperson", "MonthNum", "NetSalesValue"])
df["MonthNum"] = df["MonthNum"].astype(int)

# Month labels/order
month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
month_order  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------- Load trained artifacts ONCE (model + feature schema) ----------
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
    model, FEATURE_COLS = None, None
else:
    model = joblib.load(model_path)
    FEATURE_COLS = joblib.load(feats_path)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
years = sorted(df["Year"].unique()) if "Year" in df.columns else [2023]
year  = st.sidebar.selectbox("Year", years, index=len(years)-1)
reps  = sorted(df["Salesperson"].dropna().unique())
rep   = st.sidebar.selectbox("Salesperson", reps)

# ---------- Aggregate & prepare chart/table (NET SALES) ----------
mask = (df["Salesperson"] == rep) & ((df["Year"] == year) if "Year" in df.columns else True)

use_cols = ["MonthNum", "NetSalesValue"]
has_qty  = "QtyInvoiced" in df.columns
has_inv  = "NumInvoices" in df.columns
if has_qty: use_cols.append("QtyInvoiced")
if has_inv: use_cols.append("NumInvoices")

sub = df.loc[mask, use_cols].copy()

monthly = sub.groupby("MonthNum").sum()
monthly = monthly.reindex(range(1,13)).fillna(0)

monthly_df = pd.DataFrame({
    "Month": [month_labels[m] for m in monthly.index],
    "NetSalesValue": monthly["NetSalesValue"].values
})
monthly_df["Month"] = pd.Categorical(monthly_df["Month"], categories=month_order, ordered=True)
monthly_df = monthly_df.sort_values("Month")

# ---------- KPIs ----------
colA, colB, colC = st.columns(3)
colA.metric("Total Net Sales", f"{monthly_df['NetSalesValue'].sum():,.0f}")
colB.metric("Average / Month", f"{monthly_df['NetSalesValue'].mean():,.0f}")
best_idx = monthly_df["NetSalesValue"].idxmax()
colC.metric("Best Month", f"{monthly_df.loc[best_idx, 'Month']}  ({monthly_df.loc[best_idx, 'NetSalesValue']:,.0f})")

# ---------- Chart ----------
st.subheader(f"Monthly Net Sales — {rep} ({year})")

chart = (
    alt.Chart(monthly_df)
      .mark_bar()
      .encode(
          x=alt.X(
              "Month:N",
              sort=month_order,
              title="Month",
              axis=alt.Axis(
                  labelAngle=0,     # keep labels horizontal
                  labelLimit=220,
                  labelFontSize=18, # <-- increase x-axis label size
                  titleFontSize=17, # <-- increase x-axis title size
                  labelPadding=6
              )
          ),
          y=alt.Y(
              "NetSalesValue:Q",
              title="Net Sales",
              axis=alt.Axis(
                  labelFontSize=17, # <-- increase y-axis label size
                  titleFontSize=18  # <-- increase y-axis title size
              )
          ),
          tooltip=["Month", alt.Tooltip("NetSalesValue:Q", format=",.0f")]
      )
)

st.altair_chart(chart, use_container_width=True)


# ---------- Table ----------
with st.expander("Show data table"):
    st.dataframe(monthly_df.style.format({"NetSalesValue": "{:,.0f}"}))

# ---------- Download ----------
st.download_button(
    "Download monthly CSV",
    data=monthly_df.to_csv(index=False).encode("utf-8"),
    file_name=f"monthly_net_sales_{rep}_{year}.csv",
    mime="text/csv"
)

# =======================================================================
#     PREDICTED TARGET (SUM OF MONTHLY PREDICTIONS + MARKET GUARDRAILS)
# =======================================================================

# market-style defaults
ABS_MIN_TARGET = 50_000  # absolute minimum annual target

def bounds_from_last_year(last_year_total: float) -> tuple[float, float]:
    ly = float(last_year_total)
    if ly >= 5_000_000:         # very large books
        floor_pct, cap_pct = 0.90, 1.15
    elif ly >= 1_000_000:       # large
        floor_pct, cap_pct = 0.85, 1.20
    elif ly >= 100_000:         # mid
        floor_pct, cap_pct = 0.80, 1.25
    else:                       # tiny / seed territories
        floor_pct, cap_pct = 0.00, 1.50  # min applies
    floor_abs = max(ABS_MIN_TARGET, ly * floor_pct)
    cap_abs   = max(floor_abs, ly * cap_pct)
    return floor_abs, cap_abs

def apply_bounds(val: float, floor_abs: float, cap_abs: float) -> tuple[float, str]:
    """Clamp val to [floor_abs, cap_abs] and return (bounded_val, reason)."""
    if val < floor_abs:
        return floor_abs, "floor"
    if val > cap_abs:
        return cap_abs, "cap"
    return val, "–"

def build_features_row(sp, year_val, month_val, net_sales, qty, ninv, feature_cols):
    row = pd.DataFrame([{
        "Salesperson": str(sp),
        "ProductClass": None,
        "CustomerClass": None,
        "Year": int(year_val),
        "Month": int(month_val),
        "NetSalesValue": float(net_sales),
        "QtyInvoiced": int(qty),
        "NumInvoices": int(ninv),
    }])
    cols_to_one_hot = ["Salesperson","ProductClass","CustomerClass"]
    if FEATURE_COLS is not None and any(isinstance(c, str) and c.startswith("Year_") for c in FEATURE_COLS):
        cols_to_one_hot.append("Year")
    X = pd.get_dummies(row, columns=cols_to_one_hot, drop_first=False)
    X = X.reindex(columns=FEATURE_COLS, fill_value=0)
    return X

def predict_annual_for_rep(rep_name, base_year_monthly_df, target_year):
    """Sum 12 monthly predictions; clip negatives to 0 before guardrails."""
    preds = []
    has_qty  = "QtyInvoiced" in base_year_monthly_df.columns
    has_inv  = "NumInvoices" in base_year_monthly_df.columns
    for m in range(1, 12+1):
        ns   = float(base_year_monthly_df.loc[m, "NetSalesValue"]) if m in base_year_monthly_df.index else 0.0
        qty  = int(base_year_monthly_df.loc[m, "QtyInvoiced"]) if (has_qty and m in base_year_monthly_df.index) else 1
        ninv = int(base_year_monthly_df.loc[m, "NumInvoices"]) if (has_inv and m in base_year_monthly_df.index) else 1
        Xrow = build_features_row(rep_name, target_year, m, ns, qty, ninv, FEATURE_COLS)
        raw  = float(model.predict(Xrow)[0])
        preds.append(max(0.0, raw))  # never negative
    return float(sum(preds))

st.subheader("Predicted Target Sales for Next Year (Annual KPI)")
if model is None or FEATURE_COLS is None:
    st.info("Model files not loaded; cannot compute prediction.")
else:
    next_year = int(year) + 1

    # ---- single rep KPI ----
    base_cols = ["MonthNum","NetSalesValue"]
    if "QtyInvoiced" in df.columns: base_cols.append("QtyInvoiced")
    if "NumInvoices" in df.columns: base_cols.append("NumInvoices")
    base_monthly = df.loc[mask, base_cols].groupby("MonthNum").sum().reindex(range(1,13)).fillna(0)

    annual_raw = predict_annual_for_rep(rep, base_monthly, next_year)
    last_year_total = float(base_monthly["NetSalesValue"].sum())
    floor_abs, cap_abs = bounds_from_last_year(last_year_total)
    annual_bounded, reason = apply_bounds(annual_raw, floor_abs, cap_abs)

    if reason != "–":
        st.caption(f"Guardrail applied ({reason}): floor={floor_abs:,.0f}, cap={cap_abs:,.0f} "
                   f"({annual_raw:,.0f} → {annual_bounded:,.0f})")

    st.metric(f"Predicted Target for {rep} in {next_year}", f"{annual_bounded:,.2f}")

    # ---- short list for all reps ----
    with st.expander("Show short list for all salespeople"):
        KNOWN_SPS = {c[len('Salesperson_'): ] for c in FEATURE_COLS if isinstance(c, str) and c.startswith("Salesperson_")}
        rows = []
        for r in reps:
            msk_r = (df["Salesperson"] == r) & ((df["Year"] == year) if "Year" in df.columns else True)
            mdf = df.loc[msk_r, base_cols].groupby("MonthNum").sum().reindex(range(1,13)).fillna(0)

            r_raw  = predict_annual_for_rep(r, mdf, next_year)
            r_ly   = float(mdf["NetSalesValue"].sum())
            r_floor, r_cap = bounds_from_last_year(r_ly)
            r_pred, r_reason = apply_bounds(r_raw, r_floor, r_cap)

            rows.append({
                "Salesperson": r,
                f"Total Net Sales {year}": r_ly,
                f"Predicted Target {next_year}": r_pred,
                "Bound": r_reason,
                "Seen in training": r in KNOWN_SPS
            })

        table = pd.DataFrame(rows).sort_values(f"Predicted Target {next_year}", ascending=False)
        st.dataframe(
            table.style.format({f"Total Net Sales {year}":"{:,.0f}", f"Predicted Target {next_year}":"{:,.2f}"}),
            use_container_width=True
        )
        st.download_button(
            "Download predictions (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"predicted_targets_{next_year}.csv",
            mime="text/csv"
        )

