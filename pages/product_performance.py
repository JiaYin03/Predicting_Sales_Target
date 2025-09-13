# pages/products_customers.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

def bump_table_font(px: int = 18, header_weight: int = 600):
    st.markdown(
        f"""
        <style>
        /* Interactive tables: st.dataframe / st.data_editor (AG Grid) */
        div[data-testid="stDataFrame"] div[role="grid"] * {{
            font-size: {px}px !important;
            line-height: 1.35 !important;
        }}
        div[data-testid="stDataFrame"] div[role="columnheader"] * {{
            font-size: {px}px !important;
            font-weight: {header_weight} !important;
        }}

        /* Static st.table fallback */
        table[data-testid="stTable"] * {{
            font-size: {px}px !important;
            line-height: 1.35 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="Products & Customers", page_icon="📦", layout="wide")
st.title("📦 Product Performance & 👥 Customer Overview")
bump_table_font(16)  # change 18 to any size you prefer
# ---------- utils ----------
DATA_CANDIDATES = ["data/EDA_final_24-08.csv", "data/EDA_final_24-08.csv".replace("_", "")]

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

@st.cache_data(ttl=3600)
def load_csv(path):
    return pd.read_csv(path)

def to_month_num(x):
    if pd.isna(x):
        return None
    month_map = {
        "jan":1,"january":1, "feb":2,"february":2, "mar":3,"march":3, "apr":4,"april":4,
        "may":5, "jun":6,"june":6, "jul":7,"july":7, "aug":8,"august":8,
        "sep":9,"sept":9,"september":9, "oct":10,"october":10, "nov":11,"november":11, "dec":12,"december":12
    }
    s = str(x).strip().lower()
    try:
        n = int(s)
        if 1 <= n <= 12:
            return n
    except Exception:
        pass
    return month_map.get(s, None)

# Re-usable axis styles (horizontal x labels + larger fonts)
X_AXIS = alt.Axis(
    labelAngle=0,      # horizontal labels
    labelLimit=220,
    labelFontSize=18,  # x labels
    titleFontSize=17,  # x title
    labelPadding=6
)
Y_AXIS = alt.Axis(
    labelFontSize=17,  # y labels
    titleFontSize=18   # y title
)

# Helper to explain a class code like "CTRD-KA"
MAIN_MEANING = {"CTRD": "Trade customer", "CNTRD": "Non-trade customer"}
SUFFIX_MEANING = {
    "SD": "Sub distribution",
    "KA": "Key account",
    "HRC": "Independent",
    "ECOM": "E-commerce",
    "OS": "Other segment",
}
def class_meaning_for(code: str) -> str:
    if not isinstance(code, str):
        return ""
    parts = [p.strip().upper() for p in code.replace("_", "-").split("-") if p.strip()]
    phrases = []
    for p in parts:
        if p in MAIN_MEANING:
            phrases.append(MAIN_MEANING[p])
        elif p in SUFFIX_MEANING:
            phrases.append(SUFFIX_MEANING[p])
        else:
            phrases.append(p.title())
    # De-duplicate while preserving order
    seen, out = set(), []
    for ph in phrases:
        if ph not in seen:
            out.append(ph); seen.add(ph)
    return " | ".join(out)

# ---------- load data ----------
data_path = first_existing(DATA_CANDIDATES)
if not data_path:
    st.error("CSV not found in ./data.")
    st.stop()

df = load_csv(data_path)

# column fixes
if "NetSalesValue" not in df.columns:
    for alt_name in ["Net Sales Value", "Net_Sales_Value", "NetSales"]:
        if alt_name in df.columns:
            df = df.rename(columns={alt_name: "NetSalesValue"})
            break

if "Month" not in df.columns:
    st.error("CSV needs a 'Month' column.")
    st.stop()

if "Year" not in df.columns:
    df["Year"] = 2023  # fallback

# year picker
years = sorted(df["Year"].unique())
year = st.sidebar.selectbox("Year", years, index=len(years)-1)
d = df[df["Year"] == year].copy()

# identify key columns
prod_col = (
    "Product" if "Product" in d.columns
    else ("ProductClass" if "ProductClass" in d.columns else None)
)
cust_class_col = "CustomerClass" if "CustomerClass" in d.columns else None
salesperson_col = "Salesperson" if "Salesperson" in d.columns else None

if prod_col is None:
    st.warning("No Product / ProductClass column; product charts limited.")
if cust_class_col is None:
    st.warning("No CustomerClass column; customer overview will use row counts only by what’s available.")


# =========================================================
# A) Product priorities (ABC)
# =========================================================
st.subheader("A) Product priorities by sales (ABC)")

if prod_col:
    prod = (
        d.groupby(prod_col)["NetSalesValue"]
         .sum()
         .sort_values(ascending=False)
         .reset_index()
    )
    prod_total = prod["NetSalesValue"].sum()
    prod["cum_pct"] = prod["NetSalesValue"].cumsum() / (prod_total if prod_total > 0 else 1) * 100
    prod["ABC"] = np.select(
        [prod["cum_pct"] <= 80, (prod["cum_pct"] > 80) & (prod["cum_pct"] <= 95)],
        ["A (high)", "B (medium)"], default="C (low)"
    )

    prod_display = prod.rename(columns={
        "NetSalesValue": "Total net sales",
        "cum_pct": "Cumulative % of sales",
        "ABC": "Priority band"
    })

    # font is now controlled by bump_table_font()
    st.dataframe(
        prod_display.head(30).style.format({
            "Total net sales": "{:,.0f}",
            "Cumulative % of sales": "{:.2f}"
        }),
        use_container_width=True
    )


    # Bar chart (with horizontal labels + bigger fonts)
    bars = (
        alt.Chart(prod_display.head(30))
        .mark_bar()
        .encode(
            x=alt.X(f"{prod_col}:N", sort="-y", title="Product / Class", axis=X_AXIS),
            y=alt.Y("Total net sales:Q", title="Total net sales", axis=Y_AXIS),
            color=alt.Color(
                "Priority band:N",
                scale=alt.Scale(domain=["A (high)", "B (medium)", "C (low)"],
                                range=["#4c78a8", "#f58518", "#72b7b2"])
            ),
            tooltip=[prod_col,
                     alt.Tooltip("Total net sales:Q", format=",.0f"),
                     alt.Tooltip("Cumulative % of sales:Q", format=".2f"),
                     "Priority band"]
        )
    )
    st.altair_chart(bars, use_container_width=True)

    # ---------- Slow movers ----------
    st.subheader("Product Active Months")
    d["MonthNum"] = d["Month"].apply(to_month_num)
    mcount = (
        d.dropna(subset=["MonthNum"])
         .groupby([prod_col, "MonthNum"])["NetSalesValue"]
         .sum()
         .reset_index()
    )
    active = mcount.groupby(prod_col)["MonthNum"].nunique().reset_index(name="Active months (out of 12)")
    slow = (
        active.merge(prod_display[[prod_col, "Total net sales"]], on=prod_col, how="left")
              .sort_values(["Active months (out of 12)", "Total net sales"], ascending=[True, True])
              .head(20)
    )
    st.dataframe(
        slow.style.format({"Total net sales": "{:,.0f}"}),
        use_container_width=True
    )

# =========================================================
# B) Customer overview (simple) + top salesperson
# =========================================================
st.subheader("B) Customer overview (top classes by sales & orders)")

# Small legend for meanings
st.caption(
    "**Class references:** "
    "`CTRD` = trade customer, `CNTRD` = non-trade customer, "
    "`SD` = sub distribution, `KA` = key account, `HRC` = independent."
)

if cust_class_col:
    # Orders: use a doc/order id if available; else row count
    possible_order_cols = [c for c in ["InvoiceNo", "Invoice", "OrderID", "OrderNo", "DocumentNumber"] if c in d.columns]
    if possible_order_cols:
        order_col = possible_order_cols[0]
        group = (
            d.groupby(cust_class_col)
             .agg(**{
                 "Total net sales": ("NetSalesValue", "sum"),
                 "Number of orders": (order_col, "nunique"),
             })
             .reset_index()
        )
    else:
        group = (
            d.groupby(cust_class_col)
             .agg(**{
                 "Total net sales": ("NetSalesValue", "sum"),
                 "Number of orders": ("NetSalesValue", "size"),
             })
             .reset_index()
        )

    # Add meaning column (CTRD/CNTRD + suffix)
    group["Meaning"] = group[cust_class_col].astype(str).apply(class_meaning_for)

    # Add top salesperson per class (if we have the column)
    if salesperson_col:
        sp_sum = (
            d.groupby([cust_class_col, salesperson_col])["NetSalesValue"]
             .sum()
             .reset_index()
        )
        # pick the salesperson with max sales for each class
        idx = sp_sum.groupby(cust_class_col)["NetSalesValue"].idxmax()
        top_sp = (
            sp_sum.loc[idx]
                  .rename(columns={
                      salesperson_col: "Top salesperson",
                      "NetSalesValue": "Top salesperson sales"
                  })
        )
        group = group.merge(top_sp, on=cust_class_col, how="left")
    else:
        group["Top salesperson"] = ""
        group["Top salesperson sales"] = np.nan

    if group.empty:
        st.info("No rows for this year.")
    else:
        # KPIs: top by sales & top by orders
        top_sales_row = group.sort_values("Total net sales", ascending=False).iloc[0]
        top_orders_row = group.sort_values("Number of orders", ascending=False).iloc[0]

        k1, k2 = st.columns(2)
        k1.metric("Top class by total sales", f"{top_sales_row[cust_class_col]}", f"{top_sales_row['Total net sales']:,.0f}")
        k2.metric("Top class by number of orders", f"{top_orders_row[cust_class_col]}", f"{int(top_orders_row['Number of orders']):,}")

        # How many to show
        max_top = min(20, len(group))
        top_n = st.slider("Show top N classes", min_value=1, max_value=max_top, value=min(10, max_top), step=1)

        # Table (with Meaning + Top salesperson)
        table_cols = [
            cust_class_col, "Meaning", "Total net sales", "Number of orders", "Top salesperson", "Top salesperson sales"
        ]
        table = group.sort_values("Total net sales", ascending=False).head(top_n)[table_cols]
        st.dataframe(
            table.style.format({
                "Total net sales": "{:,.0f}",
                "Number of orders": "{:,.0f}",
                "Top salesperson sales": "{:,.0f}",
            }),
            use_container_width=True
        )

        # Charts: side-by-side bars (horizontal x-labels + bigger fonts)
        c1, c2 = st.columns(2)

        sales_bar = (
            alt.Chart(table)
            .mark_bar()
            .encode(
                x=alt.X(f"{cust_class_col}:N", sort="-y", title="Customer class", axis=X_AXIS),
                y=alt.Y("Total net sales:Q", title="Total net sales", axis=Y_AXIS),
                tooltip=[
                    cust_class_col, "Meaning",
                    alt.Tooltip("Total net sales:Q", format=",.0f"),
                    "Top salesperson",
                    alt.Tooltip("Top salesperson sales:Q", format=",.0f"),
                ],
            )
        )
        c1.altair_chart(sales_bar, use_container_width=True)

        orders_bar = (
            alt.Chart(group.sort_values("Number of orders", ascending=False).head(top_n))
            .mark_bar()
            .encode(
                x=alt.X(f"{cust_class_col}:N", sort="-y", title="Customer class", axis=X_AXIS),
                y=alt.Y("Number of orders:Q", title="Number of orders", axis=Y_AXIS),
                tooltip=[cust_class_col, "Meaning", alt.Tooltip("Number of orders:Q", format=",.0f")],
            )
        )
        c2.altair_chart(orders_bar, use_container_width=True)

        st.download_button(
            "Download customer class summary (CSV)",
            data=group.sort_values("Total net sales", ascending=False).to_csv(index=False).encode("utf-8"),
            file_name=f"customer_class_summary_{year}.csv",
            mime="text/csv"
        )
else:
    st.info("No **CustomerClass** column available. Add this field to your dataset to see the simple customer overview.")
