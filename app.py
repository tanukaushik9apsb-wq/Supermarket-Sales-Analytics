"""
Supermarket Sales Analytics — Production-Ready Streamlit App
=============================================================
Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"text.color": "#1f2328", "axes.labelcolor": "#1f2328"})

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Supermarket Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* General page typography */
        html, body, [class*="css"] { font-family: "Segoe UI", system-ui, sans-serif; }

        /* KPI cards */
        .kpi-card {
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 18px 22px;
            text-align: center;
        }
        .kpi-label  { font-size: 13px; color: #57606a; margin-bottom: 4px; }
        .kpi-value  { font-size: 30px; font-weight: 700; color: #1f2328; }
        .kpi-delta  { font-size: 12px; color: #3b82d4; margin-top: 4px; }

        /* Section headers */
        .section-header {
            font-size: 18px;
            font-weight: 600;
            color: #1f2328;
            border-left: 4px solid #3b82d4;
            padding-left: 10px;
            margin: 24px 0 12px 0;
        }

        /* Recommendation bullets */
        .rec-box {
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 18px 24px;
        }
        .rec-box li { margin-bottom: 8px; color: #1f2328; font-size: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# DATA INGESTION & HYGIENE
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "sales.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename common variants to canonical names
    rename_map = {
        "customer_type": "customer_type",
        "product_line": "product_line",
        "invoice_id": "invoice_id",
        "unit_price": "unit_price",
        "tax_5%": "tax",
        "gross_income": "gross_income",
        "gross_margin_percentage": "gross_margin_pct",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # Sanitise numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())

    # Quantity must be positive
    if "quantity" in df.columns:
        df["quantity"] = df["quantity"].abs()

    # Ensure a revenue column exists (Sales already includes tax in this dataset)
    if "sales" in df.columns and "revenue" not in df.columns:
        df["revenue"] = df["sales"]

    return df


df_raw = load_data()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔎 Filters")

    # Branch / City
    branch_col = "branch" if "branch" in df_raw.columns else None
    city_col   = "city"   if "city"   in df_raw.columns else None
    location_col = branch_col or city_col
    location_label = "Branch / City"
    if location_col:
        all_locations = sorted(df_raw[location_col].dropna().unique().tolist())
        selected_locations = st.multiselect(
            location_label, options=all_locations, default=all_locations
        )
    else:
        selected_locations = []

    # Product Line / Category
    product_col = (
        "product_line" if "product_line" in df_raw.columns
        else "category"  if "category"   in df_raw.columns
        else None
    )
    if product_col:
        all_products = sorted(df_raw[product_col].dropna().unique().tolist())
        selected_products = st.multiselect(
            "Product Line / Category", options=all_products, default=all_products
        )
    else:
        selected_products = []

    # Customer Type
    ctype_col = "customer_type" if "customer_type" in df_raw.columns else None
    if ctype_col:
        all_ctypes = sorted(df_raw[ctype_col].dropna().unique().tolist())
        selected_ctypes = st.multiselect(
            "Customer Type", options=all_ctypes, default=all_ctypes
        )
    else:
        selected_ctypes = []

    # Date range
    min_date = df_raw["date"].min().date()
    max_date = df_raw["date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )


# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df = df_raw.copy()

if location_col and selected_locations:
    df = df[df[location_col].isin(selected_locations)]
if product_col and selected_products:
    df = df[df[product_col].isin(selected_products)]
if ctype_col and selected_ctypes:
    df = df[df[ctype_col].isin(selected_ctypes)]
if len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0])
    end_dt   = pd.Timestamp(date_range[1])
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

if df.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar selections.")
    st.stop()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🛒 Supermarket Sales Analytics Dashboard")
st.markdown(
    f"Showing **{len(df):,}** transactions · "
    f"{df['date'].min().strftime('%b %d, %Y')} → {df['date'].max().strftime('%b %d, %Y')}"
)
st.markdown("---")

# ─────────────────────────────────────────────
# KPI METRIC CARDS
# ─────────────────────────────────────────────
total_revenue = df["revenue"].sum()
total_orders  = len(df)
aov           = total_revenue / total_orders if total_orders else 0
avg_rating    = df["rating"].mean() if "rating" in df.columns else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
for col, label, value, fmt in [
    (kpi1, "Total Revenue",        total_revenue, "${:,.2f}"),
    (kpi2, "Total Orders",         total_orders,  "{:,}"),
    (kpi3, "Avg Order Value (AOV)", aov,           "${:,.2f}"),
    (kpi4, "Avg Customer Rating",  avg_rating,    "{:.2f} ⭐"),
]:
    col.markdown(
        f"""<div class="kpi-card">
               <div class="kpi-label">{label}</div>
               <div class="kpi-value">{fmt.format(value)}</div>
           </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EDA — SECTION HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

# ── Row 1: Monthly Revenue Trend  +  Revenue by Product Category
col_l, col_r = st.columns(2)

with col_l:
    monthly = (
        df.groupby("month", as_index=False)["revenue"]
        .sum()
        .sort_values("month")
    )
    fig_trend = px.line(
        monthly, x="month", y="revenue",
        title="Monthly Revenue Trend",
        markers=True,
        labels={"month": "Month", "revenue": "Revenue ($)"},
        color_discrete_sequence=["#3b82d4"],
    )
    fig_trend.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font_color="#1f2328", title_font_size=15,
        xaxis=dict(tickangle=-30, showgrid=False),
        yaxis=dict(gridcolor="#e5e7eb"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_r:
    if product_col:
        prod_rev = (
            df.groupby(product_col, as_index=False)["revenue"]
            .sum()
            .sort_values("revenue")
        )
        fig_bar = px.bar(
            prod_rev, x="revenue", y=product_col,
            orientation="h",
            title="Revenue by Product Category",
            labels={"revenue": "Revenue ($)", product_col: ""},
            color="revenue",
            color_continuous_scale=["#bdd7f5", "#3b82d4"],
        )
        fig_bar.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font_color="#1f2328", title_font_size=15,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(showgrid=False),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ── Row 2: Customer Composition  +  Branch Performance
col_l2, col_r2 = st.columns(2)

with col_l2:
    if ctype_col:
        ctype_counts = df[ctype_col].value_counts().reset_index()
        ctype_counts.columns = ["customer_type", "count"]
        fig_pie = px.pie(
            ctype_counts, names="customer_type", values="count",
            title="Customer Composition",
            hole=0.45,
            color_discrete_sequence=["#3b82d4", "#7c5cd8"],
        )
        fig_pie.update_traces(textfont_size=13)
        fig_pie.update_layout(
            paper_bgcolor="#ffffff", font_color="#1f2328",
            title_font_size=15,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with col_r2:
    if location_col:
        branch_rev = (
            df.groupby(location_col, as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
        )
        fig_branch = px.bar(
            branch_rev, x=location_col, y="revenue",
            title="Branch / Regional Performance",
            labels={location_col: "Branch", "revenue": "Revenue ($)"},
            color="revenue",
            color_continuous_scale=["#bdd7f5", "#3b82d4"],
        )
        fig_branch.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font_color="#1f2328", title_font_size=15,
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#e5e7eb"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_branch, use_container_width=True)

# ─────────────────────────────────────────────
# PREDICTIVE CUSTOMER RETENTION MODEL
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🤖 Predictive Customer Retention Model</div>', unsafe_allow_html=True)

@st.cache_data
def build_rfm(data: pd.DataFrame) -> pd.DataFrame:
    id_col = "invoice_id" if "invoice_id" in data.columns else data.columns[0]
    snapshot = data["date"].max() + pd.Timedelta(days=1)

    rfm = (
        data.groupby(id_col)
        .agg(
            recency  =("date",    lambda x: (snapshot - x.max()).days),
            frequency=("revenue", "count"),
            monetary =("revenue", "sum"),
        )
        .reset_index()
    )
    rfm["aov"] = rfm["monetary"] / rfm["frequency"]

    # Churn label: recency in top quartile (longest since purchase) → high-risk (1)
    threshold = rfm["recency"].quantile(0.75)
    rfm["churn"] = (rfm["recency"] >= threshold).astype(int)
    return rfm

rfm_df = build_rfm(df)

features  = ["recency", "frequency", "monetary", "aov"]
X = rfm_df[features].values
y = rfm_df["churn"].values

# Need at least 2 classes
if len(np.unique(y)) < 2:
    st.info("Not enough class variation in filtered data to train the model. Please widen the date range or filters.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    # ── Layout: metrics  +  confusion matrix
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, label, val, fmt in [
        (mc1, "Accuracy",  acc,  "{:.1%}"),
        (mc2, "Precision", prec, "{:.1%}"),
        (mc3, "Recall",    rec,  "{:.1%}"),
        (mc4, "Test Samples", len(y_test), "{:,}"),
    ]:
        col.markdown(
            f"""<div class="kpi-card">
                   <div class="kpi-label">{label}</div>
                   <div class="kpi-value">{fmt.format(val)}</div>
               </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    cm_col, coef_col = st.columns([1, 1])

    with cm_col:
        fig_cm, ax = plt.subplots(figsize=(4, 3.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low-Risk", "High-Risk"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix", fontsize=13, color="#1f2328")
        ax.tick_params(colors="#1f2328")
        fig_cm.patch.set_facecolor("#ffffff")
        st.pyplot(fig_cm, use_container_width=False)

    with coef_col:
        st.markdown("**Model Feature Importances (Log-Reg Coefficients)**")
        coef_df = pd.DataFrame(
            {"Feature": features, "Coefficient": clf.coef_[0]}
        ).sort_values("Coefficient", ascending=True)
        fig_coef = px.bar(
            coef_df, x="Coefficient", y="Feature",
            orientation="h",
            color="Coefficient",
            color_continuous_scale=["#d64c4c", "#e5e7eb", "#3b82d4"],
            color_continuous_midpoint=0,
            labels={"Feature": "", "Coefficient": "Coefficient"},
        )
        fig_coef.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font_color="#1f2328", coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    with st.expander("ℹ️ How the model works"):
        st.markdown(
            """
            - **RFM aggregation**: each invoice is converted to four features —
              *Recency* (days since last purchase), *Frequency* (transaction count),
              *Monetary* (total spend), and *AOV* (average order value).
            - **Label**: invoices in the top 25 % of recency (longest inactive) are
              flagged as **high-risk churn (1)**; the rest are **low-risk (0)**.
            - A **Logistic Regression** classifier is trained on 75 % of records and
              evaluated on the remaining 25 %.
            - Results improve with more data — expand the date range or filters for
              a more robust model.
            """
        )

# ─────────────────────────────────────────────
# STRATEGIC RECOMMENDATIONS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">💡 Strategic Recommendations</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="rec-box">
    <ul>
      <li>
        <strong>Re-engage High-Risk Customers</strong> — Target the cohort flagged by the
        retention model with personalised discount vouchers or loyalty-point bonuses delivered
        via email/SMS within 7 days of their last purchase to reduce churn probability.
      </li>
      <li>
        <strong>Double Down on Top-Performing Product Lines</strong> — Allocate 15–20 % more
        shelf space and promotional budget to the highest-revenue product categories identified
        in the bar chart; consider bundling them with lower-performing lines to lift average
        basket size.
      </li>
      <li>
        <strong>Optimise Branch Operations by Revenue Tier</strong> — Conduct quarterly
        performance reviews for underperforming branches; deploy best-practice SOPs (staffing
        ratios, planogram layouts) from top-performing locations to close the revenue gap.
      </li>
      <li>
        <strong>Convert Normal Customers to Members</strong> — Given that Member customers
        generate comparable or higher AOV, introduce a low-friction sign-up incentive
        (e.g., 5 % off first purchase as a member) at the point-of-sale to grow the loyalty
        programme and improve long-term retention metrics.
      </li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; font-size:12px; color:#57606a; "
    "border-top:1px solid #e5e7eb; padding-top:12px;'>Made with IBM Bob</div>",
    unsafe_allow_html=True,
)
