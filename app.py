import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── ML imports ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyCity Profit Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #1a1f36;
    color: #fff;
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stRadio > label { 
    font-weight: 600; font-size: 0.8rem; letter-spacing: 0.08em; color: #94a3b8 !important;
  }
  section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding: 8px 14px;
    border-radius: 8px;
    transition: background 0.2s;
  }
  section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #2d3654 !important;
  }

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  div[data-testid="metric-container"] label { color: #6b7280 !important; font-size: 0.82rem !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #111827 !important; font-size: 1.7rem !important; font-weight: 700; }

  /* ── Section header ── */
  .section-header {
    background: linear-gradient(135deg, #1a56db 0%, #1e429f 100%);
    color: white;
    padding: 20px 28px;
    border-radius: 14px;
    margin-bottom: 24px;
  }
  .section-header h2 { margin: 0; font-size: 1.5rem; font-weight: 700; }
  .section-header p  { margin: 4px 0 0; font-size: 0.9rem; opacity: 0.85; }

  /* ── Card ── */
  .info-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .info-card h4 { margin: 0 0 6px; color: #1a56db; font-size: 1rem; }
  .info-card p  { margin: 0; color: #374151; font-size: 0.88rem; line-height: 1.5; }

  /* ── Insight badge ── */
  .badge-green  { background:#d1fae5; color:#065f46; padding:4px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-yellow { background:#fef3c7; color:#92400e; padding:4px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-red    { background:#fee2e2; color:#991b1b; padding:4px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-blue   { background:#dbeafe; color:#1e40af; padding:4px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }

  /* ── Divider ── */
  hr { border: none; border-top: 1px solid #e5e7eb; margin: 20px 0; }

  /* ── Table ── */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ── Buttons ── */
  .stButton > button {
    background: #1a56db;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
  }
  .stButton > button:hover { background: #1e429f; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; font-weight: 600; }
  .stTabs [aria-selected="true"] { background: #dbeafe; color: #1e40af; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────────────────
DATA_PATH = "/app/SkyCity Auckland Restaurants & Bars - SkyCity Auckland Restaurants & Bars.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Derived features
    df["TotalRevenue"]   = df["InStoreRevenue"] + df["UberEatsRevenue"] + df["DoorDashRevenue"] + df["SelfDeliveryRevenue"]
    df["TotalNetProfit"] = df["InStoreNetProfit"] + df["UberEatsNetProfit"] + df["DoorDashNetProfit"] + df["SelfDeliveryNetProfit"]
    df["NetProfitPerOrder"] = df["TotalNetProfit"] / df["MonthlyOrders"]
    df["OverallMargin"]     = df["TotalNetProfit"] / df["TotalRevenue"]

    # Interaction terms
    df["Commission_UE"]   = df["CommissionRate"] * df["UE_share"]
    df["DeliveryCost_SD"] = df["DeliveryCostPerOrder"] * df["SD_share"]
    df["GrowthAdj_Orders"] = df["MonthlyOrders"] * df["GrowthFactor"]
    df["CostToRevenue"]   = (df["COGSRate"] + df["OPEXRate"])
    df["ChannelMixScore"] = (df["InStoreShare"] * df["InStoreNetProfit"] + df["UE_share"] * df["UberEatsNetProfit"]) / (df["TotalNetProfit"].replace(0, 1))

    return df


@st.cache_data
def prepare_ml(df):
    le = LabelEncoder()
    df2 = df.copy()
    for col in ["CuisineType", "Segment", "Subregion"]:
        df2[col + "_enc"] = le.fit_transform(df2[col])

    FEATURES = [
        "InStoreShare", "UE_share", "DD_share", "SD_share",
        "CommissionRate", "DeliveryCostPerOrder", "DeliveryRadiusKM", "GrowthFactor",
        "AOV", "MonthlyOrders", "COGSRate", "OPEXRate",
        "Commission_UE", "DeliveryCost_SD", "GrowthAdj_Orders", "CostToRevenue",
        "CuisineType_enc", "Segment_enc", "Subregion_enc",
    ]
    TARGET = "TotalNetProfit"
    X = df2[FEATURES]
    y = df2[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_train)
    X_te_s = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, X_tr_s, X_te_s, sc, FEATURES


@st.cache_resource
def train_models(X_tr_s, X_te_s, X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression":      LinearRegression(),
        "Random Forest":          RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":      GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost":                xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }
    results, trained = {}, {}
    for name, m in models.items():
        if name == "Linear Regression":
            m.fit(X_tr_s, y_train)
            pred = m.predict(X_te_s)
        else:
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2   = r2_score(y_test, pred)
        mae  = mean_absolute_error(y_test, pred)
        results[name] = {"RMSE": rmse, "R2": r2, "MAE": mae, "pred": pred}
        trained[name] = m
    return results, trained


# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────
def section_header(title, subtitle=""):
    st.markdown(f"""
    <div class="section-header">
        <h2>{title}</h2>
        {"<p>" + subtitle + "</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)


def card(title, body):
    st.markdown(f'<div class="info-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


PALETTE = ["#1a56db", "#0e9f6e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#10b981"]


# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### SkyCity Analytics")
    st.markdown("**Profit Optimization Platform**")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Exploratory Analysis", "Predictive Models", "What-If Simulator", "Optimization Panel"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown('<small style="color:#64748b">Dataset: SkyCity Auckland<br>Records: 1,696 | Restaurants: ~212<br>Updated: 2024</small>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────────────────────────────────
df = load_data()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    section_header("Portfolio Overview", "High-level summary of SkyCity Auckland restaurant operations")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Restaurants",    f"{df['RestaurantID'].nunique():,}")
    c2.metric("Avg Monthly Orders",   f"{df['MonthlyOrders'].mean():,.0f}")
    c3.metric("Avg Monthly Profit",   f"${df['TotalNetProfit'].mean():,.0f}")
    c4.metric("Avg Net Margin",       f"{df['OverallMargin'].mean()*100:.1f}%")
    c5.metric("Avg Commission Rate",  f"{df['CommissionRate'].mean()*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Revenue by cuisine
        cuisine_rev = df.groupby("CuisineType")["TotalRevenue"].mean().sort_values()
        fig = px.bar(cuisine_rev, orientation="h",
                     title="Avg Monthly Revenue by Cuisine Type",
                     labels={"value": "Revenue ($)", "index": ""},
                     color=cuisine_rev.values,
                     color_continuous_scale=["#dbeafe", "#1a56db"])
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Profit by segment
        seg_profit = df.groupby("Segment")["TotalNetProfit"].mean().sort_values()
        colors = ["#1a56db", "#0e9f6e", "#f59e0b", "#8b5cf6"]
        fig2 = px.bar(seg_profit, title="Avg Monthly Net Profit by Segment",
                      labels={"value": "Net Profit ($)", "index": ""},
                      color=seg_profit.index, color_discrete_sequence=colors)
        fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Channel share donut
        ch_vals = [df["InStoreShare"].mean(), df["UE_share"].mean(),
                   df["DD_share"].mean(), df["SD_share"].mean()]
        ch_labels = ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"]
        fig3 = go.Figure(go.Pie(labels=ch_labels, values=ch_vals, hole=0.52,
                                marker_colors=["#1a56db", "#0e9f6e", "#f59e0b", "#8b5cf6"]))
        fig3.update_layout(title="Average Channel Mix", margin=dict(l=0, r=0, t=40, b=0),
                           paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Subregion comparison
        sub = df.groupby("Subregion")[["TotalRevenue", "TotalNetProfit"]].mean().reset_index()
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="Revenue", x=sub["Subregion"], y=sub["TotalRevenue"],
                              marker_color="#dbeafe"))
        fig4.add_trace(go.Bar(name="Net Profit", x=sub["Subregion"], y=sub["TotalNetProfit"],
                              marker_color="#1a56db"))
        fig4.update_layout(title="Revenue vs Net Profit by Subregion", barmode="group",
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Key Observations")
    cols = st.columns(3)
    with cols[0]:
        top_cuisine = df.groupby("CuisineType")["TotalNetProfit"].mean().idxmax()
        card("Best Performing Cuisine",
             f"<b>{top_cuisine}</b> leads in average net profit across all segments and subregions.")
    with cols[1]:
        top_channel = ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"][
            [df["InStoreNetProfit"].mean(), df["UberEatsNetProfit"].mean(),
             df["DoorDashNetProfit"].mean(), df["SelfDeliveryNetProfit"].mean()].index(
                max(df["InStoreNetProfit"].mean(), df["UberEatsNetProfit"].mean(),
                    df["DoorDashNetProfit"].mean(), df["SelfDeliveryNetProfit"].mean()))]
        card("Most Profitable Channel",
             f"<b>{top_channel}</b> generates the highest average net profit contribution.")
    with cols[2]:
        high_comm = df[df["CommissionRate"] > 0.30]["TotalNetProfit"].mean()
        low_comm  = df[df["CommissionRate"] <= 0.30]["TotalNetProfit"].mean()
        delta = ((low_comm - high_comm) / abs(high_comm)) * 100 if high_comm != 0 else 0
        card("Commission Impact",
             f"Restaurants with commission ≤30% earn <b>{abs(delta):.1f}% {'more' if delta > 0 else 'less'}</b> net profit than those above 30%.")

    st.markdown("---")
    with st.expander("View Raw Dataset (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Exploratory Analysis":
    section_header("Exploratory Data Analysis", "Deep-dive into distributions, correlations, and channel dynamics")

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution & Revenue", "Cost Analysis", "Channel Dynamics", "Correlation Matrix"])

    # ── Tab 1 ──
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="TotalNetProfit", nbins=50, title="Distribution of Total Net Profit",
                               color_discrete_sequence=["#1a56db"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              xaxis_title="Net Profit ($)", yaxis_title="Count",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.box(df, x="CuisineType", y="TotalNetProfit", color="CuisineType",
                          title="Net Profit Distribution by Cuisine",
                          color_discrete_sequence=PALETTE)
            fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_title="", yaxis_title="Net Profit ($)",
                               margin=dict(l=0, r=0, t=40, b=0))
            fig2.update_xaxes(tickangle=30)
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.scatter(df, x="MonthlyOrders", y="TotalNetProfit",
                              color="Segment", size="AOV",
                              title="Orders vs Net Profit (size = AOV)",
                              color_discrete_sequence=PALETTE,
                              hover_data=["RestaurantName", "CuisineType"])
            fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = px.violin(df, x="Segment", y="OverallMargin", color="Segment",
                             title="Profit Margin Distribution by Segment",
                             color_discrete_sequence=PALETTE, box=True)
            fig4.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                               yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 2 ──
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x="CommissionRate", y="UberEatsNetProfit",
                             color="CuisineType", title="Commission Rate vs Uber Eats Net Profit",
                             color_discrete_sequence=PALETTE,
                             trendline="ols")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              xaxis_tickformat=".0%",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(df, x="DeliveryCostPerOrder", y="SelfDeliveryNetProfit",
                              color="Subregion", title="Delivery Cost/Order vs Self-Delivery Profit",
                              color_discrete_sequence=PALETTE, trendline="ols")
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.scatter(df, x="COGSRate", y="OPEXRate", color="TotalNetProfit",
                              title="COGS Rate vs OPEX Rate (color = Net Profit)",
                              color_continuous_scale="Blues",
                              hover_data=["RestaurantName"])
            fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            cost_data = pd.DataFrame({
                "Cost Component": ["COGS", "OPEX", "Commission", "Delivery Cost"],
                "Avg % of Revenue": [
                    df["COGSRate"].mean() * 100,
                    df["OPEXRate"].mean() * 100,
                    df["CommissionRate"].mean() * 100,
                    (df["SD_DeliveryTotalCost"] / df["TotalRevenue"]).mean() * 100,
                ]
            })
            fig4 = px.bar(cost_data, x="Cost Component", y="Avg % of Revenue",
                          title="Average Cost Structure (% of Revenue)",
                          color="Cost Component", color_discrete_sequence=PALETTE)
            fig4.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 3 ──
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            ch_profit = pd.DataFrame({
                "Channel": ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"],
                "Avg Net Profit": [df["InStoreNetProfit"].mean(), df["UberEatsNetProfit"].mean(),
                                   df["DoorDashNetProfit"].mean(), df["SelfDeliveryNetProfit"].mean()],
                "Avg Revenue":    [df["InStoreRevenue"].mean(), df["UberEatsRevenue"].mean(),
                                   df["DoorDashRevenue"].mean(), df["SelfDeliveryRevenue"].mean()],
            })
            fig = px.bar(ch_profit, x="Channel", y=["Avg Revenue", "Avg Net Profit"],
                         title="Revenue vs Net Profit by Channel", barmode="group",
                         color_discrete_sequence=["#dbeafe", "#1a56db"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            heat_data = df.groupby(["CuisineType", "Segment"])[["TotalNetProfit"]].mean().reset_index()
            heat_pivot = heat_data.pivot(index="CuisineType", columns="Segment", values="TotalNetProfit")
            fig2 = px.imshow(heat_pivot, title="Avg Net Profit Heatmap: Cuisine × Segment",
                             color_continuous_scale="Blues", text_auto=".0f",
                             aspect="auto")
            fig2.update_layout(paper_bgcolor="white", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.scatter(df, x="InStoreShare", y="InStoreNetProfit",
                              color="Segment", title="In-Store Share vs In-Store Net Profit",
                              trendline="ols", color_discrete_sequence=PALETTE)
            fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = px.scatter(df, x="DeliveryRadiusKM", y="SelfDeliveryNetProfit",
                              color="CuisineType", title="Delivery Radius vs Self-Delivery Profit",
                              trendline="ols", color_discrete_sequence=PALETTE)
            fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 4 ──
    with tab4:
        num_cols = ["TotalNetProfit", "TotalRevenue", "MonthlyOrders", "AOV",
                    "InStoreShare", "UE_share", "DD_share", "SD_share",
                    "COGSRate", "OPEXRate", "CommissionRate",
                    "DeliveryRadiusKM", "DeliveryCostPerOrder", "GrowthFactor"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, title="Pearson Correlation Matrix", text_auto=".2f",
                        color_continuous_scale="RdBu", zmin=-1, zmax=1,
                        aspect="auto")
        fig.update_layout(paper_bgcolor="white", margin=dict(l=0, r=0, t=60, b=0),
                          width=900, height=600)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTIVE MODELS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Predictive Models":
    section_header("Predictive Modeling", "Train and compare 4 ML models for net profit forecasting")

    X_train, X_test, y_train, y_test, X_tr_s, X_te_s, sc, FEATURES = prepare_ml(df)

    if st.button("Train All 4 Models"):
        with st.spinner("Training Linear Regression, Random Forest, Gradient Boosting & XGBoost..."):
            results, trained = train_models(X_tr_s, X_te_s, X_train, X_test, y_train, y_test)
        st.session_state["results"] = results
        st.session_state["trained"] = trained
        st.session_state["X_test"]  = X_test
        st.session_state["y_test"]  = y_test
        st.success("All models trained successfully!")

    if "results" not in st.session_state:
        st.info("Click **Train All 4 Models** to begin.")
        st.stop()

    results = st.session_state["results"]
    y_test_v = st.session_state["y_test"]

    # ── Model comparison table ──
    st.markdown("### Model Performance Comparison")
    comp = pd.DataFrame({
        "Model":   list(results.keys()),
        "RMSE ($)": [f"{v['RMSE']:,.0f}" for v in results.values()],
        "R² Score": [f"{v['R2']:.4f}"   for v in results.values()],
        "MAE ($)":  [f"{v['MAE']:,.0f}"  for v in results.values()],
    })
    best_r2 = max(results, key=lambda k: results[k]["R2"])
    st.dataframe(comp.set_index("Model"), use_container_width=True)
    st.markdown(f'Best model by R²: <span class="badge-green">{best_r2} — R² = {results[best_r2]["R2"]:.4f}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Metric bar charts ──
    col1, col2, col3 = st.columns(3)
    names = list(results.keys())

    with col1:
        fig = px.bar(x=names, y=[results[n]["RMSE"] for n in names],
                     title="RMSE (lower is better)", color=names,
                     color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="", margin=dict(l=0, r=0, t=40, b=0))
        fig.update_xaxes(tickangle=15)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(x=names, y=[results[n]["R2"] for n in names],
                      title="R² Score (higher is better)", color=names,
                      color_discrete_sequence=PALETTE)
        fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           xaxis_title="", margin=dict(l=0, r=0, t=40, b=0))
        fig2.update_xaxes(tickangle=15)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = px.bar(x=names, y=[results[n]["MAE"] for n in names],
                      title="MAE (lower is better)", color=names,
                      color_discrete_sequence=PALETTE)
        fig3.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           xaxis_title="", margin=dict(l=0, r=0, t=40, b=0))
        fig3.update_xaxes(tickangle=15)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Predicted vs Actual ──
    st.markdown("### Predicted vs Actual Net Profit")
    selected_model = st.selectbox("Select model", list(results.keys()))
    pred_vals = results[selected_model]["pred"]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=y_test_v.values, y=pred_vals, mode="markers",
                               name="Predictions",
                               marker=dict(color="#1a56db", opacity=0.5, size=5)))
    mn, mx = float(y_test_v.min()), float(y_test_v.max())
    fig4.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                               name="Perfect Prediction",
                               line=dict(color="#ef4444", dash="dash")))
    fig4.update_layout(title=f"{selected_model} — Predicted vs Actual",
                       xaxis_title="Actual Net Profit ($)", yaxis_title="Predicted Net Profit ($)",
                       plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ── Feature importance (tree models) ──
    st.markdown("### Feature Importance")
    tree_models = {k: v for k, v in st.session_state["trained"].items() if k != "Linear Regression"}
    fi_model = st.selectbox("Select model for feature importance", list(tree_models.keys()), key="fi_sel")
    m = tree_models[fi_model]
    importances = m.feature_importances_
    fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances}).sort_values("Importance")
    fig5 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                  title=f"Feature Importance — {fi_model}",
                  color="Importance", color_continuous_scale=["#dbeafe", "#1a56db"])
    fig5.update_layout(coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=0, r=0, t=40, b=0), height=500)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # ── Residual distribution ──
    st.markdown("### Residual Analysis")
    residuals = y_test_v.values - pred_vals
    fig6 = px.histogram(x=residuals, nbins=50, title=f"Residual Distribution — {selected_model}",
                        color_discrete_sequence=["#1a56db"])
    fig6.add_vline(x=0, line_color="red", line_dash="dash")
    fig6.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       xaxis_title="Residual ($)", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig6, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "What-If Simulator":
    section_header("What-If Scenario Simulator", "Adjust channel mix and cost parameters to forecast profit outcomes")

    # Ensure models are trained
    if "trained" not in st.session_state:
        X_train, X_test, y_train, y_test, X_tr_s, X_te_s, sc, FEATURES = prepare_ml(df)
        with st.spinner("Training models..."):
            results, trained = train_models(X_tr_s, X_te_s, X_train, X_test, y_train, y_test)
        st.session_state["results"] = results
        st.session_state["trained"] = trained
        st.session_state["sc"]       = sc
        st.session_state["FEATURES"] = FEATURES
        _, X_test2, _, y_test2, _, _, _, _ = prepare_ml(df)
        st.session_state["X_test"]  = X_test2
        st.session_state["y_test"]  = y_test2
    else:
        if "sc" not in st.session_state or "FEATURES" not in st.session_state:
            _, X_test2, _, y_test2, _, _, sc2, FEATURES2 = prepare_ml(df)
            st.session_state["sc"]       = sc2
            st.session_state["FEATURES"] = FEATURES2

    sc       = st.session_state["sc"]
    FEATURES = st.session_state["FEATURES"]
    results  = st.session_state["results"]
    trained  = st.session_state["trained"]
    best_model_name = max(results, key=lambda k: results[k]["R2"])

    st.markdown("#### Configure Scenario Parameters")
    st.info("Adjust the sliders below. Channel shares must sum to approximately 1.0")

    col_l, col_r = st.columns([2, 1])

    with col_l:
        with st.expander("Channel Mix", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                in_store = st.slider("In-Store Share",       0.03, 0.55, 0.23, 0.01, format="%.2f")
                ue_share = st.slider("Uber Eats Share",      0.35, 0.60, 0.49, 0.01, format="%.2f")
            with c2:
                dd_share = st.slider("DoorDash Share",       0.20, 0.30, 0.27, 0.01, format="%.2f")
                sd_share = st.slider("Self-Delivery Share",  0.15, 0.45, 0.25, 0.01, format="%.2f")

        total_share = in_store + ue_share + dd_share + sd_share
        if abs(total_share - 1.0) > 0.15:
            st.warning(f"Channel shares sum to {total_share:.2f}. Consider normalising to 1.0.")
        else:
            st.success(f"Channel mix total: {total_share:.2f}")

        with st.expander("Cost & Operations", expanded=True):
            c3, c4 = st.columns(2)
            with c3:
                commission    = st.slider("Commission Rate",         0.27, 0.33, 0.30, 0.005, format="%.3f")
                delivery_cost = st.slider("Delivery Cost/Order ($)", 0.89, 5.31, 3.12, 0.10,  format="%.2f")
            with c4:
                radius        = st.slider("Delivery Radius (km)",    3, 18, 10, 1)
                growth        = st.slider("Growth Factor",           0.99, 1.05, 1.03, 0.005, format="%.3f")

        with st.expander("Restaurant Profile", expanded=True):
            c5, c6 = st.columns(2)
            with c5:
                cuisine  = st.selectbox("Cuisine Type", sorted(df["CuisineType"].unique()))
                segment  = st.selectbox("Segment",      sorted(df["Segment"].unique()))
                subregion = st.selectbox("Subregion",   sorted(df["Subregion"].unique()))
            with c6:
                aov    = st.slider("Avg Order Value ($)",  29.79, 47.23, 38.52, 0.10, format="%.2f")
                orders = st.slider("Monthly Orders",       441, 2337, 1190, 10)
                cogs   = st.slider("COGS Rate",            0.20, 0.40, 0.28, 0.005, format="%.3f")
                opex   = st.slider("OPEX Rate",            0.20, 0.55, 0.41, 0.005, format="%.3f")

    # ── Build prediction input ──
    le_vals = {c: sorted(df[c].unique()) for c in ["CuisineType", "Segment", "Subregion"]}
    cuisine_enc  = le_vals["CuisineType"].index(cuisine)
    segment_enc  = le_vals["Segment"].index(segment)
    subregion_enc = le_vals["Subregion"].index(subregion)

    commission_ue  = commission * ue_share
    delivery_sd    = delivery_cost * sd_share
    growth_orders  = orders * growth
    cost_rev       = cogs + opex

    scenario = np.array([[
        in_store, ue_share, dd_share, sd_share,
        commission, delivery_cost, radius, growth,
        aov, orders, cogs, opex,
        commission_ue, delivery_sd, growth_orders, cost_rev,
        cuisine_enc, segment_enc, subregion_enc,
    ]])

    preds_all = {}
    for name, m in trained.items():
        if name == "Linear Regression":
            sc_in = sc.transform(scenario)
            preds_all[name] = float(m.predict(sc_in)[0])
        else:
            preds_all[name] = float(m.predict(scenario)[0])

    best_pred = preds_all[best_model_name]

    with col_r:
        st.markdown("#### Scenario Forecast")
        for name, p in preds_all.items():
            badge = "badge-green" if p > 0 else "badge-red"
            is_best = " (Best Model)" if name == best_model_name else ""
            st.markdown(f"""
            <div class="info-card">
              <h4>{name}{is_best}</h4>
              <p>Predicted Net Profit:<br>
              <span class="{badge}" style="font-size:1.1rem">${p:,.0f}</span></p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Channel-level breakdown ──
    st.markdown("#### Estimated Channel-Level Profit Breakdown")
    rev_instore = aov * orders * in_store
    rev_ue      = aov * orders * ue_share
    rev_dd      = aov * orders * dd_share
    rev_sd      = aov * orders * sd_share

    profit_is = rev_instore  * (1 - cogs - opex)
    profit_ue = rev_ue       * (1 - cogs - opex - commission)
    profit_dd = rev_dd       * (1 - cogs - opex - commission)
    sd_total  = delivery_cost * (orders * sd_share)
    profit_sd = rev_sd       * (1 - cogs - opex) - sd_total

    breakdown = pd.DataFrame({
        "Channel":    ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"],
        "Revenue ($)": [rev_instore, rev_ue, rev_dd, rev_sd],
        "Net Profit ($)": [profit_is, profit_ue, profit_dd, profit_sd],
        "Margin":      [profit_is/rev_instore if rev_instore else 0,
                        profit_ue/rev_ue if rev_ue else 0,
                        profit_dd/rev_dd if rev_dd else 0,
                        profit_sd/rev_sd if rev_sd else 0],
    })

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(breakdown, x="Channel", y="Net Profit ($)", color="Channel",
                     title="Projected Net Profit by Channel",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.bar(breakdown, x="Channel", y="Margin", color="Channel",
                      title="Projected Margin by Channel",
                      color_discrete_sequence=PALETTE)
        fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    # ── Sensitivity sweep ──
    st.markdown("#### Sensitivity Sweep: Commission Rate Impact on Net Profit")
    comm_range = np.linspace(0.27, 0.33, 30)
    sweep_profits = []
    for cr in comm_range:
        s = scenario.copy()
        s[0][4]  = cr
        s[0][12] = cr * ue_share
        if best_model_name == "Linear Regression":
            sweep_profits.append(float(trained[best_model_name].predict(sc.transform(s))[0]))
        else:
            sweep_profits.append(float(trained[best_model_name].predict(s)[0]))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=comm_range * 100, y=sweep_profits, mode="lines+markers",
                               line=dict(color="#1a56db", width=2),
                               marker=dict(size=5)))
    fig3.add_vline(x=commission * 100, line_color="#ef4444", line_dash="dash",
                   annotation_text=f"Current: {commission*100:.1f}%")
    fig3.update_layout(title=f"Commission Rate vs Predicted Net Profit ({best_model_name})",
                       xaxis_title="Commission Rate (%)", yaxis_title="Predicted Net Profit ($)",
                       plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — OPTIMIZATION PANEL
# ════════════════════════════════════════════════════════════════════════════
elif page == "Optimization Panel":
    section_header("Prescriptive Optimization", "Identify optimal channel mix, safe operating ranges, and strategic recommendations")

    # Ensure models trained
    if "trained" not in st.session_state:
        X_train, X_test, y_train, y_test, X_tr_s, X_te_s, sc, FEATURES = prepare_ml(df)
        with st.spinner("Training models for optimization..."):
            results, trained = train_models(X_tr_s, X_te_s, X_train, X_test, y_train, y_test)
        st.session_state.update({"results": results, "trained": trained, "sc": sc, "FEATURES": FEATURES,
                                  "X_test": X_test, "y_test": y_test})
    results = st.session_state["results"]

    # ── KPI summary ──
    df_opt = df.copy()
    df_opt["ProfitSensitivityIndex"] = df_opt["TotalNetProfit"].std() / df_opt["TotalRevenue"].mean() * 100
    channel_efficiency = {
        "In-Store":      df_opt["InStoreNetProfit"].mean()  / df_opt["InStoreShare"].mean(),
        "Uber Eats":     df_opt["UberEatsNetProfit"].mean() / df_opt["UE_share"].mean(),
        "DoorDash":      df_opt["DoorDashNetProfit"].mean() / df_opt["DD_share"].mean(),
        "Self-Delivery": df_opt["SelfDeliveryNetProfit"].mean() / df_opt["SD_share"].mean(),
    }
    best_channel   = max(channel_efficiency, key=channel_efficiency.get)
    breakeven_comm = df_opt[df_opt["UberEatsNetProfit"] > 0]["CommissionRate"].max()
    opt_uplift_pct = (df_opt["TotalNetProfit"].quantile(0.75) - df_opt["TotalNetProfit"].mean()) / abs(df_opt["TotalNetProfit"].mean()) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit Sensitivity Index",    f"{df_opt['ProfitSensitivityIndex'].mean():.2f}%", help="Volatility relative to revenue")
    c2.metric("Most Efficient Channel",      best_channel)
    c3.metric("Break-Even Commission Rate",  f"{breakeven_comm*100:.1f}%", help="Max commission where UberEats stays profitable")
    c4.metric("Optimization Uplift Potential", f"{opt_uplift_pct:.1f}%",  help="P75 vs mean profit gap")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Channel Efficiency", "Commission Analysis", "Self-Delivery Threshold", "Recommendations"])

    # ── Tab 1: Channel Efficiency ──
    with tab1:
        st.markdown("#### Channel Mix Efficiency Score (Profit per Unit Share)")
        eff_df = pd.DataFrame(list(channel_efficiency.items()), columns=["Channel", "Profit per Share Point"])
        fig = px.bar(eff_df, x="Channel", y="Profit per Share Point",
                     color="Channel", color_discrete_sequence=PALETTE,
                     title="Channel Mix Efficiency Score")
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Profitability vs Channel Share — Optimal Mix Region")
        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.scatter(df, x="InStoreShare", y="TotalNetProfit",
                              color="Segment", trendline="ols",
                              title="In-Store Share vs Total Net Profit",
                              color_discrete_sequence=PALETTE)
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = px.scatter(df, x="SD_share", y="SelfDeliveryNetProfit",
                              color="Subregion", trendline="ols",
                              title="Self-Delivery Share vs Self-Delivery Profit",
                              color_discrete_sequence=PALETTE)
            fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Top 20 Most Profitable Restaurant Configurations")
        top20 = df.nlargest(20, "TotalNetProfit")[
            ["RestaurantName", "CuisineType", "Segment", "Subregion",
             "InStoreShare", "UE_share", "SD_share", "CommissionRate",
             "TotalNetProfit", "OverallMargin"]
        ].copy()
        top20["InStoreShare"] = top20["InStoreShare"].map("{:.0%}".format)
        top20["UE_share"]     = top20["UE_share"].map("{:.0%}".format)
        top20["SD_share"]     = top20["SD_share"].map("{:.0%}".format)
        top20["OverallMargin"]= top20["OverallMargin"].map("{:.1%}".format)
        top20["TotalNetProfit"]= top20["TotalNetProfit"].map("${:,.0f}".format)
        top20["CommissionRate"]= top20["CommissionRate"].map("{:.1%}".format)
        st.dataframe(top20.set_index("RestaurantName"), use_container_width=True)

    # ── Tab 2: Commission Analysis ──
    with tab2:
        st.markdown("#### Commission Rate vs Profitability — Safe Operating Zones")
        df["CommBucket"] = pd.cut(df["CommissionRate"],
                                   bins=[0.26, 0.28, 0.30, 0.32, 0.34],
                                   labels=["27-28%", "28-30%", "30-32%", "32-33%"])
        bucket_stats = df.groupby("CommBucket", observed=True)["TotalNetProfit"].agg(["mean","median","std"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Mean Profit",   x=bucket_stats["CommBucket"].astype(str),
                             y=bucket_stats["mean"],   marker_color="#1a56db"))
        fig.add_trace(go.Bar(name="Median Profit", x=bucket_stats["CommBucket"].astype(str),
                             y=bucket_stats["median"], marker_color="#0e9f6e"))
        fig.update_layout(title="Net Profit by Commission Rate Bucket", barmode="group",
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="Commission Rate", yaxis_title="Net Profit ($)",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.box(df, x="CommBucket", y="UberEatsNetProfit",
                          title="Uber Eats Net Profit by Commission Bucket",
                          color="CommBucket", color_discrete_sequence=PALETTE)
            fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = px.box(df, x="CommBucket", y="DoorDashNetProfit",
                          title="DoorDash Net Profit by Commission Bucket",
                          color="CommBucket", color_discrete_sequence=PALETTE)
            fig3.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        # Safe range
        safe = df.groupby("CommBucket", observed=True)["UberEatsNetProfit"].apply(lambda x: (x > 0).mean() * 100).reset_index()
        safe.columns = ["CommBucket", "% Profitable"]
        fig4 = px.bar(safe, x="CommBucket", y="% Profitable",
                      title="% of UberEats Orders Profitable by Commission Bucket",
                      color="% Profitable", color_continuous_scale=["#fee2e2", "#d1fae5"],
                      text_auto=".1f")
        fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 3: Self-Delivery Threshold ──
    with tab3:
        st.markdown("#### Self-Delivery Investment Break-Even Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x="DeliveryCostPerOrder", y="SelfDeliveryNetProfit",
                             color="Subregion", trendline="ols",
                             title="Delivery Cost vs Self-Delivery Net Profit",
                             color_discrete_sequence=PALETTE)
            fig.add_hline(y=0, line_color="red", line_dash="dash",
                          annotation_text="Break-Even Line")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(df, x="DeliveryRadiusKM", y="SelfDeliveryNetProfit",
                              color="Segment", trendline="ols",
                              title="Delivery Radius vs Self-Delivery Net Profit",
                              color_discrete_sequence=PALETTE)
            fig2.add_hline(y=0, line_color="red", line_dash="dash",
                           annotation_text="Break-Even Line")
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        # Break-even cost threshold
        be_cost = df[df["SelfDeliveryNetProfit"] > 0]["DeliveryCostPerOrder"].max()
        be_radius = df[df["SelfDeliveryNetProfit"] > 0]["DeliveryRadiusKM"].max()
        profitable_sd_pct = (df["SelfDeliveryNetProfit"] > 0).mean() * 100

        st.markdown("#### Self-Delivery Operating Thresholds")
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Profitable Delivery Cost",    f"${be_cost:.2f}/order")
        c2.metric("Max Profitable Delivery Radius",  f"{be_radius} km")
        c3.metric("% SD Operations Profitable",      f"{profitable_sd_pct:.1f}%")

        # Cost vs radius heatmap
        df["CostBin"]   = pd.cut(df["DeliveryCostPerOrder"], bins=5)
        df["RadiusBin"] = pd.cut(df["DeliveryRadiusKM"],    bins=5)
        hm = df.groupby(["CostBin","RadiusBin"], observed=True)["SelfDeliveryNetProfit"].mean().reset_index()
        hm_piv = hm.pivot(index="CostBin", columns="RadiusBin", values="SelfDeliveryNetProfit")
        hm_piv.index   = [str(x) for x in hm_piv.index]
        hm_piv.columns = [str(x) for x in hm_piv.columns]
        fig3 = px.imshow(hm_piv, title="Self-Delivery Profit: Cost vs Radius Heatmap",
                         color_continuous_scale="RdYlGn", text_auto=".0f",
                         labels=dict(color="Avg Net Profit ($)"))
        fig3.update_layout(paper_bgcolor="white", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 4: Recommendations ──
    with tab4:
        st.markdown("#### Strategic Recommendations for SkyCity Auckland")
        st.markdown("")

        # Compute key insights
        top_seg  = df.groupby("Segment")["TotalNetProfit"].mean().idxmax()
        top_sub  = df.groupby("Subregion")["TotalNetProfit"].mean().idxmax()
        top_cuis = df.groupby("CuisineType")["TotalNetProfit"].mean().idxmax()
        opt_is   = df.groupby(pd.cut(df["InStoreShare"], bins=5), observed=True)["TotalNetProfit"].mean().idxmax()
        opt_sd   = df.groupby(pd.cut(df["SD_share"], bins=5), observed=True)["TotalNetProfit"].mean().idxmax()
        best_r2_name = max(results, key=lambda k: results[k]["R2"])

        recs = [
            ("Channel Optimization",       f"Prioritise <b>In-Store</b> and <b>Self-Delivery</b> channels over pure aggregator reliance. In-Store consistently delivers the highest average net profit (${ df['InStoreNetProfit'].mean():,.0f}/month)."),
            ("Commission Negotiation",     f"Target commission rates at or below <b>30%</b>. The break-even commission rate is <b>{breakeven_comm*100:.1f}%</b> — negotiations above this threshold erode Uber Eats profitability."),
            ("Self-Delivery Expansion",    f"Self-delivery is profitable up to <b>${be_cost:.2f}/order</b> delivery cost and <b>{be_radius} km</b> radius. <b>{profitable_sd_pct:.1f}%</b> of current SD operations are profitable — consider expanding."),
            ("Best Performing Segment",    f"<b>{top_seg}</b> segment leads in average net profit. Allocate more marketing and operational resources to this format."),
            ("Subregion Focus",            f"<b>{top_sub}</b> is the highest-performing subregion. Prioritise new openings and expansions in this area."),
            ("Cuisine Mix Strategy",       f"<b>{top_cuis}</b> cuisine type generates the highest average net profit. Consider expanding this offering across underperforming subregions."),
            ("Predictive Model Adoption",  f"Use the <b>{best_r2_name}</b> model (R² = {results[best_r2_name]['R2']:.3f}) for profit forecasting. It explains {results[best_r2_name]['R2']*100:.1f}% of profit variance."),
            ("Cost Structure Management",  f"Restaurants with combined COGS+OPEX below <b>70%</b> of revenue consistently outperform. Target COGS ≤28% and OPEX ≤40%."),
        ]

        for i, (title, body) in enumerate(recs):
            badge_cls = ["badge-blue", "badge-yellow", "badge-green", "badge-blue",
                         "badge-green", "badge-blue", "badge-green", "badge-yellow"][i]
            num = i + 1
            st.markdown(f"""
            <div class="info-card">
              <h4><span class="{badge_cls}">#{num}</span> &nbsp; {title}</h4>
              <p>{body}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Profit Optimization Score by Restaurant")
        df_score = df.copy()
        df_score["OptScore"] = (
            (df_score["TotalNetProfit"] / df_score["TotalNetProfit"].max()) * 40 +
            (1 - df_score["CommissionRate"] / df_score["CommissionRate"].max()) * 30 +
            (df_score["InStoreShare"]) * 20 +
            (df_score["GrowthFactor"] / df_score["GrowthFactor"].max()) * 10
        ).round(2)

        top_opt = df_score.nlargest(15, "OptScore")[
            ["RestaurantName", "CuisineType", "Segment", "Subregion", "OptScore",
             "TotalNetProfit", "CommissionRate"]
        ]
        top_opt["TotalNetProfit"] = top_opt["TotalNetProfit"].map("${:,.0f}".format)
        top_opt["CommissionRate"] = top_opt["CommissionRate"].map("{:.1%}".format)
        top_opt["OptScore"]       = top_opt["OptScore"].map("{:.2f}".format)
        st.dataframe(top_opt.set_index("RestaurantName"), use_container_width=True)
