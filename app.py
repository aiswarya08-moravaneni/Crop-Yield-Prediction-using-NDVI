import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="AP Agri-Intel Dashboard", layout="wide", page_icon="🌾")

def _inject_theme(is_dark: bool) -> None:
    # Pure CSS theme tokens + components (cards, hero, badges).
    # Streamlit's internal DOM changes sometimes, so keep selectors minimal.
    if is_dark:
        bg = "#0B1220"
        # Slightly stronger surfaces for better contrast on dark backgrounds
        panel = "rgba(255,255,255,0.08)"
        panel2 = "rgba(255,255,255,0.10)"
        text = "rgba(255,255,255,0.92)"
        muted = "rgba(255,255,255,0.68)"
        border = "rgba(255,255,255,0.14)"
        accent = "#6EE7B7"
        accent2 = "#60A5FA"
    else:
        bg = "#F7F8FC"
        panel = "rgba(255,255,255,0.82)"
        panel2 = "rgba(255,255,255,0.92)"
        text = "rgba(15, 23, 42, 0.92)"
        muted = "rgba(15, 23, 42, 0.60)"
        border = "rgba(2, 6, 23, 0.10)"
        accent = "#16A34A"
        accent2 = "#2563EB"

    css_template = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    css = (
        css_template
        .replace("__BG__", bg)
        .replace("__PANEL__", panel)
        .replace("__PANEL2__", panel2)
        .replace("__TEXT__", text)
        .replace("__MUTED__", muted)
        .replace("__BORDER__", border)
        .replace("__ACCENT__", accent)
        .replace("__ACCENT2__", accent2)
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _kpi_card(title: str, value: str, sub: str) -> str:
    return f"""<div class="card"><div class="card-title">{title}</div><div class="card-value">{value}</div><div class="card-sub">{sub}</div></div>"""


if "ui_dark" not in st.session_state:
    st.session_state.ui_dark = False

# Plotly template is set dynamically after theme toggle (dark/light).

# --- 1. DATA LOADING ---
@st.cache_data
def load_all_data():
    # Load both datasets
    yearly = pd.read_csv('data/final_total_dataset.csv')
    seasonal = pd.read_csv('data/final_dataset.csv')
    
    # Calculate Efficiency (Yield per NDVI unit)
    yearly['Efficiency'] = yearly['Yield'] / yearly['NDVI']
    seasonal['Efficiency'] = seasonal['Yield'] / seasonal['NDVI']
    
    return yearly, seasonal

try:
    yearly_df, seasonal_df = load_all_data()
except Exception as e:
    st.error(f"Error loading CSV files: {e}. Please ensure 'final_total_dataset.csv' and 'final_dataset.csv' are in the project folder.")
    st.stop()

# --- 2. SIDEBAR: MODE SELECTOR ---
with st.sidebar:
    st.markdown("## 🌾 AP Agri‑Intel")
    st.caption("Satellite-driven crop insights for Andhra Pradesh")
    st.divider()

    st.session_state.ui_dark = st.toggle("Dark mode", value=bool(st.session_state.ui_dark))
    _inject_theme(bool(st.session_state.ui_dark))
    px.defaults.template = "plotly_dark" if st.session_state.ui_dark else "plotly_white"
    view_mode = st.radio("Analysis Mode", ["Yearly Overview", "Seasonal Deep-Dive"])

# --- TOPBAR NAV (website-style) ---
st.markdown(
    """
    <div class="topnav">
      <div class="topnav-left">
        <div class="brand">AP Agri‑Intel</div>
        <div class="brand-sub">Satellite-driven insights • Production • Yield • NDVI</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

NAV_OPTIONS = ["Dashboard", "Explorer", "Simulator", "Methodology"]
if "top_nav" not in st.session_state:
    st.session_state.top_nav = "Dashboard"

nav_cols = st.columns(4)
for i, opt in enumerate(NAV_OPTIONS):
    with nav_cols[i]:
        if st.button(
            opt,
            key=f"topnav_{opt}",
            use_container_width=True,
            type="primary" if st.session_state.top_nav == opt else "secondary",
        ):
            st.session_state.top_nav = opt

nav = st.session_state.top_nav

# Assign current working dataframe based on selection
if view_mode == "Yearly Overview":
    curr_df = yearly_df
    st.sidebar.info("Annual totals (13 districts).")
else:
    curr_df = seasonal_df
    st.sidebar.info("Kharif vs Rabi variations.")

# --- 3. SIDEBAR: FILTERS ---
st.sidebar.markdown("### Filters")
  
# Using unique keys prevents the StreamlitDuplicateElementId error
dist_list = st.sidebar.multiselect(
    "Select Districts", 
    options=sorted(curr_df['District'].unique()), 
    default=sorted(curr_df['District'].unique())[:6],
    key=f"dist_{view_mode}"
)

crop_list = st.sidebar.multiselect(
    "Select Crops", 
    options=sorted(curr_df['Crop'].unique()), 
    default=sorted(curr_df['Crop'].unique())[:5],
    key=f"crop_{view_mode}"
)

# Apply Filtering
mask = (curr_df['District'].isin(dist_list)) & (curr_df['Crop'].isin(crop_list))
f_df = curr_df[mask]

# --- 4. SIDEBAR: DOWNLOAD ---
st.sidebar.divider()
if not f_df.empty:
    csv_bytes = f_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Filtered Data",
        data=csv_bytes,
        file_name=f'ap_crop_data_{view_mode.lower().replace(" ", "_")}.csv',
        mime='text/csv',
    )
else:
    st.sidebar.warning("No data found for the current selection.")

# --- 5. MAIN HEADER & INSIGHTS ---
st.markdown(
    f"""
    <div class="hero">
      <div class="badge"><span class="dot"></span> Live analytics • {view_mode}</div>
      <h1>AP Agricultural Intelligence</h1>
      <p>District‑wise crop production, yield, and NDVI insights — with a yield simulator powered by Random Forest.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

if f_df.empty:
    st.warning("No data found. Adjust the sidebar filters.")
    st.stop()

if nav == "Dashboard":
    st.markdown("### Snapshot")

    top_crop = f_df.groupby('Crop')['Production'].sum().idxmax()
    best_district = f_df.groupby('District')['Yield'].mean().idxmax()
    ndvi_corr = f_df['NDVI'].corr(f_df['Yield'])

    total_prod_m = f_df["Production"].sum() / 1e6
    avg_yield = f_df["Yield"].mean()
    avg_ndvi = f_df["NDVI"].mean()
    avg_eff = f_df["Efficiency"].replace([np.inf, -np.inf], np.nan).mean()

    st.markdown(
        "<div class='kpi-grid'>"
        + _kpi_card("Total production", f"{total_prod_m:.2f}M", "tons (sum in filter)")
        + _kpi_card("Average yield", f"{avg_yield:.2f}", "tons / hectare")
        + _kpi_card("Average NDVI", f"{avg_ndvi:.3f}", "satellite greenness")
        + _kpi_card("Efficiency", f"{avg_eff:.2f}", "yield / NDVI")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    left, right = st.columns([2.2, 1])
    with left:
        st.markdown("### Insight summary")
        st.markdown(
            f"""
            - Dominant crop: **{top_crop}**  
            - Best district (avg yield): **{best_district}**  
            - NDVI ↔ Yield correlation: **{ndvi_corr:.2f}**
            """
        )
    with right:
        with st.container(border=True):
            st.markdown("#### NDVI signal strength")
            st.progress(float(abs(ndvi_corr)) if pd.notna(ndvi_corr) else 0.0)

elif nav == "Explorer":
    st.markdown("### Explorer")
    st.caption("Trends, efficiency, and spatial distribution for the current filters.")

    st.subheader("📈 Production trend")
    group_cols = ['Year', 'Crop'] if view_mode == "Yearly Overview" else ['Year', 'Crop', 'Season']
    trend_data = f_df.groupby(group_cols)['Production'].sum().reset_index()

    if view_mode == "Seasonal Deep-Dive":
        fig_line = px.line(
            trend_data,
            x='Year',
            y='Production',
            color='Crop',
            line_dash='Season',
            markers=True,
            title="Production over time (tons)",
        )
    else:
        fig_line = px.line(
            trend_data,
            x='Year',
            y='Production',
            color='Crop',
            markers=True,
            title="Production over time (tons)",
        )
    fig_line.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    fig_line.update_yaxes(tickformat=",")
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("🏆 Efficiency by district")
    eff_rank = (
        f_df.groupby('District')['Efficiency']
        .mean()
        .replace([np.inf, -np.inf], np.nan)
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_bar = px.bar(
        eff_rank,
        x='Efficiency',
        y='District',
        orientation='h',
        color='Efficiency',
        color_continuous_scale='Greens',
        title="Higher is better",
    )
    fig_bar.update_layout(margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("🗺️ Production distribution")
    path = ['District', 'Crop'] if view_mode == "Yearly Overview" else ['District', 'Season', 'Crop']
    fig_tree = px.treemap(
        f_df,
        path=path,
        values='Production',
        color='Yield',
        color_continuous_scale='RdYlGn',
        title="Size: production (tons) • Color: yield (tons/Ha)",
    )
    fig_tree.update_layout(margin=dict(l=10, r=10, t=50, b=10), coloraxis_colorbar_title="Yield")
    st.plotly_chart(fig_tree, use_container_width=True)

    st.subheader("⚖️ NDVI vs Yield")
    fig_scatter = px.scatter(
        f_df,
        x='NDVI',
        y='Yield',
        color='Crop',
        symbol='Season' if view_mode == "Seasonal Deep-Dive" else None,
        size='Area',
        hover_data=['District', 'Year', 'Area', 'Production'],
        title="NDVI vs Yield (tons/Ha)",
    )
    fig_scatter.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    fig_scatter.update_xaxes(range=[0, 1])
    st.plotly_chart(fig_scatter, use_container_width=True)

elif nav == "Simulator":
    st.markdown("### Yield simulator")
    st.caption("Adjust NDVI and area to forecast yield and total production.")

    import pickle

    if view_mode == "Seasonal Deep-Dive":
        model = pickle.load(open("models/seasonal_model.pkl", "rb"))
        model_cols = pickle.load(open("models/seasonal_columns.pkl", "rb"))
    else:
        model = pickle.load(open("models/yearly_model.pkl", "rb"))
        model_cols = pickle.load(open("models/yearly_columns.pkl", "rb"))

    with st.container(border=True):
        s1, s2, s3, s4 = st.columns(4)
        p_dist = s1.selectbox("District", curr_df['District'].unique())
        p_crop = s2.selectbox("Crop", curr_df['Crop'].unique())
        p_area = s3.number_input("Area (Ha)", value=1000.0, min_value=0.0, step=10.0)
        p_ndvi = s4.slider("Simulated NDVI", 0.0, 1.0, 0.35)

        p_season = None
        if view_mode == "Seasonal Deep-Dive":
            p_season = st.selectbox("Season", curr_df['Season'].unique())

        run = st.button("Generate forecast")

    if run:
        input_df = pd.DataFrame(columns=model_cols)
        input_df.loc[0] = 0
        input_df['Area'] = p_area
        input_df['NDVI'] = p_ndvi
        if f'District_{p_dist}' in model_cols:
            input_df[f'District_{p_dist}'] = 1
        if f'Crop_{p_crop}' in model_cols:
            input_df[f'Crop_{p_crop}'] = 1
        if p_season and f'Season_{p_season}' in model_cols:
            input_df[f'Season_{p_season}'] = 1

        pred = float(model.predict(input_df)[0])
        pred_prod = pred * float(p_area)

        st.markdown(
            "<div class='kpi-grid'>"
            + _kpi_card("Predicted yield", f"{pred:.2f}", "tons / hectare")
            + _kpi_card("Predicted production", f"{pred_prod:,.0f}", f"tons for {float(p_area):,.0f} Ha")
            + _kpi_card("District", str(p_dist), "selected")
            + _kpi_card("Crop", str(p_crop), "selected")
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    with st.expander("Model reliability (backtesting)", expanded=False):
        if view_mode == "Seasonal Deep-Dive":
            test_df = pd.read_csv("results/seasonal_test_results.csv")
            r2 = pickle.load(open("models/seasonal_r2.pkl", "rb"))
        else:
            test_df = pd.read_csv("results/yearly_test_results.csv")
            r2 = pickle.load(open("models/yearly_r2.pkl", "rb"))

        fig_acc = px.scatter(
            test_df,
            x='Actual',
            y='Predicted',
            trendline="ols",
            title=f"Actual vs Predicted Yield (R² = {r2:.3f})",
        )
        fig_acc.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_acc, use_container_width=True)

elif nav == "Methodology":
    st.markdown("### Methodology")
    st.markdown(
        """
        **Data sources**
        - Sentinel‑2 NDVI (satellite vegetation index)
        - Harmonized district‑level agriculture statistics (Area, Production)

        **Derived metrics**
        - Yield = Production / Area (tons/Ha)
        - Efficiency = Yield / NDVI

        **Model**
        - Random Forest Regressor trained on one‑hot encoded District/Crop (and Season in seasonal mode), with NDVI + Area as numeric features.
        """
    )

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Methodology")
st.sidebar.write("- Sentinel-2 Satellite NDVI")
st.sidebar.write("- 13-District Harmonized Data")
st.sidebar.write("- Random Forest Regression")
