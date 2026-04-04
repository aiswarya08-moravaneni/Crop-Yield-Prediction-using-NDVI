import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# --- PAGE CONFIG ---
st.set_page_config(page_title="AP Agri-Intel Dashboard", layout="wide", page_icon="🌾")

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
st.sidebar.title("🛠️ Dashboard Controls")
view_mode = st.sidebar.radio("Select Analysis Depth", ["Yearly Overview", "Seasonal Deep-Dive"])

# Assign current working dataframe based on selection
if view_mode == "Yearly Overview":
    curr_df = yearly_df
    st.sidebar.info("Viewing annual totals (13 districts).")
else:
    curr_df = seasonal_df
    st.sidebar.info("Viewing Kharif vs Rabi variations.")

# --- 3. SIDEBAR: FILTERS ---
st.sidebar.markdown("---")
  
# Using unique keys prevents the StreamlitDuplicateElementId error
dist_list = st.sidebar.multiselect(
    "Select Districts", 
    options=sorted(curr_df['District'].unique()), 
    default=sorted(curr_df['District'].unique()),
    key=f"dist_{view_mode}"
)

crop_list = st.sidebar.multiselect(
    "Select Crops", 
    options=sorted(curr_df['Crop'].unique()), 
    default=sorted(curr_df['Crop'].unique()),
    key=f"crop_{view_mode}"
)

# Apply Filtering
mask = (curr_df['District'].isin(dist_list)) & (curr_df['Crop'].isin(crop_list))
f_df = curr_df[mask]

# --- 4. SIDEBAR: DOWNLOAD ---
st.sidebar.markdown("---")
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
st.title(f"🛰️ AP Agricultural Intelligence System")
st.subheader(f"Current View: {view_mode}")

if not f_df.empty:
    st.markdown("### 📝 Automated Intelligence Summary")
    
    # Calculate key insights
    top_crop = f_df.groupby('Crop')['Production'].sum().idxmax()
    best_district = f_df.groupby('District')['Yield'].mean().idxmax()
    ndvi_corr = f_df['NDVI'].corr(f_df['Yield'])

    col_i1, col_i2 = st.columns([2, 1])
    with col_i1:
        insight_text = f"""
        * The most dominant crop in this selection is **{top_crop}**.
        * **{best_district}** is leading in terms of average productivity (Yield).
        * The correlation between Satellite Greenness (NDVI) and Final Yield is **{ndvi_corr:.2f}**. 
        """
        if ndvi_corr > 0.7:
            insight_text += " This indicates a strong positive relationship between vegetation health and crop yield."
        elif ndvi_corr > 0.3:
            insight_text += " This indicates a moderate relationship, suggesting NDVI partially influences yield."
        elif ndvi_corr > 0:
            insight_text += " This indicates a weak relationship, meaning NDVI alone is not sufficient to explain yield variations."
        else:
            insight_text += " This indicates little to no relationship. Yield is likely influenced more by irrigation, soil quality, fertilizers, and farming practices."
        st.info(insight_text)

    # KPI CARDS
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Production", f"{f_df['Production'].sum()/1e6:.2f}M Tons")
    k2.metric("Avg Yield", f"{f_df['Yield'].mean():.2f} tons/Ha")
    k3.metric("Avg NDVI", f"{f_df['NDVI'].mean():.3f}")
    k4.metric("Resource Efficiency", f"{f_df['Efficiency'].mean():.2f}")

    st.divider()

    # --- 6. VISUALIZATIONS ---
    tab_viz, tab_ml = st.tabs(["📊 Trends & Spatial Analysis", "🤖 ML Yield Simulator"])

    with tab_viz:
        c_trend, c_eff = st.columns([2, 1])
        
        with c_trend:
            st.subheader("📈 Productivity Over Time")
            group_cols = ['Year', 'Crop'] if view_mode == "Yearly Overview" else ['Year', 'Crop', 'Season']
            trend_data = f_df.groupby(group_cols)['Production'].sum().reset_index()
            
            if view_mode == "Seasonal Deep-Dive":
                fig_line = px.line(trend_data, x='Year', y='Production', color='Crop', line_dash='Season', markers=True)
            else:
                fig_line = px.line(trend_data, x='Year', y='Production', color='Crop', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

        with c_eff:
            st.subheader("🏆 Efficiency Ranking")
            eff_rank = f_df.groupby('District')['Efficiency'].mean().sort_values(ascending=True).reset_index()
            fig_bar = px.bar(eff_rank, x='Efficiency', y='District', orientation='h', color='Efficiency', color_continuous_scale='Greens')
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        
        c_tree, c_scat = st.columns(2)
        with c_tree:
            st.subheader("🗺️ Production Distribution")
            path = ['District', 'Crop'] if view_mode == "Yearly Overview" else ['District', 'Season', 'Crop']
            fig_tree = px.treemap(f_df, path=path, values='Production', color='Yield', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_tree, use_container_width=True)
            
        with c_scat:
            st.subheader("⚖️ NDVI vs Yield Correlation")
            fig_scatter = px.scatter(f_df, x='NDVI', y='Yield', color='Crop', 
                                     symbol='Season' if view_mode == "Seasonal Deep-Dive" else None,
                                     size='Area', hover_data=['District', 'Year'])
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_ml:
        st.subheader("🧪 Climate Impact & Yield Simulator")
        st.write("Predict yields by adjusting satellite greenness (NDVI) and area.")

        # ML Training logic
        st.markdown("**Model:** Random Forest Regression (Ensemble ML Model)")
        import pickle

        if view_mode == "Seasonal Deep-Dive":
            model = pickle.load(open("models/seasonal_model.pkl", "rb"))
            model_cols = pickle.load(open("models/seasonal_columns.pkl", "rb"))
        else:
            model = pickle.load(open("models/yearly_model.pkl", "rb"))
            model_cols = pickle.load(open("models/yearly_columns.pkl", "rb"))

        # Simulator Controls
        s1, s2, s3, s4 = st.columns(4)
        p_dist = s1.selectbox("Target District", curr_df['District'].unique())
        p_crop = s2.selectbox("Target Crop", curr_df['Crop'].unique())
        p_area = s3.number_input("Area (Hectares)", value=1000)
        p_ndvi = s4.slider("Simulated NDVI (Greenness)", 0.1, 0.7, 0.35)

        p_season = None
        if view_mode == "Seasonal Deep-Dive":
            p_season = st.selectbox("Target Season", curr_df['Season'].unique())

        if st.button("Generate Yield Forecast"):
            input_df = pd.DataFrame(columns=model_cols)
            input_df.loc[0] = 0
            input_df['Area'] = p_area
            input_df['NDVI'] = p_ndvi
            if f'District_{p_dist}' in model_cols: input_df[f'District_{p_dist}'] = 1
            if f'Crop_{p_crop}' in model_cols: input_df[f'Crop_{p_crop}'] = 1
            if p_season and f'Season_{p_season}' in model_cols: input_df[f'Season_{p_season}'] = 1
            
            pred = model.predict(input_df)[0]
            st.success(f"### Predicted Yield: {pred:.2f} kg/Hectare")

        st.divider()
        st.subheader("📉 Model Reliability (Backtesting)")
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
            title=f"Actual vs Predicted Yield (R² = {r2:.3f})"
        )
        st.plotly_chart(fig_acc, use_container_width=True)

else:
    st.warning("No data found. Please check your sidebar filters.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Methodology")
st.sidebar.write("- Sentinel-2 Satellite NDVI")
st.sidebar.write("- 13-District Harmonized Data")
st.sidebar.write("- Random Forest Regression")