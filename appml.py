import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np


# Page Configuration

st.set_page_config(
    page_title="Malaysia's Vehicle Registration Analytics",
    layout="wide"
)


# Model & File Path Configuration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_gbr.pkl")
DEFAULT_PATH = os.path.join(BASE_DIR, "defaults.pkl")
IMAGE_PATH = os.path.join(BASE_DIR, "traffic-volume-study.jpg") 

# Utility: Load Model / Defaults

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        defaults = joblib.load(DEFAULT_PATH)
        return model, defaults
    except FileNotFoundError:
        return None, None

model, defaults = load_assets()


# Sidebar Navigation

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    ["🏠 Home Page", "📊 Analytics Dashboard", "🔮 Prediction & Trends", "🔔 Insights & Feedback"]
)

st.sidebar.info("Model: Gradient Boosting Regressor (GBR)")


# HOME PAGE

if selected_page == "🏠 Home Page":
    st.title("Malaysia Vehicle Registration Analytics")

    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, use_container_width=True)
    
    st.markdown("""
    ### Welcome to the Market Foresight Platform.
    
    This system leverages machine learning to analyze how **fuel price fluctuations** and **macroeconomic indicators** impact the automotive market in Malaysia.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Variables Tracked", "5", "CPI, Fuel, Unemployment")
    col2.metric("Region", "Malaysia", "National Data")
    col3.metric("Model Status", "Active", "Ready for Forecast")


# ANALYTICS DASHBOARD 

elif selected_page == "📊 Analytics Dashboard":
    st.header("Dashboard")
    st.markdown("Detailed historical analysis and PowerBI integration.")
    
    # Placeholder for PowerBI
    st.info("**.")
    

    st.markdown("""
    <iframe width="100%" height="600" 
    src="https://app.powerbi.com/view?r=eyJrIjoiYTUzZmQwMWUtNDRiYi00N2QxLThhNDItNmY1OTRjMTNiNGQ5IiwidCI6IjdmMDQ4ZmMxLTJlYTMtNDhlNC1hYzkyLTkxZDFlYjA5ODA3YyIsImMiOjEwfQ%3D%3D" 
    frameborder="0" allowFullScreen="true"></iframe>
    """, unsafe_allow_html=True)
    

# PREDICTION & TRENDS PAGE 

elif selected_page == "🔮 Prediction & Trends":
    st.header("Forecast & Trend Simulation")
    st.markdown("Calculate specific volume predictions and analyze market sensitivity.")

    if model:
        # PART 1: CALCULATOR
        with st.container(border=True):
            st.subheader("1. Forecast Calculator")
            c1, c2, c3 = st.columns(3)
            ron95 = c1.number_input("RON95", 1.00, 5.00, 2.05, 0.01)
            ron97 = c2.number_input("RON97", 1.50, 7.00, 3.47, 0.01)
            diesel = c3.number_input("Diesel", 1.50, 6.00, 2.15, 0.01)

            c4, c5 = st.columns(2)
            cpi = c4.number_input("CPI (Index)", 200.0, 400.0, 280.0, 0.1)
            unemployment = c5.slider("Unemployment Rate (%)", 2.0, 10.0, 3.5, 0.1)

            # Logic
            input_dict = {
                "CPI": cpi, "RON95": ron95, "RON97": ron97, 
                "DIESEL": diesel, "UNEMPLOYMENT RATE": unemployment
            }
            expected_order = ['RON95', 'RON97', 'DIESEL', 'CPI', 'UNEMPLOYMENT RATE']
            input_df = pd.DataFrame([input_dict])[expected_order]

            if st.button("Calculate Forecast", type="primary", use_container_width=True):
                prediction = model.predict(input_df)[0]
                
                st.markdown("### Result")
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Predicted Daily Registrations", f"{int(prediction):,}")
                res_col2.metric("Avg Fuel Price", f"RM {(ron95+ron97+diesel)/3:.2f}")

        # PART 2: TREND GRAPH 
        st.write("---")
        st.subheader("2. Market Sensitivity Graph")
        st.caption("See how changing ONE variable affects the prediction above (while others stay constant).")

        # Selection for simulation
        target_var = st.selectbox("Select Variable to Simulate:", expected_order)

        if target_var:
            # Ranges
            ranges = {
                'CPI': (200.0, 400.0),
                'RON95': (1.00, 5.00),
                'RON97': (1.50, 7.00),
                'DIESEL': (1.50, 6.00),
                'UNEMPLOYMENT RATE': (2.0, 12.0)
            }

            # Generate Data
            min_val, max_val = ranges[target_var]
            steps = np.linspace(min_val, max_val, 50)
            
            # Use current inputs from the calculator above as the baseline
            sim_data = pd.DataFrame([input_dict] * 50) 
            sim_data = sim_data[expected_order]       
            sim_data[target_var] = steps              
            
            # Predict
            sim_predictions = model.predict(sim_data)
            
            # Plot
            chart_data = pd.DataFrame({
                target_var: steps,
                'Predicted Registrations': sim_predictions
            }).set_index(target_var)

            st.line_chart(chart_data, color="#FF4B4B")
            st.info(f"Visualizing how **{target_var}** impacts demand based on your current inputs.")

    else:
        st.error("⚠️ Model files (`best_model_gbr.pkl`, `defaults.pkl`) are missing.")


# INSIGHTS & FEEDBACK

elif selected_page == "🔔 Insights & Feedback":
    st.header("Strategic Notes & Feedback")

    st.warning("""
    **Market Watchlist:**
    * Monitor upcoming government subsidy rationalization plans.
    * Unemployment rate shifts above 4.0% typically dampen registration volume significantly.
    """)

    with st.form("feedback_form"):
        st.write("**Report an Issue or Suggestion**")
        category = st.selectbox("Category", ["Model Accuracy", "UI Bug", "Feature Request"])
        text = st.text_area("Details")
        if st.form_submit_button("Submit Report"):
            st.success("Thank you! Your feedback has been logged.")