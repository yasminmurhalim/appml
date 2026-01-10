import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration 
st.set_page_config(
    page_title="MY Vehicle Registration Forecaster",
    layout="centered"
)

# 1. Load Model & Assets 
@st.cache_resource
def load_assets():
    try:
        model_loaded = joblib.load("best_model_gbr.pkl")
        defaults_loaded = joblib.load("defaults.pkl")
        return model_loaded, defaults_loaded
    except FileNotFoundError:
        return None, None

model, defaults = load_assets()

# 2. Interface Design 
st.title("Malaysia Vehicle Registration Forecaster")
st.markdown("""
This dashboard predicts daily car registration trends based on fuel prices and economic indicators.
""")
st.write("---")

# 3. Main Application Logic 
if model is not None:
    # Sidebar Inputs
    st.sidebar.header("1. Fuel Prices (RM)")
    
    def get_default(key, fallback):
        if defaults and key in defaults:
            return float(defaults[key])
        return fallback

    ron95 = st.sidebar.slider("RON95 Price", 1.00, 4.00, get_default("RON95", 2.05), 0.01)
    ron97 = st.sidebar.slider("RON97 Price", 1.50, 6.00, get_default("RON97", 3.00), 0.01)
    diesel = st.sidebar.slider("Diesel Price", 1.50, 5.00, get_default("DIESEL", 2.15), 0.01)

    st.sidebar.header("2. Economic Indicators")
    cpi = st.sidebar.number_input("Consumer Price Index (CPI)", 200.0, 400.0, get_default("CPI", 280.0), 0.1)
    unemployment = st.sidebar.slider("Unemployment Rate (%)", 2.0, 10.0, get_default("UNEMPLOYMENT RATE", 3.5), 0.1)

    # Prediction Data Prep 
    input_dict = {
        "CPI": cpi,
        "RON95": ron95,
        "RON97": ron97,
        "DIESEL": diesel,
        "UNEMPLOYMENT RATE": unemployment
    }
    
    # [CRITICAL FIX] The exact order from Notebook Cell 25
    expected_order = ['RON95', 'RON97', 'DIESEL', 'CPI', 'UNEMPLOYMENT RATE']
    
    # Reorder columns to match training data
    input_df = pd.DataFrame([input_dict])[expected_order]

    # Main Dashboard
    
    # 1. Instant Prediction
    st.subheader("Current Forecast")
    if st.button("Generate Prediction", type="primary"):
        prediction = model.predict(input_df)[0]
        c1, c2 = st.columns(2)
        c1.metric("Predicted Registrations", f"{int(prediction):,}")
        c2.metric("Avg Fuel Price", f"RM {(ron95+ron97+diesel)/3:.2f}")

    # 2. Visualization Section
    st.write("---")
    st.subheader("Trend Analysis")
    st.caption("Select a factor to see how it affects vehicle registrations if all other values stay the same.")

    target_var = st.selectbox("Choose Variable to Simulate:", expected_order)

    if target_var:
        # Define ranges for simulation
        ranges = {
            'CPI': (200.0, 400.0),
            'RON95': (1.00, 5.00),
            'RON97': (1.50, 7.00),
            'DIESEL': (1.50, 6.00),
            'UNEMPLOYMENT RATE': (2.0, 12.0)
        }
        
        min_val, max_val = ranges[target_var]
        steps = np.linspace(min_val, max_val, 50)
        
        # Create simulation dataframe
        sim_data = pd.DataFrame([input_dict] * 50) 
        sim_data = sim_data[expected_order]        
        sim_data[target_var] = steps             
        
        # Predict
        sim_predictions = model.predict(sim_data)
        
        # Chart
        chart_data = pd.DataFrame({
            target_var: steps,
            'Predicted Registrations': sim_predictions
        }).set_index(target_var)

        st.line_chart(chart_data)
        
        st.info(f"Observation: The graph shows how **{target_var}** impacts registrations while holding other inputs constant.")

else:
    st.error("Model files not found.")
    st.warning("Please ensure 'best_model_gbr.pkl' and 'defaults.pkl' are in the same folder as this script.")