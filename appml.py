import streamlit as st
import pandas as pd
import joblib

# Page Configuration 
st.set_page_config(
    page_title="Malaysia Vehicle Registration Forecaster",
    layout="centered"
)

# 1. Load Model & Assets
@st.cache_resource
def load_assets():
    try:
        # Load the pre-trained model and default values
        model_loaded = joblib.load("best_model_gbr.pkl")
        defaults_loaded = joblib.load("defaults.pkl")
        return model_loaded, defaults_loaded
    except FileNotFoundError:
        return None, None

# Initialize the model
model, defaults = load_assets()

# 2. Interface Design 
st.title("Malaysia Vehicle Registration Forecaster")
st.markdown("""
This dashboard predicts daily car registration trends based on fuel prices and economic indicators.
""")
st.info("Adjust the sliders in the sidebar to simulate different economic scenarios.")
st.write("---")

# 3. Main Application Logic 
if model is not None:
    # --- Sidebar Inputs ---
    st.sidebar.header("1. Fuel Prices (RM)")
    
    # Helper to get default value safely
    def get_default(key, fallback):
        if defaults and key in defaults:
            return float(defaults[key])
        return fallback

    ron95 = st.sidebar.slider(
        "RON95 Price", 
        min_value=1.00, max_value=3.00, 
        value=get_default("RON95", 2.05), step=0.01
    )
    
    ron97 = st.sidebar.slider(
        "RON97 Price", 
        min_value=1.50, max_value=5.00, 
        value=get_default("RON97", 3.00), step=0.01
    )
    
    diesel = st.sidebar.slider(
        "Diesel Price", 
        min_value=1.50, max_value=4.00, 
        value=get_default("DIESEL", 2.15), step=0.01
    )

    st.sidebar.header("2. Economic Indicators")
    cpi = st.sidebar.number_input(
        "Consumer Price Index (CPI)", 
        min_value=200.0, max_value=400.0, 
        value=get_default("CPI", 280.0), step=0.1
    )
    
    unemployment = st.sidebar.slider(
        "Unemployment Rate (%)", 
        min_value=2.0, max_value=10.0, 
        value=get_default("UNEMPLOYMENT RATE", 3.5), step=0.1
    )

    # Prediction Logic
    
    # 1. Create Dictionary of Inputs
    input_dict = {
        "RON95": ron95,
        "RON97": ron97,
        "DIESEL": diesel,
        "CPI": cpi,
        "UNEMPLOYMENT RATE": unemployment
    }
    
    # 2. Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # 3. Enforce the exact column order found in the pickle file
    expected_order = ["RON95", "RON97", "DIESEL", "CPI", "UNEMPLOYMENT RATE"]
    input_df = input_df[expected_order]

    # Display Inputs & Output 
    
    # Show summary of inputs
    st.subheader("Simulation Parameters")
    st.dataframe(input_df)

    # Predict Button
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Analyzing market data..."):
            try:
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Display Results
                st.markdown("### Forecast Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Predicted Daily Registrations", 
                        value=f"{int(prediction):,}", 
                        delta="Model Estimate"
                    )
                
                with col2:
                    avg_fuel = (ron95 + ron97 + diesel) / 3
                    st.metric(
                        label="Average Fuel Price",
                        value=f"RM {avg_fuel:.2f}"
                    )
                
                # Optional: Add interpretation based on your domain knowledge
                if prediction > 15000:
                    st.success("High registration volume expected.")
                else:
                    st.warning("Low registration volume expected.")
                    
            except Exception as e:
                st.error(f"Prediction Failed: {e}")
                st.write("Debug Info - Model expects features:", model.feature_names_in_)

else:
    # Fallback if files are missing
    st.error("System Error: Model files not found.")
    st.warning("""
    Please ensure the following files are in the same folder as this script:
    1. `best_model_gbr.pkl` (The trained model)
    2. `defaults.pkl` (The feature averages)
    

    """)

