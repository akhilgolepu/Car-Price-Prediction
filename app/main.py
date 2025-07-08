import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

@st.cache_data
def clean_data():
    """Load and clean the car dataset, returning a DataFrame ready for use."""
    try:
        data = pd.read_csv(Path("data/cardekho_dataset.csv"))
        data = data.drop(columns=['Unnamed: 0', 'car_name', 'brand', 'model'])
        data = pd.get_dummies(data, columns=['fuel_type', 'seller_type', 'transmission_type'], drop_first=True)
        data['power_per_cc'] = data['max_power'] / data['engine']
        data['diesel_auto'] = (data['fuel_type_Diesel'] == 1) & (data['transmission_type_Manual'] == 0)
        data['diesel_auto'] = data['diesel_auto'].astype(int)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def show_radar_chart(input_data):
    """Display a radar chart based on user input values."""
    categories = ['Vehicle Age', 'KM Driven', 'Mileage', 'Engine', 'Max Power', 'Seats']
    values = [
        input_data['vehicle_age'],
        input_data['km_driven'],
        input_data['mileage'],
        input_data['engine'],
        input_data['max_power'],
        input_data['seats']
    ]
    max_vals = [30, 500000, 50, 5000, 500, 10]
    norm_values = [v / m if m else 0 for v, m in zip(values, max_vals)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_values,
        theta=categories,
        fill='toself',
        name='Your Car'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False
    )
    return fig

def add_sidebar():
    """Add sidebar widgets for user input and return the input as a dictionary."""
    st.sidebar.write("Select the features of your car to predict its price.")
    data = clean_data()
    vehicle_age = st.sidebar.slider("Vehicle Age (in years)", 0, 30, 5, help="How old is your car?")
    km_driven = st.sidebar.slider("Kilometers Driven", 0, 500000, 50000, step=1000, help="Total distance driven by the car.")
    mileage = st.sidebar.number_input("Mileage (km/l)", min_value=0.0, max_value=50.0, value=18.0, help="Fuel efficiency of the car.")
    engine = st.sidebar.number_input("Engine (cc)", min_value=500, max_value=5000, value=1200, help="Engine displacement in cubic centimeters.")
    max_power = st.sidebar.number_input("Max Power (bhp)", min_value=20.0, max_value=500.0, value=80.0, help="Maximum power output of the car.")
    seats = st.sidebar.slider("Number of Seats", 2, 10, 5, help="How many seats does the car have?")
    fuel_type = st.sidebar.selectbox("Fuel Type", ["Diesel", "Petrol", "Electric", "LPG"], help="Type of fuel used by the car.")
    seller_type = st.sidebar.selectbox("Seller Type", ["Individual", "Trustmark Dealer"], help="Who is selling the car?")
    transmission = st.sidebar.selectbox("Transmission Type", ["Manual", "Automatic"], help="Type of transmission.")
    return {
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "fuel_type": fuel_type,
        "seller_type": seller_type,
        "transmission": transmission
    }

def add_predictions(input_data, feature_names):
    """Load the model and predict the car price based on user input. Returns mean, lower, upper, model."""
    try:
        with open(Path('model/model.pkl'), 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return 0, 0, 0, None
    fuel_types = ['Diesel', 'Petrol', 'Electric', 'LPG']
    seller_types = ['Individual', 'Trustmark Dealer']
    transmissions = ['Manual', 'Automatic']
    fuel_onehot = [1 if input_data['fuel_type'] == ft else 0 for ft in fuel_types]
    seller_onehot = [1 if input_data['seller_type'] == stype else 0 for stype in seller_types]
    transmission_onehot = [1 if input_data['transmission'] == t else 0 for t in transmissions]
    power_per_cc = input_data['max_power'] / input_data['engine'] if input_data['engine'] else 0
    diesel_auto = 1 if input_data['fuel_type'] == 'Diesel' and input_data['transmission'] == 'Automatic' else 0

    input_array = np.array([
        input_data['vehicle_age'],
        input_data['km_driven'],
        input_data['mileage'],
        input_data['engine'],
        input_data['max_power'],
        input_data['seats'],
        fuel_onehot[0],  
        fuel_onehot[2],  
        fuel_onehot[3],  
        fuel_onehot[1],  
        seller_onehot[0],
        seller_onehot[1],
        transmission_onehot[0],  
        power_per_cc,
        diesel_auto
    ]).reshape(1, -1)
    input_df = pd.DataFrame(input_array, columns=pd.Index(feature_names))
    try:
        if hasattr(model, 'estimators_'):
            preds = np.array([est.predict(input_df)[0] for est in model.estimators_])
            mean_pred = preds.mean()
            std_pred = preds.std()
            lower = mean_pred - std_pred
            upper = mean_pred + std_pred
            return mean_pred, lower, upper, model
        else:
            prediction = model.predict(input_df)[0]
            return prediction, prediction, prediction, model
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0, 0, 0, model

def show_feature_importance(model, feature_names):
    """Display a bar chart of feature importances."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values('Importance', ascending=True)
        st.subheader('Feature Importance')
        st.bar_chart(imp_df.set_index('Feature'))
    else:
        st.info('Feature importance not available for this model.')

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Car Price Prediction",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    try:
        with open(Path("assets/style.css")) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load custom CSS: {e}")
    input_data = add_sidebar()
    feature_names = [
        'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
        'fuel_type_Diesel', 'fuel_type_Electric', 'fuel_type_LPG', 'fuel_type_Petrol',
        'seller_type_Individual', 'seller_type_Trustmark Dealer',
        'transmission_type_Manual', 'power_per_cc', 'diesel_auto'
    ]
    with st.container():
        st.markdown("""
            <h1 style='font-size:3rem; font-weight:700; margin-bottom:0;'>Car Price Prediction</h1>
            <h3 style='color:#555; margin-top:0;'>Get an instant, data-driven estimate for your used car 🚗</h3>
        """, unsafe_allow_html=True)
        st.write("""
        Welcome to the Car Price Prediction App! This tool uses a machine learning model trained on real-world car listings data to predict the estimated selling price of a used car based on key features like age, mileage, engine capacity, fuel type, transmission, and more.
        
        **Model Performance (on test set):**
        - **R² Score:** 0.87
        - **MAE:** ₹52,000
        - **RMSE:** ₹1,10,000
        """)
        with st.expander("How does this work?", expanded=False):
            st.markdown("""
            - Enter your car's details in the sidebar.
            - The app uses a machine learning model trained on real car sales data.
            - You'll see a radar chart of your car's specs and an estimated price.
            """)

        st.markdown("""
        <div class='summary-card'>
        <b>Vehicle Age:</b> {} years<br>
        <b>Kilometers Driven:</b> {} km<br>
        <b>Mileage:</b> {} km/l<br>
        <b>Engine:</b> {} cc<br>
        <b>Max Power:</b> {} bhp<br>
        <b>Seats:</b> {}<br>
        <b>Fuel Type:</b> {}<br>
        <b>Seller Type:</b> {}<br>
        <b>Transmission:</b> {}
        </div>
        """.format(
            input_data['vehicle_age'],
            input_data['km_driven'],
            input_data['mileage'],
            input_data['engine'],
            input_data['max_power'],
            input_data['seats'],
            input_data['fuel_type'],
            input_data['seller_type'],
            input_data['transmission']
        ), unsafe_allow_html=True)
    col1, col2 = st.columns([4,1])
    with col1:
        radar = show_radar_chart(input_data)
        st.plotly_chart(radar)
    with col2:
        with st.spinner('Predicting price...'):
            mean_pred, lower, upper, model = add_predictions(input_data, feature_names)
        if mean_pred > 0:
            st.markdown(f"""
            <div class='green-metric-container'>
                <div class='green-metric-value'>₹{mean_pred:,.0f}</div>
                <div class='green-metric-success' style='margin-top:8px;'>Estimated Price Range: ₹{lower:,.0f} – ₹{upper:,.0f}</div>
                <div class='green-metric-success'>Prediction complete! This is your car's estimated selling price.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Prediction could not be made. Please check your input.")
    show_feature_importance(model, feature_names)

if __name__ == "__main__":
    main()