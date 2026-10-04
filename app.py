import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Used Vehicle Price Predictor",
    page_icon="🚗"
)

st.title("Used Vehicle Price Predictor")

st.write(
    "Enter the vehicle information below to get an estimated price."
)

# Load the dataset and trained model
df = pd.read_csv("used_cars.csv")
model = joblib.load("vehicle_price_model.pkl")

st.subheader("Vehicle Information")

# Select a brand
brand = st.selectbox(
    "Brand",
    sorted(df["brand"].dropna().unique())
)

brand_data = df[df["brand"] == brand]

# Only show models for the selected brand
vehicle_model = st.selectbox(
    "Model",
    sorted(brand_data["model"].dropna().unique())
)

model_data = brand_data[
    brand_data["model"] == vehicle_model
]

# Only show model years found for the selected vehicle
available_years = sorted(
    model_data["model_year"].dropna().unique(),
    reverse=True
)

model_year = st.selectbox(
    "Model Year",
    available_years
)

year_data = model_data[
    model_data["model_year"] == model_year
]

# Mileage stays editable because it varies from vehicle to vehicle
mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=50000,
    step=1000
)

# Only show fuel types found for the selected vehicle
fuel_options = sorted(
    year_data["fuel_type"].dropna().unique()
)

if len(fuel_options) == 0:
    fuel_options = sorted(
        model_data["fuel_type"].dropna().unique()
    )

fuel_type = st.selectbox(
    "Fuel Type",
    fuel_options
)

fuel_data = year_data[
    year_data["fuel_type"] == fuel_type
]

if fuel_data.empty:
    fuel_data = year_data

# Only show engines that match the selected vehicle
engine_options = sorted(
    fuel_data["engine"].dropna().unique()
)

if len(engine_options) == 0:
    engine_options = sorted(
        model_data["engine"].dropna().unique()
    )

engine = st.selectbox(
    "Engine",
    engine_options
)

engine_data = fuel_data[
    fuel_data["engine"] == engine
]

if engine_data.empty:
    engine_data = fuel_data

# Only show transmissions that match the selected vehicle
transmission_options = sorted(
    engine_data["transmission"].dropna().unique()
)

if len(transmission_options) == 0:
    transmission_options = sorted(
        model_data["transmission"].dropna().unique()
    )

transmission = st.selectbox(
    "Transmission",
    transmission_options
)

configuration_data = engine_data[
    engine_data["transmission"] == transmission
]

if configuration_data.empty:
    configuration_data = engine_data

# Show exterior colors found with this vehicle configuration
exterior_options = sorted(
    configuration_data["ext_col"].dropna().unique()
)

if len(exterior_options) == 0:
    exterior_options = sorted(
        model_data["ext_col"].dropna().unique()
    )

exterior_color = st.selectbox(
    "Exterior Color",
    exterior_options
)

# Show interior colors found with this vehicle configuration
interior_options = sorted(
    configuration_data["int_col"].dropna().unique()
)

if len(interior_options) == 0:
    interior_options = sorted(
        model_data["int_col"].dropna().unique()
    )

interior_color = st.selectbox(
    "Interior Color",
    interior_options
)

# Show accident history options found for this vehicle
accident_options = sorted(
    model_data["accident"].dropna().unique()
)

accident = st.selectbox(
    "Accident History",
    accident_options
)

# Show clean title options found for this vehicle
title_options = sorted(
    model_data["clean_title"].dropna().unique()
)

clean_title = st.selectbox(
    "Clean Title",
    title_options
)

if st.button("Predict Price"):

    # Put the vehicle information into the same format used by the model
    vehicle = pd.DataFrame({
        "model_year": [model_year],
        "milage": [mileage],
        "brand": [brand],
        "model": [vehicle_model],
        "fuel_type": [fuel_type],
        "engine": [engine],
        "transmission": [transmission],
        "ext_col": [exterior_color],
        "int_col": [interior_color],
        "accident": [accident],
        "clean_title": [clean_title]
    })

    predicted_price = model.predict(vehicle)[0]

    st.success(
        f"Estimated Vehicle Price: ${predicted_price:,.2f}"
    )