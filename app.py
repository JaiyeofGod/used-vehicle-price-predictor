import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Used Vehicle Price Predictor",
    page_icon=":material/directions_car:",
    layout="centered",
)

st.html(
    """
    <style>
    .stApp {
        background: linear-gradient(#f7f5f0 0%, #ece7dc 100%) fixed;
    }
    .st-key-valuation_card {
        background: #ffffff;
        border: 1px solid rgba(195, 198, 215, 0.7);
        border-radius: 12px;
        box-shadow: 0 16px 40px rgba(25, 28, 30, 0.08);
        padding: 1.75rem 1.75rem 1.5rem;
    }
    .st-key-predict_price button {
        width: 100%;
        min-height: 3rem;
    }
    .st-key-result_panel {
        background: rgba(218, 226, 253, 0.45);
        border: 1px solid rgba(195, 198, 215, 0.55);
        border-radius: 12px;
        padding: 0.25rem 0.5rem 0.75rem;
    }
    .st-key-result_panel [data-testid="stMetric"] {
        background: transparent;
    }
    .st-key-result_panel [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
    }
    .st-key-result_panel [data-testid="stMetricValue"] {
        justify-content: center;
        width: 100%;
    }
    </style>
    """
)


@st.cache_data
def load_vehicles():
    return pd.read_csv("used_cars.csv")


@st.cache_resource
def load_model():
    return joblib.load("vehicle_price_model.pkl")


def choice(label, values, key, fallback):
    options = sorted(pd.Series(values).dropna().unique(), key=str)
    if len(options) == 0:
        options = sorted(pd.Series(fallback).dropna().unique(), key=str)
    if len(options) == 0:
        options = ["Not available"]
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return st.selectbox(label, options, key=key)


df = load_vehicles()
model = load_model()

with st.container(key="valuation_card"):
    with st.container(horizontal_alignment="center"):
        st.title(
            "Used Vehicle Price Predictor",
            icon=":material/directions_car:",
            text_alignment="center",
        )
        st.markdown(
            "Enter the vehicle information below to get an estimated price.",
            text_alignment="center",
        )

    with st.container(
        horizontal=True,
        vertical_alignment="center",
        horizontal_alignment="distribute",
    ):
        st.subheader("Vehicle basics", icon=":material/directions_car:")
        st.caption("Core identification")

    brand_col, model_col = st.columns(2)
    with brand_col:
        brand = choice("Brand", df["brand"], "brand", df["brand"])
    brand_data = df[df["brand"] == brand]
    with model_col:
        vehicle_model = choice(
            "Model",
            brand_data["model"],
            "vehicle_model",
            df["model"],
        )

    model_data = brand_data[brand_data["model"] == vehicle_model]
    year_col, mileage_col = st.columns(2)
    with year_col:
        model_year = choice(
            "Model year",
            model_data["model_year"],
            "model_year",
            model_data["model_year"],
        )
    year_data = model_data[model_data["model_year"] == model_year]
    with mileage_col:
        mileage = st.number_input(
            "Mileage",
            min_value=0,
            value=50000,
            step=1000,
            key="mileage",
        )

    with st.container(
        horizontal=True,
        vertical_alignment="center",
        horizontal_alignment="distribute",
    ):
        st.subheader(
            "Specifications and powertrain",
            icon=":material/manufacturing:",
        )
        st.caption("Mechanical setup")

    fuel_col, engine_col = st.columns(2)
    with fuel_col:
        fuel_type = choice(
            "Fuel type",
            year_data["fuel_type"],
            "fuel_type",
            model_data["fuel_type"],
        )
    fuel_data = year_data[year_data["fuel_type"] == fuel_type]
    if fuel_data.empty:
        fuel_data = year_data
    with engine_col:
        engine = choice(
            "Engine",
            fuel_data["engine"],
            "engine",
            model_data["engine"],
        )

    engine_data = fuel_data[fuel_data["engine"] == engine]
    if engine_data.empty:
        engine_data = fuel_data
    transmission = choice(
        "Transmission",
        engine_data["transmission"],
        "transmission",
        model_data["transmission"],
    )

    configuration_data = engine_data[engine_data["transmission"] == transmission]
    if configuration_data.empty:
        configuration_data = engine_data

    with st.container(
        horizontal=True,
        vertical_alignment="center",
        horizontal_alignment="distribute",
    ):
        st.subheader("Condition and history", icon=":material/verified_user:")
        st.caption("Cosmetic and legal status")

    exterior_col, interior_col = st.columns(2)
    with exterior_col:
        exterior_color = choice(
            "Exterior color",
            configuration_data["ext_col"],
            "exterior_color",
            model_data["ext_col"],
        )
    with interior_col:
        interior_color = choice(
            "Interior color",
            configuration_data["int_col"],
            "interior_color",
            model_data["int_col"],
        )

    history_col, title_col = st.columns(2)
    with history_col:
        accident = choice(
            "Accident history",
            model_data["accident"],
            "accident",
            df["accident"],
        )
    with title_col:
        clean_title = choice(
            "Clean title",
            model_data["clean_title"],
            "clean_title",
            df["clean_title"],
        )

    vehicle_signature = (
        brand,
        vehicle_model,
        model_year,
        mileage,
        fuel_type,
        engine,
        transmission,
        exterior_color,
        interior_color,
        accident,
        clean_title,
    )

    st.space("small")
    if st.button(
        "Predict price",
        type="primary",
        icon=":material/insights:",
        width="stretch",
        key="predict_price",
    ):
        vehicle = pd.DataFrame(
            {
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
                "clean_title": [clean_title],
            }
        )
        st.session_state["predicted_price"] = float(model.predict(vehicle)[0])
        st.session_state["predicted_for"] = vehicle_signature

    predicted_price = None
    if st.session_state.get("predicted_for") == vehicle_signature:
        predicted_price = st.session_state.get("predicted_price")

    with st.container(key="result_panel", border=True):
        st.caption("Valuation summary", text_alignment="center")
        if predicted_price is None:
            st.markdown(
                "Your estimate will appear here.",
                text_alignment="center",
            )
        else:
            st.metric(
                "Estimated vehicle price",
                f"${predicted_price:,.2f}",
            )
            st.caption(
                "Random Forest estimate from the vehicles in this dataset.",
                text_alignment="center",
            )
