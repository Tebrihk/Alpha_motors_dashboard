import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Alpha_Motors.csv")

    # Convert Price column to numeric (fixing issue)
    df["Price"] = df["Price"].replace('[^\d.]', '', regex=True).astype(float)

    # Create Car Age
    df["Car Age"] = 2024 - df["Year of manufacture"]

    # Avoid division by zero in Price per KM
    df["Price per KM"] = df["Price"] / df["Mileage"].replace(0, np.nan)

    return df


df = load_data()

# Load trained model
model = joblib.load("car_price_model.pkl")

# Streamlit Title
st.title("🚗 ALPHA MOTORS Dashboard")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Pricing Analysis", "Depreciation Trends", "Predict Price"])

# Page 1: Overview
if page == "Overview":
    st.header("📌 Overview of Car Listings")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cars", df.shape[0])
    col2.metric("Avg. Price", f"₦{df['Price'].mean():,.0f}")
    col3.metric("Most Common Brand", df['Make'].mode()[0])
    col4.metric("Avg. Car Age", f"{df['Car Age'].mean():.1f} years")

    st.dataframe(df.head())

# 2: Pricing Analysis
elif page == "Pricing Analysis":
    st.header("📊 Pricing Analysis")

    st.subheader("Price Distribution")
    fig = px.histogram(df, x="Price",y="amount", nbins=50, title="Distribution of Car Prices")
    st.plotly_chart(fig)

    st.subheader("Average Price by Brand")
    avg_price_brand = df.groupby("Make")["Price"].mean().sort_values(ascending=False)
    fig = px.bar(avg_price_brand, x=avg_price_brand.index, y=avg_price_brand.values, title="Avg. Price by Brand")
    st.plotly_chart(fig)

# Page 3: Depreciation Trends
elif page == "Depreciation Trends":
    st.header("📉 Depreciation & Mileage Impact")

    st.subheader("Price vs. Mileage")
    fig = px.scatter(df, x="Mileage", y="Price", title="Impact of Mileage on Price")
    st.plotly_chart(fig)

    st.subheader("Top 5 Brands Retaining Value")
    brand_depreciation = df.groupby("Make")["Price per KM"].mean().nsmallest(5)
    fig = px.bar(brand_depreciation, x=brand_depreciation.index, y=brand_depreciation.values,
                 title="Best Resale Value Brands")
    st.plotly_chart(fig)

# Page 4: Predict Price
elif page == "Predict Price":
    st.header("🔮 Predict Car Price")

    # User Inputs

    year = st.slider("Year of Manufacture", min_value=int(df["Year of manufacture"].min()), max_value=2024, value=2015)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=int(df["Mileage"].max()), value=20000)
    engine_size = st.number_input("Engine Size (cc)", min_value=500, max_value=6000, value=2000)
    make = st.selectbox("Select Car Make", df["Make"].unique())
    condition = st.selectbox("Condition", df["Condition"].unique())
    fuel = st.selectbox("Fuel Type", df["Fuel"].unique())
    transmission = st.selectbox("Transmission", df["Transmission"].unique())

    # Prediction Button
    if st.button("Predict Price"):
        preprocessor = joblib.load("preprocessor.pkl")  # Load saved preprocessor
        model = joblib.load("car_price_model.pkl")
        input_data = pd.DataFrame([[make, year, condition, mileage, engine_size, fuel, transmission]],
                                  columns=["Make", "Year of manufacture", "Condition", "Mileage", "Engine Size", "Fuel",
                                           "Transmission"])
        transformed_data = preprocessor.transform(input_data)
        prediction = model.predict(transformed_data)[0]
        st.success(f"Estimated Price: ₦{prediction:,.0f}")



