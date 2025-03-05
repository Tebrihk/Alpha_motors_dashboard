import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib

@st.cache_data
def load_data():
    df = pd.read_csv("Alpha_Motors.csv.csv")
    return df

df = load_data()

model = joblib.load("car_price_model.pkl")