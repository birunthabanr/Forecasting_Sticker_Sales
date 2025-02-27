import pandas as pd
import joblib
import streamlit as st

# Load trained model
stacking_model = joblib.load("stacking_model.pkl")

st.title("Sticker Sales Forecasting")

# Get user input
id_ = st.sidebar.number_input("ID", value=1, step=1)
country = st.sidebar.number_input("Country (as int)", value=1, step=1)
store = st.sidebar.number_input("Store (as int)", value=1, step=1)
product = st.sidebar.number_input("Product (as int)", value=1, step=1)
year = st.sidebar.number_input("Year", value=2025, step=1)
quarter = st.sidebar.number_input("Quarter", value=1.0, step=0.1)
month = st.sidebar.number_input("Month", value=1, step=1)
week = st.sidebar.number_input("Week", value=1, step=1, min_value=1, max_value=53)
day = st.sidebar.number_input("Day", value=1, step=1, min_value=1, max_value=31)
day_of_week = st.sidebar.number_input("Day of Week", value=1, step=1, min_value=0, max_value=6)
week_of_year = st.sidebar.number_input("Week of Year", value=1.0, step=0.1)
hour = st.sidebar.number_input("Hour", value=0.0, step=0.1, min_value=0.0, max_value=23.9)
minute = st.sidebar.number_input("Minute", value=0.0, step=0.1, min_value=0.0, max_value=59.9)
is_weekend = st.sidebar.selectbox("Is Weekend?", [0, 1])
sine_day = st.sidebar.number_input("Sine Day", value=0.0, step=0.01)
cos_day = st.sidebar.number_input("Cos Day", value=0.0, step=0.01)
sine_month = st.sidebar.number_input("Sine Month", value=0.0, step=0.01)
cos_month = st.sidebar.number_input("Cos Month", value=0.0, step=0.01)
sine_year = st.sidebar.number_input("Sine Year", value=0.0, step=0.01)
cos_year = st.sidebar.number_input("Cos Year", value=0.0, step=0.01)
group = st.sidebar.number_input("Group", value=1, step=1)

# Create DataFrame with correct data types
input_data = pd.DataFrame([[
    id_, country, store, product, year, quarter, month, week, day, day_of_week, 
    week_of_year, hour, minute, is_weekend, sine_day, cos_day, sine_month, cos_month, 
    sine_year, cos_year, group
]], columns=[
    'id', 'country', 'store', 'product', 'year', 'quarter', 'month', 'week', 'day', 
    'day_of_week', 'week_of_year', 'hour', 'minute', 'is_weekend', 'sine_day', 
    'cos_day', 'sine_month', 'cos_month', 'sine_year', 'cos_year', 'group'
]).astype({
    'id': 'int64',
    'country': 'int64',
    'store': 'int64',
    'product': 'int64',
    'year': 'int32',
    'quarter': 'float64',
    'month': 'int32',
    'week': 'UInt32',
    'day': 'int32',
    'day_of_week': 'int32',
    'week_of_year': 'float64',
    'hour': 'float64',
    'minute': 'float64',
    'is_weekend': 'int64',
    'sine_day': 'float64',
    'cos_day': 'float64',
    'sine_month': 'float64',
    'cos_month': 'float64',
    'sine_year': 'float64',
    'cos_year': 'float64',
    'group': 'int32'
})

# Make prediction
if st.sidebar.button("Predict"):
    prediction = stacking_model.predict(input_data)
    st.write(f"### Predicted Sales: {prediction[0]:,.2f}")