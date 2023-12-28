#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 11:30:08 2023

@author: hari
"""
#!pip install fbprophet

import streamlit as st
import pandas as pd
#from fbprophet import Prophet
import pickle


with open('/Users/hari/Desktop/Projects/P323/fb_prophet.pkl', 'rb') as model_file:
# with open('RFC.sav', 'rb') as model_file: 
    model = pickle.load(model_file)
    
    
# Streamlit UI
st.title('Oil Price Forecasting')

# User input for forecasting
st.sidebar.header('Input Parameters')
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("today") - pd.DateOffset(days=365))
forecast_days = st.sidebar.slider("Number of Days to Forecast", 1, 365, 30)

# Generate date range for forecasting
future_dates = pd.date_range(start=start_date, periods=forecast_days)

# Predict using the Prophet model
future = pd.DataFrame(future_dates, columns=['ds'])
forecast = model.predict(future)

# Display the forecasted data
st.subheader('Forecasted Oil Prices:')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Visualize the forecast
fig = model.plot(forecast)
st.write(fig)

# Show components (trends and seasonality)
fig_comp = model.plot_components(forecast)
st.write(fig_comp)

