import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('house_price_model.pkl')
columns = joblib.load('model_columns.pkl')
defaults = joblib.load('default_values.pkl')

st.title('Ames House Price Predictor')
st.write('Enter some house details to estimate the sale price.')

overall_qual = st.slider('Overall Quality (1-10)', 1, 10, 5)
gr_liv_area = st.number_input('Living Area (sq ft)', min_value=300, max_value=6000, value=1500)
garage_cars = st.slider('Garage Capacity (cars)', 0, 4, 2)
total_bsmt_sf = st.number_input('Basement Area (sq ft)', min_value=0, max_value=3000, value=1000)
full_bath = st.slider('Full Bathrooms', 0, 4, 2)
year_built = st.number_input('Year Built', min_value=1870, max_value=2024, value=2000)

if st.button('Predict Price'):
    input_data = defaults.copy()
    
    input_data['OverallQual'] = overall_qual
    input_data['GrLivArea'] = gr_liv_area
    input_data['GarageCars'] = garage_cars
    input_data['TotalBsmtSF'] = total_bsmt_sf
    input_data['FullBath'] = full_bath
    input_data['YearBuilt'] = year_built
    
    input_df = pd.DataFrame([input_data])
    input_df = input_df[columns]
    
    log_prediction = model.predict(input_df)[0]
    predicted_price = np.exp(log_prediction)
    
    st.success(f'Estimated Sale Price: ${predicted_price:,.2f}')
