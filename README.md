# Rossmann Store Sales Forecasting

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rossmann-forecasting-aqg5acezdmxn6tkxdyyeb7.streamlit.app/)

An end-to-end machine learning project predicting 6-week daily sales for Rossmann drug stores. 

## Project Overview
This repository contains the data pipeline, exploratory SQL analysis, and a machine learning model built to improve upon baseline sales forecasts. The final model predictions are served through an interactive Streamlit web dashboard.

- **Model:** XGBoost Regressor (trained on store history, promotions, and holidays)
- **Performance:** Achieved an RMSPE of 0.124, representing a 15.5% error reduction compared to the historical baseline of 0.147.
- **App:** Streamlit dashboard with Altair visualizations to compare model forecasts against baselines and actual sales.

## Repository Contents
- `01_load_data.ipynb`: Jupyter notebook containing the data loading, cleaning, XGBoost model training, and evaluation.
- `queries.sql`: SQL queries analyzing top stores, promotional lift, and rolling averages.
- `app.py`: Streamlit application file for the interactive forecast dashboard.
- `forecast_results.zip`: Pre-computed forecast results consumed by the Streamlit app.
- `requirements.txt`: Environment dependencies.

## How to Run the App Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
