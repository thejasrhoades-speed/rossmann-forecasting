import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Rossmann Forecast", layout="wide")
st.title("Rossmann 6-week store sales forecast")

@st.cache_data
def load():
    return pd.read_csv("forecast_results.zip", parse_dates=["Date"]).dropna()

df = load()

def rmspe(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    m = y != 0
    return float(np.sqrt(np.mean(((y[m] - yhat[m]) / y[m]) ** 2)))

m_model, m_base = rmspe(df.Sales, df.Forecast), rmspe(df.Sales, df.Baseline)
c1, c2, c3 = st.columns(3)
c1.metric("XGBoost RMSPE", f"{m_model:.3f}")
c2.metric("Baseline RMSPE", f"{m_base:.3f}")
c3.metric("Error reduction", f"{(1 - m_model / m_base):.1%}")

store = st.selectbox("Store", sorted(df.Store.unique()))

# Prepare data and build zero-indexed Altair chart
s = df[df.Store == store].sort_values("Date")
df_long = s[["Date", "Sales", "Forecast", "Baseline"]].melt(
    id_vars="Date", 
    var_name="Series", 
    value_name="Value"
)

chart = alt.Chart(df_long).mark_line().encode(
    x=alt.X("Date:T", axis=alt.Axis(format="%b %d")),
    y=alt.Y("Value:Q", scale=alt.Scale(zero=True), title="Sales"),
    color="Series:N"
)
st.altair_chart(chart, use_container_width=True)

st.caption(f"Store {store}: XGBoost RMSPE {rmspe(s.Sales, s.Forecast):.3f} vs baseline {rmspe(s.Sales, s.Baseline):.3f}")
