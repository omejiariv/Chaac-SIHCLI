# modules/forecasting.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf, acf
from prophet import Prophet
import plotly.graph_objects as go
from modules.config import Config

# --- Funciones de Series de Tiempo y Pronóstico ---

@st.cache_data(show_spinner=False)
def get_decomposition_results(series, period=12, model='additive'):
    """Realiza la descomposición de la serie de tiempo."""
    # Asegurar que la serie no tenga nulos y esté en frecuencia mensual para decomposition
    series_clean = series.asfreq('MS').interpolate(method='time').dropna()
    if len(series_clean) < 2 * period:
        raise ValueError("Serie demasiado corta o con demasiados nulos para la descomposición.")
    return seasonal_decompose(series_clean, model=model, period=period)

def create_acf_chart(series, max_lag):
    """Genera el gráfico de la Función de Autocorrelación (ACF) usando Plotly."""
    if len(series) <= max_lag:
        return go.Figure().update_layout(title="Datos insuficientes para ACF")
    
    # Se utiliza la función acf de statsmodels
    acf_values = acf(series, nlags=max_lag)
    lags = list(range(max_lag + 1))
    conf_interval = 1.96 / np.sqrt(len(series))
    
    fig_acf = go.Figure(data=[
        go.Bar(x=lags, y=acf_values, name='ACF'),
        go.Scatter(x=lags, y=[conf_interval] * (max_lag + 1), mode='lines',
                   line=dict(color='blue', dash='dash'), name='Límite de Confianza Superior'),
        go.Scatter(x=lags, y=[-conf_interval] * (max_lag + 1), mode='lines',
                   line=dict(color='blue', dash='dash'), fill='tonexty', 
                   fillcolor='rgba(0,0,255,0.1)', name='Límite de Confianza Inferior')
    ])
    fig_acf.update_layout(title='Función de Autocorrelación (ACF)', 
                          xaxis_title='Rezagos (Meses)', yaxis_title='Correlación', height=400)
    return fig_acf

def create_pacf_chart(series, max_lag):
    """Genera el gráfico de la Función de Autocorrelación Parcial (PACF) usando Plotly."""
    if len(series) <= max_lag:
        return go.Figure().update_layout(title="Datos insuficientes para PACF")

    # Se utiliza la función pacf de statsmodels
    pacf_values = pacf(series, nlags=max_lag)
    lags = list(range(max_lag + 1))
    conf_interval = 1.96 / np.sqrt(len(series))
    
    fig_pacf = go.Figure(data=[
        go.Bar(x=lags, y=pacf_values, name='PACF'),
        go.Scatter(x=lags, y=[conf_interval] * (max_lag + 1), mode='lines',
                   line=dict(color='red', dash='dash'), name='Límite de Confianza Superior'),
        go.Scatter(x=lags, y=[-conf_interval] * (max_lag + 1), mode='lines',
                   line=dict(color='red', dash='dash'), fill='tonexty', 
                   fillcolor='rgba(255,0,0,0.1)', name='Límite de Confianza Inferior')
    ])
    fig_pacf.update_layout(title='Función de Autocorrelación Parcial (PACF)', 
                           xaxis_title='Rezagos (Meses)', yaxis_title='Correlación', height=400)
    return fig_pacf

def generate_sarima_forecast(ts_data_raw, order, seasonal_order, horizon):
    """
    Entrena un modelo SARIMA y genera un pronóstico.
    """
    ts_data = ts_data_raw[[Config.DATE_COL, Config.PRECIPITATION_COL]].copy()
    ts_data = ts_data.set_index(Config.DATE_COL).sort_index()
    # Asegurar frecuencia y rellenar nulos para el modelo
    ts_data = ts_data[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='time').dropna()
    
    if len(ts_data) < 24:
        raise ValueError("Se necesitan al menos 24 meses de datos para SARIMA.")

    model = sm.tsa.statespace.SARIMAX(
        ts_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=horizon)
    
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # Prepara el resultado para almacenarlo en session_state
    sarima_df_export = forecast_mean.reset_index().rename(
        columns={'index': 'ds', 'predicted_mean': 'yhat'}
    )
    return ts_data, forecast_mean, forecast_ci, sarima_df_export

def generate_prophet_forecast(ts_data_raw, horizon):
    """
    Entrena un modelo Prophet y genera un pronóstico.
    """
    ts_data_prophet = ts_data_raw[[Config.DATE_COL, Config.PRECIPITATION_COL]].copy()
    ts_data_prophet.rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'}, 
                           inplace=True)
    
    if len(ts_data_prophet) < 24:
        raise ValueError("Se necesitan al menos 24 meses de datos para Prophet.")

    # Prophet requiere que los datos sean continuos, por lo que rellenamos nulos si existen
    ts_data_prophet['y'] = ts_data_prophet['y'].interpolate()
    
    # CORRECCIÓN: Se elimina 'monthly_seasonality' y se usa el argumento estándar 'yearly_seasonality'
    # para modelar el ciclo anual de 12 meses.
    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False) 
    
    model_prophet.fit(ts_data_prophet)
    
    future = model_prophet.make_future_dataframe(periods=horizon, freq='MS')
    forecast_prophet = model_prophet.predict(future)
    
    return model_prophet, forecast_prophet
