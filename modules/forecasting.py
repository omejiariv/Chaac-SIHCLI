# modules/forecasting.py

import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from modules.config import Config

@st.cache_data(show_spinner=False)
def get_decomposition_results(series, period=12, model='additive'):
    """Realiza la descomposición de la serie de tiempo."""
    series_clean = series.asfreq('MS').interpolate(method='time').dropna()
    if len(series_clean) < 2 * period:
        raise ValueError("Serie demasiado corta o con demasiados nulos para la descomposición.")
    return seasonal_decompose(series_clean, model=model, period=period)

def create_acf_chart(series, max_lag):
    """Genera el gráfico de la Función de Autocorrelación (ACF) usando Plotly."""
    if len(series) <= max_lag:
        return go.Figure().update_layout(title="Datos insuficientes para ACF")
    
    acf_values = acf(series, nlags=max_lag)
    lags = list(range(max_lag + 1))
    conf_interval = 1.96 / np.sqrt(len(series))
    
    fig_acf = go.Figure(data=[
        go.Bar(x=lags, y=acf_values, name='ACF'),
        go.Scatter(x=lags, y=[conf_interval] * (max_lag + 1), mode='lines', line=dict(color='blue', dash='dash'), name='Límite de Confianza'),
        go.Scatter(x=lags, y=[-conf_interval] * (max_lag + 1), mode='lines', line=dict(color='blue', dash='dash'), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', showlegend=False)
    ])
    fig_acf.update_layout(title='Función de Autocorrelación (ACF)', xaxis_title='Rezagos (Meses)', yaxis_title='Correlación', height=400)
    return fig_acf

def create_pacf_chart(series, max_lag):
    """Genera el gráfico de la Función de Autocorrelación Parcial (PACF) usando Plotly."""
    if len(series) <= max_lag:
        return go.Figure().update_layout(title="Datos insuficientes para PACF")

    pacf_values = pacf(series, nlags=max_lag)
    lags = list(range(max_lag + 1))
    conf_interval = 1.96 / np.sqrt(len(series))
    
    fig_pacf = go.Figure(data=[
        go.Bar(x=lags, y=pacf_values, name='PACF'),
        go.Scatter(x=lags, y=[conf_interval] * (max_lag + 1), mode='lines', line=dict(color='red', dash='dash'), name='Límite de Confianza'),
        go.Scatter(x=lags, y=[-conf_interval] * (max_lag + 1), mode='lines', line=dict(color='red', dash='dash'), fill='tonexty', fillcolor='rgba(255,0,0,0.1)', showlegend=False)
    ])
    fig_pacf.update_layout(title='Función de Autocorrelación Parcial (PACF)', xaxis_title='Rezagos (Meses)', yaxis_title='Correlación', height=400)
    return fig_pacf

def evaluate_forecast(y_true, y_pred):
    """Calcula RMSE y MAE para evaluar un pronóstico."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}

def generate_sarima_forecast(ts_data_raw, order, seasonal_order, horizon, test_size=12, regressors=None):
    """Entrena, evalúa y genera un pronóstico con SARIMAX, incluyendo regresores opcionales."""
    ts_data = ts_data_raw[[Config.DATE_COL, Config.PRECIPITATION_COL]].copy()
    ts_data = ts_data.set_index(Config.DATE_COL).sort_index()
    ts = ts_data[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='time').dropna()
    
    if len(ts) < test_size + 24:
        raise ValueError(f"Se necesitan al menos {test_size + 24} meses de datos para el pronóstico y la evaluación.")

    # Preparar regresores si existen
    exog, exog_train, exog_test, exog_future = None, None, None, None
    if regressors is not None and not regressors.empty:
        exog = regressors.set_index(Config.DATE_COL)
        exog = exog.reindex(ts.index).interpolate() # Alinear y rellenar regresores
    
    # Separar datos de entrenamiento y prueba
    train, test = ts[:-test_size], ts[-test_size:]
    if exog is not None:
        exog_train, exog_test = exog.iloc[:-test_size], exog.iloc[-test_size:]

    # 1. Entrenar y Evaluar
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, exog=exog_train, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    pred_test = results.get_forecast(steps=test_size, exog=exog_test)
    y_pred_test = pred_test.predicted_mean
    metrics = evaluate_forecast(test, y_pred_test)

    # 2. Re-entrenar con todos los datos para el pronóstico final
    if exog is not None:
        last_regressor_values = exog.iloc[-1:].values
        future_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=horizon, freq='MS')
        exog_future = pd.DataFrame(np.tile(last_regressor_values, (horizon, 1)), index=future_index, columns=exog.columns)
        
    full_model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)
    full_results = full_model.fit(disp=False)
    forecast = full_results.get_forecast(steps=horizon, exog=exog_future)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    sarima_df_export = forecast_mean.reset_index().rename(columns={'index': 'ds', 'predicted_mean': 'yhat'})
    return ts, forecast_mean, forecast_ci, metrics, sarima_df_export

def generate_prophet_forecast(ts_data_raw, horizon, test_size=12, regressors=None):
    """Entrena, evalúa y genera un pronóstico con Prophet, incluyendo regresores opcionales."""
    ts_data = ts_data_raw.rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'})
    ts_data['y'] = ts_data['y'].interpolate()
    
    if len(ts_data) < test_size + 24:
        raise ValueError(f"Se necesitan al menos {test_size + 24} meses de datos para Prophet.")

    # Unir regresores a los datos
    if regressors is not None and not regressors.empty:
        ts_data = pd.merge(ts_data, regressors.rename(columns={Config.DATE_COL: 'ds'}), on='ds', how='left')
        for col in regressors.columns:
            if col != Config.DATE_COL:
                ts_data[col] = ts_data[col].interpolate() # Rellenar posibles nulos en regresores
    
    train, test = ts_data.iloc[:-test_size], ts_data.iloc[-test_size:]

    # 1. Entrenar y Evaluar
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    if regressors is not None:
        for col in regressors.columns:
            if col != Config.DATE_COL:
                model.add_regressor(col)
    model.fit(train)
    
    future_test = model.make_future_dataframe(periods=0, freq='MS')
    future_test = future_test[future_test['ds'].isin(train['ds'])]
    if regressors is not None:
        future_test = pd.merge(future_test, regressors.rename(columns={Config.DATE_COL: 'ds'}), on='ds', how='left')

    forecast_on_train = model.predict(future_test)
    pred_df = forecast_on_train[['ds', 'yhat']].set_index('ds')
    
    # Necesitamos predecir sobre el set de prueba
    test_dates = model.make_future_dataframe(periods=test_size, freq='MS').tail(test_size)
    if regressors is not None:
        # Para la evaluación, usamos los valores conocidos del regresor en el período de prueba
        test_regressors = ts_data[ts_data['ds'].isin(test_dates['ds'])]
        test_dates = pd.merge(test_dates, test_regressors, on='ds', how='left')
        test_dates.interpolate(inplace=True)

    y_pred_test = model.predict(test_dates)['yhat']
    metrics = evaluate_forecast(test['y'], y_pred_test)

    # 2. Re-entrenar con todos los datos para el pronóstico final
    full_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    if regressors is not None:
        for col in regressors.columns:
            if col != Config.DATE_COL:
                full_model.add_regressor(col)
    full_model.fit(ts_data)
    
    future = full_model.make_future_dataframe(periods=horizon, freq='MS')
    if regressors is not None:
        last_regressor_values = regressors.iloc[-1:].drop(Config.DATE_COL, axis=1)
        future_regressors = pd.DataFrame(np.tile(last_regressor_values.values, (horizon, 1)), columns=last_regressor_values.columns)
        future_regressors['ds'] = future['ds'].iloc[-horizon:]
        future = pd.merge(future, future_regressors, on='ds', how='left')

    forecast = full_model.predict(future)
    return full_model, forecast, metrics
