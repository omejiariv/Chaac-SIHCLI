# modules/forecast_api.py

import streamlit as st
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

@st.cache_data(ttl=3600) # Guardar en caché por 1 hora
def get_weather_forecast(latitude, longitude):
    """
    Obtiene el pronóstico del tiempo para 7 días desde la API de Open-Meteo.
    """
    try:
        # Configuración para reintentos y caché
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum"],
            "timezone": "auto"
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Procesar los datos
        daily = response.Daily()
        daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
        daily_precipitation_sum = daily.Variables(3).ValuesAsNumpy()
        
        daily_data = {"date": pd.to_datetime(daily.Time(), unit = "s")}
        daily_data["temp_max (°C)"] = daily_temperature_2m_max
        daily_data["temp_min (°C)"] = daily_temperature_2m_min
        daily_data["precip_sum (mm)"] = daily_precipitation_sum
        
        daily_dataframe = pd.DataFrame(data = daily_data)
        return daily_dataframe

    except Exception as e:
        st.error(f"No se pudo obtener el pronóstico del tiempo. Error: {e}")
        return None
