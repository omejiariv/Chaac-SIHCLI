# modules/analysis.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gamma, norm
from modules.config import Config

@st.cache_data
def calculate_spi(series, window):
    """
    Calcula el Índice de Precipitación Estandarizado (SPI).
    """
    series = series.sort_index()
    rolling_sum = series.rolling(window, min_periods=window).sum()
    data_for_fit = rolling_sum.dropna()
    data_for_fit = data_for_fit[np.isfinite(data_for_fit)]
    
    if len(data_for_fit) > 0:
        params = gamma.fit(data_for_fit, floc=0)
        shape, loc, scale = params
        cdf = gamma.cdf(rolling_sum, shape, loc=loc, scale=scale)
    else:
        return pd.Series(dtype=float)

    spi = norm.ppf(cdf)
    spi = np.where(np.isinf(spi), np.nan, spi)
    return pd.Series(spi, index=rolling_sum.index)

@st.cache_data
def calculate_spei(precip_series, et_series, scale):
    """
    Calcula el Índice de Precipitación y Evapotranspiración Estandarizado (SPEI).
    """
    from scipy.stats import loglaplace
    scale = int(scale)
    df = pd.DataFrame({'precip': precip_series, 'et': et_series})
    df = df.sort_index().asfreq('MS')
    df.dropna(inplace=True)
    if len(df) < scale * 2:
        return pd.Series(dtype=float)

    water_balance = df['precip'] - df['et']
    rolling_balance = water_balance.rolling(window=scale, min_periods=scale).sum()
    data_for_fit = rolling_balance.dropna()
    data_for_fit = data_for_fit[np.isfinite(data_for_fit)]

    if len(data_for_fit) > 0:
        params = loglaplace.fit(data_for_fit)
        cdf = loglaplace.cdf(rolling_balance, *params)
    else:
        return pd.Series(dtype=float)
        
    spei = norm.ppf(cdf)
    spei = np.where(np.isinf(spei), np.nan, spei)
    return pd.Series(spei, index=rolling_balance.index)
    
@st.cache_data
def calculate_monthly_anomalies(_df_monthly_filtered, _df_long):
    """
    Calcula las anomalías mensuales con respecto al promedio de todo el período de datos.
    """
    df_monthly_filtered = _df_monthly_filtered.copy()
    df_long = _df_long.copy()
    
    df_climatology = df_long[
        df_long[Config.STATION_NAME_COL].isin(df_monthly_filtered[Config.STATION_NAME_COL].unique())
    ].groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean() \
     .reset_index().rename(columns={Config.PRECIPITATION_COL: 'precip_promedio_mes'})

    df_anomalias = pd.merge(
        df_monthly_filtered,
        df_climatology,
        on=[Config.STATION_NAME_COL, Config.MONTH_COL],
        how='left'
    )
    df_anomalias['anomalia'] = df_anomalias[Config.PRECIPITATION_COL] - df_anomalias['precip_promedio_mes']
    return df_anomalias.copy()

def calculate_percentiles_and_extremes(df_long, station_name, p_lower=10, p_upper=90):
    """
    Calcula umbrales de percentiles y clasifica eventos extremos para una estación.
    """
    df_station_full = df_long[df_long[Config.STATION_NAME_COL] == station_name].copy()
    df_thresholds = df_station_full.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].agg(
        p_lower=lambda x: np.nanpercentile(x.dropna(), p_lower),
        p_upper=lambda x: np.nanpercentile(x.dropna(), p_upper),
        mean_monthly='mean'
    ).reset_index()
    df_station_extremes = pd.merge(df_station_full, df_thresholds, on=Config.MONTH_COL, how='left')
    df_station_extremes['event_type'] = 'Normal'
    is_dry = (df_station_extremes[Config.PRECIPITATION_COL] < df_station_extremes['p_lower'])
    df_station_extremes.loc[is_dry, 'event_type'] = f'Sequía Extrema (< P{p_lower}%)'
    is_wet = (df_station_extremes[Config.PRECIPITATION_COL] > df_station_extremes['p_upper'])
    df_station_extremes.loc[is_wet, 'event_type'] = f'Húmedo Extremo (> P{p_upper}%)'
    return df_station_extremes.dropna(subset=[Config.PRECIPITATION_COL]), df_thresholds

@st.cache_data
def calculate_climatological_anomalies(_df_monthly_filtered, _df_long, baseline_start, baseline_end):
    """
    Calcula las anomalías mensuales con respecto a un período base climatológico fijo.
    """
    df_monthly_filtered = _df_monthly_filtered.copy()
    df_long = _df_long.copy()

    baseline_df = df_long[
        (df_long[Config.YEAR_COL] >= baseline_start) & 
        (df_long[Config.YEAR_COL] <= baseline_end)
    ]

    df_climatology = baseline_df.groupby(
        [Config.STATION_NAME_COL, Config.MONTH_COL]
    )[Config.PRECIPITATION_COL].mean().reset_index().rename(
        columns={Config.PRECIPITATION_COL: 'precip_promedio_climatologico'}
    )

    df_anomalias = pd.merge(
        df_monthly_filtered,
        df_climatology,
        on=[Config.STATION_NAME_COL, Config.MONTH_COL],
        how='left'
    )

    df_anomalias['anomalia'] = df_anomalias[Config.PRECIPITATION_COL] - df_anomalias['precip_promedio_climatologico']
    return df_anomalias

@st.cache_data
def analyze_events(index_series, threshold, event_type='drought'):
    """
    Identifica y caracteriza eventos de sequía o humedad en una serie de tiempo de índices.
    """
    if event_type == 'drought':
        is_event = index_series < threshold
    else: # 'wet'
        is_event = index_series > threshold

    event_blocks = (is_event.diff() != 0).cumsum()
    active_events = is_event[is_event]
    if active_events.empty:
        return pd.DataFrame()

    events = []
    for event_id, group in active_events.groupby(event_blocks):
        start_date = group.index.min()
        end_date = group.index.max()
        duration = len(group)
        
        event_values = index_series.loc[start_date:end_date]
        
        magnitude = event_values.sum()
        intensity = event_values.mean()
        peak = event_values.min() if event_type == 'drought' else event_values.max()

        events.append({
            'Fecha Inicio': start_date,
            'Fecha Fin': end_date,
            'Duración (meses)': duration,
            'Magnitud': magnitude,
            'Intensidad': intensity,
            'Pico': peak
        })

    if not events:
        return pd.DataFrame()

    events_df = pd.DataFrame(events)
    return events_df.sort_values(by='Fecha Inicio').reset_index(drop=True)
