# modules/analysis.py

import pandas as pd
import numpy as np
from scipy.stats import gamma, norm
from modules.config import Config

def calculate_spi(series, window):
    rolling_sum = series.sort_index().rolling(window, min_periods=window).sum()
    params = gamma.fit(rolling_sum.dropna(), floc=0)
    shape, loc, scale = params
    cdf = gamma.cdf(rolling_sum, shape, loc=loc, scale=scale)
    spi = norm.ppf(cdf)
    spi = np.where(np.isinf(spi), np.nan, spi)
    return pd.Series(spi, index=rolling_sum.index)

def calculate_spei(precip_series, et_series, scale):
    """
    Calculates the SPEI using a direct implementation with SciPy.
    """
    from scipy.stats import loglaplace
    
    scale = int(scale)
    
    df = pd.DataFrame({'precip': precip_series, 'et': et_series})
    df = df.sort_index().asfreq('MS')
    df.dropna(inplace=True)
    
    if len(df) < scale * 2:
        return pd.Series(dtype=float)

    # Calculate the water balance (Precipitation - Evapotranspiration)
    water_balance = df['precip'] - df['et']
    
    # Accumulate the water balance over the desired timescale
    rolling_balance = water_balance.rolling(window=scale, min_periods=scale).sum()

    # Fit a Log-Laplace distribution (equivalent to Log-Logistic)
    params = loglaplace.fit(rolling_balance.dropna())
    
    # Calculate the cumulative distribution function (CDF)
    cdf = loglaplace.cdf(rolling_balance, *params)
    
    # Transform to a Z-score of the normal distribution
    spei = norm.ppf(cdf)
    
    # Handle infinite values
    spei = np.where(np.isinf(spei), np.nan, spei)
    
    return pd.Series(spei, index=rolling_balance.index)
    
def calculate_monthly_anomalies(df_monthly_filtered, df_long):
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
    df_station_full = df_long[df_long[Config.STATION_NAME_COL] == station_name].copy()
    df_thresholds = df_station_full.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].agg(
        p_lower=lambda x: np.nanpercentile(x.dropna(), p_lower),
        p_upper=lambda x: np.nanpercentile(x.dropna(), p_upper),
        mean_monthly='mean'
    ).reset_index()
    df_station_extremes = pd.merge(
        df_station_full,
        df_thresholds,
        on=Config.MONTH_COL,
        how='left'
    )
    df_station_extremes['event_type'] = 'Normal'
    is_dry = (df_station_extremes[Config.PRECIPITATION_COL] < df_station_extremes['p_lower'])
    df_station_extremes.loc[is_dry, 'event_type'] = f'Sequía Extrema (< P{p_lower}%)'
    is_wet = (df_station_extremes[Config.PRECIPITATION_COL] > df_station_extremes['p_upper'])
    df_station_extremes.loc[is_wet, 'event_type'] = f'Húmedo Extremo (> P{p_upper}%)'
    return df_station_extremes.dropna(subset=[Config.PRECIPITATION_COL]), df_thresholds
