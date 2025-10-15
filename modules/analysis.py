# modules/analysis.py

import streamlit as st
import geopandas as gpd
import pandas as pd
import requests
import numpy as np
from scipy.stats import gamma, norm
from modules.config import Config
import rasterio
from rasterstats import zonal_stats

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

@st.cache_data
def calculate_basin_stats(_gdf_stations, _gdf_basins, _df_monthly, basin_name, basin_col_name):
    """
    Calcula estadísticas de precipitación para todas las estaciones dentro de una cuenca específica.
    """
    if _gdf_basins is None or basin_col_name not in _gdf_basins.columns:
        return pd.DataFrame(), [], "El GeoDataFrame de cuencas o la columna de nombres no es válida."

    target_basin = _gdf_basins[_gdf_basins[basin_col_name] == basin_name]
    if target_basin.empty:
        return pd.DataFrame(), [], f"No se encontró la cuenca llamada '{basin_name}'."

    stations_in_basin = gpd.sjoin(_gdf_stations, target_basin, how="inner", predicate="within")
    station_names_in_basin = stations_in_basin[Config.STATION_NAME_COL].unique().tolist()

    if not station_names_in_basin:
        return pd.DataFrame(), [], None

    df_basin_precip = _df_monthly[_df_monthly[Config.STATION_NAME_COL].isin(station_names_in_basin)]
    if df_basin_precip.empty:
        return pd.DataFrame(), station_names_in_basin, "No hay datos de precipitación para las estaciones en esta cuenca."
    
    stats = df_basin_precip[Config.PRECIPITATION_COL].describe().reset_index()
    stats.columns = ['Métrica', 'Valor']
    stats['Valor'] = stats['Valor'].round(2)
    
    return stats, station_names_in_basin, None

@st.cache_data
def get_mean_altitude_for_basin(_basin_geometry):
    """
    Calcula la altitud media de una cuenca consultando la API de Open-Elevation.
    """
    try:
        # Simplificamos la geometría para reducir el tamaño de la consulta
        simplified_geom = _basin_geometry.simplify(tolerance=0.01)
        
        # Obtenemos los puntos del contorno exterior del polígono
        exterior_coords = list(simplified_geom.exterior.coords)
        
        # Creamos la estructura de datos para la API
        locations = [{"latitude": lat, "longitude": lon} for lon, lat in exterior_coords]
        
        # Hacemos la llamada a la API de Open-Elevation
        response = requests.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": locations})
        response.raise_for_status()
        
        results = response.json()['results']
        elevations = [res['elevation'] for res in results]
        
        # Calculamos la media de las elevaciones obtenidas
        mean_altitude = np.mean(elevations)
        
        return mean_altitude, None
    except Exception as e:
        error_message = f"No se pudo obtener la altitud de la cuenca: {e}"
        st.warning(error_message)
        return None, error_message

def calculate_hydrological_balance(mean_precip_mm, basin_geometry_input):
    """
    Calcula el balance hídrico (P - ET = Q) para una cuenca o conjunto de cuencas.
    """
    results = {
        "P_media_anual_mm": mean_precip_mm,
        "Altitud_media_m": None,
        "ET_media_anual_mm": None,
        "Q_mm": None,
        "Q_m3_año": None,
        "Area_km2": None,
        "error": None
    }

    # --- INICIO DE LA CORRECCIÓN ---
    # Aseguramos que siempre trabajamos con un objeto GeoPandas
    if isinstance(basin_geometry_input, (gpd.GeoDataFrame, gpd.GeoSeries)):
        basin_geopandas_obj = basin_geometry_input
    else:
        # Si es una geometría cruda, la convertimos a un GeoSeries con el CRS correcto (WGS84)
        basin_geopandas_obj = gpd.GeoSeries([basin_geometry_input], crs="EPSG:4326")
    # --- FIN DE LA CORRECCIÓN ---

    # 1. Calcular Altitud Media
    # Pasamos la geometría unificada a la función de altitud
    mean_altitude, error = get_mean_altitude_for_basin(basin_geopandas_obj.unary_union)
    if error:
        results["error"] = error
        return results
    results["Altitud_media_m"] = mean_altitude

    # 2. Calcular Evapotranspiración (ET)
    eto_dia = 4.37 * np.exp(-0.0002 * mean_altitude)
    eto_anual_mm = eto_dia * 365.25
    results["ET_media_anual_mm"] = eto_anual_mm

    # 3. Calcular la Escorrentía (Q)
    q_mm = mean_precip_mm - eto_anual_mm
    results["Q_mm"] = q_mm

    # 4. Calcular el Caudal en Volumen
    basin_metric = basin_geopandas_obj.to_crs("EPSG:3116")
    area_m2 = basin_metric.area.sum() # Usamos .sum() para obtener el área total
    results["Area_km2"] = area_m2 / 1_000_000

    q_m = q_mm / 1000
    q_volumen_m3_anual = q_m * area_m2
    results["Q_m3_año"] = q_volumen_m3_anual
    
    return results

def calculate_morphometry(basin_gdf, dem_path):
    """
    Calcula los parámetros morfométricos de una cuenca, leyendo dinámicamente
    el valor NoData del DEM para cálculos de elevación precisos.
    """
    if basin_gdf.empty or basin_gdf.iloc[0].geometry is None:
        return {"error": "Geometría de cuenca no válida."}

    basin_metric = basin_gdf.to_crs("EPSG:3116")
    geom = basin_metric.iloc[0].geometry

    # --- Cálculos Geométricos (sin cambios) ---
    area_m2 = geom.area
    perimetro_m = geom.length
    indice_forma = perimetro_m / (2 * np.sqrt(np.pi * area_m2)) if area_m2 > 0 else 0

    # --- Cálculos de Elevación Mejorados ---
    stats = {}
    try:
        # --- SOLUCIÓN: Leer el valor NoData directamente del archivo DEM ---
        with rasterio.open(dem_path) as src:
            nodata_value = src.nodata
        # --- FIN DE LA SOLUCIÓN ---

        # Usamos el valor NoData leído para asegurar que se ignore correctamente
        z_stats = zonal_stats(basin_metric, dem_path, stats="min max mean", nodata=nodata_value)
        
        if z_stats:
            stats['alt_min'] = z_stats[0].get('min')
            stats['alt_max'] = z_stats[0].get('max')
            stats['alt_prom'] = z_stats[0].get('mean')

    except Exception as e:
        stats['error_dem'] = f"No se pudieron calcular las estadísticas de elevación: {e}"

    return {
        "area_km2": area_m2 / 1_000_000,
        "perimetro_km": perimetro_m / 1_000,
        "indice_forma": indice_forma,
        "alt_max_m": stats.get('alt_max'),
        "alt_min_m": stats.get('alt_min'),
        "alt_prom_m": stats.get('alt_prom'),
        "error": stats.get('error_dem')
    }
