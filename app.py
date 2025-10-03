# app.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os

# --- Importaciones de Módulos ---
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series, extract_elevation_from_dem
from modules.visualizer import (
    display_welcome_tab, display_spatial_distribution_tab, display_graphs_tab,
    display_advanced_maps_tab, display_anomalies_tab, display_drought_analysis_tab,
    display_frequency_analysis_tab,
    display_stats_tab, display_correlation_tab, display_enso_tab,
    display_trends_and_forecast_tab, display_downloads_tab, display_station_table_tab
)

# Desactivar Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def sync_station_selection(stations_options):
    """Sincroniza el multiselect basado en el checkbox 'Seleccionar todas'."""
    if st.session_state.get('select_all_checkbox', False): # Default a False
        st.session_state.station_multiselect = stations_options
    else:
        st.session_state.station_multiselect = []

def apply_filters_to_stations(df, min_perc, altitudes, regions, municipios, celdas):
    """Aplica una serie de filtros geográficos y de datos a un GeoDataFrame de estaciones."""
    stations_filtered = df.copy()
    if Config.PERCENTAGE_COL in stations_filtered.columns:
        if stations_filtered[Config.PERCENTAGE_COL].dtype == 'object':
            stations_filtered[Config.PERCENTAGE_COL] = pd.to_numeric(stations_filtered[Config.PERCENTAGE_COL].astype(str).str.replace(',', '.', regex=False), errors='coerce').fillna(0)
        if min_perc > 0:
            stations_filtered = stations_filtered[stations_filtered[Config.PERCENTAGE_COL] >= min_perc]
    if altitudes:
        conditions = []
        altitude_col_numeric = pd.to_numeric(stations_filtered[Config.ALTITUDE_COL], errors='coerce')
        for r in altitudes:
            if r == '0-500': conditions.append((altitude_col_numeric >= 0) & (altitude_col_numeric <= 500))
            elif r == '500-1000': conditions.append((altitude_col_numeric > 500) & (altitude_col_numeric <= 1000))
            elif r == '1000-2000': conditions.append((altitude_col_numeric > 1000) & (altitude_col_numeric <= 2000))
            elif r == '2000-3000': conditions.append((altitude_col_numeric > 2000) & (altitude_col_numeric <= 3000))
            elif r == '>3000': conditions.append(altitude_col_numeric > 3000)
        if conditions: stations_filtered = stations_filtered[pd.concat(conditions, axis=1).any(axis=1)]
    if regions: stations_filtered = stations_filtered[stations_filtered[Config.REGION_COL].isin(regions)]
    if municipios: stations_filtered = stations_filtered[stations_filtered[Config.MUNICIPALITY_COL].isin(municipios)]
    if celdas and Config.CELL_COL in stations_filtered.columns:
        stations_filtered = stations_filtered[stations_filtered[Config.CELL_COL].isin(celdas)]
    return stations_filtered

def main():
    st.set_page_config(layout="wide", page_title=Config.APP_TITLE)
    st.markdown("""<style>div.block-container{padding-top:2rem;}[data-testid="stMetricValue"]{font-size:1.8rem;}[data-testid="stMetricLabel"] {font-size: 1rem; padding-bottom:5px; } button[data-baseweb="tab"] {font-size:16px;font-weight:bold;color:#333;}</style>""", unsafe_allow_html=True)
    Config.initialize_session_state()
    
    title_col1, title_col2 = st.columns([0.05, 0.95])
    with title_col1:
        if os.path.exists(Config.LOGO_PATH):
            try: st.image(Config.LOGO_PATH, width=60)
            except Exception: pass
    with title_col2:
        st.markdown(f'<h1 style="font-size:28px; margin-top:1rem;">{Config.APP_TITLE}</h1>', unsafe_allow_html=True)
    
    st.sidebar.header("Panel de Control")
    with st.sidebar.expander("**Subir/Actualizar Archivos Base**", expanded=not st.session_state.get('data_loaded', False)):
        uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (CSV)", type="csv", key='uploaded_file_mapa')
        uploaded_file_precip = st.file_uploader("2. Cargar archivo de precipitación (CSV)", type="csv", key='uploaded_file_precip')
        uploaded_zip_shapefile = st.file_uploader("3. Cargar shapefile de municipios (.zip)", type="zip", key='uploaded_zip_shapefile')
        if st.button("Procesar y Almacenar Datos", key='process_data_button') and all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
            st.cache_data.clear()
            st.cache_resource.clear()
            keys_to_clear = list(st.session_state.keys())
            for key in keys_to_clear:
                del st.session_state[key]
            
            with st.spinner("Procesando archivos y cargando datos..."):
                gdf_stations, gdf_municipios, df_long, df_enso = load_and_process_all_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
                if gdf_stations is not None and df_long is not None and gdf_municipios is not None:
                    st.session_state.gdf_stations = gdf_stations
                    st.session_state.gdf_municipios = gdf_municipios
                    st.session_state.df_long = df_long
                    st.session_state.df_enso = df_enso
                    st.session_state.data_loaded = True
                    st.success("¡Datos cargados y listos!")
                    st.rerun()
                else:
                    st.error("Hubo un error al procesar los archivos. Verifique el formato de los datos.")

    if not st.session_state.get('data_loaded', False):
        display_welcome_tab()
        st.info("Para comenzar, cargue los archivos requeridos en el panel de la izquierda.")
        return

    st.sidebar.success("Datos base cargados.")
    if st.sidebar.button("Limpiar Caché y Reiniciar App"):
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()

    with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
        min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, st.session_state.get('min_data_perc_slider', 0), key='min_data_perc_slider')
        altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000']
        selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges, key='altitude_multiselect')
        regions_list = sorted(st.session_state.gdf_stations[Config.REGION_COL].dropna().unique())
        selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list, key='regions_multiselect')
        temp_gdf_for_mun = st.session_state.gdf_stations.copy()
        if selected_regions:
            temp_gdf_for_mun = temp_gdf_for_mun[temp_gdf_for_mun[Config.REGION_COL].isin(selected_regions)]
        municipios_list = sorted(temp_gdf_for_mun[Config.MUNICIPALITY_COL].dropna().unique())
        selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list, key='municipios_multiselect')
        celdas_list = []
        if Config.CELL_COL in temp_gdf_for_mun.columns:
            celdas_list = sorted(temp_gdf_for_mun[Config.CELL_COL].dropna().unique())
        selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, key='celdas_multiselect')
        gdf_filtered = apply_filters_to_stations(st.session_state.gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)

    with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
        stations_options = sorted(gdf_filtered[Config.STATION_NAME_COL].unique())
        
        # ==============================================================================
        # LÓGICA CLAVE PARA EVITAR LA SOBRECARGA INICIAL
        # ==============================================================================
        # Si las opciones de estaciones cambian (p. ej., por un filtro geográfico), 
        # reiniciamos la selección a una lista vacía por defecto.
        if 'filtered_station_options' not in st.session_state or st.session_state.filtered_station_options != stations_options:
            st.session_state.filtered_station_options = stations_options
            st.session_state.station_multiselect = []
            st.session_state.select_all_checkbox = False # Asegura que el checkbox también se reinicie

        st.checkbox("Seleccionar/Deseleccionar todas las estaciones", key='select_all_checkbox', on_change=sync_station_selection, args=(stations_options,))
        selected_stations = st.multiselect('Seleccionar Estaciones', options=stations_options, key='station_multiselect')
        # ==============================================================================

        years_with_data = sorted(st.session_state.df_long[Config.YEAR_COL].dropna().unique())
        year_range_default = (min(years_with_data), max(years_with_data)) if years_with_data else (1970, 2020)
        year_range = st.slider("Seleccionar Rango de Años", min_value=year_range_default[0], max_value=year_range_default[1], value=st.session_state.get('year_range', year_range_default), key='year_range')
        meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
        meses_nombres = st.multiselect("Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()), key='meses_nombres')
        meses_numeros = [meses_dict[m] for m in meses_nombres]

    with st.sidebar.expander("Opciones de Preprocesamiento"):
        st.radio("Modo de análisis", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode", help="La opción 'Completar series' solo se aplica bajo demanda en la pestaña de Pronósticos.")
        st.checkbox("Excluir datos nulos (NaN)", key='exclude_na')
        st.checkbox("Excluir valores cero (0)", key='exclude_zeros')

    tab_names = ["Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados", 
                 "Análisis de Anomalías", "Análisis de extremos hid", "Frecuencia de Extremos",
                 "Estadísticas", "Análisis de Correlación", "Análisis ENSO", 
                 "Tendencias y Pronósticos", "Descargas", "Tabla de Estaciones"]
    tabs = st.tabs(tab_names)

    stations_for_analysis = selected_stations
    if not stations_for_analysis:
        with tabs[0]: display_welcome_tab()
        st.warning("No hay estaciones seleccionadas. Por favor, seleccione al menos una estación en el panel de control para comenzar el análisis.")
        return
        
    gdf_filtered = gdf_filtered[gdf_filtered[Config.STATION_NAME_COL].isin(stations_for_analysis)]
    
    df_monthly_filtered = st.session_state.df_long[
        (st.session_state.df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
        (st.session_state.df_long[Config.DATE_COL].dt.year >= year_range[0]) &
        (st.session_state.df_long[Config.DATE_COL].dt.year <= year_range[1]) &
        (st.session_state.df_long[Config.DATE_COL].dt.month.isin(meses_numeros))
    ].copy()

    annual_data_filtered = st.session_state.df_long[
        (st.session_state.df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
        (st.session_state.df_long[Config.YEAR_COL] >= year_range[0]) &
        (st.session_state.df_long[Config.YEAR_COL] <= year_range[1])
    ].copy()

    if st.session_state.get('exclude_na', False):
        df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
        annual_data_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
            
    if st.session_state.get('exclude_zeros', False):
        df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] > 0]
        annual_data_filtered = annual_data_filtered[annual_data_filtered[Config.PRECIPITATION_COL] > 0]
    
    annual_agg = annual_data_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).agg(
        precipitation_sum=(Config.PRECIPITATION_COL, 'sum'),
        meses_validos=(Config.PRECIPITATION_COL, 'count')
    ).reset_index()
    annual_agg.loc[annual_agg['meses_validos'] < 10, 'precipitation_sum'] = np.nan
    df_anual_melted = annual_agg.rename(columns={'precipitation_sum': Config.PRECIPITATION_COL})
    
    display_args = {
        "gdf_filtered": gdf_filtered, 
        "stations_for_analysis": stations_for_analysis,
        "df_anual_melted": df_anual_melted, 
        "df_monthly_filtered": df_monthly_filtered,
        "analysis_mode": st.session_state.analysis_mode, 
        "selected_regions": selected_regions,
        "selected_municipios": selected_municipios, 
        "selected_altitudes": selected_altitudes,
        "df_full_monthly": st.session_state.df_long
    }
    
    with tabs[0]: display_welcome_tab()
    with tabs[1]: display_spatial_distribution_tab(**display_args)
    with tabs[2]: display_graphs_tab(**display_args)
    with tabs[3]: display_advanced_maps_tab(**display_args)
    with tabs[4]: display_anomalies_tab(df_long=st.session_state.df_long, **display_args)
    with tabs[5]: display_drought_analysis_tab(**display_args)
    with tabs[6]: display_frequency_analysis_tab(**display_args)
    with tabs[7]: display_stats_tab(df_long=st.session_state.df_long, **display_args)
    with tabs[8]: display_correlation_tab(**display_args)
    with tabs[9]: display_enso_tab(df_enso=st.session_state.df_enso, **display_args)
    with tabs[10]: display_trends_and_forecast_tab(**display_args)
    with tabs[11]: display_downloads_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis)
    with tabs[12]: display_station_table_tab(**display_args)

if __name__ == "__main__":
    main()
