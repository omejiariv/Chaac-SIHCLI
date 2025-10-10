# modules/sidebar.py

import streamlit as st
from modules.config import Config
import pandas as pd

# Definimos esta función aquí para que sidebar sea independiente de app.py
def _apply_geo_filters(gdf_stations, min_perc, altitudes, regions, municipios, celdas):
    """Función auxiliar para pre-filtrar estaciones basado en filtros geográficos."""
    stations_filtered = gdf_stations.copy()
    # Lógica de filtrado (simplificada de la función principal)
    if Config.PERCENTAGE_COL in stations_filtered.columns:
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
        if conditions:
            stations_filtered = stations_filtered.loc[pd.concat(conditions, axis=1).any(axis=1)]
    if regions:
        stations_filtered = stations_filtered[stations_filtered[Config.REGION_COL].isin(regions)]
    if municipios:
        stations_filtered = stations_filtered[stations_filtered[Config.MUNICIPALITY_COL].isin(municipios)]
    if celdas and Config.CELL_COL in stations_filtered.columns:
        stations_filtered = stations_filtered[stations_filtered[Config.CELL_COL].isin(celdas)]
    return stations_filtered


def create_sidebar(gdf_stations, df_long):
    """
    Crea y muestra todos los widgets de la barra lateral para filtrar los datos.
    """
    with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
        min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, 0, key="min_data_perc_slider")
        altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000']
        selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges)
        
        regions_list = sorted(gdf_stations[Config.REGION_COL].dropna().unique())
        selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list)
        
        temp_gdf_for_mun = gdf_stations.copy()
        if selected_regions:
            temp_gdf_for_mun = temp_gdf_for_mun[temp_gdf_for_mun[Config.REGION_COL].isin(selected_regions)]
        
        municipios_list = sorted(temp_gdf_for_mun[Config.MUNICIPALITY_COL].dropna().unique())
        selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list)
        
        celdas_list = sorted(temp_gdf_for_mun[Config.CELL_COL].dropna().unique()) if Config.CELL_COL in temp_gdf_for_mun.columns else []
        selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list)

    # Pre-filtramos las estaciones para que el selector de estaciones sea dinámico
    gdf_filtered_for_stations = _apply_geo_filters(gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)
    stations_options = sorted(gdf_filtered_for_stations[Config.STATION_NAME_COL].unique())

    with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
        def select_all_stations():
            if st.session_state.get('select_all_checkbox_main', False):
                st.session_state.station_multiselect = stations_options
            else:
                st.session_state.station_multiselect = []
        
        st.checkbox("Seleccionar/Deseleccionar todas", key='select_all_checkbox_main', on_change=select_all_stations)
        selected_stations = st.multiselect('Seleccionar Estaciones', options=stations_options, key='station_multiselect')
        
        years_with_data = sorted(df_long[Config.YEAR_COL].dropna().unique())
        year_range_default = (min(years_with_data), max(years_with_data)) if years_with_data else (1970, 2020)
        year_range = st.slider("Rango de Años", min_value=year_range_default[0], max_value=year_range_default[1], value=year_range_default, key='year_range')
        
        meses_dict = {m: i + 1 for i, m in enumerate(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])}
        meses_nombres = st.multiselect("Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
        meses_numeros = [meses_dict[m] for m in meses_nombres]

    with st.sidebar.expander("Opciones de Preprocesamiento"):
        analysis_mode = st.radio("Modo de análisis", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode")
        exclude_na = st.checkbox("Excluir datos nulos (NaN)", key='exclude_na')
        exclude_zeros = st.checkbox("Excluir valores cero (0)", key='exclude_zeros')

    return {
        "min_data_perc": min_data_perc,
        "selected_altitudes": selected_altitudes,
        "selected_regions": selected_regions,
        "selected_municipios": selected_municipios,
        "selected_celdas": selected_celdas,
        "selected_stations": selected_stations,
        "year_range": year_range,
        "meses_numeros": meses_numeros,
        "analysis_mode": analysis_mode,
        "exclude_na": exclude_na,
        "exclude_zeros": exclude_zeros
    }
