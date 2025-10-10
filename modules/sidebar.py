# modules/sidebar.py

import streamlit as st
from modules.config import Config

def create_sidebar(gdf_stations):
    """
    Crea y muestra todos los widgets de la barra lateral para filtrar los datos.
    Esta función solo se debe llamar DESPUÉS de que los datos han sido cargados.

    Args:
        gdf_stations (GeoDataFrame): El GeoDataFrame con la información de las estaciones.

    Returns:
        dict: Un diccionario con todos los valores seleccionados en los filtros.
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

    with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
        # Primero filtramos las estaciones con los filtros geográficos para el dropdown
        from app import get_filtered_data # Importación local para evitar importación circular
        gdf_filtered_for_stations = get_filtered_data(gdf_stations, min_data_perc, tuple(selected_altitudes), tuple(selected_regions), tuple(selected_municipios), tuple(selected_celdas))
        stations_options = sorted(gdf_filtered_for_stations[Config.STATION_NAME_COL].unique())
        
        def select_all_stations():
            if st.session_state.get('select_all_checkbox_main', False):
                st.session_state.station_multiselect = stations_options
            else:
                st.session_state.station_multiselect = []
        
        st.checkbox("Seleccionar/Deseleccionar todas", key='select_all_checkbox_main', on_change=select_all_stations)
        selected_stations = st.multiselect('Seleccionar Estaciones', options=stations_options, key='station_multiselect')
        
        years_with_data = sorted(st.session_state.df_long[Config.YEAR_COL].dropna().unique())
        year_range_default = (min(years_with_data), max(years_with_data)) if years_with_data else (1970, 2020)
        year_range = st.slider("Rango de Años", min_value=year_range_default[0], max_value=year_range_default[1], value=year_range_default, key='year_range')
        
        meses_dict = {m: i + 1 for i, m in enumerate(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])}
        meses_nombres = st.multiselect("Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
        meses_numeros = [meses_dict[m] for m in meses_nombres]

    with st.sidebar.expander("Opciones de Preprocesamiento"):
        analysis_mode = st.radio("Modo de análisis", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode")
        exclude_na = st.checkbox("Excluir datos nulos (NaN)", key='exclude_na')
        exclude_zeros = st.checkbox("Excluir valores cero (0)", key='exclude_zeros')

    # Devolvemos un diccionario con todos los valores seleccionados
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
