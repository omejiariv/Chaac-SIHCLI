# Última actualización para forzar la reconstrucción del entorno

# app.py

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
import requests_cache
import time

#--- Importaciones de Módulos Propios ---
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series, extract_elevation_from_dem, download_and_load_remote_dem
from modules.visualizer import (
    display_welcome_tab, display_spatial_distribution_tab, display_graphs_tab,
    display_advanced_maps_tab, display_anomalies_tab, display_drought_analysis_tab,
    display_stats_tab, display_correlation_tab, display_enso_tab,
    display_trends_and_forecast_tab, display_downloads_tab, display_station_table_tab,
    display_forecast_tab
)
from modules.reporter import generate_pdf_report
from modules.analysis import calculate_monthly_anomalies
from modules.analysis import calculate_basin_stats
from modules.github_loader import load_csv_from_url, load_zip_from_url

#--- Desactivar Advertencias ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def apply_filters_to_stations(df, min_perc, altitudes, regions, municipios, celdas):
    """Aplica una serie de filtros al DataFrame de estaciones."""
    stations_filtered = df.copy()
    if Config.PERCENTAGE_COL in stations_filtered.columns:
        stations_filtered[Config.PERCENTAGE_COL] = pd.to_numeric(
            stations_filtered[Config.PERCENTAGE_COL].astype(str).str.replace(',', '.', regex=False), 
            errors='coerce'
        ).fillna(0)
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
            stations_filtered = stations_filtered[pd.concat(conditions, axis=1).any(axis=1)]
    if regions:
        stations_filtered = stations_filtered[stations_filtered[Config.REGION_COL].isin(regions)]
    if municipios:
        stations_filtered = stations_filtered[stations_filtered[Config.MUNICIPALITY_COL].isin(municipios)]
    if celdas and Config.CELL_COL in stations_filtered.columns:
        stations_filtered = stations_filtered[stations_filtered[Config.CELL_COL].isin(celdas)]
    return stations_filtered

# In app.py

def main():
    # --- Definiciones de Funciones Internas ---
    def process_and_store_data(file_mapa, file_precip, file_shape):
        with st.spinner("Procesando archivos y cargando datos..."):
            gdf_stations, gdf_municipios, df_long, df_enso, gdf_subcuencas = \
                load_and_process_all_data(file_mapa, file_precip, file_shape)

            if gdf_stations is not None and df_long is not None and gdf_municipios is not None:
                st.session_state.update({
                    'gdf_stations': gdf_stations, 'gdf_municipios': gdf_municipios,
                    'df_long': df_long, 'df_enso': df_enso,
                    'gdf_subcuencas': gdf_subcuencas,
                    'data_loaded': True
                })
                st.success("¡Datos cargados y listos!")
                st.rerun()
            else:
                st.error("Hubo un error al procesar los archivos.")
                st.session_state['data_loaded'] = False

    def display_map_controls(container_object, key_prefix):
        base_map_options = {
            "CartoDB Positron": {"tiles": "cartodbpositron", "attr": "CartoDB"},
            "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": "OpenStreetMap"},
            "Topografía (Open TopoMap)": {
                "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                "attr": "Open TopoMap"
            },
        }
        overlay_map_options = {
            "Mapa de Colombia (WMS IDEAM)": {
                "url": "https://geoservicios.ideam.gov.co/geoserver/ideam/wms",
                "layers": "ideam:col_admin",
                "fmt": 'image/png',
                "transparent": True,
                "attr": "IDEAM",
                "overlay": True
            }
        }
        selected_base_map_name = container_object.selectbox(
            "Seleccionar Mapa Base",
            list(base_map_options.keys()),
            key=f"{key_prefix}_base_map"
        )
        selected_overlays_names = container_object.multiselect(
            "Seleccionar Capas Adicionales",
            list(overlay_map_options.keys()),
            key=f"{key_prefix}_overlays"
        )
        selected_base_map_config = base_map_options[selected_base_map_name]
        selected_overlays_config = [overlay_map_options[name] for name in selected_overlays_names]
        return selected_base_map_config, selected_overlays_config

    # --- Inicio de la Ejecución de la App ---
    st.set_page_config(layout="wide", page_title=Config.APP_TITLE)
    st.markdown("""<style>div.block-container{padding-top:1rem;} [data-testid="stMetricValue"] {font-size:1.8rem;} [data-testid="stMetricLabel"] {font-size: 1rem; padding-bottom:5px; } button [data-baseweb="tab"] {font-size:16px;font-weight:bold;color:#333;}</style>""", unsafe_allow_html=True)
    Config.initialize_session_state()

    # --- Guía Interactiva ---
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
    if not st.session_state.get('data_loaded', False) and st.session_state.first_visit:
        st.toast("¡Bienvenido a Chaac SIHCLI! 👋")
        time.sleep(1.5)
        st.toast("Para empezar, carga tus datos desde GitHub...")
        time.sleep(2)
        st.toast("...o sube tus archivos manualmente en el panel de la izquierda. 👈")
        st.session_state.first_visit = False

    progress_placeholder = st.empty()
    title_col1, title_col2 = st.columns([0.05, 0.95])
    with title_col1:
        if os.path.exists(Config.LOGO_PATH):
            st.image(Config.LOGO_PATH, width=60)
    with title_col2:
        st.markdown(f'<h1 style="font-size:28px; margin-top:1rem;">{Config.APP_TITLE}</h1>', unsafe_allow_html=True)

    # --- Panel de Control (Sidebar) ---
    st.sidebar.header("Panel de Control")
    with st.sidebar.expander("**Subir/Actualizar Archivos Base**", expanded=not st.session_state.get('data_loaded', False)):
        load_mode = st.radio("Modo de Carga", ("GitHub", "Manual"), key="load_mode", horizontal=True)
        if load_mode == "Manual":
            uploaded_file_mapa = st.file_uploader("1. Archivo de estaciones (CSV)", type="csv")
            uploaded_file_precip = st.file_uploader("2. Archivo de precipitación (CSV)", type="csv")
            uploaded_zip_shapefile = st.file_uploader("3. Shapefile de municipios (.zip)", type="zip")
            if st.button("Procesar Datos Manuales"):
                if all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
                    process_and_store_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
                else:
                    st.warning("Por favor, suba los tres archivos requeridos.")
        else:
            st.info(f"Datos desde: **{Config.GITHUB_USER}/{Config.GITHUB_REPO}**")
            if st.button("Cargar Datos desde GitHub"):
                with st.spinner("Descargando archivos..."):
                    github_files = {
                        'mapa': load_csv_from_url(Config.URL_ESTACIONES_CSV),
                        'precip': load_csv_from_url(Config.URL_PRECIPITACION_CSV),
                        'shape': load_zip_from_url(Config.URL_SHAPEFILE_ZIP)
                    }
                    if all(github_files.values()):
                        process_and_store_data(github_files['mapa'], github_files['precip'], github_files['shape'])
                    else:
                        st.error("No se pudieron descargar los archivos desde GitHub.")

    if not st.session_state.get('data_loaded', False):
        display_welcome_tab()
        st.warning("Para comenzar, cargue los datos usando el panel de la izquierda.")
        return

    st.sidebar.success("Datos cargados.")
    if st.sidebar.button("Limpiar Caché y Reiniciar"):
        st.cache_data.clear()
        st.cache_resource.clear()
        requests_cache.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
        min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, st.session_state.get('min_data_perc_slider', 0))
        altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000']
        selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges)
        regions_list = sorted(st.session_state.gdf_stations[Config.REGION_COL].dropna().unique())
        selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list, key='regions_multiselect')
        temp_gdf_for_mun = st.session_state.gdf_stations.copy()
        if selected_regions:
            temp_gdf_for_mun = temp_gdf_for_mun[temp_gdf_for_mun[Config.REGION_COL].isin(selected_regions)]
        municipios_list = sorted(temp_gdf_for_mun[Config.MUNICIPALITY_COL].dropna().unique())
        selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list, key='municipios_multiselect')
        celdas_list = sorted(temp_gdf_for_mun[Config.CELL_COL].dropna().unique()) if Config.CELL_COL in temp_gdf_for_mun.columns else []
        selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, key='celdas_multiselect')
        gdf_filtered = apply_filters_to_stations(st.session_state.gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)


    with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
        stations_options = sorted(gdf_filtered[Config.STATION_NAME_COL].unique())
        def select_all_stations():
            if st.session_state.get('select_all_checkbox_main', False):
                st.session_state.station_multiselect = stations_options
            else:
                st.session_state.station_multiselect = []
        st.checkbox("Seleccionar/Deseleccionar todas", key='select_all_checkbox_main', on_change=select_all_stations)
        selected_stations = st.multiselect('Seleccionar Estaciones', options=stations_options, key='station_multiselect')
        years_with_data = sorted(st.session_state.df_long[Config.YEAR_COL].dropna().unique())
        year_range_default = (min(years_with_data), max(years_with_data)) if years_with_data else (1970, 2020)
        year_range = st.slider("Rango de Años", min_value=year_range_default[0], max_value=year_range_default[1], value=st.session_state.get('year_range', year_range_default), key='year_range')
        meses_dict = {m: i + 1 for i, m in enumerate(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])}
        meses_nombres = st.multiselect("Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
        meses_numeros = [meses_dict[m] for m in meses_nombres]

    with st.sidebar.expander("Opciones de Preprocesamiento"):
        st.radio("Modo de análisis", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode")
        st.checkbox("Excluir datos nulos (NaN)", key='exclude_na')
        st.checkbox("Excluir valores cero (0)", key='exclude_zeros')

    # --- Definición de Pestañas ---
    tab_names = [
        "Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados",
        "Análisis de Anomalías", "Análisis de Extremos", "Estadísticas",
        "Correlación", "Análisis ENSO", "Tendencias y Pronósticos",
        "Pronóstico del Tiempo", "Descargas", "Análisis por Cuenca",
        "Comparación de Periodos",  # <--- AÑADE ESTA NUEVA PESTAÑA
        "Tabla de Estaciones", "Generar Reporte"
    ]
    tabs = st.tabs(tab_names)

    stations_for_analysis = selected_stations

    if not stations_for_analysis:
        with tabs[0]:
            display_welcome_tab()
        st.info("Para comenzar, seleccione al menos una estación en el panel de la izquierda.")
        for tab in tabs[1:]:
            with tab:
                st.info("Seleccione al menos una estación para ver el contenido.")
        return

    # --- Procesamiento de Datos Post-Filtros ---
    df_monthly_filtered = st.session_state.df_long[
        (st.session_state.df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
        (st.session_state.df_long[Config.DATE_COL].dt.year >= year_range[0]) &
        (st.session_state.df_long[Config.DATE_COL].dt.year <= year_range[1]) &
        (st.session_state.df_long[Config.DATE_COL].dt.month.isin(meses_numeros))
    ].copy()

    if st.session_state.analysis_mode == "Completar series (interpolación)":
        bar = progress_placeholder.progress(0, text="Iniciando interpolación...")
        df_monthly_filtered = complete_series(df_monthly_filtered, _progress_bar=bar)
        progress_placeholder.empty()

    if st.session_state.get('exclude_na', False):
        df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
    if st.session_state.get('exclude_zeros', False):
        df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] > 0]

    annual_agg = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).agg(
        precipitation_sum=(Config.PRECIPITATION_COL, 'sum'),
        meses_validos=(Config.MONTH_COL, 'nunique')
    ).reset_index()
    annual_agg.loc[annual_agg['meses_validos'] < 10, 'precipitation_sum'] = np.nan
    df_anual_melted = annual_agg.rename(columns={'precipitation_sum': Config.PRECIPITATION_COL})

    display_args = {
        "gdf_filtered": gdf_filtered, "stations_for_analysis": stations_for_analysis,
        "df_anual_melted": df_anual_melted, "df_monthly_filtered": df_monthly_filtered,
        "analysis_mode": st.session_state.analysis_mode, "selected_regions": selected_regions,
        "selected_municipios": selected_municipios, "selected_altitudes": selected_altitudes
    }

    # --- Renderizado de Pestañas ---
    with tabs[0]: display_welcome_tab()
    with tabs[1]: display_spatial_distribution_tab(**display_args)
    with tabs[2]: display_graphs_tab(**display_args)
    with tabs[3]: display_advanced_maps_tab(**display_args)
    with tabs[4]: display_anomalies_tab(df_long=st.session_state.df_long, **display_args)
    with tabs[5]: display_drought_analysis_tab(**display_args)
    with tabs[6]: display_stats_tab(df_long=st.session_state.df_long, **display_args)
    with tabs[7]: display_correlation_tab(**display_args)
    with tabs[8]: display_enso_tab(df_enso=st.session_state.df_enso, **display_args)
    with tabs[9]: display_trends_and_forecast_tab(df_full_monthly=st.session_state.df_long, **display_args)
    with tabs[10]: display_forecast_tab(**display_args)
    with tabs[11]: display_downloads_tab(
        df_anual_melted=df_anual_melted, df_monthly_filtered=df_monthly_filtered,
        stations_for_analysis=stations_for_analysis,
        analysis_mode=st.session_state.analysis_mode
    )

    # --- PESTAÑA NUEVA: ANÁLISIS POR CUENCA ---
    with tabs[12]:
        st.header("Análisis Agregado por Cuenca Hidrográfica")
        if st.session_state.gdf_subcuencas is not None and not st.session_state.gdf_subcuencas.empty:
            BASIN_NAME_COLUMN = 'SUBC_LBL'
            if BASIN_NAME_COLUMN in st.session_state.gdf_subcuencas.columns:
                relevant_basins_gdf = gpd.sjoin(st.session_state.gdf_subcuencas, gdf_filtered, how="inner", predicate="intersects")
                if not relevant_basins_gdf.empty:
                    basin_names = sorted(relevant_basins_gdf[BASIN_NAME_COLUMN].dropna().unique())
                else:
                    basin_names = []

                if not basin_names:
                    st.info("Ninguna cuenca contiene estaciones que coincidan con los filtros actuales.")
                else:
                    selected_basin = st.selectbox(
                        "Seleccione una cuenca para analizar:",
                        options=basin_names,
                        key="basin_selector"
                    )
                    if selected_basin:
                        stats_df, stations, error_msg = calculate_basin_stats(
                            gdf_filtered,
                            st.session_state.gdf_subcuencas,
                            df_monthly_filtered,
                            selected_basin,
                            BASIN_NAME_COLUMN
                        )
                        if error_msg: st.warning(error_msg)
                        if stations:
                            st.subheader(f"Resultados para la cuenca: {selected_basin}")
                            st.metric("Número de Estaciones en la Cuenca", len(stations))
                            with st.expander("Ver estaciones incluidas"): st.write(", ".join(stations))
                            if not stats_df.empty:
                                st.markdown("---")
                                st.write("**Estadísticas de Precipitación Mensual (Agregada)**")
                                st.dataframe(stats_df, use_container_width=True)
                            else:
                                st.info("Aunque se encontraron estaciones, no hay datos de precipitación para el período seleccionado.")
                        else:
                            st.warning("No se encontraron estaciones dentro de la cuenca seleccionada.")
            else:
                st.error(f"Error Crítico: No se encontró la columna de nombres '{BASIN_NAME_COLUMN}' en el archivo de subcuencas.")
        else:
            st.warning("Los datos de las subcuencas no están cargados.")

    # --- PESTAÑA: COMPARACIÓN DE PERIODOS (VERSIÓN MEJORADA) ---
    with tabs[13]: # Ajusta el índice si es necesario
        st.header("Comparación de Periodos de Tiempo")

        # --- NIVEL DE ANÁLISIS ---
        analysis_level = st.radio(
            "Seleccione el nivel de análisis para la comparación:",
            ("Promedio Regional (Todas las estaciones seleccionadas)", "Por Cuenca Específica"),
            key="compare_level_radio"
        )

        df_to_compare = pd.DataFrame()

        # --- LÓGICA DE FILTRADO ---
        if analysis_level == "Por Cuenca Específica":
            st.markdown("---")
            if st.session_state.gdf_subcuencas is not None and not st.session_state.gdf_subcuencas.empty:
                BASIN_NAME_COLUMN = 'SUBC_LBL'
                if BASIN_NAME_COLUMN in st.session_state.gdf_subcuencas.columns:
                    relevant_basins_gdf = gpd.sjoin(st.session_state.gdf_subcuencas, gdf_filtered, how="inner", predicate="intersects")
                    if not relevant_basins_gdf.empty:
                        basin_names = sorted(relevant_basins_gdf[BASIN_NAME_COLUMN].dropna().unique())
                    else:
                        basin_names = []

                    if not basin_names:
                        st.warning("Ninguna cuenca contiene estaciones que coincidan con los filtros actuales.", icon="⚠️")
                    else:
                        selected_basin = st.selectbox(
                            "Seleccione la cuenca a comparar:",
                            options=basin_names,
                            key="compare_basin_selector"
                        )
                        # Filtramos las estaciones que están dentro de la cuenca seleccionada
                        target_basin_geom = st.session_state.gdf_subcuencas[st.session_state.gdf_subcuencas[BASIN_NAME_COLUMN] == selected_basin]
                        stations_in_basin = gpd.sjoin(gdf_filtered, target_basin_geom, how="inner", predicate="within")
                        station_names_in_basin = stations_in_basin[Config.STATION_NAME_COL].unique().tolist()
                        
                        # Usamos solo los datos de las estaciones de esa cuenca
                        df_to_compare = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL].isin(station_names_in_basin)]
                        st.info(f"Análisis para **{len(station_names_in_basin)}** estaciones encontradas en la cuenca **{selected_basin}**.", icon="🏞️")

                else:
                    st.error(f"Error Crítico: No se encontró la columna de nombres '{BASIN_NAME_COLUMN}' en el archivo de subcuencas.")
            else:
                st.warning("Los datos de las subcuencas no están cargados.", icon="⚠️")
        
        else: # Promedio Regional
            df_to_compare = df_monthly_filtered
        
        st.markdown("---")
        
        # --- LÓGICA DE COMPARACIÓN (SIN CAMBIOS) ---
        if df_to_compare.empty:
            st.warning("Seleccione una cuenca con estaciones para poder realizar la comparación.", icon="👇")
        else:
            years_with_data = sorted(df_to_compare[Config.YEAR_COL].dropna().unique())
            min_year, max_year = int(years_with_data[0]), int(years_with_data[-1])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Periodo 1")
                periodo1 = st.slider(
                    "Seleccione el rango de años para el Periodo 1",
                    min_year, max_year,
                    (min_year, min_year + 10 if min_year + 10 < max_year else max_year),
                    key="periodo1_slider_comp"
                )
            
            with col2:
                st.markdown("#### Periodo 2")
                periodo2 = st.slider(
                    "Seleccione el rango de años para el Periodo 2",
                    min_year, max_year,
                    (max_year - 10 if max_year - 10 > min_year else min_year, max_year),
                    key="periodo2_slider_comp"
                )

            df_periodo1 = df_to_compare[
                (df_to_compare[Config.DATE_COL].dt.year >= periodo1[0]) &
                (df_to_compare[Config.DATE_COL].dt.year <= periodo1[1])
            ]
            df_periodo2 = df_to_compare[
                (df_to_compare[Config.DATE_COL].dt.year >= periodo2[0]) &
                (df_to_compare[Config.DATE_COL].dt.year <= periodo2[1])
            ]

            st.markdown("---")
            st.subheader("Resultados Comparativos")

            if df_periodo1.empty or df_periodo2.empty:
                st.warning("Uno o ambos periodos seleccionados no contienen datos. Por favor, ajuste los rangos.", icon=" ADJUST RANGES")
            else:
                stats1_mean = df_periodo1[Config.PRECIPITATION_COL].mean()
                stats2_mean = df_periodo2[Config.PRECIPITATION_COL].mean()
                delta = ((stats2_mean - stats1_mean) / stats1_mean) * 100 if stats1_mean != 0 else 0

                st.metric(
                    label=f"Precipitación Media Mensual ({periodo1[0]}-{periodo1[1]} vs. {periodo2[0]}-{periodo2[1]})",
                    value=f"{stats2_mean:.1f} mm",
                    delta=f"{delta:.2f}% (respecto a {stats1_mean:.1f} mm del Periodo 1)"
                )
                
                st.markdown("##### Desglose Estadístico Completo")
                col1_stats, col2_stats = st.columns(2)
                with col1_stats:
                    st.write(f"**Periodo 1 ({periodo1[0]}-{periodo1[1]})**")
                    st.dataframe(df_periodo1[Config.PRECIPITATION_COL].describe().round(2))
                
                with col2_stats:
                    st.write(f"**Periodo 2 ({periodo2[0]}-{periodo2[1]})**")
                    st.dataframe(df_periodo2[Config.PRECIPITATION_COL].describe().round(2))
    
    with tabs[14]: display_station_table_tab(**display_args)

    with tabs[15]:
        st.header("Generación de Reporte PDF")
        report_title = st.text_input("Título del Reporte:", value="Análisis Hidroclimático")
        report_sections_options = [
            "Resumen Ejecutivo", "Tabla de Estaciones", "Distribución Espacial",
            "Gráficos de Series Temporales", "Mapas Avanzados de Interpolación",
            "Análisis de Anomalías", "Análisis de Extremos Hidrológicos",
            "Estadísticas Descriptivas", "Análisis de Correlación",
            "Análisis de El Niño/La Niña (ENSO)", "Análisis de Tendencias y Pronósticos",
            "Disponibilidad de Datos", "Metodología y Fuentes de Datos"
        ]
        st.markdown("**Seleccione las secciones a incluir:**")
        def select_all_report_sections_on_change():
            if st.session_state.select_all_report_sections_checkbox:
                st.session_state.selected_report_sections_multiselect = report_sections_options
            else:
                st.session_state.selected_report_sections_multiselect = []
        st.checkbox(
            "Seleccionar/Deseleccionar todas",
            key='select_all_report_sections_checkbox',
            on_change=select_all_report_sections_on_change
        )
        selected_report_sections = st.multiselect(
            "Secciones a incluir:",
            options=report_sections_options,
            key='selected_report_sections_multiselect'
        )
        if st.button("Generar Reporte PDF"):
            if not selected_report_sections:
                st.warning("Seleccione al menos una sección.")
            else:
                with st.spinner("Generando reporte..."):
                    try:
                        summary_data = {
                            "Estaciones": f"{len(stations_for_analysis)}/{len(st.session_state.gdf_stations)}",
                            "Periodo": f"{year_range[0]}-{year_range[1]}",
                            "Modo de Análisis": st.session_state.analysis_mode
                        }
                        df_anomalies = calculate_monthly_anomalies(df_monthly_filtered, st.session_state.df_long)
                        report_pdf_bytes = generate_pdf_report(
                            report_title=report_title,
                            sections_to_include=selected_report_sections,
                            **display_args,
                            summary_data=summary_data,
                            df_anomalies=df_anomalies
                        )
                        st.download_button(
                            label="Descargar Reporte PDF",
                            data=report_pdf_bytes,
                            file_name=f"{report_title.replace(' ', '_')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("¡Reporte generado!")
                    except Exception as e:
                        st.error(f"Error al generar el reporte: {e}")
                        st.exception(e)

if __name__ == "__main__":
    main()
