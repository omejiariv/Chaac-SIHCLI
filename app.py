# app.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import pymannkendall as mk
import plotly.graph_objects as go
from scipy import stats

#--- Importaciones de Módulos
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
from modules.github_loader import load_csv_from_url, load_zip_from_url

#--- Desactivar Advertencias
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def apply_filters_to_stations(df, min_perc, altitudes, regions, municipios, celdas):
    stations_filtered = df.copy()
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
            stations_filtered = stations_filtered[pd.concat(conditions, axis=1).any(axis=1)]
    if regions:
        stations_filtered = stations_filtered[stations_filtered[Config.REGION_COL].isin(regions)]
    if municipios:
        stations_filtered = stations_filtered[stations_filtered[Config.MUNICIPALITY_COL].isin(municipios)]
    if celdas and Config.CELL_COL in stations_filtered.columns:
        stations_filtered = stations_filtered[stations_filtered[Config.CELL_COL].isin(celdas)]
    return stations_filtered

def main():
    st.set_page_config(layout="wide", page_title=Config.APP_TITLE)
    progress_placeholder = st.empty()
    st.markdown("""<style>div.block-container{padding-top:1rem;} [data-testid="stMetricValue"] {font-size:1.8rem;} [data-testid="stMetricLabel"] {font-size: 1rem; padding-bottom:5px; } button[data-baseweb="tab"] {font-size:16px;font-weight:bold;color:#333;}</style>""", unsafe_allow_html=True)
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
        load_mode = st.radio("Modo de Carga de Archivos", ("Automático (desde GitHub)", "Manual (Subir archivos)"), key="load_mode", horizontal=True)
        def process_and_store_data(file_mapa, file_precip, file_shape):
            st.cache_data.clear()
            st.cache_resource.clear()
            for key in list(st.session_state.keys()): del st.session_state[key]
            Config.initialize_session_state()
            with st.spinner("Procesando archivos y cargando datos..."):
                gdf_stations, gdf_municipios, df_long, df_enso = load_and_process_all_data(file_mapa, file_precip, file_shape)
                if gdf_stations is not None and df_long is not None and gdf_municipios is not None:
                    st.session_state.gdf_stations, st.session_state.gdf_municipios, st.session_state.df_long, st.session_state.df_enso = gdf_stations, gdf_municipios, df_long, df_enso
                    st.session_state.data_loaded = True
                    st.success("¡Datos cargados y listos!")
                    st.rerun()
                else: st.error("Hubo un error al procesar los archivos. Verifique el formato y contenido.")
        if load_mode == "Manual (Subir archivos)":
            uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (CSV)", type="csv", key='uploaded_file_mapa')
            uploaded_file_precip = st.file_uploader("2. Cargar archivo de precipitación (CSV)", type="csv", key='uploaded_file_precip')
            uploaded_zip_shapefile = st.file_uploader("3. Cargar shapefile de municipios (.zip)", type="zip", key='uploaded_zip_shapefile')
            if st.button("Procesar Datos Manuales", key='process_data_manual_button'):
                if all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
                    process_and_store_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
                else: st.warning("Por favor, suba los tres archivos requeridos.")
        else:
            st.info(f"Los datos se cargarán desde el repositorio de GitHub: **{Config.GITHUB_USER}/{Config.GITHUB_REPO}**")
            if st.button("Cargar Datos desde GitHub", key='process_data_github_button'):
                with st.spinner("Descargando archivos desde GitHub..."):
                    github_file_mapa = load_csv_from_url(Config.URL_ESTACIONES_CSV)
                    github_file_precip = load_csv_from_url(Config.URL_PRECIPITACION_CSV)
                    github_file_shape = load_zip_from_url(Config.URL_SHAPEFILE_ZIP)
                if all([github_file_mapa, github_file_precip, github_file_shape]):
                    process_and_store_data(github_file_mapa, github_file_precip, github_file_shape)
                else: st.error("No se pudieron descargar uno o más archivos desde GitHub. Revisa las URLs en config.py y que el repositorio sea público.")

    if not st.session_state.get('data_loaded', False):
        display_welcome_tab()
        st.info("Para comenzar, cargue los archivos requeridos en el panel de la izquierda.")
        return

    st.sidebar.success("Datos base cargados.")
    if st.sidebar.button("Limpiar Caché y Reiniciar App"):
        for key in list(st.session_state.keys()): del st.session_state[key]
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
        celdas_list = sorted(temp_gdf_for_mun[Config.CELL_COL].dropna().unique()) if Config.CELL_COL in temp_gdf_for_mun.columns else []
        selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, key='celdas_multiselect')

    gdf_filtered = apply_filters_to_stations(st.session_state.gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)

    with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
        stations_options = sorted(gdf_filtered[Config.STATION_NAME_COL].unique())
        if 'select_all_prev_state' not in st.session_state:
            st.session_state.select_all_prev_state = False
        select_all_checkbox = st.checkbox("Seleccionar/Deseleccionar todas las estaciones", value=st.session_state.select_all_prev_state, key='select_all_checkbox_main')
        if select_all_checkbox != st.session_state.select_all_prev_state:
            st.session_state.station_multiselect = stations_options if select_all_checkbox else []
            st.session_state.select_all_prev_state = select_all_checkbox
            st.rerun()
        selected_stations = st.multiselect('Seleccionar Estaciones', options=stations_options, key='station_multiselect')
        years_with_data = sorted(st.session_state.df_long[Config.YEAR_COL].dropna().unique())
        year_range_default = (min(years_with_data), max(years_with_data)) if years_with_data else (1970, 2020)
        year_range = st.slider("Seleccionar Rango de Años", min_value=year_range_default[0], max_value=year_range_default[1], value=st.session_state.get('year_range', year_range_default), key='year_range')
        meses_dict = {m: i + 1 for i, m in enumerate(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])}
        meses_nombres = st.multiselect("Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()), key='meses_nombres')
        meses_numeros = [meses_dict[m] for m in meses_nombres]

    with st.sidebar.expander("Opciones de Preprocesamiento"):
        st.radio("Modo de análisis", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode", help="La opción 'Completar series' utiliza interpolación para rellenar los datos faltantes. Afecta a todas las pestañas de análisis y a las descargas.")
        st.checkbox("Excluir datos nulos (NaN)", key='exclude_na')
        st.checkbox("Excluir valores cero (0)", key='exclude_zeros')

    with st.sidebar.expander("Opciones de Elevación (DEM)"):
        dem_source_option = st.radio("Fuente del Modelo de Elevación Digital (DEM)", ["No usar DEM", "Subir archivo GeoTIFF", "Cargar desde URL"], key='dem_source_radio')
        if dem_source_option == "Subir archivo GeoTIFF":
            uploaded_dem_file = st.file_uploader("Cargar archivo .tif", type=['tif', 'tiff'], key='dem_uploader')
            if uploaded_dem_file: st.session_state.gdf_stations = extract_elevation_from_dem(st.session_state.gdf_stations, uploaded_dem_file)
        elif dem_source_option == "Cargar desde URL":
            dem_url = st.text_input("URL del archivo GeoTIFF remoto", value=Config.DEM_SERVER_URL)
            if st.button("Descargar y aplicar DEM"):
                with st.spinner("Descargando DEM remoto..."):
                    try:
                        remote_dem_data = download_and_load_remote_dem(dem_url)
                        st.session_state.gdf_stations = extract_elevation_from_dem(st.session_state.gdf_stations, remote_dem_data)
                    except Exception as e: st.error(f"No se pudo cargar el DEM desde la URL: {e}")

    tab_names = ["Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados", "Análisis de Anomalías", "Análisis de extremos hid", "Estadísticas", "Análisis de Correlación", "Análisis ENSO", "Tendencias y Pronósticos", "Pronóstico del Tiempo", "Descargas", "Tabla de Estaciones", "Generar Reporte"]
    tabs = st.tabs(tab_names)
    stations_for_analysis = selected_stations

    if not stations_for_analysis:
        with tabs[0]:
            display_welcome_tab()
            st.info("Para comenzar, cargue los archivos requeridos en el panel de la izquierda y seleccione estaciones.")
        return

    gdf_filtered = gdf_filtered[gdf_filtered[Config.STATION_NAME_COL].isin(stations_for_analysis)]
    df_monthly_filtered = st.session_state.df_long[(st.session_state.df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) & (st.session_state.df_long[Config.DATE_COL].dt.year >= year_range[0]) & (st.session_state.df_long[Config.DATE_COL].dt.year <= year_range[1]) & (st.session_state.df_long[Config.DATE_COL].dt.month.isin(meses_numeros))].copy()
    if st.session_state.analysis_mode == "Completar series (interpolación)":
        bar = progress_placeholder.progress(0, text="Iniciando interpolación...")
        df_monthly_filtered = complete_series(df_monthly_filtered, _progress_bar=bar)
        progress_placeholder.empty()
    if st.session_state.get('exclude_na', False): df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
    if st.session_state.get('exclude_zeros', False): df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] > 0]
    
    annual_agg = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).agg(precipitation_sum=(Config.PRECIPITATION_COL, 'sum'), meses_validos=(Config.MONTH_COL, 'nunique')).reset_index()
    annual_agg.loc[annual_agg['meses_validos'] < 10, 'precipitation_sum'] = np.nan
    df_anual_melted = annual_agg.rename(columns={'precipitation_sum': Config.PRECIPITATION_COL})

    display_args = {"gdf_filtered": gdf_filtered, "stations_for_analysis": stations_for_analysis, "df_anual_melted": df_anual_melted, "df_monthly_filtered": df_monthly_filtered, "analysis_mode": st.session_state.analysis_mode, "selected_regions": selected_regions, "selected_municipios": selected_municipios, "selected_altitudes": selected_altitudes}
    
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
    with tabs[11]: 
        display_downloads_tab(
            df_anual_melted=df_anual_melted,
            df_monthly_filtered=df_monthly_filtered,
            stations_for_analysis=stations_for_analysis,
            analysis_mode=st.session_state.analysis_mode
        )
    with tabs[12]: display_station_table_tab(**display_args)
    with tabs[13]:
        st.header("Generación de Reporte PDF")
        
        # Opciones para el reporte
        report_title = st.text_input("Título del Reporte:", value="Análisis Hidroclimático de Estaciones Seleccionadas")
        st.markdown("**Seleccione las secciones a incluir:**")
        
        # Checkbox para seleccionar todas las secciones
        select_all_report_sections = st.checkbox("Seleccionar/Deseleccionar todas las secciones", value=st.session_state.get('select_all_report_sections_default', False), key='select_all_report_sections')
        
        # Lista de secciones para el reporte
        report_sections_options = [
            "Resumen Ejecutivo", "Tabla de Estaciones", "Distribución Espacial",
            "Gráficos de Series Temporales", "Mapas Avanzados de Interpolación",
            "Análisis de Anomalías", "Análisis de Extremos Hidrológicos",
            "Estadísticas Descriptivas", "Análisis de Correlación",
            "Análisis de El Niño/La Niña (ENSO)", "Análisis de Tendencias y Pronósticos",
            "Disponibilidad de Datos", "Metodología y Fuentes de Datos"
        ]
        
        # Actualizar selección de secciones si el checkbox "seleccionar todas" cambia
        if select_all_report_sections != st.session_state.get('select_all_report_sections_default', False):
            st.session_state.selected_report_sections = report_sections_options if select_all_report_sections else []
            st.session_state.select_all_report_sections_default = select_all_report_sections
            st.rerun()

        # Multiselect para las secciones del reporte
        selected_report_sections = st.multiselect(
            "Secciones del Reporte",
            options=report_sections_options,
            default=st.session_state.get('selected_report_sections', report_sections_options),
            key='selected_report_sections_multiselect'
        )

        st.session_state.selected_report_sections = selected_report_sections # Asegura que el estado de la sesión se actualice

        if st.button("Generar Reporte PDF", key="generate_report_button"):
            if not stations_for_analysis:
                st.warning("Seleccione al menos una estación para generar el reporte.")
            elif not selected_report_sections:
                st.warning("Seleccione al menos una sección para incluir en el reporte.")
            else:
                with st.spinner("Generando reporte PDF..."):
                    try:
                        report_pdf_bytes = generate_pdf_report(
                            report_title=report_title,
                            selected_sections=selected_report_sections,
                            gdf_stations=st.session_state.gdf_stations,
                            gdf_municipios=st.session_state.gdf_municipios,
                            df_long=st.session_state.df_long,
                            df_enso=st.session_state.df_enso,
                            gdf_filtered=gdf_filtered,
                            stations_for_analysis=stations_for_analysis,
                            df_anual_melted=df_anual_melted,
                            df_monthly_filtered=df_monthly_filtered,
                            analysis_mode=st.session_state.analysis_mode,
                            selected_regions=selected_regions,
                            selected_municipios=selected_municipios,
                            selected_altitudes=selected_altitudes,
                            year_range=year_range,
                            meses_numeros=meses_numeros
                        )
                        st.download_button(
                            label="Descargar Reporte PDF",
                            data=report_pdf_bytes,
                            file_name=f"{report_title.replace(' ', '_')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("¡Reporte generado con éxito!")
                    except Exception as e:
                        st.error(f"Error al generar el reporte: {e}")
                        st.exception(e) # Muestra el traceback completo para depuración


if __name__ == "__main__":
    main()
