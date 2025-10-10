# app.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
import os
from datetime import datetime
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from modules.db_manager import create_table

# --- Importaciones de Módulos Propios ---
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series
from modules.sidebar import create_sidebar # ¡NUEVA IMPORTACIÓN!
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

#--- Desactivar Advertencias ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- FUNCIONES CON CACHÉ PARA ESTABILIDAD Y RENDIMIENTO ---

@st.cache_data
def get_filtered_data(_gdf_stations, min_perc, altitudes, regions, municipios, celdas):
    """Aplica filtros al DataFrame de estaciones de forma optimizada."""
    stations_filtered = _gdf_stations.copy()
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

@st.cache_data
def get_processed_time_series(_df_long, stations_for_analysis, year_range, meses_numeros, analysis_mode, exclude_na, exclude_zeros):
    """Procesa los datos de series de tiempo de forma optimizada."""
    df_monthly = _df_long[
        (_df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
        (_df_long[Config.DATE_COL].dt.year >= year_range[0]) &
        (_df_long[Config.DATE_COL].dt.year <= year_range[1]) &
        (_df_long[Config.DATE_COL].dt.month.isin(meses_numeros))
    ].copy()

    if analysis_mode == "Completar series (interpolación)":
        bar = st.progress(0, text="Iniciando interpolación...")
        df_monthly = complete_series(df_monthly, _progress_bar=bar)
        bar.empty()

    if exclude_na:
        df_monthly.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
    if exclude_zeros:
        df_monthly = df_monthly[df_monthly[Config.PRECIPITATION_COL] > 0]
    
    annual_agg = df_monthly.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).agg(
        precipitation_sum=(Config.PRECIPITATION_COL, 'sum'),
        meses_validos=(Config.MONTH_COL, 'nunique')
    ).reset_index()
    annual_agg.loc[annual_agg['meses_validos'] < 10, 'precipitation_sum'] = np.nan
    df_anual = annual_agg.rename(columns={'precipitation_sum': Config.PRECIPITATION_COL})
    
    return df_monthly, df_anual

def main():
    st.set_page_config(layout="wide", page_title=Config.APP_TITLE)
    create_table()

    try:
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
    except FileNotFoundError:
        st.error("Error: El archivo 'config.yaml' no se encontró. Por favor, créelo para configurar los usuarios.")
        return

    authenticator = stauth.Authenticate(
        config['credentials'], config['cookie']['name'], config['cookie']['key'],
        config['cookie']['expiry_days'], config['preauthorized']
    )

    name, authentication_status, username = authenticator.login('main')

    if st.session_state.get("authentication_status"):
        st.sidebar.write(f'Bienvenido *{st.session_state["name"]}*')
        authenticator.logout('Logout', 'sidebar')

        title_col1, title_col2 = st.columns([0.05, 0.95])
        with title_col1:
            if os.path.exists(Config.LOGO_PATH):
                st.image(Config.LOGO_PATH, width=60)
        with title_col2:
            st.markdown(f'<h1 style="font-size:28px; margin-top:1rem;">{Config.APP_TITLE}</h1>', unsafe_allow_html=True)
        
        st.sidebar.header("Panel de Control")
        
        with st.sidebar.expander("**Subir/Actualizar Archivos Base**", expanded=not st.session_state.get('data_loaded', False)):
            load_mode = st.radio("Modo de Carga", ("GitHub", "Manual"), key="load_mode", horizontal=True)

            def process_and_store_data(file_mapa, file_precip, file_shape):
                st.cache_data.clear()
                st.cache_resource.clear()
                auth_info = {
                    "authentication_status": st.session_state.get("authentication_status"),
                    "name": st.session_state.get("name"),
                    "username": st.session_state.get("username")
                }
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                for key, value in auth_info.items():
                    st.session_state[key] = value

                Config.initialize_session_state()
                
                with st.spinner("Procesando archivos y cargando datos..."):
                    gdf_stations, gdf_municipios, df_long, df_enso = load_and_process_all_data(file_mapa, file_precip, file_shape)
                    if gdf_stations is not None and df_long is not None and gdf_municipios is not None:
                        st.session_state.update({
                            'gdf_stations': gdf_stations, 'gdf_municipios': gdf_municipios,
                            'df_long': df_long, 'df_enso': df_enso, 'data_loaded': True
                        })
                        st.success("¡Datos cargados y listos!")
                    else:
                        st.error("Hubo un error al procesar los archivos.")

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

        # --- INICIO DE LA LÓGICA CONDICIONAL ---
        if st.session_state.get('data_loaded', False):
            if 'geojson_loaded' not in st.session_state:
                try:
                    st.session_state['gdf_municipios_ant'] = gpd.read_file("data/MunicipiosAntioquia.geojson")
                    st.session_state['gdf_predios'] = gpd.read_file("data/PrediosEjecutados.geojson")
                    st.session_state['gdf_subcuencas'] = gpd.read_file("data/SubcuencasAinfluencia.geojson")
                    st.session_state['geojson_loaded'] = True
                except Exception as e:
                    st.error(f"Error al cargar los archivos GeoJSON locales: {e}")

            st.sidebar.success("Datos cargados.")
            if st.sidebar.button("Limpiar Caché y Reiniciar"):
                auth_info = {
                    "authentication_status": st.session_state.get("authentication_status"),
                    "name": st.session_state.get("name"),
                    "username": st.session_state.get("username")
                }
                st.cache_data.clear()
                st.cache_resource.clear()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                for key, value in auth_info.items():
                    st.session_state[key] = value

            # --- Creación de la Barra Lateral y Obtención de Filtros ---
            filters = create_sidebar(st.session_state.gdf_stations, st.session_state.df_long)
            
            gdf_filtered = get_filtered_data(
                st.session_state.gdf_stations, filters["min_data_perc"], 
                tuple(filters["selected_altitudes"]), tuple(filters["selected_regions"]), 
                tuple(filters["selected_municipios"]), tuple(filters["selected_celdas"])
            )
            
            stations_for_analysis = filters["selected_stations"]

            tab_names = [
                "Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados", 
                "Análisis de Anomalías", "Análisis de Extremos", "Estadísticas", 
                "Correlación", "Análisis ENSO", "Tendencias y Pronósticos", 
                "Pronóstico del Tiempo", "Descargas", "Tabla de Estaciones", "Generar Reporte"
            ]
            tabs = st.tabs(tab_names)

            if not stations_for_analysis:
                with tabs[0]:
                    display_welcome_tab()
                    st.info("Para comenzar, seleccione al menos una estación en el panel de la izquierda.")
                for i, tab in enumerate(tabs):
                    if i > 0:
                        with tab:
                            st.info("Seleccione al menos una estación para ver el contenido.")
                return

            df_monthly_filtered, df_anual_melted = get_processed_time_series(
                st.session_state.df_long, tuple(stations_for_analysis), filters["year_range"], 
                tuple(filters["meses_numeros"]), filters["analysis_mode"], 
                filters["exclude_na"], filters["exclude_zeros"]
            )
            
            display_args = {
                "gdf_filtered": gdf_filtered, "stations_for_analysis": stations_for_analysis, 
                "df_anual_melted": df_anual_melted, "df_monthly_filtered": df_monthly_filtered, 
                "analysis_mode": filters["analysis_mode"], "selected_regions": filters["selected_regions"], 
                "selected_municipios": filters["selected_municipios"], "selected_altitudes": filters["selected_altitudes"]
            }
            
            # --- Renderizado de Pestañas ---
            with tabs[0]:
                display_welcome_tab()
            with tabs[1]:
                display_spatial_distribution_tab(**display_args)
            with tabs[2]:
                display_graphs_tab(**display_args)
            with tabs[3]:
                display_advanced_maps_tab(**display_args)
            with tabs[4]:
                display_anomalies_tab(df_long=st.session_state.df_long, **display_args)
            with tabs[5]:
                display_drought_analysis_tab(**display_args)
            with tabs[6]:
                display_stats_tab(df_long=st.session_state.df_long, **display_args)
            with tabs[7]:
                display_correlation_tab(**display_args)
            with tabs[8]:
                display_enso_tab(df_enso=st.session_state.df_enso, **display_args)
            with tabs[9]:
                display_trends_and_forecast_tab(df_full_monthly=st.session_state.df_long, **display_args)
            with tabs[10]:
                display_forecast_tab(**display_args)
            with tabs[11]:
                display_downloads_tab(
                    df_anual_melted=df_anual_melted, df_monthly_filtered=df_monthly_filtered,
                    stations_for_analysis=stations_for_analysis, analysis_mode=filters["analysis_mode"]
                )
            with tabs[12]:
                display_station_table_tab(**display_args)
            
            with tabs[13]:
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
                
                selected_report_sections = st.multiselect(
                    "Secciones a incluir:", 
                    options=report_sections_options, 
                    default=report_sections_options
                )

                if st.button("Generar Reporte PDF"):
                    if not selected_report_sections:
                        st.warning("Seleccione al menos una sección.")
                    else:
                        with st.spinner("Generando reporte..."):
                            try:
                                summary_data = {
                                    "Estaciones": f"{len(stations_for_analysis)}/{len(st.session_state.gdf_stations)}",
                                    "Periodo": f"{filters['year_range'][0]}-{filters['year_range'][1]}",
                                    "Modo de Análisis": filters['analysis_mode']
                                }
                                df_anomalies = calculate_monthly_anomalies(df_monthly_filtered, st.session_state.df_long)
                                
                                report_pdf_bytes = generate_pdf_report(
                                    report_title=report_title,
                                    sections_to_include=selected_report_sections,
                                    summary_data=summary_data,
                                    df_anomalies=df_anomalies,
                                    **display_args
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
        else: # Si los datos no están cargados
            display_welcome_tab()
            st.warning("Para comenzar, cargue los datos usando el panel de la izquierda.")
    
    elif st.session_state.get("authentication_status") is False:
        st.error('Usuario/contraseña incorrecto')
    elif st.session_state.get("authentication_status") is None:
        st.warning('Por favor, ingrese su usuario y contraseña para continuar')

if __name__ == "__main__":
    main()
