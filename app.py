# app.py

# --- Importaciones Esenciales ---
import streamlit as st
import pandas as pd
import numpy as np
import warnings

# --- Importaciones de tus Módulos ---
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series
from modules.visualizer import (
    display_welcome_tab,
    display_spatial_distribution_tab,
    display_graphs_tab,
    display_advanced_maps_tab,
    display_anomalies_tab,
    display_drought_analysis_tab,
    display_stats_tab,
    display_correlation_tab,
    display_enso_tab,
    display_trends_and_forecast_tab,
    display_downloads_tab,
    display_station_table_tab
)

# Desactivar Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Función Principal de Streamlit ---
def main():
    st.set_page_config(layout="wide", page_title=Config.APP_TITLE)
    
    st.markdown("""
        <style>
        div.block-container {padding-top: 2rem;}
        .sidebar .sidebar-content {font-size: 13px; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; }
        [data-testid="stMetricLabel"] { font-size: 1rem; padding-bottom: 5px; }
        button[data-baseweb="tab"] { font-size: 16px; font-weight: bold; color: #333; }
        </style>
    """, unsafe_allow_html=True)

    Config.initialize_session_state()

    title_col1, title_col2 = st.columns([0.07, 0.93])
    with title_col1:
        if os.path.exists(Config.LOGO_DROP_PATH): st.image(Config.LOGO_DROP_PATH, width=50)
    with title_col2:
        st.markdown(f'<h1 style="font-size:28px; margin-top:1rem;">{Config.APP_TITLE}</h1>', unsafe_allow_html=True)
    
    st.sidebar.header("Panel de Control")

    with st.sidebar.expander("**Cargar Archivos**", expanded=not st.session_state.data_loaded):
        uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (mapaCVENSO.csv)", type="csv")
        uploaded_file_precip = st.file_uploader("2. Cargar archivo de precipitación mensual y ENSO (DatosPptnmes_ENSO.csv)", type="csv")
        uploaded_zip_shapefile = st.file_uploader("3. Cargar shapefile de municipios (.zip)", type="zip")

        if not st.session_state.data_loaded and all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
            with st.spinner("Procesando archivos y cargando datos..."):
                gdf_stations, gdf_municipios, df_long, df_enso = load_and_process_all_data(
                    uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
                if gdf_stations is not None and df_long is not None:
                    st.session_state.gdf_stations = gdf_stations
                    st.session_state.gdf_municipios = gdf_municipios
                    st.session_state.df_long = df_long
                    st.session_state.df_enso = df_enso
                    st.session_state.data_loaded = True
                    st.rerun()
                else:
                    st.error("Hubo un error al procesar los archivos.")
        
        if st.button("Recargar Datos"):
            st.cache_data.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if st.session_state.data_loaded:
        with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
            min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, st.session_state.get('min_data_perc_slider', 0), key='min_data_perc_slider')
            altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000']
            selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges, default=st.session_state.get('altitude_multiselect', []), key='altitude_multiselect')
            regions_list = sorted(st.session_state.gdf_stations[Config.REGION_COL].dropna().unique())
            selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list, default=st.session_state.get('regions_multiselect', []), key='regions_multiselect')
            
            # Lógica para filtrar municipios dinámicamente
            temp_gdf = st.session_state.gdf_stations.copy()
            if selected_regions:
                temp_gdf = temp_gdf[temp_gdf[Config.REGION_COL].isin(selected_regions)]
            municipios_list = sorted(temp_gdf[Config.MUNICIPALITY_COL].dropna().unique())
            selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list, default=st.session_state.get('municipios_multiselect', []), key='municipios_multiselect')

        with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
            years_with_data = sorted(st.session_state.df_long[Config.YEAR_COL].unique())
            year_range = st.slider("Seleccionar Rango de Años", min_value=min(years_with_data), max_value=max(years_with_data), value=(min(years_with_data), max(years_with_data)))
            
            meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
            meses_nombres = st.multiselect("Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
            meses_numeros = [meses_dict[m] for m in meses_nombres]

        # --- LÓGICA CENTRAL DE PREPROCESAMIENTO ---
        df_monthly_filtered = st.session_state.df_long.copy() # Usamos el df original para los filtros
        
        # ... (Aplicar filtros) ...

        # --- Pestañas y Visualización ---
        tab_names = [
            "🏠 Bienvenida", "🗺️ Distribución Espacial", "📊 Gráficos", "✨ Mapas Avanzados", 
            "📉 Análisis de Anomalías", "🌪️ Análisis de extremos hid", "🔢 Estadísticas", 
            "🤝 Análisis de Correlación", "🌊 Análisis ENSO", "📈 Tendencias y Pronósticos", 
            "📥 Descargas", "📋 Tabla de Estaciones"
        ]
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            display_welcome_tab()
        # ... (Y así sucesivamente para el resto de las pestañas) ...
            
    else:
        display_welcome_tab()
        st.info("👋 Para comenzar, por favor cargue los 3 archivos requeridos en el panel de la izquierda.")

if __name__ == "__main__":
    main()
