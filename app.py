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
    display_stats_tab, display_correlation_tab, display_enso_tab,
    display_trends_and_forecast_tab, display_downloads_tab, display_station_table_tab
)

# Desactivar Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Función para sincronizar la selección de todas las estaciones
def sync_station_selection():
    options = sorted(st.session_state.get('filtered_station_options', []))
    if st.session_state.get('select_all_checkbox', True):
        st.session_state.station_multiselect = options
    else:
        st.session_state.station_multiselect = []

# Función para aplicar filtros a las estaciones
def apply_filters_to_stations(df, min_perc, altitudes, regions, municipios, celdas):
    stations_filtered = df.copy()
    if Config.PERCENTAGE_COL in stations_filtered.columns:
        if stations_filtered[Config.PERCENTAGE_COL].dtype == 'object':
            stations_filtered[Config.PERCENTAGE_COL] = \
                pd.to_numeric(stations_filtered[Config.PERCENTAGE_COL].astype(str).str.replace(',', '.', regex=False),
                              errors='coerce').fillna(0)
        stations_filtered = stations_filtered[stations_filtered[Config.PERCENTAGE_COL] >= min_perc]
    if altitudes:
        conditions = []
        for r in altitudes:
            if r == '0-500': conditions.append((stations_filtered[Config.ALTITUDE_COL] >= 0) & (stations_filtered[Config.ALTITUDE_COL] <= 500))
            elif r == '500-1000': conditions.append((stations_filtered[Config.ALTITUDE_COL] > 500) & (stations_filtered[Config.ALTITUDE_COL] <= 1000))
            elif r == '1000-2000': conditions.append((stations_filtered[Config.ALTITUDE_COL] > 1000) & (stations_filtered[Config.ALTITUDE_COL] <= 2000))
            elif r == '2000-3000': conditions.append((stations_filtered[Config.ALTITUDE_COL] > 2000) & (stations_filtered[Config.ALTITUDE_COL] <= 3000))
            elif r == '>3000': conditions.append(stations_filtered[Config.ALTITUDE_COL] > 3000)
        if conditions: stations_filtered = stations_filtered[pd.concat(conditions, axis=1).any(axis=1)]
    if regions: stations_filtered = stations_filtered[stations_filtered[Config.REGION_COL].isin(regions)]
    if municipios: stations_filtered = stations_filtered[stations_filtered[Config.MUNICIPALITY_COL].isin(municipios)]
    if celdas: stations_filtered = stations_filtered[stations_filtered[Config.CELL_COL].isin(celdas)]
    return stations_filtered

def main():
    st.set_page_config(layout="wide", page_title=Config.APP_TITLE)
    
    st.markdown("""
    <style>
    div.block-container {padding-top: 2rem;}
    .sidebar .sidebar-content {font-size: 13px; }
    [data-testid="stMetricValue"] {font-size: 1.8rem; }
    [data-testid="stMetricLabel"] {font-size: 1rem; padding-bottom: 5px; }
    button[data-baseweb="tab"] {font-size: 16px; font-weight: bold; color: #333; }
    </style>
    """, unsafe_allow_html=True)

    Config.initialize_session_state()

    title_col1, title_col2 = st.columns([0.07, 0.93])
    with title_col1:
        if os.path.exists(Config.LOGO_PATH):
            try:
                with open(Config.LOGO_PATH, "rb") as f:
                    logo_bytes = f.read()
                st.image(logo_bytes, width=50)
            except Exception:
                pass 
    with title_col2:
        st.markdown(f'<h1 style="font-size:28px; margin-top:1rem;">{Config.APP_TITLE}</h1>', unsafe_allow_html=True)

    st.sidebar.header("Panel de Control")

    # --- 1. LÓGICA DE CARGA ---
    # El expander ahora se controla con el estado 'data_loaded'
    with st.sidebar.expander("**Subir/Actualizar Archivos Base**", expanded=not st.session_state.get('data_loaded', False)):
        uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (CSV)", type="csv", key='uploaded_file_mapa')
        uploaded_file_precip = st.file_uploader("2. Cargar archivo de precipitación (CSV)", type="csv", key='uploaded_file_precip')
        uploaded_zip_shapefile = st.file_uploader("3. Cargar shapefile de municipios (.zip)", type="zip", key='uploaded_zip_shapefile')

        if st.button("Procesar y Almacenar Datos", key='process_data_button') and \
           all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
            
            st.cache_resource.clear() 
            
            with st.spinner("Procesando archivos y cargando datos..."):
                gdf_stations, gdf_municipios, df_long, df_enso = load_and_process_all_data(
                    uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
                    
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

    # --- LÓGICA DE FILTROS Y DESPLIEGUE (Solo si los datos están cargados) ---
    if st.session_state.get('data_loaded', False) and st.session_state.get('df_long') is not None:
        st.sidebar.success("✅ Datos base cargados y persistentes.")
        
        # --- FILTROS GEOGRÁFICOS ---
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
            
            temp_gdf_for_celdas = temp_gdf_for_mun.copy()
            if selected_municipios:
                temp_gdf_for_celdas = temp_gdf_for_celdas[temp_gdf_for_celdas[Config.MUNICIPALITY_COL].isin(selected_municipios)]
            
            celdas_list = sorted(temp_gdf_for_celdas[Config.CELL_COL].dropna().unique())
            selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, key='celdas_multiselect')

            if st.button("🔄 Limpiar Filtros Geográficos"):
                keys_to_clear = ['min_data_perc_slider', 'altitude_multiselect', 'regions_multiselect', 'municipios_multiselect', 'celdas_multiselect', 'station_multiselect', 'select_all_checkbox', 'year_range', 'meses_nombres']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Ejecutar filtro geográfico
        gdf_filtered = apply_filters_to_stations(st.session_state.gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)

        # --- FILTROS TEMPORALES Y PREPROCESAMIENTO ---
        with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
            stations_options = sorted(gdf_filtered[Config.STATION_NAME_COL].unique())
            st.session_state['filtered_station_options'] = stations_options

            st.checkbox("Seleccionar/Deseleccionar todas las estaciones", key='select_all_checkbox', on_change=sync_station_selection, value=st.session_state.get('select_all_checkbox', True))

            if st.session_state.get('select_all_checkbox', True) and st.session_state.get('station_multiselect') != stations_options:
                st.session_state.station_multiselect = stations_options
            
            selected_stations = st.multiselect('Seleccionar Estaciones', options=stations_options, key='station_multiselect')

            years_with_data = sorted(st.session_state.df_long[Config.YEAR_COL].dropna().unique())
            if not years_with_data:
                st.error("No hay datos de año válidos en el archivo de precipitación.")
                return
                
            year_range_default = (min(years_with_data), max(years_with_data))
            year_range = st.slider("Seleccionar Rango de Años", min_value=min(years_with_data), max_value=max(years_with_data), value=st.session_state.get('year_range', year_range_default), key='year_range')

            meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
            meses_nombres = st.multiselect("Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()), key='meses_nombres')
            meses_numeros = [meses_dict[m] for m in meses_nombres]

        # --- OPCIONES DE PREPROCESAMIENTO Y DEM (MOVIDOS AL FINAL) ---
        with st.sidebar.expander("Opciones de Preprocesamiento de Datos"):
            st.radio("Análisis de Series Mensuales", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode")
            st.checkbox("Excluir datos nulos (NaN)", key='exclude_na')
            st.checkbox("Excluir valores cero (0)", key='exclude_zeros')

        # --- LÓGICA DE DEM ---
        with st.sidebar.expander("Opciones de Modelo Digital de Elevación (DEM)"):
            dem_source = st.radio("Fuente de DEM para KED (Kriging):", ("No usar DEM", "Subir DEM propio (GeoTIFF)", "Cargar DEM desde Servidor"), key="dem_source")
            
            uploaded_dem_file = None
            if dem_source == "Subir DEM propio (GeoTIFF)":
                uploaded_dem_file = st.file_uploader("Cargar GeoTIFF (.tif) para elevación", type=["tif", "tiff"], key="dem_uploader")
            
            if dem_source == "Cargar DEM desde Servidor":
                if st.button("Descargar y Usar DEM Remoto", key="download_dem_button"):
                    with st.spinner("Descargando DEM del servidor..."):
                        try:
                            # La función de descarga está en data_processor.py
                            st.session_state.dem_raster = download_and_load_remote_dem(Config.DEM_SERVER_URL) 
                            st.success("DEM remoto cargado y listo para KED.")
                        except Exception as e:
                            st.error(f"Error al cargar DEM remoto: {e}. Verifique la URL en Config.py")
                            st.session_state.dem_raster = None

            # Procesamiento del DEM (aplica altitud a gdf_stations)
            if dem_source == "No usar DEM":
                st.session_state.dem_raster = None 

            if uploaded_dem_file or st.session_state.get('dem_raster') is not None:
                if f'original_{Config.ALTITUDE_COL}' not in st.session_state:
                    st.session_state[f'original_{Config.ALTITUDE_COL}'] = st.session_state.gdf_stations.get(Config.ALTITUDE_COL, None)

                dem_data = uploaded_dem_file if uploaded_dem_file else st.session_state.dem_raster
                
                st.session_state.gdf_stations = extract_elevation_from_dem(st.session_state.gdf_stations.copy(), dem_data)
                st.sidebar.success("Altitud de estaciones actualizada.")
            else:
                if st.session_state.get(f'original_{Config.ALTITUDE_COL}') is not None and Config.ALTITUDE_COL in st.session_state.gdf_stations.columns:
                     st.session_state.gdf_stations[Config.ALTITUDE_COL] = st.session_state[f'original_{Config.ALTITUDE_COL}']

        # --- PREPARACIÓN DE DATAFRAMES FINALES ---
        stations_for_analysis = selected_stations
        gdf_filtered = gdf_filtered[gdf_filtered[Config.STATION_NAME_COL].isin(stations_for_analysis)]
        st.session_state.meses_numeros = meses_numeros

        # Aplicar el modo de análisis (interpolación o no)
        if st.session_state.analysis_mode == "Completar series (interpolación)":
            # La función complete_series debe estar importada desde data_processor
            df_monthly_processed = complete_series(st.session_state.df_long.copy())
        else:
            df_monthly_processed = st.session_state.df_long.copy()
            
        st.session_state.df_monthly_processed = df_monthly_processed

        # Filtrar datos mensuales por tiempo y estación
        df_monthly_filtered = df_monthly_processed[
            (df_monthly_processed[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
            (df_monthly_processed[Config.DATE_COL].dt.year >= year_range[0]) &
            (df_monthly_processed[Config.DATE_COL].dt.year <= year_range[1]) &
            (df_monthly_processed[Config.DATE_COL].dt.month.isin(meses_numeros))
        ].copy()

        # Filtrar datos anuales para agregación
        annual_data_filtered = st.session_state.df_long[
            (st.session_state.df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
            (st.session_state.df_long[Config.YEAR_COL] >= year_range[0]) &
            (st.session_state.df_long[Config.YEAR_COL] <= year_range[1])
        ].copy()

        # Aplicar exclusión de NaN/Ceros
        if st.session_state.get('exclude_na', False):
            df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
            annual_data_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)

        if st.session_state.get('exclude_zeros', False):
            df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] > 0]
            annual_data_filtered = \
                annual_data_filtered[annual_data_filtered[Config.PRECIPITATION_COL] > 0]

        # Agregación anual (para series anuales)
        annual_agg = annual_data_filtered.groupby([Config.STATION_NAME_COL,
                                                     Config.YEAR_COL]).agg(
            precipitation_sum=(Config.PRECIPITATION_COL, 'sum'),
            meses_validos=(Config.PRECIPITATION_COL, 'count')
        ).reset_index()

        annual_agg.loc[annual_agg['meses_validos'] < 10, 'precipitation_sum'] = np.nan
        df_anual_melted = annual_agg.rename(columns={'precipitation_sum': Config.PRECIPITATION_COL})

        # --- DESPLIEGUE DE PESTAÑAS ---
        tab_names = [
            "Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados",
            "Análisis de Anomalías", "Análisis de extremos hid", "Estadísticas",
            "Análisis de Correlación", "Análisis ENSO", "Tendencias y Pronósticos",
            "Descargas", "Tabla de Estaciones"
        ]
        
        tabs = st.tabs(tab_names)

        # CRÍTICO: Verificar que haya datos para evitar fallos en funciones downstream
        if df_anual_melted.empty or df_monthly_filtered.empty or gdf_filtered.empty:
            with tabs[0]:
                 st.warning("No hay datos disponibles para los filtros aplicados. Ajuste la selección de años, meses o estaciones.")
            return

        with tabs[0]:
            display_welcome_tab()
        with tabs[1]:
            display_spatial_distribution_tab(gdf_filtered, stations_for_analysis, df_anual_melted,
                                             df_monthly_filtered)
        with tabs[2]:
            display_graphs_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis,
                               gdf_filtered)
        with tabs[3]:
            display_advanced_maps_tab(gdf_filtered, stations_for_analysis, df_anual_melted,
                                      df_monthly_filtered)
        with tabs[4]:
            display_anomalies_tab(st.session_state.df_long, df_monthly_filtered, stations_for_analysis)
        with tabs[5]:
            display_drought_analysis_tab(df_monthly_filtered, gdf_filtered, stations_for_analysis)
        with tabs[6]:
            # 💥 Se añadió gdf_filtered como 5to argumento 💥
            display_stats_tab(st.session_state.df_long, df_anual_melted, df_monthly_filtered,
                              stations_for_analysis, gdf_filtered) 
        with tabs[7]:
            display_correlation_tab(df_monthly_filtered, stations_for_analysis)
        with tabs[8]:
            display_enso_tab(df_monthly_filtered, st.session_state.df_enso, gdf_filtered,
                             stations_for_analysis)
        with tabs[9]:
            display_trends_and_forecast_tab(df_anual_melted,
                                            st.session_state.df_monthly_processed, stations_for_analysis)
        with tabs[10]:
            display_downloads_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis)
        with tabs[11]:
            display_station_table_tab(gdf_filtered, df_anual_melted, stations_for_analysis)

    else:
        # Si los datos no se han cargado, mostrar la pantalla de bienvenida
        display_welcome_tab()
        st.info("Para comenzar, por favor cargue los 3 archivos requeridos en el panel de la izquierda y haga clic en 'Procesar y Almacenar Datos'.")


if __name__ == "__main__":
    main()
