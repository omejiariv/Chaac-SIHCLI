iariv# modules/config.py

import os
import streamlit as st
import pandas as pd

class Config:
    #--- Configuración de la Aplicación
    APP_TITLE = "Sistema de Información de Lluvias y Clima en el norte de la región Andina"

    #--- URLs para carga automática desde GitHub ---
    # REEMPLAZA con tu usuario y nombre de repositorio
    GITHUB_USER = "omejiariv"
    GITHUB_REPO = "https://github.com/omejiariv/Chaac-SIHCLI/edit/main/modules/config.py"
    BRANCH = "main" # O la rama que estés usando ('master', etc.)

    # Construimos las URLs a los archivos RAW en GitHub
    URL_ESTACIONES_CSV = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/mapaCVENSO.csv"
    URL_PRECIPITACION_CSV = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/DatosPptnmes_ENSO.csv"
    URL_SHAPEFILE_ZIP = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/mapaCVENSO.zip"

    #--- RUTAS ROBUSTAS A LOS ARCHIVOS DEL PROYECTO ---
    # 1. Obtenemos la ruta a la carpeta donde está este archivo (la carpeta 'modules')
    _MODULES_DIR = os.path.dirname(__file__)
    # 2. Subimos un nivel para llegar a la raíz del proyecto
    _PROJECT_ROOT = os.path.abspath(os.path.join(_MODULES_DIR, '..'))
    # 3. Construimos las rutas completas y correctas a los archivos
    GIF_PATH = os.path.join(_PROJECT_ROOT, 'assets', 'PPAM.gif')
    LOGO_PATH = os.path.join(_PROJECT_ROOT, 'assets', 'CuencaVerde_Logo.jpg')
    
    WELCOME_TEXT = """
    "El futuro, también depende del pasado y de nuestra capacidad presente para anticiparlo"
    omr.
    Esta plataforma interactiva está diseñada para la visualización y análisis de datos históricos de
    precipitación y su relación con el fenómeno ENSO en el norte de la región Andina.
    #### ¿Cómo empezar?
    1. **Cargar Archivos:** En el panel de la izquierda, suba los archivos de estaciones, precipitación
    y el shapefile de municipios.
    2. **Aplicar Filtros:** Utilice el **Panel de Control** para filtrar las estaciones y seleccionar el
    período de análisis.
    3. **Explorar Análisis:** Navegue a través de las pestañas para visualizar los datos.
    """
    
    # --- Nombres de Columnas Estándar (deben coincidir con la lógica de data_processor.py)
    DATE_COL = 'fecha_mes_año'
    PRECIPITATION_COL = 'precipitation'
    STATION_NAME_COL = 'nom_est'
    ALTITUDE_COL = 'alt_est'
    LATITUDE_COL = 'latitud_wgs84'
    LONGITUDE_COL = 'longitud_wgs84'
    MUNICIPALITY_COL = 'municipio'
    REGION_COL = 'depto_region'
    PERCENTAGE_COL = 'porc_datos'
    YEAR_COL = 'año'
    MONTH_COL = 'mes'
    ORIGIN_COL = 'origin'
    CELL_COL = 'celda_xy'
    ET_COL = 'et_mmy' # Evapotranspiración
    ELEVATION_COL = 'elevation_dem' # Usado para KED desde DEM

    # Índices Climáticos
    ENSO_ONI_COL = 'anomalia_oni'
    SOI_COL = 'soi'
    IOD_COL = 'iod'

    #--- Configuración para DEM
    DEM_SERVER_URL = "https://tu-bucket.storage.com/srtm_antioquia.tif" # URL de ejemplo

    @staticmethod
    def initialize_session_state():
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'gdf_stations' not in st.session_state:
            st.session_state.gdf_stations = None
        if 'df_long' not in st.session_state:
            st.session_state.df_long = None
        if 'df_enso' not in st.session_state:
            st.session_state.df_enso = None
        if 'gdf_municipios' not in st.session_state:
            st.session_state.gdf_municipios = None
        if 'df_monthly_processed' not in st.session_state:
            st.session_state.df_monthly_processed = pd.DataFrame()
        if 'meses_numeros' not in st.session_state:
            st.session_state.meses_numeros = list(range(1, 13))
        if 'dem_source' not in st.session_state:
            st.session_state.dem_source = "No usar DEM"
        if 'dem_raster' not in st.session_state:
            st.session_state.dem_raster = None
        if 'sarima_forecast' not in st.session_state:
            st.session_state.sarima_forecast = None
        if 'prophet_forecast' not in st.session_state:
            st.session_state.prophet_forecast = None
        if 'gif_reload_key' not in st.session_state:
            st.session_state.gif_reload_key = 0
