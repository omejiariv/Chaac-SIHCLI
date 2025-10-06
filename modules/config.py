# modules/config.py

import os
import streamlit as st
import pandas as pd

class Config:
    #--- Configuración de la Aplicación
    APP_TITLE = "Sistema de Información de Lluvias y Clima en el norte de la región Andina"

    #--- URLs para carga automática desde GitHub ---
   
    GITHUB_USER = "omejiariv"
    GITHUB_REPO = "Chaac-SIHCLI" # Solo el nombre del repositorio
    BRANCH = "main" 
    
    # Construimos las URLs a los archivos RAW en GitHub, añadiendo la carpeta "data"
    URL_ESTACIONES_CSV = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/data/mapaCVENSO.csv"
    URL_PRECIPITACION_CSV = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/data/DatosPptnmes_ENSO.csv"
    URL_SHAPEFILE_ZIP = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/data/mapaCVENSO.zip"


    #--- RUTAS ROBUSTAS A LOS ARCHIVOS DEL PROYECTO ---
    _MODULES_DIR = os.path.dirname(__file__)
    _PROJECT_ROOT = os.path.abspath(os.path.join(_MODULES_DIR, '..'))
    GIF_PATH = os.path.join(_PROJECT_ROOT, 'assets', 'PPAM.gif')
    LOGO_PATH = os.path.join(_PROJECT_ROOT, 'assets', 'CuencaVerde_Logo.jpg')
    
    CHAAC_IMAGE_PATH = os.path.join(_PROJECT_ROOT, 'assets', 'chaac.png') # Asumiremos que guardarás una imagen llamada 'chaac.png' en tu carpeta 'assets'
    CHAAC_STORY = """
    ### Chaac, el Señor de la Lluvia

    En la mitología maya, **Chaac** es una de las deidades más importantes y veneradas. No es un solo dios, sino una deidad cuádruple que reside en los cuatro puntos cardinales, cada uno asociado a un color y un destino.

    Representado como un anciano con rasgos de reptil, una nariz larga y curva, y dos colmillos, Chaac blande su hacha de relámpagos para golpear las nubes y producir la lluvia, esencial para la vida y la cosecha del maíz. Su llanto eran las gotas que caían sobre la tierra.

    Esta plataforma lleva su nombre como un homenaje a la vital importancia del agua y la lluvia en nuestra región, buscando, al igual que los mayas, entender sus patrones para anticipar nuestro futuro.
    """
    
    QUOTE_TEXT = '"El futuro, también depende del pasado y de nuestra capacidad presente para anticiparlo"'
    QUOTE_AUTHOR = "omr."

    WELCOME_TEXT = """
    Esta plataforma interactiva está diseñada para la visualización y análisis de datos históricos de
    precipitación y su relación con el fenómeno ENSO en el norte de la región Andina.
    #### ¿Cómo empezar?
    1. **Cargar Archivos:** En el panel de la izquierda, suba los archivos de estaciones, precipitación
    y el shapefile de municipios.
    2. **Aplicar Filtros:** Utilice el **Panel de Control** para filtrar las estaciones y seleccionar el
    período de análisis.
    3. **Explorar Análisis:** Navegue a través de las pestañas para visualizar los datos.
    """
    
    # --- Nombres de Columnas Estándar
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
    ET_COL = 'et_mmy'
    ELEVATION_COL = 'elevation_dem'
    ENSO_ONI_COL = 'anomalia_oni'
    SOI_COL = 'soi'
    IOD_COL = 'iod'

    #--- Configuración para DEM
    DEM_SERVER_URL = "https://tu-bucket.storage.com/srtm_antioquia.tif"

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
