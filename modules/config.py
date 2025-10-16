# modules/config.py

import os
import streamlit as st
import pandas as pd
import geopandas as gpd # Es buena práctica inicializar con un GeoDataFrame vacío si se espera uno.

class Config:
    """
    Clase para centralizar toda la configuración de la aplicación Streamlit.
    Contiene constantes como títulos, URLs, rutas a archivos y nombres de columnas,
    así como la lógica para inicializar de forma segura el estado de la sesión.
    """
    # --- Configuración General de la Aplicación ---
    APP_TITLE = "Sistema de Información de Lluvias y Clima en el norte de la región Andina"

    # --- Constantes para carga de datos desde GitHub ---
    GITHUB_USER = "omejiariv"
    GITHUB_REPO = "Chaac-SIHCLI"
    BRANCH = "main"

    # --- URLs directas a los archivos RAW en GitHub ---
    URL_ESTACIONES_CSV = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/data/mapaCVENSO.csv"
    URL_PRECIPITACION_CSV = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/data/DatosPptnmes_ENSO.csv"
    URL_SHAPEFILE_ZIP = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/data/mapaCVENSO.zip"
    URL_SUBCUENCAS_GEOJSON = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/data/SubcuencasAinfluencia.geojson"
    DEM_SERVER_URL = "https://tu-bucket.storage.com/srtm_antioquia.tif" # URL de ejemplo para el DEM

    # --- Rutas robustas a los archivos locales del proyecto ---
    # Se calcula la ruta base del proyecto para que funcione en cualquier sistema operativo.
    _MODULES_DIR = os.path.dirname(__file__)
    _PROJECT_ROOT = os.path.abspath(os.path.join(_MODULES_DIR, '..'))
    
    # --- Rutas a los assets (imágenes, logos, etc.) ---
    GIF_PATH = os.path.join(_PROJECT_ROOT, 'assets', 'PPAM.gif')
    LOGO_PATH = os.path.join(_PROJECT_ROOT, 'assets', 'CuencaVerde_Logo.jpg')
    CHAAC_IMAGE_PATH = os.path.join(_PROJECT_ROOT, 'assets', 'chaac.png')

    # --- Textos estáticos para la interfaz de usuario ---
    CHAAC_STORY = """
    ### Chaac, el Señor de la Lluvia
    En la mitología maya, **Chaac** es una de las deidades más importantes. Reside en los cuatro puntos cardinales y blande su hacha de relámpagos para golpear las nubes y producir la lluvia, esencial para la vida. Esta plataforma lleva su nombre como homenaje a la vital importancia del agua en nuestra región.
    """
    QUOTE_TEXT = '"El futuro, también depende del pasado y de nuestra capacidad presente para anticiparlo"'
    QUOTE_AUTHOR = "omr."
    WELCOME_TEXT = """
    Esta plataforma interactiva está diseñada para la visualización y análisis de datos históricos de
    precipitación y su relación con el fenómeno ENSO en el norte de la región Andina.
    #### ¿Cómo empezar?
    1. **Cargar Archivos:** En el panel de la izquierda, suba los archivos requeridos.
    2. **Aplicar Filtros:** Utilice el **Panel de Control** para filtrar y seleccionar el período de análisis.
    3. **Explorar Análisis:** Navegue a través de las pestañas para visualizar los datos.
    """

    # --- Nombres de columnas estándar para evitar errores de tipeo ---
    # Usar constantes centralizadas aquí previene errores si un nombre de columna cambia en el futuro.
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

    @staticmethod
    def initialize_session_state():
        """
        Inicializa todas las claves necesarias en st.session_state de forma centralizada y segura.
        Este método evita la re-inicialización en cada recarga de la página, previniendo errores
        y la pérdida de datos durante la interacción del usuario.
        
        **Importante:** Llamar a esta función UNA SOLA VEZ al inicio del script principal de la app.
        """
        # Define el estado inicial por defecto para todas las variables de sesión.
        # Agruparlas en un diccionario hace que el código sea más limpio y fácil de mantener.
        default_state = {
            # Banderas de estado
            'data_loaded': False,
            'run_balance': False,
            
            # DataFrames y GeoDataFrames
            'gdf_stations': None,
            'df_long': None,
            'df_enso': None,
            'gdf_municipios': None,
            'gdf_subcuencas': None,
            'unified_basin_gdf': None,
            'df_monthly_processed': pd.DataFrame(),

            # Datos para modelos y pronósticos
            'sarima_forecast': None,
            'prophet_forecast': None,

            # Variables de la interfaz (widgets y filtros)
            'meses_numeros': list(range(1, 13)),
            'select_all_report_sections_checkbox': False,
            'selected_report_sections_multiselect': [],
            'selected_basins_title': "",

            # Objetos complejos (figuras, rasters)
            'dem_source': "No usar DEM",
            'dem_raster': None,
            'fig_basin': None,
            
            # Variables de control y misceláneos
            'mean_precip': None,
            'error_msg': None,
            'gif_reload_key': 0, # Para forzar la recarga de elementos si es necesario
        }

        # Itera sobre el diccionario y solo inicializa las claves que no existen.
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
