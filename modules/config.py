# modules/config.py

import streamlit as st
import pandas as pd
import os

# 1. Definir BASE_DIR aquí, en el nivel superior del archivo.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Ahora, definir la clase. Ya puede "ver" y usar la variable BASE_DIR.
class Config:
    # Nombres de Columnas de Datos
    STATION_NAME_COL = 'nom_est'
    PRECIPITATION_COL = 'precipitation'
    # ... (el resto de tus nombres de columnas) ...
    
    # Rutas de Archivos (usando la ruta absoluta)
    LOGO_PATH = os.path.join(BASE_DIR, "data", "CuencaVerde Logo.jpg")
    LOGO_DROP_PATH = os.path.join(BASE_DIR, "data", "CuencaVerdeGoticaLogo.JPG")
    GIF_PATH = os.path.join(BASE_DIR, "data", "PPAM.gif")
    
    # Mensajes de la UI
    APP_TITLE = "Sistema de información de las lluvias y el Clima en el norte de la región Andina"
    # ... (el resto de tu clase Config se mantiene igual) ...
    WELCOME_TEXT = """
    Esta plataforma interactiva está diseñada para la visualización y análisis de datos históricos de precipitación y su
    relación con el fenómeno ENSO en el norte de la región Andina.
    
    **¿Cómo empezar?**
    1.  **Cargue sus archivos**: Si es la primera vez que usa la aplicación, el panel de la izquierda le solicitará cargar los archivos de estaciones,
    precipitación y el shapefile de municipios. La aplicación recordará estos archivos en su sesión.
    2.  **Filtre los datos**: Una vez cargados los datos, utilice el **Panel de Control** en la barra lateral para filtrar las estaciones por ubicación (región, municipio), altitud,
    porcentaje de datos disponibles, y para seleccionar el período de análisis (años y meses).
    3.  **Explore las pestañas**: Cada pestaña ofrece una perspectiva diferente de los datos. Navegue a través de ellas para descubrir:
        - **Distribución Espacial**: Mapas interactivos de las estaciones.
        - **Gráficos**: Series de tiempo anuales, mensuales, comparaciones y distribuciones.
        - **Mapas Avanzados**: Animaciones y mapas de interpolación.
        - **Análisis de Anomalías**: Desviaciones de la precipitación respecto a la media histórica.
        - **Tendencias y Pronósticos**: Análisis de tendencias a largo plazo y modelos de pronóstico.
    
    Utilice el botón **🧹 Limpiar Filtros** en el panel lateral para reiniciar su selección en cualquier momento.
    
    ¡Esperamos que esta herramienta le sea de gran utilidad para sus análisis climáticos!
    """
    
    @staticmethod
    def initialize_session_state():
        # ... (este método se mantiene igual) ...
        state_defaults = {
            'data_loaded': False,
            'analysis_mode': "Usar datos originales",
            'select_all_stations_state': False,
            'df_monthly_processed': pd.DataFrame(),
            'gdf_stations': None,
            'df_precip_anual': None,
            'gdf_municipios': None,
            'df_long': None,
            'df_enso': None,
            'min_data_perc_slider': 0,
            'altitude_multiselect': [],
            'regions_multiselect': [],
            'municipios_multiselect': [],
            'celdas_multiselect': [],
            'station_multiselect': [],
            'exclude_na': False,
            'exclude_zeros': False,
            'uploaded_forecast': None,
            'sarima_forecast': None, 
            'prophet_forecast': None 
        }
        for key, value in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
