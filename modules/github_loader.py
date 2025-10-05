# modules/github_loader.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import requests
import io

@st.cache_data(ttl=3600) # Cache por 1 hora para no descargar en cada re-ejecución
def load_csv_from_url(url):
    """Carga un archivo CSV desde una URL a un DataFrame de Pandas."""
    try:
        df = pd.read_csv(url, sep=";")
        # Convierte el DataFrame a un objeto de bytes en memoria, simulando un archivo subido
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        return io.BytesIO(csv_bytes)
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV desde la URL: {url}\nError: {e}")
        return None

@st.cache_data(ttl=3600)
def load_zip_from_url(url):
    """Descarga un archivo ZIP (shapefile) desde una URL y lo retorna como un objeto de bytes en memoria."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza un error si la descarga falla
        return io.BytesIO(response.content)
    except Exception as e:
        st.error(f"Error al descargar el archivo ZIP desde la URL: {url}\nError: {e}")
        return None
