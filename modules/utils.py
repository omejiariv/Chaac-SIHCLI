# modules/utils.py

import streamlit as st
import io
import plotly.graph_objects as go 
import folium 
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import rasterio
from rasterio.mask import mask
from pykrige.ok import OrdinaryKriging
import warnings

# --- NUEVA FUNCIÓN PARA CORRECCIÓN NUMÉRICA ---
@st.cache_data
def standardize_numeric_column(series):
    """
    Convierte una serie de Pandas a valores numéricos de manera robusta,
    reemplazando comas por puntos como separador decimal.
    """
    series_clean = series.astype(str).str.replace(',', '.', regex=False)
    return pd.to_numeric(series_clean, errors='coerce')


def display_plotly_download_buttons(fig, file_prefix):
    """Muestra botones de descarga para un gráfico Plotly (HTML y PNG).""" 
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, include_plotlyjs='cdn')
        st.download_button(
            label="Descargar Gráfico (HTML)",
            data=html_buffer.getvalue(),
            file_name=f"{file_prefix}.html",
            mime="text/html",
            key=f"dl_html_{file_prefix}",
            use_container_width=True
        )

    with col2:
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
            st.download_button(
                label="Descargar Gráfico (PNG)",
                data=img_bytes,
                file_name=f"{file_prefix}.png",
                mime="image/png",
                key=f"dl_png_{file_prefix}",
                use_container_width=True
            )
        except Exception as e:
            st.warning("No se pudo generar la imagen PNG. Asegúrate de tener la librería 'kaleido' instalada ('pip install kaleido').")

def add_folium_download_button(map_object, file_name):
    """Muestra un botón de descarga para un mapa de Folium (HTML)."""
    st.markdown("---")
    map_buffer = io.BytesIO()
    map_object.save(map_buffer, close_file=False)
    st.download_button(
        label="Descargar Mapa (HTML)",
        data=map_buffer.getvalue(),
        file_name=file_name,
        mime="text/html",
        key=f"dl_map_{file_name.replace('.', '_')}",
        use_container_width=True
    )


# --- FUNCIÓN DE KRIGING
@st.cache_data
def create_kriging_by_basin(_gdf_points, grid_lon, grid_lat, value_col='Valor', variogram_model='linear'):
    """
    Realiza la interpolación de Kriging Ordinario y la cachea.
    El argumento _gdf_points es ignorado por el hasher de Streamlit gracias al '_'.
    """
    if _gdf_points.empty or len(_gdf_points) < 3:
        warnings.warn("No hay suficientes puntos de datos para la interpolación de Kriging.")
        return None, None

    # Extraer coordenadas y valores
    x = _gdf_points.geometry.x.values
    y = _gdf_points.geometry.y.values
    z = _gdf_points[value_col].values

    try:
        # Configurar y ejecutar el Kriging
        ok = OrdinaryKriging(
            x, y, z,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False
        )
        # Crear la malla y ejecutar la predicción
        grid, variance = ok.execute('grid', grid_lon, grid_lat)
        return grid, variance
    except Exception as e:
        warnings.warn(f"Error durante la ejecución de Kriging: {e}")
        return None, None

# --- CÁLCULOS MORFOMÉTRICOS Y DE ELEVACIÓN ---

def calculate_morphometry(basin_gdf, dem_path):
    """
    Calcula los parámetros morfométricos de una cuenca unificada.
    
    Args:
        basin_gdf (GeoDataFrame): GeoDataFrame con la geometría de la cuenca (una sola fila).
        dem_path (str): Ruta al archivo DEM temporal.
        
    Returns:
        dict: Un diccionario con los resultados morfométricos.
    """
    results = {}
    try:
        # --- 1. Cálculos geométricos (no requieren el DEM) ---
        # Asegurarse que el GDF está en un sistema de coordenadas proyectadas para cálculos precisos
        if basin_gdf.crs.is_geographic:
            # Estima y convierte a la zona UTM apropiada
            projected_gdf = basin_gdf.to_crs(basin_gdf.estimate_utm_crs())
        else:
            projected_gdf = basin_gdf

        area_m2 = projected_gdf.geometry.area.iloc[0]
        perimetro_m = projected_gdf.geometry.length.iloc[0]
        
        results['area_km2'] = area_m2 / 1_000_000
        results['perimetro_km'] = perimetro_m / 1_000
        
        # Índice de Forma (Gravelius)
        if results['area_km2'] > 0:
            results['indice_forma'] = results['perimetro_km'] / (2 * np.sqrt(np.pi * results['area_km2']))
        else:
            results['indice_forma'] = 0

        # --- 2. Cálculos de elevación (requieren el DEM) ---
        with rasterio.open(dem_path) as src:
            # Reproyectar la cuenca al CRS del DEM para enmascarar correctamente
            basin_for_mask = basin_gdf.to_crs(src.crs)
            
            # Recortar el DEM a la extensión de la cuenca
            out_image, out_transform = mask(src, basin_for_mask.geometry, crop=True)
            
            # Extraer las elevaciones válidas
            elevations = out_image[0]
            no_data_value = src.nodata
            if no_data_value is not None:
                valid_elevations = elevations[elevations != no_data_value]
            else:
                valid_elevations = elevations.flatten()

            valid_elevations = valid_elevations[~np.isnan(valid_elevations)]

            if valid_elevations.size > 0:
                results['alt_max_m'] = float(np.max(valid_elevations))
                results['alt_min_m'] = float(np.min(valid_elevations))
                results['alt_prom_m'] = float(np.mean(valid_elevations))
            else:
                results['alt_max_m'] = None
                results['alt_min_m'] = None
                results['alt_prom_m'] = None

    except Exception as e:
        results['error'] = f"Error en 'calculate_morphometry': {e}"
        
    return results

def calculate_hypsometric_curve(basin_gdf, dem_path):
    """
    Calcula los datos para la curva hipsométrica de la cuenca.

    Args:
        basin_gdf (GeoDataFrame): GeoDataFrame con la geometría de la cuenca.
        dem_path (str): Ruta al archivo DEM temporal.

    Returns:
        dict: Un diccionario con los datos para graficar la curva.
    """
    results = {}
    try:
        with rasterio.open(dem_path) as src:
            basin_for_mask = basin_gdf.to_crs(src.crs)
            out_image, out_transform = mask(src, basin_for_mask.geometry, crop=True)
            
            elevations = out_image[0]
            no_data_value = src.nodata
            if no_data_value is not None:
                valid_elevations = elevations[elevations != no_data_value]
            else:
                valid_elevations = elevations.flatten()
            
            valid_elevations = valid_elevations[~np.isnan(valid_elevations)]

            if valid_elevations.size == 0:
                results['error'] = "No se encontraron datos de elevación válidos en el área de la cuenca."
                return results

            # Ordenar las elevaciones de menor a mayor
            sorted_elevations = np.sort(valid_elevations)
            
            # Calcular el área acumulada como un porcentaje
            # Se invierte para que la elevación más alta tenga 0% de área por encima de ella
            n = len(sorted_elevations)
            cumulative_area_percent = 100 * (1.0 - np.arange(n) / n)

            results['elevations'] = sorted_elevations.tolist()
            results['cumulative_area_percent'] = cumulative_area_percent.tolist()

    except Exception as e:
        results['error'] = f"Error en 'calculate_hypsometric_curve': {e}"
        
    return results



