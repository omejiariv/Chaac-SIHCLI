# modules/utils.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import folium
from folium.raster_layers import WmsTileLayer
from modules.config import Config
import warnings
from scipy.stats import gamma, norm

def add_plotly_download_buttons(fig, file_prefix):
    """Muestra botones de descarga para un gráfico Plotly (HTML y PNG)."""
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, include_plotlyjs='cdn')
        st.download_button(
            label="📥 Descargar Gráfico (HTML)",
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
                label="📥 Descargar Gráfico (PNG)",
                data=img_bytes,
                file_name=f"{file_prefix}.png",
                mime="image/png",
                key=f"dl_png_{file_prefix}",
                use_container_width=True
            )
        except Exception as e:
            st.warning("No se pudo generar la imagen PNG. Asegúrate de tener la librería 'kaleido' instalada (`pip install kaleido`).")

def add_folium_download_button(map_object, file_name):
    """Muestra un botón de descarga para un mapa de Folium (HTML)."""
    st.markdown("---")
    map_buffer = io.BytesIO()
    map_object.save(map_buffer, close_file=False)
    st.download_button(
        label="📥 Descargar Mapa (HTML)",
        data=map_buffer.getvalue(),
        file_name=file_name,
        mime="text/html",
        key=f"dl_map_{file_name.replace('.', '_')}",
        use_container_width=True
    )

def get_map_options():
    return {
        "CartoDB Positron (Predeterminado)": {"tiles": "cartodbpositron", "attr": '&copy; <a href="https://carto.com/attributions">CartoDB</a>', "overlay": False},
        "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', "overlay": False},
        "Topografía (OpenTopoMap)": {"tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", "attr": 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)', "overlay": False},
        "Relieve (Stamen Terrain)": {"tiles": "Stamen Terrain", "attr": 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', "overlay": False},
        "Relieve y Océanos (GEBCO)": {"url": "https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/web_map_service.php", "layers": "GEBCO_2021_Surface", "transparent": False, "attr": "GEBCO 2021", "overlay": True},
        "Mapa de Colombia (WMS IDEAM)": {"url": "https://geoservicios.ideam.gov.co/geoserver/ideam/wms", "layers": "ideam:col_admin", "transparent": True, "attr": "IDEAM", "overlay": True},
        "Cobertura de la Tierra (WMS IGAC)": {"url": "https://servicios.igac.gov.co/server/services/IDEAM/IDEAM_Cobertura_Corine/MapServer/WMSServer", "layers": "IDEAM_Cobertura_Corine_Web", "transparent": True, "attr": "IGAC", "overlay": True},
    }

def display_map_controls(container_object, key_prefix):
    map_options = get_map_options()
    base_maps = {k: v for k, v in map_options.items() if not v.get("overlay")}
    overlays = {k: v for k, v in map_options.items() if v.get("overlay")}
    
    selected_base_map_name = container_object.selectbox("Seleccionar Mapa Base", list(base_maps.keys()), key=f"{key_prefix}_base_map")
    default_overlays = ["Mapa de Colombia (WMS IDEAM)"]
    selected_overlays = container_object.multiselect("Seleccionar Capas Adicionales", list(overlays.keys()), default=default_overlays, key=f"{key_prefix}_overlays")
    
    return base_maps[selected_base_map_name], [overlays[k] for k in selected_overlays]

def create_folium_map(location, zoom, base_map_config, overlays_config, fit_bounds_data=None):
    """Crea un mapa base de Folium con las capas y configuraciones especificadas."""
    m = folium.Map(
        location=location,
        zoom_start=zoom,
        tiles=base_map_config.get("tiles", "OpenStreetMap"),
        attr=base_map_config.get("attr", None)
    )
    if fit_bounds_data is not None and not fit_bounds_data.empty:
        bounds = fit_bounds_data.total_bounds
        if np.all(np.isfinite(bounds)):
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    for layer_config in overlays_config:
        WmsTileLayer(
            url=layer_config["url"],
            layers=layer_config["layers"],
            fmt='image/png',
            transparent=layer_config.get("transparent", False),
            overlay=True,
            control=True,
            name=layer_config.get("attr", "Overlay")
        ).add_to(m)
        
    return m
