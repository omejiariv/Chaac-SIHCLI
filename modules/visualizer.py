# modules/visualizer.py
import streamlit as st
import pandas as pd
import base64
import geopandas as gpd
import altair as alt
import folium
from folium.plugins import MarkerCluster, MiniMap
from folium.raster_layers import WmsTileLayer
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import branca.colormap as cm
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy import stats
from prophet.plot import plot_plotly
import io
from datetime import datetime, timedelta
from modules import db_manager

# --- Importaciones de Módulos Propios
from modules.analysis import (
    calculate_spi, 
    calculate_spei, 
    calculate_monthly_anomalies,
    calculate_percentiles_and_extremes, 
    analyze_events,
    calculate_climatological_anomalies
)
from modules.config import Config
from modules.utils import add_folium_download_button
from modules.interpolation import create_interpolation_surface, perform_loocv_for_all_methods
from modules.forecasting import (
    generate_sarima_forecast, 
    generate_prophet_forecast,
    get_decomposition_results, 
    create_acf_chart, 
    create_pacf_chart,
    auto_arima_search
)
from modules.data_processor import complete_series
from modules.forecast_api import get_weather_forecast

# --- FUNCIONES DE UTILIDAD DE VISUALIZACIÓN
def display_filter_summary(total_stations_count, selected_stations_count, year_range,
                           selected_months_count, analysis_mode, selected_regions,
                           selected_municipios, selected_altitudes):
    if isinstance(year_range, tuple) and len(year_range) == 2:
        year_text = f"{year_range[0]}-{year_range[1]}"
    else:
        year_text = "N/A"
    
    mode_text = "Original (con huecos)"
    if analysis_mode == "Completar series (interpolación)":
        mode_text = "Completado (interpolado)"
    
    summary_parts = [
        f"**Estaciones:** {selected_stations_count}/{total_stations_count}",
        f"**Período:** {year_text}",
        f"**Datos:** {mode_text}"
    ]
    
    if selected_regions:
        summary_parts.append(f"**Región:** {', '.join(selected_regions)}")
    if selected_municipios:
        summary_parts.append(f"**Municipio:** {', '.join(selected_municipios)}")
    if selected_altitudes:
        summary_parts.append(f"**Altitud:** {', '.join(selected_altitudes)}")
        
    st.info(" | ".join(summary_parts))

def display_map_controls(container_object, key_prefix):
    """Muestra los controles para seleccionar mapa base y capas adicionales."""
    base_map_options = {
        "CartoDB Positron": {"tiles": "cartodbpositron", "attr": "CartoDB"},
        "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": "OpenStreetMap"},
        "Topografía (OpenTopoMap)": {
            "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            "attr": "OpenTopoMap"
        },
    }
    
    overlay_map_options = {
        "Municipios de Antioquia": {
            "name": "Municipios de Antioquia", "type": "geojson", 
            "data_key": "gdf_municipios_ant", "attr": "Municipios"
        },
        "Predios Ejecutados": {
            "name": "Predios Ejecutados", "type": "geojson", 
            "data_key": "gdf_predios", "attr": "Predios"
        },
        "Subcuencas de Influencia": {
            "name": "Subcuencas de Influencia", "type": "geojson", 
            "data_key": "gdf_subcuencas", "attr": "Subcuencas"
        },
        "Mapa de Colombia (WMS IDEAM)": {
            "url": "https://geoservicios.ideam.gov.co/geoserver/ideam/wms",
            "layers": "ideam:col_admin", "fmt": 'image/png',
            "transparent": True, "attr": "IDEAM", "type": "wms"
        }
    }

    selected_base_map_name = container_object.selectbox(
        "Seleccionar Mapa Base",
        list(base_map_options.keys()),
        key=f"{key_prefix}_base_map"
    )

    selected_overlays_names = container_object.multiselect(
        "Seleccionar Capas Adicionales",
        list(overlay_map_options.keys()),
        key=f"{key_prefix}_overlays"
    )

    selected_base_map_config = base_map_options[selected_base_map_name]
    selected_overlays_config = [overlay_map_options[name] for name in selected_overlays_names]

    return selected_base_map_config, selected_overlays_config

def create_enso_chart(enso_data):
    if enso_data.empty or Config.ENSO_ONI_COL not in enso_data.columns:
        return go.Figure()

    data = enso_data.copy().sort_values(Config.DATE_COL)
    data.dropna(subset=[Config.ENSO_ONI_COL], inplace=True)
    
    if data.empty:
        return go.Figure()

    conditions = [data[Config.ENSO_ONI_COL] >= 0.5, data[Config.ENSO_ONI_COL] <= -0.5]
    phases = ['El Niño', 'La Niña']
    colors = ['red', 'blue']
    
    data['phase'] = np.select(conditions, phases, default='Neutral')
    data['color'] = np.select(conditions, colors, default='grey')
    
    y_range = [data[Config.ENSO_ONI_COL].min() - 0.5, data[Config.ENSO_ONI_COL].max() + 0.5]

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data[Config.DATE_COL], y=[y_range[1] - y_range[0]] * len(data),
        base=y_range[0], marker_color=data['color'], opacity=0.3,
        hoverinfo='none', showlegend=False
    ))
    
    legend_map = {'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'grey'}
    for phase, color in legend_map.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=15, color=color, symbol='square', opacity=0.5), name=phase, showlegend=True))
        
    fig.add_trace(go.Scatter(x=data[Config.DATE_COL], y=data[Config.ENSO_ONI_COL], mode='lines', name='Anomalía ONI', line=dict(color='black', width=2), showlegend=True))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue")
    
    fig.update_layout(height=600, title="Fases del Fenómeno ENSO y Anomalía ONI",
                      yaxis_title="Anomalía ONI (°C)", xaxis_title="Fecha", showlegend=True,
                      legend_title_text='Fase', yaxis_range=y_range)
                      
    return fig
    
def create_anomaly_chart(df_plot):
    if df_plot.empty:
        return go.Figure()
        
    df_plot['color'] = np.where(df_plot['anomalia'] < 0, 'red', 'blue')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot[Config.DATE_COL], 
        y=df_plot['anomalia'],
        marker_color=df_plot['color'], 
        name='Anomalía de Precipitación'
    ))
                         
    fig.update_layout(
        height=600, 
        title="Anomalías Mensuales de Precipitación y Fases ENSO",
        yaxis_title="Anomalía de Precipitación (mm)", 
        xaxis_title="Fecha", 
        showlegend=True
    )
                      
    return fig

def generate_station_popup_html(row, df_anual_melted):
    station_name = row.get(Config.STATION_NAME_COL, 'N/A')
    
    try:
        year_range_val = st.session_state.get('year_range', (2000, 2020))
        year_min, year_max = year_range_val
        total_years_in_period = year_max - year_min + 1
        
        df_station_data = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station_name]
        
        precip_media_anual = df_station_data['precipitation'].mean() if not df_station_data.empty else 0
        valid_years = df_station_data['precipitation'].count() if not df_station_data.empty else 0
        
        precip_formatted = f"{precip_media_anual:.0f}" if pd.notna(precip_media_anual) else "N/A"
        
        text_html = f"<h4>{station_name}</h4>"
        text_html += f"<p><b>Municipio:</b> {row.get(Config.MUNICIPALITY_COL, 'N/A')}</p>"
        text_html += f"<p><b>Altitud:</b> {row.get(Config.ALTITUDE_COL, 'N/A')} m</p>"
        text_html += f"<p><b>Promedio Anual:</b> {precip_formatted} mm</p>"
        text_html += f"<small>(Calculado con <b>{valid_years}</b> de <b>{total_years_in_period}</b> años del período)</small>"
        
        return folium.Popup(text_html, max_width=450)
        
    except Exception as e:
        st.warning(f"No se pudo generar el popup para '{station_name}'. Razón: {e}")
        return folium.Popup(f"<h4>{station_name}</h4><p>Error al cargar datos del popup.</p>", max_width=450)

def generate_annual_map_popup_html(row, df_anual_melted_full_period):
    station_name = row.get(Config.STATION_NAME_COL, 'N/A')
    municipality = row.get(Config.MUNICIPALITY_COL, 'N/A')
    altitude = row.get(Config.ALTITUDE_COL, 'N/A')
    precip_year = row.get(Config.PRECIPITATION_COL, 'N/A')
    
    station_full_data = df_anual_melted_full_period[df_anual_melted_full_period[Config.STATION_NAME_COL] == station_name]
    
    precip_avg, precip_max, precip_min = "N/A", "N/A", "N/A"
    
    if not station_full_data.empty:
        precip_avg = f"{station_full_data[Config.PRECIPITATION_COL].mean():.0f}"
        precip_max = f"{station_full_data[Config.PRECIPITATION_COL].max():.0f}"
        precip_min = f"{station_full_data[Config.PRECIPITATION_COL].min():.0f}"
        
    altitude_formatted = f"{altitude:.0f}" if isinstance(altitude, (int, float)) and np.isfinite(altitude) else "N/A"
    precip_year_formatted = f"{precip_year:.0f}" if isinstance(precip_year, (int, float)) and np.isfinite(precip_year) else "N/A"
    
    html = f"""
    <h4>{station_name}</h4>
    <p><b>Municipio:</b> {municipality}</p>
    <p><b>Altitud:</b> {altitude_formatted} m</p>
    <hr>
    <p><b>Precipitación del Año:</b> {precip_year_formatted} mm</p>
    <p><b>Promedio Anual (histórico):</b> {precip_avg} mm</p>
    <p><small><b>Máxima del período:</b> {precip_max} mm</small></p>
    <p><small><b>Mínima del período:</b> {precip_min} mm</small></p>
    """
    return folium.Popup(html, max_width=300)

def create_folium_map(location, zoom, base_map_config, overlays_config,
                      fit_bounds_data=None):
    """Crea un mapa base de Folium y le añade capas de overlay de forma inteligente."""
    m = folium.Map(
        location=location,
        zoom_start=zoom,
        tiles=base_map_config.get('tiles', 'OpenStreetMap'),
        attr=base_map_config.get('attr', 'OpenStreetMap')
    )

    if fit_bounds_data is not None and not fit_bounds_data.empty:
        if len(fit_bounds_data) > 1:
            bounds = fit_bounds_data.total_bounds
            if np.all(np.isfinite(bounds)):
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        elif len(fit_bounds_data) == 1:
            point = fit_bounds_data.iloc[0].geometry
            m.location = [point.y, point.x]
            m.zoom_start = 12

    if overlays_config:
        for layer_config in overlays_config:
            layer_type = layer_config.get("type")
            layer_name = layer_config.get("name", "Overlay")

            # Lógica para capas WMS (como la de IDEAM)
            if layer_type == "wms":
                WmsTileLayer(
                    url=layer_config["url"],
                    layers=layer_config["layers"],
                    fmt=layer_config.get("fmt", 'image/png'),
                    transparent=layer_config.get("transparent", True),
                    overlay=True,
                    control=True,
                    name=layer_name
                ).add_to(m)
            
            # Lógica para capas GeoJSON (Municipios, Predios, etc.)
            elif layer_type == "geojson":
                data_key = layer_config.get("data_key")
                if data_key in st.session_state and st.session_state[data_key] is not None and not st.session_state[data_key].empty:
                    folium.GeoJson(
                        st.session_state[data_key],
                        name=layer_name
                    ).add_to(m)
    return m
    
# --- MAIN TAB DISPLAY FUNCTIONS
def display_welcome_tab():
    st.header("Bienvenido al Sistema de Información de Lluvias y Clima")
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css?family=Playfair+Display:wght@700&display=swap');
    .quote { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 22px; text-align: center; padding: 20px; }
    .author { font-family: 'Playfair Display', serif; text-align: right; font-style: italic; font-size: 18px; padding-right: 20px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<p class="quote">{Config.QUOTE_TEXT}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="author">- {Config.QUOTE_AUTHOR}</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.markdown(Config.WELCOME_TEXT, unsafe_allow_html=True)
    
    with st.expander("La Inspiración: Chaac, Divinidad Maya", expanded=False):
        st.markdown(Config.CHAAC_STORY)
        
    with col2:
        if os.path.exists(Config.CHAAC_IMAGE_PATH):
            st.image(Config.CHAAC_IMAGE_PATH, caption="Representación de Chaac, Códice de Dresde.")
        if os.path.exists(Config.LOGO_PATH):
            st.image(Config.LOGO_PATH, width=250, caption="Corporación Cuenca Verde")
    st.markdown("---")
    
    with st.expander("Conceptos Clave, Métodos y Ecuaciones", expanded=True):
        st.markdown("""
        Esta sección proporciona una descripción de los métodos y conceptos analíticos utilizados en la plataforma.
        
        ### Análisis de Anomalías
        Una **anomalía** representa la diferencia entre el valor observado en un momento dado y un valor de referencia o "normal".
        
        - **Anomalía vs. Período Seleccionado**: Compara la lluvia de cada mes con el promedio histórico de todos los meses iguales (ej. todos los eneros, febreros, etc.) en el conjunto de datos.
        - **Anomalía vs. Normal Climatológica**: Compara la lluvia de cada mes con el promedio de un período de referencia de 30 años (ej. 1991-2020), según recomienda la Organización Meteorológica Mundial.

        ### Métodos de Interpolación Espacial
        La interpolación estima la precipitación en lugares sin estaciones de medición.
        
        - **IDW (Inverso de la Distancia Ponderada)**: Método que asume que los puntos más cercanos tienen más influencia.
        - **Kriging**: Método geoestadístico que usa la autocorrelación espacial (variograma) para estimaciones más precisas.
        - **Spline (Thin Plate)**: Método matemático que ajusta una superficie flexible a los datos.

        ### Índices de Sequía
        Estandarizan la precipitación para comparar la intensidad de sequías y períodos húmedos.
        
        - **SPI (Índice de Precipitación Estandarizado)**: Mide la desviación de la precipitación respecto a su media histórica.
        - **SPEI (Índice Estandarizado de Precipitación-Evapotranspiración)**: Versión avanzada del SPI que incluye la evapotranspiración, haciéndolo más relevante en estudios de cambio climático.

        ### Análisis de Frecuencia de Extremos
        - **Período de Retorno**: Estimación estadística de la probabilidad de que un evento extremo ocurra. Un período de retorno de 100 años tiene una probabilidad del 1% de ocurrir en cualquier año.

        ### Análisis de Tendencias
        - **Prueba de Mann-Kendall**: Prueba no paramétrica para detectar tendencias monótonas (crecientes o decrecientes) en una serie de tiempo.
        - **Pendiente de Sen**: Cuantifica la magnitud de la tendencia detectada por Mann-Kendall (ej. "aumento de 5 mm/año").

        ### Modelos de Pronóstico
        - **SARIMA**: Modelo estadístico clásico para series de tiempo que descompone los datos en tendencia, estacionalidad y ruido.
        - **Prophet**: Modelo de Facebook, automático y robusto, ideal para series con fuertes efectos estacionales y datos faltantes.
        """)

def display_spatial_distribution_tab(gdf_filtered, stations_for_analysis, df_anual_melted,
                                     df_monthly_filtered, analysis_mode, selected_regions,
                                     selected_municipios, selected_altitudes, **kwargs):
    st.header("Distribución espacial de las Estaciones de Lluvia")
    display_filter_summary(total_stations_count=len(st.session_state.gdf_stations),
                           selected_stations_count=len(stations_for_analysis),
                           year_range=st.session_state.year_range,
                           selected_months_count=len(st.session_state.meses_numeros),
                           analysis_mode=analysis_mode, selected_regions=selected_regions,
                           selected_municipios=selected_municipios,
                           selected_altitudes=selected_altitudes)

    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    
    gdf_display = gdf_filtered.copy()
    sub_tab_mapa, sub_tab_grafico = st.tabs(["Mapa Interactivo", "Gráfico de Disponibilidad de Datos"])

    with sub_tab_mapa:
        controls_col, map_col = st.columns([1, 3])
        with controls_col:
            st.subheader("Controles del Mapa")
            if st.button("Guardar Mapa en Dashboard", key="pin_spatial_map"):
                params = {"stations": stations_for_analysis}
                db_manager.save_preference(
                    username=st.session_state["username"],
                    widget_type="spatial_map",
                    params=params
                )
                st.toast("¡Mapa de distribución guardado!", icon="🗺️")
            
            selected_base_map_config, selected_overlays_config = display_map_controls(st, "dist_esp")
            if not gdf_filtered.empty:
                st.metric("Estaciones en Vista", len(gdf_filtered))

        with map_col:
            if not gdf_filtered.empty:
                m = create_folium_map(
                    location=[6.2, -75.5],
                    zoom=7,
                    base_map_config=selected_base_map_config,
                    overlays_config=selected_overlays_config,
                    fit_bounds_data=gdf_filtered
                )
                marker_cluster = MarkerCluster(name='Estaciones').add_to(m)
                for _, row in gdf_filtered.iterrows():
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        tooltip=row[Config.STATION_NAME_COL]
                    ).add_to(marker_cluster)
                
                # CORRECCIÓN FINAL APLICADA AQUÍ
                st_folium(m, height=500, use_container_width=True)
            else:
                st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

    with sub_tab_grafico:
        st.subheader("Disponibilidad y Composición de Datos por Estación")
        if not gdf_display.empty:
            if analysis_mode == "Completar series (interpolación)":
                st.info("Mostrando la composición de datos originales vs. completados para el período seleccionado.")
                if not df_monthly_filtered.empty and Config.ORIGIN_COL in df_monthly_filtered.columns:
                    data_composition = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.ORIGIN_COL]).size().unstack(fill_value=0)
                    if 'Original' not in data_composition: data_composition['Original'] = 0
                    if 'Completado' not in data_composition: data_composition['Completado'] = 0

                    data_composition['total'] = data_composition['Original'] + data_composition['Completado']
                    data_composition['% Original'] = (data_composition['Original'] / data_composition['total']) * 100
                    data_composition['% Completado'] = (data_composition['Completado'] / data_composition['total']) * 100

                    sort_order_comp = st.radio("Ordenar por:", ["% Datos Originales (Mayor a Menor)", "% Datos Originales (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_comp")
                    if "Mayor a Menor" in sort_order_comp:
                        data_composition = data_composition.sort_values("% Original", ascending=False)
                    elif "Menor a Mayor" in sort_order_comp:
                        data_composition = data_composition.sort_values("% Original", ascending=True)
                    else:
                        data_composition = data_composition.sort_index(ascending=True)

                    df_plot = data_composition.reset_index().melt(id_vars=Config.STATION_NAME_COL, value_vars=['% Original', '% Completado'], var_name='Tipo de Dato', value_name='Porcentaje')
                    
                    fig_comp = px.bar(
                        df_plot,
                        x=Config.STATION_NAME_COL,
                        y='Porcentaje',
                        color='Tipo de Dato',
                        title='Composición de Datos por Estación',
                        labels={Config.STATION_NAME_COL: 'Estación', 'Porcentaje': '% del Período'},
                        text_auto='.1f',
                        color_discrete_map={'% Original': '#1f77b4', '% Completado': '#ff7f0e'}
                    )
                    fig_comp.update_layout(height=500, xaxis={'categoryorder': 'trace'}, barmode='stack')
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.warning("No hay datos mensuales procesados para mostrar la composición.")
            else:
                st.info("Mostrando el porcentaje de disponibilidad de datos según el archivo de estaciones.")
                sort_order_disp = st.radio("Ordenar estaciones por:", ["% Datos (Mayor a Menor)", "% Datos (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_disp")
                df_chart = gdf_display.copy()
                if Config.PERCENTAGE_COL in df_chart.columns:
                    if "% Datos (Mayor a Menor)" in sort_order_disp:
                        df_chart = df_chart.sort_values(Config.PERCENTAGE_COL, ascending=False)
                    elif "% Datos (Menor a Mayor)" in sort_order_disp:
                        df_chart = df_chart.sort_values(Config.PERCENTAGE_COL, ascending=True)
                    else:
                        df_chart = df_chart.sort_values(Config.STATION_NAME_COL, ascending=True)
                    
                    fig_disp = px.bar(
                        df_chart,
                        x=Config.STATION_NAME_COL,
                        y=Config.PERCENTAGE_COL,
                        title='Porcentaje de Disponibilidad de Datos Históricos',
                        labels={Config.STATION_NAME_COL: 'Estación', Config.PERCENTAGE_COL: '% de Datos Disponibles'},
                        color=Config.PERCENTAGE_COL,
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    fig_disp.update_layout(height=500, xaxis={'categoryorder': 'trace'})
                    st.plotly_chart(fig_disp, use_container_width=True)
                else:
                    st.warning("La columna con el porcentaje de datos no se encuentra en el archivo de estaciones.")
        else:
            st.warning("No hay estaciones seleccionadas para mostrar el gráfico.")

def display_graphs_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis,
                       gdf_filtered, analysis_mode, selected_regions, selected_municipios, selected_altitudes, **kwargs):
    st.header("Visualizaciones de Precipitación")
    display_filter_summary(total_stations_count=len(st.session_state.gdf_stations),
                           selected_stations_count=len(stations_for_analysis),
                           year_range=st.session_state.year_range,
                           selected_months_count=len(st.session_state.get('meses_numeros', [])),
                           analysis_mode=analysis_mode, selected_regions=selected_regions,
                           selected_municipios=selected_municipios, selected_altitudes=selected_altitudes)

    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    year_range_val = st.session_state.get('year_range', (2000, 2020))
    if isinstance(year_range_val, tuple) and len(year_range_val) == 2 and isinstance(year_range_val[0], int):
        year_min, year_max = year_range_val
    else:
        # Fallback por si el rango no está bien definido
        year_min, year_max = (df_anual_melted[Config.YEAR_COL].min(), df_anual_melted[Config.YEAR_COL].max()) if not df_anual_melted.empty else (2000, 2020)

    # --- Lógica para enriquecer los datos con metadatos ---
    metadata_cols = [Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.ALTITUDE_COL]
    gdf_metadata = gdf_filtered[metadata_cols].drop_duplicates(subset=[Config.STATION_NAME_COL]).copy()
    
    if Config.ALTITUDE_COL in gdf_metadata.columns:
        gdf_metadata[Config.ALTITUDE_COL] = pd.to_numeric(gdf_metadata[Config.ALTITUDE_COL], errors='coerce').fillna(-9999).astype(int).astype(str)
    if Config.MUNICIPALITY_COL in gdf_metadata.columns:
        gdf_metadata[Config.MUNICIPALITY_COL] = gdf_metadata[Config.MUNICIPALITY_COL].astype(str).str.strip().replace('nan', 'Sin Dato')

    cols_to_drop = [col for col in [Config.MUNICIPALITY_COL, Config.ALTITUDE_COL] if col != Config.STATION_NAME_COL]
    
    df_anual_pre_merge = df_anual_melted.drop(columns=cols_to_drop, errors='ignore')
    df_anual_rich = pd.merge(df_anual_pre_merge, gdf_metadata, on=Config.STATION_NAME_COL, how='left')

    df_monthly_pre_merge = df_monthly_filtered.drop(columns=cols_to_drop, errors='ignore')
    df_monthly_rich = pd.merge(df_monthly_pre_merge, gdf_metadata, on=Config.STATION_NAME_COL, how='left')
    
    # --- Creación de Pestañas ---
    tab_keys = ["Análisis Anual", "Análisis Mensual", "Comparación Rápida", "Boxplot Anual", "Distribución", "Acumulada", "Relación Altitud", "Serie Regional", "Descargar Datos"]
    sub_tab_anual, sub_tab_mensual, sub_tab_comparacion, sub_tab_boxplot, sub_tab_distribucion, sub_tab_acumulada, sub_tab_altitud, sub_tab_regional, sub_tab_descarga = st.tabs(tab_keys)
                           
    with sub_tab_anual:
        anual_graf_tab, anual_analisis_tab = st.tabs(["Gráfico de Serie Anual", "Análisis Multianual"])

        with anual_graf_tab:
            if not df_anual_rich.empty:
                st.subheader("Precipitación Anual (mm)")
                if st.button("📌 Guardar en Dashboard", key="pin_anual_series"):
                    params = {
                        "stations": stations_for_analysis,
                        "year_range": list(year_range_val)
                    }
                    db_manager.save_preference(username=st.session_state["username"], widget_type="annual_series_chart", params=params)
                    st.toast("¡Gráfico de serie anual guardado!", icon="✅")
                
                st.info("Solo se muestran los años con 10 o más meses de datos válidos.")
                chart_anual = alt.Chart(df_anual_rich.dropna(subset=[Config.PRECIPITATION_COL])).mark_line(point=True).encode(
                    x=alt.X(f'{Config.YEAR_COL}:O', title='Año'),
                    y=alt.Y(f'{Config.PRECIPITATION_COL}:Q', title='Precipitación (mm)'),
                    color=f'{Config.STATION_NAME_COL}:N',
                    tooltip=[
                        alt.Tooltip(Config.STATION_NAME_COL),
                        alt.Tooltip(Config.YEAR_COL, format='d', title='Año'),
                        alt.Tooltip(f'{Config.PRECIPITATION_COL}:Q', format='.0f', title='Ppt. Anual (mm)'),
                        alt.Tooltip(f'{Config.MUNICIPALITY_COL}:N', title='Municipio'),
                        alt.Tooltip(f'{Config.ALTITUDE_COL}:N', title='Altitud (m)')
                    ]
                ).properties(title=f"Precipitación Anual por Estación ({year_min} - {year_max})").interactive()
                st.altair_chart(chart_anual, use_container_width=True)
            else:
                st.warning("No hay datos anuales para mostrar la serie.")

        with anual_analisis_tab:
            if not df_anual_rich.empty:
                st.subheader("Precipitación Media Multianual")
                if st.button("📌 Guardar en Dashboard", key="pin_anual_avg_bar"):
                    params = {"stations": stations_for_analysis}
                    db_manager.save_preference(username=st.session_state["username"], widget_type="annual_avg_bar_chart", params=params)
                    st.toast("¡Gráfico de promedio anual guardado!", icon="✅")
                
                st.caption(f"Período de análisis: {year_min} - {year_max}")
                chart_type_annual = st.radio("Seleccionar tipo de gráfico:", ("Gráfico de Barras (Promedio)", "Gráfico de Cajas (Distribución)"), key="avg_chart_type_annual", horizontal=True)

                if chart_type_annual == "Gráfico de Barras (Promedio)":
                    df_summary = df_anual_rich.groupby(Config.STATION_NAME_COL, as_index=False)[Config.PRECIPITATION_COL].mean().round(0)
                    sort_order = st.radio("Ordenar estaciones por:", ["Promedio (Mayor a Menor)", "Promedio (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_annual_avg")
                    
                    if "Mayor a Menor" in sort_order: df_summary = df_summary.sort_values(Config.PRECIPITATION_COL, ascending=False)
                    elif "Menor a Mayor" in sort_order: df_summary = df_summary.sort_values(Config.PRECIPITATION_COL, ascending=True)
                    else: df_summary = df_summary.sort_values(Config.STATION_NAME_COL, ascending=True)

                    fig_avg = px.bar(df_summary, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL, 
                                     title=f'Promedio de Precipitación Anual ({year_min} - {year_max})',
                                     labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Ppt. Media Anual (mm)'},
                                     color=Config.PRECIPITATION_COL)
                    st.plotly_chart(fig_avg, use_container_width=True)
                else: 
                    fig_box = px.box(df_anual_rich, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL,
                                     color=Config.STATION_NAME_COL, points='all', title='Distribución de la Precipitación Anual',
                                     labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Precipitación Anual (mm)'})
                    st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("No hay datos anuales para el análisis multianual.")

    with sub_tab_mensual:
        mensual_graf_tab, mensual_enso_tab, mensual_datos_tab = st.tabs(["Gráfico de Serie Mensual", "Análisis ENSO en el Período", "Tabla de Datos"])
        with mensual_graf_tab:
            if not df_monthly_rich.empty:
                controls_col, chart_col = st.columns([1, 4])
                with controls_col:
                    st.markdown("##### Opciones del Gráfico")
                    chart_type = st.radio("Tipo de Gráfico:", ["Líneas y Puntos", "Nube de Puntos", "Gráfico de Cajas (Distribución Mensual)"], key="monthly_chart_type")
                    color_by_disabled = (chart_type == "Gráfico de Cajas (Distribución Mensual)")
                    color_by = st.radio("Colorear por:", ["Estación", "Mes"], key="monthly_color_by", disabled=color_by_disabled)
                with chart_col:
                    if chart_type != "Gráfico de Cajas (Distribución Mensual)":
                        base_chart = alt.Chart(df_monthly_rich).encode(
                            x=alt.X(f'{Config.DATE_COL}:T', title='Fecha'),
                            y=alt.Y(f'{Config.PRECIPITATION_COL}:Q', title='Precipitación (mm)'),
                            tooltip=[
                                alt.Tooltip(f'{Config.DATE_COL}:T', format='%Y-%m', title='Fecha'),
                                alt.Tooltip(f'{Config.PRECIPITATION_COL}:Q', format='.0f', title='Ppt. Mensual'),
                                alt.Tooltip(f'{Config.STATION_NAME_COL}:N', title='Estación'),
                                alt.Tooltip(f'{Config.MONTH_COL}:N', title="Mes"),
                                alt.Tooltip(f'{Config.MUNICIPALITY_COL}:N', title='Municipio'),
                                alt.Tooltip(f'{Config.ALTITUDE_COL}:N', title='Altitud (m)')
                            ]
                        )
                        if color_by == "Mes":
                            color_encoding = alt.Color(f'month({Config.DATE_COL}):N', legend=alt.Legend(title="Meses"), scale=alt.Scale(scheme='tableau20'))
                        else:
                            color_encoding = alt.Color(f'{Config.STATION_NAME_COL}:N', legend=alt.Legend(title="Estaciones"))
                        
                        if chart_type == "Líneas y Puntos":
                            line_chart = base_chart.mark_line(opacity=0.4, color='lightgray').encode(detail=f'{Config.STATION_NAME_COL}:N')
                            point_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                            final_chart = line_chart + point_chart
                        else:
                            final_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                        st.altair_chart(final_chart.properties(height=500, title=f"Serie de Precipitación Mensual ({year_min} - {year_max})").interactive(), use_container_width=True)
                    else:
                        st.subheader("Distribución de la Precipitación Mensual")
                        fig_box_monthly = px.box(df_monthly_rich, x=Config.MONTH_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, title='Distribución de la Precipitación por Mes', labels={Config.MONTH_COL: 'Mes', Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)', Config.STATION_NAME_COL: 'Estación'})
                        fig_box_monthly.update_layout(height=500)
                        st.plotly_chart(fig_box_monthly, use_container_width=True)
            else:
                st.warning("No hay datos mensuales para mostrar el gráfico.")

        with mensual_enso_tab:
            if 'df_enso' in st.session_state and st.session_state.df_enso is not None:
                # Obtenemos los años del slider para filtrar los datos del ENSO
                year_min = st.session_state.get('year_range', [2000, 2020])[0]
                year_max = st.session_state.get('year_range', [2000, 2020])[1]
                
                enso_filtered = st.session_state.df_enso[
                    (st.session_state.df_enso[Config.DATE_COL].dt.year >= year_min) &
                    (st.session_state.df_enso[Config.DATE_COL].dt.year <= year_max) &
                    (st.session_state.df_enso[Config.DATE_COL].dt.month.isin(st.session_state.get('meses_numeros', range(1, 13))))
                ]
                fig_enso_mensual = create_enso_chart(enso_filtered)
                st.plotly_chart(fig_enso_mensual, use_container_width=True, key="enso_chart_mensual")
            else:
                st.info("No hay datos ENSO disponibles para este análisis.")

        with mensual_datos_tab:
            st.subheader("Datos de Precipitación Mensual Detallados")
            if not df_monthly_rich.empty:
                df_values = df_monthly_rich.pivot_table(index=Config.DATE_COL, columns=Config.STATION_NAME_COL, values=Config.PRECIPITATION_COL).round(1)
                st.dataframe(df_values, use_container_width=True)
            else:
                st.info("No hay datos mensuales detallados.")

    with sub_tab_comparacion:
        st.subheader("Comparación de Precipitación Mensual entre Estaciones")
        if len(stations_for_analysis) < 2:
            st.info("Seleccione al menos dos estaciones para comparar.")
        else:
            st.markdown("##### Precipitación Mensual Promedio")

            # 1. Agrupa y calcula solo los valores numéricos
            df_monthly_avg = df_monthly_rich.groupby([Config.STATION_NAME_COL, Config.MONTH_COL]).agg(
                precip_promedio=(Config.PRECIPITATION_COL, 'mean'),
                precip_max=(Config.PRECIPITATION_COL, 'max'),
                precip_min=(Config.PRECIPITATION_COL, 'min')
            ).reset_index()

            # 2. Obtiene los metadatos únicos de las estaciones
            station_meta = df_monthly_rich[[
                Config.STATION_NAME_COL, 
                Config.MUNICIPALITY_COL, 
                Config.ALTITUDE_COL
            ]].drop_duplicates(subset=[Config.STATION_NAME_COL])

            # 3. Une los datos agregados con los metadatos
            df_monthly_avg_rich = pd.merge(df_monthly_avg, station_meta, on=Config.STATION_NAME_COL, how='left')

            fig_avg_monthly = px.line(df_monthly_avg_rich, x=Config.MONTH_COL,
                y='precip_promedio', color=Config.STATION_NAME_COL,
                labels={Config.MONTH_COL: 'Mes', 'precip_promedio': 'Precipitación Promedio (mm)'},
                title='Promedio de Precipitación Mensual por Estación',
                hover_data={
                    Config.MUNICIPALITY_COL: True, 
                    Config.ALTITUDE_COL: True, 
                    'precip_max': ':.0f',
                    'precip_min': ':.0f'
                })
            
            meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
            fig_avg_monthly.update_layout(height=500, xaxis=dict(tickmode='array', tickvals=list(meses_dict.values()), ticktext=list(meses_dict.keys())))
            st.plotly_chart(fig_avg_monthly, use_container_width=True)

    with sub_tab_boxplot:
        st.subheader("Distribución de Precipitación Anual por Estación (Boxplot)")
        if not df_anual_rich.empty:
            fig_box_annual = px.box(df_anual_rich, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL,
                                    color=Config.STATION_NAME_COL, points='all', title='Distribución de la Precipitación Anual por Estación',
                                    labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Precipitación Anual (mm)'})
            st.plotly_chart(fig_box_annual, use_container_width=True)
        else:
            st.warning("No hay datos anuales para mostrar el gráfico de cajas.")

    with sub_tab_distribucion:
        st.subheader("Distribución de la Precipitación")
        distribucion_tipo = st.radio("Seleccionar tipo de distribución:", ("Anual", "Mensual"), horizontal=True)
        plot_type = st.radio("Seleccionar tipo de gráfico:", ("Histograma", "Gráfico de Violín"), horizontal=True, key="distribucion_plot_type")
        
        df_to_plot = df_anual_rich if distribucion_tipo == "Anual" else df_monthly_rich
        x_axis = Config.STATION_NAME_COL if distribucion_tipo == "Anual" else Config.MONTH_COL

        if not df_to_plot.empty:
            if plot_type == "Histograma":
                fig_dist = px.histogram(df_to_plot, x=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                       title=f'Distribución {distribucion_tipo} de Precipitación',
                                       labels={Config.PRECIPITATION_COL: f'Precipitación {distribucion_tipo} (mm)'})
            else: # Gráfico de Violín
                fig_dist = px.violin(df_to_plot, y=Config.PRECIPITATION_COL, x=x_axis, color=Config.STATION_NAME_COL,
                                     box=True, points="all", title=f'Distribución {distribucion_tipo} con Gráfico de Violín')
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning(f"No hay datos {distribucion_tipo.lower()}es para mostrar.")

    with sub_tab_acumulada:
        st.subheader("Precipitación Acumulada Anual")
        if not df_anual_rich.empty:
            fig_acumulada = px.bar(df_anual_rich, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                   title=f'Precipitación Anual por Año ({year_min} - {year_max})',
                                   labels={Config.YEAR_COL: 'Año', Config.PRECIPITATION_COL: 'Precipitación (mm)'},
                                   barmode='group')
            st.plotly_chart(fig_acumulada, use_container_width=True)
        else:
            st.warning("No hay datos para calcular la precipitación acumulada.")

    with sub_tab_altitud:
        st.subheader("Relación entre Altitud y Precipitación Media Anual")
        if not df_anual_rich.empty and pd.to_numeric(df_anual_rich[Config.ALTITUDE_COL], errors='coerce').notna().any():
            df_relacion = df_anual_rich.groupby(Config.STATION_NAME_COL).agg(
                precip_media=(Config.PRECIPITATION_COL, 'mean'),
                altitud=(Config.ALTITUDE_COL, 'first')
            ).reset_index()
            fig_relacion = px.scatter(df_relacion, x='altitud', y='precip_media',
                                      color=Config.STATION_NAME_COL, title='Precipitación Media Anual vs. Altitud',
                                      labels={'altitud': 'Altitud (m)', 'precip_media': 'Precipitación Media Anual (mm)'})
            st.plotly_chart(fig_relacion, use_container_width=True)
        else:
            st.info("No hay datos de altitud o precipitación disponibles para analizar la relación.")
            
    with sub_tab_regional:
        st.subheader("Serie de Tiempo Promedio Regional (Múltiples Estaciones)")
        if not df_monthly_rich.empty:
            df_regional_avg = df_monthly_rich.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean().reset_index()
            fig_regional = px.line(df_regional_avg, x=Config.DATE_COL, y=Config.PRECIPITATION_COL, title="Serie de Tiempo Promedio Regional")
            st.plotly_chart(fig_regional, use_container_width=True)
        else:
            st.warning("No hay datos mensuales para calcular la serie regional.")

    with sub_tab_descarga:
        st.subheader("Descargar Datos de la Pestaña Gráficos")
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False, sep=';').encode('utf-8')

        if not df_anual_rich.empty:
            st.download_button("Descargar Datos Anuales (CSV)", convert_df_to_csv(df_anual_rich), "datos_anuales.csv", "text/csv")
        
        if not df_monthly_rich.empty:
            st.download_button("Descargar Datos Mensuales (CSV)", convert_df_to_csv(df_monthly_rich), "datos_mensuales.csv", "text/csv")

def display_advanced_maps_tab(gdf_filtered, stations_for_analysis, df_anual_melted,
                              df_monthly_filtered, analysis_mode, selected_regions, selected_municipios,
                              selected_altitudes, **kwargs):
    st.header("Mapas Avanzados")
    display_filter_summary(total_stations_count=len(st.session_state.gdf_stations),
                           selected_stations_count=len(stations_for_analysis),
                           year_range=st.session_state.year_range,
                           selected_months_count=len(st.session_state.meses_numeros),
                           analysis_mode=analysis_mode, selected_regions=selected_regions,
                           selected_municipios=selected_municipios, selected_altitudes=selected_altitudes)

    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    tab_names = ["Animación GIF", "Superficies de Interpolación", "Validación Cruzada (LOOCV)", "Visualización Temporal", "Gráfico de Carrera", "Mapa Animado", "Comparación de Mapas"]
    gif_tab, kriging_tab, validation_tab, temporal_tab, race_tab, anim_tab, compare_tab = st.tabs(tab_names)

    with gif_tab:
        st.subheader("Distribución Espacio-Temporal de la Lluvia en Antioquia")
        gif_path = Config.GIF_PATH
        if os.path.exists(gif_path):
            try:
                with open(gif_path, "rb") as f:
                    gif_bytes = f.read()
                gif_b64 = base64.b64encode(gif_bytes).decode("utf-8")
                html_string = f'<img src="data:image/gif;base64,{gif_b64}" width="600" alt="Animación PPAM">'
                st.markdown(html_string, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Ocurrió un error al intentar mostrar el GIF: {e}")
        else:
            st.error(f"No se pudo encontrar el archivo GIF en la ruta especificada: {gif_path}")
                
    with temporal_tab:
        st.subheader("Explorador Anual de Precipitación")
        df_anual_melted_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        if not df_anual_melted_non_na.empty:
            all_years_int = sorted(df_anual_melted_non_na[Config.YEAR_COL].unique())
            controls_col, map_col = st.columns([1, 3])
            with controls_col:
                st.markdown("##### Opciones de Visualización")
                selected_base_map_config, selected_overlays_config = display_map_controls(st, "temporal")
                selected_year = None
                if len(all_years_int) > 1:
                    selected_year = st.slider('Seleccione un Año para Explorar', min_value=min(all_years_int), max_value=max(all_years_int), value=min(all_years_int), key="temporal_year_slider")
                elif len(all_years_int) == 1:
                    selected_year = all_years_int[0]
                    st.info(f"Mostrando único año disponible: {selected_year}")

            with map_col:
                if selected_year:
                    m_temporal = create_folium_map([4.57, -74.29], 5, selected_base_map_config, selected_overlays_config)

                    df_year_filtered = df_anual_melted_non_na[df_anual_melted_non_na[Config.YEAR_COL] == selected_year]
                    if not df_year_filtered.empty:
                        cols_to_merge = [Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.ALTITUDE_COL, 'geometry']
                        df_map_data = pd.merge(df_year_filtered, gdf_filtered[cols_to_merge].drop_duplicates(subset=[Config.STATION_NAME_COL]),
                                               on=Config.STATION_NAME_COL, how="inner")
                        
                        if not df_map_data.empty:
                            min_val, max_val = df_anual_melted_non_na[Config.PRECIPITATION_COL].min(), df_anual_melted_non_na[Config.PRECIPITATION_COL].max()
                            if min_val >= max_val: max_val = min_val + 1
                            colormap = cm.LinearColormap(colors=plt.cm.viridis.colors, vmin=min_val, vmax=max_val)
                            
                            for _, row in df_map_data.iterrows():
                                ## AQUÍ ESTÁ LA CORRECCIÓN ##
                                popup_object = generate_annual_map_popup_html(row, df_anual_melted_non_na)
                            folium.CircleMarker(
                                location=[row['geometry'].y, row['geometry'].x], radius=5,
                                color=colormap(row[Config.PRECIPITATION_COL]), fill=True,
                                fill_color=colormap(row[Config.PRECIPITATION_COL]), fill_opacity=0.8,
                                tooltip=row[Config.STATION_NAME_COL], popup=popup_object
                            ).add_to(m_temporal)

                            temp_gdf = gpd.GeoDataFrame(df_map_data, geometry='geometry', crs=gdf_filtered.crs)
                            if not temp_gdf.empty:
                                bounds = temp_gdf.total_bounds
                                if np.all(np.isfinite(bounds)):
                                    m_temporal.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                            folium.LayerControl().add_to(m_temporal)
                            st_folium(m_temporal, height=700, use_container_width=True)

    with race_tab:
        st.subheader("Ranking Anual de Precipitación por Estación")
        df_anual_valid = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        if not df_anual_valid.empty:
            fig_racing = px.bar(
                df_anual_valid, x=Config.PRECIPITATION_COL, y=Config.STATION_NAME_COL,
                animation_frame=Config.YEAR_COL, orientation='h',
                labels={Config.PRECIPITATION_COL: 'Precipitación Anual (mm)', Config.STATION_NAME_COL: 'Estación'},
                title="Evolución de Precipitación Anual por Estación"
            )
            fig_racing.update_layout(
                height=max(600, len(stations_for_analysis) * 35),
                yaxis=dict(categoryorder='total ascending')
            )
            st.plotly_chart(fig_racing, use_container_width=True)
        else:
            st.warning("No hay suficientes datos anuales con los filtros actuales para generar el Gráfico de Carrera.")

    with anim_tab:
        st.subheader("Mapa Animado de Precipitación Anual")
        df_anual_valid = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        if not df_anual_valid.empty:
            df_anim_merged = pd.merge(
                df_anual_valid,
                gdf_filtered.drop_duplicates(subset=[Config.STATION_NAME_COL]),
                on=Config.STATION_NAME_COL, how="inner"
            )
            if not df_anim_merged.empty:
                fig_mapa_animado = px.scatter_geo(
                    df_anim_merged,
                    lat=Config.LATITUDE_COL, lon=Config.LONGITUDE_COL,
                    color=Config.PRECIPITATION_COL, size=Config.PRECIPITATION_COL,
                    hover_name=Config.STATION_NAME_COL,
                    animation_frame=Config.YEAR_COL,
                    projection='natural earth',
                    title='Precipitación Anual por Estación'
                )
                fig_mapa_animado.update_geos(fitbounds="locations", visible=True)
                st.plotly_chart(fig_mapa_animado, use_container_width=True)
            else:
                st.warning("No se pudieron combinar los datos anuales con la información geográfica de las estaciones.")
        else:
            st.warning("No hay suficientes datos anuales con los filtros actuales para generar el Mapa Animado.")

    with compare_tab:
        st.subheader("Comparación de Mapas Anuales")
        df_anual_valid = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        all_years = sorted(df_anual_valid[Config.YEAR_COL].unique())

        if len(all_years) > 1:
            control_col, map_col1, map_col2 = st.columns([1, 2, 2])
            
            with control_col:
                st.markdown("##### Controles de Mapa")
                selected_base_map_config, selected_overlays_config = display_map_controls(st, "compare")
                min_year, max_year = int(all_years[0]), int(all_years[-1])
                st.markdown("**Mapa 1**")
                year1 = st.selectbox("Seleccione el primer año", options=all_years, index=len(all_years)-1, key="compare_year1")
                st.markdown("**Mapa 2**")
                year2 = st.selectbox("Seleccione el segundo año", options=all_years, index=len(all_years)-2 if len(all_years) > 1 else 0, key="compare_year2")
                min_precip, max_precip = int(df_anual_valid[Config.PRECIPITATION_COL].min()), int(df_anual_valid[Config.PRECIPITATION_COL].max())
                if min_precip >= max_precip: max_precip = min_precip + 1
                color_range = st.slider("Rango de Escala de Color (mm)", min_precip, max_precip, (min_precip, max_precip), key="color_compare")
                colormap = cm.LinearColormap(colors=plt.cm.viridis.colors, vmin=color_range[0], vmax=color_range[1])

            def create_compare_map(data, year, col, gdf_stations_info, df_anual_full):
                col.markdown(f"**Precipitación en {year}**")
                m = create_folium_map([6.24, -75.58], 6, selected_base_map_config, selected_overlays_config)
                if not data.empty:
                    data_with_geom = pd.merge(data, gdf_stations_info, on=Config.STATION_NAME_COL)
                    gpd_data = gpd.GeoDataFrame(data_with_geom, geometry='geometry', crs=gdf_stations_info.crs)
                    for index, row in gpd_data.iterrows():
                        if pd.notna(row[Config.PRECIPITATION_COL]):
                            popup_object = generate_annual_map_popup_html(row, df_anual_full)
                            folium.CircleMarker(
                                location=[row['geometry'].y, row['geometry'].x], radius=5,
                                color=colormap(row[Config.PRECIPITATION_COL]),
                                fill=True, fill_color=colormap(row[Config.PRECIPITATION_COL]),
                                fill_opacity=0.8,
                                tooltip=row[Config.STATION_NAME_COL], popup=popup_object
                            ).add_to(m)
                    if not gpd_data.empty:
                        m.fit_bounds(gpd_data.total_bounds.tolist())
                with col:
                    st_folium(m, height=450, use_container_width=True)

            gdf_geometries = gdf_filtered[[Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.ALTITUDE_COL, 'geometry']].drop_duplicates(subset=[Config.STATION_NAME_COL])
            data_year1 = df_anual_valid[df_anual_valid[Config.YEAR_COL] == year1]
            data_year2 = df_anual_valid[df_anual_valid[Config.YEAR_COL] == year2]
            create_compare_map(data_year1, year1, map_col1, gdf_geometries, df_anual_valid)
            create_compare_map(data_year2, year2, map_col2, gdf_geometries, df_anual_valid)
        else:
            st.warning("Se necesitan datos de al menos dos años diferentes para generar la Comparación de Mapas.")

    with kriging_tab:
        st.subheader("Comparación de Superficies de Interpolación Anual")
        df_anual_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        if not stations_for_analysis:
            st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        elif df_anual_non_na.empty or len(df_anual_non_na[Config.YEAR_COL].unique()) == 0:
            st.warning("No hay suficientes datos anuales para realizar la interpolación.")
        else:
            min_year, max_year = int(df_anual_non_na[Config.YEAR_COL].min()), int(df_anual_non_na[Config.YEAR_COL].max())
            control_col, map_col1, map_col2 = st.columns([1, 2, 2])
            with control_col:
                st.markdown("#### Controles de los Mapas")
                interpolation_methods = ["Kriging Ordinario", "IDW", "Spline (Thin Plate)"]
                if Config.ELEVATION_COL in gdf_filtered.columns:
                    interpolation_methods.insert(1, "Kriging con Deriva Externa (KED)")
                st.markdown("**Mapa 1**")
                year1 = st.slider("Seleccione el año", min_year, max_year, max_year, key="interp_year1")
                method1 = st.selectbox("Método de interpolación", options=interpolation_methods, key="interp_method1")
                variogram_model1 = None
                if "Kriging" in method1:
                    variogram_options = ['linear', 'spherical', 'exponential', 'gaussian']
                    variogram_model1 = st.selectbox("Modelo de Variograma para Mapa 1", variogram_options, key="var_model_1")
                st.markdown("---")
                st.markdown("**Mapa 2**")
                year2 = st.slider("Seleccione el año", min_year, max_year, max_year - 1 if max_year > min_year else max_year, key="interp_year2")
                method2 = st.selectbox("Método de interpolación", options=interpolation_methods, index=1, key="interp_method2")
                variogram_model2 = None
                if "Kriging" in method2:
                    variogram_options = ['linear', 'spherical', 'exponential', 'gaussian']
                    variogram_model2 = st.selectbox("Modelo de Variograma para Mapa 2", variogram_options, key="var_model_2")
            
            gdf_bounds = gdf_filtered.total_bounds.tolist()
            gdf_metadata = pd.DataFrame(gdf_filtered.drop(columns='geometry', errors='ignore'))
            
            fig1, fig_var1, error1 = create_interpolation_surface(year1, method1, variogram_model1, gdf_bounds, gdf_metadata, df_anual_non_na)
            fig2, fig_var2, error2 = create_interpolation_surface(year2, method2, variogram_model2, gdf_bounds, gdf_metadata, df_anual_non_na)
            
            with map_col1:
                if fig1: st.plotly_chart(fig1, use_container_width=True)
                else: st.info(error1)
            with map_col2:
                if fig2: st.plotly_chart(fig2, use_container_width=True)
                else: st.info(error2)

            st.markdown("---")
            st.markdown("##### Variogramas de los Mapas")
            col3, col4 = st.columns(2)
            
            with col3:
                if fig_var1:
                    buf = io.BytesIO()
                    fig_var1.savefig(buf, format="png")
                    st.image(buf)
                    st.download_button(label="Descargar Variograma 1 (PNG)", data=buf.getvalue(), file_name=f"variograma_1_{year1}_{method1}.png", mime="image/png")
                    plt.close(fig_var1)
                else:
                    st.info("El variograma no está disponible para este método.")
            
            with col4:
                if fig_var2:
                    buf = io.BytesIO()
                    fig_var2.savefig(buf, format="png")
                    st.image(buf)
                    st.download_button(label="Descargar Variograma 2 (PNG)", data=buf.getvalue(), file_name=f"variograma_2_{year2}_{method2}.png", mime="image/png")
                    plt.close(fig_var2)
                else:
                    st.info("El variograma no está disponible para este método.")

    with validation_tab:
        st.subheader("Validación Cruzada Comparativa de Métodos de Interpolación")
        if len(stations_for_analysis) < 4:
            st.warning("Se necesitan al menos 4 estaciones con datos para realizar una validación robusta.")
        else:
            df_anual_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
            all_years_int = sorted(df_anual_non_na[Config.YEAR_COL].unique())
            if not all_years_int:
                st.warning("No hay años con datos válidos para la validación.")
            else:
                selected_year = st.selectbox("Seleccione un año para la validación:", options=all_years_int, index=len(all_years_int)-1, key="validation_year_select")
                if st.button(f"Ejecutar Validación para el año {selected_year}", key="run_validation_button"):
                    with st.spinner("Realizando validación cruzada..."):
                        gdf_metadata = pd.DataFrame(gdf_filtered.drop(columns='geometry', errors='ignore'))
                        validation_results_df = perform_loocv_for_all_methods(selected_year, gdf_metadata, df_anual_non_na)
                        if not validation_results_df.empty:
                            st.subheader(f"Resultados de la Validación para el Año {selected_year}")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Error Cuadrático Medio (RMSE)**")
                                fig_rmse = px.bar(validation_results_df.sort_values("RMSE"), x="Método", y="RMSE", color="Método", text_auto='.2f')
                                fig_rmse.update_layout(showlegend=False)
                                st.plotly_chart(fig_rmse, use_container_width=True)
                            with col2:
                                st.markdown("**Error Absoluto Medio (MAE)**")
                                fig_mae = px.bar(validation_results_df.sort_values("MAE"), x="Método", y="MAE", color="Método", text_auto='.2f')
                                fig_mae.update_layout(showlegend=False)
                                st.plotly_chart(fig_mae, use_container_width=True)
                            st.markdown("**Tabla Comparativa de Errores**")
                            st.dataframe(validation_results_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}"}))
                            best_rmse = validation_results_df.loc[validation_results_df['RMSE'].idxmin()]
                            st.success(f"**Mejor método según RMSE:** {best_rmse['Método']} (RMSE: {best_rmse['RMSE']:.2f})")
                        else:
                            st.error("No se pudieron calcular los resultados de la validación.")

# --- NUEVA FUNCIÓN PARA MOSTRAR EL ANÁLISIS DE EVENTOS ---
def display_event_analysis(index_values, index_type):
    """Muestra el panel de control y los resultados del análisis de eventos de sequía/humedad."""
    st.markdown("---")
    st.subheader(f"Análisis de Eventos de Sequía y Humedad ({index_type})")
    
    col1, col2 = st.columns(2)
    with col1:
        drought_threshold = st.slider("Umbral de Sequía Moderada", -2.0, 0.0, -1.0, 0.1, key=f"drought_thresh_{index_type}", help="Un evento de sequía comienza cuando el índice cae por debajo de este valor.")
        extreme_drought_threshold = st.slider("Umbral de Sequía Extrema", -3.0, -1.0, -1.5, 0.1, key=f"extreme_drought_thresh_{index_type}", help="Eventos que alcanzan un pico por debajo de este valor se consideran extremos.")
    with col2:
        wet_threshold = st.slider("Umbral de Período Húmedo", 0.0, 2.0, 1.0, 0.1, key=f"wet_thresh_{index_type}", help="Un período húmedo comienza cuando el índice supera este valor.")
        extreme_wet_threshold = st.slider("Umbral de Período Húmedo Extremo", 1.0, 3.0, 1.5, 0.1, key=f"extreme_wet_thresh_{index_type}", help="Eventos que alcanzan un pico por encima de este valor se consideran extremos.")

    # Realizar el análisis
    droughts_df = analyze_events(index_values, drought_threshold, 'drought')
    wet_periods_df = analyze_events(index_values, wet_threshold, 'wet')

    st.markdown("#### Panel Informativo de Eventos")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 💧 **Resumen de Sequías**")
        if not droughts_df.empty:
            longest_drought = droughts_df.loc[droughts_df['Duración (meses)'].idxmax()]
            most_intense = droughts_df.loc[droughts_df['Pico'].idxmin()]
            extreme_count = (droughts_df['Pico'] < extreme_drought_threshold).sum()
            
            st.metric("Sequía más Larga", f"{longest_drought['Duración (meses)']} meses", f"Inició en {longest_drought['Fecha Inicio'].strftime('%Y-%m')}")
            st.metric("Sequía más Intensa (Pico)", f"{most_intense['Pico']:.2f}", f"Inició en {most_intense['Fecha Inicio'].strftime('%Y-%m')}")
            st.metric(f"Nº de Eventos de Sequía (<{drought_threshold})", len(droughts_df))
            st.metric(f"Nº de Eventos Extremos (<{extreme_drought_threshold})", extreme_count)
        else:
            st.info("No se identificaron eventos de sequía con los umbrales seleccionados.")

    with col2:
        st.markdown("##### 🌧️ **Resumen de Períodos Húmedos**")
        # CORRECCIÓN: Comprobación robusta de que el DataFrame no está vacío
        if not wet_periods_df.empty:
            longest_wet = wet_periods_df.loc[wet_periods_df['Duración (meses)'].idxmax()]
            most_intense_wet = wet_periods_df.loc[wet_periods_df['Pico'].idxmax()]
            extreme_wet_count = (wet_periods_df['Pico'] > extreme_wet_threshold).sum()

            st.metric("Período Húmedo más Largo", f"{longest_wet['Duración (meses)']} meses", f"Inició en {longest_wet['Fecha Inicio'].strftime('%Y-%m')}")
            st.metric("Período Húmedo más Intenso (Pico)", f"{most_intense_wet['Pico']:.2f}", f"Inició en {most_intense_wet['Fecha Inicio'].strftime('%Y-%m')}")
            st.metric(f"Nº de Eventos Húmedos (>{wet_threshold})", len(wet_periods_df))
            st.metric(f"Nº de Eventos Extremos (>{extreme_wet_threshold})", extreme_wet_count)
        else:
            st.info("No se identificaron períodos húmedos con los umbrales seleccionados.")

    st.markdown("---")
    st.subheader("Visualización y Datos de Eventos")

    tab_drought, tab_wet = st.tabs(["Eventos de Sequía", "Períodos Húmedos"])

    with tab_drought:
        if not droughts_df.empty:
            fig_droughts = px.bar(
                droughts_df, x='Fecha Inicio', y='Duración (meses)', color='Intensidad',
                title='Duración e Intensidad de los Eventos de Sequía',
                hover_data=['Magnitud', 'Pico', 'Fecha Fin'],
                color_continuous_scale=px.colors.sequential.Reds_r
            )
            fig_droughts.update_layout(coloraxis_colorbar=dict(title=f"Intensidad<br>({index_type})"))
            st.plotly_chart(fig_droughts, use_container_width=True)
            with st.expander("Ver tabla de datos de eventos de sequía"):
                st.dataframe(droughts_df.style.format({
                    'Fecha Inicio': '{:%Y-%m}', 'Fecha Fin': '{:%Y-%m}',
                    'Magnitud': '{:.2f}', 'Intensidad': '{:.2f}', 'Pico': '{:.2f}'
                }))
        else:
            st.info("No hay datos de sequía para mostrar.")

    with tab_wet:
        if not wet_periods_df.empty:
            fig_wet = px.bar(
                wet_periods_df, x='Fecha Inicio', y='Duración (meses)', color='Intensidad',
                title='Duración e Intensidad de los Períodos Húmedos',
                hover_data=['Magnitud', 'Pico', 'Fecha Fin'],
                color_continuous_scale=px.colors.sequential.Blues
            )
            fig_wet.update_layout(coloraxis_colorbar=dict(title=f"Intensidad<br>({index_type})"))
            st.plotly_chart(fig_wet, use_container_width=True)
            with st.expander("Ver tabla de datos de períodos húmedos"):
                st.dataframe(wet_periods_df.style.format({
                    'Fecha Inicio': '{:%Y-%m}', 'Fecha Fin': '{:%Y-%m}',
                    'Magnitud': '{:.2f}', 'Intensidad': '{:.2f}', 'Pico': '{:.2f}'
                }))
        else:
            st.info("No hay datos de períodos húmedos para mostrar.")

def display_drought_analysis_tab(df_monthly_filtered, stations_for_analysis,
                                 df_anual_melted, gdf_filtered, analysis_mode, selected_regions, selected_municipios,
                                 selected_altitudes, **kwargs):
    st.header("Análisis de Extremos Hidrológicos")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )

    if not stations_for_analysis:
        st.warning("Seleccione al menos una estación.")
        return

    # 1. Mover los controles y la lógica de cálculo fuera y encima de las pestañas
    st.subheader("Configuración del Análisis por Percentiles")
    station_to_analyze_perc = st.selectbox(
        "Seleccione una estación para el análisis de percentiles:",
        options=sorted(stations_for_analysis),
        key="percentile_station_select"
    )
    col1, col2 = st.columns(2)
    p_lower = col1.slider("Percentil Inferior (Sequía):", 1, 40, 10, key="p_lower_perc")
    p_upper = col2.slider("Percentil Superior (Húmedo):", 60, 99, 90, key="p_upper_perc")

    df_extremes, df_thresholds = pd.DataFrame(), pd.DataFrame()
    if station_to_analyze_perc:
        df_long = st.session_state.get('df_long')
        if df_long is not None and not df_long.empty:
            try:
                with st.spinner(f"Calculando percentiles P{p_lower} y P{p_upper}..."):
                    df_extremes, df_thresholds = calculate_percentiles_and_extremes(
                        df_long, station_to_analyze_perc, p_lower, p_upper
                    )
            except Exception as e:
                st.error(f"Error al calcular el análisis de percentiles: {e}")

    # 2. Crear las nuevas pestañas
    (
        percentile_series_tab,
        percentile_thresholds_tab,
        indices_sub_tab,
        frequency_sub_tab,
    ) = st.tabs([
        "Serie de Tiempo por Percentiles",
        "Umbrales de Percentil Mensual",
        "Índices de Sequía (SPI/SPEI)",
        "Análisis de Frecuencia de Extremos"
    ])

    # 3. Lógica de la primera pestaña (Serie de Tiempo)
    with percentile_series_tab:
        if not df_extremes.empty:
            year_range_val = st.session_state.get('year_range', (2000, 2020))
            if isinstance(year_range_val, tuple) and len(year_range_val) == 2 and isinstance(year_range_val[0], int):
                year_min, year_max = year_range_val
            else:
                year_min, year_max = st.session_state.get('year_range_single', (2000, 2020))
            
            df_plot = df_extremes[
                (df_extremes[Config.DATE_COL].dt.year >= year_min) &
                (df_extremes[Config.DATE_COL].dt.year <= year_max) &
                (df_extremes[Config.DATE_COL].dt.month.isin(st.session_state.meses_numeros))
            ].copy()

            if not df_plot.empty:
                st.subheader(f"Serie de Tiempo con Eventos Extremos (P{p_lower} y P{p_upper} Percentiles)")
                color_map = {f'Sequía Extrema (<P{p_lower}%)': 'red', f'Húmedo Extremo (>P{p_upper}%)': 'blue', 'Normal': 'gray'}
                fig_series = px.scatter(
                    df_plot, x=Config.DATE_COL, y=Config.PRECIPITATION_COL,
                    color='event_type', color_discrete_map=color_map,
                    title=f"Precipitación Mensual y Eventos Extremos en {station_to_analyze_perc}",
                    labels={Config.PRECIPITATION_COL: "Precipitación (mm)", Config.DATE_COL: "Fecha"},
                    hover_data={'event_type': True, 'p_lower': ':.0f', 'p_upper': ':.0f'}
                )
                
                mean_precip = st.session_state.df_long[st.session_state.df_long[Config.STATION_NAME_COL] == station_to_analyze_perc][Config.PRECIPITATION_COL].mean()
                fig_series.add_hline(y=mean_precip, line_dash="dash", line_color="green", annotation_text="Media Histórica")
                fig_series.update_layout(height=500)
                st.plotly_chart(fig_series, use_container_width=True)
            else:
                st.warning("No hay datos que coincidan con los filtros de tiempo para la estación seleccionada.")
        else:
            st.info("Seleccione una estación para ver el análisis.")

    # 4. Lógica de la nueva pestaña (Umbrales)
    with percentile_thresholds_tab:
        if not df_thresholds.empty:
            st.subheader("Umbrales de Percentil Mensual (Climatología Histórica)")
            meses_map_inv = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
            df_thresholds['Month_Name'] = df_thresholds[Config.MONTH_COL].map(meses_map_inv)
            
            fig_thresh = go.Figure()
            fig_thresh.add_trace(go.Scatter(x=df_thresholds['Month_Name'], y=df_thresholds['p_upper'], mode='lines+markers', name=f'Percentil Superior (P{p_upper}%)', line=dict(color='blue')))
            fig_thresh.add_trace(go.Scatter(x=df_thresholds['Month_Name'], y=df_thresholds['p_lower'], mode='lines+markers', name=f'Percentil Inferior (P{p_lower}%)', line=dict(color='red')))
            fig_thresh.add_trace(go.Scatter(x=df_thresholds['Month_Name'], y=df_thresholds['mean_monthly'], mode='lines', name='Media Mensual', line=dict(color='green', dash='dot')))
            
            fig_thresh.update_layout(title='Umbrales de Precipitación por Mes (Basado en Climatología)', xaxis_title="Mes", yaxis_title="Precipitación (mm)", height=400)
            st.plotly_chart(fig_thresh, use_container_width=True)
        else:
            st.info("Seleccione una estación para ver los umbrales.")
            
    with indices_sub_tab:
        st.subheader("Análisis con Índices Estandarizados")
        col1_idx, col2_idx = st.columns([1, 3])
        index_values = pd.Series(dtype=float)
        
        with col1_idx:
            index_type = st.radio("Índice a Calcular:", ("SPI", "SPEI"), key="index_type_radio")
            station_to_analyze_idx = st.selectbox("Estación para análisis:", options=sorted(stations_for_analysis), key="index_station_select")
            index_window = st.select_slider("Escala de tiempo (meses):", options=[3, 6, 9, 12, 24], value=12, key="index_window_slider")
        
        if station_to_analyze_idx:
            df_station_idx = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station_to_analyze_idx].copy().set_index(Config.DATE_COL).sort_index()
            
            with col2_idx:
                with st.spinner(f"Calculando {index_type}-{index_window}..."):
                    if index_type == "SPI":
                        precip_series = df_station_idx[Config.PRECIPITATION_COL]
                        if len(precip_series.dropna()) < index_window * 2:
                            st.warning(f"No hay suficientes datos ({len(precip_series.dropna())} meses) para calcular el SPI-{index_window}.")
                        else:
                            index_values = calculate_spi(precip_series, index_window)
                    
                    elif index_type == "SPEI":
                        if Config.ET_COL not in df_station_idx.columns or df_station_idx[Config.ET_COL].isnull().all():
                            st.error(f"No hay datos de evapotranspiración ('{Config.ET_COL}') disponibles.")
                        else:
                            precip_series, et_series = df_station_idx[Config.PRECIPITATION_COL], df_station_idx[Config.ET_COL]
                            if len(precip_series.dropna()) < index_window * 2 or len(et_series.dropna()) < index_window * 2:
                                st.warning(f"No hay suficientes datos de precipitación o ETP para calcular el SPEI-{index_window}.")
                            else:
                                index_values = calculate_spei(precip_series, et_series, index_window)
            
                if not index_values.empty and not index_values.isnull().all():
                    with col2_idx:
                        df_plot = pd.DataFrame({'index_val': index_values}).dropna()
                        conditions = [df_plot['index_val']<=-2.0, (df_plot['index_val']>-2.0)&(df_plot['index_val']<=-1.5), (df_plot['index_val']>-1.5)&(df_plot['index_val']<=-1.0), (df_plot['index_val']>-1.0)&(df_plot['index_val']<1.0), (df_plot['index_val']>=1.0)&(df_plot['index_val']<1.5), (df_plot['index_val']>=1.5)&(df_plot['index_val']<2.0), df_plot['index_val']>=2.0]
                        colors = ['#b2182b', '#ef8a62', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
                        df_plot['color'] = np.select(conditions, colors, default='grey')
                        fig = go.Figure(go.Bar(x=df_plot.index, y=df_plot['index_val'], marker_color=df_plot['color'], name=index_type))
                        fig.update_layout(title=f"Índice {index_type}-{index_window} para {station_to_analyze_idx}", yaxis_title=f"Valor {index_type}", xaxis_title="Fecha", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    display_event_analysis(index_values, index_type)

    with frequency_sub_tab:
        st.subheader("Análisis de Frecuencia de Precipitaciones Anuales Máximas")
        st.markdown("Este análisis estima la probabilidad de ocurrencia de un evento de precipitación de cierta magnitud utilizando la distribución de Gumbel para calcular los **períodos de retorno**.")
        station_to_analyze = st.selectbox("Seleccione una estación para el análisis de frecuencia:", options=sorted(stations_for_analysis), key="freq_station_select")
        if station_to_analyze:
            station_data = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station_to_analyze].copy()
            annual_max_precip = station_data['precipitation'].dropna()
            if len(annual_max_precip) < 10:
                st.warning("Se recomiendan al menos 10 años de datos para un análisis de frecuencia confiable.")
            else:
                with st.spinner("Calculando períodos de retorno..."):
                    params = stats.gumbel_r.fit(annual_max_precip)
                    return_periods = np.array([2, 5, 10, 25, 50, 100, 200, 500])
                    non_exceed_prob = 1 - 1 / return_periods
                    precip_values = stats.gumbel_r.ppf(non_exceed_prob, *params)
                    results_df = pd.DataFrame({"Período de Retorno (años)": return_periods, "Precipitación Anual Esperada (mm)": precip_values})
                    
                    st.subheader(f"Resultados para la estación: {station_to_analyze}")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("#### Tabla de Resultados")
                        st.dataframe(results_df.style.format({"Precipitación Anual Esperada (mm)": "{:.1f}"}))
                    
                    with col2:
                        st.markdown("#### Curva de Frecuencia")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=station_data[Config.YEAR_COL], y=annual_max_precip, mode='markers', name='Máximos Anuales Observados'))
                        
                        x_plot = np.linspace(annual_max_precip.min(), precip_values[-1] * 1.1, 100)
                        y_plot_prob = stats.gumbel_r.cdf(x_plot, *params)
                        # Evitar división por cero
                        y_plot_prob = np.clip(y_plot_prob, 0, 0.999999)
                        y_plot_return_period = 1 / (1 - y_plot_prob)
                        
                        fig.add_trace(go.Scatter(x=y_plot_return_period, y=x_plot, mode='lines', name='Curva de Gumbel Ajustada', line=dict(color='red')))
                        fig.update_layout(title="Curva de Períodos de Retorno", xaxis_title="Período de Retorno (años)", yaxis_title="Precipitación Anual (mm)", xaxis_type="log")
                        st.plotly_chart(fig, use_container_width=True)
                        
def display_anomalies_tab(df_long, df_monthly_filtered, stations_for_analysis, analysis_mode, selected_regions, selected_municipios, selected_altitudes, **kwargs):
    st.header("Análisis de Anomalías de Precipitación")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )
    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    st.subheader("Configuración del Análisis")
    analysis_type = st.radio(
        "Calcular anomalía con respecto a:",
        ("El promedio de todo el período", "Una Normal Climatológica (período base fijo)"),
        key="anomaly_type"
    )
    
    df_anomalias = pd.DataFrame()
    avg_col_name = '' 

    if analysis_type == "Una Normal Climatológica (período base fijo)":
        years_in_long = sorted(df_long[Config.YEAR_COL].unique())
        default_start = 1991 if 1991 in years_in_long else years_in_long[0]
        default_end = 2020 if 2020 in years_in_long else years_in_long[-1]
        c1, c2 = st.columns(2)
        with c1:
            baseline_start = st.selectbox("Año de inicio del período base:", years_in_long, index=years_in_long.index(default_start))
        with c2:
            baseline_end = st.selectbox("Año de fin del período base:", years_in_long, index=years_in_long.index(default_end))
        
        if baseline_start >= baseline_end:
            st.error("El año de inicio del período base debe ser anterior al año de fin.")
            return
        
        with st.spinner(f"Calculando anomalías vs. normal climatológica ({baseline_start}-{baseline_end})..."):
            df_anomalias = calculate_climatological_anomalies(df_monthly_filtered, df_long, baseline_start, baseline_end)
            avg_col_name = 'precip_promedio_climatologico'
    else:
        with st.spinner("Calculando anomalías vs. promedio de todo el período..."):
            df_anomalias = calculate_monthly_anomalies(df_monthly_filtered, df_long)
            avg_col_name = 'precip_promedio_mes'

    if df_anomalias.empty or df_anomalias['anomalia'].isnull().all():
        st.warning("No hay suficientes datos históricos para calcular y mostrar las anomalías con los filtros actuales.")
        return

    anom_graf_tab, anom_fase_tab, anom_extremos_tab = st.tabs(["Gráfico de Anomalías", "Anomalías por Fase ENSO", "Tabla de Eventos Extremos"])
    
    with anom_graf_tab:
        df_plot = df_anomalias.groupby(Config.DATE_COL).agg(anomalia=('anomalia', 'mean')).reset_index()
        fig = create_anomaly_chart(df_plot)
        st.plotly_chart(fig, use_container_width=True)

    with anom_fase_tab:
        if Config.ENSO_ONI_COL in df_anomalias.columns:
            df_anomalias_enso = df_anomalias.dropna(subset=[Config.ENSO_ONI_COL]).copy()
            conditions = [df_anomalias_enso[Config.ENSO_ONI_COL] >= 0.5, df_anomalias_enso[Config.ENSO_ONI_COL] <= -0.5]
            phases = ['El Niño', 'La Niña']
            df_anomalias_enso['enso_fase'] = np.select(conditions, phases, default='Neutral')
            fig_box = px.box(df_anomalias_enso, x='enso_fase', y='anomalia', color='enso_fase', title="Distribución de Anomalías de Precipitación por Fase ENSO", labels={'anomalia': 'Anomalía de Precipitación (mm)', 'enso_fase': 'Fase ENSO'}, points='all')
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning(f"La columna '{Config.ENSO_ONI_COL}' no está disponible para este análisis.")

    with anom_extremos_tab:
        st.subheader("Eventos Mensuales Extremos (Basado en Anomalías)")
        df_extremos = df_anomalias.dropna(subset=['anomalia']).copy()
        df_extremos['fecha'] = df_extremos[Config.DATE_COL].dt.strftime('%Y-%m')
        
        if avg_col_name and avg_col_name in df_extremos.columns:
            cols_to_show = ['fecha', Config.STATION_NAME_COL, 'anomalia', Config.PRECIPITATION_COL, avg_col_name]
            col_rename_dict = {
                Config.STATION_NAME_COL: 'Estación',
                'anomalia': 'Anomalía (mm)',
                Config.PRECIPITATION_COL: 'Ppt. (mm)',
                avg_col_name: 'Ppt. Media (mm)'
            }
        else:
            cols_to_show = ['fecha', Config.STATION_NAME_COL, 'anomalia', Config.PRECIPITATION_COL]
            col_rename_dict = {
                Config.STATION_NAME_COL: 'Estación',
                'anomalia': 'Anomalía (mm)',
                Config.PRECIPITATION_COL: 'Ppt. (mm)'
            }

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 10 Meses más Secos")
            secos = df_extremos.nsmallest(10, 'anomalia')[cols_to_show]
            st.dataframe(secos.rename(columns=col_rename_dict).round(0), use_container_width=True)
        with col2:
            st.markdown("##### 10 Meses más Húmedos")
            humedos = df_extremos.nlargest(10, 'anomalia')[cols_to_show]
            st.dataframe(humedos.rename(columns=col_rename_dict).round(0), use_container_width=True)

def display_stats_tab(df_long, df_anual_melted, df_monthly_filtered, stations_for_analysis, gdf_filtered, analysis_mode, selected_regions, selected_municipios, selected_altitudes, **kwargs):
    st.header("Estadísticas de Precipitación")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )
    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    matriz_tab, resumen_mensual_tab, series_tab, sintesis_tab = st.tabs(["Matriz de Disponibilidad", "Resumen Mensual", "Datos Series Pptn", "Síntesis General"])
    
    with matriz_tab:
        st.subheader("Matriz de Disponibilidad de Datos Anual")
        heatmap_df = pd.DataFrame()
        title_text = ""
        color_scale = "Greens"
        
        if analysis_mode == "Completar series (interpolación)":
            view_mode = st.radio(
                "Seleccione la vista de la matriz:",
                ("Porcentaje de Datos Originales", "Porcentaje de Datos Completados", "Porcentaje de Datos Totales"),
                horizontal=True, key="matriz_view_mode"
            )
            if view_mode == "Porcentaje de Datos Completados":
                df_counts = df_monthly_filtered[df_monthly_filtered[Config.ORIGIN_COL] == 'Completado'].groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).size().reset_index(name='count')
                df_counts['porc_value'] = (df_counts['count'] / 12) * 100
                heatmap_df = df_counts.pivot(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values='porc_value').fillna(0)
                color_scale = "Reds"
                title_text = "Porcentaje de Datos Completados (Interpolados)"
            elif view_mode == "Porcentaje de Datos Totales":
                df_counts = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).size().reset_index(name='count')
                df_counts['porc_value'] = (df_counts['count'] / 12) * 100
                heatmap_df = df_counts.pivot(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values='porc_value').fillna(0)
                color_scale = "Blues"
                title_text = "Disponibilidad de Datos Totales (Original + Completado)"
            else: # Porcentaje de Datos Originales
                df_original_filtered = df_long[(df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) & (df_long[Config.DATE_COL].dt.year >= st.session_state.year_range[0]) & (df_long[Config.DATE_COL].dt.year <= st.session_state.year_range[1])]
                df_counts = df_original_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).size().reset_index(name='count')
                df_counts['porc_value'] = (df_counts['count'] / 12) * 100
                heatmap_df = df_counts.pivot(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values='porc_value').fillna(0)
                title_text = "Disponibilidad de Datos Originales"
        else: # Modo de datos originales
            df_counts = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).size().reset_index(name='count')
            df_counts['porc_value'] = (df_counts['count'] / 12) * 100
            heatmap_df = df_counts.pivot(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values='porc_value').fillna(0)
            title_text = "Disponibilidad de Datos Originales"

        if not heatmap_df.empty:
            st.markdown(f"**{title_text}**")
            styled_df = heatmap_df.style.background_gradient(cmap=color_scale, axis=None, vmin=0, vmax=100).format("{:.0f}%", na_rep="-")
            st.dataframe(styled_df)
        else:
            st.info("No hay datos para mostrar en la matriz con la selección actual.")

    with resumen_mensual_tab:
        st.subheader("Resumen de Estadísticas Mensuales por Estación")
        if not df_monthly_filtered.empty:
            summary_data = []
            for station_name, group in df_monthly_filtered.groupby(Config.STATION_NAME_COL):
                if not group[Config.PRECIPITATION_COL].dropna().empty:
                    max_row = group.loc[group[Config.PRECIPITATION_COL].idxmax()]
                    min_row = group.loc[group[Config.PRECIPITATION_COL].idxmin()]
                    summary_data.append({
                        "Estación": station_name,
                        "Ppt. Máxima Mensual (mm)": max_row[Config.PRECIPITATION_COL],
                        "Fecha Máxima": max_row[Config.DATE_COL].strftime('%Y-%m'),
                        "Ppt. Mínima Mensual (mm)": min_row[Config.PRECIPITATION_COL],
                        "Fecha Mínima": min_row[Config.DATE_COL].strftime('%Y-%m'),
                        "Promedio Mensual (mm)": group[Config.PRECIPITATION_COL].mean()
                    })
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df.round(1), use_container_width=True)
            else:
                st.info("No hay datos suficientes para calcular el resumen mensual.")
        else:
            st.info("No hay datos para mostrar el resumen mensual.")
    
    with series_tab:
        st.subheader("Series de Precipitación Anual por Estación (mm)")
        if not df_anual_melted.empty:
            ppt_series_df = df_anual_melted.pivot_table(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values=Config.PRECIPITATION_COL)
            st.dataframe(ppt_series_df.style.format("{:.0f}", na_rep="-").background_gradient(cmap='viridis', axis=1))
        else:
            st.info("No hay datos anuales para mostrar en la tabla.")

    with sintesis_tab:
        st.subheader("Síntesis General de Precipitación")
        if not df_monthly_filtered.empty and not df_anual_melted.empty:
            df_anual_valid = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
            df_monthly_valid = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL])
            if not df_anual_valid.empty and not df_monthly_valid.empty and not gdf_filtered.empty:
                # --- A. EXTREMOS DE PRECIPITACIÓN ---
                max_monthly_row = df_monthly_valid.loc[df_monthly_valid[Config.PRECIPITATION_COL].idxmax()]
                min_monthly_row = df_monthly_valid.loc[df_monthly_valid[Config.PRECIPITATION_COL].idxmin()]
                max_annual_row = df_anual_valid.loc[df_anual_valid[Config.PRECIPITATION_COL].idxmax()]
                min_annual_row = df_anual_valid.loc[df_anual_valid[Config.PRECIPITATION_COL].idxmin()]
                # --- B. PROMEDIOS REGIONALES/CLIMATOLÓGICOS ---
                df_yearly_avg = df_anual_valid.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                year_max_avg = df_yearly_avg.loc[df_yearly_avg[Config.PRECIPITATION_COL].idxmax()]
                year_min_avg = df_yearly_avg.loc[df_yearly_avg[Config.PRECIPITATION_COL].idxmin()]
                df_monthly_avg = df_monthly_valid.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                month_max_avg = df_monthly_avg.loc[df_monthly_avg[Config.PRECIPITATION_COL].idxmax()][Config.MONTH_COL]
                month_min_avg = df_monthly_avg.loc[df_monthly_avg[Config.PRECIPITATION_COL].idxmin()][Config.MONTH_COL]
                # --- C. EXTREMOS DE ALTITUD ---
                df_stations_valid = gdf_filtered.dropna(subset=[Config.ALTITUDE_COL])
                station_max_alt = None
                station_min_alt = None
                if not df_stations_valid.empty:
                    df_stations_valid[Config.ALTITUDE_COL] = pd.to_numeric(df_stations_valid[Config.ALTITUDE_COL], errors='coerce')
                    if not df_stations_valid[Config.ALTITUDE_COL].isnull().all():
                        station_max_alt = df_stations_valid.loc[df_stations_valid[Config.ALTITUDE_COL].idxmax()]
                        station_min_alt = df_stations_valid.loc[df_stations_valid[Config.ALTITUDE_COL].idxmin()]
                # --- D. CÁLCULO DE TENDENCIAS (SEN'S SLOPE) ---
                trend_results = []
                for station in stations_for_analysis:
                    station_data = df_anual_valid[df_anual_valid[Config.STATION_NAME_COL] == station].copy()
                    if len(station_data) >= 4:
                        mk_result_table = mk.original_test(station_data[Config.PRECIPITATION_COL])
                        trend_results.append({'slope_sen': mk_result_table.slope, 'p_value': mk_result_table.p, Config.STATION_NAME_COL: station})
                df_trends = pd.DataFrame(trend_results)
                max_pos_trend_row, min_neg_trend_row = None, None
                if not df_trends.empty:
                    df_pos_trends = df_trends[df_trends['slope_sen'] > 0]
                    df_neg_trends = df_trends[df_trends['slope_sen'] < 0]
                    if not df_pos_trends.empty: max_pos_trend_row = df_pos_trends.loc[df_pos_trends['slope_sen'].idxmax()]
                    if not df_neg_trends.empty: min_neg_trend_row = df_neg_trends.loc[df_neg_trends['slope_sen'].idxmin()]
                
                meses_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
                
                # --- DISPLAY DE RESULTADOS ---
                st.markdown("#### 1. Extremos de Precipitación")
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Máxima Ppt. Anual", f"{max_annual_row[Config.PRECIPITATION_COL]:.0f} mm", f"{max_annual_row[Config.STATION_NAME_COL]} ({int(max_annual_row[Config.YEAR_COL])})")
                with col2: st.metric("Mínima Ppt. Anual", f"{min_annual_row[Config.PRECIPITATION_COL]:.0f} mm", f"{min_annual_row[Config.STATION_NAME_COL]} ({int(min_annual_row[Config.YEAR_COL])})")
                with col3: st.metric("Máxima Ppt. Mensual", f"{max_monthly_row[Config.PRECIPITATION_COL]:.0f} mm", f"{max_monthly_row[Config.STATION_NAME_COL]} ({meses_map.get(max_monthly_row[Config.MONTH_COL])} {max_monthly_row[Config.DATE_COL].year})")
                with col4: st.metric("Mínima Ppt. Mensual", f"{min_monthly_row[Config.PRECIPITATION_COL]:.0f} mm", f"{min_monthly_row[Config.STATION_NAME_COL]} ({meses_map.get(min_monthly_row[Config.MONTH_COL])} {min_monthly_row[Config.DATE_COL].year})")
                
                st.markdown("#### 2. Promedios Históricos y Climatológicos")
                col5, col6, col7 = st.columns(3)
                with col5: st.metric("Año más Lluvioso (Promedio Regional)", f"{year_max_avg[Config.PRECIPITATION_COL]:.0f} mm", f"Año: {int(year_max_avg[Config.YEAR_COL])}")
                with col6: st.metric("Año menos Lluvioso (Promedio Regional)", f"{year_min_avg[Config.PRECIPITATION_COL]:.0f} mm", f"Año: {int(year_min_avg[Config.YEAR_COL])}")
                with col7: st.metric("Mes Climatológico más Lluvioso", f"{df_monthly_avg.loc[df_monthly_avg[Config.MONTH_COL] == month_max_avg, Config.PRECIPITATION_COL].iloc[0]:.0f} mm", f"{meses_map.get(month_max_avg)} (Min: {meses_map.get(month_min_avg)})")

                st.markdown("#### 3. Geografía y Tendencias")
                col8, col9, col10, col11 = st.columns(4)
                with col8:
                    if station_max_alt is not None: st.metric("Estación a Mayor Altitud", f"{float(station_max_alt[Config.ALTITUDE_COL]):.0f} m", f"{station_max_alt[Config.STATION_NAME_COL]}")
                    else: st.info("No hay datos de altitud.")
                with col9:
                    if station_min_alt is not None: st.metric("Estación a Menor Altitud", f"{float(station_min_alt[Config.ALTITUDE_COL]):.0f} m", f"{station_min_alt[Config.STATION_NAME_COL]}")
                    else: st.info("No hay datos de altitud.")
                with col10:
                    if max_pos_trend_row is not None: st.metric("Mayor Tendencia Positiva", f"+{max_pos_trend_row['slope_sen']:.2f} mm/año", f"{max_pos_trend_row[Config.STATION_NAME_COL]} (p={max_pos_trend_row['p_value']:.3f})")
                    else: st.info("No hay tendencias positivas.")
                with col11:
                    if min_neg_trend_row is not None: st.metric("Mayor Tendencia Negativa", f"{min_neg_trend_row['slope_sen']:.2f} mm/año", f"{min_neg_trend_row[Config.STATION_NAME_COL]} (p={min_neg_trend_row['p_value']:.3f})")
                    else: st.info("No hay tendencias negativas.")
            else:
                st.info("No hay datos anuales, mensuales o geográficos válidos para mostrar la síntesis.")
        else:
            st.info("No hay datos para mostrar la síntesis general.")
            
def display_correlation_tab(df_monthly_filtered, stations_for_analysis, analysis_mode,
                            selected_regions, selected_municipios, selected_altitudes, **kwargs):
    st.header("Análisis de Correlación")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )
    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
        
    st.markdown("Esta sección cuantifica la relación lineal entre la precipitación y diferentes variables utilizando el coeficiente de correlación de Pearson.")

    # AÑADIMOS UNA NUEVA PESTAÑA A LA LISTA
    tab_names = ["Correlación con ENSO (ONI)", "Matriz entre Estaciones", "Comparación 1 a 1", "Correlación con Otros Índices"]
    enso_corr_tab, matrix_corr_tab, station_corr_tab, indices_climaticos_tab = st.tabs(tab_names)

    with enso_corr_tab:
        if Config.ENSO_ONI_COL not in df_monthly_filtered.columns or df_monthly_filtered[Config.ENSO_ONI_COL].isnull().all():
            st.warning(f"No se puede realizar el análisis de correlación con ENSO. La columna '{Config.ENSO_ONI_COL}' no fue encontrada o no tiene datos en el período seleccionado.")
            return

        st.subheader("Configuración del Análisis de Correlación con ENSO")
        lag_months = st.slider(
            "Seleccionar desfase temporal (meses)",
            min_value=0, max_value=12, value=0,
            help="Analiza la correlación de la precipitación con el ENSO de 'x' meses atrás. Un desfase de 3 significa correlacionar la lluvia de hoy con el ENSO de hace 3 meses."
        )
        
        df_corr_analysis = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL, Config.ENSO_ONI_COL])
        if df_corr_analysis.empty:
            st.warning("No hay datos coincidentes entre la precipitación y el ENSO para la selección actual.")
            return

        analysis_level = st.radio("Nivel de Análisis de Correlación con ENSO", ["Promedio de la selección", "Por Estación Individual"], horizontal=True, key="enso_corr_level")
        
        df_plot_corr = pd.DataFrame()
        title_text = ""
        
        if analysis_level == "Por Estación Individual":
            station_to_corr = st.selectbox("Seleccione Estación:", options=sorted(df_corr_analysis[Config.STATION_NAME_COL].unique()), key="enso_corr_station")
            if station_to_corr:
                df_plot_corr = df_corr_analysis[df_corr_analysis[Config.STATION_NAME_COL] == station_to_corr].copy()
                title_text = f"Correlación para la estación: {station_to_corr}"
            else:
                return # Si no se selecciona estación
        else:
            df_plot_corr = df_corr_analysis.groupby(Config.DATE_COL).agg(
                precipitation=(Config.PRECIPITATION_COL, 'mean'),
                anomalia_oni=(Config.ENSO_ONI_COL, 'first')
            ).reset_index()
            title_text = "Correlación para el promedio de las estaciones seleccionadas"

        if not df_plot_corr.empty and len(df_plot_corr) > 2:
            if lag_months > 0:
                df_plot_corr['anomalia_oni_shifted'] = df_plot_corr['anomalia_oni'].shift(lag_months)
                df_plot_corr.dropna(subset=['anomalia_oni_shifted'], inplace=True)
                oni_column_to_use = 'anomalia_oni_shifted'
                lag_text = f" (con desfase de {lag_months} meses)"
            else:
                oni_column_to_use = 'anomalia_oni'
                lag_text = ""

            corr, p_value = stats.pearsonr(df_plot_corr[oni_column_to_use], df_plot_corr['precipitation'])
            
            st.subheader(title_text + lag_text)
            col1, col2 = st.columns(2)
            col1.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")
            col2.metric("Significancia (valor p)", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("La correlación es estadísticamente significativa.")
            else:
                st.warning("La correlación no es estadísticamente significativa.")

            fig_corr = px.scatter(
                df_plot_corr, x=oni_column_to_use, y='precipitation', trendline='ols',
                title=f"Dispersión: Precipitación vs. Anomalía ONI{lag_text}",
                labels={oni_column_to_use: f'Anomalía ONI (°C) [desfase {lag_months}m]', 'precipitation': 'Precipitación Mensual (mm)'}
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No hay suficientes datos superpuestos para calcular la correlación.")

    with matrix_corr_tab:
        st.subheader("Matriz de Correlación de Precipitación entre Estaciones")
        if len(stations_for_analysis) < 2:
            st.info("Seleccione al menos dos estaciones para generar la matriz de correlación.")
        else:
            with st.spinner("Calculando matriz de correlación..."):
                df_pivot = df_monthly_filtered.pivot_table(
                    index=Config.DATE_COL, 
                    columns=Config.STATION_NAME_COL, 
                    values=Config.PRECIPITATION_COL
                )
                corr_matrix = df_pivot.corr()
                
                fig_matrix = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Mapa de Calor de Correlaciones de Precipitación Mensual"
                )
                fig_matrix.update_layout(height=max(400, len(stations_for_analysis) * 25))
                st.plotly_chart(fig_matrix, use_container_width=True)

    with station_corr_tab:
        if len(stations_for_analysis) < 2:
            st.info("Seleccione al menos dos estaciones para comparar la correlación entre ellas.")
        else:
            st.subheader("Correlación de Precipitación entre dos Estaciones")
            station_options = sorted(stations_for_analysis)
            col1, col2 = st.columns(2)
            station1_name = col1.selectbox("Estación 1:", options=station_options, key="corr_station_1")
            station2_name = col2.selectbox("Estación 2:", options=station_options, index=1 if len(station_options) > 1 else 0, key="corr_station_2")
            
            if station1_name and station2_name and station1_name != station2_name:
                df_station1 = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station1_name][[Config.DATE_COL, Config.PRECIPITATION_COL]]
                df_station2 = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station2_name][[Config.DATE_COL, Config.PRECIPITATION_COL]]
                
                df_merged = pd.merge(df_station1, df_station2, on=Config.DATE_COL, suffixes=('_1', '_2')).dropna()
                df_merged.rename(columns={f'{Config.PRECIPITATION_COL}_1': station1_name, f'{Config.PRECIPITATION_COL}_2': station2_name}, inplace=True)
                
                if not df_merged.empty and len(df_merged) > 2:
                    corr, p_value = stats.pearsonr(df_merged[station1_name], df_merged[station2_name])
                    st.markdown(f"#### Resultados de la correlación ({station1_name} vs. {station2_name})")
                    st.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")
                    
                    if p_value < 0.05:
                        st.success(f"La correlación es estadísticamente significativa (p={p_value:.4f}).")
                    else:
                        st.warning(f"La correlación no es estadísticamente significativa (p={p_value:.4f}).")
                    
                    slope, intercept, _, _, _ = stats.linregress(df_merged[station1_name], df_merged[station2_name])
                    st.info(f"Ecuación de regresión: y = {slope:.2f}x + {intercept:.2f}")

                    fig_scatter = px.scatter(
                        df_merged, x=station1_name, y=station2_name, trendline='ols',
                        title=f'Dispersión de Precipitación: {station1_name} vs. {station2_name}',
                        labels={station1_name: f'Precipitación en {station1_name} (mm)', station2_name: f'Precipitación en {station2_name} (mm)'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("No hay suficientes datos superpuestos para calcular la correlación para las estaciones seleccionadas.")

    with indices_climaticos_tab:
        st.subheader("Análisis de Correlación con Índices Climáticos (SOI, IOD)")
        available_indices = []
        if Config.SOI_COL in df_monthly_filtered.columns and not df_monthly_filtered[Config.SOI_COL].isnull().all():
            available_indices.append("SOI")
        if Config.IOD_COL in df_monthly_filtered.columns and not df_monthly_filtered[Config.IOD_COL].isnull().all():
            available_indices.append("IOD")

        if not available_indices:
            st.warning("No se encontraron columnas para los índices climáticos (SOI o IOD) en el archivo principal o no hay datos en el período seleccionado.")
        else:
            col1_corr, col2_corr = st.columns(2)
            selected_index = col1_corr.selectbox("Seleccione un índice climático:", available_indices)
            selected_station_corr = col2_corr.selectbox("Seleccione una estación:", options=sorted(stations_for_analysis), key="station_for_index_corr")

            if selected_index and selected_station_corr:
                index_col_map = {"SOI": Config.SOI_COL, "IOD": Config.IOD_COL}
                index_col_name = index_col_map.get(selected_index)
                
                df_merged_indices = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == selected_station_corr].copy()
                
                if index_col_name in df_merged_indices.columns:
                    df_merged_indices.dropna(subset=[Config.PRECIPITATION_COL, index_col_name], inplace=True)
                else:
                    st.error(f"La columna para el índice '{selected_index}' no se encontró en los datos de la estación.")
                    return

                if not df_merged_indices.empty and len(df_merged_indices) > 2:
                    corr, p_value = stats.pearsonr(df_merged_indices[index_col_name], df_merged_indices[Config.PRECIPITATION_COL])
                    
                    st.markdown(f"#### Resultados de la correlación ({selected_index} vs. Precipitación de {selected_station_corr})")
                    st.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")
                    if p_value < 0.05:
                        st.success("La correlación es estadísticamente significativa.")
                    else:
                        st.warning("La correlación no es estadísticamente significativa.")

                    fig_scatter_indices = px.scatter(
                        df_merged_indices, x=index_col_name, y=Config.PRECIPITATION_COL, trendline='ols',
                        title=f'Dispersión: {selected_index} vs. Precipitación de {selected_station_corr}',
                        labels={index_col_name: f'Valor del índice {selected_index}', Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)'}
                    )
                    st.plotly_chart(fig_scatter_indices, use_container_width=True)
                else:
                    st.warning("No hay suficientes datos superpuestos entre la estación y el índice para calcular la correlación.")

def display_enso_tab(df_enso, df_monthly_filtered, gdf_filtered, stations_for_analysis,
                     analysis_mode, selected_regions, selected_municipios, selected_altitudes, **kwargs):
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )
    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    if df_enso is None or df_enso.empty:
        st.warning("No se encontraron datos del fenómeno ENSO en el archivo de precipitación cargado.")
        return

    enso_series_tab, enso_anim_tab = st.tabs(["Series de Tiempo ENSO", "Mapa Interactivo ENSO"])

    with enso_series_tab:
        enso_vars_available = {
            Config.ENSO_ONI_COL: 'Anomalía ONI',
            'temp_sst': 'Temp. Superficial del Mar (SST)',
            'temp_media': 'Temp. Media'
        }
        available_tabs = [name for var, name in enso_vars_available.items() if var in df_enso.columns]
        if not available_tabs:
            st.warning("No hay variables ENSO disponibles en el archivo de datos para visualizar.")
        else:
            enso_variable_tabs = st.tabs(available_tabs)
            for i, var_name in enumerate(available_tabs):
                with enso_variable_tabs[i]:
                    var_code = [code for code, name in enso_vars_available.items() if name == var_name][0]
                    enso_filtered = df_enso
                    if not enso_filtered.empty and var_code in enso_filtered.columns and not enso_filtered[var_code].isnull().all():
                        fig_enso_series = px.line(enso_filtered, x=Config.DATE_COL, y=var_code, title=f"Serie de Tiempo para {var_name}")
                        st.plotly_chart(fig_enso_series, use_container_width=True)
                    else:
                        st.warning(f"No hay datos disponibles para '{var_code}' en el período seleccionado.")

    with enso_anim_tab:
        st.subheader("Explorador Mensual del Fenómeno ENSO")
        if gdf_filtered.empty or Config.ENSO_ONI_COL not in df_enso.columns:
            st.warning("Datos insuficientes para generar esta visualización. Se requiere información de estaciones y la columna 'anomalia_oni'.")
            return
        
        controls_col, map_col = st.columns([1, 3])
        enso_anim_data = df_enso[[Config.DATE_COL, Config.ENSO_ONI_COL]].copy().dropna(subset=[Config.ENSO_ONI_COL])
        conditions = [
            enso_anim_data[Config.ENSO_ONI_COL] >= 0.5,
            enso_anim_data[Config.ENSO_ONI_COL] <= -0.5
        ]
        phases = ['El Niño', 'La Niña']
        enso_anim_data['fase'] = np.select(conditions, phases, default='Neutral')
        
        year_range_val = st.session_state.get('year_range', (2000, 2020))
        if isinstance(year_range_val, tuple) and len(year_range_val) == 2 and isinstance(year_range_val[0], int):
            year_min, year_max = year_range_val
        else:
            year_min, year_max = st.session_state.get('year_range_single', (2000, 2020))
            
        enso_anim_data_filtered = enso_anim_data[
            (enso_anim_data[Config.DATE_COL].dt.year >= year_min) &
            (enso_anim_data[Config.DATE_COL].dt.year <= year_max)
        ]
        selected_date = None
        
        with controls_col:
            st.markdown("##### Controles de Mapa")
            selected_base_map_config, selected_overlays_config = display_map_controls(st, "enso_anim")
            st.markdown("##### Selección de Fecha")
            available_dates = sorted(enso_anim_data_filtered[Config.DATE_COL].unique())
            if available_dates:
                selected_date = st.select_slider("Seleccione una fecha (Año-Mes)",
                                                 options=available_dates,
                                                 format_func=lambda date: pd.to_datetime(date).strftime('%Y-%m'))
                phase_info = enso_anim_data_filtered[enso_anim_data_filtered[Config.DATE_COL] == selected_date]
                if not phase_info.empty:
                    current_phase = phase_info['fase'].iloc[0]
                    current_oni = phase_info[Config.ENSO_ONI_COL].iloc[0]
                    st.metric(f"Fase ENSO en {pd.to_datetime(selected_date).strftime('%Y-%m')}", current_phase, f"Anomalía ONI: {current_oni:.2f}°C")
                else:
                    st.warning("No hay datos de ENSO para el período seleccionado.")
            else:
                st.warning("No hay fechas con datos ENSO en el rango seleccionado.")
        
        with map_col:
            if selected_date:
                m_enso = create_folium_map([4.57, -74.29], 5, selected_base_map_config, selected_overlays_config)
                phase_color_map = {'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'gray'}
                phase_info = enso_anim_data_filtered[enso_anim_data_filtered[Config.DATE_COL] == selected_date]
                current_phase_str = phase_info['fase'].iloc[0] if not phase_info.empty else "N/A"
                marker_color = phase_color_map.get(current_phase_str, 'black')
                for _, station in gdf_filtered.iterrows():
                    folium.Marker(
                        location=[station['geometry'].y, station['geometry'].x],
                        tooltip=f"{station[Config.STATION_NAME_COL]}<br>Fase: {current_phase_str}",
                        icon=folium.Icon(color=marker_color, icon='cloud')
                    ).add_to(m_enso)
                if not gdf_filtered.empty:
                    bounds = gdf_filtered.total_bounds
                    if np.all(np.isfinite(bounds)):
                        m_enso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                st_folium(m_enso, height=700, use_container_width=True)
            else:
                st.info("Seleccione una fecha para visualizar el mapa.")

def display_trends_and_forecast_tab(df_full_monthly, stations_for_analysis, df_anual_melted, df_monthly_filtered, analysis_mode, selected_regions, selected_municipios, selected_altitudes, **kwargs):
    st.header("Análisis de Tendencias y Pronósticos")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros),
        analysis_mode=analysis_mode,
        selected_regions=selected_regions,
        selected_municipios=selected_municipios,
        selected_altitudes=selected_altitudes
    )
    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
        
    tab_names = ["Análisis Lineal", "Tendencia Mann-Kendall", "Tabla Comparativa", "Descomposición de Series", "Autocorrelación (ACF/PACF)", "Pronóstico SARIMA", "Pronóstico Prophet", "SARIMA vs Prophet"]
    tendencia_individual_tab, mann_kendall_tab, tendencia_tabla_tab, descomposicion_tab, autocorrelacion_tab, pronostico_sarima_tab, pronostico_prophet_tab, compare_forecast_tab = st.tabs(tab_names)

    with tendencia_individual_tab:
        st.subheader("Tendencia de Precipitación Anual (Regresión Lineal)")
        analysis_type = st.radio("Tipo de Análisis de Tendencia:", ["Promedio de la selección", "Estación individual"], horizontal=True, key="linear_trend_type")
        df_to_analyze = None
        
        if analysis_type == "Promedio de la selección":
            df_to_analyze = df_anual_melted.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        else:
            station_to_analyze = st.selectbox("Seleccione una estación para analizar:", options=stations_for_analysis, key="tendencia_station_select")
            if station_to_analyze:
                df_to_analyze = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station_to_analyze]
        
        if df_to_analyze is not None and len(df_to_analyze.dropna(subset=[Config.PRECIPITATION_COL])) > 2:
            df_to_analyze['año_num'] = pd.to_numeric(df_to_analyze[Config.YEAR_COL])
            df_clean = df_to_analyze.dropna(subset=[Config.PRECIPITATION_COL])
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['año_num'], df_clean[Config.PRECIPITATION_COL])
            tendencia_texto = "aumentando" if slope > 0 else "disminuyendo"
            significancia_texto = "**estadísticamente significativa**" if p_value < 0.05 else "no es **estadísticamente significativa**"
            st.markdown(f"La tendencia de la precipitación es de **{slope:.2f} mm/año** (es decir, está {tendencia_texto}). Con un valor p de **{p_value:.3f}**, esta tendencia {significancia_texto}.")
            df_to_analyze['tendencia'] = slope * df_to_analyze['año_num'] + intercept
            
            fig_tendencia = px.scatter(df_to_analyze, x='año_num', y=Config.PRECIPITATION_COL, title='Tendencia de la Precipitación Anual')
            fig_tendencia.add_trace(go.Scatter(x=df_to_analyze['año_num'], y=df_to_analyze['tendencia'], mode='lines', name='Línea de Tendencia', line=dict(color='red')))
            fig_tendencia.update_layout(xaxis_title="Año", yaxis_title="Precipitación Anual (mm)")
            st.plotly_chart(fig_tendencia, use_container_width=True)
        else:
            st.warning("No hay suficientes datos en el período seleccionado para calcular una tendencia.")

    with mann_kendall_tab:
        st.subheader("Tendencia de Precipitación Anual (Prueba de Mann-Kendall y Pendiente de Sen)")
        with st.expander("¿Qué es la prueba de Mann-Kendall?"):
            st.markdown("""
            - **Prueba de Mann-Kendall**: Detecta si existe una tendencia (creciente o decreciente) en el tiempo.
            - **Valor p**: Si es < 0.05, la tendencia es estadísticamente significativa.
            - **Pendiente de Sen**: Cuantifica la magnitud de la tendencia (ej. "aumento de 5 mm/año"). Es un método robusto que no se ve muy afectado por valores atípicos.
            """)
        
        mk_analysis_type = st.radio("Tipo de Análisis de Tendencia:", ["Promedio de la selección", "Estación individual"], horizontal=True, key="mk_trend_type")
        df_to_analyze_mk = None
        if mk_analysis_type == "Promedio de la selección":
            df_to_analyze_mk = df_anual_melted.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        else:
            station_to_analyze_mk = st.selectbox("Seleccione una estación para analizar:", options=stations_for_analysis, key="mk_station_select")
            if station_to_analyze_mk:
                df_to_analyze_mk = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station_to_analyze_mk]

        if df_to_analyze_mk is not None and len(df_to_analyze_mk.dropna(subset=[Config.PRECIPITATION_COL])) > 3:
            df_clean_mk = df_to_analyze_mk.dropna(subset=[Config.PRECIPITATION_COL]).sort_values(by=Config.YEAR_COL)
            mk_result = mk.original_test(df_clean_mk[Config.PRECIPITATION_COL])
            
            title = 'Promedio de la selección' if mk_analysis_type == 'Promedio de la selección' else station_to_analyze_mk
            st.markdown(f"#### Resultados para: {title}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Tendencia Detectada", mk_result.trend.capitalize())
            col2.metric("Valor p", f"{mk_result.p:.4f}")
            col3.metric("Pendiente de Sen (mm/año)", f"{mk_result.slope:.2f}")

            df_clean_mk['año_num'] = pd.to_numeric(df_clean_mk[Config.YEAR_COL])
            median_x = df_clean_mk['año_num'].median()
            median_y = df_clean_mk[Config.PRECIPITATION_COL].median()
            intercept = median_y - (mk_result.slope * median_x)
            df_clean_mk['tendencia_sen'] = (mk_result.slope * df_clean_mk['año_num']) + intercept

            fig_mk = go.Figure()
            fig_mk.add_trace(go.Scatter(x=df_clean_mk['año_num'], y=df_clean_mk[Config.PRECIPITATION_COL], mode='markers', name='Datos Anuales'))
            fig_mk.add_trace(go.Scatter(x=df_clean_mk['año_num'], y=df_clean_mk['tendencia_sen'], mode='lines', name="Tendencia (Sen's Slope)", line=dict(color='orange')))
            fig_mk.update_layout(title=f"Tendencia de Mann-Kendall para {title}", xaxis_title="Año", yaxis_title="Precipitación Anual (mm)")
            st.plotly_chart(fig_mk, use_container_width=True)
        else:
            st.warning("No hay suficientes datos (se requieren al menos 4 puntos) para calcular la tendencia de Mann-Kendall.")

    with tendencia_tabla_tab:
        st.subheader("Tabla Comparativa de Tendencias de Precipitación Anual")
        st.info("Presione el botón para calcular los valores para todas las estaciones seleccionadas.")
        if st.button("Calcular Tendencias para Todas las Estaciones Seleccionadas"):
            with st.spinner("Calculando tendencias..."):
                results = []
                df_anual_calc = df_anual_melted.copy()
                for station in stations_for_analysis:
                    station_data = df_anual_calc[df_anual_calc[Config.STATION_NAME_COL] == station].dropna(subset=[Config.PRECIPITATION_COL]).sort_values(by=Config.YEAR_COL)
                    slope_lin, p_lin = np.nan, np.nan
                    trend_mk, p_mk, slope_sen = "Datos insuficientes", np.nan, np.nan
                    if len(station_data) > 2:
                        station_data['año_num'] = pd.to_numeric(station_data[Config.YEAR_COL])
                        res = stats.linregress(station_data['año_num'], station_data[Config.PRECIPITATION_COL])
                        slope_lin, p_lin = res.slope, res.pvalue
                    if len(station_data) > 3:
                        mk_result_table = mk.original_test(station_data[Config.PRECIPITATION_COL])
                        trend_mk = mk_result_table.trend.capitalize()
                        p_mk = mk_result_table.p
                        slope_sen = mk_result_table.slope
                    results.append({"Estación": station, "Años Analizados": len(station_data), "Tendencia Lineal (mm/año)": slope_lin, "Valor p (Lineal)": p_lin, "Tendencia MK": trend_mk, "Valor p (MK)": p_mk, "Pendiente de Sen (mm/año)": slope_sen})
                if results:
                    results_df = pd.DataFrame(results)
                    def style_p_value(val):
                        if pd.isna(val) or isinstance(val, str): return ""
                        color = 'lightgreen' if val < 0.05 else 'lightcoral'
                        return f'background-color: {color}'
                    st.dataframe(results_df.style.format({"Tendencia Lineal (mm/año)": "{:.2f}", "Valor p (Lineal)": "{:.4f}", "Valor p (MK)": "{:.4f}", "Pendiente de Sen (mm/año)": "{:.2f}"}).applymap(style_p_value, subset=['Valor p (Lineal)', 'Valor p (MK)']), use_container_width=True)

    with descomposicion_tab:
        st.subheader("Descomposición de Series de Tiempo Mensual")
        station_to_decompose = st.selectbox("Seleccione una estación para la descomposición:", options=stations_for_analysis, key="decompose_station_select")
        if station_to_decompose:
            df_station = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station_to_decompose].copy()
            if not df_station.empty:
                df_station.set_index(Config.DATE_COL, inplace=True)
                try:
                    series_for_decomp = df_station[Config.PRECIPITATION_COL].asfreq('MS').interpolate(method='time')
                    result = get_decomposition_results(series_for_decomp)
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Observado", "Tendencia", "Estacionalidad", "Residuo"))
                    fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed, mode='lines', name='Observado'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Tendencia'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Estacionalidad'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='markers', name='Residuo'), row=4, col=1)
                    fig.update_layout(height=700, title_text=f"Descomposición de la Serie para {station_to_decompose}", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"No se pudo realizar la descomposición. Error: {e}")

    with autocorrelacion_tab:
        st.subheader("Análisis de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)")
        station_to_analyze_acf = st.selectbox("Seleccione una estación:", options=stations_for_analysis, key="acf_station_select")
        max_lag = st.slider("Número máximo de rezagos (meses):", min_value=12, max_value=60, value=24, step=12)
        if station_to_analyze_acf:
            df_station_acf = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station_to_analyze_acf].copy()
            if not df_station_acf.empty:
                df_station_acf.set_index(Config.DATE_COL, inplace=True)
                df_station_acf = df_station_acf.asfreq('MS')
                df_station_acf[Config.PRECIPITATION_COL] = df_station_acf[Config.PRECIPITATION_COL].interpolate(method='time').dropna()
                if len(df_station_acf) > max_lag:
                    try:
                        fig_acf = create_acf_chart(df_station_acf[Config.PRECIPITATION_COL], max_lag)
                        st.plotly_chart(fig_acf, use_container_width=True)
                        fig_pacf = create_pacf_chart(df_station_acf[Config.PRECIPITATION_COL], max_lag)
                        st.plotly_chart(fig_pacf, use_container_width=True)
                    except Exception as e:
                        st.error(f"No se pudieron generar los gráficos de autocorrelación. Error: {e}")
                else:
                    st.warning(f"No hay suficientes datos para el análisis de autocorrelación.")

    with pronostico_sarima_tab:
        st.subheader("Pronóstico (Modelo SARIMA)")
        st.info("Los pronósticos se generan utilizando los datos procesados según la opción seleccionada en 'Modo de análisis' en el panel de control. Si el modo 'Completar series' está activo, se usarán los datos interpolados.", icon="ℹ️")
        station_to_forecast = st.selectbox("Seleccione una estación:", options=stations_for_analysis, key="sarima_station_select")
        c1, c2 = st.columns(2)
        with c1:
            forecast_horizon = st.slider("Meses a pronosticar:", 12, 36, 12, step=12, key="sarima_horizon")
        with c2:
            test_size = st.slider("Meses para evaluación:", 12, 36, 12, step=6, key="sarima_test_size")
        use_auto_arima = st.checkbox("Encontrar parámetros óptimos automáticamente (Auto-ARIMA)", value=True)
        if station_to_forecast and st.button("Generar Pronóstico SARIMA"):
            ts_data_sarima = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station_to_forecast].copy()
            if len(ts_data_sarima.dropna(subset=[Config.PRECIPITATION_COL])) < test_size + 36:
                st.warning("No hay suficientes datos para un pronóstico confiable (se necesitan al menos 3 años más que el período de evaluación).")
            else:
                try:
                    if use_auto_arima:
                        with st.spinner("Buscando el mejor modelo Auto-ARIMA (esto puede tardar)..."):
                            order, seasonal_order = auto_arima_search(ts_data_sarima, test_size)
                        st.success(f"Modelo óptimo encontrado: orden={order}, orden estacional={seasonal_order}")
                    else:
                        order, seasonal_order = (1, 1, 1), (1, 1, 1, 12)
                    with st.spinner("Entrenando y evaluando modelo SARIMA..."):
                        ts_hist, forecast_mean, forecast_ci, metrics, sarima_df_export = generate_sarima_forecast(ts_data_sarima, order, seasonal_order, forecast_horizon, test_size)
                    st.session_state['sarima_results'] = {'forecast': sarima_df_export, 'metrics': metrics, 'history': ts_hist}
                    st.markdown("##### Resultados del Pronóstico")
                    fig_pronostico = go.Figure()
                    fig_pronostico.add_trace(go.Scatter(x=ts_hist.index, y=ts_hist, mode='lines', name='Datos Históricos'))
                    fig_pronostico.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Pronóstico SARIMA', line=dict(color='red', dash='dash')))
                    fig_pronostico.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0], mode='lines', line=dict(width=0), showlegend=False))
                    fig_pronostico.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.2)', name='Intervalo de Confianza'))
                    st.plotly_chart(fig_pronostico, use_container_width=True)
                    st.markdown("##### Evaluación del Modelo")
                    st.info(f"El modelo se evaluó usando los últimos **{test_size} meses** de datos históricos como conjunto de prueba.")
                    m1, m2 = st.columns(2)
                    m1.metric("RMSE (Error Cuadrático Medio)", f"{metrics['RMSE']:.2f}")
                    m2.metric("MAE (Error Absoluto Medio)", f"{metrics['MAE']:.2f}")
                except Exception as e:
                    st.error(f"No se pudo generar el pronóstico SARIMA. Error: {e}")

    with pronostico_prophet_tab:
        st.subheader("Pronóstico (Modelo Prophet)")
        station_to_forecast_prophet = st.selectbox("Seleccione una estación:", options=stations_for_analysis, key="prophet_station_select")
        c1, c2 = st.columns(2)
        with c1:
            forecast_horizon_prophet = st.slider("Meses a pronosticar:", 12, 36, 12, step=12, key="prophet_horizon")
        with c2:
            test_size_prophet = st.slider("Meses para evaluación:", 12, 36, 12, step=6, key="prophet_test_size")
        if station_to_forecast_prophet and st.button("Generar Pronóstico Prophet"):
            with st.spinner(f"Preparando y completando datos para {station_to_forecast_prophet}..."):
                original_station_data = df_full_monthly[df_full_monthly[Config.STATION_NAME_COL] == station_to_forecast_prophet].copy()
                ts_data_prophet = complete_series(original_station_data)
            if len(ts_data_prophet.dropna(subset=[Config.PRECIPITATION_COL])) < test_size_prophet + 24:
                st.warning(f"Incluso después de completar, no hay suficientes datos para un pronóstico confiable.")
            else:
                try:
                    with st.spinner("Entrenando y evaluando modelo Prophet..."):
                        model, forecast, metrics = generate_prophet_forecast(ts_data_prophet, forecast_horizon_prophet, test_size_prophet, regressors=None)
                    st.session_state['prophet_results'] = {'forecast': forecast[['ds', 'yhat']], 'metrics': metrics}
                    st.markdown("##### Resultados del Pronóstico")
                    fig_prophet = plot_plotly(model, forecast)
                    st.plotly_chart(fig_prophet, use_container_width=True)
                    st.markdown("##### Evaluación del Modelo")
                    st.info(f"El modelo se evaluó usando los últimos **{test_size_prophet} meses** de datos históricos como conjunto de prueba.")
                    m1, m2 = st.columns(2)
                    m1.metric("RMSE", f"{metrics['RMSE']:.2f}")
                    m2.metric("MAE", f"{metrics['MAE']:.2f}")
                except Exception as e:
                    st.error(f"No se pudo generar el pronóstico con Prophet. Error: {e}")

    with compare_forecast_tab:
        st.subheader("Comparación de Pronósticos: SARIMA vs Prophet")
        sarima_results = st.session_state.get('sarima_results')
        prophet_results = st.session_state.get('prophet_results')
        if not sarima_results or not prophet_results:
            st.warning("Debe generar un pronóstico SARIMA y Prophet en sus respectivas pestañas para poder compararlos.")
        else:
            fig_compare = go.Figure()
            if sarima_results.get('history') is not None:
                hist_data = sarima_results['history']
                fig_compare.add_trace(go.Scatter(x=hist_data.index, y=hist_data, mode='lines', name='Histórico', line=dict(color='gray')))
            if sarima_results.get('forecast') is not None:
                sarima_fc = sarima_results['forecast']
                fig_compare.add_trace(go.Scatter(x=sarima_fc['ds'], y=sarima_fc['yhat'], mode='lines', name='Pronóstico SARIMA', line=dict(color='red', dash='dash')))
            if prophet_results.get('forecast') is not None:
                prophet_fc = prophet_results['forecast']
                fig_compare.add_trace(go.Scatter(x=prophet_fc['ds'], y=prophet_fc['yhat'], mode='lines', name='Pronóstico Prophet', line=dict(color='blue', dash='dash')))
            fig_compare.update_layout(title="Pronóstico Comparativo", xaxis_title="Fecha", yaxis_title="Precipitación (mm)", height=500, legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig_compare, use_container_width=True)
            st.markdown("#### Comparación de Precisión (sobre el conjunto de prueba)")
            sarima_metrics = sarima_results.get('metrics')
            prophet_metrics = prophet_results.get('metrics')
            if sarima_metrics and prophet_metrics:
                m_data = {'Métrica': ['RMSE', 'MAE'], 'SARIMA': [sarima_metrics['RMSE'], sarima_metrics['MAE']], 'Prophet': [prophet_metrics['RMSE'], prophet_metrics['MAE']]}
                metrics_df = pd.DataFrame(m_data)
                st.dataframe(metrics_df.style.format({'SARIMA': '{:.2f}', 'Prophet': '{:.2f}'}))
                rmse_winner = 'SARIMA' if sarima_metrics['RMSE'] < prophet_metrics['RMSE'] else 'Prophet'
                mae_winner = 'SARIMA' if sarima_metrics['MAE'] < prophet_metrics['MAE'] else 'Prophet'
                st.success(f"**Ganador (menor error):** **{rmse_winner}** basado en RMSE y **{mae_winner}** basado en MAE.")
            else:
                st.info("Genere ambos pronósticos (SARIMA y Prophet) para ver la comparación de precisión.")

def display_downloads_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis, analysis_mode):
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    st.header("Opciones de Descarga")
    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para activar las descargas.")
        return
    st.markdown("Aquí puedes descargar los datos actualmente visualizados, según los filtros aplicados en el panel de control.")
    st.markdown("---")
    st.markdown("#### Datos de Precipitación Anual (Filtrados)")
    if not df_anual_melted.empty:
        csv_anual = convert_df_to_csv(df_anual_melted)
        st.download_button(label="📥 Descargar CSV Anual", data=csv_anual, file_name='precipitacion_anual_filtrada.csv', mime='text/csv', key='download-anual')
    else:
        st.info("No hay datos anuales para descargar con los filtros actuales.")
    st.markdown("---")
    if analysis_mode == "Completar series (interpolación)":
        st.markdown("#### Datos de Series Mensuales Completas (Interpoladas)")
        st.info("Los datos a continuación han sido completados (interpolados) para rellenar los vacíos en las series de tiempo.")
        csv_completed = convert_df_to_csv(df_monthly_filtered)
        st.download_button(label="📥 Descargar CSV de Series Completas", data=csv_completed, file_name='precipitacion_mensual_completa.csv', mime='text/csv', key='download-completed')
    else:
        st.markdown("#### Datos de Precipitación Mensual (Originales Filtrados)")
        if not df_monthly_filtered.empty:
            csv_mensual = convert_df_to_csv(df_monthly_filtered)
            st.download_button(label="📥 Descargar CSV Mensual", data=csv_mensual, file_name='precipitacion_mensual_filtrada.csv', mime='text/csv', key='download-mensual')
        else:
            st.info("No hay datos mensuales para descargar con los filtros actuales.")

@st.cache_data
def calculate_comprehensive_stats(_df_anual, _df_monthly, _stations):
    """Calcula un conjunto completo de estadísticas para cada estación seleccionada."""
    results = []
    for station in _stations:
        stats = {"Estación": station}
        station_anual = _df_anual[_df_anual[Config.STATION_NAME_COL] == station].dropna(subset=[Config.PRECIPITATION_COL])
        station_monthly = _df_monthly[_df_monthly[Config.STATION_NAME_COL] == station].dropna(subset=[Config.PRECIPITATION_COL])
        if not station_anual.empty:
            stats['Años con Datos'] = int(station_anual[Config.PRECIPITATION_COL].count())
            stats['Ppt. Media Anual (mm)'] = station_anual[Config.PRECIPITATION_COL].mean()
            stats['Desv. Estándar Anual (mm)'] = station_anual[Config.PRECIPITATION_COL].std()
            max_anual_row = station_anual.loc[station_anual[Config.PRECIPITATION_COL].idxmax()]
            stats['Ppt. Máxima Anual (mm)'] = max_anual_row[Config.PRECIPITATION_COL]
            stats['Año Ppt. Máxima'] = int(max_anual_row[Config.YEAR_COL])
            min_anual_row = station_anual.loc[station_anual[Config.PRECIPITATION_COL].idxmin()]
            stats['Ppt. Mínima Anual (mm)'] = min_anual_row[Config.PRECIPITATION_COL]
            stats['Año Ppt. Mínima'] = int(min_anual_row[Config.YEAR_COL])
            if len(station_anual) >= 4:
                mk_result = mk.original_test(station_anual[Config.PRECIPITATION_COL])
                stats['Tendencia (mm/año)'] = mk_result.slope
                stats['Significancia (p-valor)'] = mk_result.p
            else:
                stats['Tendencia (mm/año)'] = np.nan
                stats['Significancia (p-valor)'] = np.nan
        if not station_monthly.empty:
            meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            monthly_means = station_monthly.groupby(station_monthly[Config.DATE_COL].dt.month)[Config.PRECIPITATION_COL].mean()
            for i, mes in enumerate(meses, 1):
                stats[f'Ppt Media {mes} (mm)'] = monthly_means.get(i, 0)
        results.append(stats)
    return pd.DataFrame(results)

def display_station_table_tab(gdf_filtered, df_anual_melted, df_monthly_filtered, stations_for_analysis, **kwargs):
    st.header("Información Detallada de las Estaciones")
    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    st.info("Presiona el botón para generar una tabla detallada con estadísticas calculadas para cada estación seleccionada.")
    if st.button("Calcular Estadísticas Detalladas"):
        with st.spinner("Realizando cálculos, por favor espera..."):
            try:
                detailed_stats_df = calculate_comprehensive_stats(df_anual_melted, df_monthly_filtered, stations_for_analysis)
                base_info_df = gdf_filtered[[Config.STATION_NAME_COL, Config.ALTITUDE_COL, Config.MUNICIPALITY_COL, Config.REGION_COL]].copy()
                base_info_df.rename(columns={Config.STATION_NAME_COL: 'Estación'}, inplace=True)
                final_df = pd.merge(base_info_df.drop_duplicates(subset=['Estación']), detailed_stats_df, on="Estación", how="right")
                column_order = ['Estación', 'municipio', 'depto_region', 'alt_est', 'Años con Datos', 'Ppt. Media Anual (mm)', 'Desv. Estándar Anual (mm)', 'Ppt. Máxima Anual (mm)', 'Año Ppt. Máxima', 'Ppt. Mínima Anual (mm)', 'Año Ppt. Mínima', 'Tendencia (mm/año)', 'Significancia (p-valor)', 'Ppt Media Ene (mm)', 'Ppt Media Feb (mm)', 'Ppt Media Mar (mm)', 'Ppt Media Abr (mm)', 'Ppt Media May (mm)', 'Ppt Media Jun (mm)', 'Ppt Media Jul (mm)', 'Ppt Media Ago (mm)', 'Ppt Media Sep (mm)', 'Ppt Media Oct (mm)', 'Ppt Media Nov (mm)', 'Ppt Media Dic (mm)']
                display_columns = [col for col in column_order if col in final_df.columns]
                final_df_display = final_df[display_columns]
                st.dataframe(final_df_display.style.format({'Ppt. Media Anual (mm)': '{:.1f}', 'Desv. Estándar Anual (mm)': '{:.1f}', 'Ppt. Máxima Anual (mm)': '{:.1f}', 'Ppt. Mínima Anual (mm)': '{:.1f}', 'Tendencia (mm/año)': '{:.2f}', 'Significancia (p-valor)': '{:.3f}', 'Ppt Media Ene (mm)': '{:.1f}', 'Ppt Media Feb (mm)': '{:.1f}', 'Ppt Media Mar (mm)': '{:.1f}', 'Ppt Media Abr (mm)': '{:.1f}', 'Ppt Media May (mm)': '{:.1f}', 'Ppt Media Jun (mm)': '{:.1f}', 'Ppt Media Jul (mm)': '{:.1f}', 'Ppt Media Ago (mm)': '{:.1f}', 'Ppt Media Sep (mm)': '{:.1f}', 'Ppt Media Oct (mm)': '{:.1f}', 'Ppt Media Nov (mm)': '{:.1f}', 'Ppt Media Dic (mm)': '{:.1f}'}))
            except Exception as e:
                st.error(f"Ocurrió un error al calcular las estadísticas: {e}")

def display_percentile_analysis_subtab(df_monthly_filtered, station_to_analyze_perc):
    """Realiza y muestra el análisis de sequías y eventos extremos por percentiles mensuales para una estación."""
    df_long = st.session_state.get('df_long')
    if df_long is None or df_long.empty:
        st.warning("No se puede realizar el análisis de percentiles. El DataFrame histórico no está disponible.")
        return

    st.markdown("#### Parámetros del Análisis")
    col1, col2 = st.columns(2)
    p_lower = col1.slider("Percentil Inferior (Sequía):", 1, 40, 10, key="p_lower_perc")
    p_upper = col2.slider("Percentil Superior (Húmedo):", 60, 99, 90, key="p_upper_perc")
    st.markdown("---")
    
    with st.spinner(f"Calculando percentiles P{p_lower} y P{p_upper} para {station_to_analyze_perc}..."):
        try:
            df_extremes, df_thresholds = calculate_percentiles_and_extremes(df_long, station_to_analyze_perc, p_lower, p_upper)
            
            year_range_val = st.session_state.get('year_range', (2000, 2020))
            if isinstance(year_range_val, tuple) and len(year_range_val) == 2 and isinstance(year_range_val[0], int):
                year_min, year_max = year_range_val
            else:
                year_min, year_max = st.session_state.get('year_range_single', (2000, 2020))

            df_plot = df_extremes[
                (df_extremes[Config.DATE_COL].dt.year >= year_min) &
                (df_extremes[Config.DATE_COL].dt.year <= year_max) &
                (df_extremes[Config.DATE_COL].dt.month.isin(st.session_state.meses_numeros))
            ].copy()

            if df_plot.empty:
                st.warning("No hay datos que coincidan con los filtros de tiempo para la estación seleccionada.")
                return

            st.subheader(f"Serie de Tiempo con Eventos Extremos (P{p_lower} y P{p_upper} Percentiles)")
            color_map = {f'Sequía Extrema (<P{p_lower}%)': 'red', f'Húmedo Extremo (>P{p_upper}%)': 'blue', 'Normal': 'gray'}
            fig_series = px.scatter(
                df_plot, x=Config.DATE_COL, y=Config.PRECIPITATION_COL,
                color='event_type',
                color_discrete_map=color_map,
                title=f"Precipitación Mensual y Eventos Extremos en {station_to_analyze_perc}",
                labels={Config.PRECIPITATION_COL: "Precipitación (mm)", Config.DATE_COL: "Fecha"},
                hover_data={'event_type': True, 'p_lower': ':.0f', 'p_upper': ':.0f'}
            )
            
            mean_precip = df_long[df_long[Config.STATION_NAME_COL] == station_to_analyze_perc][Config.PRECIPITATION_COL].mean()
            fig_series.add_hline(y=mean_precip, line_dash="dash", line_color="green", annotation_text="Media Histórica")
            fig_series.update_layout(height=500)
            st.plotly_chart(fig_series, use_container_width=True)

            # --- CÓDIGO ELIMINADO ---
            # El bloque de código que generaba el gráfico de "Umbrales" ha sido removido de aquí
            # porque ahora vive en su propia sub-pestaña.
            
        except Exception as e:
            st.error(f"Error al calcular el análisis de percentiles: {e}")
            st.info("Asegúrese de que el archivo histórico de datos ('df_long') contenga datos suficientes para la estación seleccionada.")

def display_forecast_tab(gdf_filtered, stations_for_analysis, **kwargs):
    st.header("Pronóstico del Tiempo a 7 Días (Open-Meteo)")

    if not stations_for_analysis:
        st.warning("Seleccione al menos una estación para ver el pronóstico.")
        return

    station_to_forecast = st.selectbox(
        "Seleccione una estación para obtener su pronóstico:",
        options=sorted(stations_for_analysis),
        key="forecast_station_select"
    )

    if station_to_forecast:
        station_info = gdf_filtered[gdf_filtered[Config.STATION_NAME_COL] == station_to_forecast].iloc[0]
        lat = station_info.geometry.y
        lon = station_info.geometry.x

        st.info(f"Obteniendo pronóstico para **{station_to_forecast}** (Lat: {lat:.4f}, Lon: {lon:.4f})")
        
        forecast_df = get_weather_forecast(lat, lon)

        if forecast_df is not None:
            # Gráfico de Pronóstico
            fig = go.Figure()

            # Barras para precipitación
            fig.add_trace(go.Bar(
                x=forecast_df['date'],
                y=forecast_df['precip_sum (mm)'],
                name='Precipitación (mm)',
                marker_color='blue',
                yaxis='y2' # Asignar al eje Y secundario
            ))

            # Líneas para temperatura
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['temp_max (°C)'],
                mode='lines+markers',
                name='Temp. Máxima',
                line=dict(color='red')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['temp_min (°C)'],
                mode='lines+markers',
                name='Temp. Mínima',
                line=dict(color='orange'),
                fill='tonexty', # Rellenar el área entre min y max
                fillcolor='rgba(255, 165, 0, 0.2)'
            ))
            
            fig.update_layout(
                title=f'Pronóstico para {station_to_forecast}',
                yaxis=dict(title='Temperatura (°C)'),
                yaxis2=dict(
                    title='Precipitación (mm)',
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                legend=dict(x=0, y=1.1, orientation="h")
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver datos del pronóstico en tabla"):
                st.dataframe(forecast_df.style.format({
                    "temp_max (°C)": "{:.1f}",
                    "temp_min (°C)": "{:.1f}",
                    "precip_sum (mm)": "{:.1f}"
                }))

def display_dashboard_tab(df_anual_melted, gdf_filtered, **kwargs):
    """
    Muestra el dashboard personalizado para el usuario que ha iniciado sesión.
    """
    st.header(f"Dashboard Personalizado de {st.session_state.get('name', 'Usuario')}")

    username = st.session_state.get("username")
    if not username:
        st.error("No se pudo identificar al usuario.")
        return

    preferences = db_manager.get_preferences(username)

    if not preferences:
        st.info("Aún no has guardado nada en tu dashboard. Busca el ícono 📌 para añadir gráficos.")
        return

    st.success(f"Mostrando {len(preferences)} elementos guardados.")
    st.markdown("---")

    for pref in preferences:
        widget_type = pref.get('type')
        params = pref.get('params', {})

        # --- Lógica para renderizar el Gráfico de Serie Anual ---
        if widget_type == 'annual_series_chart':
            st.subheader("Serie de Precipitación Anual Guardada")
            
            stations = params.get('stations', [])
            year_range = params.get('year_range', [1970, 2023])

            # Filtramos el DataFrame anual ya procesado
            df_plot = df_anual_melted[
                (df_anual_melted[Config.STATION_NAME_COL].isin(stations)) &
                (df_anual_melted[Config.YEAR_COL] >= year_range[0]) &
                (df_anual_melted[Config.YEAR_COL] <= year_range[1])
            ]

            if not df_plot.empty:
                chart = alt.Chart(df_plot.dropna(subset=['precipitation'])).mark_line(point=True).encode(
                    x=alt.X(f'{Config.YEAR_COL}:O', title='Año'),
                    y=alt.Y('precipitation:Q', title='Precipitación (mm)'),
                    color=f'{Config.STATION_NAME_COL}:N'
                ).properties(title=f"Estaciones: {', '.join(stations)}").interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("No se encontraron datos para los parámetros de este gráfico.")
            
            st.divider()

        # --- Lógica para renderizar el Mapa de Distribución ---
        elif widget_type == 'annual_avg_bar_chart':
            st.subheader("Precipitación Media Multianual Guardada")
            stations = params.get('stations', [])
            
            df_plot = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL].isin(stations)]

            if not df_plot.empty:
                df_summary = df_plot.groupby(Config.STATION_NAME_COL, as_index=False)['precipitation'].mean().round(0)
                df_summary = df_summary.sort_values('precipitation', ascending=False)
                
                fig = px.bar(df_summary, x=Config.STATION_NAME_COL, y='precipitation',
                             title=f"Promedio para: {', '.join(stations)}",
                             labels={Config.STATION_NAME_COL: 'Estación', 'precipitation': 'Precipitación Media Anual (mm)'},
                             color='precipitation')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No se encontraron datos para los parámetros de este gráfico de barras.")

            st.divider()
        
        # Puedes añadir más bloques 'elif' aquí para otros tipos de gráficos
        # elif widget_type == 'otro_tipo_de_grafico':
        #     ...
