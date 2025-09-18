# modules/visualizer.py

# --- Importaciones ---
import streamlit as st
import pandas as pd
import altair as alt
import folium
from folium.plugins import MarkerCluster, MiniMap
from folium.raster_layers import WmsTileLayer
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import base64
import branca.colormap as cm
from pykrige.ok import OrdinaryKriging
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf
from prophet import Prophet
from prophet.plot import plot_plotly
import pymannkendall as mk

# --- Importaciones de Módulos Propios ---
from modules.config import Config
from modules.utils import add_folium_download_button, add_plotly_download_buttons
from modules.data_processor import calculate_spi, interpolate_idw, interpolate_rbf_spline

# --- Funciones de Creación de Gráficos y Mapas ---

def create_enso_chart(enso_data):
    if enso_data.empty or Config.ENSO_ONI_COL not in enso_data.columns:
        return go.Figure()

    data = enso_data.copy().sort_values(Config.DATE_COL)
    data.dropna(subset=[Config.ENSO_ONI_COL], inplace=True)

    conditions = [data[Config.ENSO_ONI_COL] >= 0.5, data[Config.ENSO_ONI_COL] <= -0.5]
    phases = ['El Niño', 'La Niña']
    colors = ['red', 'blue']
    data['phase'] = np.select(conditions, phases, default='Neutral')
    data['color'] = np.select(conditions, colors, default='grey')
    y_range = [data[Config.ENSO_ONI_COL].min() - 0.5, data[Config.ENSO_ONI_COL].max() + 0.5]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data[Config.DATE_COL], y=[y_range[1] - y_range[0]] * len(data),
        base=y_range[0], marker_color=data['color'], width=30*24*60*60*1000,
        opacity=0.3, hoverinfo='none', showlegend=False
    ))
    legend_map = {'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'grey'}
    for phase, color in legend_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=15, color=color, symbol='square', opacity=0.5),
            name=phase, showlegend=True
        ))
    fig.add_trace(go.Scatter(
        x=data[Config.DATE_COL], y=data[Config.ENSO_ONI_COL],
        mode='lines', name='Anomalía ONI', line=dict(color='black', width=2), showlegend=True
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue")
    fig.update_layout(
        height=600, title="Fases del Fenómeno ENSO y Anomalía ONI",
        yaxis_title="Anomalía ONI (°C)", xaxis_title="Fecha", showlegend=True,
        legend_title_text='Fase', yaxis_range=y_range
    )
    return fig

def create_anomaly_chart(df_plot):
    if df_plot.empty:
        return go.Figure()
    df_plot['color'] = np.where(df_plot['anomalia'] < 0, 'red', 'blue')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot[Config.DATE_COL], y=df_plot['anomalia'],
        marker_color=df_plot['color'], name='Anomalía de Precipitación'
    ))
    if Config.ENSO_ONI_COL in df_plot.columns:
        df_plot_enso = df_plot.dropna(subset=[Config.ENSO_ONI_COL])
        nino_periods = df_plot_enso[df_plot_enso[Config.ENSO_ONI_COL] >= 0.5]
        for _, row in nino_periods.iterrows():
            fig.add_vrect(x0=row[Config.DATE_COL] - pd.DateOffset(days=15), x1=row[Config.DATE_COL] + pd.DateOffset(days=15),
                          fillcolor="red", opacity=0.15, layer="below", line_width=0)
        nina_periods = df_plot_enso[df_plot_enso[Config.ENSO_ONI_COL] <= -0.5]
        for _, row in nina_periods.iterrows():
            fig.add_vrect(x0=row[Config.DATE_COL] - pd.DateOffset(days=15), x1=row[Config.DATE_COL] + pd.DateOffset(days=15),
                          fillcolor="blue", opacity=0.15, layer="below", line_width=0)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', color='rgba(255, 0, 0, 0.3)'), name='Fase El Niño'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', color='rgba(0, 0, 255, 0.3)'), name='Fase La Niña'))
    fig.update_layout(
        height=600, title="Anomalías Mensuales de Precipitación y Fases ENSO",
        yaxis_title="Anomalía de Precipitación (mm)", xaxis_title="Fecha", showlegend=True
    )
    return fig

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
    m = folium.Map(location=location, zoom_start=zoom, tiles=base_map_config.get("tiles", "OpenStreetMap"), attr=base_map_config.get("attr", None))
    if fit_bounds_data is not None and not fit_bounds_data.empty:
        bounds = fit_bounds_data.total_bounds
        if np.all(np.isfinite(bounds)):
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    for layer_config in overlays_config:
        WmsTileLayer(url=layer_config["url"], layers=layer_config["layers"], fmt='image/png',
                     transparent=layer_config.get("transparent", False), overlay=True, control=True,
                     name=layer_config.get("attr", "Overlay")).add_to(m)
    return m

# --- Funciones para las Pestañas de la UI ---

def display_welcome_tab():
    st.header("Bienvenido al Sistema de Información de Lluvias y Clima")
    st.markdown(Config.WELCOME_TEXT, unsafe_allow_html=True)
    if os.path.exists(Config.LOGO_PATH):
        st.image(Config.LOGO_PATH, width=400, caption="Corporación Cuenca Verde")

def display_spatial_distribution_tab(gdf_filtered, stations_for_analysis, df_anual_melted, df_monthly_filtered):
    st.header("Distribución espacial de las Estaciones de Lluvia")
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")

    if not df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL]).empty:
        summary_stats = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].agg(['mean', 'count']).reset_index()
        summary_stats.rename(columns={'mean': 'precip_media_anual', 'count': 'años_validos'}, inplace=True)
        gdf_filtered = gdf_filtered.merge(summary_stats, on=Config.STATION_NAME_COL, how='left')
    else:
        gdf_filtered['precip_media_anual'] = np.nan
        gdf_filtered['años_validos'] = 0

    gdf_filtered['precip_media_anual'] = gdf_filtered['precip_media_anual'].fillna(0)
    gdf_filtered['años_validos'] = gdf_filtered['años_validos'].fillna(0).astype(int)

    sub_tab_mapa, sub_tab_grafico = st.tabs(["Mapa Interactivo", "Gráfico de Disponibilidad de Datos"])

    with sub_tab_mapa:
        controls_col, map_col = st.columns([1, 3])
        with controls_col:
            st.subheader("Controles del Mapa")
            selected_base_map_config, selected_overlays_config = display_map_controls(st, "dist_esp")
            if not gdf_filtered.empty:
                st.markdown("---")
                m1, m2 = st.columns([1, 3])
                with m1:
                    if os.path.exists(Config.LOGO_DROP_PATH):
                        st.image(Config.LOGO_DROP_PATH, width=50)
                with m2:
                    st.metric("Estaciones en Vista", len(gdf_filtered))
                st.markdown("---")
                map_centering = st.radio("Opciones de centrado:", ("Automático", "Vistas Predefinidas"), key="map_centering_radio")
                if 'map_view' not in st.session_state:
                    st.session_state.map_view = {"location": [4.57, -74.29], "zoom": 5}
                if map_centering == "Vistas Predefinidas":
                    if st.button("Ver Colombia"):
                        st.session_state.map_view = {"location": [4.57, -74.29], "zoom": 5}
                    if st.button("Ver Antioquia"):
                        st.session_state.map_view = {"location": [6.24, -75.58], "zoom": 8}
                    if st.button("Ajustar a Selección"):
                        if not gdf_filtered.empty:
                            bounds = gdf_filtered.total_bounds
                            if np.all(np.isfinite(bounds)):
                                center_lat = (bounds[1] + bounds[3]) / 2
                                center_lon = (bounds[0] + bounds[2]) / 2
                                st.session_state.map_view = {"location": [center_lat, center_lon], "zoom": 9}
                st.markdown("---")
                with st.expander("Resumen de Filtros Activos", expanded=True):
                    summary_text = f"**Período:** {st.session_state.year_range[0]} - {st.session_state.year_range[1]}\n\n"
                    summary_text += f"**% Mínimo de Datos:** {st.session_state.min_data_perc_slider}%\n\n"
                    if 'selected_altitudes' in st.session_state and st.session_state.selected_altitudes: summary_text += f"**Altitud:** {', '.join(st.session_state.selected_altitudes)}\n\n"
                    if 'selected_regions' in st.session_state and st.session_state.selected_regions: summary_text += f"**Región:** {', '.join(st.session_state.selected_regions)}\n\n"
                    if 'selected_municipios' in st.session_state and st.session_state.selected_municipios: summary_text += f"**Municipio:** {', '.join(st.session_state.selected_municipios)}\n\n"
                    if 'selected_celdas' in st.session_state and st.session_state.selected_celdas: summary_text += f"**Celda XY:** {', '.join(st.session_state.selected_celdas)}\n\n"
                    st.info(summary_text)

        with map_col:
            if not gdf_filtered.empty:
                m = create_folium_map(
                    location=st.session_state.map_view["location"],
                    zoom=st.session_state.map_view["zoom"],
                    base_map_config=selected_base_map_config,
                    overlays_config=selected_overlays_config,
                    fit_bounds_data=gdf_filtered if map_centering == "Automático" else None
                )
                
                if st.session_state.gdf_municipios is not None:
                    folium.GeoJson(st.session_state.gdf_municipios.to_json(), name='Municipios').add_to(m)
                
                marker_cluster = MarkerCluster(name='Estaciones').add_to(m)
                
                gdf_filtered_map = gdf_filtered.dropna(subset=[Config.LATITUDE_COL, Config.LONGITUDE_COL]).copy()

                for _, row in gdf_filtered_map.iterrows():
                    try:
                        total_years_in_period = st.session_state.year_range[1] - st.session_state.year_range[0] + 1
                        valid_years = row.get('años_validos', 0)
                        
                        popup_html = f"""
                            <b>Estación:</b> {row[Config.STATION_NAME_COL]}<br>
                            <b>Municipio:</b> {row.get(Config.MUNICIPALITY_COL, 'N/A')}<br>
                            <b>Promedio Anual:</b> {row.get('precip_media_anual', 0):.0f} mm<br>
                            <small>(Calculado con <b>{valid_years}</b> de <b>{total_years_in_period}</b> años del período)</small>
                        """
                        folium.Marker(
                            location=[row[Config.LATITUDE_COL], row[Config.LONGITUDE_COL]],
                            tooltip=row[Config.STATION_NAME_COL],
                            popup=popup_html
                        ).add_to(marker_cluster)
                    except Exception:
                        continue

                folium.LayerControl().add_to(m)
                m.add_child(MiniMap(toggle_display=True))
                folium_static(m, height=700, width="100%")
                add_folium_download_button(m, "mapa_distribucion.html")
            else:
                st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

    with sub_tab_grafico:
        st.subheader("Disponibilidad y Composición de Datos por Estación")
        if not gdf_filtered.empty:
            if st.session_state.analysis_mode == "Completar series (interpolación)":
                st.info("Mostrando la composición de datos originales vs. completados para el período seleccionado.")
                if not df_monthly_filtered.empty:
                    data_composition = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.ORIGIN_COL]).size().unstack(fill_value=0)
                    if 'Original' not in data_composition: data_composition['Original'] = 0
                    if 'Completado' not in data_composition: data_composition['Completado'] = 0
                    data_composition['total'] = data_composition['Original'] + data_composition['Completado']
                    data_composition['% Original'] = (data_composition['Original'] / data_composition['total']) * 100
                    data_composition['% Completado'] = (data_composition['Completado'] / data_composition['total']) * 100
                    sort_order_comp = st.radio("Ordenar por:", ["% Datos Originales (Mayor a Menor)", "% Datos Originales (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_comp")
                    if "Mayor a Menor" in sort_order_comp: data_composition = data_composition.sort_values("% Original", ascending=False)
                    elif "Menor a Mayor" in sort_order_comp: data_composition = data_composition.sort_values("% Original", ascending=True)
                    else: data_composition = data_composition.sort_index(ascending=True)
                    
                    df_plot = data_composition.reset_index().melt(
                        id_vars=Config.STATION_NAME_COL, value_vars=['% Original', '% Completado'],
                        var_name='Tipo de Dato', value_name='Porcentaje')
                    
                    fig_comp = px.bar(df_plot, x=Config.STATION_NAME_COL, y='Porcentaje', color='Tipo de Dato',
                                      title='Composición de Datos por Estación',
                                      labels={Config.STATION_NAME_COL: 'Estación', 'Porcentaje': '% del Período'},
                                      color_discrete_map={'% Original': '#1f77b4', '% Completado': '#ff7f0e'}, text_auto='.1f')
                    fig_comp.update_layout(height=600, xaxis={'categoryorder': 'trace'})
                    st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("Mostrando el porcentaje de disponibilidad de datos según el archivo de estaciones.")
                sort_order_disp = st.radio("Ordenar estaciones por:", ["% Datos (Mayor a Menor)", "% Datos (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_disp")
                df_chart = gdf_filtered.copy()
                if "% Datos (Mayor a Menor)" in sort_order_disp: df_chart = df_chart.sort_values(Config.PERCENTAGE_COL, ascending=False)
                elif "% Datos (Menor a Mayor" in sort_order_disp: df_chart = df_chart.sort_values(Config.PERCENTAGE_COL, ascending=True)
                else: df_chart = df_chart.sort_values(Config.STATION_NAME_COL, ascending=True)
                
                fig_disp = px.bar(df_chart, x=Config.STATION_NAME_COL, y=Config.PERCENTAGE_COL,
                                  title='Porcentaje de Disponibilidad de Datos Históricos',
                                  labels={Config.STATION_NAME_COL: 'Estación', Config.PERCENTAGE_COL: '% de Datos Disponibles'},
                                  color=Config.PERCENTAGE_COL, color_continuous_scale=px.colors.sequential.Viridis)
                fig_disp.update_layout(height=600, xaxis={'categoryorder':'trace'})
                st.plotly_chart(fig_disp, use_container_width=True)
        else:
            st.warning("No hay estaciones seleccionadas para mostrar el gráfico.")

def display_graphs_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis):
    st.header("Visualizaciones de Precipitación")
    
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]} y meses {', '.join([str(m) for m in st.session_state.meses_numeros])}.")

    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    sub_tab_anual, sub_tab_mensual, sub_tab_comparacion, sub_tab_distribucion, sub_tab_acumulada, sub_tab_altitud, sub_tab_regional = st.tabs(["Análisis Anual", "Análisis Mensual", "Comparación Rápida", "Distribución", "Acumulada", "Relación Altitud", "Serie Regional"])

    with sub_tab_anual:
        anual_graf_tab, anual_analisis_tab = st.tabs(["Gráfico de Serie Anual", "Análisis Multianual"])
        with anual_graf_tab:
            if not df_anual_melted.empty:
                st.subheader("Precipitación Anual (mm)")
                st.info("Solo se muestran los años con 10 o más meses de datos.")
                chart_anual = alt.Chart(df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])).mark_line(point=True).encode(
                    x=alt.X(f'{Config.YEAR_COL}:O', title='Año'),
                    y=alt.Y(f'{Config.PRECIPITATION_COL}:Q', title='Precipitación (mm)'),
                    color=f'{Config.STATION_NAME_COL}:N',
                    tooltip=[alt.Tooltip(Config.STATION_NAME_COL), alt.Tooltip(Config.YEAR_COL, format='d'), alt.Tooltip(f'{Config.PRECIPITATION_COL}:Q', format='.0f')]
                ).properties(title=f'Precipitación Anual por Estación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})').interactive()
                st.altair_chart(chart_anual, use_container_width=True)

        with anual_analisis_tab:
            if not df_anual_melted.empty:
                st.subheader("Precipitación Media Multianual")
                st.caption(f"Período de análisis: {st.session_state.year_range[0]} - {st.session_state.year_range[1]}")
                chart_type_annual = st.radio("Seleccionar tipo de gráfico:", ("Gráfico de Barras (Promedio)", "Gráfico de Cajas (Distribución)"), key="avg_chart_type_annual", horizontal=True)
                if chart_type_annual == "Gráfico de Barras (Promedio)":
                    df_summary = df_anual_melted.groupby(Config.STATION_NAME_COL, as_index=False)[Config.PRECIPITATION_COL].mean().round(0)
                    sort_order = st.radio("Ordenar estaciones por:", ["Promedio (Mayor a Menor)", "Promedio (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_annual_avg")
                    if "Mayor a Menor" in sort_order: df_summary = df_summary.sort_values(Config.PRECIPITATION_COL, ascending=False)
                    elif "Menor a Mayor" in sort_order: df_summary = df_summary.sort_values(Config.PRECIPITATION_COL, ascending=True)
                    else: df_summary = df_summary.sort_values(Config.STATION_NAME_COL, ascending=True)
                    
                    fig_avg = px.bar(df_summary, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL, title=f'Promedio de Precipitación Anual por Estación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})', labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Precipitación Media Anual (mm)'}, color=Config.PRECIPITATION_COL, color_continuous_scale=px.colors.sequential.Blues_r)
                    fig_avg.update_layout(height=600, xaxis={'categoryorder':'total descending' if "Mayor a Menor" in sort_order else ('total ascending' if "Menor a Mayor" in sort_order else 'trace')})
                    st.plotly_chart(fig_avg, use_container_width=True)
                else:
                    df_anual_filtered_for_box = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL].isin(stations_for_analysis)]
                    fig_box_annual = px.box(df_anual_filtered_for_box, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, points='all', title='Distribución de la Precipitación Anual por Estación', labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Precipitación Anual (mm)'})
                    fig_box_annual.update_layout(height=600)
                    st.plotly_chart(fig_box_annual, use_container_width=True)

    with sub_tab_mensual:
        mensual_graf_tab, mensual_enso_tab, mensual_datos_tab = st.tabs(["Gráfico de Serie Mensual", "Análisis ENSO en el Período", "Tabla de Datos"])
        with mensual_graf_tab:
            if not df_monthly_filtered.empty:
                control_col1, control_col2 = st.columns(2)
                chart_type = control_col1.radio("Tipo de Gráfico:", ["Líneas y Puntos", "Nube de Puntos", "Gráfico de Cajas (Distribución Mensual)"], key="monthly_chart_type")
                color_by = control_col2.radio("Colorear por:", ["Estación", "Mes"], key="monthly_color_by", disabled=(chart_type == "Gráfico de Cajas (Distribución Mensual)"))
                if chart_type != "Gráfico de Cajas (Distribución Mensual)":
                    base_chart = alt.Chart(df_monthly_filtered).encode(x=alt.X(f'{Config.DATE_COL}:T', title='Fecha'), y=alt.Y(f'{Config.PRECIPITATION_COL}:Q', title='Precipitación (mm)'), tooltip=[alt.Tooltip(Config.DATE_COL, format='%Y-%m'), alt.Tooltip(Config.PRECIPITATION_COL, format='.0f'), Config.STATION_NAME_COL, Config.ORIGIN_COL, alt.Tooltip(f'{Config.MONTH_COL}:N', title="Mes")])
                    if color_by == "Estación":
                        color_encoding = alt.Color(f'{Config.STATION_NAME_COL}:N', legend=alt.Legend(title="Estaciones"))
                    else:
                        color_encoding = alt.Color(f'month({Config.DATE_COL}):N', legend=alt.Legend(title="Meses"), scale=alt.Scale(scheme='tableau20'))
                    
                    if chart_type == "Líneas y Puntos":
                        line_chart = base_chart.mark_line(opacity=0.4, color='lightgray').encode(detail=f'{Config.STATION_NAME_COL}:N')
                        point_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                        final_chart = (line_chart + point_chart)
                    else:
                        point_chart = base_chart.mark_point(filled=True, size=60).encode(color=color_encoding)
                        final_chart = point_chart
                    st.altair_chart(final_chart.properties(title=f"Serie de Precipitación Mensual ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})").interactive(), use_container_width=True)
                else:
                    st.subheader("Distribución de la Precipitación Mensual")
                    fig_box_monthly = px.box(df_monthly_filtered, x=Config.MONTH_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, title='Distribución de la Precipitación por Mes', labels={Config.MONTH_COL: 'Mes', Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)', Config.STATION_NAME_COL: 'Estación'})
                    fig_box_monthly.update_layout(height=600)
                    st.plotly_chart(fig_box_monthly, use_container_width=True)
        with mensual_enso_tab:
            if st.session_state.df_enso is not None:
                enso_filtered = st.session_state.df_enso[(st.session_state.df_enso[Config.DATE_COL].dt.year >= st.session_state.year_range[0]) & (st.session_state.df_enso[Config.DATE_COL].dt.year <= st.session_state.year_range[1]) & (st.session_state.df_enso[Config.DATE_COL].dt.month.isin(st.session_state.meses_numeros))]
                fig_enso_mensual = create_enso_chart(enso_filtered)
                st.plotly_chart(fig_enso_mensual, use_container_width=True, key="enso_chart_mensual")

        with mensual_datos_tab:
            st.subheader("Datos de Precipitación Mensual Detallados")
            if not df_monthly_filtered.empty:
                df_values = df_monthly_filtered.pivot_table(index=Config.DATE_COL, columns=Config.STATION_NAME_COL, values=Config.PRECIPITATION_COL).round(0)
                st.dataframe(df_values)

    with sub_tab_comparacion:
        st.subheader("Comparación de Precipitación entre Estaciones")
        if len(stations_for_analysis) < 2:
            st.info("Seleccione al menos dos estaciones para comparar.")
        else:
            st.markdown("##### Precipitación Mensual Promedio")
            df_monthly_avg = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean().reset_index()
            fig_avg_monthly = px.line(df_monthly_avg, x=Config.MONTH_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                      labels={Config.MONTH_COL: 'Mes', Config.PRECIPITATION_COL: 'Precipitación Promedio (mm)'},
                                      title='Promedio de Precipitación Mensual por Estación')
            meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
            fig_avg_monthly.update_layout(height=600, xaxis = dict(tickmode = 'array', tickvals = list(meses_dict.values()), ticktext = list(meses_dict.keys())))
            st.plotly_chart(fig_avg_monthly, use_container_width=True)
            st.markdown("##### Distribución de Precipitación Anual")
            df_anual_filtered_for_box = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL].isin(stations_for_analysis)]
            fig_box_annual = px.box(df_anual_filtered_for_box, x=Config.STATION_NAME_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, points='all', title='Distribución de la Precipitación Anual por Estación', labels={Config.STATION_NAME_COL: 'Estación', Config.PRECIPITATION_COL: 'Precipitación Anual (mm)'})
            fig_box_annual.update_layout(height=600)
            st.plotly_chart(fig_box_annual, use_container_width=True)

    with sub_tab_distribucion:
        st.subheader("Distribución de la Precipitación")
        distribucion_tipo = st.radio("Seleccionar tipo de distribución:", ("Anual", "Mensual"), horizontal=True)
        plot_type = st.radio("Seleccionar tipo de gráfico:", ("Histograma", "Gráfico de Violín"), horizontal=True, key="distribucion_plot_type")
        
        if distribucion_tipo == "Anual":
            if not df_anual_melted.empty:
                if plot_type == "Histograma":
                    fig_hist_anual = px.histogram(df_anual_melted, x=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                                 title=f'Distribución Anual de Precipitación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})',
                                                 labels={Config.PRECIPITATION_COL: 'Precipitación Anual (mm)', 'count': 'Frecuencia'})
                    fig_hist_anual.update_layout(height=600)
                    st.plotly_chart(fig_hist_anual, use_container_width=True)
                else: # Gráfico de Violín Anual
                    fig_violin_anual = px.violin(df_anual_melted, y=Config.PRECIPITATION_COL, x=Config.STATION_NAME_COL, color=Config.STATION_NAME_COL, 
                                                box=True, points="all", title='Distribución Anual con Gráfico de Violín',
                                                labels={Config.PRECIPITATION_COL: 'Precipitación Anual (mm)', Config.STATION_NAME_COL: 'Estación'})
                    fig_violin_anual.update_layout(height=600)
                    st.plotly_chart(fig_violin_anual, use_container_width=True)
            else:
                st.info("No hay datos anuales para mostrar la distribución.")
        else:
            if not df_monthly_filtered.empty:
                if plot_type == "Histograma":
                    fig_hist_mensual = px.histogram(df_monthly_filtered, x=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                                    title=f'Distribución Mensual de Precipitación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})',
                                                    labels={Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)', 'count': 'Frecuencia'})
                    fig_hist_mensual.update_layout(height=600)
                    st.plotly_chart(fig_hist_mensual, use_container_width=True)
                else: # Gráfico de Violín Mensual
                    fig_violin_mensual = px.violin(df_monthly_filtered, y=Config.PRECIPITATION_COL, x=Config.MONTH_COL, color=Config.STATION_NAME_COL, 
                                                   box=True, points="all", title='Distribución Mensual con Gráfico de Violín',
                                                   labels={Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)', Config.MONTH_COL: 'Mes'})
                    fig_violin_mensual.update_layout(height=600)
                    st.plotly_chart(fig_violin_mensual, use_container_width=True)
            else:
                st.info("No hay datos mensuales para mostrar la distribución.")

    with sub_tab_acumulada:
        st.subheader("Precipitación Acumulada Anual")
        if not df_anual_melted.empty:
            df_acumulada = df_anual_melted.groupby([Config.YEAR_COL, Config.STATION_NAME_COL])[Config.PRECIPITATION_COL].sum().reset_index()
            fig_acumulada = px.bar(df_acumulada, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                   title=f'Precipitación Acumulada por Año ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})',
                                   labels={Config.YEAR_COL: 'Año', Config.PRECIPITATION_COL: 'Precipitación Acumulada (mm)'})
            fig_acumulada.update_layout(barmode='group', height=600)
            st.plotly_chart(fig_acumulada, use_container_width=True)
        else:
            st.info("No hay datos para calcular la precipitación acumulada.")

    with sub_tab_altitud:
        st.subheader("Relación entre Altitud y Precipitación")
        if not df_anual_melted.empty and not st.session_state.gdf_filtered[Config.ALTITUDE_COL].isnull().all():
            df_relacion = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
            df_relacion = df_relacion.merge(st.session_state.gdf_filtered[[Config.STATION_NAME_COL, Config.ALTITUDE_COL]].drop_duplicates(), on=Config.STATION_NAME_COL, how='left')
            fig_relacion = px.scatter(df_relacion, x=Config.ALTITUDE_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL,
                                      title='Relación entre Precipitación Media Anual y Altitud',
                                      labels={Config.ALTITUDE_COL: 'Altitud (m)', Config.PRECIPITATION_COL: 'Precipitación Media Anual (mm)'})
            fig_relacion.update_layout(height=600)
            st.plotly_chart(fig_relacion, use_container_width=True)
        else:
            st.info("No hay datos de altitud o precipitación disponibles para analizar la relación.")

    with sub_tab_regional: 
        st.subheader("Serie de Tiempo Promedio Regional (Múltiples Estaciones)")

        if len(stations_for_analysis) == 0:
            st.warning("Seleccione una o más estaciones en el panel lateral para calcular la serie regional.")
            return

        with st.spinner("Calculando serie de tiempo regional..."):
            # 1. Agrupar los datos mensuales filtrados por fecha y calcular el promedio de precipitación
            df_regional_avg = df_monthly_filtered.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean().reset_index()
            df_regional_avg.rename(columns={Config.PRECIPITATION_COL: 'Precipitación Promedio Regional'}, inplace=True)
            
            df_regional_avg['Estación'] = 'Promedio Regional'

            # 2. Preparar los datos individuales para comparación (solo si hay más de una estación)
            if len(stations_for_analysis) > 1:
                col_display = st.columns(1)[0]
                show_individual = col_display.checkbox("Superponer estaciones individuales", value=False)
                
                if show_individual and len(stations_for_analysis) <= 6:
                    # Combinar datos individuales y promedio regional para superposición
                    df_plot_combined = df_monthly_filtered.copy()
                    df_plot_combined.rename(columns={Config.PRECIPITATION_COL: 'Precipitación'}, inplace=True)
                    df_plot_combined['Tipo'] = df_plot_combined[Config.STATION_NAME_COL]

                    df_regional_avg.rename(columns={'Precipitación Promedio Regional': 'Precipitación'}, inplace=True)
                    df_regional_avg['Tipo'] = 'Promedio Regional'
                    
                    df_final_plot = pd.concat([
                        df_plot_combined[[Config.DATE_COL, 'Precipitación', 'Tipo']], 
                        df_regional_avg[[Config.DATE_COL, 'Precipitación', 'Tipo']]
                    ], ignore_index=True)

                    fig_regional = go.Figure()
                    
                    # Series individuales (gris, delgada)
                    for station in stations_for_analysis:
                        df_s = df_final_plot[df_final_plot['Tipo'] == station]
                        fig_regional.add_trace(go.Scatter(
                            x=df_s[Config.DATE_COL], 
                            y=df_s['Precipitación'], 
                            mode='lines', 
                            name=station, 
                            line=dict(color='rgba(128, 128, 128, 0.5)', width=1), 
                            showlegend=True
                        ))
                    
                    # Serie regional (azul, gruesa)
                    df_regional_trace = df_final_plot[df_final_plot['Tipo'] == 'Promedio Regional']
                    fig_regional.add_trace(go.Scatter(
                        x=df_regional_trace[Config.DATE_COL], 
                        y=df_regional_trace['Precipitación'], 
                        mode='lines', 
                        name='Promedio Regional', 
                        line=dict(color='#1f77b4', width=4)
                    ))
                    
                    fig_regional.update_layout(
                        title='Serie de Tiempo Regional vs. Estaciones Individuales',
                        xaxis_title="Fecha",
                        yaxis_title="Precipitación Mensual (mm)",
                        height=650
                    )
                    st.plotly_chart(fig_regional, use_container_width=True)
                
                else:
                    if show_individual:
                        st.info("Demasiadas estaciones seleccionadas para superponer. Mostrando solo el promedio regional.")
                        
                    # Graficar solo el promedio regional
                    df_regional_avg.rename(columns={'Precipitación Promedio Regional': 'Precipitación'}, inplace=True)
                    fig_regional = px.line(df_regional_avg, x=Config.DATE_COL, y='Precipitación',
                                           title=f'Serie de Tiempo Promedio Regional ({len(stations_for_analysis)} Estaciones)',
                                           labels={'Precipitación': 'Precipitación Promedio (mm)'},
                                           line_shape='linear')
                    fig_regional.update_traces(line=dict(width=3, color='#1f77b4'))
                    fig_regional.update_layout(height=650)
                    st.plotly_chart(fig_regional, use_container_width=True)
                    
            else:
                # Graficar solo el promedio regional si solo hay una estación o por defecto
                df_regional_avg.rename(columns={'Precipitación Promedio Regional': 'Precipitación'}, inplace=True)
                fig_regional = px.line(df_regional_avg, x=Config.DATE_COL, y='Precipitación',
                                       title=f'Serie de Tiempo Promedio Regional ({len(stations_for_analysis)} Estaciones)',
                                       labels={'Precipitación': 'Precipitación Promedio (mm)'},
                                       line_shape='linear')
                fig_regional.update_traces(line=dict(width=3, color='#1f77b4'))
                fig_regional.update_layout(height=650)
                st.plotly_chart(fig_regional, use_container_width=True)

        st.markdown("#### Datos de la Serie Regional")
        df_regional_avg.rename(columns={'Precipitación': 'Precipitación Promedio Regional'}, inplace=True) # Renombrar de vuelta si se usó la variable para plot
        st.dataframe(df_regional_avg.round(1), use_container_width=True)

def display_advanced_maps_tab(gdf_filtered, df_anual_melted, stations_for_analysis, df_monthly_filtered):
    st.header("Mapas Avanzados")
    st.info(f"Mostrando análisis para {len(stations_for_analysis)} estaciones en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")

    tab_names = ["Animación GIF (Antioquia)", "Mapa Interactivo de Estaciones", "Visualización Temporal", 
                 "Gráfico de Carrera", "Mapa Animado", "Comparación de Mapas", "Interpolación Comparativa"]
    gif_tab, mapa_interactivo_tab, temporal_tab, race_tab, anim_tab, compare_tab, kriging_tab = st.tabs(tab_names)

    with gif_tab:
        st.subheader("Distribución Espacio-Temporal de la Lluvia en Antioquia")
        if os.path.exists(Config.GIF_PATH):
            with open(Config.GIF_PATH, "rb") as file:
                contents = file.read()
                data_url = base64.b64encode(contents).decode("utf-8")
            st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="Animación PPAM" style="width:100%;">', unsafe_allow_html=True)
        else:
            st.warning("No se encontró el archivo GIF. Verifique la ruta en config.py")

    with mapa_interactivo_tab:
        st.subheader("Visualización de una Estación con Mini-gráfico de Precipitación")
        if len(stations_for_analysis) == 0:
            st.warning("Por favor, seleccione al menos una estación en el panel lateral para ver esta sección.")
        else:
            station_to_show = st.selectbox("Seleccione la estación a visualizar:", options=sorted(stations_for_analysis), key="station_map_select")
            if station_to_show:
                controls_col, map_col = st.columns([1, 3])
                with controls_col:
                    st.subheader("Controles del Mapa")
                    selected_base_map_config, selected_overlays_config = display_map_controls(st, "avanzado_estaciones")
                
                with map_col:
                    station_data_list = gdf_filtered[gdf_filtered[Config.STATION_NAME_COL] == station_to_show]
                    if not station_data_list.empty:
                        station_data = station_data_list.iloc[0]
                        m = create_folium_map(
                            location=[station_data[Config.LATITUDE_COL], station_data[Config.LONGITUDE_COL]],
                            zoom=12,
                            base_map_config=selected_base_map_config,
                            overlays_config=selected_overlays_config
                        )
                        
                        df_station_monthly_avg = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station_to_show]
                        if not df_station_monthly_avg.empty:
                            df_monthly_avg = df_station_monthly_avg.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                            
                            fig = go.Figure(data=[go.Bar(x=df_monthly_avg[Config.MONTH_COL], y=df_monthly_avg[Config.PRECIPITATION_COL])])
                            fig.update_layout(title=f"Ppt. Mensual Media<br>{station_data[Config.STATION_NAME_COL]}", 
                                              xaxis_title="Mes", yaxis_title="Ppt. (mm)", height=250, width=350,
                                              margin=dict(t=50, b=20, l=20, r=20))
                            
                            popup_html_chart = fig.to_html(full_html=False, include_plotlyjs='cdn')
                            
                            html_popup = f"""
                                <h4>{station_data[Config.STATION_NAME_COL]}</h4>
                                <p><b>Municipio:</b> {station_data.get(Config.MUNICIPALITY_COL, 'N/A')}</p>
                                <p><b>Altitud:</b> {station_data.get(Config.ALTITUDE_COL, 'N/A')} m</p>
                                {popup_html_chart}
                            """
                            folium.Marker(location=[station_data[Config.LATITUDE_COL], station_data[Config.LONGITUDE_COL]], popup=folium.Popup(html_popup, max_width=400)).add_to(m)
                        else:
                            st.warning(f"No hay datos mensuales para {station_to_show}. Se mostrará un marcador básico.")
                            html_popup = f"""
                                <h4>{station_data[Config.STATION_NAME_COL]}</h4>
                                <p><b>Municipio:</b> {station_data.get(Config.MUNICIPALITY_COL, 'N/A')}</p>
                                <p><b>Altitud:</b> {station_data.get(Config.ALTITUDE_COL, 'N/A')} m</p>
                            """
                            folium.Marker(location=[station_data[Config.LATITUDE_COL], station_data[Config.LONGITUDE_COL]], popup=html_popup).add_to(m)
                        
                        folium.LayerControl().add_to(m)
                        folium_static(m, height=700, width="100%")
        
    with temporal_tab:
        st.subheader("Explorador Anual de Precipitación")
        if len(stations_for_analysis) == 0:
            st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        elif not df_anual_melted.empty:
            df_anual_melted_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
            if not df_anual_melted_non_na.empty:
                all_years_int = sorted(df_anual_melted_non_na[Config.YEAR_COL].unique())
                selected_year = st.slider('Seleccione un Año para Explorar', min_value=min(all_years_int), max_value=max(all_years_int), value=min(all_years_int))
                
                min_precip_slider, max_precip_slider = int(df_anual_melted_non_na[Config.PRECIPITATION_COL].min()), int(df_anual_melted_non_na[Config.PRECIPITATION_COL].max())
                if min_precip_slider >= max_precip_slider: max_precip_slider = min_precip_slider + 1

                min_precip_filter, max_precip_filter = st.slider("Filtrar por rango de Precipitación Anual (mm)",
                                                                 min_value=min_precip_slider, max_value=max_precip_slider,
                                                                 value=(min_precip_slider, max_precip_slider), key="precip_range_filter")
                
                controls_col, map_col = st.columns([1, 3])
                with controls_col:
                    st.markdown("##### Opciones de Visualización")
                    selected_base_map_config, selected_overlays_config = display_map_controls(st, "temporal")
                    st.markdown(f"#### Resumen del Año: {selected_year}")
                    df_year_filtered = df_anual_melted[
                        (df_anual_melted[Config.YEAR_COL] == selected_year) & 
                        (df_anual_melted[Config.PRECIPITATION_COL] >= min_precip_filter) &
                        (df_anual_melted[Config.PRECIPITATION_COL] <= max_precip_filter)
                    ].dropna(subset=[Config.PRECIPITATION_COL])

                    logo_col, info_col = st.columns([1, 4])
                    with logo_col:
                        if os.path.exists(Config.LOGO_DROP_PATH): st.image(Config.LOGO_DROP_PATH, width=40)
                    with info_col:
                        st.metric(f"Estaciones con Datos en {selected_year}", f"{len(df_year_filtered)} de {len(stations_for_analysis)}")
                    if not df_year_filtered.empty:
                        max_row = df_year_filtered.loc[df_year_filtered[Config.PRECIPITATION_COL].idxmax()]
                        min_row = df_year_filtered.loc[df_year_filtered[Config.PRECIPITATION_COL].idxmin()]
                        st.info(f"""
                        **Ppt. Máxima ({selected_year}):**
                        {max_row[Config.STATION_NAME_COL]} ({max_row[Config.PRECIPITATION_COL]:.0f} mm)

                        **Ppt. Mínima ({selected_year}):**
                        {min_row[Config.STATION_NAME_COL]} ({min_row[Config.PRECIPITATION_COL]:.0f} mm)
                        """)
                    else:
                        st.warning(f"No hay datos de precipitación para el año {selected_year} con los filtros aplicados.")
                with map_col:
                    m_temporal = create_folium_map([6.24, -75.58], 7, selected_base_map_config, selected_overlays_config)
                    if not df_year_filtered.empty:
                        min_val, max_val = df_anual_melted_non_na[Config.PRECIPITATION_COL].min(), df_anual_melted_non_na[Config.PRECIPITATION_COL].max()
                        if min_val >= max_val: max_val = min_val + 1 
                        colormap = cm.linear.YlGnBu_09.scale(vmin=min_val, vmax=max_val)
                        for _, row in df_year_filtered.iterrows():
                            folium.CircleMarker(
                                location=[row[Config.LATITUDE_COL], row[Config.LONGITUDE_COL]], radius=5,
                                color=colormap(row[Config.PRECIPITATION_COL]), fill=True, fill_color=colormap(row[Config.PRECIPITATION_COL]),
                                fill_opacity=0.8, tooltip=f"{row[Config.STATION_NAME_COL]}: {row[Config.PRECIPITATION_COL]:.0f} mm"
                            ).add_to(m_temporal)
                        bounds = st.session_state.gdf_stations.loc[st.session_state.gdf_stations[Config.STATION_NAME_COL].isin(df_year_filtered[Config.STATION_NAME_COL])].total_bounds
                        if np.all(np.isfinite(bounds)):
                            m_temporal.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                    folium.LayerControl().add_to(m_temporal)
                    folium_static(m_temporal, height=700, width="100%")

    with race_tab:
        st.subheader("Ranking Anual de Precipitación por Estación")
        if not df_anual_melted.empty:
            df_anual_melted_sorted = df_anual_melted.sort_values([Config.YEAR_COL, Config.PRECIPITATION_COL])
            fig_racing = px.bar(
                df_anual_melted_sorted, x=Config.PRECIPITATION_COL, y=Config.STATION_NAME_COL,
                animation_frame=Config.YEAR_COL, orientation='h',
                labels={Config.PRECIPITATION_COL: 'Precipitación Anual (mm)', Config.STATION_NAME_COL: 'Estación'},
                title=f"Evolución de Precipitación Anual por Estación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})"
            )
            fig_racing.update_traces(texttemplate='%{x:.0f}', textposition='outside')
            fig_racing.update_layout(
                xaxis_range=[0, df_anual_melted[Config.PRECIPITATION_COL].max() * 1.15],
                height=max(600, len(stations_for_analysis) * 35),
                title_font_size=20, font_size=12,
                yaxis=dict(categoryorder='total ascending')
            )
            fig_racing.layout.sliders[0]['currentvalue']['font']['size'] = 24
            fig_racing.layout.sliders[0]['currentvalue']['prefix'] = '<b>Año: </b>'
            fig_racing.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
            fig_racing.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
            st.plotly_chart(fig_racing, use_container_width=True)

    with anim_tab:
        st.subheader("Mapa Animado de Precipitación Anual")
        st.info("Este mapa utiliza Plotly. Los controles de mapa se encuentran en las otras pestañas de mapas.")
        if not df_anual_melted.empty:
            all_years = sorted(df_anual_melted[Config.YEAR_COL].unique())
            if all_years:
                all_selected_stations_info = st.session_state.gdf_stations.loc[st.session_state.gdf_stations[Config.STATION_NAME_COL].isin(stations_for_analysis)][[Config.STATION_NAME_COL, Config.LATITUDE_COL, Config.LONGITUDE_COL, Config.ALTITUDE_COL]].drop_duplicates()
                full_grid = pd.MultiIndex.from_product([all_selected_stations_info[Config.STATION_NAME_COL], all_years], names=[Config.STATION_NAME_COL, Config.YEAR_COL]).to_frame(index=False)
                full_grid = pd.merge(full_grid, all_selected_stations_info, on=Config.STATION_NAME_COL)
                df_anim_complete = pd.merge(full_grid, df_anual_melted[[Config.STATION_NAME_COL, Config.YEAR_COL, Config.PRECIPITATION_COL]], on=[Config.STATION_NAME_COL, Config.YEAR_COL], how='left')
                df_anim_complete['texto_tooltip'] = df_anim_complete.apply(lambda row: f"<b>Estación:</b> {row[Config.STATION_NAME_COL]}<br><b>Precipitación:</b> {row[Config.PRECIPITATION_COL]:.0f} mm" if pd.notna(row[Config.PRECIPITATION_COL]) else f"<b>Estación:</b> {row[Config.STATION_NAME_COL]}<br><b>Precipitación:</b> Sin datos", axis=1)
                df_anim_complete['precipitacion_plot'] = df_anim_complete[Config.PRECIPITATION_COL].fillna(0)
                min_precip_anim, max_precip_anim = df_anual_melted[Config.PRECIPITATION_COL].min(), df_anual_melted[Config.PRECIPITATION_COL].max()
                
                fig_mapa_animado = px.scatter_geo(df_anim_complete,
                                                  lat=Config.LATITUDE_COL, lon=Config.LONGITUDE_COL,
                                                  color='precipitacion_plot', size='precipitacion_plot',
                                                  hover_name=Config.STATION_NAME_COL,
                                                  hover_data={Config.LATITUDE_COL: False, Config.LONGITUDE_COL: False, 'precipitacion_plot': False, 'texto_tooltip': True},
                                                  animation_frame=Config.YEAR_COL,
                                                  projection='natural earth', 
                                                  title=f'Precipitación Anual por Estación ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})',
                                                  color_continuous_scale=px.colors.sequential.YlGnBu, range_color=[min_precip_anim, max_precip_anim])
                fig_mapa_animado.update_traces(hovertemplate='%{customdata[0]}')
                fig_mapa_animado.update_geos(fitbounds="locations", visible=True, showcoastlines=True, coastlinewidth=0.5, showland=True, landcolor="rgb(243, 243, 243)", showocean=True, oceancolor="rgb(220, 235, 255)", showcountries=True, countrywidth=0.5)
                fig_mapa_animado.update_layout(height=700, sliders=[dict(currentvalue=dict(font=dict(size=24, color="#707070"), prefix='<b>Año: </b>', visible=True))])
                st.plotly_chart(fig_mapa_animado, use_container_width=True)
    
    with compare_tab:
        st.subheader("Comparación de Mapas Anuales")
        df_anual_melted_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        if len(stations_for_analysis) == 0:
            st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        elif not df_anual_melted_non_na.empty and len(df_anual_melted_non_na[Config.YEAR_COL].unique()) > 0:
            control_col, map_col1, map_col2 = st.columns([1, 2, 2])
            with control_col:
                st.markdown("##### Controles de Mapa")
                selected_base_map_config, selected_overlays_config = display_map_controls(st, "compare")
                min_year, max_year = int(df_anual_melted_non_na[Config.YEAR_COL].min()), int(df_anual_melted_non_na[Config.YEAR_COL].max())
                year1 = st.slider("Seleccione el año para el Mapa 1", min_year, max_year, max_year, key="compare_year1")
                year2 = st.slider("Seleccione el año para el Mapa 2", min_year, max_year, max_year - 1 if max_year > min_year else max_year, key="compare_year2")
                
                min_precip_comp, max_precip_comp = int(df_anual_melted_non_na[Config.PRECIPITATION_COL].min()), int(df_anual_melted_non_na[Config.PRECIPITATION_COL].max())
                if min_precip_comp >= max_precip_comp:
                    max_precip_comp = min_precip_comp + 1
                
                color_range_comp = st.slider("Rango de Escala de Color (mm)", min_precip_comp, max_precip_comp, (min_precip_comp, max_precip_comp), key="color_comp")

            data_year1 = df_anual_melted[df_anual_melted[Config.YEAR_COL] == year1]
            data_year2 = df_anual_melted[df_anual_melted[Config.YEAR_COL] == year2]
            
            colormap = cm.linear.YlGnBu_09.scale(vmin=color_range_comp[0], vmax=color_range_comp[1])

            def create_compare_map(data, year, col):
                col.markdown(f"**Precipitación en {year}**")
                m = create_folium_map([6.24, -75.58], 6, selected_base_map_config, selected_overlays_config)
                if not data.empty:
                    for _, row in data.iterrows():
                        if pd.notna(row[Config.PRECIPITATION_COL]):
                            folium.CircleMarker(
                                location=[row[Config.LATITUDE_COL], row[Config.LONGITUDE_COL]], radius=5, color=colormap(row[Config.PRECIPITATION_COL]),
                                fill=True, fill_color=colormap(row[Config.PRECIPITATION_COL]), fill_opacity=0.8,
                                tooltip=f"{row[Config.STATION_NAME_COL]}: {row[Config.PRECIPITATION_COL]:.0f} mm"
                            ).add_to(m)
                    bounds = st.session_state.gdf_stations.loc[st.session_state.gdf_stations[Config.STATION_NAME_COL].isin(data[Config.STATION_NAME_COL])].total_bounds
                    if np.all(np.isfinite(bounds)):
                        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                folium.LayerControl().add_to(m)
                with col:
                    folium_static(m, height=600, width="100%")

            create_compare_map(data_year1, year1, map_col1)
            create_compare_map(data_year2, year2, map_col2)
        else:
            st.warning("No hay años disponibles para la comparación.")

    with kriging_tab:
        st.subheader("Comparación de Superficies de Interpolación Anual")
        df_anual_non_na = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
        if df_anual_non_na.empty or len(df_anual_non_na[Config.YEAR_COL].unique()) == 0:
            st.warning("No hay suficientes datos anuales para realizar la interpolación.")
        else:
            min_year, max_year = int(df_anual_non_na[Config.YEAR_COL].min()), int(df_anual_non_na[Config.YEAR_COL].max())
            control_col, map_col1, map_col2 = st.columns([1, 2, 2])
            
            with control_col:
                st.markdown("##### Controles de los Mapas")
                st.markdown("**Mapa 1**")
                year1 = st.slider("Seleccione el año", min_year, max_year, max_year, key="interp_year1")
                method1 = st.selectbox("Método de interpolación", options=["Kriging Ordinario", "IDW", "Spline (Thin Plate)"], key="interp_method1")
                st.markdown("---")
                st.markdown("**Mapa 2**")
                year2 = st.slider("Seleccione el año", min_year, max_year, max_year - 1 if max_year > min_year else max_year, key="interp_year2")
                method2 = st.selectbox("Método de interpolación", options=["Kriging Ordinario", "IDW", "Spline (Thin Plate)"], index=1, key="interp_method2")

            def generate_interpolation_map(year, method, gdf_filtered):
                data_year = df_anual_non_na[df_anual_non_na[Config.YEAR_COL] == year].copy()
                
                if len(data_year) < 4:
                    fig = go.Figure()
                    fig.update_layout(title=f"Datos insuficientes para {method} en {year} (se necesitan >= 4)", xaxis_visible=False, yaxis_visible=False)
                    return fig

                lons, lats, vals = data_year[Config.LONGITUDE_COL], data_year[Config.LATITUDE_COL], data_year[Config.PRECIPITATION_COL]
                bounds = gdf_filtered.total_bounds
                grid_lon = np.linspace(bounds[0] - 0.1, bounds[2] + 0.1, 100)
                grid_lat = np.linspace(bounds[1] - 0.1, bounds[3] + 0.1, 100)
                z_grid = None

                try:
                    if method == "Kriging Ordinario":
                        ok = OrdinaryKriging(lons.values, lats.values, vals.values, variogram_model='linear', verbose=False, enable_plotting=False)
                        z_grid, _ = ok.execute('grid', grid_lon, grid_lat)
                    elif method == "IDW":
                        z_grid = interpolate_idw(lons.values, lats.values, vals.values, grid_lon, grid_lat)
                    elif method == "Spline (Thin Plate)":
                        z_grid = interpolate_rbf_spline(lons.values, lats.values, vals.values, grid_lon, grid_lat, function='thin_plate_spline')
                except Exception as e:
                    st.error(f"Error al calcular {method} para el año {year}: {e}")
                    return go.Figure().update_layout(title=f"Error en {method} para {year}")
                
                if z_grid is not None:
                    fig = go.Figure(data=go.Contour(z=z_grid.T, x=grid_lon, y=grid_lat, 
                                                     colorscale='YlGnBu', contours=dict(showlabels=True, labelfont=dict(size=10, color='white'))))
                    fig.add_trace(go.Scatter(x=lons, y=lats, mode='markers', marker=dict(color='red', size=5), name='Estaciones'))
                    fig.update_layout(title=f"Precipitación en {year} ({method})", height=600)
                    return fig
                return go.Figure().update_layout(title="Error: Método no implementado")

            with map_col1:
                with st.spinner(f"Generando mapa 1 ({year1}, {method1})..."):
                    fig1 = generate_interpolation_map(year1, method1, gdf_filtered)
                    st.plotly_chart(fig1, use_container_width=True)
            
            with map_col2:
                with st.spinner(f"Generando mapa 2 ({year2}, {method2})..."):
                    fig2 = generate_interpolation_map(year2, method2, gdf_filtered)
                    st.plotly_chart(fig2, use_container_width=True)

def display_drought_analysis_tab(df_monthly_filtered, stations_for_analysis):
    st.header("Análisis de Extremos Hidrológicos")
    st.markdown("Esta sección ofrece dos metodologías para identificar eventos extremos: el **análisis de percentiles** para extremos puntuales y el **Índice Estandarizado de Precipitación (SPI)** para evaluar la intensidad de la sequía o humedad.")

    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    percentile_sub_tab, spi_sub_tab = st.tabs(["📈 Análisis por Percentiles", "💧 Análisis SPI"])

    with percentile_sub_tab:
        station_to_analyze_perc = st.selectbox("Seleccione una estación para el análisis de percentiles:",
                                               options=sorted(stations_for_analysis),
                                               key="percentile_station_select")
        if station_to_analyze_perc:
            display_percentile_analysis_subtab(df_monthly_filtered, station_to_analyze_perc)

    with spi_sub_tab:
        st.subheader("Análisis con el Índice Estandarizado de Precipitación (SPI)")
        col1_spi, col2_spi = st.columns([1, 2])
        with col1_spi:
            station_to_analyze_spi = st.selectbox("Seleccione una estación para el análisis SPI:",
                                                  options=sorted(stations_for_analysis),
                                                  key="spi_station_select")
            spi_window = st.select_slider("Seleccione la escala de tiempo del SPI (meses):",
                                          options=[3, 6, 9, 12, 24], value=12, key="spi_window_slider",
                                          help="Una escala corta (3 meses) refleja sequías agrícolas. Una escala larga (12-24 meses) refleja sequías hidrológicas.")

        if station_to_analyze_spi:
            df_station_spi = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station_to_analyze_spi].copy()
            df_station_spi = df_station_spi.set_index(Config.DATE_COL).sort_index()
            precip_series = df_station_spi[Config.PRECIPITATION_COL]

            if len(precip_series.dropna()) < spi_window * 2:
                with col2_spi:
                    st.warning(f"No hay suficientes datos ({len(precip_series.dropna())} meses) para calcular el SPI-{spi_window}.")
            else:
                with st.spinner(f"Calculando SPI-{spi_window}..."):
                    df_station_spi['spi'] = calculate_spi(precip_series, spi_window)
                df_plot = df_station_spi.dropna(subset=['spi']).copy()
                conditions = [
                    df_plot['spi'] <= -2.0, (df_plot['spi'] > -2.0) & (df_plot['spi'] <= -1.5),
                    (df_plot['spi'] > -1.5) & (df_plot['spi'] <= -1.0), (df_plot['spi'] > -1.0) & (df_plot['spi'] < 1.0),
                    (df_plot['spi'] >= 1.0) & (df_plot['spi'] < 1.5), (df_plot['spi'] >= 1.5) & (df_plot['spi'] < 2.0),
                    df_plot['spi'] >= 2.0
                ]
                colors = ['#b2182b', '#ef8a62', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
                df_plot['color'] = np.select(conditions, colors, default='grey')
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['spi'], marker_color=df_plot['color'], name='SPI'))
                fig.update_layout(title=f"Índice Estandarizado de Precipitación (SPI-{spi_window}) para {station_to_analyze_spi}",
                                  yaxis_title="Valor SPI", xaxis_title="Fecha", height=600)
                with col2_spi:
                    st.plotly_chart(fig, use_container_width=True)
                with st.expander("Ver tabla de datos SPI"):
                    st.dataframe(df_plot[['spi']].style.format("{:.2f}"))
                    
def display_anomalies_tab(df_long, df_monthly_filtered, stations_for_analysis):
    st.header("Análisis de Anomalías de Precipitación")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    if df_long is not None and not df_long.empty:
        df_long_filtered_stations = df_long[df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)]
        if df_long_filtered_stations.empty:
            st.warning("No hay datos de anomalías para la selección actual.")
            return

        df_climatology = df_long_filtered_stations.groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean().reset_index().rename(columns={Config.PRECIPITATION_COL: 'precip_promedio_mes'})
        df_anomalias = pd.merge(df_monthly_filtered, df_climatology, on=[Config.STATION_NAME_COL, Config.MONTH_COL], how='left')
        df_anomalias['anomalia'] = df_anomalias[Config.PRECIPITATION_COL] - df_anomalias['precip_promedio_mes']

        if st.session_state.exclude_na:
            df_anomalias.dropna(subset=['anomalia'], inplace=True)

        if df_anomalias.empty or df_anomalias['anomalia'].isnull().all():
            st.warning("No hay suficientes datos históricos para las estaciones y el período seleccionado para calcular y mostrar las anomalías.")
            return

        anom_graf_tab, anom_fase_tab, anom_extremos_tab = st.tabs(["Gráfico de Anomalías", "Anomalías por Fase ENSO", "Tabla de Eventos Extremos"])

        with anom_graf_tab:
            df_plot = df_anomalias.groupby(Config.DATE_COL).agg(
                anomalia=('anomalia', 'mean'),
                anomalia_oni=(Config.ENSO_ONI_COL, 'first')
            ).reset_index()
            fig = create_anomaly_chart(df_plot)
            st.plotly_chart(fig, use_container_width=True)

        with anom_fase_tab:
            if Config.ENSO_ONI_COL in df_anomalias.columns:
                df_anomalias_enso = df_anomalias.dropna(subset=[Config.ENSO_ONI_COL]).copy()
                conditions = [df_anomalias_enso[Config.ENSO_ONI_COL] >= 0.5, df_anomalias_enso[Config.ENSO_ONI_COL] <= -0.5]
                phases = ['El Niño', 'La Niña']
                df_anomalias_enso['enso_fase'] = np.select(conditions, phases, default='Neutral')
                fig_box = px.box(df_anomalias_enso, x='enso_fase', y='anomalia', color='enso_fase',
                                 title="Distribución de Anomalías de Precipitación por Fase ENSO",
                                 labels={'anomalia': 'Anomalía de Precipitación (mm)', 'enso_fase': 'Fase ENSO'},
                                 points='all')
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("La columna 'anomalia_oni' no está disponible para este análisis.")

        with anom_extremos_tab:
            st.subheader("Eventos Mensuales Extremos (Basado en Anomalías)")
            df_extremos = df_anomalias.dropna(subset=['anomalia']).copy()
            df_extremos['fecha'] = df_extremos[Config.DATE_COL].dt.strftime('%Y-%m')
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 10 Meses más Secos")
                secos = df_extremos.nsmallest(10, 'anomalia')[['fecha', Config.STATION_NAME_COL, 'anomalia', Config.PRECIPITATION_COL, 'precip_promedio_mes']]
                st.dataframe(secos.rename(columns={Config.STATION_NAME_COL: 'Estación', 'anomalia': 'Anomalía (mm)', Config.PRECIPITATION_COL: 'Ppt. (mm)', 'precip_promedio_mes': 'Ppt. Media (mm)'}).round(0), use_container_width=True)
            with col2:
                humedos = df_extremos.nlargest(10, 'anomalia')[['fecha', Config.STATION_NAME_COL, 'anomalia', Config.PRECIPITATION_COL, 'precip_promedio_mes']]
                st.dataframe(humedos.rename(columns={Config.STATION_NAME_COL: 'Estación', 'anomalia': 'Anomalía (mm)', Config.PRECIPITATION_COL: 'Ppt. (mm)', 'precip_promedio_mes': 'Ppt. Media (mm)'}).round(0), use_container_width=True)
    else:
        st.warning("No se puede realizar el análisis de anomalías. El DataFrame de datos mensuales no está disponible.")

def display_stats_tab(df_long, df_anual_melted, df_monthly_filtered, stations_for_analysis):
    st.header("Estadísticas de Precipitación")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    matriz_tab, resumen_mensual_tab, sintesis_tab = st.tabs(["Matriz de Disponibilidad", "Resumen Mensual", "Síntesis General"])

    with matriz_tab:
        st.subheader("Matriz de Disponibilidad de Datos Anual")
        
        # Determinar el DataFrame base según el modo de análisis
        df_base_raw = st.session_state.df_monthly_processed.copy()
        
        # Controles de vista (solo si la interpolación está activada)
        if st.session_state.analysis_mode == "Completar series (interpolación)":
            
            view_mode = st.radio("Seleccione la vista de la matriz:", 
                                 ("Porcentaje de Datos Originales", 
                                  "Porcentaje de Datos Totales (Original + Completado)",
                                  "Porcentaje de Datos Completados (Interpolados)"), 
                                 horizontal=True)
            
            df_base_filtered = df_base_raw[df_base_raw[Config.STATION_NAME_COL].isin(stations_for_analysis)].copy()
            
            if view_mode == "Porcentaje de Datos Completados (Interpolados)":
                df_counts = df_base_filtered[df_base_filtered[Config.ORIGIN_COL] == 'Completado'].groupby(
                    [Config.STATION_NAME_COL, Config.YEAR_COL]
                ).size().reset_index(name='count_completed')
                
                # Merge con la cuadrícula completa de estaciones/años
                all_months_count = df_base_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).size().reset_index(name='total_months')
                df_merged = pd.merge(all_months_count, df_counts, on=[Config.STATION_NAME_COL, Config.YEAR_COL], how='left').fillna(0)
                
                df_merged['porc_value'] = (df_merged['count_completed'] / 12) * 100
                heatmap_df = df_merged.pivot(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values='porc_value').fillna(0)
                color_scale = "Reds"
                title_text = "Porcentaje de Datos Completados (Interpolados)"
            
            elif view_mode == "Porcentaje de Datos Totales (Original + Completado)":
                # Contar el total de meses *disponibles* (que es casi siempre 12 si la interpolación funciona)
                df_counts = df_base_filtered.groupby(
                    [Config.STATION_NAME_COL, Config.YEAR_COL]
                ).size().reset_index(name='count_total')
                df_counts['porc_value'] = (df_counts['count_total'] / 12) * 100
                heatmap_df = df_counts.pivot(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values='porc_value').fillna(0)
                color_scale = "Blues"
                title_text = "Disponibilidad Promedio de Datos Totales (Originales + Completados)"
            
            else: # Porcentaje de Datos Originales
                df_original_filtered = df_long[df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)]
                df_counts = df_original_filtered.groupby(
                    [Config.STATION_NAME_COL, Config.YEAR_COL]
                ).size().reset_index(name='count_original')
                df_counts['porc_value'] = (df_counts['count_original'] / 12) * 100
                heatmap_df = df_counts.pivot(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values='porc_value').fillna(0)
                color_scale = "Greens"
                title_text = "Disponibilidad Promedio de Datos Originales"

        else: # Usar datos originales (Modo por defecto)
            df_base_filtered = df_long[df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)].copy()
            df_counts = df_base_filtered.groupby(
                [Config.STATION_NAME_COL, Config.YEAR_COL]
            ).size().reset_index(name='count_total')
            df_counts['porc_value'] = (df_counts['count_total'] / 12) * 100
            heatmap_df = df_counts.pivot(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values='porc_value').fillna(0)
            color_scale = "Greens"
            title_text = "Disponibilidad Promedio de Datos Originales"
            
        # --------------------
        # Aplicación de filtros de exclusión para cálculo de la métrica (si aplica)
        # --------------------
        if st.session_state.exclude_na:
            heatmap_df = heatmap_df.loc[:, ~heatmap_df.isnull().all()].dropna(how='all')
        if st.session_state.exclude_zeros:
            # Los ceros se manejan mejor en el filtrado mensual/anual, pero aquí se muestra la disponibilidad
            pass
        # --------------------
        
        if not heatmap_df.empty:
            avg_availability = heatmap_df.stack().mean()
            logo_col, metric_col = st.columns([1, 5])
            with logo_col:
                if os.path.exists(Config.LOGO_DROP_PATH): st.image(Config.LOGO_DROP_PATH, width=50)
            with metric_col: st.metric(label=title_text, value=f"{avg_availability:.1f}%")
            
            styled_df = heatmap_df.style.background_gradient(cmap=color_scale, axis=None, vmin=0, vmax=100).format("{:.0f}%", na_rep="-").set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white'), ('font-size', '14px')]},
                {'selector': 'td', 'props': [('text-align', 'center')]}])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No hay datos para mostrar en la matriz con la selección actual.")

    with resumen_mensual_tab:
        st.subheader("Resumen de Estadísticas Mensuales por Estación")
        if not df_monthly_filtered.empty:
            summary_data = []
            for station_name, group in df_monthly_filtered.groupby(Config.STATION_NAME_COL):
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
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.round(0), use_container_width=True)
        else:
            st.info("No hay datos para mostrar el resumen mensual.")

    with sintesis_tab:
        st.subheader("Síntesis General de Precipitación")
        if not df_monthly_filtered.empty and not df_anual_melted.empty:
            df_anual_valid = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
            if not df_anual_valid.empty:
                max_annual_row = df_anual_valid.loc[df_anual_valid[Config.PRECIPITATION_COL].idxmax()]
                max_monthly_row = df_monthly_filtered.loc[df_monthly_filtered[Config.PRECIPITATION_COL].idxmax()]
                meses_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Máxima Ppt. Anual Registrada",
                        f"{max_annual_row[Config.PRECIPITATION_COL]:.0f} mm",
                        f"{max_annual_row[Config.STATION_NAME_COL]} (Año {int(max_annual_row[Config.YEAR_COL])})"
                    )
                with col2:
                    st.metric(
                        "Máxima Ppt. Mensual Registrada",
                        f"{max_monthly_row[Config.PRECIPITATION_COL]:.0f} mm",
                        f"{max_monthly_row[Config.STATION_NAME_COL]} ({meses_map.get(max_monthly_row[Config.MONTH_COL])} {max_monthly_row[Config.DATE_COL].year})"
                    )
            else:
                st.info("No hay datos anuales válidos para mostrar la síntesis.")
        else:
            st.info("No hay datos para mostrar la síntesis general.")

def display_correlation_tab(df_monthly_filtered, stations_for_analysis):
    st.header("Análisis de Correlación")
    st.markdown("Esta sección cuantifica la relación lineal entre la precipitación y diferentes variables (otras estaciones o índices climáticos) utilizando el coeficiente de correlación de Pearson.")
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    enso_corr_tab, station_corr_tab, indices_climaticos_tab = st.tabs(["Correlación con ENSO (ONI)", "Comparación entre Estaciones", "Correlación con Otros Índices"])
    
    with enso_corr_tab:
        if Config.ENSO_ONI_COL not in df_monthly_filtered.columns or df_monthly_filtered[Config.ENSO_ONI_COL].isnull().all():
            st.warning("No se puede realizar el análisis de correlación con ENSO. La columna 'anomalia_oni' no fue encontrada o no tiene datos en el período seleccionado.")
            return
        
        df_corr_analysis = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL, Config.ENSO_ONI_COL])
        if df_corr_analysis.empty:
            st.warning("No hay datos coincidentes entre la precipitación y el ENSO para la selección actual.")
            return

        analysis_level = st.radio("Nivel de Análisis de Correlación con ENSO", ["Promedio de la selección", "Por Estación Individual"], key="enso_corr_level")
        
        df_plot_corr = pd.DataFrame()
        title_text = ""
        if analysis_level == "Por Estación Individual":
            station_to_corr = st.selectbox("Seleccione Estación:", options=sorted(df_corr_analysis[Config.STATION_NAME_COL].unique()), key="enso_corr_station")
            if station_to_corr:
                df_plot_corr = df_corr_analysis[df_corr_analysis[Config.STATION_NAME_COL] == station_to_corr]
                title_text = f"Correlación para la estación: {station_to_corr}"
        else: # Promedio
            df_plot_corr = df_corr_analysis.groupby(Config.DATE_COL).agg(
                precipitation=(Config.PRECIPITATION_COL, 'mean'),
                anomalia_oni=(Config.ENSO_ONI_COL, 'first')
            ).reset_index()
            title_text = "Correlación para el promedio de las estaciones seleccionadas"

        if not df_plot_corr.empty and len(df_plot_corr) > 2:
            corr, p_value = stats.pearsonr(df_plot_corr['anomalia_oni'], df_plot_corr['precipitation'])
            st.subheader(title_text)
            col1, col2 = st.columns(2)
            col1.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")
            col2.metric("Significancia (valor p)", f"{p_value:.4f}")
            if p_value < 0.05:
                st.success("La correlación es estadísticamente significativa, lo que sugiere una relación lineal entre las variables.")
            else:
                st.warning("La correlación no es estadísticamente significativa. No hay evidencia de una relación lineal fuerte.")
            
            fig_corr = px.scatter(
                df_plot_corr, x='anomalia_oni', y='precipitation', trendline='ols',
                title="Gráfico de Dispersión: Precipitación vs. Anomalía ONI",
                labels={'anomalia_oni': 'Anomalía ONI (°C)', 'precipitation': 'Precipitación Mensual (mm)'}
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    with station_corr_tab:
        if len(stations_for_analysis) < 2:
            st.info("Seleccione al menos dos estaciones para comparar la correlación entre ellas.")
        else:
            st.subheader("Correlación de Precipitación entre dos Estaciones")
            station_options = sorted(stations_for_analysis)
            col1, col2 = st.columns(2)
            station1_name = col1.selectbox("Estación 1:", options=station_options, key="corr_station_1")
            station2_name = col2.selectbox("Estación 2:", options=station_options, index=1 if len(station_options)>1 else 0, key="corr_station_2")

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
                        st.success("La correlación es estadísticamente significativa (p < 0.05).")
                    else:
                        st.warning("La correlación no es estadísticamente significativa (p ≥ 0.05).")
                    
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
                index_col_name = selected_index.lower()
                df_merged_indices = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == selected_station_corr].copy()
                df_merged_indices.dropna(subset=[Config.PRECIPITATION_COL, index_col_name], inplace=True)
                
                if not df_merged_indices.empty and len(df_merged_indices) > 2:
                    corr, p_value = stats.pearsonr(df_merged_indices[index_col_name], df_merged_indices[Config.PRECIPITATION_COL])

                    st.markdown(f"#### Resultados de la correlación ({selected_index} vs. Precipitación de {selected_station_corr})")
                    st.metric("Coeficiente de Correlación (r)", f"{corr:.3f}")
                    if p_value < 0.05:
                        st.success("La correlación es estadísticamente significativa (p < 0.05).")
                    else:
                        st.warning("La correlación no es estadísticamente significativa (p ≥ 0.05).")

                    fig_scatter_indices = px.scatter(
                        df_merged_indices, x=index_col_name, y=Config.PRECIPITATION_COL, trendline='ols',
                        title=f'Dispersión: {selected_index} vs. Precipitación de {selected_station_corr}',
                        labels={index_col_name: 'Valor del Índice {selected_index}', Config.PRECIPITATION_COL: 'Precipitación Mensual (mm)'}
                    )
                    st.plotly_chart(fig_scatter_indices, use_container_width=True)
                else:
                    st.warning("No hay suficientes datos superpuestos entre la estación y el índice para calcular la correlación.")

def display_enso_tab(df_monthly_filtered, df_enso, gdf_filtered, stations_for_analysis):
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
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
                    df_enso_filtered = df_enso[(df_enso[Config.DATE_COL].dt.year >= st.session_state.year_range[0]) & (df_enso[Config.DATE_COL].dt.year <= st.session_state.year_range[1]) & (df_enso[Config.DATE_COL].dt.month.isin(st.session_state.meses_numeros))]
                    if not df_enso_filtered.empty and var_code in df_enso_filtered.columns and not df_enso_filtered[var_code].isnull().all():
                        fig_enso_series = px.line(df_enso_filtered, x=Config.DATE_COL, y=var_code, title=f"Serie de Tiempo para {var_name}")
                        st.plotly_chart(fig_enso_series, use_container_width=True)
                    else:
                        st.warning(f"No hay datos disponibles para '{var_code}' en el período seleccionado.")

    with enso_anim_tab:
        st.subheader("Explorador Mensual del Fenómeno ENSO")
        if st.session_state.gdf_stations.empty or Config.ENSO_ONI_COL not in df_enso.columns:
            st.warning("Datos insuficientes para generar esta visualización. Se requiere información de estaciones y la columna 'anomalia_oni'.")
            return
        
        controls_col, map_col = st.columns([1, 3])
        enso_anim_data = df_enso[[Config.DATE_COL, Config.ENSO_ONI_COL]].copy().dropna(subset=[Config.ENSO_ONI_COL])
        conditions = [enso_anim_data[Config.ENSO_ONI_COL] >= 0.5, enso_anim_data[Config.ENSO_ONI_COL] <= -0.5]
        phases = ['El Niño', 'La Niña']
        enso_anim_data['fase'] = np.select(conditions, phases, default='Neutral')
        enso_anim_data_filtered = enso_anim_data[(enso_anim_data[Config.DATE_COL].dt.year >= st.session_state.year_range[0]) & (enso_anim_data[Config.DATE_COL].dt.year <= st.session_state.year_range[1])]

        with controls_col:
            st.markdown("##### Controles de Mapa")
            selected_base_map_config, selected_overlays_config = display_map_controls(st, "enso_anim")
            st.markdown("##### Selección de Fecha")
            available_dates = sorted(enso_anim_data_filtered[Config.DATE_COL].unique())
            if available_dates:
                selected_date = st.select_slider("Seleccione una fecha (Año-Mes)", options=available_dates, format_func=lambda date: date.strftime('%Y-%m'))
                phase_info = enso_anim_data_filtered[enso_anim_data_filtered[Config.DATE_COL] == selected_date]
                if not phase_info.empty:
                    current_phase = phase_info['fase'].iloc[0]
                    current_oni = phase_info[Config.ENSO_ONI_COL].iloc[0]
                    st.metric(f"Fase ENSO en {selected_date.strftime('%Y-%m')}", current_phase, f"Anomalía ONI: {current_oni:.2f}°C")
                else:
                    st.warning("No hay datos de ENSO para el período seleccionado.")

        with map_col:
            if 'selected_date' in locals():
                m_enso = create_folium_map([4.57, -74.29], 5, selected_base_map_config, selected_overlays_config)
                phase_color_map = {'El Niño': 'red', 'La Niña': 'blue', 'Neutral': 'grey'}
                marker_color = phase_color_map.get(locals().get('current_phase', 'black'), 'black')
                for _, station in gdf_filtered.iterrows():
                    folium.Marker(
                        location=[station[Config.LATITUDE_COL], station[Config.LONGITUDE_COL]],
                        tooltip=f"{station[Config.STATION_NAME_COL]}<br>Fase: {locals().get('current_phase', 'N/A')}",
                        icon=folium.Icon(color=marker_color, icon='cloud')
                    ).add_to(m_enso)
                if not gdf_filtered.empty:
                    bounds = gdf_filtered.total_bounds
                    if np.all(np.isfinite(bounds)):
                        m_enso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                folium.LayerControl().add_to(m_enso)
                folium_static(m_enso, height=700, width="100%")

def display_trends_and_forecast_tab(df_anual_melted, df_monthly_to_process, stations_for_analysis):
    st.header("Análisis de Tendencias y Pronósticos")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")

    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
        
    tendencia_individual_tab, mann_kendall_tab, tendencia_tabla_tab, descomposicion_tab, autocorrelacion_tab, pronostico_sarima_tab, pronostico_prophet_tab, compare_forecast_tab = st.tabs([
        "Análisis Lineal", "Tendencia Mann-Kendall", "Tabla Comparativa", "Descomposición de Series", 
        "Autocorrelación (ACF/PACF)", "Pronóstico SARIMA", "Pronóstico Prophet", "SARIMA vs Prophet"
    ])

    # --- Lógica de Pronóstico SARIMA (Mejora 2.1) ---
    with pronostico_sarima_tab:
        st.subheader("Pronóstico de Precipitación Mensual (Modelo SARIMA)")
        with st.expander("Ajuste de Parámetros SARIMA y Descripción"):
            st.markdown("""
                El modelo **SARIMA** (Seasonal Auto-Regressive Integrated Moving Average) utiliza datos históricos para predecir valores futuros.
                Ajuste los parámetros para optimizar el pronóstico:
                - **(p, d, q):** Componentes no estacionales (AR, I, MA).
                - **(P, D, Q):** Componentes estacionales (AR, I, MA). s=12 (mensual).
            """)
            col_p, col_d, col_q = st.columns(3)
            p = col_p.slider("p (AR no estacional)", 0, 3, 1, key="sarima_p")
            d = col_d.slider("d (I no estacional)", 0, 2, 1, key="sarima_d")
            q = col_q.slider("q (MA no estacional)", 0, 3, 1, key="sarima_q")

            col_P, col_D, col_Q = st.columns(3)
            P = col_P.slider("P (AR estacional)", 0, 2, 1, key="sarima_P")
            D = col_D.slider("D (I estacional)", 0, 2, 1, key="sarima_D")
            Q = col_Q.slider("Q (MA estacional)", 0, 2, 1, key="sarima_Q")

        station_to_forecast = st.selectbox("Seleccione una estación para el pronóstico:", options=stations_for_analysis, key="sarima_station_select")
        forecast_horizon = st.slider("Meses a pronosticar:", 12, 36, 12, step=12, key="sarima_forecast_horizon_slider")

        sarima_order = (p, d, q)
        seasonal_order = (P, D, Q, 12)
        
        ts_data_sarima = df_monthly_to_process[df_monthly_to_process[Config.STATION_NAME_COL] == station_to_forecast]

        if ts_data_sarima.empty or len(ts_data_sarima) < 24:
            st.warning("Se necesitan al menos 24 meses de datos continuos para un pronóstico SARIMA confiable.")
        else:
            with st.spinner(f"Entrenando modelo y generando pronóstico para {station_to_forecast} con SARIMA{sarima_order}x{seasonal_order[:-1]}..."):
                try:
                    ts_data = ts_data_sarima[[Config.DATE_COL, Config.PRECIPITATION_COL]].copy()
                    ts_data = ts_data.set_index(Config.DATE_COL).sort_index()
                    ts_data = ts_data[Config.PRECIPITATION_COL].asfreq('MS')

                    model = sm.tsa.statespace.SARIMAX(
                        ts_data, 
                        order=sarima_order, 
                        seasonal_order=seasonal_order, 
                        enforce_stationarity=False, 
                        enforce_invertibility=False
                    )
                    results = model.fit(disp=False)
                    forecast = results.get_forecast(steps=forecast_horizon)
                    forecast_mean = forecast.predicted_mean
                    forecast_ci = forecast.conf_int()
                    
                    # Almacenar pronóstico SARIMA para comparación
                    st.session_state['sarima_forecast'] = forecast_mean.reset_index().rename(columns={'index': 'ds', 'predicted_mean': 'yhat'})
                    
                    fig_pronostico = go.Figure()
                    fig_pronostico.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Datos Históricos'))
                    fig_pronostico.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Pronóstico SARIMA', line=dict(color='red', dash='dash')))
                    fig_pronostico.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0], fill=None, mode='lines', line=dict(color='rgba(255,0,0,0.2)'), showlegend=False))
                    fig_pronostico.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1], fill='tonexty', mode='lines', line=dict(color='rgba(255,0,0,0.2)'), name='Intervalo de Confianza'))
                    fig_pronostico.update_layout(title=f"Pronóstico de Precipitación SARIMA {sarima_order}x{seasonal_order[:-1]} para {station_to_forecast}", xaxis_title="Fecha", yaxis_title="Precipitación (mm)")
                    st.plotly_chart(fig_pronostico, use_container_width=True)
                    st.info(f"El modelo SARIMA fue entrenado con la configuración: **Orden={sarima_order}**, **Estacional={seasonal_order}**.")
                    
                    forecast_df = pd.DataFrame({
                        'fecha': forecast_mean.index, 'pronostico': forecast_mean.values,
                        'limite_inferior': forecast_ci.iloc[:, 0].values, 'limite_superior': forecast_ci.iloc[:, 1].values
                    })
                    csv_data = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar Pronóstico SARIMA en CSV", data=csv_data,
                        file_name=f'pronostico_sarima_{station_to_forecast.replace(" ", "_")}.csv', mime='text/csv',
                        key='download-sarima'
                    )
                except Exception as e:
                    st.error(f"No se pudo generar el pronóstico SARIMA. El modelo no pudo converger. Error: {e}")

    # --- Lógica de Pronóstico Prophet ---
    with pronostico_prophet_tab:
        st.subheader("Pronóstico de Precipitación Mensual (Modelo Prophet)")
        
        station_to_forecast_prophet = st.selectbox("Seleccione una estación para el pronóstico:", options=stations_for_analysis, key="prophet_station_select", help="El pronóstico se realiza para una única serie de tiempo con Prophet.")
        forecast_horizon_prophet = st.slider("Meses a pronosticar:", 12, 36, 12, step=12, key="prophet_forecast_horizon_slider")
        
        ts_data_prophet_raw = df_monthly_to_process[df_monthly_to_process[Config.STATION_NAME_COL] == station_to_forecast_prophet]
        
        if ts_data_prophet_raw.empty or len(ts_data_prophet_raw) < 24:
            st.warning("Se necesitan al menos 24 meses de datos para que Prophet funcione correctamente.")
        else:
            with st.spinner(f"Entrenando modelo Prophet y generando pronóstico para {station_to_forecast_prophet}..."):
                try:
                    ts_data_prophet = ts_data_prophet_raw[[Config.DATE_COL, Config.PRECIPITATION_COL]].copy()
                    ts_data_prophet.rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'}, inplace=True)
                    model_prophet = Prophet()
                    model_prophet.fit(ts_data_prophet)
                    future = model_prophet.make_future_dataframe(periods=forecast_horizon_prophet, freq='MS')
                    forecast_prophet = model_prophet.predict(future)
                    
                    # Almacenar pronóstico Prophet para comparación
                    st.session_state['prophet_forecast'] = forecast_prophet[['ds', 'yhat']].copy()
                    
                    st.success("Pronóstico generado exitosamente.")
                    fig_prophet = plot_plotly(model_prophet, forecast_prophet)
                    fig_prophet.update_layout(title=f"Pronóstico de Precipitación con Prophet para {station_to_forecast_prophet}", yaxis_title="Precipitación (mm)")
                    st.plotly_chart(fig_prophet, use_container_width=True)

                    csv_data = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar Pronóstico Prophet en CSV", data=csv_data,
                        file_name=f'pronostico_prophet_{station_to_forecast_prophet.replace(" ", "_")}.csv', mime='text/csv',
                        key='download-prophet'
                    )
                except Exception as e:
                    st.error(f"Ocurrió un error al generar el pronóstico con Prophet. Esto puede deberse a que la serie de datos es demasiado corta o inestable. Error: {e}")

    # --- Lógica de Comparación de Pronósticos (Mejora 2.2) ---
    with compare_forecast_tab:
        st.subheader("Comparación de Pronósticos: SARIMA vs Prophet")
        
        if 'sarima_forecast' not in st.session_state or 'prophet_forecast' not in st.session_state:
            st.warning("Debe generar un pronóstico SARIMA y un pronóstico Prophet en las pestañas anteriores antes de poder compararlos.")
            st.stop()
            
        sarima_df = st.session_state['sarima_forecast'].copy()
        prophet_df = st.session_state['prophet_forecast'].copy()
        
        # Usar la estación seleccionada en Prophet para mostrar los datos históricos
        # Asumimos que el usuario generó los pronósticos para la misma estación (es la mejor práctica)
        station_id_for_history = st.selectbox("Seleccione la estación para la serie histórica (debe coincidir con la usada para los pronósticos):", options=stations_for_analysis, key="compare_station_history")
        ts_data = df_monthly_to_process[df_monthly_to_process[Config.STATION_NAME_COL] == station_id_for_history].copy()
        
        if ts_data.empty:
            st.warning("Datos históricos no encontrados para la comparación.")
            st.stop()

        # Asegurar que las fechas sean comparables
        sarima_df['ds'] = pd.to_datetime(sarima_df['ds'])
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        ts_data['ds'] = ts_data[Config.DATE_COL]
        
        # Combinar datos históricos y pronósticos (Solo los valores de predicción y la fecha)
        df_combined = pd.merge(sarima_df[['ds', 'yhat']], prophet_df[['ds', 'yhat']], on='ds', suffixes=('_sarima', '_prophet'), how='outer')
        df_combined = pd.merge(df_combined, ts_data[['ds', Config.PRECIPITATION_COL]], on='ds', how='left')
        
        # Crear el gráfico combinado
        fig_compare = go.Figure()
        
        # 1. Datos Históricos
        fig_compare.add_trace(go.Scatter(
            x=df_combined['ds'], y=df_combined[Config.PRECIPITATION_COL], 
            mode='lines+markers', name='Histórico', line=dict(color='gray', width=2)
        ))
        
        # 2. Pronóstico SARIMA
        fig_compare.add_trace(go.Scatter(
            x=df_combined['ds'], y=df_combined['yhat_sarima'], 
            mode='lines', name='Pronóstico SARIMA', line=dict(color='red', dash='dash', width=3)
        ))
        
        # 3. Pronóstico Prophet
        fig_compare.add_trace(go.Scatter(
            x=df_combined['ds'], y=df_combined['yhat_prophet'], 
            mode='lines', name='Pronóstico Prophet', line=dict(color='blue', dash='dash', width=3)
        ))

        fig_compare.update_layout(
            title=f"Pronóstico Comparativo SARIMA vs Prophet para {station_id_for_history}",
            xaxis_title="Fecha",
            yaxis_title="Precipitación (mm)",
            height=650,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    with tendencia_individual_tab:
        st.subheader("Tendencia de Precipitación Anual (Regresión Lineal)")
        analysis_type = st.radio("Tipo de Análisis de Tendencia:", ["Promedio de la selección", "Estación individual"], horizontal=True, key="linear_trend_type")
        df_to_analyze = None
        title_for_download = "promedio"
        if analysis_type == "Promedio de la selección":
            df_to_analyze = df_anual_melted.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        else:
            station_to_analyze = st.selectbox("Seleccione una estación para analizar:", options=stations_for_analysis, key="tendencia_station_select")
            if station_to_analyze: 
                df_to_analyze = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station_to_analyze]
                title_for_download = station_to_analyze.replace(" ","_")

        if df_to_analyze is not None and len(df_to_analyze.dropna(subset=[Config.PRECIPITATION_COL])) > 2:
            df_to_analyze['año_num'] = pd.to_numeric(df_to_analyze[Config.YEAR_COL])
            df_clean = df_to_analyze.dropna(subset=[Config.PRECIPITATION_COL])
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['año_num'], df_clean[Config.PRECIPITATION_COL])
            tendencia_texto = "aumentando" if slope > 0 else "disminuyendo"
            significancia_texto = "**estadísticamente significativa**" if p_value < 0.05 else "no es estadísticamente significativa"
            
            st.markdown(f"La tendencia de la precipitación es de **{slope:.2f} mm/año** (es decir, está {tendencia_texto}). Con un valor p de **{p_value:.3f}**, esta tendencia **{significancia_texto}**.")
            
            df_to_analyze['tendencia'] = slope * df_to_analyze['año_num'] + intercept
            fig_tendencia = px.scatter(df_to_analyze, x='año_num', y=Config.PRECIPITATION_COL, title=f'Tendencia de la Precipitación Anual ({st.session_state.year_range[0]} - {st.session_state.year_range[1]})')
            fig_tendencia.add_trace(go.Scatter(x=df_to_analyze['año_num'], y=df_to_analyze['tendencia'], mode='lines', name='Línea de Tendencia', line=dict(color='red')))
            fig_tendencia.update_layout(xaxis_title="Año", yaxis_title="Precipitación Anual (mm)")
            st.plotly_chart(fig_tendencia, use_container_width=True)
            
            csv_data = df_to_analyze.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar datos de Tendencia Anual", data=csv_data,
                file_name=f'tendencia_anual_{title_for_download}.csv', mime='text/csv',
                key='download-anual-tendencia'
            )
        else:
            st.warning("No hay suficientes datos en el período seleccionado para calcular una tendencia.")

    with mann_kendall_tab:
        st.subheader("Tendencia de Precipitación Anual (Prueba de Mann-Kendall)")
        with st.expander("¿Qué es la prueba de Mann-Kendall?"):
            st.markdown("""
            La **Prueba de Mann-Kendall** es un método estadístico no paramétrico utilizado para detectar tendencias en series de tiempo. A diferencia de la regresión lineal, no asume que los datos sigan una distribución particular.
            - **Objetivo**: Determinar si existe una tendencia monotónica (consistentemente creciente o decreciente) a lo largo del tiempo.
            - **Resultados Clave**:
                - **Tendencia**: Indica si es 'increasing' (creciente), 'decreasing' (decreciente) o 'no trend' (sin tendencia).
                - **Valor p**: Si es menor a 0.05, la tendencia se considera estadísticamente significativa.
                - **Pendiente de Sen (Sen's Slope)**: Es un método para cuantificar la magnitud de la tendencia, calculando la mediana de todas las pendientes entre pares de puntos. Es robusto frente a valores atípicos.
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
            
            st.markdown(f"#### Resultados para: {mk_analysis_type if mk_analysis_type == 'Promedio de la selección' else station_to_analyze_mk}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Tendencia Detectada", mk_result.trend.capitalize())
            col2.metric("Valor p", f"{mk_result.p:.4f}")
            col3.metric("Pendiente de Sen (mm/año)", f"{mk_result.slope:.2f}")

            if mk_result.p < 0.05:
                st.success("La tendencia es estadísticamente significativa (p < 0.05).")
            else:
                st.warning("La tendencia no es estadísticamente significativa (p ≥ 0.05).")

            # Visualización
            fig_mk = px.scatter(df_clean_mk, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL, title="Análisis de Tendencia con Pendiente de Sen")
            
            median_x = df_clean_mk[Config.YEAR_COL].median()
            median_y = df_clean_mk[Config.PRECIPITATION_COL].median()
            intercept_sen = median_y - mk_result.slope * median_x
            
            x_vals = np.array(df_clean_mk[Config.YEAR_COL])
            y_vals = mk_result.slope * x_vals + intercept_sen
            
            fig_mk.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name="Pendiente de Sen", line=dict(color='orange')))
            fig_mk.update_layout(xaxis_title="Año", yaxis_title="Precipitación Anual (mm)")
            st.plotly_chart(fig_mk, use_container_width=True)

        else:
            st.warning("No hay suficientes datos (se requieren al menos 4 puntos) para calcular la tendencia de Mann-Kendall.")

    with tendencia_tabla_tab:
        st.subheader("Tabla Comparativa de Tendencias de Precipitación Anual")
        st.info("Esta tabla resume los resultados de dos métodos de análisis de tendencia. Presione el botón para calcular los valores para todas las estaciones seleccionadas.")

        if st.button("Calcular Tendencias para Todas las Estaciones Seleccionadas"):
            with st.spinner("Calculando tendencias..."):
                results = []
                df_anual_calc = df_anual_melted.copy()
                if st.session_state.exclude_zeros:
                    df_anual_calc = df_anual_calc[df_anual_calc[Config.PRECIPITATION_COL] > 0]
                
                for station in stations_for_analysis:
                    station_data = df_anual_calc[df_anual_calc[Config.STATION_NAME_COL] == station].dropna(subset=[Config.PRECIPITATION_COL]).sort_values(by=Config.YEAR_COL)
                    
                    # Inicializar valores por defecto
                    slope_lin, p_lin = np.nan, np.nan
                    trend_mk, p_mk, slope_sen = "Datos insuficientes", np.nan, np.nan
                    
                    # Cálculo de Regresión Lineal
                    if len(station_data) > 2:
                        station_data['año_num'] = pd.to_numeric(station_data[Config.YEAR_COL])
                        slope_lin, _, _, p_lin, _ = stats.linregress(station_data['año_num'], station_data[Config.PRECIPITATION_COL])
                    
                    # Cálculo de Mann-Kendall
                    if len(station_data) > 3:
                        mk_result = mk.original_test(station_data[Config.PRECIPITATION_COL])
                        trend_mk = mk_result.trend.capitalize()
                        p_mk = mk.result.p
                        slope_sen = mk_result.slope

                    results.append({
                        "Estación": station,
                        "Años Analizados": len(station_data),
                        "Tendencia Lineal (mm/año)": slope_lin,
                        "Valor p (Lineal)": p_lin,
                        "Tendencia MK": trend_mk,
                        "Valor p (MK)": p_mk,
                        "Pendiente de Sen (mm/año)": slope_sen,
                    })

                if results:
                    results_df = pd.DataFrame(results)
                    
                    def style_p_value(val):
                        if pd.isna(val): return ''
                        color = 'lightgreen' if val < 0.05 else 'lightcoral'
                        return f'background-color: {color}'
                    
                    st.dataframe(results_df.style.format({
                        "Tendencia Lineal (mm/año)": "{:.2f}",
                        "Valor p (Lineal)": "{:.4f}",
                        "Valor p (MK)": "{:.4f}",
                        "Pendiente de Sen (mm/año)": "{:.2f}",
                    }).applymap(style_p_value, subset=['Valor p (Lineal)', 'Valor p (MK)']), use_container_width=True)
                    
                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar tabla de tendencias en CSV", data=csv_data,
                        file_name='tabla_tendencias_comparativa.csv', mime='text/csv', key='download-tabla-tendencias'
                    )
                else:
                    st.warning("No se pudieron calcular tendencias para las estaciones seleccionadas.")
    
    with descomposicion_tab:
        st.subheader("Descomposición de Series de Tiempo Mensual")
        st.markdown("""
        La **descomposición de una serie de tiempo** separa sus componentes principales:
        - **Tendencia**: Muestra la dirección a largo plazo de los datos (ascendente o descendente).
        - **Estacionalidad**: Revela patrones que se repiten a intervalos regulares (por ejemplo, anualmente).
        - **Residuo**: Representa la variabilidad restante después de eliminar la tendencia y la estacionalidad.
        """)
        
        station_to_decompose = st.selectbox("Seleccione una estación para la descomposición:", options=stations_for_analysis, key="decompose_station_select")
        
        if station_to_decompose:
            df_station = df_monthly_to_process[df_monthly_to_process[Config.STATION_NAME_COL] == station_to_decompose].copy()
            if not df_station.empty:
                df_station.set_index(Config.DATE_COL, inplace=True)
                df_station = df_station.asfreq('MS')

                if df_station[Config.PRECIPITATION_COL].isnull().values.any():
                    st.info("La serie tiene datos faltantes. Se rellenarán con interpolación lineal para la descomposición.")
                    df_station[Config.PRECIPITATION_COL] = df_station[Config.PRECIPITATION_COL].interpolate(method='time')
                
                try:
                    result = seasonal_decompose(df_station[Config.PRECIPITATION_COL].dropna(), model='additive', period=12)
                    
                    fig_decomp = go.Figure()
                    
                    fig_decomp.add_trace(go.Scatter(x=df_station.index, y=df_station[Config.PRECIPITATION_COL], mode='lines', name='Original'))
                    fig_decomp.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Tendencia'))
                    fig_decomp.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Estacionalidad'))
                    fig_decomp.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residuo'))
                    
                    fig_decomp.update_layout(title=f"Descomposición de la Serie de Precipitación para {station_to_decompose}",
                                             height=600,
                                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    
                    st.plotly_chart(fig_decomp, use_container_width=True)

                except Exception as e:
                    st.error(f"No se pudo realizar la descomposición de la serie para la estación seleccionada. Es posible que la serie de datos sea demasiado corta o no tenga una estructura estacional clara. Error: {e}")
            else:
                st.warning(f"No hay datos mensuales para la estación {station_to_decompose} en el período seleccionado.")

    with autocorrelacion_tab:
        st.subheader("Análisis de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)")
        st.markdown("Estos gráficos ayudan a entender la dependencia de la precipitación con sus valores pasados (rezagos). Las barras que superan el área azul sombreada indican una correlación estadísticamente significativa.")
        
        station_to_analyze_acf = st.selectbox("Seleccione una estación:", options=stations_for_analysis, key="acf_station_select")
        max_lag = st.slider("Número máximo de rezagos (meses):", min_value=12, max_value=60, value=24, step=12)
        
        if station_to_analyze_acf:
            df_station_acf = df_monthly_to_process[df_monthly_to_process[Config.STATION_NAME_COL] == station_to_analyze_acf].copy()
            if not df_station_acf.empty:
                df_station_acf.set_index(Config.DATE_COL, inplace=True)
                df_station_acf = df_station_acf.asfreq('MS')
                df_station_acf[Config.PRECIPITATION_COL] = df_station_acf[Config.PRECIPITATION_COL].interpolate(method='time').dropna()

                if len(df_station_acf) > max_lag:
                    try:
                        acf_values = sm.tsa.acf(df_station_acf[Config.PRECIPITATION_COL], nlags=max_lag)
                        lags = list(range(max_lag + 1))
                        
                        conf_interval = 1.96 / np.sqrt(len(df_station_acf))
                        
                        fig_acf = go.Figure(data=[
                            go.Bar(x=lags, y=acf_values, name='ACF'),
                            go.Scatter(x=lags, y=[conf_interval] * (max_lag + 1), mode='lines', line=dict(color='blue', dash='dash'), name='Límite de Confianza Superior'),
                            go.Scatter(x=lags, y=[-conf_interval] * (max_lag + 1), mode='lines', line=dict(color='blue', dash='dash'), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', name='Límite de Confianza Inferior')
                        ])
                        fig_acf.update_layout(title='Función de Autocorrelación (ACF)', xaxis_title='Rezagos (Meses)', yaxis_title='Correlación', height=400)
                        st.plotly_chart(fig_acf, use_container_width=True)

                        pacf_values = pacf(df_station_acf[Config.PRECIPITATION_COL], nlags=max_lag)
                        
                        fig_pacf = go.Figure(data=[
                            go.Bar(x=lags, y=pacf_values, name='PACF'),
                            go.Scatter(x=lags, y=[conf_interval] * (max_lag + 1), mode='lines', line=dict(color='red', dash='dash'), name='Límite de Confianza Superior'),
                            go.Scatter(x=lags, y=[-conf_interval] * (max_lag + 1), mode='lines', line=dict(color='red', dash='dash'), fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name='Límite de Confianza Inferior')
                        ])
                        fig_pacf.update_layout(title='Función de Autocorrelación Parcial (PACF)', xaxis_title='Rezagos (Meses)', yaxis_title='Correlación', height=400)
                        st.plotly_chart(fig_pacf, use_container_width=True)

                    except Exception as e:
                        st.error(f"No se pudieron generar los gráficos de autocorrelación. Error: {e}")
                else:
                    st.warning(f"No hay suficientes datos (se requieren > {max_lag} meses) para la estación {station_to_analyze_acf} para realizar el análisis de autocorrelación.")
            else:
                st.warning(f"No hay datos para la estación {station_to_analyze_acf} en el período seleccionado.")
    
    # Lógica de Pronóstico SARIMA ya implementada arriba
    
def display_downloads_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis):
    st.header("Opciones de Descarga")
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para activar las descargas.")
        return
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    st.markdown("**Datos de Precipitación Anual (Filtrados)**")
    csv_anual = convert_df_to_csv(df_anual_melted)
    st.download_button("Descargar CSV Anual", csv_anual, 'precipitacion_anual.csv', 'text/csv', key='download-anual')

    st.markdown("**Datos de Precipitación Mensual (Filtrados)**")
    csv_mensual = convert_df_to_csv(df_monthly_filtered)
    st.download_button("Descargar CSV Mensual", csv_mensual, 'precipitacion_mensual.csv', 'text/csv', key='download-mensual')

    if st.session_state.analysis_mode == "Completar series (interpolación)":
        st.markdown("**Datos de Precipitación Mensual (Series Completadas y Filtradas)**")
        
        df_completed_to_download = st.session_state.df_monthly_processed[
            (st.session_state.df_monthly_processed[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
            (st.session_state.df_monthly_processed[Config.DATE_COL].dt.year >= st.session_state.year_range[0]) &
            (st.session_state.df_monthly_processed[Config.DATE_COL].dt.year <= st.session_state.year_range[1]) &
            (st.session_state.df_monthly_processed[Config.DATE_COL].dt.month.isin(st.session_state.meses_numeros))
        ].copy()
        
        # Aplicar los filtros de exclusión (NaN/Ceros) al DataFrame de descarga
        if st.session_state.exclude_na:
            df_completed_to_download.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
        if st.session_state.exclude_zeros:
            df_completed_to_download = df_completed_to_download[df_completed_to_download[Config.PRECIPITATION_COL] > 0]
        
        csv_completado = convert_df_to_csv(df_completed_to_download)
        st.download_button("📥 Descargar CSV con Series Completadas", csv_completado, 'precipitacion_mensual_completada.csv', 'text/csv', key='download-completado')
    else:
        st.info("Para descargar las series completadas, seleccione la opción **Completar series (interpolación)** en el panel lateral.")

def display_station_table_tab(gdf_filtered, df_anual_melted, stations_for_analysis):
    st.header("Información Detallada de las Estaciones")
    selected_stations_str = f"{len(stations_for_analysis)} estaciones" if len(stations_for_analysis) > 1 else f"1 estación: {stations_for_analysis[0]}"
    st.info(f"Mostrando análisis para {selected_stations_str} en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    if len(stations_for_analysis) == 0:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return
    if not df_anual_melted.empty:
        df_info_table = gdf_filtered[[Config.STATION_NAME_COL, Config.ALTITUDE_COL, Config.MUNICIPALITY_COL, Config.REGION_COL, Config.PERCENTAGE_COL]].copy()
        df_mean_precip = df_anual_melted.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().round(0).reset_index()
        df_mean_precip.rename(columns={Config.PRECIPITATION_COL: 'Precipitación media anual (mm)'}, inplace=True)
        df_info_table = df_info_table.merge(df_mean_precip, on=Config.STATION_NAME_COL, how='left')
        st.dataframe(df_info_table)
    else:
        st.info("No hay datos de precipitación anual (con >= 10 meses) para mostrar en la selección actual.")
