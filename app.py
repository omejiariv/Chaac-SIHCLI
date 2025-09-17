# -*- coding: utf-8 -*-
# Importaciones
# ---
import streamlit as st
import pandas as pd
import altair as alt
import warnings 
from pykrige.ok import OrdinaryKriging
from scipy import stats
from scipy.stats import gamma, norm
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf
from prophet import Prophet
from prophet.plot import plot_plotly
import base64
import pymannkendall as mk
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# Importaciones de módulos locales
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series
from modules.visualizer import create_enso_chart, create_anomaly_chart, add_plotly_download_buttons, add_folium_download_button, display_map_controls, create_folium_map
from folium.plugins import MarkerCluster, MiniMap
from folium.raster_layers import WmsTileLayer

# Desactivar UserWarning de scipy/statsmodels que son
# comunes durante el fitting de distribución
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---
# Funciones para las Pestañas de la UI
# ---
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
                    if st.session_state.selected_altitudes: summary_text += f"**Altitud:** {', '.join(st.session_state.selected_altitudes)}\n\n"
                    if st.session_state.selected_regions: summary_text += f"**Región:** {', '.join(st.session_state.selected_regions)}\n\n"
                    if st.session_state.selected_municipios: summary_text += f"**Municipio:** {', '.join(st.session_state.selected_municipios)}\n\n"
                    if st.session_state.selected_celdas: summary_text += f"**Celda XY:** {', '.join(st.session_state.selected_celdas)}\n\n"
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
                    sort_order_dis = st.radio("Ordenar por:", ["% Datos Disponibles (Mayor a Menor)", "% Datos Disponibles (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_dis")
                    if "Mayor a Menor" in sort_order_dis: gdf_filtered = gdf_filtered.sort_values(Config.PERCENTAGE_COL, ascending=False)
                    elif "Menor a Mayor" in sort_order_dis: gdf_filtered = gdf_filtered.sort_values(Config.PERCENTAGE_COL, ascending=True)
                    else: gdf_filtered = gdf_filtered.sort_values(Config.STATION_NAME_COL, ascending=True)
                    
                    fig_avail = px.bar(gdf_filtered, x=Config.STATION_NAME_COL, y=Config.PERCENTAGE_COL,
                                       title="Porcentaje de Disponibilidad de Datos por Estación",
                                       labels={Config.STATION_NAME_COL: "Estación", Config.PERCENTAGE_COL: "% de Datos"},
                                       text_auto='.1f')
                    fig_avail.update_layout(height=600, xaxis={'categoryorder': 'trace'})
                    st.plotly_chart(fig_avail, use_container_width=True)
            else:
                st.info("Mostrando el porcentaje de disponibilidad de datos según el archivo de estaciones.")
                sort_order_dis = st.radio("Ordenar por:", ["% Datos Disponibles (Mayor a Menor)", "% Datos Disponibles (Menor a Mayor)", "Alfabético"], horizontal=True, key="sort_dis")
                if "Mayor a Menor" in sort_order_dis: gdf_filtered = gdf_filtered.sort_values(Config.PERCENTAGE_COL, ascending=False)
                elif "Menor a Mayor" in sort_order_dis: gdf_filtered = gdf_filtered.sort_values(Config.PERCENTAGE_COL, ascending=True)
                else: gdf_filtered = gdf_filtered.sort_values(Config.STATION_NAME_COL, ascending=True)
                
                fig_avail = px.bar(gdf_filtered, x=Config.STATION_NAME_COL, y=Config.PERCENTAGE_COL,
                                   title="Porcentaje de Disponibilidad de Datos por Estación",
                                   labels={Config.STATION_NAME_COL: "Estación", Config.PERCENTAGE_COL: "% de Datos"},
                                   text_auto='.1f')
                fig_avail.update_layout(height=600, xaxis={'categoryorder': 'trace'})
                st.plotly_chart(fig_avail, use_container_width=True)
        else:
            st.info("No hay estaciones seleccionadas para mostrar el gráfico de disponibilidad.")

def display_analysis_tab(df_monthly_filtered, stations_for_analysis):
    st.header("Análisis de Precipitación Mensual y Anual")
    st.info(f"Mostrando análisis para {len(stations_for_analysis)} estaciones en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")

    if df_monthly_filtered.empty or len(stations_for_analysis) == 0:
        st.warning("No hay datos para mostrar con los filtros actuales. Por favor, ajuste los filtros.")
        return

    # Tabs for Annual and Monthly analysis
    tab_anual, tab_mensual, tab_comparativa, tab_distribucion = st.tabs(["Precipitación Anual", "Precipitación Mensual", "Comparativa ENSO", "Distribuciones"])

    with tab_anual:
        st.subheader("Serie de Tiempo de Precipitación Anual por Estación")
        df_anual = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()
        df_anual.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
        
        station_filter = st.selectbox("Seleccionar Estación para Gráfico Anual", stations_for_analysis, key="anual_station")
        df_plot = df_anual[df_anual[Config.STATION_NAME_COL] == station_filter]
        
        fig = px.line(df_plot, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL,
                      title=f"Precipitación Anual en la Estación: {station_filter}",
                      labels={Config.YEAR_COL: 'Año', Config.PRECIPITATION_COL: 'Precipitación Anual (mm)'})
        fig.update_traces(mode='markers+lines')
        st.plotly_chart(fig, use_container_width=True)
        add_plotly_download_buttons(fig, f"precipitacion_anual_{station_filter}")
        
    with tab_mensual:
        st.subheader("Patrón Climático Mensual")
        df_monthly_avg = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean().reset_index()
        station_filter = st.selectbox("Seleccionar Estación para Gráfico Mensual", stations_for_analysis, key="mensual_station")
        df_plot = df_monthly_avg[df_monthly_avg[Config.STATION_NAME_COL] == station_filter]
        fig = px.line(df_plot, x=Config.MONTH_COL, y=Config.PRECIPITATION_COL,
                      title=f"Precipitación Media Mensual en la Estación: {station_filter}",
                      labels={Config.MONTH_COL: 'Mes', Config.PRECIPITATION_COL: 'Precipitación Media Mensual (mm)'})
        fig.update_xaxes(tickvals=list(range(1, 13)))
        st.plotly_chart(fig, use_container_width=True)
        add_plotly_download_buttons(fig, f"precipitacion_mensual_{station_filter}")

    with tab_comparativa:
        st.subheader("Comparación de Precipitación con la Anomalía ONI")
        
        df_enso = st.session_state.df_enso
        if df_enso is None or df_enso.empty:
            st.warning("No se cargaron datos de anomalía ONI. Por favor, asegúrese de que el archivo de precipitación contenga la columna 'anomalia_oni'.")
            return
            
        fig_enso = create_enso_chart(df_enso)
        st.plotly_chart(fig_enso, use_container_width=True)
        add_plotly_download_buttons(fig_enso, "enso_chart")

        st.subheader("Anomalías de Precipitación Mensual por Estación")
        station_filter = st.selectbox("Seleccionar Estación para Anomalías", stations_for_analysis, key="anom_station")
        
        df_plot = df_monthly_filtered.copy()
        df_plot[Config.DATE_COL] = pd.to_datetime(df_plot[Config.DATE_COL])
        df_plot['month'] = df_plot[Config.DATE_COL].dt.month
        df_plot['year'] = df_plot[Config.DATE_COL].dt.year
        
        df_monthly_avg_global = df_plot.groupby(['month'])[Config.PRECIPITATION_COL].mean().reset_index()
        df_monthly_avg_global.rename(columns={Config.PRECIPITATION_COL: 'promedio_historico'}, inplace=True)
        
        df_plot = pd.merge(df_plot, df_monthly_avg_global, on='month', how='left')
        df_plot['anomalia'] = df_plot[Config.PRECIPITATION_COL] - df_plot['promedio_historico']
        
        df_plot = df_plot[df_plot[Config.STATION_NAME_COL] == station_filter].copy()
        
        df_plot = pd.merge(df_plot, df_enso[[Config.DATE_COL, Config.ENSO_ONI_COL]], on=Config.DATE_COL, how='left')
        
        fig_anom = create_anomaly_chart(df_plot)
        st.plotly_chart(fig_anom, use_container_width=True)
        add_plotly_download_buttons(fig_anom, f"anomalias_{station_filter}")

    with tab_distribucion:
        st.subheader("Distribución de la Precipitación Mensual")
        selected_distribution = st.selectbox("Seleccionar Tipo de Distribución", ["Gamma", "Normal"], key="dist_type")
        station_filter = st.selectbox("Seleccionar Estación para Distribución", stations_for_analysis, key="dist_station")
        
        df_plot = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station_filter].copy()
        data = df_plot[Config.PRECIPITATION_COL].dropna()
        if data.empty:
            st.warning("No hay datos de precipitación para esta estación. Seleccione otra o ajuste los filtros.")
            return

        fig = go.Figure()
        
        # Histograma de los datos
        fig.add_trace(go.Histogram(x=data, histnorm='density', name='Histograma de Datos',
                                   marker_color='#1f77b4'))
        
        x_plot = np.linspace(data.min(), data.max(), 100)
        
        # Ajuste de la distribución
        try:
            if selected_distribution == "Gamma":
                params = gamma.fit(data, floc=0)
                pdf = gamma.pdf(x_plot, *params)
                fig.add_trace(go.Scatter(x=x_plot, y=pdf, mode='lines', name='Distribución Gamma Ajustada',
                                         line=dict(color='red', width=2)))
                st.write(f"Parámetros de la distribución Gamma: a={params[0]:.2f}, loc={params[1]:.2f}, scale={params[2]:.2f}")
            elif selected_distribution == "Normal":
                mu, sigma = norm.fit(data)
                pdf = norm.pdf(x_plot, mu, sigma)
                fig.add_trace(go.Scatter(x=x_plot, y=pdf, mode='lines', name='Distribución Normal Ajustada',
                                         line=dict(color='red', width=2)))
                st.write(f"Parámetros de la distribución Normal: media={mu:.2f}, desviación estándar={sigma:.2f}")
        except RuntimeError:
            st.error("No se pudo ajustar la distribución. Intente con otra estación o un período diferente.")

        fig.update_layout(title=f'Distribución de Precipitación en {station_filter}',
                          xaxis_title='Precipitación (mm)',
                          yaxis_title='Densidad')
        st.plotly_chart(fig, use_container_width=True)
        add_plotly_download_buttons(fig, f"distribucion_{station_filter}_{selected_distribution}")

def display_advanced_maps_tab(gdf_filtered, df_monthly_filtered, stations_for_analysis):
    st.header("Mapas Avanzados de Precipitación")
    st.info(f"Mostrando análisis para {len(stations_for_analysis)} estaciones en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    if gdf_filtered.empty or df_monthly_filtered.empty or len(stations_for_analysis) < 4:
        st.warning("Se requieren al menos 4 estaciones para el análisis de interpolación espacial.")
        return

    tab_kriging, tab_animado = st.tabs(["Mapa de Interpolación (Kriging)", "Mapa Animado"])

    with tab_kriging:
        st.subheader("Mapa de Precipitación Interpolada por Kriging")
        
        if not all(col in gdf_filtered.columns for col in [Config.LATITUDE_COL, Config.LONGITUDE_COL]):
            st.error("Los datos de las estaciones deben tener las columnas de latitud y longitud.")
            return

        gdf_kriging = gdf_filtered.copy()
        gdf_kriging.set_index(Config.STATION_NAME_COL, inplace=True)
        
        # Agrupar los datos mensuales para obtener un promedio anual
        df_monthly_avg = df_monthly_filtered.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index()
        
        gdf_kriging = gdf_kriging.merge(df_monthly_avg, on=Config.STATION_NAME_COL)
        
        gdf_kriging = gdf_kriging.dropna(subset=[Config.LATITUDE_COL, Config.LONGITUDE_COL, Config.PRECIPITATION_COL])
        
        if len(gdf_kriging) < 4:
            st.warning("Se requieren al menos 4 estaciones con datos válidos para realizar la interpolación.")
            return

        lons = gdf_kriging[Config.LONGITUDE_COL].values
        lats = gdf_kriging[Config.LATITUDE_COL].values
        vals = gdf_kriging[Config.PRECIPITATION_COL].values
        
        try:
            # Crear la cuadrícula para la interpolación
            lon_min, lon_max = lons.min() - 0.5, lons.max() + 0.5
            lat_min, lat_max = lats.min() - 0.5, lats.max() + 0.5
            
            grid_lon = np.arange(lon_min, lon_max, 0.05)
            grid_lat = np.arange(lat_min, lat_max, 0.05)

            # Realizar Kriging Ordinario
            OK = OrdinaryKriging(
                lons, lats, vals,
                variogram_model='spherical',
                verbose=False,
                enable_plotting=False
            )
            z, ss = OK.execute('grid', grid_lon, grid_lat)
            
            # Crear el mapa de Folium
            map_krig = create_folium_map(
                location=[lats.mean(), lons.mean()],
                zoom=8,
                base_map_config=display_map_controls(st, "kriging")[0],
                overlays_config=display_map_controls(st, "kriging")[1],
                fit_bounds_data=gdf_kriging
            )

            # Convertir los datos interpolados a un formato de imagen para la superposición
            colored_img = cm.linear.PuBuGn.scale(z.min(), z.max()).to_image(z)
            
            # Crear la superposición
            folium.raster_layers.ImageOverlay(
                name='Precipitación Interpolada (Kriging)',
                image=colored_img.to_url(),
                bounds=[[grid_lat.min(), grid_lon.min()], [grid_lat.max(), grid_lon.max()]],
                opacity=0.6,
                interactive=True,
                cross_origin=False,
                zindex=1
            ).add_to(map_krig)

            # Añadir una leyenda de color
            legend = cm.linear.PuBuGn.scale(z.min(), z.max()).to_step(n=10)
            legend.caption = 'Precipitación Media Anual (mm)'
            map_krig.add_child(legend)

            folium_static(map_krig, width="100%", height=700)
        
        except Exception as e:
            st.error(f"Error al realizar la interpolación por Kriging: {e}")
            st.info("Asegúrese de que los datos tengan suficiente variación espacial para el análisis.")
    
    with tab_animado:
        st.subheader("Mapa Animado de Precipitación Mensual")
        
        df_monthly_filtered['fecha_str'] = df_monthly_filtered[Config.DATE_COL].dt.strftime('%Y-%m')
        
        if df_monthly_filtered.empty:
            st.warning("No hay datos mensuales para crear el mapa animado.")
            return

        fig_anim = px.scatter_geo(
            df_monthly_filtered,
            lat=Config.LATITUDE_COL,
            lon=Config.LONGITUDE_COL,
            color=Config.PRECIPITATION_COL,
            animation_frame='fecha_str',
            size=Config.PRECIPITATION_COL,
            hover_name=Config.STATION_NAME_COL,
            projection="natural earth",
            title="Precipitación Mensual Animada",
            labels={Config.PRECIPITATION_COL: "Precipitación (mm)"},
            template="plotly_white",
            height=700
        )
        
        fig_anim.update_layout(
            geo=dict(
                scope='south america',
                showland=True,
                landcolor="rgb(243, 243, 243)",
                countrycolor="rgb(204, 204, 204)",
                subunitcolor="rgb(204, 204, 204)",
                lataxis_range=[min(df_monthly_filtered[Config.LATITUDE_COL]) - 1, max(df_monthly_filtered[Config.LATITUDE_COL]) + 1],
                lonaxis_range=[min(df_monthly_filtered[Config.LONGITUDE_COL]) - 1, max(df_monthly_filtered[Config.LONGITUDE_COL]) + 1],
            )
        )
        st.plotly_chart(fig_anim, use_container_width=True)
        add_plotly_download_buttons(fig_anim, "mapa_animado_precipitacion")

def display_trends_forecast_tab(df_monthly_filtered, stations_for_analysis):
    st.header("Análisis de Tendencias y Pronósticos")
    st.info(f"Mostrando análisis para {len(stations_for_analysis)} estaciones en el período {st.session_state.year_range[0]} - {st.session_state.year_range[1]}.")
    
    if df_monthly_filtered.empty or len(stations_for_analysis) == 0:
        st.warning("No hay datos para mostrar con los filtros actuales. Por favor, ajuste los filtros.")
        return

    station_filter = st.selectbox("Seleccionar Estación para el Análisis de Tendencias y Pronósticos", stations_for_analysis, key="trends_station")
    df_station = df_monthly_filtered[df_monthly_filtered[Config.STATION_NAME_COL] == station_filter].copy()
    
    tab_tendencias, tab_prophet, tab_sarima = st.tabs(["Tendencias (Mann-Kendall)", "Pronóstico (Prophet)", "Pronóstico (SARIMA)"])

    with tab_tendencias:
        st.subheader("Análisis de Tendencia de Precipitación (Mann-Kendall)")
        
        df_anual_station = df_station.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].sum().reset_index()
        
        if df_anual_station.empty or len(df_anual_station) < 5:
            st.warning("Se requieren al menos 5 años de datos para realizar el análisis de tendencia.")
        else:
            trend_data = df_anual_station[Config.PRECIPITATION_COL].values
            result = mk.original_test(trend_data)
            
            if result.p > 0.05:
                st.info(f"El análisis de Mann-Kendall no detectó una tendencia significativa en la precipitación anual (p-value = {result.p:.2f}).")
            else:
                trend_direction = "ascendente" if result.slope > 0 else "descendente"
                st.success(f"El análisis de Mann-Kendall detectó una tendencia **{trend_direction}** significativa en la precipitación anual (p-value = {result.p:.2f}).")
            
            fig = px.line(df_anual_station, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL,
                          title=f"Serie Anual de Precipitación para {station_filter}",
                          labels={Config.YEAR_COL: 'Año', Config.PRECIPITATION_COL: 'Precipitación Anual (mm)'})
            fig.update_traces(mode='markers+lines')
            st.plotly_chart(fig, use_container_width=True)
            add_plotly_download_buttons(fig, f"tendencia_anual_{station_filter}")

    with tab_prophet:
        st.subheader("Pronóstico de Precipitación con Prophet")
        
        prophet_data = df_station.copy()
        prophet_data = prophet_data.dropna(subset=[Config.PRECIPITATION_COL])
        prophet_data.rename(columns={Config.DATE_COL: 'ds', Config.PRECIPITATION_COL: 'y'}, inplace=True)
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        if len(prophet_data) < 24:
            st.warning("Se necesitan al menos 24 meses de datos para un pronóstico con Prophet.")
            return

        with st.spinner("Entrenando el modelo Prophet..."):
            m = Prophet()
            m.fit(prophet_data)
            
        future_periods = st.slider("Meses para pronosticar", min_value=1, max_value=36, value=12)
        future = m.make_future_dataframe(periods=future_periods, freq='MS')
        forecast = m.predict(future)
        
        st.session_state.prophet_forecast = forecast
        
        fig = plot_plotly(m, forecast)
        fig.update_layout(title=f'Pronóstico de Precipitación para {station_filter} con Prophet',
                          xaxis_title='Fecha', yaxis_title='Precipitación (mm)')
        
        st.plotly_chart(fig, use_container_width=True)
        add_plotly_download_buttons(fig, f"prophet_forecast_{station_filter}")
        
    with tab_sarima:
        st.subheader("Pronóstico de Precipitación con SARIMA")
        
        sarima_data = df_station.copy()
        sarima_data = sarima_data.dropna(subset=[Config.PRECIPITATION_COL])
        sarima_data.set_index(Config.DATE_COL, inplace=True)
        
        if len(sarima_data) < 24:
            st.warning("Se necesitan al menos 24 meses de datos para un pronóstico con SARIMA.")
            return

        with st.spinner("Entrenando el modelo SARIMA..."):
            try:
                # Descomposición estacional
                result_decomp = seasonal_decompose(sarima_data[Config.PRECIPITATION_COL], model='additive', period=12)
                
                # Gráfico de la descomposición
                fig_decomp = go.Figure()
                fig_decomp.add_trace(go.Scatter(x=result_decomp.observed.index, y=result_decomp.observed.values, mode='lines', name='Observado'))
                fig_decomp.add_trace(go.Scatter(x=result_decomp.trend.index, y=result_decomp.trend.values, mode='lines', name='Tendencia'))
                fig_decomp.add_trace(go.Scatter(x=result_decomp.seasonal.index, y=result_decomp.seasonal.values, mode='lines', name='Estacional'))
                fig_decomp.add_trace(go.Scatter(x=result_decomp.resid.index, y=result_decomp.resid.values, mode='lines', name='Residuo'))
                fig_decomp.update_layout(title=f'Descomposición de la Serie de Tiempo para {station_filter}',
                                         xaxis_title='Fecha', yaxis_title='Precipitación (mm)')
                st.plotly_chart(fig_decomp, use_container_width=True)
                add_plotly_download_buttons(fig_decomp, f"sarima_decomp_{station_filter}")
                
                # Autocorrelación Parcial para identificar 'p'
                pacf_vals = pacf(sarima_data[Config.PRECIPITATION_COL], nlags=20, method='ywm')
                
                # Usar valores de la autocorrelación parcial para una sugerencia
                # La lógica para elegir los parámetros SARIMA es compleja,
                # aquí solo se muestra un ejemplo básico
                
                # Modelo SARIMA
                # Intentar un modelo SARIMA(p, d, q)(P, D, Q)s
                # Parámetros sugeridos por la descomposición
                p, d, q = 1, 1, 1
                P, D, Q = 1, 1, 1
                s = 12
                
                model = sm.tsa.statespace.SARIMAX(sarima_data[Config.PRECIPITATION_COL],
                                                  order=(p, d, q),
                                                  seasonal_order=(P, D, Q, s),
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False)
                results = model.fit(disp=False)
                
                st.success("Modelo SARIMA ajustado exitosamente.")
                
                # Pronóstico
                forecast_periods = st.slider("Meses para pronosticar", min_value=1, max_value=36, value=12, key="sarima_periods")
                forecast = results.get_forecast(steps=forecast_periods)
                
                st.session_state.sarima_forecast = forecast.predicted_mean
                
                forecast_ci = forecast.conf_int()
                
                fig_sarima = go.Figure()
                fig_sarima.add_trace(go.Scatter(x=sarima_data.index, y=sarima_data[Config.PRECIPITATION_COL],
                                              mode='lines', name='Histórico'))
                fig_sarima.add_trace(go.Scatter(x=forecast.predicted_mean.index, y=forecast.predicted_mean,
                                              mode='lines', name='Pronóstico', line=dict(color='red')))
                fig_sarima.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci['lower ' + Config.PRECIPITATION_COL],
                                              mode='lines', line=dict(width=0), name='Intervalo de Confianza Inferior',
                                              showlegend=False))
                fig_sarima.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci['upper ' + Config.PRECIPITATION_COL],
                                              fill='tonexty', mode='lines', line=dict(width=0),
                                              name='Intervalo de Confianza', showlegend=True))
                
                fig_sarima.update_layout(title=f'Pronóstico de Precipitación para {station_filter} con SARIMA',
                                         xaxis_title='Fecha', yaxis_title='Precipitación (mm)')
                st.plotly_chart(fig_sarima, use_container_width=True)
                add_plotly_download_buttons(fig_sarima, f"sarima_forecast_{station_filter}")
            
            except Exception as e:
                st.error(f"Error al ajustar el modelo SARIMA: {e}")
                st.info("Ajuste los parámetros del modelo o seleccione otra estación. Este modelo es sensible a la calidad de los datos.")


# ---
# Lógica Principal de Streamlit
# ---
def main():
    Config.initialize_session_state()
    st.set_page_config(
        page_title=Config.APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title(Config.APP_TITLE)
    
    with st.sidebar:
        st.header("⚙️ Panel de Control")
        if os.path.exists(Config.LOGO_DROP_PATH):
            st.image(Config.LOGO_DROP_PATH, width=150)
            
        st.markdown("---")
        st.subheader("1. Carga de Archivos")
        uploaded_file_mapa = st.file_uploader("Cargar Archivo de Estaciones (.csv)", type=["csv"], key="upload_map")
        uploaded_file_precip = st.file_uploader("Cargar Archivo de Precipitación (.csv)", type=["csv"], key="upload_precip")
        uploaded_zip_shapefile = st.file_uploader("Cargar Shapefile de Municipios (.zip)", type=["zip"], key="upload_shp")

        if uploaded_file_mapa and uploaded_file_precip and uploaded_zip_shapefile and not st.session_state.data_loaded:
            with st.spinner("Procesando archivos... Esto puede tomar unos minutos."):
                gdf_stations, gdf_municipios, df_long, df_enso = load_and_process_all_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
                if gdf_stations is not None:
                    st.session_state.data_loaded = True
                    st.session_state.gdf_stations = gdf_stations
                    st.session_state.gdf_municipios = gdf_municipios
                    st.session_state.df_long = df_long
                    st.session_state.df_enso = df_enso
                    st.success("Archivos cargados y procesados exitosamente!")
                else:
                    st.session_state.data_loaded = False
                    st.error("Error al cargar los archivos. Por favor, revise el formato y vuelva a intentarlo.")

        if st.session_state.data_loaded:
            st.markdown("---")
            st.subheader("2. Opciones de Análisis")
            
            df_monthly = st.session_state.df_long.copy()
            stations_list = sorted(df_monthly[Config.STATION_NAME_COL].unique())
            
            st.session_state.analysis_mode = st.radio("Seleccionar Modo de Análisis", ["Usar datos originales", "Completar series (interpolación)"], help="La interpolación lineal llenará los datos faltantes en cada serie de tiempo.")
            
            if st.session_state.analysis_mode == "Completar series (interpolación)":
                st.session_state.df_monthly_processed = complete_series(df_monthly)
            else:
                st.session_state.df_monthly_processed = df_monthly
            
            if st.button("🧹 Limpiar Filtros"):
                st.session_state.min_data_perc_slider = 0
                st.session_state.altitude_multiselect = []
                st.session_state.regions_multiselect = []
                st.session_state.municipios_multiselect = []
                st.session_state.celdas_multiselect = []
                st.session_state.station_multiselect = []
                st.session_state.select_all_stations_state = False
                st.experimental_rerun()
            
            st.markdown("---")
            st.subheader("3. Filtrar Datos")
            
            df_filtered_summary = st.session_state.df_monthly_processed.groupby(Config.STATION_NAME_COL).agg(
                total_meses=(Config.DATE_COL, 'count')
            ).reset_index()
            
            df_filtered_summary['meses_totales_periodo'] = (st.session_state.df_monthly_processed[Config.DATE_COL].max() - st.session_state.df_monthly_processed[Config.DATE_COL].min()).days / 30.44
            df_filtered_summary[Config.PERCENTAGE_COL] = (df_filtered_summary['total_meses'] / df_filtered_summary['meses_totales_periodo']) * 100
            
            gdf_filtered_base = st.session_state.gdf_stations.merge(df_filtered_summary[[Config.STATION_NAME_COL, Config.PERCENTAGE_COL]], on=Config.STATION_NAME_COL, how='left')
            gdf_filtered_base[Config.PERCENTAGE_COL] = gdf_filtered_base[Config.PERCENTAGE_COL].fillna(0)
            
            min_data_perc = st.slider("Porcentaje Mínimo de Datos Disponibles (%)", 0, 100, st.session_state.min_data_perc_slider, key='min_data_perc_slider')
            
            gdf_filtered = gdf_filtered_base[gdf_filtered_base[Config.PERCENTAGE_COL] >= min_data_perc].copy()
            
            # Filtros adicionales
            if Config.ALTITUDE_COL in gdf_filtered.columns:
                unique_altitudes = sorted(gdf_filtered[Config.ALTITUDE_COL].dropna().unique())
                st.session_state.altitude_multiselect = st.multiselect("Filtrar por Altitud (m)", unique_altitudes, default=st.session_state.altitude_multiselect, key='altitude_multiselect')
                if st.session_state.altitude_multiselect:
                    gdf_filtered = gdf_filtered[gdf_filtered[Config.ALTITUDE_COL].isin(st.session_state.altitude_multiselect)]
            
            if Config.REGION_COL in gdf_filtered.columns:
                unique_regions = sorted(gdf_filtered[Config.REGION_COL].dropna().unique())
                st.session_state.regions_multiselect = st.multiselect("Filtrar por Región", unique_regions, default=st.session_state.regions_multiselect, key='regions_multiselect')
                if st.session_state.regions_multiselect:
                    gdf_filtered = gdf_filtered[gdf_filtered[Config.REGION_COL].isin(st.session_state.regions_multiselect)]

            if Config.MUNICIPALITY_COL in gdf_filtered.columns:
                unique_municipios = sorted(gdf_filtered[Config.MUNICIPALITY_COL].dropna().unique())
                st.session_state.municipios_multiselect = st.multiselect("Filtrar por Municipio", unique_municipios, default=st.session_state.municipios_multiselect, key='municipios_multiselect')
                if st.session_state.municipios_multiselect:
                    gdf_filtered = gdf_filtered[gdf_filtered[Config.MUNICIPALITY_COL].isin(st.session_state.municipios_multiselect)]

            if Config.CELL_COL in gdf_filtered.columns:
                unique_celdas = sorted(gdf_filtered[Config.CELL_COL].dropna().unique())
                st.session_state.celdas_multiselect = st.multiselect("Filtrar por Celda XY", unique_celdas, default=st.session_state.celdas_multiselect, key='celdas_multiselect')
                if st.session_state.celdas_multiselect:
                    gdf_filtered = gdf_filtered[gdf_filtered[Config.CELL_COL].isin(st.session_state.celdas_multiselect)]
            
            filtered_stations_list = sorted(gdf_filtered[Config.STATION_NAME_COL].unique())
            
            st.session_state.select_all_stations_state = st.checkbox("Seleccionar todas las estaciones filtradas", value=st.session_state.select_all_stations_state)
            if st.session_state.select_all_stations_state:
                st.session_state.station_multiselect = filtered_stations_list
            else:
                st.session_state.station_multiselect = st.multiselect("Seleccionar Estación para el Análisis", filtered_stations_list, default=st.session_state.station_multiselect, key='station_multiselect')

            stations_for_analysis = st.session_state.station_multiselect
            
            st.markdown("---")
            st.subheader("4. Seleccionar Período de Tiempo")
            
            start_year = int(st.session_state.df_long[Config.YEAR_COL].min())
            end_year = int(st.session_state.df_long[Config.YEAR_COL].max())
            st.session_state.year_range = st.slider("Rango de Años", start_year, end_year, (start_year, end_year), key='year_range')
            
            start_month, end_month = st.select_slider("Rango de Meses", options=list(range(1, 13)), value=(1, 12), key='month_range')
            
            # Aplicar filtros
            df_monthly_filtered = st.session_state.df_monthly_processed[
                (st.session_state.df_monthly_processed[Config.YEAR_COL].between(st.session_state.year_range[0], st.session_state.year_range[1])) &
                (st.session_state.df_monthly_processed[Config.MONTH_COL].between(start_month, end_month)) &
                (st.session_state.df_monthly_processed[Config.STATION_NAME_COL].isin(stations_for_analysis))
            ].copy()
            
            df_anual_melted = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL].sum().reset_index()

            gdf_filtered = st.session_state.gdf_stations[st.session_state.gdf_stations[Config.STATION_NAME_COL].isin(stations_for_analysis)].copy()

            # Pestañas principales
            tab_inicio, tab_espacial, tab_analisis, tab_avanzado, tab_tendencias = st.tabs(["Inicio", "Distribución Espacial", "Análisis de Datos", "Mapas Avanzados", "Tendencias y Pronósticos"])

            with tab_inicio:
                display_welcome_tab()
            
            with tab_espacial:
                display_spatial_distribution_tab(gdf_filtered, stations_for_analysis, df_anual_melted, df_monthly_filtered)
            
            with tab_analisis:
                display_analysis_tab(df_monthly_filtered, stations_for_analysis)
                
            with tab_avanzado:
                display_advanced_maps_tab(gdf_filtered, df_monthly_filtered, stations_for_analysis)
            
            with tab_tendencias:
                display_trends_forecast_tab(df_monthly_filtered, stations_for_analysis)
        
    else:
        st.warning("Por favor, suba todos los archivos requeridos en la barra lateral para comenzar.")
        display_welcome_tab()

if __name__ == "__main__":
    main()
