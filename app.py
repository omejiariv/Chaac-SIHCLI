# app.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import pymannkendall as mk
import plotly.graph_objects as go
from scipy import stats

#--- Importaciones de Módulos
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series
from modules.visualizer import (
    display_welcome_tab, display_spatial_distribution_tab, display_graphs_tab,
    display_advanced_maps_tab, display_anomalies_tab, display_drought_analysis_tab,
    display_frequency_analysis_tab,
    display_stats_tab, display_correlation_tab, display_enso_tab,
    display_trends_and_forecast_tab, display_downloads_tab, display_station_table_tab,
    display_validation_tab
)
from modules.reporter import generate_pdf_report
from modules.analysis import calculate_monthly_anomalies
# --- INICIO DE LA MODIFICACIÓN: Importar el nuevo módulo ---
from modules.github_loader import load_csv_from_url, load_zip_from_url
# --- FIN DE LA MODIFICACIÓN ---


#--- Desactivar Advertencias
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def sync_station_selection(stations_options):
    """Sincroniza el multiselect basado en el checkbox 'Seleccionar todas'."""
    if st.session_state.get('select_all_checkbox', False):
        st.session_state.station_multiselect = stations_options
    else:
        st.session_state.station_multiselect = []

def apply_filters_to_stations(df, min_perc, altitudes, regions, municipios, celdas):
    """Aplica una serie de filtros geográficos y de datos a un GeoDataFrame de estaciones."""
    stations_filtered = df.copy()
    if Config.PERCENTAGE_COL in stations_filtered.columns:
        stations_filtered[Config.PERCENTAGE_COL] = pd.to_numeric(
            stations_filtered[Config.PERCENTAGE_COL].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        ).fillna(0)
    if min_perc > 0:
        stations_filtered = stations_filtered[stations_filtered[Config.PERCENTAGE_COL] >= min_perc]
    if altitudes:
        conditions = []
        altitude_col_numeric = pd.to_numeric(stations_filtered[Config.ALTITUDE_COL], errors='coerce')
        for r in altitudes:
            if r == '0-500': conditions.append((altitude_col_numeric >= 0) & (altitude_col_numeric <= 500))
            elif r == '500-1000': conditions.append((altitude_col_numeric > 500) & (altitude_col_numeric <= 1000))
            elif r == '1000-2000': conditions.append((altitude_col_numeric > 1000) & (altitude_col_numeric <= 2000))
            elif r == '2000-3000': conditions.append((altitude_col_numeric > 2000) & (altitude_col_numeric <= 3000))
            elif r == '>3000': conditions.append(altitude_col_numeric > 3000)
        if conditions:
            stations_filtered = stations_filtered[pd.concat(conditions, axis=1).any(axis=1)]
    if regions:
        stations_filtered = stations_filtered[stations_filtered[Config.REGION_COL].isin(regions)]
    if municipios:
        stations_filtered = stations_filtered[stations_filtered[Config.MUNICIPALITY_COL].isin(municipios)]
    if celdas and Config.CELL_COL in stations_filtered.columns:
        stations_filtered = stations_filtered[stations_filtered[Config.CELL_COL].isin(celdas)]
    return stations_filtered

def main():
    st.set_page_config(layout="wide", page_title=Config.APP_TITLE)
    progress_placeholder = st.empty()

    st.markdown("""<style>div.block-container{padding-top:1rem;} [data-testid="stMetricValue"] {font-size:1.8rem;} [data-testid="stMetricLabel"] {font-size: 1rem; padding-bottom:5px; } button[data-baseweb="tab"] {font-size:16px;font-weight:bold;color:#333;}</style>""", unsafe_allow_html=True)

    Config.initialize_session_state()
    title_col1, title_col2 = st.columns([0.05, 0.95])
    with title_col1:
        if os.path.exists(Config.LOGO_PATH):
            try: st.image(Config.LOGO_PATH, width=60)
            except Exception: pass
    with title_col2:
        st.markdown(f'<h1 style="font-size:28px; margin-top:1rem;">{Config.APP_TITLE}</h1>', unsafe_allow_html=True)

    st.sidebar.header("Panel de Control")
    
    # --- INICIO DE LA MODIFICACIÓN: Lógica de carga automática/manual ---
    with st.sidebar.expander("**Subir/Actualizar Archivos Base**", expanded=not st.session_state.get('data_loaded', False)):
        
        load_mode = st.radio(
            "Modo de Carga de Archivos",
            ("Automático (desde GitHub)", "Manual (Subir archivos)"),
            key="load_mode",
            horizontal=True
        )

        # Lógica para procesar los datos, sea cual sea el origen
        def process_and_store_data(file_mapa, file_precip, file_shape):
            st.cache_data.clear()
            st.cache_resource.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            Config.initialize_session_state()
            with st.spinner("Procesando archivos y cargando datos..."):
                gdf_stations, gdf_municipios, df_long, df_enso = load_and_process_all_data(file_mapa, file_precip, file_shape)
                if gdf_stations is not None and df_long is not None and gdf_municipios is not None:
                    st.session_state.gdf_stations = gdf_stations
                    st.session_state.gdf_municipios = gdf_municipios
                    st.session_state.df_long = df_long
                    st.session_state.df_enso = df_enso
                    st.session_state.data_loaded = True
                    st.success("¡Datos cargados y listos!")
                    st.rerun()
                else:
                    st.error("Hubo un error al procesar los archivos. Verifique el formato y contenido.")

        if load_mode == "Manual (Subir archivos)":
            uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (CSV)", type="csv", key='uploaded_file_mapa')
            uploaded_file_precip = st.file_uploader("2. Cargar archivo de precipitación (CSV)", type="csv", key='uploaded_file_precip')
            uploaded_zip_shapefile = st.file_uploader("3. Cargar shapefile de municipios (.zip)", type="zip", key='uploaded_zip_shapefile')
            
            if st.button("Procesar Datos Manuales", key='process_data_manual_button'):
                if all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
                    process_and_store_data(uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
                else:
                    st.warning("Por favor, suba los tres archivos requeridos.")

        else: # Modo Automático
            st.info(f"Los datos se cargarán desde el repositorio de GitHub: **{Config.GITHUB_USER}/{Config.GITHUB_REPO}**")
            if st.button("Cargar Datos desde GitHub", key='process_data_github_button'):
                with st.spinner("Descargando archivos desde GitHub..."):
                    github_file_mapa = load_csv_from_url(Config.URL_ESTACIONES_CSV)
                    github_file_precip = load_csv_from_url(Config.URL_PRECIPITACION_CSV)
                    github_file_shape = load_zip_from_url(Config.URL_SHAPEFILE_ZIP)
                
                if all([github_file_mapa, github_file_precip, github_file_shape]):
                    process_and_store_data(github_file_mapa, github_file_precip, github_file_shape)
                else:
                    st.error("No se pudieron descargar uno o más archivos desde GitHub. Revisa las URLs en config.py y que el repositorio sea público.")
    # --- FIN DE LA MODIFICACIÓN ---

    if not st.session_state.get('data_loaded', False):
        display_welcome_tab()
        st.info("Para comenzar, cargue los archivos requeridos en el panel de la izquierda.")
        return

    # ... (El resto del archivo, desde st.sidebar.success("Datos base cargados.") hasta el final, permanece exactamente igual) ...
    st.sidebar.success("Datos base cargados.")
    if st.sidebar.button("Limpiar Caché y Reiniciar App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True):
        min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, st.session_state.get('min_data_perc_slider', 0), key='min_data_perc_slider')
        altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000']
        selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges, key='altitude_multiselect')
        regions_list = sorted(st.session_state.gdf_stations[Config.REGION_COL].dropna().unique())
        selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list, key='regions_multiselect')
        temp_gdf_for_mun = st.session_state.gdf_stations.copy()
        if selected_regions:
            temp_gdf_for_mun = temp_gdf_for_mun[temp_gdf_for_mun[Config.REGION_COL].isin(selected_regions)]
        municipios_list = sorted(temp_gdf_for_mun[Config.MUNICIPALITY_COL].dropna().unique())
        selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list, key='municipios_multiselect')
        celdas_list = []
        if Config.CELL_COL in temp_gdf_for_mun.columns:
            celdas_list = sorted(temp_gdf_for_mun[Config.CELL_COL].dropna().unique())
        selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, key='celdas_multiselect')

    gdf_filtered = apply_filters_to_stations(st.session_state.gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas)

    with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True):
        stations_options = sorted(gdf_filtered[Config.STATION_NAME_COL].unique())

        # --- LÓGICA CORREGIDA PARA EVITAR EL ERROR "Bad message format" ---
        
        # Guardamos el estado anterior del checkbox para detectar cuándo cambia.
        if 'select_all_prev_state' not in st.session_state:
            st.session_state.select_all_prev_state = False

        # Creamos el checkbox sin el 'on_change'
        select_all_checkbox = st.checkbox(
            "Seleccionar/Deseleccionar todas las estaciones", 
            value=st.session_state.select_all_prev_state, # Usamos el estado guardado
            key='select_all_checkbox_main'
        )

        # Si el estado del checkbox cambió en esta ejecución...
        if select_all_checkbox != st.session_state.select_all_prev_state:
            if select_all_checkbox:  # Si se acaba de marcar
                st.session_state.station_multiselect = stations_options
            else:  # Si se acaba de desmarcar
                st.session_state.station_multiselect = []
            
            # Actualizamos el estado guardado y forzamos un refresco de la app.
            st.session_state.select_all_prev_state = select_all_checkbox
            st.rerun()
        
        # El widget multiselect ahora usa el st.session_state que ya hemos preparado.
        selected_stations = st.multiselect(
            'Seleccionar Estaciones', 
            options=stations_options, 
            key='station_multiselect'
        )
        # --- FIN DE LA LÓGICA CORREGIDA ---

        years_with_data = sorted(st.session_state.df_long[Config.YEAR_COL].dropna().unique())
        year_range_default = (min(years_with_data), max(years_with_data)) if years_with_data else (1970, 2020)
        year_range = st.slider(
            "Seleccionar Rango de Años", 
            min_value=year_range_default[0], 
            max_value=year_range_default[1], 
            value=st.session_state.get('year_range', year_range_default), 
            key='year_range'
        )
        meses_dict = {m: i+1 for i, m in enumerate(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])}
        meses_nombres = st.multiselect(
            "Seleccionar Meses", 
            list(meses_dict.keys()), 
            default=list(meses_dict.keys()), 
            key='meses_nombres'
        )
        meses_numeros = [meses_dict[m] for m in meses_nombres]
    with st.sidebar.expander("Opciones de Preprocesamiento"):
        st.radio("Modo de análisis", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode", help="La opción 'Completar series' utiliza interpolación para rellenar los datos faltantes. Afecta a todas las pestañas de análisis y a las descargas.")
        st.checkbox("Excluir datos nulos (NaN)", key='exclude_na')
        st.checkbox("Excluir valores cero (0)", key='exclude_zeros')

    tab_names = ["Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados", "Validación de Interpolación", "Análisis de Anomalías", "Análisis de extremos hid", "Frecuencia de Extremos", "Estadísticas", "Análisis de Correlación", "Análisis ENSO", "Tendencias y Pronósticos", "Descargas", "Tabla de Estaciones", "Generar Reporte"]
    tabs = st.tabs(tab_names)
    stations_for_analysis = selected_stations

    if not stations_for_analysis:
        with tabs[0]:
            display_welcome_tab()
            st.warning("No hay estaciones seleccionadas. Por favor, seleccione al menos una estación en el panel de control para comenzar el análisis.")
        return

    gdf_filtered = gdf_filtered[gdf_filtered[Config.STATION_NAME_COL].isin(stations_for_analysis)]
    df_monthly_filtered = st.session_state.df_long[
        (st.session_state.df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
        (st.session_state.df_long[Config.DATE_COL].dt.year >= year_range[0]) &
        (st.session_state.df_long[Config.DATE_COL].dt.year <= year_range[1]) &
        (st.session_state.df_long[Config.DATE_COL].dt.month.isin(meses_numeros))
    ].copy()

    if st.session_state.analysis_mode == "Completar series (interpolación)":
        bar = progress_placeholder.progress(0, text="Iniciando interpolación de series...")
        df_monthly_filtered = complete_series(df_monthly_filtered, _progress_bar=bar)
        progress_placeholder.empty()
        annual_agg = df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).agg(
            precipitation_sum=(Config.PRECIPITATION_COL, 'sum'),
            meses_validos=(Config.MONTH_COL, 'nunique')
        ).reset_index()
        annual_agg.loc[annual_agg['meses_validos'] < 12, 'precipitation_sum'] = np.nan
        df_anual_melted = annual_agg.rename(columns={'precipitation_sum': Config.PRECIPITATION_COL})
    else:
        annual_data_filtered = st.session_state.df_long[
            (st.session_state.df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
            (st.session_state.df_long[Config.YEAR_COL] >= year_range[0]) &
            (st.session_state.df_long[Config.YEAR_COL] <= year_range[1])
        ].copy()
        if st.session_state.get('exclude_na', False):
            annual_data_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
        if st.session_state.get('exclude_zeros', False):
            annual_data_filtered = annual_data_filtered[annual_data_filtered[Config.PRECIPITATION_COL] > 0]
        annual_agg = annual_data_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).agg(
            precipitation_sum=(Config.PRECIPITATION_COL, 'sum'),
            meses_validos=(Config.PRECIPITATION_COL, 'count')
        ).reset_index()
        annual_agg.loc[annual_agg['meses_validos'] < 10, 'precipitation_sum'] = np.nan
        df_anual_melted = annual_agg.rename(columns={'precipitation_sum': Config.PRECIPITATION_COL})

    if st.session_state.get('exclude_na', False):
        df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True)
    if st.session_state.get('exclude_zeros', False):
        df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] > 0]

    display_args = {
        "gdf_filtered": gdf_filtered, "stations_for_analysis": stations_for_analysis,
        "df_anual_melted": df_anual_melted, "df_monthly_filtered": df_monthly_filtered,
        "analysis_mode": st.session_state.analysis_mode, "selected_regions": selected_regions,
        "selected_municipios": selected_municipios, "selected_altitudes": selected_altitudes
    }

    with tabs[0]: display_welcome_tab()
    with tabs[1]: display_spatial_distribution_tab(**display_args)
    with tabs[2]: display_graphs_tab(**display_args)
    with tabs[3]: display_advanced_maps_tab(**display_args)
    with tabs[4]: display_validation_tab(df_anual_melted=df_anual_melted, gdf_filtered=gdf_filtered, stations_for_analysis=stations_for_analysis)
    with tabs[5]: display_anomalies_tab(df_long=st.session_state.df_long, **display_args)
    with tabs[6]: display_drought_analysis_tab(**display_args)
    with tabs[7]: display_frequency_analysis_tab(**display_args)
    with tabs[8]: display_stats_tab(df_long=st.session_state.df_long, **display_args)
    with tabs[9]: display_correlation_tab(**display_args)
    with tabs[10]: display_enso_tab(df_enso=st.session_state.df_enso, **display_args)
    with tabs[11]: display_trends_and_forecast_tab(df_full_monthly=st.session_state.df_long, **display_args)
    with tabs[12]: display_downloads_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis, analysis_mode=st.session_state.analysis_mode)
    with tabs[13]: display_station_table_tab(**display_args)
    
    with tabs[14]:
        st.header("Generación de Reporte PDF")
        with st.expander("Opciones del Reporte", expanded=True):
            report_title = st.text_input("Título del Reporte", "Análisis Hidroclimático de Estaciones Seleccionadas")
            st.markdown("**Seleccione las secciones a incluir:**")
            
            select_all = st.checkbox("Seleccionar/Deseleccionar todas las secciones", value=True, key="select_all_report_sections")

            col1, col2, col3 = st.columns(3)
            
            sections_to_include = {
                "Resumen de Filtros": col1.checkbox("Resumen de Filtros Aplicados", value=select_all),
                "Mapa de Distribución": col1.checkbox("Mapa de Distribución Espacial", value=select_all),
                "Serie Anual": col1.checkbox("Gráfico de Serie de Tiempo Anual", value=select_all),
                "Anomalías Mensuales": col1.checkbox("Gráfico de Anomalías Mensuales", value=select_all),
                "Resumen Mensual": col1.checkbox("Resumen Mensual de Estadísticas", value=select_all),
                "Matriz de Disponibilidad": col2.checkbox("Matriz de Disponibilidad de Datos", value=select_all),
                "Matriz de Correlación": col2.checkbox("Matriz de Correlación", value=select_all),
                "Serie Regional": col2.checkbox("Gráfico de Serie Regional", value=select_all),
                "Síntesis General": col2.checkbox("Síntesis General de Estadísticas", value=select_all),
                "Tabla de Tendencias": col3.checkbox("Tabla de Tendencias (MK)", value=select_all),
                "SARIMA vs Prophet": col3.checkbox("Gráfico Comparativo de Pronósticos", value=select_all),
            }

        if st.button("Generar y Descargar Reporte PDF"):
            with st.spinner("Generando reporte... Este proceso puede tardar varios segundos."):
                
                summary_data = {
                    "Estaciones Seleccionadas": f"{len(stations_for_analysis)} de {len(st.session_state.gdf_stations)}",
                    "Período de Análisis": f"{year_range[0]} - {year_range[1]}",
                    "Regiones": ", ".join(selected_regions) if selected_regions else "Todas",
                    "Municipios": ", ".join(selected_municipios) if selected_municipios else "Todos",
                    "Modo de Análisis": st.session_state.analysis_mode
                }
                df_anomalies = calculate_monthly_anomalies(df_monthly_filtered, st.session_state.df_long)
                
                df_summary_report = pd.DataFrame()
                if sections_to_include.get("Resumen Mensual"):
                    summary_data_list = []
                    for station_name, group in df_monthly_filtered.groupby(Config.STATION_NAME_COL):
                        if not group[Config.PRECIPITATION_COL].dropna().empty:
                            max_row = group.loc[group[Config.PRECIPITATION_COL].idxmax()]
                            min_row = group.loc[group[Config.PRECIPITATION_COL].idxmin()]
                            summary_data_list.append({
                                "Estación": station_name,
                                "Ppt. Máx (mm)": max_row[Config.PRECIPITATION_COL],
                                "Fecha Máx": max_row[Config.DATE_COL].strftime('%Y-%m'),
                                "Ppt. Mín (mm)": min_row[Config.PRECIPITATION_COL],
                                "Fecha Mín": min_row[Config.DATE_COL].strftime('%Y-%m'),
                                "Promedio (mm)": group[Config.PRECIPITATION_COL].mean()
                            })
                    df_summary_report = pd.DataFrame(summary_data_list)

                synthesis_stats_report = {}
                if sections_to_include.get("Síntesis General"):
                    df_anual_valid = df_anual_melted.dropna(subset=[Config.PRECIPITATION_COL])
                    df_monthly_valid = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL])
                    if not df_anual_valid.empty and not df_monthly_valid.empty:
                        max_monthly_row = df_monthly_valid.loc[df_monthly_valid[Config.PRECIPITATION_COL].idxmax()]
                        max_annual_row = df_anual_valid.loc[df_anual_valid[Config.PRECIPITATION_COL].idxmax()]
                        df_yearly_avg = df_anual_valid.groupby(Config.YEAR_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                        year_max_avg = df_yearly_avg.loc[df_yearly_avg[Config.PRECIPITATION_COL].idxmax()]
                        synthesis_stats_report = {
                            "max_ppt_anual": f"{max_annual_row[Config.PRECIPITATION_COL]:.0f} mm ({max_annual_row[Config.STATION_NAME_COL]}, {int(max_annual_row[Config.YEAR_COL])})",
                            "max_ppt_mensual": f"{max_monthly_row[Config.PRECIPITATION_COL]:.0f} mm ({max_monthly_row[Config.STATION_NAME_COL]}, {max_monthly_row[Config.DATE_COL].strftime('%Y-%m')})",
                            "ano_mas_lluvioso": f"{year_max_avg[Config.PRECIPITATION_COL]:.0f} mm (Año: {int(year_max_avg[Config.YEAR_COL])})"
                        }

                trend_results = []
                for station in stations_for_analysis:
                    station_data = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station].dropna(subset=[Config.PRECIPITATION_COL])
                    if len(station_data) >= 4:
                        mk_res = mk.original_test(station_data[Config.PRECIPITATION_COL])
                        trend_results.append({"Estación": station, "Tendencia": mk_res.trend, "p-valor": mk_res.p, "Pendiente Sen": mk_res.slope})
                df_trends_report = pd.DataFrame(trend_results)

                df_counts = st.session_state.df_long[(st.session_state.df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) & (st.session_state.df_long[Config.YEAR_COL] >= year_range[0]) & (st.session_state.df_long[Config.YEAR_COL] <= year_range[1])].groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).size().reset_index(name='count')
                df_counts['porc_value'] = (df_counts['count'] / 12) * 100
                heatmap_df_report = df_counts.pivot(index=Config.STATION_NAME_COL, columns=Config.YEAR_COL, values='porc_value').fillna(0)

                df_regional_report = df_monthly_filtered.groupby(Config.DATE_COL)[Config.PRECIPITATION_COL].mean().reset_index()
                df_regional_report.rename(columns={Config.PRECIPITATION_COL: 'Precipitación Promedio'}, inplace=True)
                
                fig_compare_forecast_report = None
                sarima_res = st.session_state.get('sarima_results')
                prophet_res = st.session_state.get('prophet_results')
                if sarima_res and prophet_res:
                    fig_compare_forecast_report = go.Figure()
                    if sarima_res.get('history') is not None:
                        hist = sarima_res['history']
                        fig_compare_forecast_report.add_trace(go.Scatter(x=hist.index, y=hist, mode='lines', name='Histórico'))
                    if sarima_res.get('forecast') is not None:
                        sarima_fc = sarima_res['forecast']
                        fig_compare_forecast_report.add_trace(go.Scatter(x=sarima_fc['ds'], y=sarima_fc['yhat'], mode='lines', name='SARIMA'))
                    if prophet_res.get('forecast') is not None:
                        prophet_fc = prophet_res['forecast']
                        fig_compare_forecast_report.add_trace(go.Scatter(x=prophet_fc['ds'], y=prophet_fc['yhat'], mode='lines', name='Prophet'))
                    fig_compare_forecast_report.update_layout(title="Comparación de Pronósticos")

                try:
                    pdf_bytes = generate_pdf_report(
                        report_title=report_title,
                        sections_to_include=sections_to_include,
                        gdf_filtered=gdf_filtered,
                        df_anual_melted=df_anual_melted,
                        df_monthly_filtered=df_monthly_filtered,
                        summary_data=summary_data,
                        df_anomalies=df_anomalies,
                        df_trends=df_trends_report,
                        heatmap_df=heatmap_df_report,
                        df_regional=df_regional_report,
                        fig_compare_forecast=fig_compare_forecast_report,
                        df_summary_monthly=df_summary_report,
                        synthesis_stats=synthesis_stats_report
                    )
                    file_name_safe = "".join([c for c in report_title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
                    st.download_button(
                        label="📥 Descargar PDF",
                        data=bytes(pdf_bytes),
                        file_name=f"{file_name_safe.replace(' ', '_').lower()}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error al generar el PDF: {e}")
                    st.error("Asegúrese de tener el navegador (Chromium) instalado y accesible en su sistema para la generación de reportes.")


if __name__ == "__main__":
    main()
