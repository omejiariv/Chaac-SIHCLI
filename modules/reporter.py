# modules/reporter.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import folium
from fpdf import FPDF
import io
import os
import tempfile
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options

from modules.config import Config
from modules.visualizer import create_folium_map, generate_station_popup_html

# --- Configuración para Selenium ---
def setup_driver():
    """Configura y retorna un driver de Selenium para usar Chromium."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1200,800")
    chrome_options.binary_location = "/usr/bin/chromium"
    service = ChromeService(executable_path="/usr/bin/chromedriver")
    
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        st.error(f"Error al iniciar el WebDriver para generar el reporte: {e}")
        st.warning("Ocurrió un problema al configurar Chromium. Revisa los logs.")
        return None

# --- Clase para generar el PDF ---
class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.WIDTH = 210
        self.HEIGHT = 297

    def header(self):
        # Logo - se verifica si existe la ruta del logo
        if os.path.exists(Config.LOGO_PATH):
            self.image(Config.LOGO_PATH, 10, 8, 25) # x, y, ancho
        
        self.set_font('Arial', 'B', 14)
        # Se mueve el título un poco para no superponerse con el logo
        self.cell(80) 
        self.cell(30, 10, 'Reporte de Análisis Hidroclimático', 0, 1, 'C')
        # Salto de línea más grande para dar espacio
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def add_section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def add_body_text(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln()

    def add_dataframe(self, df):
        self.set_font('Arial', '', 8)
        if df.empty:
            self.cell(0, 10, "No hay datos disponibles.", 0, 1)
            return

        self.set_font('Arial', 'B', 8)
        col_widths = [30] + [(self.WIDTH - 50) / (len(df.columns) - 1)] * (len(df.columns) - 1)
        for i, col in enumerate(df.columns):
            self.cell(col_widths[i], 10, str(col), 1, 0, 'C')
        self.ln()

        self.set_font('Arial', '', 8)
        for index, row in df.iterrows():
            for i, item in enumerate(row):
                text = f"{item:.2f}" if isinstance(item, (int, float)) else str(item)
                self.cell(col_widths[i], 10, text, 1, 0, 'L')
            self.ln()
        self.ln(5)

    def add_plotly_fig(self, fig, width=190):
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
            self.image(io.BytesIO(bytes(img_bytes)), w=width)
            self.ln(5)
        except Exception as e:
            self.add_body_text(f"Error al generar la imagen del gráfico: {e}")

    def add_altair_chart(self, chart, width=190):
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                chart.save(tmpfile.name, format='png', scale_factor=2.0)
                self.image(tmpfile.name, w=width)
                os.unlink(tmpfile.name)
            self.ln(5)
        except Exception as e:
            self.add_body_text(f"Error al generar la imagen del gráfico Altair: {e}")

    def add_folium_map(self, map_obj, width=190):
        driver = setup_driver()
        if not driver:
            self.add_body_text("No se pudo generar la imagen del mapa (WebDriver no disponible).")
            return

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_html:
            map_obj.save(tmp_html.name)
        
        driver.get(f"file://{tmp_html.name}")
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
            driver.save_screenshot(tmp_png.name)
        
        driver.quit()
        
        try:
            # Esta es la corrección clave: pasamos la ruta del archivo, no un objeto de imagen.
            self.image(tmp_png.name, w=width)
            self.ln(5)
        finally:
            os.unlink(tmp_html.name)
            os.unlink(tmp_png.name)

    def add_synthesis(self, stats_dict):
        self.set_font('Arial', 'B', 10)
        self.cell(0, 8, "Máximos Históricos", 0, 1)
        self.set_font('Arial', '', 10)
        self.cell(0, 6, f"  - Ppt. Máxima Anual: {stats_dict.get('max_ppt_anual', 'N/A')}", 0, 1)
        self.cell(0, 6, f"  - Ppt. Máxima Mensual: {stats_dict.get('max_ppt_mensual', 'N/A')}", 0, 1)
        self.cell(0, 6, f"  - Año Promedio más Lluvioso: {stats_dict.get('ano_mas_lluvioso', 'N/A')}", 0, 1)
        self.ln(5)

# --- Función Principal ---
def generate_pdf_report(report_title, sections_to_include, gdf_filtered, df_anual_melted, df_monthly_filtered, summary_data, df_anomalies, **kwargs):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, report_title, 0, 1, 'C') # Título principal debajo del header
    pdf.ln(10)

    if sections_to_include.get("Resumen de Filtros"):
        pdf.add_section_title("Resumen de Filtros Aplicados")
        filter_text = ""
        for key, value in summary_data.items():
            filter_text += f"- {key}: {value}\n"
        pdf.add_body_text(filter_text)

    if sections_to_include.get("Mapa de Distribución"):
        pdf.add_section_title("Mapa de Distribución Espacial")
        if not gdf_filtered.empty:
            m = create_folium_map(location=[4.57, -74.29], zoom=5, base_map_config={"tiles": "cartodbpositron", "attr": "CartoDB"}, overlays_config=[], fit_bounds_data=gdf_filtered)
            for _, row in gdf_filtered.iterrows():
                popup = generate_station_popup_html(row, df_anual_melted)
                folium.Marker(location=[row.geometry.y, row.geometry.x], tooltip=row[Config.STATION_NAME_COL], popup=popup).add_to(m)
            pdf.add_folium_map(m)
        else:
            pdf.add_body_text("No hay estaciones seleccionadas para mostrar en el mapa.")

    if sections_to_include.get("Serie Anual"):
        pdf.add_section_title("Precipitación Anual por Estación")
        if not df_anual_melted.empty:
            fig = px.line(df_anual_melted, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, title="Precipitación Anual por Estación", markers=True)
            pdf.add_plotly_fig(fig)
        else:
            pdf.add_body_text("No hay datos anuales para mostrar.")

    if sections_to_include.get("Anomalías Mensuales"):
        pdf.add_section_title("Anomalías Mensuales de Precipitación")
        if not df_anomalies.empty:
            df_plot = df_anomalies.groupby(Config.DATE_COL).agg(anomalia=('anomalia', 'mean')).reset_index()
            df_plot['color'] = np.where(df_plot['anomalia'] < 0, 'red', 'blue')
            fig = go.Figure(go.Bar(x=df_plot[Config.DATE_COL], y=df_plot['anomalia'], marker_color=df_plot['color'], name='Anomalía'))
            fig.update_layout(title="Anomalías Mensuales de Precipitación (Promedio Regional)")
            pdf.add_plotly_fig(fig)
        else:
            pdf.add_body_text("No hay datos de anomalías para mostrar.")
            
    if sections_to_include.get("Resumen Mensual"):
        pdf.add_section_title("Resumen de Estadísticas Mensuales por Estación")
        df_summary = kwargs.get('df_summary_monthly')
        if df_summary is not None and not df_summary.empty:
            # --- INICIO DE LA CORRECCIÓN DE SANGRÍA ---
            pdf.add_dataframe(df_summary.round(1))
            # --- FIN DE LA CORRECCIÓN DE SANGRÍA ---
        else:
            pdf.add_body_text("No se calcularon los datos del resumen mensual.")
            
    if sections_to_include.get("Síntesis General"):
        pdf.add_section_title("Síntesis General de Estadísticas")
        synthesis_data = kwargs.get('synthesis_stats')
        if synthesis_data:
            pdf.add_synthesis(synthesis_data)
        else:
            pdf.add_body_text("No se calcularon los datos de la síntesis general.")

    if sections_to_include.get("Matriz de Disponibilidad"):
        pdf.add_section_title("Matriz de Disponibilidad de Datos Originales (%)")
        heatmap_df = kwargs.get('heatmap_df')
        if heatmap_df is not None and not heatmap_df.empty:
             fig = px.imshow(heatmap_df, text_auto=True, aspect="auto", color_continuous_scale='Greens', labels=dict(color="%"))
             pdf.add_plotly_fig(fig, width=180)
        else:
            pdf.add_body_text("No se pudieron generar los datos de disponibilidad.")

    if sections_to_include.get("Matriz de Correlación"):
        pdf.add_section_title("Matriz de Correlación entre Estaciones")
        num_stations_str = summary_data.get("Estaciones Seleccionadas", "0 de 0").split(" de ")[0]
        if num_stations_str.isdigit() and int(num_stations_str) > 1:
            df_pivot = df_monthly_filtered.pivot_table(index=Config.DATE_COL, columns=Config.STATION_NAME_COL, values=Config.PRECIPITATION_COL)
            corr_matrix = df_pivot.corr()
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r')
            pdf.add_plotly_fig(fig, width=180)
        else:
            pdf.add_body_text("Se necesitan al menos dos estaciones para generar la matriz de correlación.")

    if sections_to_include.get("Serie Regional"):
        pdf.add_section_title("Serie de Tiempo Promedio Regional")
        df_regional = kwargs.get('df_regional')
        if df_regional is not None and not df_regional.empty:
            fig = px.line(df_regional, x=Config.DATE_COL, y='Precipitación Promedio', title="Serie de Tiempo Promedio Regional")
            pdf.add_plotly_fig(fig)
        else:
            pdf.add_body_text("No hay datos para la serie regional.")

    if sections_to_include.get("Tabla de Tendencias"):
        pdf.add_section_title("Tabla Comparativa de Tendencias (Mann-Kendall)")
        df_trends = kwargs.get('df_trends')
        if df_trends is not None and not df_trends.empty:
            pdf.add_dataframe(df_trends)
        else:
            pdf.add_body_text("No se calcularon los datos de tendencias para el reporte.")

    if sections_to_include.get("SARIMA vs Prophet"):
        pdf.add_section_title("Comparación de Pronósticos: SARIMA vs Prophet")
        fig_compare = kwargs.get('fig_compare_forecast')
        if fig_compare:
            pdf.add_plotly_fig(fig_compare)
        else:
            pdf.add_body_text("Genere ambos pronósticos en la aplicación para incluirlos en el reporte.")

    return pdf.output()
