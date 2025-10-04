# reporter.py

import io
import pandas as pd
from fpdf import FPDF
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from modules.config import Config

# Clase personalizada para manejar encabezado y pie de página en el PDF
class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "Reporte de Análisis Hidroclimático"

    def set_report_title(self, title):
        self.report_title = title

    def header(self):
        # Añadir logo si existe
        if os.path.exists(Config.LOGO_PATH):
            self.image(Config.LOGO_PATH, x=10, y=8, w=15)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, self.report_title, 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

# --- Funciones Auxiliares para el Reporte ---

def add_graph_to_pdf(pdf, fig, width=190):
    """Convierte una figura de Plotly a imagen y la añade al PDF."""
    try:
        img_bytes = fig.to_image(format="png", scale=2)
        pdf.image(io.BytesIO(img_bytes), w=width)
        pdf.ln(5)
    except Exception as e:
        pdf.set_font('Arial', 'I', 10)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"Error al generar gráfico: {e}")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

def add_dataframe_to_pdf(pdf, df):
    """Renderiza un DataFrame de Pandas como una tabla simple en el PDF."""
    pdf.set_font('Arial', 'B', 10)
    # Encabezados
    for col in df.columns:
        pdf.cell(45, 10, col, 1, 0, 'C')
    pdf.ln()
    
    pdf.set_font('Arial', '', 9)
    # Datos
    for index, row in df.iterrows():
        for col in df.columns:
            pdf.cell(45, 10, str(row[col]), 1, 0, 'L')
        pdf.ln()
    pdf.ln(10)

def add_summary_to_pdf(pdf, summary_dict):
    """Añade un resumen de filtros al PDF."""
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "1. Resumen de Filtros Aplicados", 0, 1)
    pdf.set_font('Arial', '', 10)
    for key, value in summary_dict.items():
        if value: # Solo muestra si hay valor
            pdf.multi_cell(0, 6, f"- **{key}:** {value}")
    pdf.ln(10)


# --- Función Principal del Módulo ---

def generate_pdf_report(
    report_title,
    sections_to_include,
    gdf_filtered,
    df_anual_melted,
    df_monthly_filtered,
    summary_data,
    df_anomalies
):
    pdf = PDF()
    pdf.set_report_title(report_title)
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    # --- Construcción del Reporte por Secciones ---

    if sections_to_include.get("Resumen de Filtros"):
        add_summary_to_pdf(pdf, summary_data)

    if sections_to_include.get("Mapa de Estaciones"):
        # Esta sección es compleja de renderizar en PDF. Por ahora, añadimos un placeholder.
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "2. Mapa de Distribución de Estaciones", 0, 1)
        pdf.set_font('Arial', 'I', 10)
        pdf.multi_cell(0, 6, "La generación de mapas interactivos en PDF no es soportada directamente. Se recomienda tomar una captura de pantalla del mapa en la pestaña 'Distribución Espacial' y añadirla manualmente si es necesario.")
        pdf.ln(10)

    if sections_to_include.get("Serie Anual") and not df_anual_melted.empty:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "3. Gráfico de Serie de Tiempo Anual", 0, 1)
        fig = px.line(df_anual_melted, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, title="Precipitación Anual por Estación")
        add_graph_to_pdf(pdf, fig)

    if sections_to_include.get("Anomalías Mensuales") and not df_anomalies.empty:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "4. Gráfico de Anomalías Mensuales", 0, 1)
        df_plot = df_anomalies.groupby(Config.DATE_COL).agg(anomalia=('anomalia', 'mean')).reset_index()
        df_plot['color'] = np.where(df_plot['anomalia'] < 0, 'red', 'blue')
        fig = go.Figure(go.Bar(x=df_plot[Config.DATE_COL], y=df_plot['anomalia'], marker_color=df_plot['color']))
        fig.update_layout(title="Anomalías Mensuales de Precipitación (Promedio Regional)")
        add_graph_to_pdf(pdf, fig)

    if sections_to_include.get("Estadísticas de Tendencia") and not df_anual_melted.empty:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "5. Tabla de Estadísticas de Tendencia (Mann-Kendall)", 0, 1)
        
        trend_results = []
        for station in gdf_filtered[Config.STATION_NAME_COL].unique():
            station_data = df_anual_melted[df_anual_melted[Config.STATION_NAME_COL] == station].copy()
            if len(station_data.dropna(subset=[Config.PRECIPITATION_COL])) >= 4:
                mk_result = mk.original_test(station_data[Config.PRECIPITATION_COL].dropna())
                trend_results.append({
                    "Estación": station,
                    "Tendencia": mk_result.trend,
                    "Pendiente (mm/año)": f"{mk_result.slope:.2f}",
                    "Valor p": f"{mk_result.p:.3f}"
                })
        if trend_results:
            df_trends = pd.DataFrame(trend_results)
            add_dataframe_to_pdf(pdf, df_trends)
        else:
            pdf.cell(0, 10, "No hay suficientes datos para calcular tendencias.")
            pdf.ln()

    # Devolver el PDF como bytes
    return pdf.output(dest='S').encode('latin-1')
