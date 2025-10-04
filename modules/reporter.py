import streamlit as st
import pandas as pd
import io
import os
import numpy as np
import pymannkendall as mk
from fpdf import FPDF
import plotly.graph_objects as go
import plotly.express as px
from modules.config import Config

class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "Reporte"

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

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def add_plotly_fig(self, fig, width=190):
        try:
            img_bytes = fig.to_image(format="png", scale=2)
            self.image(io.BytesIO(img_bytes), w=width)
            self.ln(5)
        except Exception as e:
            self.set_text_color(255, 0, 0)
            self.multi_cell(0, 6, f"Error al generar gráfico: {e}")
            self.set_text_color(0, 0, 0)

def add_summary_to_pdf(pdf, summary_dict):
    pdf.chapter_title("Resumen de Filtros Aplicados")
    pdf.set_font('Arial', '', 10)
    
    effective_width = pdf.w - 2 * pdf.l_margin
    key_width = effective_width * 0.3
    value_width = effective_width * 0.7

    for key, value in summary_dict.items():
        if value:
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(key_width, 6, f"{key}:", border=0)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(value_width, 6, str(value), border=0)
    pdf.ln(10)

def add_dataframe_to_pdf(pdf, df):
    """Añade un DataFrame de Pandas como una tabla al PDF."""
    pdf.set_font("Arial", 'B', 8)
    
    num_cols = len(df.columns)
    effective_width = pdf.w - 2 * pdf.l_margin
    
    col_widths = [effective_width * 0.4] + [(effective_width * 0.6) / (num_cols - 1)] * (num_cols - 1) if num_cols > 1 else [effective_width]
    
    for i, header in enumerate(df.columns):
        pdf.cell(col_widths[i], 7, str(header), 1, 0, 'C')
    pdf.ln()
    
    pdf.set_font("Arial", '', 7)
    for index, row in df.iterrows():
        for i, item in enumerate(row):
            if isinstance(item, float):
                item = f"{item:.2f}"
            pdf.cell(col_widths[i], 6, str(item), 1)
        pdf.ln()
    pdf.ln(10)

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

    if sections_to_include.get("Resumen de Filtros"):
        add_summary_to_pdf(pdf, summary_data)

    if sections_to_include.get("Serie Anual"):
        if not df_anual_melted.empty:
            pdf.chapter_title("Serie de Tiempo de Precipitación Anual")
            fig_anual = px.line(df_anual_melted, x=Config.YEAR_COL, y=Config.PRECIPITATION_COL, color=Config.STATION_NAME_COL, title="Precipitación Anual por Estación")
            pdf.add_plotly_fig(fig_anual)

    if sections_to_include.get("Anomalías Mensuales"):
        if not df_anomalies.empty:
            pdf.chapter_title("Anomalías Mensuales de Precipitación")
            df_plot = df_anomalies.groupby(Config.DATE_COL).agg(anomalia=('anomalia', 'mean')).reset_index()
            df_plot['color'] = np.where(df_plot['anomalia'] < 0, 'red', 'blue')
            fig_anom = go.Figure(go.Bar(x=df_plot[Config.DATE_COL], y=df_plot['anomalia'], marker_color=df_plot['color']))
            fig_anom.update_layout(title="Anomalías Mensuales Promedio")
            pdf.add_plotly_fig(fig_anom)
            
    if sections_to_include.get("Estadísticas de Tendencia"):
        pdf.chapter_title("Tabla Comparativa de Tendencias (Mann-Kendall)")
        
        stations_for_analysis = gdf_filtered[Config.STATION_NAME_COL].unique()
        results = []
        df_anual_calc = df_anual_melted.copy()

        for station in stations_for_analysis:
            station_data = df_anual_calc[df_anual_calc[Config.STATION_NAME_COL] == station].dropna(subset=[Config.PRECIPITATION_COL]).sort_values(by=Config.YEAR_COL)
            trend_mk, p_mk, slope_sen = "Datos insuficientes", np.nan, np.nan
            
            if len(station_data) > 3:
                mk_result_table = mk.original_test(station_data[Config.PRECIPITATION_COL])
                trend_mk = mk_result_table.trend.capitalize()
                p_mk = mk_result_table.p
                slope_sen = mk_result_table.slope

            results.append({
                "Estación": station,
                "Tendencia MK": trend_mk,
                "Valor p (MK)": p_mk,
                "Pendiente (mm/año)": slope_sen,
            })
        
        if results:
            trends_df = pd.DataFrame(results)
            trends_df['Estación'] = trends_df['Estación'].str.slice(0, 25)
            add_dataframe_to_pdf(pdf, trends_df)

    # Se elimina el guion final que causaba el SyntaxError
    return pdf.output(dest='S')
