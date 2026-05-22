import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="SIHCLIM - Dashboard de Lluvia Mensual",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Sistema de Información Hidroclimática: Agregación Mensual")
st.markdown("""
Esta aplicación despliega la visualización de datos de lluvia históricos optimizados mediante 
la arquitectura híbrida de procesamiento por fragmentos.
""")

# Ruta al archivo mensual ligero generado por tu script backend
archivo_resumen = "lluvia_mensual_consolidado.csv"

if os.path.exists(archivo_resumen):
    df = pd.read_csv(archivo_resumen)
    # Asegurar el orden cronológico
    df = df.sort_values('periodo_mensual')

    # Sección de Indicadores Clave (KPIs)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Meses Analizados", len(df))
    with m2:
        st.metric("Precipitación Promedio Mensual", f"{df.iloc[:,1].mean():.1f} mm")
    with m3:
        st.metric("Máximo Histórico Mensual", f"{df.iloc[:,1].max():.1f} mm")

    st.markdown("---")
    col_izq, col_der = st.columns([2, 1])

    with col_izq:
        st.write("### Gráfico de Tendencia Histórica (Acumulado Mensual)")
        fig, ax = plt.subplots(figsize=(10, 4))
        # Ajusta el nombre de la columna según tu archivo final
        ax.plot(df['periodo_mensual'], df.iloc[:,1], marker='o', color='#1f77b4', linewidth=2)
        ax.set_xlabel("Periodo (Año-Mes)")
        ax.set_ylabel("Precipitación (mm)")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=90, fontsize=8)
        st.pyplot(fig)

    with col_der:
        st.write("### Tabla de Datos Consolidados")
        st.dataframe(df, use_container_width=True, height=320)
else:
    st.warning(f"⚠️ No se encontró el archivo '{archivo_resumen}'. Ejecuta primero el script de procesamiento en tu repositorio para consolidar los datos mensuales.")
