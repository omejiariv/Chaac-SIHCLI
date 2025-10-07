# app.py (Versión de Diagnóstico)

import streamlit as st

st.set_page_config(layout="wide")
st.title("🩺 Diagnóstico de Módulos de la Aplicación")

st.write("Intentando importar cada módulo uno por uno. Si un módulo tiene un error de sintaxis, verás un mensaje de error debajo de él.")

try:
    from modules.config import Config
    st.success("✅ Módulo `config.py` importado correctamente.")
except Exception as e:
    st.error(f"❌ ERROR al importar `config.py`: {e}")

try:
    from modules.github_loader import load_csv_from_url
    st.success("✅ Módulo `github_loader.py` importado correctamente.")
except Exception as e:
    st.error(f"❌ ERROR al importar `github_loader.py`: {e}")

try:
    from modules.data_processor import load_and_process_all_data
    st.success("✅ Módulo `data_processor.py` importado correctamente.")
except Exception as e:
    st.error(f"❌ ERROR al importar `data_processor.py`: {e}")

try:
    from modules.visualizer import display_welcome_tab
    st.success("✅ Módulo `visualizer.py` importado correctamente.")
except Exception as e:
    st.error(f"❌ ERROR al importar `visualizer.py`: {e}")

try:
    from modules.reporter import generate_pdf_report
    st.success("✅ Módulo `reporter.py` importado correctamente.")
except Exception as e:
    st.error(f"❌ ERROR al importar `reporter.py`: {e}")
    
try:
    from modules.analysis import calculate_monthly_anomalies
    st.success("✅ Módulo `analysis.py` importado correctamente.")
except Exception as e:
    st.error(f"❌ ERROR al importar `analysis.py`: {e}")

st.info("Si todos los módulos se importaron correctamente, el problema podría estar en la lógica principal de tu app.py original.")
