import pandas as pd
import requests
import os
import time
import numpy as np

def descargar_y_actualizar_ideam():
    print("Iniciando la descarga automatizada desde el portal de Datos Abiertos...")
    inicio_total = time.time()
    
    DATASET_ID = "s54a-sgyg"
    URL_API = f"https://datos.gov.co/resource/{DATASET_ID}.json"
    ARCHIVO_SALIDA = "data/lluvia_mensual_consolidado.csv"
    
    departamentos = [
        "ANTIOQUIA", "CHOCO", "CHOCÓ", "CORDOBA", "CÓRDOBA", 
        "CALDAS", "TOLIMA", "BOYACA", "BOYACÁ", "SANTANDER", "RISARALDA"
    ]
    
    deptos_query = ", ".join([f"'{d}'" for d in departamentos])
    select_clause = "codigoestacion, fechaobservacion, valorobservado"
    where_clause = f"upper(departamento) IN ({deptos_query}) AND unidadmedida = 'mm'"
    
    limit = 100000  
    offset = 0
    resultados_parciales = []
    
    while True:
        query_url = f"{URL_API}?$select={select_clause}&$where={where_clause}&$limit={limit}&$offset={offset}"
        
        try:
            response = requests.get(query_url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error al conectar con la API del IDEAM: {e}")
            break
            
        if not data:
            break 
            
        chunk_df = pd.DataFrame(data)
        
        # Procesamiento de fechas
        chunk_df['fechaobservacion'] = pd.to_datetime(chunk_df['fechaobservacion'], errors='coerce')
        chunk_df = chunk_df.dropna(subset=['fechaobservacion'])
        chunk_df['periodo_mensual'] = chunk_df['fechaobservacion'].dt.to_period('M')
        
        # --- CONTROL DE CALIDAD HIDROCLIMÁTICO ---
        chunk_df['valorobservado'] = pd.to_numeric(chunk_df['valorobservado'], errors='coerce')
        
        # 1. Eliminar números enormes atípicos (fallas de sensor > 150 mm/hora) y negativos
        chunk_df.loc[(chunk_df['valorobservado'] > 150) | (chunk_df['valorobservado'] < 0), 'valorobservado'] = np.nan
        
        # Conteo de registros totales por bloque para control de calidad institucional
        chunk_df['conteo_horas'] = 1
        
        # Agrupamos temporalmente sumando la lluvia y contando cuántas horas válidas aportó el bloque
        chunk_agrupado = chunk_df.groupby(['codigoestacion', 'periodo_mensual']).agg(
            lluvia_sum=('valorobservado', 'sum'),
            horas_validas=('valorobservado', 'count'),
            horas_totales=('conteo_horas', 'sum')
        ).reset_index()
        
        resultados_parciales.append(chunk_agrupado)
        offset += limit
        print(f"📦 Descargados y procesados {offset} registros históricos...")
        time.sleep(0.2)

    if resultados_parciales:
        print("🔄 Consolidando y aplicando umbral de representatividad mensual...")
        df_total = pd.concat(resultados_parciales)
        
        # Volvemos a agrupar por si el mes quedó dividido entre dos bloques de descarga
        df_consolidado = df_total.groupby(['codigoestacion', 'periodo_mensual']).agg(
            precipitacion=('lluvia_sum', 'sum'),
            total_horas_validas=('horas_validas', 'sum'),
            total_horas_medidas=('horas_totales', 'sum')
        ).reset_index()
        
        # 2. Filtro de representatividad: Si el mes no tiene al menos el 80% de sus horas esperadas, 
        # lo descartamos para evitar subestimar la lluvia mensual con ceros vacíos.
        # Un mes promedio tiene ~720 horas (24h * 30d), exigimos mínimo 576 horas con datos.
        df_consolidado = df_consolidado[df_consolidado['total_horas_validas'] >= 576]
        
        # Convertir periodo a texto para compatibilidad con el dashboard
        df_consolidado['periodo_mensual'] = df_consolidado['periodo_mensual'].astype(str)
        
        # Seleccionar y renombrar columnas finales para la App
        df_final = df_consolidado[['codigoestacion', 'periodo_mensual', 'precipitacion']].copy()
        df_final.columns = ['codigo_estacion', 'periodo_mensual', 'precipitacion']
        
        # Guardar archivo optimizado y limpio
        os.makedirs("data", exist_ok=True)
        df_final.to_csv(ARCHIVO_SALIDA, index=False, sep=',')
        
        print(f"🏁 ¡Pipeline finalizado! Archivo depurado guardado en: {ARCHIVO_SALIDA}")
    else:
        print("❌ No se recuperaron datos de la API.")

if __name__ == "__main__":
    descargar_y_actualizar_ideam()