# actualizar_datos.py

import pandas as pd
import requests
import os
import time
import numpy as np

def descargar_y_actualizar_ideam():
    print("Iniciando la descarga automatizada con enriquecimiento geografico...")
    inicio_total = time.time()
    
    DATASET_ID = "s54a-sgyg"
    URL_API = f"https://datos.gov.co/resource/{DATASET_ID}.json"
    ARCHIVO_SALIDA = "data/lluvia_mensual_consolidado.csv"
    ARCHIVO_ESTACIONES_NUEVAS = "data/estaciones_automaticas_catalogo.csv"
    
    departamentos = [
        "ANTIOQUIA", "CHOCO", "CHOCÓ", "CORDOBA", "CÓRDOBA", 'SUCRE', 'QUINDÍO', 
        "CALDAS", "TOLIMA", "BOYACA", "BOYACÁ", "SANTANDER", "RISARALDA", 'CUNDINAMARCA'
    ]
    
    deptos_query = ", ".join([f"'{d}'" for d in departamentos])
    
    # IMPORTANTE: Ahora si traemos las columnas geograficas esenciales para la App
    select_clause = "codigoestacion, nombreestacion, departamento, municipio, latitud, longitud, fechaobservacion, valorobservado"
    where_clause = f"upper(departamento) IN ({deptos_query}) AND unidadmedida = 'mm'"
    
    limit = 100000  
    offset = 0
    resultados_parciales = []
    catalogo_estaciones = {} # Diccionario para almacenar la ubicacion unica de cada estacion
    
    while True:
        query_url = f"{URL_API}?$select={select_clause}&$where={where_clause}&$limit={limit}&$offset={offset}"
        
        try:
            response = requests.get(query_url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error al conectar con la API: {e}")
            break
            
        if not data:
            break 
            
        chunk_df = pd.DataFrame(data)
        
        # Procesamiento y limpieza del bloque
        chunk_df['fechaobservacion'] = pd.to_datetime(chunk_df['fechaobservacion'], errors='coerce')
        chunk_df = chunk_df.dropna(subset=['fechaobservacion'])
        chunk_df['periodo_mensual'] = chunk_df['fechaobservacion'].dt.to_period('M')
        chunk_df['valorobservado'] = pd.to_numeric(chunk_df['valorobservado'], errors='coerce')
        
        # Control de calidad (Atípicos y negativos)
        chunk_df.loc[(chunk_df['valorobservado'] > 150) | (chunk_df['valorobservado'] < 0), 'valorobservado'] = np.nan
        chunk_df['conteo_horas'] = 1
        
        # Lógica de Captura del Catálogo Geográfico
        for _, fila in chunk_df.dropna(subset=['latitud', 'longitud']).iterrows():
            cod = str(fila['codigoestacion'])
            if cod not in catalogo_estaciones:
                catalogo_estaciones[cod] = {
                    'codigo_estacion': cod,
                    'nombre_estacion': str(fila['nombreestacion']).strip(),
                    'municipio': str(fila['municipio']).strip().title(),
                    'departamento': str(fila['departamento']).strip().upper(),
                    'latitud': fila['latitud'],
                    'longitud': fila['longitud']
                }
        
        # Agrupación por bloque
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
        print("Consolidando series, aplicando purga climatica y exportando catalogos...")
        df_total = pd.concat(resultados_parciales)
        
        df_consolidado = df_total.groupby(['codigoestacion', 'periodo_mensual']).agg(
            precipitacion=('lluvia_sum', 'sum'),
            total_horas_validas=('horas_validas', 'sum')
        ).reset_index()
        
        # 1. Filtro de representatividad horaria (Umbral ajustado para estaciones nuevas)
        df_consolidado = df_consolidado[df_consolidado['total_horas_validas'] >= 400]
        
        # 2. PURGA CLIMÁTICA DEFINTIVA A NIVEL MENSUAL (Extermina picos corruptos y ceros falsos)
        df_consolidado = df_consolidado[
            (df_consolidado['precipitacion'] > 0.0) & 
            (df_consolidado['precipitacion'] <= 1200)
        ]
        
        df_consolidado['periodo_mensual'] = df_consolidado['periodo_mensual'].astype(str)
        
        # Preparar DataFrame final de lluvias para la App
        df_final = df_consolidado[['codigoestacion', 'periodo_mensual', 'precipitacion']].copy()
        df_final.columns = ['codigo_estacion', 'periodo_mensual', 'precipitacion']
        
        os.makedirs("data", exist_ok=True)
        
        # Guardar 1: El archivo indexado de lluvia purgado y limpio
        df_final.to_csv(ARCHIVO_SALIDA, index=False, sep=',')
        
        # Guardar 2: El catálogo geográfico estructurado para las 429 estaciones automáticas
        if catalogo_estaciones:
            df_cat = pd.DataFrame(catalogo_estaciones.values())
            df_cat.to_csv(ARCHIVO_ESTACIONES_NUEVAS, index=False, sep=',')
            print(f"Catalogo geografico actualizado con exito ({len(df_cat)} estaciones).")
        
        print(f"Pipeline finalizado con exito. Archivos guardados en la carpeta data/.")
    else:
        print("Error: No se recuperaron datos validos de la API.")

if __name__ == "__main__":
    descargar_y_actualizar_ideam()
