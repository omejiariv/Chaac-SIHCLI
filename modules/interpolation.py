import streamlit as st
import pandas as pd
import numpy as np
import gstools as gs
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error
from modules.config import Config

def interpolate_idw(lons, lats, vals, grid_lon, grid_lat, power=2):
    """Realiza una interpolación por el método IDW."""
    nx, ny = len(grid_lon), len(grid_lat)
    grid_z = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            x, y = grid_lon[i], grid_lat[j]
            distances = np.sqrt((lons - x)**2 + (lats - y)**2)
            if np.any(distances < 1e-10):
                grid_z[j, i] = vals[np.argmin(distances)]
                continue
            weights = 1.0 / (distances**power)
            weighted_sum = np.sum(weights * vals)
            total_weight = np.sum(weights)
            if total_weight > 0:
                grid_z[j, i] = weighted_sum / total_weight
            else:
                grid_z[j, i] = np.nan
    return grid_z.T

# -----------------------------------------------------------------------------
# NUEVA FUNCIÓN INTERNA PARA REUTILIZAR LA LÓGICA DE VALIDACIÓN CRUZADA
# -----------------------------------------------------------------------------
def _perform_loocv(method, lons, lats, vals, elevs=None):
    """
    Función auxiliar interna que realiza la validación cruzada (LOOCV).
    """
    if len(vals) <= 1:
        return {'RMSE': np.nan, 'MAE': np.nan}

    loo = LeaveOneOut()
    true_values, predicted_values = [], []

    for train_index, test_index in loo.split(lons):
        lons_train, lons_test = lons[train_index], lons[test_index]
        lats_train, lats_test = lats[train_index], lats[test_index]
        vals_train, vals_test = vals[train_index], vals[test_index]
        
        try:
            z_pred = None
            if method == "Kriging Ordinario" and len(lons_train) > 0:
                model_cv = gs.Spherical(dim=2)
                bin_center_cv, gamma_cv = gs.vario_estimate((lons_train, lats_train), vals_train)
                model_cv.fit_variogram(bin_center_cv, gamma_cv, nugget=True)
                krig_cv = gs.krige.Ordinary(model_cv, (lons_train, lats_train), vals_train)
                z_pred, _ = krig_cv((lons_test[0], lats_test[0]))
            
            elif method == "Kriging con Deriva Externa (KED)" and elevs is not None and len(lons_train) > 0:
                elevs_train, elevs_test = elevs[train_index], elevs[test_index]
                model_cv = gs.Spherical(dim=2)
                bin_center_cv, gamma_cv = gs.vario_estimate((lons_train, lats_train), vals_train)
                model_cv.fit_variogram(bin_center_cv, gamma_cv, nugget=True)
                krig_cv = gs.krige.ExtDrift(model_cv, (lons_train, lats_train), vals_train, drift_src=elevs_train)
                z_pred, _ = krig_cv((lons_test[0], lats_test[0]), drift_tgt=elevs_test)

            elif method == "IDW":
                z_pred = interpolate_idw(lons_train, lats_train, vals_train, lons_test, lats_test)[0, 0]
            
            elif method == "Spline (Thin Plate)" and len(lons_train) > 2:
                rbf_cv = Rbf(lons_train, lats_train, vals_train, function='thin_plate')
                z_pred = rbf_cv(lons_test, lats_test)[0]

            if z_pred is not None:
                predicted_values.append(z_pred)
                true_values.append(vals_test[0])
        except Exception:
            continue
            
    if true_values and predicted_values:
        rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
        mae = mean_absolute_error(true_values, predicted_values)
        return {'RMSE': rmse, 'MAE': mae}
    else:
        return {'RMSE': np.nan, 'MAE': np.nan}

# -----------------------------------------------------------------------------
# NUEVA FUNCIÓN PÚBLICA PARA LA PESTAÑA DE VALIDACIÓN
# -----------------------------------------------------------------------------
@st.cache_data
def perform_loocv_for_year(year, method, gdf_metadata, df_anual_non_na):
    """
    Realiza una Validación Cruzada Dejando Uno Afuera (LOOCV) para un año y método dados.
    Devuelve las métricas de error (RMSE y MAE).
    """
    df_year = pd.merge(
        df_anual_non_na[df_anual_non_na[Config.YEAR_COL] == year],
        gdf_metadata,
        on=Config.STATION_NAME_COL
    )
    
    clean_cols = [Config.LONGITUDE_COL, Config.LATITUDE_COL, Config.PRECIPITATION_COL]
    if method == "Kriging con Deriva Externa (KED)" and Config.ELEVATION_COL in df_year.columns:
        clean_cols.append(Config.ELEVATION_COL)

    df_clean = df_year.dropna(subset=clean_cols).copy()
    df_clean = df_clean[np.isfinite(df_clean[clean_cols]).all(axis=1)]
    df_clean = df_clean.drop_duplicates(subset=[Config.LONGITUDE_COL, Config.LATITUDE_COL])

    if len(df_clean) < 4:
        return {'RMSE': np.nan, 'MAE': np.nan}

    lons = df_clean[Config.LONGITUDE_COL].values
    lats = df_clean[Config.LATITUDE_COL].values
    vals = df_clean[Config.PRECIPITATION_COL].values
    elevs = df_clean[Config.ELEVATION_COL].values if Config.ELEVATION_COL in df_clean else None
    
    return _perform_loocv(method, lons, lats, vals, elevs)

@st.cache_data
def perform_loocv_for_all_methods(_year, _gdf_metadata, _df_anual_non_na):
    """Ejecuta LOOCV para todos los métodos de interpolación para un año dado."""
    methods = ["Kriging Ordinario", "IDW", "Spline (Thin Plate)"]
    if Config.ELEVATION_COL in _gdf_metadata.columns:
        methods.insert(1, "Kriging con Deriva Externa (KED)")
    
    results = []
    for method in methods:
        metrics = perform_loocv_for_year(_year, method, _gdf_metadata, _df_anual_non_na)
        if metrics:
            results.append({
                "Método": method,
                "Año": _year,
                "RMSE": metrics.get('RMSE'),
                "MAE": metrics.get('MAE')
            })
    return pd.DataFrame(results)
# -----------------------------------------------------------------------------
# FUNCIÓN ORIGINAL, AHORA ACTUALIZADA PARA USAR LA FUNCIÓN AUXILIAR
# -----------------------------------------------------------------------------
@st.cache_data
def create_interpolation_surface(year, method, variogram_model, gdf_bounds, gdf_metadata, df_anual_non_na):
    """Crea una superficie de interpolación y calcula el error RMSE."""
    fig_var = None
    error_msg = None

    df_year = pd.merge(
        df_anual_non_na[df_anual_non_na[Config.YEAR_COL] == year],
        gdf_metadata,
        on=Config.STATION_NAME_COL
    )
    
    clean_cols = [Config.LONGITUDE_COL, Config.LATITUDE_COL, Config.PRECIPITATION_COL]
    if method == "Kriging con Deriva Externa (KED)" and Config.ELEVATION_COL in df_year.columns:
        clean_cols.append(Config.ELEVATION_COL)

    df_clean = df_year.dropna(subset=clean_cols).copy()
    df_clean = df_clean[np.isfinite(df_clean[clean_cols]).all(axis=1)]
    df_clean = df_clean.drop_duplicates(subset=[Config.LONGITUDE_COL, Config.LATITUDE_COL])
    
    if len(df_clean) < 4:
        error_msg = f"Se necesitan al menos 4 estaciones con datos para el año {year} para interpolar."
        fig = go.Figure().update_layout(title=error_msg, xaxis_visible=False, yaxis_visible=False)
        return fig, None, error_msg

    lons = df_clean[Config.LONGITUDE_COL].values
    lats = df_clean[Config.LATITUDE_COL].values
    vals = df_clean[Config.PRECIPITATION_COL].values
    elevs = df_clean[Config.ELEVATION_COL].values if Config.ELEVATION_COL in df_clean else None

    # Llama a la función auxiliar para obtener métricas
    metrics = _perform_loocv(method, lons, lats, vals, elevs)
    rmse = metrics.get('RMSE')

    grid_lon = np.linspace(gdf_bounds[0] - 0.1, gdf_bounds[2] + 0.1, 100)
    grid_lat = np.linspace(gdf_bounds[1] - 0.1, gdf_bounds[3] + 0.1, 100)
    z_grid, fig_variogram, error_message = None, None, None

    try:
        if method in ["Kriging Ordinario", "Kriging con Deriva Externa (KED)"]:
            model_map = {'gaussian': gs.Gaussian(dim=2), 'exponential': gs.Exponential(dim=2), 'spherical': gs.Spherical(dim=2), 'linear': gs.Linear(dim=2)}
            model = model_map.get(variogram_model, gs.Spherical(dim=2))
            bin_center, gamma = gs.vario_estimate((lons, lats), vals)
            model.fit_variogram(bin_center, gamma, nugget=True)
            
            fig_variogram, ax = plt.subplots()
            ax.plot(bin_center, gamma, 'o', label='Experimental')
            model.plot(ax=ax, label='Modelo Ajustado')
            ax.set_xlabel('Distancia'); ax.set_ylabel('Semivarianza')
            ax.set_title(f'Variograma para {year}'); ax.legend()

            if method == "Kriging Ordinario":
                krig = gs.krige.Ordinary(model, (lons, lats), vals)
                z_grid, _ = krig.structured([grid_lon, grid_lat])
            else: # KED
                # Necesitamos crear una grilla de elevación para la predicción
                rbf_elev = Rbf(lons, lats, elevs, function='thin_plate')
                grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
                drift_grid = rbf_elev(grid_x, grid_y)
                krig = gs.krige.ExtDrift(model, (lons, lats), vals, drift_src=elevs)
                z_grid, _ = krig.structured([grid_lon, grid_lat], drift_tgt=drift_grid.T)

        elif method == "IDW":
            z_grid = interpolate_idw(lons, lats, vals, grid_lon, grid_lat)
            
        elif method == "Spline (Thin Plate)":
            rbf = Rbf(lons, lats, vals, function='thin_plate')
            grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
            z_grid = rbf(grid_x, grid_y)
            
    except Exception as e:
        error_message = f"Error al calcular {method}: {e}"
        fig = go.Figure().update_layout(title=error_message, xaxis_visible=False, yaxis_visible=False)
        return fig, None, error_message

    if z_grid is not None:
        fig = go.Figure(data=go.Contour(
            z=z_grid.T, x=grid_lon, y=grid_lat,
            colorscale=px.colors.sequential.YlGnBu,
            colorbar_title='Precipitación (mm)',
            contours=dict(showlabels=True, labelfont=dict(size=10, color='white'), labelformat=".0f")
        ))
        
        fig.add_trace(go.Scatter(
            x=lons, y=lats, mode='markers', marker=dict(color='red', size=5, line=dict(width=1, color='black')),
            name='Estaciones',
            hoverinfo='text',
            text=[f"<b>{row[Config.STATION_NAME_COL]}</b><br>" +
                  f"Municipio: {row[Config.MUNICIPALITY_COL]}<br>" +
                  f"Altitud: {row[Config.ALTITUDE_COL]} m<br>" +
                  f"Precipitación: {row[Config.PRECIPITATION_COL]:.0f} mm"
                  for _, row in df_clean.iterrows()]
        ))
        
        if rmse is not None:
            fig.add_annotation(
                x=0.01, y=0.99, xref="paper", yref="paper",
                text=f"<b>RMSE: {rmse:.1f} mm</b>", align='left',
                showarrow=False, font=dict(size=12, color="black"),
                bgcolor="rgba(255, 255, 255, 0.7)", bordercolor="black", borderwidth=1
            )
            
        fig.update_layout(
            title=f"Precipitación en {year} ({method})",
            xaxis_title="Longitud", yaxis_title="Latitud", height=600,
            legend=dict(x=0.01, y=0.01, bgcolor="rgba(0,0,0,0)")
        )
        return fig, fig_variogram, None

    return go.Figure().update_layout(title="Error: Método no implementado"), None, "Método no implementado"
