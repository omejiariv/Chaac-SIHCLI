# modules/sidebar.py

import dash_bootstrap_components as dbc
from dash import html, dcc

def create_sidebar():
    """
    Crea el layout de la barra lateral con todos sus controles.
    """
    return html.Div(
        [
            html.H2("Panel de Control", className="display-6"),
            html.Hr(),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            dcc.Upload(
                                id='upload-mapa',
                                children=html.Div(['Arrastra o ', html.A('Selecciona el archivo de estaciones')]),
                                style={'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'padding': '10px'},
                            ),
                            html.Br(),
                            dcc.Upload(
                                id='upload-precip',
                                children=html.Div(['Arrastra o ', html.A('Selecciona el archivo de precipitación')]),
                                style={'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'padding': '10px'},
                            ),
                            # Puedes añadir el de shapefiles después si lo deseas
                        ],
                        title="Cargar Archivos",
                        item_id="accordion-item-1"
                    ),
                    dbc.AccordionItem(
                        [
                            html.P("Rango de Años:", className="lead"),
                            dcc.RangeSlider(id='year-slider', min=1970, max=2021, step=1, value=[1980, 2010], marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Hr(),
                            html.P("Meses:", className="lead"),
                            dcc.Dropdown(
                                id='month-dropdown',
                                options=[
                                    {'label': 'Enero', 'value': 1}, {'label': 'Febrero', 'value': 2},
                                    {'label': 'Marzo', 'value': 3}, {'label': 'Abril', 'value': 4},
                                    {'label': 'Mayo', 'value': 5}, {'label': 'Junio', 'value': 6},
                                    {'label': 'Julio', 'value': 7}, {'label': 'Agosto', 'value': 8},
                                    {'label': 'Septiembre', 'value': 9}, {'label': 'Octubre', 'value': 10},
                                    {'label': 'Noviembre', 'value': 11}, {'label': 'Diciembre', 'value': 12},
                                ],
                                value=[1,2,3,4,5,6,7,8,9,10,11,12], # Por defecto todos
                                multi=True
                            ),
                            html.Hr(),
                            html.P("Estaciones:", className="lead"),
                            dcc.Dropdown(id='station-dropdown', multi=True, placeholder="Selecciona estaciones...")
                        ],
                        title="Selección de Período y Estaciones",
                        item_id="accordion-item-2"
                    ),
                ],
                start_collapsed=False,
                always_open=True
            ),
        ],
        style={'padding': '2rem 1rem', 'background-color': '#f8f9fa'}
    )
