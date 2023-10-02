from dash import Dash, Input, Output, State, ALL, dcc, html, callback_context
from madina.zonal import Zonal
import dash_bootstrap_components as dbc
import dash_deck
import os
import json

INITIAL_CITY = 'Somerville'
CURRENT_CITY = INITIAL_CITY
city = Zonal()

dash_app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=False,
    title='Urban Network Analysis',
    prevent_initial_callbacks=True
)

def create_layer_table():
    print (f"create_layer_table is called")
    layer_rows = []
    for layer_name in city.layers.layers:
        layer_rows.append(
            html.Tr([
                html.Td(layer_name),
                html.Td(
                    dbc.Switch(
                        id={"type": "layer_show", "index": layer_name},
                        value=True if city.layers[layer_name].show else False,
                    )
                ),
                html.Td(
                    dbc.Input(
                        type="color",
                        id={"type": "layer_color", "index": layer_name},
                        value="#000000",
                        style={"width": 25, "height": 25, 'padding': 0},
                    )
                )
            ])
        )

    layer_table = dbc.Table(
        [
            #html.Thead(html.Tr([html.Th("Layer"), html.Th("Weight"),  html.Th("Show")])),
            html.Tbody(layer_rows)
        ],
        bordered=False,
        id='layer_table'
    )
    return layer_table

def create_city_drop_down(picked_city_name):
    print (f"create_city_drop_down is called")

    create_city_form = dbc.Form(
        dbc.Row(
            [
                dbc.Col(
                    dbc.Input(type="text", placeholder="City Name", id='create_city_name'),
                    className="me-3",
                ),
                dbc.Col(
                    dbc.Button("Create New City", id='create_city_button', color="primary"),
                    width="auto"
                ),
            ],
            className="g-2",
        )
    )

    city_menu_items = []
    for city_name in os.listdir(os.path.join("workflows", "Cities")):
        city_menu_items.append(
            dbc.DropdownMenuItem(
                city_name,
                active=True if city_name == picked_city_name else False,
                id={"type": "city_menu_item", "index": city_name}
            )
        )

    city_drop_down = dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem("Cities", header=True),
        ] + city_menu_items + [
            dbc.DropdownMenuItem(divider=True),
            create_city_form,
        ],
        label=f"City: {picked_city_name}",
        id='city_drop_down'
    )
    return city_drop_down

def create_add_layer_form(picked_city_name):
    print (f"create_add_layer_form is called")

    city_files = os.listdir(os.path.join("workflows", "Cities", picked_city_name, 'Data'))

    file_menu_items = [dbc.DropdownMenuItem(file_name, id={"type": "file_menu_item", "index": file_name}) for file_name in city_files]
    file_drop_down = dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem("Files", header=True),
        ] + file_menu_items + [
            dbc.DropdownMenuItem(divider=True),
            dbc.Button("Upload FIle", id='upload_button', color="primary"),
        ],
        label="File",
        id='file_drop_down'
    )

    add_layer_form = dbc.Form(
        [
            html.H3('Add Layer'),
            dbc.Row(
                [
                    dbc.Col(file_drop_down, width="auto"),
                    dbc.Col(
                        dbc.Input(type="text", placeholder="Layer Name", id='create_layer_name'),
                        width="auto"
                    ),
                    dbc.Col(
                        dbc.Button("Add", id='create_layer_button', color="primary"),
                        width="auto"
                    ),
                ],
                className="g-2",
            )
        ],
        id='add_layer_form'
    )
    return add_layer_form

def create_network_map():
    print (f"create_network_map is called, CURRENT LAYERS {city.layers.layers = }")
    deck_component = dash_deck.DeckGL(
        city.create_map().to_json(),
        enableEvents=False,
        tooltip=False,
        style={
            "width": "61vw",
            "height": "80vh",
            "position": "relative"
        },
        id="network_map",
    )
    return deck_component
    
def create_network_form():
    layer_pick = dbc.Row(
    [
        dbc.Label("Layer", html_for="network_layer", width=3),
        dbc.Col(
            dcc.Dropdown(
                id="network_layer",
                options=[{"label": layer_name, "value": layer_name} for layer_name in city.layers.layers],
                value='streets' if 'streets' in city.layers.layers else None
            ),
            width=9,
        )
    ],
    className="mb-3",
    )

    weight_pick = dbc.Row(
    [
        dbc.Label("Weight", html_for="network_weight", width=3),
        dbc.Col(
            dcc.Dropdown(
                id="network_weight",
                options=[{"label": 'Geometric Distance', "value": 'Geometric Distance'}],#+[{"label": column, "value": layer_name} for layer_name in city.layers.layers],
                value='Geometric Distance'
            ),
            width=9,
        )
    ],
    className="mb-3",
    )

    create_network_button = dbc.Button("Create Network", color="primary", id='create_network_button')

    return dbc.Form([layer_pick, weight_pick, create_network_button])

def create_origin_form():
    layer_pick = dbc.Row([
        dbc.Label("Layer", html_for="origin_layer", width=3),
        dbc.Col(
            dcc.Dropdown(
                id="origin_layer",
                options=[{"label": layer_name, "value": layer_name} for layer_name in city.layers.layers],
            ),
            width=9,
        )
    ])

    weight_pick = dbc.Row( [
        dbc.Label("Weight", html_for="origin_weight", width=3),
        dbc.Col(
            dcc.Dropdown(
                id="origin_weight",
                options=[{"label": 'Count', "value": 'Count'}],
                value='Count'
            ),
            width=9,
        )
    ])
    create_origin_button = dbc.Button("Add Origin", color="primary", id='add_origin_button')
    return dbc.Form([layer_pick, weight_pick, create_origin_button])

def create_destination_form():
    '''
    
    '''
    layer_pick = dbc.Row([
        dbc.Label("Layer", html_for="destination_layer", width=3),
        dbc.Col(
            dcc.Dropdown(
                id="destination_layer",
                options=[{"label": layer_name, "value": layer_name} for layer_name in city.layers.layers],
            ),
            width=9,
        )
    ])

    weight_pick = dbc.Row( [
        dbc.Label("Weight", html_for="destination_weight", width=3),
        dbc.Col(
            dcc.Dropdown(
                id="destination_weight",
                options=[{"label": 'Count', "value": 'Count'}],
                value='Count'
            ),
            width=9,
        )
    ])
    create_destination_button = dbc.Button("Add Destination", color="primary", id='add_destination_button')
    return dbc.Form([layer_pick, weight_pick, create_destination_button])


def create_reach_form():
    return "reach form...."


def create_service_area_form():
    return "service_area..."
def creat_flow_form():
    return "flow form...."


@dash_app.callback(
    Output("create_network_button", "children"),
    Output("create_network_button", "color"),
    State('network_layer', 'value'),
    State('network_weight', 'value'),
    Input('create_network_button', "n_clicks"),
    prevent_initial_call=True,
)
def button_creates_a_network(layer_name, weight_attribute, n_clicks):
    ## TODO: Do some error checking... Alert if layer_name not in layers or weight_attribute not in columns

    city.create_street_network(
        source_layer=layer_name,
        weight_attribute=None if weight_attribute == "Geometric Distance" else weight_attribute
    )
    return "Network Created", 'success'

@dash_app.callback(
    Output("add_origin_button", "children"),
    Output("add_origin_button", "color"),
    State('origin_layer', 'value'),
    State('origin_weight', 'value'),
    Input('add_origin_button', "n_clicks"),
    prevent_initial_call=True,
)
def button_adds_origin(layer_name, weight_attribute, n_clicks):
    city.insert_node(
        layer_name=layer_name,
        label='origin',
        weight_attribute=None if weight_attribute == "Count" else weight_attribute
    )
    return "Origin Inserted", 'success'

@dash_app.callback(
    Output("add_destination_button", "children"),
    Output("add_destination_button", "color"),
    State('destination_layer', 'value'),
    State('destination_weight', 'value'),   
    Input('add_destination_button', "n_clicks"),
    prevent_initial_call=True,
)
def button_adds_destination(layer_name, weight_attribute, n_clicks):
    city.insert_node(
        layer_name=layer_name,
        label='destination',
        weight_attribute=None if weight_attribute == "Count" else weight_attribute
    )
    return "Destination Inserted", 'success'

@dash_app.callback(
    Output("network_weight", "options"),
    Input('network_layer', "value"),
    prevent_initial_call=True,
)
def update_layer_weight_dropdown(layer_name):
    return [{"label": 'Geometric Distance', "value": 'Geometric Distance'}]+[{"label": column, "value": column} for column in city.layers[layer_name].gdf.columns]

def madina_layout(picked_city_name):
    layers_card = dbc.Spinner(     
        dbc.Card(
            [
                dbc.CardHeader(create_city_drop_down(picked_city_name), id='city_drop_down_card'),
                dbc.CardBody(create_layer_table(), id='layer_table_card'),
                dbc.CardFooter(create_add_layer_form(picked_city_name), id='add_layer_form_card'),
            ],
            color="light",
            inverse=True
        ),
        color='light'
    )

    map_card = dbc.Spinner(     
        dbc.Card("Create a Layer to see a Map", color="light", inverse=True, id='map_card'),
        color='light'
    )
    network_card = dbc.Card(
        [
            dbc.CardHeader("Network"),
            dbc.CardBody(create_network_form(), id='network_form_card'),
            #dbc.CardFooter([html.H3('Add Layer'), add_layer_form])
        ],
        color="light",
        inverse=True
    )

    origin_card = dbc.Card(
        [
            dbc.CardHeader("Origns"),
            dbc.CardBody(create_origin_form(), id='origin_form_card'),
            #dbc.CardFooter([html.H3('Add Layer'), add_layer_form])
        ],
        color="light",
        inverse=True
    )

    destination_card = dbc.Card(
        [
            dbc.CardHeader("Destinations"),
            dbc.CardBody(create_destination_form(), id='destination_form_card'),
            #dbc.CardFooter([html.H3('Add Layer'), add_layer_form])
        ],
        color="light",
        inverse=True
    )

    ## wrapping cards in spinners
    network_card = dbc.Spinner(network_card, color='light')
    origin_card = dbc.Spinner(origin_card, color='light')
    destination_card = dbc.Spinner(destination_card, color='light')
           
    reach_tab = dbc.Card(
        dbc.CardBody(create_reach_form(), id='reach_form_card')
    )

    service_area_tab = dbc.Card(
        dbc.CardBody(create_service_area_form(), id='service_area_form_card')
    )

    flow_tab = dbc.Card(
        dbc.CardBody(creat_flow_form(), id='flow_form_card')
    )


    tool_tabs = dbc.Tabs(
        [
            dbc.Tab(reach_tab, label="Access - Reach"),
            dbc.Tab(service_area_tab, label="Catchment"),
            dbc.Tab(flow_tab, label="Flow"),
        ]
    )


    tool_card = dbc.Card(
        [
            dbc.CardHeader("Urban Network Analysis"),
            dbc.CardBody(tool_tabs),
            #dbc.CardFooter([html.H3('Add Layer'), add_layer_form])
        ],
        color="light",
        inverse=True
    )

    layout = html.Div(
        children=[
            dbc.Row([
                dbc.Col(layers_card, width=4),
                dbc.Col(map_card, width=8),
            ], 
            className="h-750"
            ),
            dbc.Row([
                dbc.Col(network_card),
                dbc.Col(origin_card),
                dbc.Col(destination_card),
            ], 
            className="h-250"
            ),
            dbc.Row(dbc.Col(tool_card))
        ],
        id='city_layers'
    )
    return layout

@dash_app.callback(
    Output('file_drop_down', 'label'),
    Input({"type": "file_menu_item", "index": ALL}, "n_clicks"),
)
def file_name_picked(values):
    button_id = callback_context.triggered[0]["prop_id"].replace(".n_clicks", "", 1)
    file_name = json.loads(button_id)['index']
    return file_name

@dash_app.callback(
    Output("layer_table_card", "children"),
    Output('city_drop_down_card', 'children'), 
    Output('add_layer_form_card', 'children'), 
    Output('map_card', 'children'), 
    Output('network_form_card', 'children'), 
    Output('origin_form_card', 'children'), 
    Output('destination_form_card', 'children'), 
    State('file_drop_down', 'label'), 
    State('create_layer_name', 'value'), 
    Input('create_layer_button', "n_clicks"),
    Input({"type": "city_menu_item", "index": ALL}, "n_clicks"),
    prevent_initial_call=False,
)
def create_layer(file_name, layer_name, create_layer_n_clicks, city_menu_item_n_clicks):
    global CURRENT_CITY
    # an initial call would give:
    # callback_context.triggered = [{'prop_id': '.', 'value': None}]

    button_id = callback_context.triggered[0]["prop_id"].replace(".n_clicks", "", 1)
    print (f'{callback_context.triggered = }')
    print (f"creating layer, {button_id = } {file_name = }, {create_layer_n_clicks = }")


    if callback_context.triggered[0]["prop_id"] == '.':
        print ("Initial Callback, filling in current state of city..")
    elif button_id == "create_layer_button":
        city.load_layer(
            layer_name=layer_name,
            file_path=os.path.join('workflows', "Cities", CURRENT_CITY, "Data", file_name)
        )
    else: ## indexed input, parse source and index...
        # since the property ID of menuItems is a dict, we parse it here, and get the 'index' of the button, which is the city name
        CURRENT_CITY = json.loads(button_id)['index']


    return create_layer_table(), create_city_drop_down(CURRENT_CITY), create_add_layer_form(CURRENT_CITY), create_network_map() if len (city.layers.layers) > 0 else dbc.Alert("Add a layer to see a Map", color='primary'), create_network_form(), create_origin_form(), create_destination_form()

dash_app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=True),
        html.Div(
            children=['Header Content'],
            id="header_container"
        ),
        madina_layout(INITIAL_CITY),
        html.Div(
            children=['Footer COntent'],
            id='footer_container'
        )
    ],
    id='app-container',
    style={"width": "1200px", "margin": "auto"},
)

dash_app.server.run(host="0.0.0.0", port=80, debug=True, use_reloader=False)
