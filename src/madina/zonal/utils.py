import time
import random
import warnings

import numpy as np
import pydeck as pdk

from geopandas import GeoDataFrame
from pydeck.types import String

DEFAULT_COLORS = {
    "streets": [150, 150, 150],
    "blocks": [100, 100, 100],
    "parcels": [50, 50, 50],
    "network_edges": {
        "__attribute_name__": "type",
        "street": [155, 155, 155],
        "project_line": [0, 0, 150],
        "__other__": [0, 0, 0]
    },
    "network_nodes": {
        "__attribute_name__": "type",
        "street_node": [0, 255, 255],
        "project_node": [0, 0, 255],
        "destination": [239,89,128],
        "origin": [86,5,255],
        "__other__": [0, 0, 0]
    }
}


def prepare_geometry(geometry_gdf: GeoDataFrame):
    geometry_gdf = geometry_gdf.copy(deep=True)

    # making sure if geometry contains polygons, they are corrected by using the polygon exterior as a line.
    polygon_idxs = geometry_gdf[geometry_gdf["geometry"].geom_type ==
                                "Polygon"].index
    geometry_gdf.loc[polygon_idxs,
    "geometry"] = geometry_gdf.loc[polygon_idxs, "geometry"].boundary

    # if geometry is multilineString, convert to lineString
    if (geometry_gdf["geometry"].geom_type == 'MultiLineString').all():  #\
            #and (np.array(list(map(len, geometry_gdf["geometry"].values))) == 1).all():
        geometry_gdf["geometry"] = geometry_gdf["geometry"].apply(lambda x: x.geoms[0])


    # deleting any Z coordinate if they exist
    from shapely.ops import transform

    def _to_2d(x, y, z):
        return tuple(filter(None, [x, y]))

    if geometry_gdf.has_z.any():
        geometry_gdf["geometry"] = geometry_gdf["geometry"].apply(
            lambda s: transform(_to_2d, s))
    return geometry_gdf



def color_gdf(gdf, color_by_attribute=None, color_method=None, color=None):
    """Sets geometry color

    :param gdf: GeoDataFrame to be colored.
    :param color_by_attribute: string, attribute name, or column name to visualize geometry by
    :param color_method: string, "single_color" to color all geometry by the same color.
    "categorical" to use distinct color to distinct value, "gradient": to use a gradient of colors for a numeric, scalar attribute.
    :param color: if color method is single color, expects one color. if categorical, expects nothing and would give automatic assignment, or a dict {"val': [0,0,0]}. if color_method is gradient, expects nothing for a default color map, or a color map name

    """
    if color_method is None:
        if color_by_attribute is not None:
            color_method = "categorical"
        else:
            color_method = "single_color"

    if color_by_attribute is None and color is None:
        # if "color_by_attribute" is not given, and its not a default layer, assuming color_method == "single_color"
        # if no color is given, assign random color, else, color=color
        color = [random.random() * 255, random.random() * 255, random.random() * 255]
        color_method = "single_color"
    elif color is None:
        # color by attribute ia given, but no color is given..
        if color_method == "single_color":
            # if color by attribute is given, and color method is single color, this is redundant but just in case:
            color = [random.random() * 255, random.random() * 255, random.random() * 255]
        if color_method == "categorical":
            color = {"__other__": [255, 255, 255]}
            for distinct_value in gdf[color_by_attribute].unique():
                color[distinct_value] = [random.random() * 255, random.random() * 255, random.random() * 255]

    # create color column
    if color_method == "single_color":
        color_column = [color] * len(gdf)
    elif color_method == "categorical":
        # color = {"__other__": [255, 255, 255]}
        color_column = []
        for value in gdf[color_by_attribute]:
            if value in color.keys():
                color_column.append(color[value])
            else:
                color_column.append(color["__other__"])
    elif color_method == "gradient":
        cbc = gdf[color_by_attribute]  # color by column
        nc = 255 * (cbc - cbc.min()) / (cbc.max() - cbc.min())  # normalized column
        color_column = [[255 - v, 0 + v, 0] if not np.isnan(v) else [255, 255, 255] for v in
                        list(nc)]  # convert normalized values to color spectrom.
        # TODO: insert color map options here..
    elif color_method == 'quantile':
        scaled_percentile_rank = 255 * gdf[color_by_attribute].rank(pct=True)
        color_column = [[255.0 - v, 0.0 + v, 0] if not np.isnan(v) else [255, 255, 255] for v in
                        scaled_percentile_rank]  # convert normalized values to color spectrom.

    gdf["color"] = color_column
    return gdf


def create_deckGL_map(gdf_list=[], centerX=46.6725, centerY=24.7425, basemap=False, zoom=17, filename=None):
    start = time.time()
    pdk_layers = []
    for layer_number, gdf_dict in enumerate(gdf_list):
        local_gdf = gdf_dict["gdf"].copy(deep=True).reset_index()
        local_gdf["geometry"] = local_gdf["geometry"].to_crs("EPSG:4326")
        # print(f"{(time.time()-start)*1000:6.2f}ms\t {layer_number = }, gdf copied")
        start = time.time()

        radius_attribute = 1
        if "radius" in gdf_dict:
            radius_attribute = gdf_dict["radius"]
            local_gdf = local_gdf[~local_gdf[radius_attribute].isna()]
            r_series = local_gdf[radius_attribute]

            radius_min = gdf_dict["radius_min"] if "radius_min" in gdf_dict else 5
            radius_max = gdf_dict["radius_max"] if "radius_max" in gdf_dict else 10
            #r_series = radius_min + (r_series - r_series.mean()) / r_series.std() * radius_max
            r_series = radius_min + (r_series - r_series.min()) / (r_series.max()-r_series.min()) * (radius_max-radius_min)

            # r_series = r_series.apply(lambda x: (x - r_series.mean()) / r_series.std() if not np.isnan(x) else np.nan)

            # r_series = r_series.apply(lambda x: max(1,x) + 3 if not np.isnan(x) else np.nan)
            local_gdf['__radius__'] = r_series

        width_attribute = 1
        width_scale = 1
        if "width" in gdf_dict:
            width_attribute = gdf_dict["width"]
            if "width_scale" in gdf_dict:
                width_scale = gdf_dict["width_scale"]
            local_gdf['__width__'] = local_gdf[width_attribute] * width_scale

        if "opacity" in gdf_dict:
            opacity = gdf_dict["opacity"]
        else:
            opacity = 1

        if ("color_by_attribute" in gdf_dict) or ("color_method" in gdf_dict) or ("color" in gdf_dict):
            args = {arg: gdf_dict[arg] for arg in ['color_by_attribute', 'color_method', 'color'] if
                    arg in gdf_dict}
            local_gdf = color_gdf(local_gdf, **args)
            # print (local_gdf['color'])

        pdk_layer = pdk.Layer(
            'GeoJsonLayer',
            local_gdf.reset_index(),
            opacity=opacity,
            stroked=True,
            filled=True,
            wireframe=True,
            get_line_width='__width__',
            get_radius='__radius__',
            get_line_color='color',
            get_fill_color="color",
            pickable=True,
        )
        pdk_layers.append(pdk_layer)
        # print(f"{(time.time()-start)*1000:6.2f}ms\t {layer_number = }, pdk.Layer created.")
        start = time.time()

        if "text" in gdf_dict:
            # if numerical, round within four decimals, else, do nothing and treat as string
            try:
                local_gdf["text"] = round(local_gdf[gdf_dict["text"]], 2).astype('string')
            except TypeError:
                local_gdf["text"] = local_gdf[gdf_dict["text"]].astype('string')

            # formatting a centroid point to be [lat, long]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                local_gdf["coordinates"] = local_gdf["geometry"].centroid
            local_gdf["coordinates"] = [[p.coords[0][0], p.coords[0][1]] for p in local_gdf["coordinates"]]

            layer = pdk.Layer(
                "TextLayer",
                local_gdf.reset_index(),
                pickable=True,
                get_position="coordinates",
                get_text="text",
                get_size=16,
                get_color='color',
                get_angle=0,
                background=True,
                get_background_color=[0, 0, 0, 125],
                # Note that string constants in pydeck are explicitly passed as strings
                # This distinguishes them from columns in a data set
                get_text_anchor=String("middle"),
                get_alignment_baseline=String("center"),
            )
            pdk_layers.append(layer)
            # print(f"{(time.time()-start)*1000:6.2f}ms\t {layer_number = }, text layer created and added.")
            start = time.time()

    initial_view_state = pdk.ViewState(
        latitude=centerY,
        longitude=centerX,
        zoom=zoom,
        max_zoom=20,
        pitch=0,
        bearing=0
    )

    if basemap:
        r = pdk.Deck(
            layers=pdk_layers,
            initial_view_state=initial_view_state,
        )
    else:
        r = pdk.Deck(
            layers=pdk_layers,
            initial_view_state=initial_view_state,
            map_provider=None,
            parameters={
                "clearColor": [0.00, 0.00, 0.00, 1]
            },
        )

    if filename is not None:
        r.to_html(
            filename,
            css_background_color="cornflowerblue"
        )
    # print(f"{(time.time()-start)*1000:6.2f}ms\t {layer_number = }, map rendered.")
    start = time.time()
    return r

def color_layer(self, layer_name, color_by_attribute=None, color_method="single_color", color=None):
    '''
    This function is used internally to apply default color settings for given layers
    '''
    if layer_name in self.DEFAULT_COLORS.keys() and color_by_attribute is None and color is None:
        # set default colors first. all default layers call without specifying "color_by_attribute"
        # default layer creation always calls self.color_layer(layer_name) without any other parameters
        color = self.DEFAULT_COLORS[layer_name].copy()
        color_method = "single_color"
        if type(color) is dict:
            # the default color is categorical..
            color_by_attribute = color["__attribute_name__"]
            color_method = "categorical"
    self.layers[layer_name]["gdf"] = self.color_gdf(
        self.layers[layer_name]["gdf"],
        color_by_attribute=color_by_attribute,
        color_method=color_method,
        color=color
    )
    return

