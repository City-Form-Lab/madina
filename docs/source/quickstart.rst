Quickstart
==================================

This page gives an introduction in how to create and populate a ``Zonal`` workspace with layers and
run analysis on it using madina's Urban Network Analysis toolkit.

First, make sure that:

- madina is installed
- madina is up-to-date

Let's start by exploring some simple examples.

Getting started with ``madina``
--------------
In the madina library, a workspace is represented as a ``Zonal`` object, analogous to
a "project" in QGIS or Rhino 3D.

Begin by importing the class.

>>> from madina import Zonal

Now let's instantiate it.

>>> cambridge = Zonal()

To look at details about the ``Zonal`` object, call the
``describe`` function.

>>> cambridge.describe()

By default, a Zonal's projected CRS is EPSG:3857,
while it's geographic CRS is EPSG:4326. This can be changed as follows.

    >>> cambridge.projected_crs = "EPSG:32118"


In the next section, we'll learn about adding layers to
``Zonal`` objects.


Loading Data Layers
--------------
``madina`` uses GeoPandas internally, with layers stored as GeoDataFrames. GeoPandas extends the functionality of
the Pandas library, so if you're familiar with Pandas DataFrames, you can apply that knowledge. The main distinction
is that a GeoDataFrame always includes a geometry column and supports various spatial operations.

Just like GIS and Rhino, the way to incorporate data into your analysis is
through loading data in a layering system. The library supports different
formats like .geoJSON and .shp. To load a layer, call the function
``load_layer()`` giving it a layer name for your new layer, and a path
to where the data is stored.

.. code::

     cambridge.load_layer(
        layer_name='sidewalks',
        file_path='./path/to/building_entrances.geojson'
    )

You can also assign an RGB color to a layer:

    >>> cambridge.color_layer('sidewalks', color=[128, 128, 128])


Creating a street network
--------------
A network, comprising nodes and edges, is essential for analysis.
While the loaded sidewalk data may visually resemble a network, it must be converted into a topologically
connected set of nodes and edges to be functional. This conversion is achieved through the
``create_street_network``` method of the Zonal class.

    >>> cambridge.create_street_network('sidewalks')

We can visualize the network using the ``create_map`` function.

    >>> cambridge.create_map()

Constructing a graph
--------------
Before performing an analysis, finalize the graph by calling the ``create_graph()`` function after setting up your
network, origins, and destinations.

    >>> cambridge.create_graph()


Urban Network Analysis
==================================

Setting Origins and Destinations
--------------
Origins represent the starting points where trips are generated, while destinations are the endpoints where trips
are distributed to. The library's UNA (urban network analysis) use these to calculate
metrics like pedestrian accessibility, gravity, and betweenness.

To illustrate this, add a few more layers to the Zonal.

.. code::

     cambridge.load_layer(
        layer_name="building_entrances",
        file_path="./path/to/building_entrances.geojson"
    )

    cambridge.load_layer(
        layer_name="sidewalks",
        file_path="./path/to/sidewalks.geojson"
    )


To set a previously-inserted layer as a node to the Zonal graph, one
can do the following:

.. code::

    # inserting origins:
    cambridge.insert_node(
        label='origin',
        layer_name="subway",
    )

    # inserting destinations
    cambridge.insert_node(
        label='destination',
        layer_name="building_entrances",
    )



Before conducting an analysis, finalize the graph by calling the
``create_graph`` function after setting up your network,
origins, and destinations.
    >>> cambridge.create_graph()


Accessibility analyses
--------------
Reach Index

We can calculate a "reach index" from the nodes of the graph. In this
case, we are concerned with how many building entrances can be reached
from subway stations in a ~ 5 minute (300m) walk-shed.

To do this, we must import the first import the ``una`` module, which contains
tools for UNA.

    >>> from madina import una

There's an ``accessibility`` function, which adds a column...

.. code::

    una.accessibility(
        cambridge,
        reach=True,
        search_radius=300
    )



    cambridge.create_map(
        layer_list=[
            {"gdf": cambridge.network.edges, 'color': [125, 125, 125]},
            {"gdf": cambridge.network.nodes, "radius": "una_reach", 'text':'una_reach', 'color': [255, 0, 0]},
        ]
    )

Service Area

The ``service_area()`` function outputs two GeoDataFrames: ``destinations``
(covered destinations) and ``network_edges`` (segments within the service area).

Additionally, scope_gdf contains the service area boundaries.
The ``create_deck_map()`` function accepts either a layer or a
gdf (GeoDataFrame) as input.


.. code::

    destinations, network_edges, scope_gdf = una.service_area(
        cambridge,
        search_radius=100,
    )

    cambridge.create_map(
        layer_list=[
            {"layer": 'sidewalks'},
            {"layer": 'building_entrances'},
            {"gdf": network_edges, "color": [0, 255, 0]},
            {"gdf": destinations, "color": [255, 0, 0]},
            {"gdf": scope_gdf[scope_gdf['name'] == 'service area border'], "color": [0, 0, 255], 'opacity': 0.10},
        ]
    )

Setting up Attributes
--------------

In a coding environment, changing or setting a single attribute of a layer
element is typically uncommon. Attributes are usually data inputs or calculated
results. However, in debugging, validation, or experimentation scenarios, setting
a single attribute can be valuable. The ``set_attribute()`` function facilitates this
process.

In a visual interface, you can use a mouse to select elements. However, in a coding environment, each object is identified with a unique identifier. Once a layer is loaded, it is assigned an `id` attribute. To locate the building entrances with IDs 2 and 115, hover over the previous map. These entrances correspond to the points mentioned in Rhino bullet point 19. They both belong to the `layer="building_entrances"` and we want to set the `attribute='student'` to the values 10 and 10, for IDs 2 and 115 respectively.

.. code::

   # Example code to set attributes for building entrances
   set_attribute(layer="building_entrances", ids=[2, 115], attribute='student', value=[10, 10])
```







