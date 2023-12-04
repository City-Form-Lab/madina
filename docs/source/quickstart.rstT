Quickstart
==================================

This page gives an introduction in how to create and populate a ``Zonal`` workspace with layers and
run analysis on it using madina's Urban Network Analysis toolkit.

First, make sure that:

- madina is :ref:`installed <install>`
- madina is :ref:`up-to-date <updates>`

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


Accessibility analysis
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


