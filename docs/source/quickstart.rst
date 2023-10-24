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


