Instructions for Pedestrian Flow Simulation
============



1) Data preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^
* There is a prepared data example in the ``workflows/Cities/SOmerville`` folder. The next few points explain how to prepare a folder from scratch for a new city.

* Inside the ``workflows/Cities`` Folder, create a folder for the city you want to simulate, for instance, the example folder ``Somerville`` already exists/ Inside the ``Somerville`` folder, create a folder called ``Data``


* Inside the ``Cities/Somerville/Data`` folder you created, add your data files as either geoJSON files or shapefiles.

* It is preferred that all files in the ``Data`` folder be cropped into a specific scope that's as narrow as possible, for instance, clip a layer of the Massachusetts  road network into a scope that works best for your analysis: like: a buffer around your area of interest that is equal or larger than your search radius.

.. note::
    Unneccesarly large files might make the simulation slow

* Be sure to use a projected CRS (example: ``"EPSG:3857"``) (not  a geographic crs like ``"EPSG:4326"``). The units of the projection system would be used as is. Make sure you know the unit of the projection system (feet or meters). The simulation would use the same unit as the projection system, so make sure any numbers in the pairing table are the same unit as your projection system. 

.. warning::
    All data files must be in the same projection, all numbers are in the same unit as the CRS


2) Pairing Table preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* There is already a prepared pairing table in the SOmerville example. ``Cities/Somerville/Data/pairings.csv``, the following steps explain how to create one from scratch. Feel free to copy the Somerville pairing table and modify it for your own city.
* Inside your ``Cities/Somerville/Data`` folder, you need to create a ``pairings.csv`` file, This file contains the simulation settings for the origin-destination pairs you want to simulate. The “Pairing.csv” file contains the following columns.

.. warning::
    Keywords in the table, and column name from data files must be spelled the same way, case sensitive, and space/tab sensitive. Make sure the csv contain keywords and column names as they are intended

``Flow_Name``
----------------
* This is a descriptive name for the flow, It will be used to create columns in the network file for the simulation results.
* The name should explain this flow: "b_homes_jobs" for instance is a good name for betweenness flows between homes as origins and jobs as a destination.
* if a certain OD pair is repeated, but with different settings, make sure the name reflects this, for example, for trips from home to work, name one flow “b_home_jobs_geometric” and “b_home_jobs_percieved” if you used geometric distance for the first and perceived distance for the second.


``Network_File``
----------------
* This should be the name for your network file, for example ``sidewalks.geoJSON``.
.. warning:: 
    You need to use the same network file for all rows in the pairing table as results from all pairs are reported to the same network.


``Network_Cost``
-------------------
* Each OD pair in the pairing table can have its own network cost.
* For geometric distance, use the **keyword** ``Geometric``.
If you needed to use a specific network weight, perceived distance for example, you need to specify a column name that exists in the network file specified in ``Network_File``.
* If you specified a network cost, and the column you specified contained zero or negative values, the ``Geometric`` distance would be used for network segment that have 0 or negative values.



``Origin_File``
--------------------
* Specify the file name that contains the origins in this OD pair. For example, ``homes.geoJSON``.

``Origin_Name``
-------------------
* A name for your origin, this name would be used to name the origin file, and internally keep track of the origin data. For example: ``Homes``.

``Origin_Weight``
---------------------
* The origin weight that will be used for this origin in the betweenness flow simulation. It can either be:
    * The **keyword** ``Count``: Sets all origin weights to 1.
    * A name of a column in the file specified in ``Origin_File``, for example: ``pop_total``.


``Destination_File``
----------------------
* Specify the filename that contains the destinations in this OD pair. For example, ``jobs.geoJSON``


``Destination_Name``
----------------------
A name for your destination, this name would be used to name the destination file, and internally keep track of the destination data. For example: ``Jobs``.


``Destination_Weight``
----------------------------

* The destination weight that will be used for this destination in the betweenness flow simulation. It can either be:
    * The **keyword** ``Count``: Sets all destination weights to 1.
    * A name of a column in the file specified in ``Destination_File``, for example: ``EMPNUM``.


``Radius``
------------------

* A number, For example, ``800``.
* This number is the same units as the input data CRS. Make sure you are using a CRS in meters if you want the search radius to be 800 meters.

``Beta``
----------
* A "sensitivity to walk" parameter. A low value means individuals are less sensitive to walking long distance (More willing to walk more). 
* A typical value ranges between ``0.001`` (Low sensitivity to distance) to ``0.004`` (HIgh sensitivity to distance). These values assume a meter unit of distance.

``Decay``
-----------
* **keyword** ``TRUE`` to enable decay in the betweenness flow simulation.
* **keyword** ``FALSE`` to disable decay.

``Decay_Mode``
-----------------
* Would only be used if ``Decay_Mode`` is specified as the keyword ``TRUE``. Options are:
    * **keyword** ``exponent`` for exponential penalty (Current preferred method).
    * **keyword** ``power`` for 1/(X^2) penalty.


``Closest_destination``
----------------------------
* **keyword** ``TRUE`` to only route trip to the closest reachable destination/
* **keyword** ``FALSE`` route trips to all destinaitons reachable within the search radius. Trips would be distributed to destinations according to the Huff Model based on destination distancs and destination attractiveness (as measured by ``Destination_Weight``).


``Detour``
--------------
* A number, For example, ``1.15``.
* This number would specify the detour ration in the simulation. ``1.15`` means routes that are 15% longer than the shortest path would be considered.


``Elastic_Weights``
----------------------
* **keyword** ``TRUE`` to enable the K Nearest Neighbor (KNN) Elastic Weight.
* **keyword** ``FALSE`` to disable elastic weight.


``KNN_Weight``
-----------------
* If ``Elastic_Weights`` is set to **keyword** ``TRUE``, this parameter must be provided.
Example ``[0.5; 0.3; 0.2]`` means that only three neighbors would be considered to give this origin a weight. The first neighbor is weighted ``0.5``, the second ``0.3`` and the third ``0.2``. 
* The number of weights in the list equals the number of destinations that would be considered.
.. warning:: 
    The simulation does not check for correctness, but numbers in the list are expected to sum up to 1.

``Plateau``
-------------
* If ``Elastic_Weights`` is set to **keyword** ``TRUE``, this parameter must be provided.
* A number, for example ``400``, in the same unit as the unit if the CRS.
* If a destination is closer than this number, it gets assigned its full KNN weight. If its further, the KNN weight is penalized exponentially for the additional distance.

``Turns``
------------
* **keyword** ``TRUE`` to enable turn penalty.
* **keyword** ``FALSE`` to disable turn penalty.

``Turn_Threshold``
-------------------------

* If ``Turns`` is set to **keyword** ``TRUE``, this parameter must be provided.
* A number, in degrees, that represents the minimum deviation from a straight line that defines a turn subject to penalty. For example: ``45``.


``Turn_Penalty``
--------------------
* If ``Turns`` is set to keyword ``TRUE``, this parameter must be provided.
* A number, in the same unit as CRS of the ``Network_File``, that represents the distance penalty each turn incurs. For example, ``60`` means each turns would be equivalent to walking 60 units of crs distance (meters or feet).


3) Running a Simulation 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To run a simulation, import the function  ```betweenness_flow_simulation`` and give it a ``city_name`` parameter that represent a folder inside your ``Cities`` folder. The example provided in the installation guide comes with a ``Cities\SOmerville\Data`` folder that comes with a prepared ``pairings.csv`` file.  `The examples folder <https://www.dropbox.com/scl/fi/1fbidbc5bqz7ccn61u1yq/examples.zip?rlkey=y0ppgukbyck0scw6pakrcn7f5&dl=1>`_

run the simulation::

   from madina.una.betweenness import betweenness_flow_simulation

   betweenness_flow_simulation(
      city_name="Somerville"
   )
