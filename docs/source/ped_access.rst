Running a Pedestrian Accessibility Workflow
====================================================
The Accessibility workflow is meant to sreamline the process of generating various accessibility metrics from one origin to a set of destinations. These accessibility metrics are:

* **Reach**: The Reach index, also known as a “cumulative opportunities accessibility index” captures the number of destinations that can be reached from each origin within a given network radius, optionally weighted by numeric destination attributes. 
* **Gravity**: The gravity index additionally accounts for travel costs required to reach each of the destinations. It therefore offers a more precise definition of accessibility, where each spatially separated origin has a unique accessibility to surrounding destinations due to differences in travel costs.
* **KNN Access**: K-Nearest-Neighbor (KNN) accessibility metric is similar to the Gravity index, with the  exception that only K nearest destinations (e.g. three nearest bus stops) are used to compute the index, ignoring the rest, and the relative impact of each of the K nearest destinations is controlled by predefined weights.
..
    * **KNN Weight**: This is simply the resilt of multiplying an origin's **KNN Access** by the origin weight. This metric could be used as an origin weight, for instance, to represent elastic trip generation: Origins generate trips based on how well they score on KNN access: Origins with perfect KNN access score generate thier full potential; origins with less KNN access generate less trips.


1) Workflow Summary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To run a simulation, import the function  ``KNN_accessibility`` and give it a ``city_name`` parameter that represent a folder inside your ``Cities`` folder. The example provided in the installation guide comes with a ``Cities\SOmerville\Data`` folder that comes with a prepared ``access_pairings.csv`` file.  `Download example notebooks <https://www.dropbox.com/scl/fo/vvhukdl6vc2wcprzp9kwc/h?rlkey=3zteo0dj08d5mhbeyo95v8qd2&dl=1>`_. The function to run the workflow is::

   from madina.una.workflows import KNN_accessibility

    KNN_accessibility(
        city_name=None,
        data_folder=None,
        output_folder=None,
        pairings_file="pairing.csv",
        num_cores=8,
    )

For thew code to run, two things must be prepared first:

* **The data folder**: place all the network, origin and destination layers into one folder. More information about data preparation is below.
* **The pairing table**: Prepare a pairing table that pairs the origin layer with destination layers, and specify the required accessibility parameters. More on preparing the pairing table is below.

if the inpt data is placed inside ``Cities/Somerville/Data``, and the pairing table is named ``access_pairings.csv`` you can run the workflow using 8 cpu cores by calling::


   from madina.una.workflows import betweenness_flow_simulation

    KNN_accessibility(
        city_name='Somerville',
        pairings_file="access_pairings.csv",
        num_cores=8,
    )


2) Data preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^
* There is a prepared data example in the ``Cities/Somerville/Data`` folder in the `example notebooks <https://www.dropbox.com/scl/fo/vvhukdl6vc2wcprzp9kwc/h?rlkey=3zteo0dj08d5mhbeyo95v8qd2&dl=1>`_. The next few points explain how to prepare a folder from scratch for a new city.

* Inside the ``Cities`` Folder, create a folder for the city you want to simulate, for instance, the example folder ``Somerville`` already exists. Inside the ``Somerville`` folder, create a folder called ``Data``


* Inside the ``Cities/Somerville/Data`` folder you created, add your data files as either geoJSON files or shapefiles.

* It is necessary that all files in the ``Data`` folder be cropped into a specific scope that's as narrow as possible, for instance, clip a layer of the Massachusetts road network into a scope that works best for your analysis: like: a buffer around your area of interest that is equal or larger than your search radius.

.. warning::
    Unneccesarly large files might make the simulation slow, and if there are origins or destinations outsite the scope of the network file, origins and destinations would be snapped to the closest network element causing boundary issues. 

* Be sure to use a projected CRS (example: ``"EPSG:3857"`` WGS 84 / Pseudo-Mercator) (not  a geographic crs like ``"EPSG:4326"`` World Geodetic System). 
* The units of the projection system would be used as is. Make sure you know the unit of the projection system (feet or meters). The simulation parameters would use the same unit as the projection system, so make sure any numbers in the pairing table you're preparing are the same unit as your projection system. 
* In spatial analysis in general, it is always best to use a localized projection. For instance, in Somerville, MA, the recommended projectied coordinate system  `for use in Massachussetts <https://www.mass.gov/info-details/learn-about-massgis-data>`_ is the "Massachusetts State Plane Coordinate System, Mainland Zone meters" ``EPSG:26986``. Notice that this CRS is in meters, and all data reported in `MassGIS <https://www.mass.gov/info-details/massgis-data-layers>`_ and used in the Somerville data folder is in this CRS. 

.. warning::
    All data files must be in the same projection. All numbers used for setting parameters are in the same unit as the CRS. Use a local projected CRS for more accurate calculations.


2) Pairing Table preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



* The pairing table for the accessibility workflow is a CSV that contains these columns:

    * Flow_Name
    * Network_File
    * Network_Cost
    * Origin_File
    * Origin_Name
    * Origin_Weight
    * Destination_File
    * Destination_Name
    * Destination_Weight
    * Radius
    * Beta
    * KNN_Weight
    * Plateau
    * Turns
    * Turn_Threshold
    * Turn_Penalty


* There is a prepared pairing table (shown below) in the Somerville example. ``Cities/Somerville/Data/access_pairings.csv``. This section contains an explanation of each column in the table and how to properly fill it for youw own case. Feel free to copy the Somerville pairing table as a template and modify it for your own city. It is easy to edit CSV files in Excel or any other spreadsheet software.
* Once done creating your pairing table, place it inside your ``Cities/city_name/Data`` folder.

.. warning::
    Keywords in the table, and column names from data files (for use in network cost, origin and destination weights) must be spelled the same way, **case sensitive**, and space/tab sensitive. Make sure the csv contains keywords and column names properly spelled.


.. csv-table:: Sample Access Workflow Pairing Table for Somerville 
    :file: notebooks//Cities//Somerville//Data//access_pairings.csv

``Flow_Name``
----------------
* This is a descriptive name for the origin-destination pair, It will be used to create columns in the origin file for accessibility metrics as a prefix. 
* The name should explain this pair and any special settings it might have to properly document columns: ``Homes_to_Subway`` for instance is a good name for accessibility metrics from homes as origins and jobs as a destinations.
* It is possible to create multiple rows for the same origin and destination pairs when you want to vary a certain parameter like search radius, beta, or anything else. In such case, make sure to name each pair to distinguish what is unique about it. for instance, ``Homes_to_Subway_radius400``, ``Homes_to_Subway_radius800`` allows you to distinguish between access from home to subway stations while varying the search radius. 
* Whatever name you give for each pair would be used to store the accessibility metrics as columns in a geojson copy of the origin layer's input file specified in ``Origin_File``, for example, when the workflow for Somerville is dome, a copy of the file ``homes.geojson`` would be created in the output folder that contains three new columns for the pair named ``Homes_to_Subway``: 

    * ``Homes_to_Subway_reach``: this is the reach accessibility metric using the specified parameters in the pair named ``Homes_to_Subway```. between the specified origin and destination, reach accessibility is sensitive to these paramerers: ``Destination_Weight``, ``Radius``, ``Turns``, ``Turn_Threshold``, ``Turn_Penalty``.
    * ``Homes_to_Subway_gravity``: this is the gravity accessibility metric uaing the specified parameters in the pair named ``Homes_to_Subway``. Between the specified origin and destination, gravity accessibility is sensitive to these paramerers: ``Destination_Weight``, ``Radius``, ``Beta``, ``Turns``, ``Turn_Threshold``, ``Turn_Penalty``.
    * ``Homes_to_Subway_knn_access``: this is the KNN-access metric uaing the specified parameters in the pair named ``Homes_to_Subway``. Between the specified origin and destination, KNN-access is sensitive to these paramerers: ``Destination_Weight``, ``Radius``, ``Beta``, ``KNN_Weight``, ``Plateau``, ``Turns``, ``Turn_Threshold``, ``Turn_Penalty``.
..
        * ``Homes_to_Subway_knn_weight``: this is the KNN weight metric uaing the specified parameters in the pair named ``Homes_to_Subway``. Between the specified origin and destination, KNN weight is sensitive to these paramerers: ``Origin_Weight``, ``Destination_Weight``, ``Radius``, ``Beta``, ``KNN_Weight``, ``Plateau``, ``Turns``, ``Turn_Threshold``, ``Turn_Penalty``.





``Network_File``
----------------
* This should be the name for your network file, for example ``network.geojson``.

.. note:: 
    You can create multiple pairs of the same origin and destination and set them to use different networks. For instance, to see the difference in accessibility metrics when using street centerlines versus a sidewalk network. or to compare accessibility metrics on multiple versions of the network showing multiple interventions. 


``Network_Cost``
-------------------
* Each OD pair in the pairing table can have its own network cost.
* For geometric distance, use the **keyword** ``Geometric``.
* If you needed to use a specific network weight, to show a different perceived distance for example, you need to specify a column name that exists in the network file specified in ``Network_File``.
* If you specified a ``Network_Cost``, and the column you specified contains:

    * **Positive Values**: would be used as is for network cost.
    * **Zero**: the ``Geometric`` distance would be used for network segment that have a 0 ``Network_Cost`` value. This is useful if you only modify a small subset of the network. Segments that are assigned 0 would fall back to the geometric distance.
    * **Negative Values**: any negative value (probably resulting from errors in percieved distance calculations) would be replaced by a value of 0.01. Negative network costs are not allowed in distance calculations.



``Origin_File``
--------------------
* Specify the file name that contains the origins in this OD pair. For example, ``homes.geojson``.

``Origin_Name``
-------------------
* A name for your origin, this name would be used to name the origin layer, and internally keep track of the origin data. For example: ``Homes``.

``Origin_Weight``
---------------------
..    
    * The origin weight is only used to calculate a metric called ``knn_weight``:

        * The **keyword** ``Count``: Sets all origin weights to 1 and in this case ``knn_weight`` would be equal to ``knn_access``
        * A name of a column in the file specified in ``Origin_File``, for example: ``pop_total``. in this case, ``knn_weight`` = ``pop_total`` * ``knn_access``. 
        * The ``knn_weight`` is useful to approximat an origin's trip generation potential depending on the destinations it can access. 

    * origin weight only have an impact on ``knn_weight``, but not on ``reach``, ``gravity``, and ``knn_access``.
    
    
    .. note:: 
    ``Origin_Weight`` is accounted for in the ``KNN_accessibility()`` workflow starting in version 0.0.15. To check your current verison:

    * ``import madina as md``
    * ``print (md.zonal.VERSION)``

Origin weights have no impact on accesibility metrics, feel free to use the **keyword** ``Count``: Sets all origin weights to 1


``Destination_File``
----------------------
* Specify the filename that contains the destinations in this OD pair. For example, ``subway.geojson``. 


``Destination_Name``
----------------------
A name for your destination, this name would be used to name the destination layer, and internally keep track of the destination data. For example: ``Jobs``.


``Destination_Weight``
----------------------------

* The destination weight have the follwoing settings:

    * The **keyword** ``Count``: Sets all destination weights to 1, and all destinations are weighted equally. 
    * Using a name of a column in the file specified in ``Destination_File``. This would be used to weigh destination differently based on their importance for the reach and gravity accesibility metrics. 

* destination weight only have an impact on ``reach`` and  ``gravity``, but not on ``knn_access``.


.. note:: 
    ``Destination_Weight`` is accounted for in the ``KNN_accessibility()`` workflow starting in version ``0.0.15``. To check your current verison:

    * ``import madina as md``
    * ``print (md.zonal.VERSION)``

    if the version is older than 0.0.15, run the following commands in your terminal to update to the latest release:
    
    * ``conda activate madina_env``
    * ``pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple madina --upgrade``
``Radius``
------------------

* A number, For example, ``800``.
* This number is the same units as the input data CRS. Make sure you are using a CRS in meters if you want the search radius to be 800 meters.

``Beta``
----------
* A "sensitivity to walk" parameter. A low value means individuals are less sensitive to walking long distance (More willing to walk more). 
* A typical value ranges between ``0.001`` (Low sensitivity to distance) to ``0.004`` (HIgh sensitivity to distance). These values assume a meter unit of distance.



``KNN_Weight``
-----------------
* This parameter is used to calculate the ``knn_access`` metric. 
* Example ``[0.5, 0.3, 0.2]`` means that only three neighbors would be considered to give this origin a weight. The first neighbor is weighted ``0.5``, the second ``0.3`` and the third ``0.2``.
* The number of weights in the list equals the number of destinations that would be considered for suffecuent access. reachable destinations that excces the count of this list do not account towards ``knn_acceess``. 
* The sum of this list reflect the relative importance of this pair in the ``total_knn_access`` and ``normalized_knn_access``. if lists in all pairs sum up to one (or the same number), all pairs are weigted equally in these Summary paramerers.


.. warning:: 
    if you only care about  ``reach`` and ``gravity``,  set ``KNN_Weight`` to ``[1]`` in order to avoid errors. 

``Plateau``
-------------
* A number, for example ``400``, in the same unit as the unit if the CRS.
* If a destination is closer than this number, it gets assigned its full KNN weight. If its further, the KNN weight is penalized exponentially based on the value of ``beta`` for the additional distance.


.. warning:: 
    if you only care about  ``reach`` and ``gravity``,  set ``KNN_Weight`` to ``0`` in order to avoid errors. 

``Turns``
------------
This oculd be one of two options: 

* **keyword** ``TRUE`` to enable turn penalty.
* **keyword** ``FALSE`` to disable turn penalty.

``Turn_Threshold``
-------------------------

* If ``Turns`` is set to **keyword** ``TRUE``, this parameter must be provided.
* A number, in degrees, that represents the minimum deviation from a straight line that defines a turn subject to penalty. For example: ``45``.


``Turn_Penalty``
--------------------
* If ``Turns`` is set to keyword ``TRUE``, this parameter must be provided.
* A number, in the same unit as CRS of the ``Network_File``, that represents the distance penalty each turn incurs. For example, ``60`` means each identified turn would be equivalent to walking 60 units of distance (in the CRS distanc, either meters or feet).


3) Running a Simulation 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* To run the workflow, import the function  ```KNN_accessibility`` and give it a ``city_name`` parameter that represent a folder inside your ``Cities`` folder. The example notebooka comes with a ``Cities\Somerville\Data`` folder that comes with a prepared ``access_pairings.csv`` file that you can use as a sammple.  `Download example notebooks <https://www.dropbox.com/scl/fo/vvhukdl6vc2wcprzp9kwc/h?rlkey=3zteo0dj08d5mhbeyo95v8qd2&dl=1>`_. The output would be stored in ``Citied\Somerville\KNN_workflow``. 
* The parameter ``num_cores`` specify how many CPU cores to use to speed up the workflow. Setting it to the maximum number of CPU cores in your computer provide maximum speed, but reduces the responsiveness of your computer. Setting it to a lower number would make your computer more responive if you plan to keep using it while the workflow is running,

run the simulation::

   from madina.una.workflows import KNN_accessibility

   KNN_accessibility(
      city_name="Somerville", 
      pairings_file="access_pairings.csv",
      num_cores=8,
   )


* if you prefer to manually set up the input and outout files, the input file must contain the pairing table specified, and you don't need to specify the city name. To avoid potential issues with different path representations between Windows/Linux/Mac, use the os.path.join() function to provide the location of the input folder and the output folder. The output folder would be created if it didn't exist. ::
    
    import os
    from madina.una.workflows import KNN_accessibility

    KNN_accessibility(
        data_folder=os.path.join(r"C:\Users\username\Desktop\research\data"),
        output_folder=os.path.join(r"C:\Users\username\Desktop\research\output"),
        pairings_file="access_pairings.csv",
        num_cores=8,
    )