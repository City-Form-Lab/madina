.. madina documentation master file, created by
   sphinx-quickstart on Mon Oct 23 15:39:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Madina
============
Madina **(Arabic for the word 'city')** is a python package of classes and functions to streamline the representation and analysis of urban data. The package enables layer management (Similar to layers in CAD/GIS), spatial network representation, and spatial visualizations. The package also includes a Python implemetation of the **Urban Network Analysis** tools (`Homepage <https://unatoolbox.notion.site/>`_). The source code for Madina is available `on github <https://github.com/City-Form-Lab/madina>`_.

.. toctree::
   :maxdepth: 2
   :caption: Installation Guides:

   installation_guide



Examples and Notebooks
=========================
The example notebooks and the datat needed to run them are available in the `github repository  <https://github.com/City-Form-Lab/madina/tree/main/examples>`_, or could be  `downloaded from this link <https://www.dropbox.com/scl/fi/6dilb8o02gq5q5i75fdx5/examples.zip?rlkey=qiewt7o1non5nxk845o7gcmoq&dl=0>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   notebooks/1-loading_data.ipynb
   notebooks/2-creating_networks.ipynb
   notebooks/3-UNA_tools_access.ipynb
   notebooks/4-UNA_tools_paths.ipynb
   notebooks/5-Pedestrian_flow_multiple_ODs.ipynb
   notebooks/6-UNA_workflows.ipynb


Here is a link to a generated map : `Light Map <_static/flow_map_light.html>`_  and  `Dark Map <_static/flow_map_dark.html>`_




UNA Pedestrain Flow Simulation
====================================
The package enables modelling pedestrian accessibility and movement in urban areas between pairs of pre-specified origins and destinations. This can be done by following these steps:
* Prepare input data files for the network, and each origin and destination. Place all data in a folder called ``Cities/city_name/Data``

* Fill in the pairing table to specify origin-destination pairs, and specify specific parameters for each pair. Save the filled pairing table in the same ``Cities/city_name/Data`` folder

* run the simulation::

   from madina.una.betweenness import betweenness_flow_simulation

   betweenness_flow_simulation(
      city_name="new_york"
   )

* Output would be saved in ``Cities/city_name/Simulations``.
* More instructions on running a pedestrain flow simulation, preparing data and creating the pairing table are found in the documentation here

.. toctree::
   :maxdepth: 2
   :caption: Instructions for Preparing and Running a Simulation:

   ped_flow


.. image:: img//nyc_flow.png
  :width: 600
  :alt: Instruction Guide



.. toctree::
   :maxdepth: 5
   :caption: Documentation: Package Modules, Classes and Functions


   madina.una
   madina.zonal
