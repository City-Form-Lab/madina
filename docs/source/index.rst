.. madina documentation master file, created by
   sphinx-quickstart on Mon Oct 23 15:39:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Madina
============
Madina *(Arabic for the word 'city')* is a package of classes and functions to streamline the representation and analysis of urban networks. The package includes a python implemetation of the **Urban Network Analysis** tools (`Homepage <https://unatoolbox.notion.site/>`_). The source code for Madina is available `on github <https://github.com/City-Form-Lab/madina>`_.

.. toctree::
   :maxdepth: 2
   :caption: Installation Guides:

   installation_guide



Referencing in Your Research
=======================================

To reference this package in your research, you can cite the paper available on SSRN:

* Alhassan, Abdulaziz and Sevtsuk, Andres, **Madina Python Package: Scalable Urban Network Analysis for Modeling Pedestrian and Bicycle Trips in Cities.** Available at SSRN: https://ssrn.com/abstract=4748255 or http://dx.doi.org/10.2139/ssrn.4748255 ::


   @article{alhassan2024madina,
   title={Madina Python Package: Scalable Urban Network Analysis for Modeling Pedestrian and Bicycle Trips in Cities},
      author={Alhassan, Abdulaziz and Sevtsuk, Andres},
      journal={SSRN},
      year={2024},
      publisher={Elsevier}, 
      doi={10.2139/ssrn.4748255},
      url={https://ssrn.com/abstract=4748255}
   }





Examples and Notebooks
=========================
The example notebooks and the datat needed to run them are available in the `github repository  <https://github.com/City-Form-Lab/madina/tree/main/docs/source/notebooks>`_, or could be  `downloaded from this link <https://www.dropbox.com/scl/fo/vvhukdl6vc2wcprzp9kwc/h?rlkey=3zteo0dj08d5mhbeyo95v8qd2&dl=1>`_

.. toctree::
   :maxdepth: 2
   :caption: Notebooks:

   notebooks/1-loading_data.ipynb
   notebooks/2-creating_networks.ipynb
   notebooks/3-UNA_tools_access.ipynb
   notebooks/4-UNA_tools_paths.ipynb
   notebooks/5-Pedestrian_flow_multiple_ODs.ipynb
   notebooks/6-UNA_workflows.ipynb


Here is a link to a generated map : `Light Map <_static/flow_map_light.html>`_  and  `Dark Map <_static/flow_map_dark.html>`_



UNA Access Metrics Workflow
===================================
The package provides a streamlined way to measure multiple access metrics from one origin to multiple destinations 

* Prepare input data files (Network, origin and destination layers) inside a folder  ``Cities/city_name/Data``
* Prepare and fill a pairing table and save it in the input folder
* Run this code snippet::

   from madina.una.workflows import betweenness_flow_simulation

    KNN_accessibility(
        city_name='Somerville',
        pairings_file="pairing.csv",
        num_cores=8,
    )

.. toctree::
   :maxdepth: 2
   :caption: Instructions for Preparing and Running Accessibility workflows:

   ped_access


.. image:: img//access_map_3_panels.png
  :width: 600
  :alt: Instruction Guide


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



.. image:: img//flow_map_3_panels.png
  :width: 600
  :alt: Instruction Guide

.. image:: img//NYC_homes_to_amenities.jpeg
  :width: 600
  :alt: Instruction Guide



.. toctree::
   :maxdepth: 5
   :caption: Documentation: Package Modules, Classes and Functions


   madina.una
   madina.zonal
