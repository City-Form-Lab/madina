# Madina


![Example of trip flows generated for New York City](/docs/source/img/NYCFlows.jpeg "Example of flows generated for New York CIty")



Madina *(Arabic for the word 'city')* is a package of classes and functions to streamline the representation and analysis of urban networks. The package provide layer management (Similar to layers in CAD/GIS), urban network representation and visualization. The package also includes a python implemetation of the ***Urban Network Analysis Toolbox*** ([Homepage](https://cityform.mit.edu/projects/una-rhino-toolbox) - [User Guide](https://unatoolbox.notion.site/)). More detailed documentation of the package is available [here](https://madinadocs.readthedocs.io/)

To reference this package in your research, you can cite the paper available on SSRN:

Alhassan, Abdulaziz and Sevtsuk, Andres, **Madina Python Package: Scalable Urban Network Analysis for Modeling Pedestrian and Bicycle Trips in Cities.** Available at SSRN: https://ssrn.com/abstract=4748255 or http://dx.doi.org/10.2139/ssrn.4748255

```
@article{alhassan2024madina,
  title={Madina Python Package: Scalable Urban Network Analysis for Modeling Pedestrian and Bicycle Trips in Cities},
  author={Alhassan, Abdulaziz and Sevtsuk, Andres},
  journal={SSRN},
  year={2024},
  publisher={Elsevier}, 
  doi={10.2139/ssrn.4748255},
  url={https://ssrn.com/abstract=4748255}
}
```


## HIghlights
* Organization of data layers using [Geopandas](https://geopandas.org/en/stable)
* Creation of topological (Routable) networks from a geometric representaion. Network is represented using [NetworkX](https://networkx.org/)
* Insertion of origin and destination points from data layers into topological network
* Creating maps using [DeckGL](https://deck.gl/) with various streamlined styling options
* Improved implementation of [UNA Tools](https://cityform.mit.edu/projects/una-rhino-toolbox) that uses multiprocessing and novel path generation algorithoms to enable effecient betweenness flow simulations on large scale.
* Added functionalities for UNA including percieved distance, elastic trip generation, turn penalties,
* streamlined workflow for pedestrian flow simulation in urban environments.

## Pedestrain Flow Simulation
The package features a streamlined way to model pedestrian activity in urban areas between pairs of pre-specified origins and destinations. This can be done by following these steps:
* Prepare input data files for the network, and each origin and destination. Place all data in a folder called `Cities/city_name/Data`
* Fill in the pairing table to specify origin-destination pairs, and specify specific parameters for each pair. Save the filled pairing table in the same `Cities/city_name/Data` folder
* run the simulation:
```
from madina.una.betweenness import betweenness_flow_simulation

betweenness_flow_simulation(
    city_name="new_york"
)
```
* Output would be saved in `Cities/city_name/Simulations`.
* More instructions on running a pedestrain flow simulation, preparing data and creating the pairing table are found in the documentation [here](https://madinadocs.readthedocs.io/)


![Simulated Flow in New York City](docs/source//img//NYC_homes_to_amenities.jpeg)

## Installation
First, install geopandas through conda in a new environment
```
conda create -n madina_env -c conda-forge --strict-channel-priority geopandas
```
Activate the newly created environment

```
conda activate madina_env
```
Install Madina through pip
```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple madina
```

Detailed instructions are available in the documentation here

## Library Structure
* `Zonal` class: This is the main class that the user interacts with. A user would create a Zonal object, populate it with data layers and calls functions to create a network object within a Zonal object. 
* `Network` Class: Created inside a Zonal ovject to represent a network of origins, destinations and 'street' connections. This object is used internally as input to most network algorithms.
* `UNA` Module: A set of functions implementing the UNA functionalities. Each function tales a `Zonal` object as input.
* `Workflows` module: A set of standarized workdlows that takes a set of structured inpurs. Examples for Pedesstrain flow simulartiob

