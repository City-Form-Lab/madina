# Madina


![Example of trip flows generated for New York City](/docs/source/img/NYCFlows.jpeg "Example of flows generated for New York CIty")

> [!IMPORTANT]
> **Madina has been superseded by the [UNA Python package](https://github.com/City-Form-Lab/urban_network_analysis)**,
> which continues its development under a new API with expanded accessibility metrics,
> a scalable aggregate-flow engine, and full documentation at
> [urban-network-analysis.readthedocs.io](https://urban-network-analysis.readthedocs.io).
> This repository is preserved for reference and for reproducibility of
> Sevtsuk & Alhassan (2025), *Journal of Transport Geography*, which describes it.
> New users should start with UNA.

**About the Archived Madina Version**

Madina *(Arabic for the word 'city')* is a package of classes and functions to streamline the representation and analysis of urban networks. The package includes a python implemetation of the ***Urban Network Analysis Toolbox*** ([Homepage](https://cityform.mit.edu/projects/una-rhino-toolbox) - [User Guide](https://unatoolbox.notion.site/)). More detailed documentation of the package is available [here](https://madinadocs.readthedocs.io/).

To reference this package in your research, you can cite the paper available on SSRN:

Sevtsuk, A., & Alhassan, A. (2025). **Madina Python Package: Scalable Urban Network Analysis for Modeling Pedestrian and Bicycle Trips in Cities.** Journal of Transportation Geography, Online (Feb 2025). https://doi.org/10.1016/j.jtrangeo.2025.104130

```
 @article{Sevtsuk2025,
   author = {Andres Sevtsuk and Abdulaziz Alhassan},
   journal = {Journal of Transportation Geography},
   title = {Madina Python Package: Scalable Urban Network Analysis for Modeling Pedestrian and Bicycle Trips in Cities},
   volume = {Online (Feb 2025)},
   url = {https://doi.org/10.1016/j.jtrangeo.2025.104130},
   year = {2025}
}

```


## Highlights
* Organization of data layers using [Geopandas](https://geopandas.org/en/stable)
* Creation of topological (Routable) networks from a geometric representaion. Networks are represented using [NetworkX](https://networkx.org/)
* Insertion of origin and destination nodes from data layers into topological networks
* Creating maps using [DeckGL](https://deck.gl/) with various streamlined styling options
* Improved implementation of [UNA Tools](https://cityform.mit.edu/projects/una-rhino-toolbox) that use multiprocessing and novel path generation algorithoms to enable effecient pedestrian accessibility and flow simulations on large-scale networks.
* Added functionalities for UNA, including percieved segment costs that can account for segment quality attributes, elastic trip generation with respect to destination availability, KNN-Access metrics that allows WalkScore-type access calculations, turn penalties, etc.
* Automated workflows for pedestrian flow simulation in urban environments.
* User-friendly workspace environment that requires minimal coding experience.


## Pedestrain Flow Simulations
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

Detailed instructions are available in the documentation [here](https://madinadocs.readthedocs.io/).

## Library Structure
* `Zonal` class: This is the main class that the user interacts with. A user would create a Zonal object, populate it with data layers and calls functions to create a network object within a Zonal object. 
* `Network` Class: Created inside a Zonal ovject to represent a network of origins, destinations and 'street' connections. This object is used internally as input to most network algorithms.
* `UNA` Module: A set of functions implementing the UNA functionalities. Each function tales a `Zonal` object as input.
* `Workflows` module: A set of standarized workdlows that takes a set of structured inpurs. Examples for Pedesstrain flow simulartiob

