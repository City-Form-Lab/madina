# madina
A library to simplify creating workflows for urban planning research. Tools to work with geometry, networks and geoprocessing

## Code Structure (06/05/2023)

- `zonal`: contains the Layer(s), Zonal and Network classes
  - `Layer` class: a container for geographic layers
  - `Layers` class: a data structure to organize multiple `Layer` objects with an order
  - `Zonal` class: a collection of geographic layers and the Network they generate.
    - `zonal.py`: The class itself
    - `zonal_utils.py`: Currently a collection of non-computation functionalities. Will be separated into different sub-collections (e. g. map cleaning, visualization, etc.)
  - `Network` class: street networks generated from geographic layers with nodes, edges, and special nodes
    - `network.py`: The class itself
    - `network_utils.py`: Currently containing some function skeletons not used, will be used to migrate some helper functions in
- `una`: contains operations of UNA
  - `betweenness.py`: Entry points to betweenness calculation
  - `paths.py`: Functions for finding all paths between one O-D pair
  - `elastic.py`: Entry point to Elastic Weight calculation
  - `una_utils.py`: Helper functions used by multiple UNA operations

## TODO List (06/05/2023)

- Add turn penalty argument to the betweenness pipeline
- Rewrite specs for most UNA functions
- Add map cleaning, visualization, and generative geometry