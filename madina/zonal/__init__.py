from madina.zonal.layer import *
from madina.zonal.network import *
from madina.zonal.network_utils import *
from madina.zonal.zonal import *
from madina.zonal.zonal_utils import *


# We set_origin and destination
# network idea extensibility
# rename `create_street_network` to `create_network`

"""
To do analysis, you need to load three layers at minimum
Create a topological network
add origins and destinations -- set_origins() and set_destinations(). Wrapper around insert_node


create the graph network --moving this to internal would make sense

set_network
set_origins
set_destination

Layers
- make the layers get_item work for the Zonal as a whole
- refactor it to allow layer styling
- figure out conda AND pip hosting
"""
