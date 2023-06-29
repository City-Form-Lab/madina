print("hello... its me..")


# create a zonal object, load sample network, origin, destination


# create network


# Test out n_o_scope

# test out subgraph and alternative path generation


# have a loop of betweennerss settings, tested aginst various result files coming from Rhino



### Generare report about inputs, outputs, statistice, betweenness summary...

### HARVARD SQUARE TEST CASE ###


tests = [
    {
        'test_name': '_Betweenness1',
        'Origin weight': None, 
        'Destination weight': None,
        'Search radius': 300,
        'Detour Ratio': 1,
        'Elastic weights': False, 
        'Turns': False,
        'Turn threshold': 0, 
        'Turn penalty':	0,
    }, 
    {
        'test_name': '_Betweenness2',
        'Origin weight': 'people', 
        'Destination weight': None,
        'Search radius': 300,
        'Detour Ratio': 1,
        'Elastic weights': False, 
        'Turns': False,
        'Turn threshold': 0, 
        'Turn penalty':	0,
    }, 
    {
        'test_name': '_Betweenness3',
        'Origin weight': 'people', 
        'Destination weight': 'lines',
        'Search radius': 300,
        'Detour Ratio': 1,
        'Elastic weights': False, 
        'Turns': False,
        'Turn threshold': 0, 
        'Turn penalty':	0,
    }, 
    {
        'test_name': '_Betweenness4',
        'Origin weight': 'people', 
        'Destination weight': 'lines',
        'Search radius': 300,
        'Detour Ratio': 1.15,
        'Elastic weights': False, 
        'Turns': False,
        'Turn threshold': 0, 
        'Turn penalty':	0,
    }, 
    {
        'test_name': '_Betweenness5',
        'Origin weight': 'people', 
        'Destination weight': 'lines',
        'Search radius': 300,
        'Detour Ratio': 1.15,
        'Elastic weights': True, 
        'Turns': False,
        'Turn threshold': 0, 
        'Turn penalty':	0,
    }, 
    {
        'test_name': '_Betweenness6',
        'Origin weight': 'people', 
        'Destination weight': 'lines',
        'Search radius': 300,
        'Detour Ratio': 1.15,
        'Elastic weights': True, 
        'Turns': True,
        'Turn threshold': 45, 
        'Turn penalty':	30,
    }
]

buildings_file = r'C:\Users\abdul\Dropbox (MIT)\PhD Thesis\Madina\UNA test\building_entranceas.geojson'
subway_file = r'C:\Users\abdul\Dropbox (MIT)\PhD Thesis\Madina\UNA test\subway.geojson'
network_file = r'C:\Users\abdul\Dropbox (MIT)\PhD Thesis\Madina\UNA test\network.geojson'

import sys
sys.path.append('../')
from madina.zonal.zonal import Zonal
harvard_square = Zonal()

harvard_square.load_layer(
    layer_name='streets',
    file_path=network_file
    )


harvard_square.load_layer(
    layer_name='buildings',
    file_path=buildings_file
    )




harvard_square.load_layer(
    layer_name='subway',
    file_path=buildings_file
    )