import os
from madina.una.workflows import betweenness_flow_simulation


if __name__ == '__main__':
    city_name = 'Milan_heat'
    #city_name = 'Somerville'

    betweenness_flow_simulation(
        city_name=city_name,
        #data_folder=os.path.join('examples', 'Cities', city_name, 'Data'),
        #output_folder=os.path.join('examples', 'Cities', city_name, 'Simulations', 'Baseline'),
        pairings_file="pairings.csv",       
        num_cores=6,
    )