import os
from madina.una.workflows import KNN_accessibility


if __name__ == '__main__':
    city_name = 'NYC'
    KNN_accessibility(
        city_name=city_name,
        #data_folder=os.path.join('examples', 'Cities', city_name, 'Data'),
        #output_folder=os.path.join('examples', 'Cities', city_name, 'Simulations', 'Baseline'),
        pairings_file="knn_pairings.csv",       
        num_cores=8,
    )

 