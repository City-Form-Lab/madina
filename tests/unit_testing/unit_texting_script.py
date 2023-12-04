
if __name__ == '__main__':
    import os
    os.environ['USE_PYGEOS'] = '0'
    import sys
    #sys.path.append('../../')
    import os
    print(os.getcwd())
    sys.path.append(os.getcwd())
    #print (sys.path)
    from madina.zonal.zonal import Zonal
    #from madina import Zonal
    #import madina
    from madina.una.betweenness import parallel_betweenness, paralell_betweenness_exposure, betweenness_exposure


    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import time 



    print (f"{'test name':30s} | {'madina_flow_sum':15s} | {'Rhino_flow_sum':15s} | {'sum_diff':8s} | {'sum_smlr_%':10s} | {'sle_mean':8s} | {'sle_median':10s} | {'sle_max':8s} | {'sle>0.1%':8s} | {'sle>1.0%':8s} | {'sle>3.0%':8s} | {'sle>5.0%':8s} | {'time_spent':10s}")
    #for test_case in os.listdir("Test Cases"):
    for test_case in ['Harvard Square', 'Somerville']:   # 'Harvard Square',
        start = time.time()
        # TODO: Check OS compatibility, ensure this is compatible with Unix systems..
        test_case_folder = 'tests\\unit_testing\\' "Test Cases" + "\\" + test_case + "\\"
        test_config = pd.read_csv(test_case_folder + "test_configs.csv")
        test_flows =  pd.read_csv(test_case_folder + "test_flows.csv")

        if test_case == 'Somerville':
            ready_tests = test_config.index[0:3]
        else:
            ready_tests = test_config.index

        for test_idx in ready_tests:

            harvard_square = Zonal()

            harvard_square.load_layer(
                layer_name='streets',
                file_path=  test_case_folder + test_config.at[test_idx, 'Network_File']
                )

            harvard_square.load_layer(
                layer_name=test_config.at[test_idx, 'Origin_Name'],
                file_path= test_case_folder + test_config.at[test_idx, 'Origin_File']
                )

            harvard_square.load_layer(
                layer_name=test_config.at[test_idx, 'Destination_Name'],
                file_path= test_case_folder + test_config.at[test_idx, 'Destination_File']
                )
            

            harvard_square.create_street_network(
                source_layer='streets', 
                discard_redundant_edges=True,
                split_redundant_edges=False,
                node_snapping_tolerance=0.1,  # TODO: check for sensitivity... pick one as default snapping.
                weight_attribute=test_config.at[test_idx, 'Network_Cost'] if test_config.at[test_idx, 'Network_Cost'] != "Geometric" else None
            )

            #print (f"{harvard_square.network.edges.shape = }")
            #print (f"{harvard_square.network.nodes.shape = }")
            #harvard_square.network.nodes,  harvard_square.network.edges = _discard_redundant_edges(harvard_square.network.nodes, harvard_square.network.edges)
            #print (f"{harvard_square.network.edges.shape = }")
            #print (f"{harvard_square.network.nodes.shape = }")

            harvard_square.insert_node(
                layer_name=test_config.at[test_idx, 'Origin_Name'], 
                label='origin', 
                weight_attribute=test_config.at[test_idx, 'Origin_Weight'] if test_config.at[test_idx, 'Origin_Weight'] != "Count" else None
            )
            harvard_square.insert_node(
                layer_name=test_config.at[test_idx, 'Destination_Name'], 
                label='destination', 
                weight_attribute=test_config.at[test_idx, 'Destination_Weight'] if test_config.at[test_idx, 'Destination_Weight'] != "Count" else None
            )

            harvard_square.create_graph()

            node_gdf = harvard_square.network.nodes
            origin_gdf = node_gdf[node_gdf['type'] == 'origin']

            harvard_square.network.nodes["original_weight"] = harvard_square.network.nodes["weight"]

            harvard_square.network.turn_penalty_amount = test_config.at[test_idx, 'Turn_Penalty']
            harvard_square.network.turn_threshold_degree = test_config.at[test_idx, 'Turn_Threshold']

            if test_config.at[test_idx, 'Elastic_Weights']:
                '''
                harvard_square.network.nodes["weight"] = harvard_square.network.nodes["original_weight"]
                get_elastic_weight(
                    harvard_square.network,
                    search_radius=test_config.at[test_idx, 'Radius'],
                    detour_ratio=test_config.at[test_idx, 'Detour'],
                    beta=test_config.at[test_idx, 'Beta'],
                    decay=True, #test_config.at[test_idx, 'Decay'],
                    #turn_penalty=test_config.at[test_idx, 'Turns'],
                    turn_penalty=False,
                )
                for o_idx in origin_gdf.index:
                    harvard_square.network.nodes.at[o_idx, 'weight'] =  harvard_square.network.nodes.at[o_idx, 'elastic_weight']
                '''
                continue
            '''
            return_dict = parallel_betweenness(
                harvard_square.network,
                search_radius=test_config.at[test_idx, 'Radius'],
                detour_ratio=test_config.at[test_idx, 'Detour'],
                decay=test_config.at[test_idx, 'Decay'], #if test['Elastic weights'] else True,
                decay_method=test_config.at[test_idx, 'Decay_Mode'],  # "power", "exponent"
                beta=test_config.at[test_idx, 'Beta'],
                path_detour_penalty='equal', # "equal",  # "power", "exponent", "equal"
                origin_weights=False if type(test_config.at[test_idx, 'Origin_Weight']) != str else True,
                closest_destination=test_config.at[test_idx, 'Closest_Destination'],
                destination_weights=False if type(test_config.at[test_idx, 'Destination_Weight']) != str  else True,    #or (test['Elastic weights'])
                # perceived_distance=False,
                num_cores=6,
                light_graph=True,
                turn_penalty=test_config.at[test_idx, 'Turns'],
            )
            
            '''
            harvard_square.network.max_chunck_size = 50
            harvard_square.network.chunking_method = 'pizza_chunks' # ['no_chunking', 'cocentric-chunks', 'random_chunks', 'pizza_chunks']
            return_dict = paralell_betweenness_exposure(
                harvard_square,
                search_radius=test_config.at[test_idx,'Radius'],
                detour_ratio=test_config.at[test_idx,'Detour'],
                decay=False if test_config.at[test_idx,'Elastic_Weights'] else test_config.at[test_idx,'Decay'],  # elastic weight already reduces origin weight factoring in decay. if this pairing uses elastic weights, don't decay again,
                decay_method=test_config.at[test_idx,'Decay_Mode'],
                beta=test_config.at[test_idx,'Beta'],
                num_cores=3,
                path_detour_penalty='equal', # "power" | "exponent" | "equal"
                closest_destination=test_config.at[test_idx,'Closest_Destination'],
                elastic_weight=test_config.at[test_idx,'Elastic_Weights'],
                turn_penalty=test_config.at[test_idx,'Turns'],
                path_exposure_attribute=None,
                return_path_record=False, 
                destniation_cap=None
            )

            



            simulated_sum_of_flow = return_dict['edge_gdf']['betweenness'].sum()
            test_flow = test_flows[test_config.at[test_idx, 'Flow_Name']].sum()


            ## create segment level comparison
            # creating connector lines

            import shapely.geometry as geo
            simulated_betweenness = return_dict['edge_gdf'][['betweenness', 'parent_street_id']].rename(columns={"betweenness": "simulated_betweenness"}).drop_duplicates(subset=['parent_street_id']).set_index("parent_street_id")
            simulated_betweenness = harvard_square.layers["streets"].gdf[["geometry", "__GUID"]].join(simulated_betweenness).set_index("__GUID")

            test_name = test_config.at[test_idx, 'Flow_Name']
            test_betweenness = test_flows[['__GUID', test_name]].set_index("__GUID").rename(columns = {test_name: "test_flow"})


            comparison = simulated_betweenness.join(test_betweenness)
            comparison["difference"] = comparison["simulated_betweenness"] - comparison["test_flow"]
            comparison["difference_pct"] = abs(comparison["difference"]) / comparison["simulated_betweenness"] *100
            # segment level error
            sle = comparison[~comparison["difference_pct"].isna()]['difference_pct']

            
            node_gdf = harvard_square.network.nodes
            edge_gdf = harvard_square.network.edges


            origin_nodes = node_gdf[node_gdf['type'] == 'origin']
            '''
            origin_layer = harvard_square.layers[test_config.at[test_idx, 'Origin_Name']].gdf
            origin_joined = origin_layer.join(origin_nodes.set_index('source_id'),lsuffix='_origin')
            origin_joined['connector_line'] = origin_joined.apply(lambda x:geo.LineString([x['geometry'], x["geometry_origin"]]), axis=1)
            origin_joined["geometry"] = origin_joined['connector_line']


            destination_layer = harvard_square.layers[test_config.at[test_idx, 'Destination_Name']].gdf
            destination_nodes = node_gdf[node_gdf['type'] == 'destination']
            destination_joined = destination_layer.join(destination_nodes.set_index('source_id'),lsuffix='_destination')
            destination_joined['connector_line'] = destination_joined.apply(lambda x:geo.LineString([x['geometry'], x["geometry_destination"]]), axis=1)
            destination_joined["geometry"] = destination_joined['connector_line']


            #print (f"{origin_nodes.shape = }\t{destination_nodes.shape = }")

            streets = harvard_square.layers["streets"].gdf 
            network_file = gpd.read_file(test_case_folder + test_config.at[test_idx, 'Network_File'], engine='pyogrio')


            flow_difference = comparison[
                ((comparison['test_flow' ] > 0) & (comparison['simulated_betweenness'] == 0)) | 
                ((comparison['test_flow' ] == 0) & (comparison['simulated_betweenness'] > 0))
            ]
            harvard_square.create_map(
                [
                    #{'gdf': streets[streets[test_config.at[test_idx, 'Network_Cost']] > 0], 'color': [255, 255, 255], 'text': test_config.at[test_idx, 'Network_Cost']},
                    {'gdf': streets, 'color': [100, 100, 100], 'opacity': 0.1},
                    {'gdf': edge_gdf[edge_gdf['betweenness'] > 0], 'color': ['125, 125, 0'], 'text': 'betweenness', 'opacity': 0.2},
                    #{'gdf': comparison[abs(comparison["difference"]) > 0.01], 'color_by_attribute': 'difference', 'color_method': 'gradient', 'text': 'difference'},
                    {'gdf': comparison[(comparison["difference_pct"] >= 0.1) & (comparison["difference_pct"] < 100)], 'color_by_attribute': 'difference_pct', 'color_method': 'gradient', 'text': 'difference_pct'},
                    {'gdf': edge_gdf[edge_gdf['snapped'] == True], 'color': [255, 0, 255], 'opacity': 0.2},
                    {'gdf': network_file[network_file["geometry"].geom_type == 'Polygon'], 'color': [125, 0, 125], 'text': '__GUID'},
                    {'gdf': flow_difference, 'color': [255, 255, 0] , 'text': 'difference'},        
                    #{'gdf': comparison[comparison['difference'] != 0], 'color_by_attribute': 'difference', 'color_method': 'gradient', 'text': 'difference', 'opacity':  0.1},
                    {'gdf': origin_layer, 'color': [0, 0, 255]},
                    {'gdf': origin_joined[['geometry']], 'color': [0, 0, 255]},
                    {'gdf': destination_layer, 'color': [255, 0, 0]},
                    {'gdf': destination_joined[['geometry']], 'color': [255, 0, 0]},
                    {'gdf': harvard_square.network.nodes[['geometry', 'type', 'weight']].reset_index(), 'color': [255, 0, 255], 'text': 'id'},
                ],
                save_as="Test Cases\\" + test_case + '\\'  + test_name + "._difference_map.html"
            )
            '''
            
            #print (test_config.loc[test_idx])
            print (f"{test_config.at[test_idx, 'Flow_Name'][:30]:30s} | {simulated_sum_of_flow:15.2f} | {test_flow:15.2f} | {simulated_sum_of_flow - test_flow:8.2f} | {1-(simulated_sum_of_flow - test_flow)/ test_flow:10.2%} | {sle.mean():7.4f}% | {sle.median():9.4f}% | {sle.max():7.4f}% | {sle[sle > 0.1].count():8} | {sle[sle > 1.0].count():8} | {sle[sle > 3.0].count():8} | {sle[sle > 5.0].count():8} | {time.time() - start: 10.5f}")
            #print ("DOne Case...")
            #break
        break