import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
import json
import torch
import torch.nn as nn
from math import sin, cos, sqrt, atan2, radians, exp
import copy
from sklearn.cluster import *
import random
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.neighbors import KDTree
import os
import argparse
import datetime
import warnings


import python.data as data
import python.display as dp
import python.voxels as voxel
import python.metric as metric
import python.clustering as cl
import python.RNN as RNN
import python.validation as validation
import python.graphs as graphs  

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None



def create_dict_modif(G, dict_cluster, df_simplified):
    """
    Creates 2 dictionaries containing the modifications to apply to a road graph with respect to observations.
    The first dictionay contains modifications with respect to clusters of observations, the second with respect to all observations at once.

    Parameters
    ----------
    G : MultiDiGraph (Networkx)
        The road graph that will need modifications.       
    dict_cluster : dict
        Dict containing the information about the clusters. The keys are cluster numbers.
        The values are lists containing the number of the routes in each cluster.
    df_simplified : DataFrame with columns=['lat', 'lon', 'route_num']
        Dataframe containing the observations.
    Returns
    -------
    dict_modif : dict
        A dictionary of dictionary. The keys are cluster numbers.
        The values are dictionaries which keys are edges under the form 'n1;n2' (n1 and n2 are nodes of G).
        The values of the dictionaries are coefficients in [0, 1] to multiply with the weight of the edge 
        to make the modification.
    dict_modif_global : dict
        Same that dict_modif but with only one cluster containing all roads.
    """
    dict_modif = {}
    dict_modif_global = {0:{}}
    print("Creating the dictionaries containing the road graph's modifications, this might take a while depending on the road graph size but \
    this operation only needs to be done once.")
    print("start:", datetime.datetime.now().time())
    dict_dict_voxels_cluster = {}
    num_nodes_processed = 0
    for cl in dict_cluster:
        if(cl > -1):
            df_cluster = pd.DataFrame(columns=["lat", "lon", "route_num"])
            for num_route in range(len(dict_cluster[cl])):
                df_temp = df_simplified[df_simplified["route_num"]==dict_cluster[cl][num_route]]
                df_temp["route_num"] = num_route
                df_cluster = df_cluster.append(df_temp)
            _, _, dict_voxels_cluster = voxel.generate_voxels(df_cluster, df_cluster.iloc[0]["route_num"], df_cluster.iloc[-1]["route_num"])
            dict_dict_voxels_cluster[cl] = dict_voxels_cluster

    df_cluster_global = pd.DataFrame(columns=["lat", "lon", "route_num"])
    for num_route in range(len(tab_clusters)+1):
        df_temp = df_simplified[df_simplified["route_num"]==num_route]
        df_temp["route_num"] = num_route
        df_cluster_global = df_cluster_global.append(df_temp)
    _, _, dict_voxels_cluster_global = voxel.generate_voxels(df_cluster_global, df_cluster_global.iloc[0]["route_num"], df_cluster_global.iloc[-1]["route_num"])
    for v in G:
        print("\rNode {}/{}.".format(num_nodes_processed, G.number_of_nodes()), end="")
        for v_n in G[v]:
            df_line = pd.DataFrame([[G.nodes[v]['y'], G.nodes[v]['x'], 0], [G.nodes[v_n]['y'], G.nodes[v_n]['x'], 0]], columns=["lat", "lon", "route_num"])
            tab_voxels, _, _ = voxel.generate_voxels(df_line, 0, 0)
            for cl in dict_dict_voxels_cluster:
                nb_vox_found = 0
                tot_coeff = 0
                dict_voxels_cluster = dict_dict_voxels_cluster[cl]
                for vox in tab_voxels[0]:
                    if vox in dict_voxels_cluster:
                        nb_vox_found += 1
                        tot_coeff += dict_voxels_cluster[vox]["cyclability_coeff"]
                if(nb_vox_found > 0):
                    if(cl not in dict_modif):
                        dict_modif[cl] = {}
                    tot_coeff /= nb_vox_found
                    dict_modif[cl][str(v)+";"+str(v_n)] = tot_coeff

            nb_vox_global_found = 0
            tot_coeff_global = 0
            for vox in tab_voxels[0]:
                if vox in dict_voxels_cluster_global:
                    nb_vox_global_found += 1
                    tot_coeff_global += dict_voxels_cluster_global[vox]["cyclability_coeff"]
            if(nb_vox_global_found > 0):
                tot_coeff_global /= nb_vox_global_found
                dict_modif_global[0][str(v)+";"+str(v_n)] = tot_coeff_global
        num_nodes_processed += 1

    print("end:", datetime.datetime.now().time())
    return dict_modif, dict_modif_global


    

def create_path_compute_distance(d_point, f_point, df, tree, G, nodes, global_metric):

    route = data.pathfind_route_osmnx(d_point, f_point, tree, G, nodes)
    route_coord = [[G.nodes[x]["y"], G.nodes[x]["x"]] for x in route]
    route_coord = [x + [0] for x in route_coord]

    df_route = pd.DataFrame(route_coord, columns=["lat", "lon", "route_num"])
    df_route = data.rd_compression(df_route, 0, 1, verbose=False)
   

    tab_route_voxels, _, _ = voxel.generate_voxels(df_route, 0, 0)
    df["route_num"] = 1
    df["type"] = 0     
    df_route["type"] = 2
    df_coeff = df_route.append(df)

    tab_voxels, tab_voxels_global, dict_voxels = voxel.generate_voxels(df_coeff, 0, 1)

    #\\\DEBUG///
    '''tab_voxels_min_route = voxel.get_voxels_with_min_routes(dict_voxels, 2, global_metric)
    df = pd.DataFrame(tab_voxels_min_route, columns=["lat", "lon", "route_num", "type"])
    df_coeff = df_coeff.append(df)
    dp.display_mapbox(df_coeff, color="type")'''
    

    if(global_metric):
        coeff = metric.get_distance_voxels(0, 1, tab_voxels_global)
    else:
        coeff = metric.get_distance_voxels(0, 1, tab_voxels)

    return df_route, tab_route_voxels[0], coeff


def modify_network_graph(cl, dict_modif, G, coeff_diminution = 1):
    for key in dict_modif[cl]:
        vertexes = key.split(";")
        v = int(vertexes[0])
        v_n = int(vertexes[1]) 
        if(v in G):
            G[v][v_n][0]['length'] -= G[v][v_n][0]['length']*dict_modif[cl][key] #min(1, exp(dict_modif[cl][key])-1)
        else: 
            print("ERROR IN GRAPH MODIFICATION!")
            return False
    return True



def cancel_network_graph_modifications(cl, dict_modif, G, G_base):
    for key in dict_modif[cl]:
        vertexes = key.split(";")
        v = int(vertexes[0])
        v_n = int(vertexes[1])
        if(v in G):
            G[v][v_n][0]['length'] = G_base[v][v_n][0]['length']
        else:
            break

    
    
            
        


def main_global(global_metric, deviation):

    tab_coeff_modified = []

    tab_diff_coeff = []

    global G
    modify_network_graph(0, dict_modif_global, G)

    print("GLOBAL :")

    for i, num_route in enumerate(tab_num_routes):

        print("\rRoute {}/{}".format(i, len(tab_num_routes)-1), end="")

        df_route_tested = df_simplified[df_simplified["route_num"]==num_route]

        d_point = df_route_tested.iloc[0].tolist()[:2]
        f_point = df_route_tested.iloc[-1].tolist()[:2]

        df_route, tab_route_voxels, coeff_modified = create_path_compute_distance(d_point, f_point, df_route_tested, tree, G, nodes, global_metric)

        tab_coeff_modified.append(min(coeff_modified))
    
    
    print()
    print("Mean modified path distance:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("===============================")

    return tab_coeff_modified
        

def main_clusters(global_metric, deviation):

    tab_coeff_modified = []

    tab_diff_coeff = []

    for key in dict_cluster:
        if(key != -1):
            df_temp = df_simplified[df_simplified["route_num"]==dict_cluster[key][0]]

            modify_network_graph(key, dict_modif, G)

    print("ORACLE :")
    
    for i, num_route in enumerate(tab_num_routes):

        print("\rRoute {}/{}".format(i, len(tab_num_routes)-1), end="")

        df_route_tested = df_simplified[df_simplified["route_num"]==num_route]
        
        
        d_point = df_route_tested.iloc[0].tolist()[:2]
        f_point = df_route_tested.iloc[-1].tolist()[:2]

        df_route, tab_route_voxels, coeff_modified = create_path_compute_distance(d_point, f_point, df_route_tested, tree, G, nodes, global_metric)

        tab_coeff_modified.append(min(coeff_modified))


    print()
    print("Mean modified path distance:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("===============================")
    

    return tab_coeff_modified

    


    

def main_clusters_NN(dict_tab_route_voxels, df_computed, global_metric, deviation, full_print=False):
    


    with open("./files/"+project_folder+"/neural_networks/network.param",'rb') as infile:
        param = pickle.load(infile)

    size_data = 1

    network = RNN.RNN_LSTM(size_data, max(tab_clusters)+1, param.hidden_size, param.num_layers, param.bidirectional, param.dropout)
    network.load_state_dict(torch.load("files/"+project_folder+"/neural_networks/network_temp.pt"))
    network.eval()

    nb_good_predict = 0

    tab_coeff_modified = [[], []]

    tab_diff_coeff = [[], []]

    print("LSTM :")
    tab_num_routes.sort()
    for i, num_route in enumerate(tab_num_routes): #len(tab_clusters)):
        print("\rRoute {}/{}".format(i, len(tab_num_routes)-1), end="")

        good_predict = False
        df_route_tested = df_simplified[df_simplified["route_num"]==num_route]
        
        d_point = df_route_tested.iloc[0].tolist()[:2]
        f_point = df_route_tested.iloc[-1].tolist()[:2]


        cl, nb_new_cluster = validation.find_cluster(dict_tab_route_voxels[num_route], network, param.voxels_frequency, dict_voxels_clustered, 
                                    kmeans, df_route_tested)
        if(cl == tab_clusters[num_route]):
            #print("good predict")
            nb_good_predict += 1
            good_predict = True
            #dp.display_cluster_heatmap_mapbox(df_simplified, dict_cluster[cl])
        #dp.display_mapbox(df_route)

        ################################################################################_

        modify_network_graph(cl, dict_modif, G)


        df_route_modified,_,coeff_modified = create_path_compute_distance(d_point, f_point, df_simplified[df_simplified["route_num"]==num_route], tree, G, nodes, global_metric)

        cancel_network_graph_modifications(cl, dict_modif, G, G_base)
        
        df_sp = df_computed[df_computed["route_num"]==num_route]
        
        df_route_tested["route_num"] = 0 #orange
        df_route_tested["type"] = 0
        
        df_sp["route_num"] = 1 #bleu
        df_sp["type"] = 1
        
        df_route_modified["route_num"]= 2 #rouge
        df_route_modified["type"] = 2
        
        
        df_route_modified = df_route_modified.append(df_sp)
        df_route_modified = df_route_modified.append(df_route_tested)
        '''map = dp.display(df_route_modified, color="route_num")
        map.save("files/"+project_folder+"/images/maps/"+str(num_route)+".html")'''
        

        if(good_predict):
            tab_coeff_modified[0].append(min(coeff_modified))
        else:
            tab_coeff_modified[1].append(min(coeff_modified))

    print()     
    if(full_print):
        print("===============================")
        print("GOOD PREDICTIONS :")
        print("===============================")
        print("Mean modified path distance:", sum(tab_coeff_modified[0])/len(tab_coeff_modified[0])*100, "%")
        print("===============================")
        print("BAD PREDICTIONS :")
        print("===============================")
        print("Mean modified path distance:", sum(tab_coeff_modified[1])/len(tab_coeff_modified[1])*100, "%")
        print("===============================")
        
    print("Accuracy:", len(tab_coeff_modified[0])/(len(tab_coeff_modified[0])+len(tab_coeff_modified[1]))*100, "%")
    print("Mean modified path distance:", sum(sum(tab_coeff_modified,[]))/sum(len(row) for row in tab_coeff_modified)*100, "%")


    return tab_coeff_modified[0]+tab_coeff_modified[1]




def main_clusters_full_predict(global_metric, deviation):

    tab_coeff_modified = []

    tab_diff_coeff = []

    print("ORACLE :")

    for i, num_route in enumerate(tab_num_routes): #len(tab_clusters)):

        print("\rRoute {}/{}".format(i, len(tab_num_routes)-1), end="")

        df_route_tested = df_simplified[df_simplified["route_num"]==num_route]
        df_mapbox = df_mapbox_routes_test[df_mapbox_routes_test["route_num"]==num_route]
        
        d_point = df_route_tested.iloc[0].tolist()[:2]
        f_point = df_route_tested.iloc[-1].tolist()[:2]

        modify_network_graph(tab_clusters[num_route], dict_modif, G)


        df_route_modified,_,coeff_modified = create_path_compute_distance(d_point, f_point, df_simplified[df_simplified["route_num"]==num_route], tree, G, nodes, global_metric)

        cancel_network_graph_modifications(tab_clusters[num_route], dict_modif, G, G_base)
        
        tab_coeff_modified.append(min(coeff_modified))
        
        '''df_route_tested["route_num"] = 0
        df_route_modified["route_num"] = 1
        df_mapbox["route_num"] = 2
        df_route["route_num"] = 3
        
        df_display = df_route_tested.append(df_route_modified)
        df_display = df_display.append(df_route)
        df_display = df_display.append(df_mapbox)
        dp.display_mapbox(df_display, color="route_num")'''

       
    
    print()
    print("Mean modified path distance:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("===============================")

    return tab_coeff_modified







def main_mapbox(global_metric):
    global df_simplified
    global df_mapbox_routes_test
        
    tab_coeff_modified = []
    
    print("MAPBOX :")

    for i, num_route in enumerate(tab_num_routes):

        print("\rRoute {}/{}".format(i, len(tab_num_routes)-1), end="")

        df_observation = df_simplified[df_simplified["route_num"]==num_route]
        df_mapbox = df_mapbox_routes_test[df_mapbox_routes_test["route_num"]==num_route]

        df_observation["route_num"] = 0
        df_observation["type"] = 0    
        df_mapbox["route_num"] = 1
        df_mapbox["type"] = 2
        
        df_coeff = df_observation.append(df_mapbox)

        tab_voxels, tab_voxels_global, dict_voxels = voxel.generate_voxels(df_coeff, 0, 1)

        #\\\DEBUG///
        '''tab_voxels_min_route = voxel.get_voxels_with_min_routes(dict_voxels, 2)
        df = pd.DataFrame(tab_voxels_min_route, columns=["lat", "lon", "route_num", "type"])
        df_coeff = df_coeff.append(df)
        dp.display_mapbox(df_coeff, color="type") '''

        if(global_metric):
            coeff = metric.get_distance_voxels(0, 1, tab_voxels_global)
        else:
            coeff = metric.get_distance_voxels(0, 1, tab_voxels)
        
        tab_coeff_modified.append(min(coeff))


        '''df_observation["route_num"] = 0
        df_mapbox["route_num"] = 1
        
        df_display = df_observation.append(df_mapbox)
        dp.display_mapbox(df_display, color="route_num")'''
        
            
    print()
    print("Mean modified path distance:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("===============================")
        
    return tab_coeff_modified






if __name__ == "__main__": 

    parse = argparse.ArgumentParser()
    parse.add_argument('--project-folder', type=str, default="monresovelo", help='folder of the project')
    parse.add_argument('--set', type=str, default="test", help='whether to use the testing or the training set')
    
    args = parse.parse_args()


    project_folder = args.project_folder
    dataset = args.set

    global_metric = True


    pd.options.mode.chained_assignment = None

    with open("files/"+project_folder+"/data_processed/mapbox_pathfinding_simplified.df",'rb') as infile:
        df_mapbox_routes_test = pickle.load(infile)

    with open("files/"+project_folder+"/data_processed/observations_matched_simplified.df",'rb') as infile:
        df_simplified = pickle.load(infile)


    if not data.check_file("files/"+project_folder+"/city_graphs/city.ox"):
        exec(open("data/"+project_folder+"/load_city.py").read())
        
        
    print("Loading the city graph, this might take several minutes depending on its size...")
    with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
        G = pickle.load(infile)
    with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
        G_base = pickle.load(infile)
    nodes, _ = ox.graph_to_gdfs(G)
    tree = KDTree(nodes[['y', 'x']], metric='euclidean')


    with open("./files/"+project_folder+"/clustering/voxels_clustered_osmnx.dict",'rb') as infile:
        dict_voxels_clustered = pickle.load(infile)
    with open("files/"+project_folder+"/clustering/kmeans_voxels_osmnx.sk",'rb') as infile:
        kmeans = pickle.load(infile)
    with open("./files/"+project_folder+"/neural_networks/num_test.tab",'rb') as infile:
        tab_num_test = pickle.load(infile)
    with open("./files/"+project_folder+"/clustering/dbscan_observations.tab",'rb') as infile:
        tab_clusters = pickle.load(infile)
    dict_cluster = cl.tab_clusters_to_dict(tab_clusters)
        
        

    if(not(os.path.isfile("files/"+project_folder+"/city_graphs/graph_modifications.dict"))
    or not(os.path.isfile("files/"+project_folder+"/city_graphs/graph_modifications_global.dict"))):

        dict_modif, dict_modif_global = create_dict_modif(G, dict_cluster, df_simplified)
                
            
        with open("files/"+project_folder+"/city_graphs/graph_modifications.dict",'wb') as outfile:
            pickle.dump(dict_modif, outfile)
            
        with open("files/"+project_folder+"/city_graphs/graph_modifications_global.dict",'wb') as outfile:
            pickle.dump(dict_modif_global, outfile)
        
                
    with open("files/"+project_folder+"/city_graphs/graph_modifications.dict",'rb') as infile:
        dict_modif = pickle.load(infile)

    with open("files/"+project_folder+"/city_graphs/graph_modifications_global.dict",'rb') as infile:
        dict_modif_global = pickle.load(infile)
            
                
    with open("files/"+project_folder+"/city_graphs/graph_modifications.dict",'rb') as infile:
        dict_modif = pickle.load(infile)

    with open("files/"+project_folder+"/city_graphs/graph_modifications_global.dict",'rb') as infile:
        dict_modif_global = pickle.load(infile)
        
        
    if(dataset == "train"):
        tab_num_routes = np.arange(df_simplified.iloc[-1]["route_num"]+1).tolist()
        for i in dict_cluster[-1]:
            tab_num_routes.remove(i)
        for i in tab_num_test:  
            tab_num_routes.remove(i)
            
        print("Using the training dataset of",len(tab_num_routes),"tracks... Global metric :", global_metric)
        print("===============================")
    else:
        tab_num_routes = tab_num_test
        for i in tab_num_test:
            if i not in dict_cluster[tab_clusters[i]]:
                print("Error removing route from cluster")
            else:
                dict_cluster[tab_clusters[i]].remove(i)  
                
        print("Using the testing dataset of",len(tab_num_routes),"tracks... Global metric :", global_metric)
        print("===============================")

            
    for i in tab_num_routes:
        if(tab_clusters[i] == -1):
            print("Error: noise in dataset")





    tab_mean_results_base = []
    tab_boxplot = []
    tab_boxplot_NN = []

    tab_coeff_simplified = []

    dict_tab_route_voxels = {}

    deviation = 0


                                                        
    df_computed = pd.DataFrame(columns=["lat", "lon", "route_num"])

    print("SHORTEST PATHS :")

    for i, num_route in enumerate(tab_num_routes):
        print("\rRoute {}/{}".format(i, len(tab_num_routes)-1), end="")

        df_route_tested = df_simplified[df_simplified["route_num"]==num_route]
        d_point = df_route_tested.iloc[0].tolist()[:2]
        f_point = df_route_tested.iloc[-1].tolist()[:2]
        df_route, tab_route_voxels, coeff_simplified = create_path_compute_distance(d_point, f_point, df_route_tested, tree, G_base, nodes, global_metric)
        tab_coeff_simplified.append(min(coeff_simplified))
        dict_tab_route_voxels[num_route] = tab_route_voxels
        df_route["route_num"]= num_route
        df_computed = df_computed.append(df_route)
    print()     
    print("Mean shortest path distance:", sum(tab_coeff_simplified)/len(tab_coeff_simplified)*100, "%")
    print("===============================")
        
        
    tab_mean_results_base.append(sum(tab_coeff_simplified)/len(tab_coeff_simplified))
    tab_boxplot.append(tab_coeff_simplified)

    tab_coeff_modified_mapbox = main_mapbox(global_metric)

    tab_mean_results_base.append(sum(tab_coeff_modified_mapbox)/len(tab_coeff_modified_mapbox))
    tab_boxplot.append(tab_coeff_modified_mapbox)

    tab_coeff_modified_global = main_global(global_metric, deviation)

    tab_mean_results_base.append(sum(tab_coeff_modified_global)/len(tab_coeff_modified_global))
    tab_boxplot.append(tab_coeff_modified_global)
    tab_boxplot_NN.append(tab_coeff_modified_global)

    G = deepcopy(G_base)


    tab_coeff_modified_oracle = main_clusters_full_predict(global_metric, deviation)

    tab_mean_results_base.append(sum(tab_coeff_modified_oracle)/len(tab_coeff_modified_oracle))
    tab_boxplot.append(tab_coeff_modified_oracle)


    tab_coeff_modified_NN = main_clusters_NN(dict_tab_route_voxels, df_computed, global_metric, deviation)
    tab_boxplot_NN.append(tab_coeff_modified_NN)
    tab_boxplot_NN.append(tab_coeff_modified_oracle)

    print("===============================")
    

    graphs.distance_boxplot(tab_boxplot, project_folder, dataset)
    #graphs.mean_distance_barplot(tab_mean_results_base, [], project_folder, dataset)

    graphs.distance_boxplot_NN(tab_boxplot_NN, project_folder, dataset)
