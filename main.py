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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.neighbors import KDTree
import os
import argparse
import datetime


import python.data as data
import python.display as dp
import python.voxels as voxel
import python.metric as metric
import python.clustering as cl
import python.RNN as RNN
import python.validation as validation
import python.graphs as graphs
import python.MultiPass as MP


if __name__ == "__main__": 
    parse = argparse.ArgumentParser()
    parse.add_argument('--project-folder', type=str, default="veleval_walk", help='folder of the project')
    parse.add_argument('--set', type=str, default="test", help='whether to use the testing or the training set')
    
args = parse.parse_args()


project_folder = args.project_folder
dataset = args.set

global_metric = True



def create_dict_modif(G, dict_cluster, df_simplified):
    a = 0
    for i in dict_cluster:
        a+=len(dict_cluster[i])
    print(a)
    dict_modif = {}
    print("start:", datetime.datetime.now().time())
    dict_dict_voxels_cluster = {}
    for cl in dict_cluster:
        if(cl > -1):
            df_cluster = pd.DataFrame(columns=["lat", "lon", "route_num"])
            for num_route in range(len(dict_cluster[cl])):
                df_temp = df_simplified[df_simplified["route_num"]==dict_cluster[cl][num_route]]
                df_temp["route_num"] = num_route
                df_cluster = df_cluster.append(df_temp)
            _, _, dict_voxels_cluster = voxel.generate_voxels(df_cluster, df_cluster.iloc[0]["route_num"], df_cluster.iloc[-1]["route_num"])
            dict_dict_voxels_cluster[cl] = dict_voxels_cluster
    for v in G:
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
    print("end:", datetime.datetime.now().time())
    return dict_modif


pd.options.mode.chained_assignment = None

with open("files/"+project_folder+"/data_processed/mapbox_pathfinding.df",'rb') as infile:
    df_mapbox_routes_test = pickle.load(infile)

with open("files/"+project_folder+"/data_processed/observations_matched_simplified.df",'rb') as infile:
    df_simplified = pickle.load(infile)


if not os.path.exists(os.path.dirname("files/"+project_folder+"/city_graphs/city.ox")):
    print("Creating city.ox")
    exec(open("files/"+project_folder+"/load_city.py").read())
    
    
if("veleval" in project_folder):

    with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
        G_1 = pickle.load(infile)
    with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
        G_base_1 = pickle.load(infile)
    nodes_1, _ = ox.graph_to_gdfs(G_1)
    tree_1 = KDTree(nodes_1[['y', 'x']], metric='euclidean')

    with open("files/"+project_folder+"/city_graphs/city_2.ox", "rb") as infile:
        G_2 = pickle.load(infile)
    with open("files/"+project_folder+"/city_graphs/city_2.ox", "rb") as infile:
        G_base_2 = pickle.load(infile)
    nodes_2, _ = ox.graph_to_gdfs(G_2)
    tree_2 = KDTree(nodes_2[['y', 'x']], metric='euclidean')

else:
    with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
        G = pickle.load(infile)
    with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
        G_base = pickle.load(infile)
    nodes, _ = ox.graph_to_gdfs(G)
    tree = KDTree(nodes[['y', 'x']], metric='euclidean')
    
    
print(len(G_1.edges), len(G_1.nodes), len(G_2.edges), len(G_2.nodes))


with open("./files/"+project_folder+"/clustering/voxels_clustered_osmnx.dict",'rb') as infile:
    dict_voxels_clustered = pickle.load(infile)
with open("files/"+project_folder+"/clustering/kmeans_voxels_osmnx.sk",'rb') as infile:
    kmeans = pickle.load(infile)
with open("./files/"+project_folder+"/neural_networks/saved/num_test.tab",'rb') as infile:
    tab_num_test = pickle.load(infile)
with open("./files/"+project_folder+"/clustering/dbscan_observations.tab",'rb') as infile:
    tab_clusters = pickle.load(infile)
dict_cluster = cl.tab_clusters_to_dict(tab_clusters)

if(dataset == "train"):
    tab_num_routes = np.arange(df_simplified.iloc[-1]["route_num"]).tolist()
    for i in dict_cluster[-1]:
        tab_num_routes.remove(i)
    for i in tab_num_test:   
        tab_num_routes.remove(i)
        
    print("Using the training dataset of",len(tab_num_routes),"tracks... Global metric :", global_metric)
    print("===============================")
else:
    dataset == "test"
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

data.check_file("files/"+project_folder+"/data_processed/unreachable_routes.tab", [[],[]])
with open("files/"+project_folder+"/data_processed/unreachable_routes.tab",'rb') as infile:
    tab_unreachable_routes = pickle.load(infile) 
    
    

if(not(os.path.isfile("files/"+project_folder+"/city_graphs/graph_modifications.dict"))):
    if("veleval" in project_folder):
        dict_modif = create_dict_modif(G_1, dict_cluster, df_simplified)
        dict_modif_se = create_dict_modif(G_2, dict_cluster, df_simplified)
        for cl in dict_modif_se:
            if(cl not in dict_modif):
                dict_modif[cl] = dict_modif_se[cl]
    else:
        dict_modif = create_dict_modif(G, dict_cluster, df_simplified)
            
        
    with open("files/"+project_folder+"/city_graphs/graph_modifications.dict",'wb') as outfile:
        pickle.dump(dict_modif, outfile)
   

            
if(not(os.path.isfile("files/"+project_folder+"/city_graphs/graph_modifications_global.dict"))):    
    dict_modif_global = {}
    dict_cluster_global = {0: range(len(tab_clusters)+1)}
    if("veleval" in project_folder):
        dict_modif_global[0] = create_dict_modif(G_1, dict_cluster_global, df_simplified)[0]
        dict_modif_global[1] = create_dict_modif(G_2, dict_cluster_global, df_simplified)[0]
    else:
        dict_modif_global[0] = create_dict_modif(G, dict_cluster_global, df_simplified)[0]
        
    with open("files/"+project_folder+"/city_graphs/graph_modifications_global.dict",'wb') as outfile:
        pickle.dump(dict_modif_global, outfile)
    
            
with open("files/"+project_folder+"/city_graphs/graph_modifications.dict",'rb') as infile:
    dict_modif = pickle.load(infile)

with open("files/"+project_folder+"/city_graphs/graph_modifications_global.dict",'rb') as infile:
    dict_modif_global = pickle.load(infile)
    

def create_path_compute_distance(d_point, f_point, df, tree, G, nodes, global_metric):

    route = data.pathfind_route_osmnx(d_point, f_point, tree, G, nodes)
    route_coord = [[G.nodes[x]["y"], G.nodes[x]["x"]] for x in route]
    route_coord = [x + [0] for x in route_coord]

    df_route = pd.DataFrame(route_coord, columns=["lat", "lon", "route_num"])
    df_route = data.rd_compression(df_route, 0, 1)
   

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
        
def choose_route_endpoints(df_route, num_route, deviation):
        global tab_unreachable_routes
        d_point = [df_route.iloc[0]["lat"], df_route.iloc[0]["lon"]]
        if(num_route in tab_unreachable_routes[0]):
            d_point = [df_route.iloc[1]["lat"], df_route.iloc[1]["lon"]]
        f_point = [df_route.iloc[-1]["lat"], df_route.iloc[-1]["lon"]]
        if(num_route in tab_unreachable_routes[1]):
            f_point = [df_route.iloc[-2]["lat"], df_route.iloc[-2]["lon"]]
        rand = random.uniform(-deviation, deviation)
        d_point[0] += rand
        rand = random.uniform(-deviation, deviation)
        d_point[1] += rand
        rand = random.uniform(-deviation, deviation)
        f_point[0] += rand
        rand = random.uniform(-deviation, deviation)
        f_point[1] += rand
        
        return d_point, f_point
    
    
def choose_network_graph(df_route, project_folder):
    if("veleval" in project_folder):
        if(df_route.iloc[0]["lat"] <= 45.5):
            return G_2, nodes_2, tree_2, G_base_2
        else:
            return G_1, nodes_1, tree_1, G_base_1
    return G, nodes, tree, G_base


def compute_kspwlo(tab_k_sp):              
    for i in range(len(tab_k_sp), df_simplified.iloc[-1]["route_num"]+1): #len(tab_clusters)):
        print("Computing kspwlo for", i)
        df_route_tested = df_simplified[df_simplified["route_num"]==i]       
        d_point, f_point = choose_route_endpoints(df_route_tested, i, deviation)

        G, nodes, tree, G_base = choose_network_graph(df_route_tested, project_folder)

        tab_k_sp.append(MP.ESX(G, tree, nodes, d_point, f_point))
                       
        '''with open("files/"+project_folder+"/data_processed/kspwlo.tab",'wb') as outfile:
            pickle.dump(tab_k_sp, outfile)'''
            
    return tab_k_sp


def create_logit_data(df_k_sp):
    
    print("Creating logit data...")

    X_train=[]
    y_train=[]
    y_train_global=[]

    X_test=[]
    y_test=[]
    y_test_global=[]

    for i in range(df_k_sp.iloc[-1]["route_num"]+1):

        tab_features = []
        tab_distances = []
        tab_distances_global = []

        df_spe_k_sp = df_k_sp[df_k_sp["route_num"]==i]
        
        if(not df_spe_k_sp.empty):
            df_obs = df_simplified[df_simplified["route_num"]==i]

            for k in range(df_spe_k_sp.iloc[-1]["k_num"]+1):
                df_sp = df_spe_k_sp[df_spe_k_sp["k_num"]==k]

                distance = 0

                for p in range(len(df_sp.values)-1):
                    point = df_sp.values[p]
                    next_point = df_sp.values[p+1]
                    distance+=data.distance_between_points([point[0], point[1]], [next_point[0], next_point[1]])

                tab_features.append([len(df_sp.values)-1, distance])

                df_sp=df_sp[["lat", "lon", "route_num"]]
                df_sp["route_num"]=0
                df_obs["route_num"]=1

                df_sp = df_sp.append(df_obs)

                tab_voxels, tab_voxels_global, dict_voxels = voxel.generate_voxels(df_sp, 0, 1)

                tab_distances.append(min(metric.get_distance_voxels(0, 1, tab_voxels)))
                tab_distances_global.append(min(metric.get_distance_voxels(0, 1, tab_voxels_global)))

            if(i in tab_num_test):                                    
                X_test.append(tab_features)
                y_test.append(tab_distances)
                y_test_global.append(tab_distances_global)
            else:
                X_train.append(tab_features)
                y_train.append(tab_distances)
                y_train_global.append(tab_distances_global)            
            
    X = [X_train, X_test]
    y = [y_train, y_test]
    y_global = [y_train_global, y_test_global]
    
    with open("files/"+project_folder+"/data_processed/logit_X.tab",'wb') as outfile:
        pickle.dump(X, outfile)
    with open("files/"+project_folder+"/data_processed/logit_y.tab",'wb') as outfile:
        pickle.dump(y, outfile)
    with open("files/"+project_folder+"/data_processed/logit_y_global.tab",'wb') as outfile:
        pickle.dump(y_global, outfile)
            
            

def create_df_k_sp(tab_k_sp, df_k_sp):                       
                        
    for i in range(df_k_sp.iloc[-1]["route_num"], len(tab_k_sp)-1):
        
        print("Computing kspwlo dataframe for", i)
        
        if(len(tab_k_sp[i]) > 1):
            k=0
            G, nodes, tree, G_base = choose_network_graph(df_simplified[df_simplified["route_num"]==i], project_folder)
            for sp in tab_k_sp[i]:
                pos_sp = []
                for n in sp:
                    pos_sp.append([G.nodes[n]["y"], G.nodes[n]["x"], i])

                df_sp = pd.DataFrame(pos_sp, columns=["lat", "lon", "route_num"])

                df_sp["k_num"]=k
                k+=1
                df_k_sp = df_k_sp.append(df_sp)           
          
    '''with open("files/"+project_folder+"/data_processed/kspwlo.df",'wb') as outfile:
        pickle.dump(df_k_sp, outfile)'''
        
    return df_k_sp


def main_global(global_metric, deviation):

    tab_coeff_modified = []

    tab_diff_coeff = []


    if("veleval" in project_folder):
        modify_network_graph(0, dict_modif_global, G_1)
        modify_network_graph(1, dict_modif_global, G_2)
    else:
        global G
        modify_network_graph(0, dict_modif_global, G)

    for i in tab_num_routes:
        df_route_tested = df_simplified[df_simplified["route_num"]==i]

        d_point, f_point = choose_route_endpoints(df_route_tested, i, deviation)

        G, nodes, tree, G_base = choose_network_graph(df_route_tested, project_folder)

        df_route, tab_route_voxels, coeff_modified = create_path_compute_distance(d_point, f_point, df_route_tested, tree, G, nodes, global_metric)

        tab_coeff_modified.append(min(coeff_modified))
    
    
    print("GLOBAL :")
    print("Mean modified path distance:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("===============================")

    return tab_coeff_modified
        

def main_clusters(global_metric, deviation):

    tab_coeff_modified = []

    tab_diff_coeff = []

    for key in dict_cluster:
        if(key != -1):
            df_temp = df_simplified[df_simplified["route_num"]==dict_cluster[key][0]]
            
            G, nodes, tree, G_base = choose_network_graph(df_temp, project_folder)

            modify_network_graph(key, dict_modif, G)
    
    for i in tab_num_routes: #len(tab_clusters)):e
        df_route_tested = df_simplified[df_simplified["route_num"]==i]
        
        
        d_point, f_point = choose_route_endpoints(df_route_tested, i, deviation)

        G, nodes, tree, G_base = choose_network_graph(df_route_tested, project_folder)

        df_route, tab_route_voxels, coeff_modified = create_path_compute_distance(d_point, f_point, df_route_tested, tree, G, nodes, global_metric)

        tab_coeff_modified.append(min(coeff_modified))


    print("CLUSTERS ONLY:")
    print("Mean modified path distance:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("===============================")
    

    return tab_coeff_modified

    




def main_clusters_NN(dict_tab_route_voxels, df_computed, global_metric, deviation, full_print=False):
    


    with open("./files/"+project_folder+"/neural_networks/saved/network.param",'rb') as infile:
        param = pickle.load(infile)

    size_data = 1

    network = RNN.RNN_LSTM(size_data, max(tab_clusters)+1, param.hidden_size, param.num_layers, param.bidirectional, param.dropout)
    network.load_state_dict(torch.load("files/"+project_folder+"/neural_networks/saved/network_temp.pt"))
    network.eval()

    nb_good_predict = 0

    tab_coeff_modified = [[], []]

    tab_diff_coeff = [[], []]

    for i in tab_num_routes: #len(tab_clusters)):
        good_predict = False
        df_route_tested = df_simplified[df_simplified["route_num"]==i]
        
        d_point, f_point = choose_route_endpoints(df_route_tested, i, deviation)

        G, nodes, tree, G_base = choose_network_graph(df_route_tested, project_folder)


        cl, nb_new_cluster = validation.find_cluster(dict_tab_route_voxels[i], network, param.voxels_frequency, dict_voxels_clustered, 
                                    kmeans, df_route_tested)
        #print(cl, tab_clusters[i])
        if(cl == tab_clusters[i]):
            #print("good predict")
            nb_good_predict += 1
            good_predict = True
            #dp.display_cluster_heatmap_mapbox(df_simplified, dict_cluster[cl])
        #dp.display_mapbox(df_route)

        ################################################################################_

        modify_network_graph(cl, dict_modif, G)


        df_route_modified,_,coeff_modified = create_path_compute_distance(d_point, f_point, df_simplified[df_simplified["route_num"]==i], tree, G, nodes, global_metric)

        cancel_network_graph_modifications(cl, dict_modif, G, G_base)
        
        df_sp = df_computed[df_computed["route_num"]==i]
        
        df_route_tested["route_num"] = 0 #orange
        df_route_tested["type"] = 0
        
        df_sp["route_num"] = 1 #bleu
        df_sp["type"] = 1
        
        df_route_modified["route_num"]= 2 #rouge
        df_route_modified["type"] = 2
        
        
        df_route_modified = df_route_modified.append(df_sp)
        df_route_modified = df_route_modified.append(df_route_tested)
        map = dp.display(df_route_modified, color="route_num")
        map.save("files/"+project_folder+"/images/maps/"+str(i)+".html")
        

        if(good_predict):
            tab_coeff_modified[0].append(min(coeff_modified))
        else:
            tab_coeff_modified[1].append(min(coeff_modified))

        
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
        
    print("LSTM (Good predict:", len(tab_coeff_modified[0])/(len(tab_coeff_modified[0])+len(tab_coeff_modified[1]))*100, "%):")
    print("Mean modified path distance:", sum(sum(tab_coeff_modified,[]))/sum(len(row) for row in tab_coeff_modified)*100, "%")


    return tab_coeff_modified[0]+tab_coeff_modified[1]




def main_clusters_full_predict(global_metric, deviation):

    tab_coeff_modified = []

    tab_diff_coeff = []

    for i in tab_num_routes: #len(tab_clusters)):
        df_route_tested = df_simplified[df_simplified["route_num"]==i]
        df_mapbox = df_mapbox_routes_test[df_mapbox_routes_test["route_num"]==i]
        
        d_point, f_point = choose_route_endpoints(df_route_tested, i, deviation)
        

        G, nodes, tree, G_base = choose_network_graph(df_route_tested, project_folder)


        modify_network_graph(tab_clusters[i], dict_modif, G)


        df_route_modified,_,coeff_modified = create_path_compute_distance(d_point, f_point, df_simplified[df_simplified["route_num"]==i], tree, G, nodes, global_metric)

        cancel_network_graph_modifications(tab_clusters[i], dict_modif, G, G_base)
        
        tab_coeff_modified.append(min(coeff_modified))
        
        '''df_route_tested["route_num"] = 0
        df_route_modified["route_num"] = 1
        df_mapbox["route_num"] = 2
        df_route["route_num"] = 3
        
        df_display = df_route_tested.append(df_route_modified)
        df_display = df_display.append(df_route)
        df_display = df_display.append(df_mapbox)
        dp.display_mapbox(df_display, color="route_num")'''

       
    
    print("ORACLE :")
    print("Mean modified path distance:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("===============================")

    return tab_coeff_modified







def main_mapbox(global_metric):
    global df_simplified
    global df_mapbox_routes_test
        
    tab_coeff_modified = []
    
    
    for i in tab_num_routes:
        df_observation = df_simplified[df_simplified["route_num"]==i]
        df_mapbox = df_mapbox_routes_test[df_mapbox_routes_test["route_num"]==i]

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
        
            
    print("MAPBOX :")
    print("Mean modified path distance:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("===============================")
        
    return tab_coeff_modified


def main_logit(global_metric, deviation, k=3):

    tab_coeff_modified = []

    tab_diff_coeff = []
                        
    data.check_file("files/"+project_folder+"/data_processed/kspwlo.tab", [])
                        
    with open("files/"+project_folder+"/data_processed/kspwlo.tab",'rb') as infile:
        tab_k_sp = pickle.load(infile)              
                        
    tab_k_sp = compute_kspwlo(tab_k_sp)
    
    data.check_file("files/"+project_folder+"/data_processed/kspwlo.tab", [])
    
    with open("files/"+project_folder+"/data_processed/kspwlo.df",'rb') as infile:
        df_k_sp = pickle.load(infile)     
    
    df_k_sp = create_df_k_sp(tab_k_sp, df_k_sp)
    
    #create_logit_data(df_k_sp)
    
    with open("files/"+project_folder+"/data_processed/logit_X.tab",'rb') as infile:
        X = pickle.load(infile)
        
    if(global_metric):
        with open("files/"+project_folder+"/data_processed/logit_y_global.tab",'rb') as infile:
            y = pickle.load(infile)
    else:
        with open("files/"+project_folder+"/data_processed/logit_y.tab",'rb') as infile:
            y = pickle.load(infile)
        
    X_train = []
    X_test = []
    
    y_train = []
    y_test = []
    
    for i in range(len(X[0])):
        features=[]
        for j in range(len(X[0][i])):
            if(j<k):
                #features.append(X[0][i][j][0])
                features.append(X[0][i][j][1])
                
        X_train.append(features)
        
        distances = y[0][i][:k]
        y_train.append(distances.index(min(distances)))
            
            
    for i in range(len(X[1])):
        features=[]
        for j in range(len(X[1][i])):
            if(j<k):
                #features.append(X[1][i][j][0])
                features.append(X[1][i][j][1])
                
        X_test.append(features)
        distances = y[1][i][:k]        
        
        y_test.append(distances.index(min(distances)))
        
    '''for i in range(k):
        print(y_train.count(i))
    print("---------------------")
    for i in range(k):
        print(y_test.count(i))'''
    
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    #print(len(X_train_minmax[0]))
    clf = LogisticRegression(random_state=0).fit(X_train_minmax, y_train)
    
    X_test_minmax = min_max_scaler.transform(X_test)
    
    
    if(dataset == "train"):
        X_to_use = X_train_minmax
        y_to_use = y_train
        y_index=0
    else:
        X_to_use = X_test_minmax
        y_to_use = y_test
        y_index=1
    
    
    for i in range(len(X_to_use)):
        index = clf.predict([X_to_use[i]])[0]
        tab_coeff_modified.append(y[y_index][i][index])
        
        
       
    print("LOGIT (Good predict:", clf.score(X_to_use, y_to_use)*100, "%):")
    print("Mean modified path distance:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("===============================")

    return tab_coeff_modified

tab_mean_results_base = []
tab_boxplot = []
tab_boxplot_NN = []

tab_coeff_simplified = []

dict_tab_route_voxels = {}

deviation = 0

main_logit(global_metric, deviation)


                                                     
df_computed = pd.DataFrame(columns=["lat", "lon", "route_num"])

for i in tab_num_routes:
    df_route_tested = df_simplified[df_simplified["route_num"]==i]
    d_point, f_point = choose_route_endpoints(df_route_tested, i, deviation)
    _, nodes, tree, G_base = choose_network_graph(df_route_tested, project_folder)
    df_route, tab_route_voxels, coeff_simplified = create_path_compute_distance(d_point, f_point, df_route_tested, tree, G_base, nodes, global_metric)
    tab_coeff_simplified.append(min(coeff_simplified))
    dict_tab_route_voxels[i] = tab_route_voxels
    df_route["route_num"]= i
    df_computed = df_computed.append(df_route)
    
print("Mean shortest path distance:", sum(tab_coeff_simplified)/len(tab_coeff_simplified)*100, "%")
print("===============================")
    
    
tab_mean_results_base.append(sum(tab_coeff_simplified)/len(tab_coeff_simplified))
tab_boxplot.append(tab_coeff_simplified)

tab_coeff_modified_mapbox = main_mapbox(global_metric)

tab_mean_results_base.append(sum(tab_coeff_modified_mapbox)/len(tab_coeff_modified_mapbox))
tab_boxplot.append(tab_coeff_modified_mapbox)

tab_coeff_modified_logit = main_logit(global_metric, deviation)

tab_mean_results_base.append(sum(tab_coeff_modified_logit)/len(tab_coeff_modified_logit))
tab_boxplot.append(tab_coeff_modified_logit)

tab_coeff_modified_global = main_global(global_metric, deviation)

tab_mean_results_base.append(sum(tab_coeff_modified_global)/len(tab_coeff_modified_global))
tab_boxplot.append(tab_coeff_modified_global)
tab_boxplot_NN.append(tab_coeff_modified_global)


if("veleval" in project_folder):
    G_1 = deepcopy(G_base_1)
    G_2 = deepcopy(G_base_2)
else:
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
