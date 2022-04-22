import json 
import pandas as pd
import pickle
import sys
from rdp import *
import requests
import numpy as np
import xml.etree.ElementTree as ET
import os
from math import sin, cos, sqrt, atan2, radians
from geopy.distance import geodesic
import networkx as nx
import osmnx as ox
from sklearn.neighbors import KDTree
from datetime import datetime
#import python.voxels as voxel

def check_file(file, content=None):
    """
    Check if a file exist at a given path. Creates all the directories in the path that do not exist.
    Creates the file with a given content if asked.

    Parameters
    ----------
    file : string
        The path to the file.     
    content : 
        The content to put in the file if it does not exist. If the content is None, the file is not created.

    Returns
    -------
    True
        If the file existed before the execution of the function.
    False
        If the file did not existed before the execution, wether the function created it since or not.
    """
    if(not(os.path.isfile(file))):
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        if(not isinstance(content, pd.DataFrame)):
            if(content == None):
                return False
        print("Warning: creating", file)
        open(file, "x")
        with open(file,'wb') as infile:
            pickle.dump(content, infile)
        return False
    return True
        

def request_map_matching(df_route, token):
    route = df_route.to_numpy()
    coord=""
    tab_requests = []
    i=0
    for i in range(len(route)):
        coord += str(route[i][1])+","+str(route[i][0])+";"
        if(i!=0 and i%99 == 0):
            coord = coord[:-1]
            tab_requests.append(requests.get("https://api.mapbox.com/matching/v5/mapbox/cycling/"+coord+"?access_token="+token))
            coord = ""
    if(i!=0 and i%99 != 0):
        coord = coord[:-1]
        tab_requests.append(requests.get("https://api.mapbox.com/matching/v5/mapbox/cycling/"+coord+"?access_token="+token))
    return tab_requests


def clean_dataframe(df, tab_unreachable_routes=None):
    """
    Clean a dataframe, i.e. change the route numbers if some routes are missing (if the route numbers in the dataframe
    are 0, 1, 3, 5 change it to 0, 1, 2, 3).

    Parameters
    ----------
    df : pandas' DataFrame with columns=['lat', 'lon', 'route_num']
        Dataframe to clean    
    tab_unreachable_routes : list of list
        Useful for bad gps points that cannot be accessed by osmnx, not relevant for monresovelo.
    Returns
    -------
    df_final : pandas' DataFrame with columns=['lat', 'lon', 'route_num']
        The clean dataframe.
    """
    print("Cleaning dataframe...")
    nb_empty = 0
    df_final = pd.DataFrame(columns=df.columns)
    for i in range(df.iloc[0]["route_num"], df.iloc[-1]["route_num"]+1):
        df_temp = df[df["route_num"]==i]
        if(len(df_temp)<=1):
            nb_empty += 1
        else:
            df_temp["route_num"] = i-nb_empty
            df_final = df_final.append(df_temp)
            if(tab_unreachable_routes != None):
                if(i in tab_unreachable_routes[0]):
                    tab_unreachable_routes[0].remove(i)
                    tab_unreachable_routes[0].append(i-nb_empty)
                if(i in tab_unreachable_routes[1]):
                    tab_unreachable_routes[1].remove(i)
                    tab_unreachable_routes[1].append(i-nb_empty)

    return df_final



def simplify_gps(infile, outfile, nb_routes=sys.maxsize, dim=2):
    if(nb_routes > 0):
        with open(infile,'rb') as infile:
            df = pickle.load(infile)
        check_file(outfile, pd.DataFrame(columns=df.columns))
        with open(outfile,'rb') as infile:
            df_simplified = pickle.load(infile)
        if(len(df_simplified) == 0):
            last_route_simplified = 0
        else:
            last_route_simplified = df_simplified.iloc[-1]["route_num"]+1
        nb_routes = min(df.iloc[-1]["route_num"] - last_route_simplified, nb_routes)
        df_simplified = df_simplified.append(rd_compression(df, last_route_simplified, last_route_simplified+nb_routes+1, dim))
        with open(outfile, 'wb') as outfile:
            pickle.dump(df_simplified, outfile)



def rd_compression(df, start, end, verbose=True, dim=2, eps=1e-4):
    """
    Compress a dataframe with douglas-peucker's algorithm.
    Parameters
    ----------
    df : pandas' DataFrame with columns=['lat', 'lon', 'route_num']
        Dataframe to compress
    eps : int in [0, 1[ , optional
        Precision of the compression (high value = few points)
    nb_routes : int
        Number of routes to compress
    Returns
    -------
    pandas' DataFrame with columns=['lat', 'lon', 'route_num']
        the compressed DataFrame
    """
    df_simplified = pd.DataFrame()
    for i in range(start, end):
        if(verbose):
            print("\rSimplifying route {}/{} ".format(i, end-1), end="")
        route = df[df['route_num']==i].values
        if(len(route)>0):
            simplified = np.delete(route, range(dim, route.shape[1]), 1)
            simplified = rdp(simplified.tolist(), epsilon=eps)
            if(dim == 2):
                df_temp = pd.DataFrame(simplified, columns=['lat', 'lon'])
            else:
                df_temp = pd.DataFrame(simplified, columns=['lat', 'lon', 'time_elapsed'])
            df_temp["route_num"]=route[0][-1]
            df_simplified = df_simplified.append(df_temp)
    return df_simplified



def request_route(lat1, long1, lat2, long2, token, mode="cycling"):
    coord = str(long1)+","+str(lat1)+";"+str(long2)+","+str(lat2)
    return requests.get("https://api.mapbox.com/directions/v5/mapbox/"+mode+"/"+coord, 
                            params={"alternatives": "true", "geometries": "geojson", "steps": "true", "access_token": token}) 


def pathfinding_mapbox(infilestr, outfilestr, token, nb_routes=sys.maxsize):
    if(nb_routes > 0):
        with open(infilestr,'rb') as infile:
            df_map_matched_simplified = pickle.load(infile)
        check_file(outfilestr, pd.DataFrame(columns=['lat', 'lon', 'route_num']))
        with open(outfilestr,'rb') as infile:
            df_pathfinding = pickle.load(infile)
        tab_num_test = None

        if(len(df_pathfinding) == 0):
            last_route_pathfound = 0
        else:
            last_route_pathfound = df_pathfinding.iloc[-1]["route_num"]+1

        nb_routes = min(df_map_matched_simplified.iloc[-1]["route_num"] - last_route_pathfound, nb_routes)
        for i in range(last_route_pathfound, last_route_pathfound+nb_routes+1):
            if(tab_num_test == None or i in tab_num_test):
                print("\rFinding the shortest path for route {}/{} using Mapbox.".format(i, last_route_pathfound+nb_routes), end="")
                df_temp = df_map_matched_simplified[df_map_matched_simplified["route_num"]==i]
                d_point = [df_temp.iloc[0]["lat"], df_temp.iloc[0]["lon"]]
                f_point = [df_temp.iloc[-1]["lat"], df_temp.iloc[-1]["lon"]]
                df_temp = pathfind_route_mapbox(d_point, f_point, token, df_pathfinding, i)
                df_pathfinding = df_pathfinding.append(df_temp)
                with open(outfilestr, 'wb') as outfile:
                    pickle.dump(df_pathfinding, outfile)


def pathfind_route_mapbox(d_point, f_point, token, df_pathfinding=pd.DataFrame(), num_route=1):
    save_route = True
    req = request_route(d_point[0], d_point[1], f_point[0], f_point[1]) #mapbox request to find a route between the stations
    response = req.json()
    if(response['code']=='Ok'): #if a route have been found
        steps = response['routes'][0]['legs'][0]['steps'] #we browse all the steps of the route
        for step in steps:
            if(step['maneuver']['instruction'].find("Wharf") != -1):
                save_route = False #if the route is not good (using a boat) we don't save it
                break
        if(save_route): #if we save the route
            df_temp = pd.DataFrame.from_records(response['routes'][0]['geometry']['coordinates'], 
                                    columns=['lon', 'lat']) #create a DF from the route (nparray)
            df_temp["route_num"] = num_route
            return df_temp[["lat", "lon", "route_num"]]
    return None




def pathfinding_osmnx(infile_str, outfile_str, graphfile_str, unreachableroutesfile_str, nb_routes=sys.maxsize):
    if(nb_routes > 0):
        with open(infile_str,'rb') as infile:
            df_simplified = pickle.load(infile)

        with open(graphfile_str,'rb') as infile:
            G = pickle.load(infile)
            nodes, _ = ox.graph_to_gdfs(G)
            tree = KDTree(nodes[['y', 'x']], metric='euclidean')

        check_file(outfile_str, pd.DataFrame(columns=['lat', 'lon', 'route_num']))
        with open(outfile_str,'rb') as infile:
            df_pathfinding = pickle.load(infile)
            
        check_file(unreachableroutesfile_str, [[],[]])
        with open(unreachableroutesfile_str,'rb') as infile:
            tab_unreachable_routes = pickle.load(infile)

        if(len(df_pathfinding) == 0):
            last_route_pathfound = 0
        else:
            last_route_pathfound = df_pathfinding.iloc[-1]["route_num"]+1

        nb_routes = min(df_simplified.iloc[-1]["route_num"] - last_route_pathfound, nb_routes)
        for i in range(last_route_pathfound, last_route_pathfound+nb_routes+1):
            print("\rFinding the shortest path for route {}/{} with OSMNX.".format(i, last_route_pathfound+nb_routes), end="")
            df_temp = df_simplified[df_simplified["route_num"]==i]
            d_point = [df_temp.iloc[0]["lat"], df_temp.iloc[0]["lon"]]
            if(i in tab_unreachable_routes[0]):
                d_point = [df_temp.iloc[1]["lat"], df_temp.iloc[1]["lon"]]
            f_point = [df_temp.iloc[-1]["lat"], df_temp.iloc[-1]["lon"]]
            if(i in tab_unreachable_routes[1]):
                f_point = [df_temp.iloc[-2]["lat"], df_temp.iloc[-2]["lon"]]
            route = pathfind_route_osmnx(d_point, f_point, tree, G, nodes)
            route_coord = [[G.nodes[x]["y"], G.nodes[x]["x"]] for x in route]
            route_coord = [x + [i] for x in route_coord]
            df_pathfinding = df_pathfinding.append(pd.DataFrame(route_coord, columns=["lat", "lon", "route_num"]))
            with open(outfile_str, 'wb') as outfile:
                pickle.dump(df_pathfinding, outfile)
                
                



def pathfind_route_osmnx(d_point, f_point, tree, G, nodes):
    d_idx = tree.query([d_point], k=1, return_distance=False)[0]
    f_idx = tree.query([f_point], k=1, return_distance=False)[0]
    closest_node_to_d = nodes.iloc[d_idx].index.values[0]
    closest_node_to_f = nodes.iloc[f_idx].index.values[0]
    route = nx.shortest_path(G, 
                             closest_node_to_d,
                             closest_node_to_f,
                             weight='length')
    return route





def distance_between_points(p1, p2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(p1[0])
    lon1 = radians(p1[1])
    lat2 = radians(p2[0])
    lon2 = radians(p2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


def compute_distance(infile, outfile):
    with open(infile,'rb') as infile:
        df = pickle.load(infile)
    tab_distances = []
    for i in range(df.iloc[-1]["route_num"]+1):
        print("\rCalculating the distance of route {}/{}.".format(i, df.iloc[-1]["route_num"]), end="")
        df_temp = df[df["route_num"]==i]
        dist = 0
        if(df_temp.shape[0] >= 2):
            for j in range(df_temp.shape[0]-1):
                dist += distance_between_points(df_temp.iloc[j], df_temp.iloc[j+1])
                #dist += geodesic((df_temp.iloc[j][0],df_temp.iloc[j][1]), (df_temp.iloc[j+1][0], df_temp.iloc[j+1][1])).kilometers
            tab_distances.append(dist)
        else:
            tab_distances.append(0)
    print()
    check_file(outfile, [])
    with open(outfile, 'wb') as outfile:
        pickle.dump(tab_distances, outfile)



def dataframe_to_array(df, n=2):
    tab = df.to_numpy().tolist()
    for i in range(len(tab)):
        tab[i] = tab[i][:n]
    return tab
