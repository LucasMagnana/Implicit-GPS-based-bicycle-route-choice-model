
import plotly.express as px
import pandas as pd
import folium
from folium.plugins import HeatMap

import python.voxels as voxel


def display_mapbox(dfdisplay, token, n=75, line_group="route_num", color=None, filename=None):
    """
    Display a dataframe of gps points on a mapbox map.
    Parameters
    ----------
    df or str : pandas' DataFrame with columns=['lat', 'lon', 'route_num'] or the name of a file containing one
        Dataframe to display or the file where it is located
    n : int, optional
        Number of routes to display
    line_group : str, optional
        Dataframe's attribute used to differenciate routes
    color : str, optional
        Dataframe's attribute used to color routes
    """
    if(type(dfdisplay) == str): #if df is a file location
        with open(dfdisplay,'rb') as infile:
            n+=1
            dfdisplay = pickle.load(infile) #open the file to load the dataframe
            dfdisplay = dfdisplay[dfdisplay[line_group]<n]
    fig = px.line_mapbox(dfdisplay, lat="lat", lon="lon", line_group=line_group, color=color, zoom=11)
    fig.show()
    if(filename != None):
        fig.write_image(filename)



def display(df_display, zoom=11, line_group="route_num", color=None):
    """
    Display a dataframe of gps points on a mapbox map.
    Parameters
    ----------
    df or str : pandas' DataFrame with columns=['lat', 'lon', 'route_num'] or the name of a file containing one
        Dataframe to display or the file where it is located
    n : int, optional
        Number of routes to display
    line_group : str, optional
        Dataframe's attribute used to differenciate routes
    color : str, optional
        Dataframe's attribute used to color routes
    """
    base_map = folium.Map(location=[df_display.iloc[0]["lat"],df_display.iloc[0]["lon"]], control_scale=True, zoom_start=zoom, tiles = 'Stamen Toner')
    tab_colors = ["orange", "blue", "red", "green", "yellow", "black"]
    i = 0
    cont = True
    while(cont):
        if(color != None):
            df_temp = df_display[df_display[color]==i]
        else:
            df_temp = df_display
            cont = False
        if(not df_temp.empty):
            if(df_temp.iloc[-1][line_group]>0):
                r = range(int(df_temp.iloc[0][line_group]), int(df_temp.iloc[-1][line_group])+1)
            else :
                r = range(int(df_temp.iloc[0][line_group]), int(df_temp.iloc[-1][line_group])-1, -1)
            for j in r:
                points = df_temp[df_temp[line_group]==j][["lat","lon"]].values.tolist()
                folium.PolyLine(points, color=tab_colors[i], weight=6).add_to(base_map)
        else:
            cont = False
        i+=1
    return base_map



def display_routes(df, tab_routes, tab_voxels=[], line_group="route_num", color=None):
    dfdisplay = pd.DataFrame(columns=["lat", "lon", "route_num"])
    for i in range(len(tab_routes)):
        dfdisplay = dfdisplay.append(df[df["route_num"]==tab_routes[i]])
    display(dfdisplay, len(tab_routes), line_group, color)




def create_df_heatmap(df, tab_routes, tab_voxels=[], line_group="route_num", color=None):
    dfdisplay = pd.DataFrame(columns=["lat", "lon", "route_num"])
    for i in range(len(tab_routes)):
        df_temp = df[df["route_num"]==tab_routes[i]]
        df_temp["num_route"] = i
        dfdisplay = dfdisplay.append(df_temp)
    _, _, dict_voxels = voxel.generate_voxels(dfdisplay, 0, dfdisplay.iloc[-1]["route_num"])
    tab = []
    for key in dict_voxels:
        tab_routes = dict_voxels[key]["tab_routes_real"]+dict_voxels[key]["tab_routes_extended"]
        vox_str = key.split(";")
        vox_int = [int(vox_str[0]), int(vox_str[1])]
        vox_pos = voxel.get_voxel_points(vox_int, 0)
        if(dict_voxels[key]["cyclability_coeff"]):
            tab.append([vox_pos[0][0], vox_pos[0][1], dict_voxels[key]["cyclability_coeff"]])

    return pd.DataFrame(tab, columns=["lat", "lon", "value"])



def display_cluster_heatmap(df, tab_routes, tab_voxels=[], line_group="route_num", color=None):
    dfdisplay = create_df_heatmap(df, tab_routes, tab_voxels=[], line_group="route_num", color=None)

    map = folium.Map(location=[dfdisplay.iloc[0]["lat"],dfdisplay.iloc[0]["lon"]], control_scale=True, zoom_start=11, tiles = 'Stamen Toner')
    HeatMap(data=dfdisplay.values.tolist(), max_zoom=13, radius=9, blur = 1, min_opacity = 0, max_val = 1).add_to(map)
    return map


def display_cluster_heatmap_mapbox(df, tab_routes, tab_voxels=[], line_group="route_num", color=None):
    dfdisplay = create_df_heatmap(df, tab_routes, tab_voxels=[], line_group="route_num", color=None)

    fig = px.scatter_mapbox(dfdisplay, lat="lat", lon="lon",  color="value", size="value", zoom=10)
    fig.show()
