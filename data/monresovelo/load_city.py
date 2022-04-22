import osmnx as ox
import pickle
import pandas as pd
import sys

with open("files/monresovelo/data_processed/observations_matched_simplified.df", "rb") as infile:
    df = pickle.load(infile)



df = df[["lat", "lon"]]
df_values = df.values

nort_lat = -sys.maxsize
sout_lat = sys.maxsize
east_lon = -sys.maxsize
west_lon = sys.maxsize

for coord in df_values:

    if(coord[0]>nort_lat):
        nort_lat = coord[0]
    if(coord[0]<sout_lat):
        sout_lat = coord[0]

    if(coord[1]>east_lon):
        east_lon = coord[1]
    if(coord[1]<west_lon):
        west_lon = coord[1]

df_test = pd.DataFrame([[nort_lat, west_lon, 0],
[nort_lat, east_lon, 0],
[sout_lat, east_lon, 0],
[sout_lat, west_lon, 0]] ,columns=["lat", "lon", "route_num"])

#dp.display_mapbox(df_test)

print("Creating city.ox, this might take several minutes...")    


G = ox.graph_from_bbox(nort_lat, sout_lat, east_lon, west_lon)

with open("files/monresovelo/city_graphs/city.ox", "wb") as outfile:
    pickle.dump(G, outfile)
