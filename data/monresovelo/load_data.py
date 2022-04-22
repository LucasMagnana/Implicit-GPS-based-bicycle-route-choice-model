import json
import pandas as pd
import pickle
import os

tab_routes = []

with open('data/monresovelo/trip5000.json') as f:
  d = json.load(f)

print("Loading data...")

for i in range(len(d["features"])):
  for coord in d["features"][i]["geometry"]["coordinates"]:
    tab_routes.append([coord[1], coord[0], i])

  print("\r{}th route ".format(i), end="")
print()

df = pd.DataFrame(tab_routes, columns=["lat", "lon", "route_num"], dtype=object)

print("Warning: creating files/monresovelo/data_processed/observations_matched.df")

with open("files/monresovelo/data_processed/observations_matched.df", "wb") as infile:
  pickle.dump(df, infile)
