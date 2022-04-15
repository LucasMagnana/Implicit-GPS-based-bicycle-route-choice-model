from OSMPythonTools.overpass import overpassQueryBuilder
from OSMPythonTools.overpass import Overpass
from OSMPythonTools.data import Data, dictRangeYears, ALL
import pandas as pd
import pickle

from OSMPythonTools.nominatim import Nominatim

from collections import OrderedDict

project_folder = "veleval"

overpass = Overpass()
nominatim = Nominatim()

dimensions = OrderedDict([
  ('typeOfRoad', OrderedDict({
    0: '"highway"="cycleway"',
    1: '"cycleway"',
    2: '"cycleway:left"',
    3: '"cycleway:right"',
    4: '"cycleway:lane"'
  }))
])

def fetch(typeOfRoad):
    areaId = nominatim.query("Lyon").areaId()
    query = overpassQueryBuilder(area=areaId, elementType='way', selector=typeOfRoad, out='body', includeGeometry=True)
    return overpass.query(query)
data = Data(fetch, dimensions)

route_coord = []
route_num = 0

for i in range(len(dimensions['typeOfRoad'])):
    print(data.select(typeOfRoad=i).countElements())
    json = data.select(typeOfRoad=i).toJSON()
    for el in json["elements"] :
      for pos in el['geometry']:
        route_coord.append([pos["lon"], pos["lat"], route_num])
      route_num += 1

df = pd.DataFrame(route_coord, columns=["lon", "lat", "route_num"])

with open("files/"+project_folder+"/data_processed/osm_bikepath.df", "wb") as infile:
    pickle.dump(df, infile)

