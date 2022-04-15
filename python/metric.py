from dipy.segment.metric import Metric, ResampleFeature
from dipy.segment.clustering import QuickBundles
import python.voxels as voxel
import pickle
import os


def get_distance_voxels(num_route1, num_route2, tab_routes_voxels=None):
    
    common_parts = len(list(set(tab_routes_voxels[num_route1]) & set(tab_routes_voxels[num_route2])))
    
    union = len(list(set(tab_routes_voxels[num_route1]) | set(tab_routes_voxels[num_route2])))
    

    #print(len(tab_routes_voxels[num_route1-1]))
    
    sim1 = -1
    sim2 = -1
    
    if(len(tab_routes_voxels[num_route1-1]) == 0):
        sim1 = 0.0
    if(len(tab_routes_voxels[num_route2-1]) == 0):
        sim2 = 0.0
        
    if(sim1<0):
        sim1 = common_parts/union
    if(sim2<0):
        sim2 = common_parts/union
    
    #print("cp: ", common_parts)

    return [1-sim1, 1-sim2]
    
    #return max(len(tab_routes_voxels[num_route1-1]), len(tab_routes_voxels[num_route2-1]))-common_parts
    
    #return (common_parts/max(len(tab_routes_voxels[num_route1-1]), len(tab_routes_voxels[num_route2-1])))*100

def get_distance_voxels_symetric(num_route1, num_route2):
    dist = get_distance_voxels(num_route1, num_route2)
    return max(dist[0], dist[1])


def get_distance_euclidian(route1, route2, pca):
    v1 = dataframe_to_array(route1)
    v2 = dataframe_to_array(route2)
    harmonize_route(v1, v2)
    dist = euclidean(pca.fit_transform(v1), pca.fit_transform(v2))
    return [dist, dist]

def get_distance_hausdorff(route1, route2):
    v1 = dataframe_to_array(route1)
    v2 = dataframe_to_array(route2)
    return [directed_hausdorff(v1, v2)[0], directed_hausdorff(v2, v1)[0]]

class GPSDistanceTuto(Metric):
    def __init__(self):
        super(GPSDistanceTuto, self).__init__(feature=ResampleFeature(nb_points=256))
        
    def are_compatible(self, shape1, shape2):
        return shape1 == shape2
    
    def dist(self, v1, v2):
        x = [geopy.distance.geodesic([p[0][0],p[0][1]], [p[1][0],p[1][1]]).km for p in list(zip(v1,v2))]
        currD = np.mean(x)
        return currD


class GPSDistanceCustom(Metric):
    def __init__(self):
        super(GPSDistanceCustom, self).__init__(feature=ResampleFeature(nb_points=256))
        self.tab_route_voxel = []
        self.dict_routes = {}
        self.tab_dist = []
        
    def are_compatible(self, shape1, shape2):
        return shape1 == shape2
    
    def dist(self, v1, v2):
        tab_route_voxel = []
        tab_route_voxel.append([])
        tab_route_voxel.append([])
        dict1 = get_voxels_from_route(v1)
        dict2 = get_voxels_from_route(v2)
        
        for key in dict1:
            tab_route_voxel[0].append(key)
            
        for key in dict2:
            tab_route_voxel[1].append(key)
        
        dist = get_distance_voxels(tab_route_voxel, 0, 1)
        #print(dist)
        self.tab_dist.append(dist)
        return dist



def neuro_cluster(infile, outfile, m, threshold):
    with open(infile,'rb') as infile:
        df_metric = pickle.load(infile)  

    streams = []
    for i in range(df_metric.iloc[-1]["route_num"]+1):
        df_temp = df_metric[df_metric["route_num"]==i]
        del df_temp["route_num"]
        if(not(df_temp.empty)):
            streams.append(np.array(df_temp.values.tolist()))
    
    #metric = GPSDistanceTuto()
    #qb = QuickBundles(threshold=threshold, metric=metric)
    
    qb = QuickBundles(threshold=threshold, metric=m)
    streams = np.array(streams)
    clusters = qb.cluster(streams)
    print(len(clusters))
    
    '''with open(outfile, 'wb') as outfile:
        pickle.dump(clusters, outfile)'''
    
        