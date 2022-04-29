def get_distance_voxels(num_route1, num_route2, tab_routes_voxels):
    """
    Get the distance between 2 routes using voxels (see our paper for more information).
    Parameters
    ----------
    num_route1 : int
        The index of the first route in tab_routes_voxels.
    num_route2 : int
        The index of the second route in tab_routes_voxels.
    tab_routes_voxels : list
        A list of lists, each list contains the voxels affiliated to a route.
    """
    
    common_parts = len(list(set(tab_routes_voxels[num_route1]) & set(tab_routes_voxels[num_route2])))
    
    union = len(list(set(tab_routes_voxels[num_route1]) | set(tab_routes_voxels[num_route2])))
    
    
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
    

def get_distance_voxels_symetric(num_route1, num_route2):
    dist = get_distance_voxels(num_route1, num_route2)
    return max(dist[0], dist[1])