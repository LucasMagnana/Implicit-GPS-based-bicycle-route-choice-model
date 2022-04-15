import math
import pandas as pd

n_voxel = 3
vox_divider = 2
nb_subvox = (10/vox_divider)

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def find_voxel_int(p, new_voxel=True):
    """
    Find the voxel in which a point is by truncating its position. Voxel's position are transformed into 
    int to be manipulated in an easier way.
    Parameters
    ----------
    p : list of two int
        The point 
    n : int, optional
        Number of digits to truncate
        
    Returns
    -------
    list of two int
        Position of the voxel's low left point
    """
    v_lat = math.trunc(p[0]*10**(n_voxel+1))
    v_lon = math.trunc(p[1]*10**(n_voxel+1))

    if(new_voxel):
        if(v_lat < 0 and v_lat%nb_subvox == 0 and v_lat != p[0]*10**(n_voxel+1)):
            v_lat -= 1
        if(v_lon < 0 and v_lon%nb_subvox == 0 and v_lon != p[1]*10**(n_voxel+1)):
            v_lon -= 1
    
    while(v_lat%nb_subvox != 0):
        v_lat -= 1
    while(v_lon%nb_subvox != 0):
        v_lon -= 1
    
    return [v_lat, v_lon]


def line_intersection(line1, line2):
    """
    Find the point of intersection between two lines
    Parameters
    ----------
    line1 : list of two points (a point is a list of two int)
        First line  
    line2 : list of two points (a point is a list of two int)
        Second line  
        
    Returns
    -------
    list of two int
        Position of the intersection
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        #print("Line does not intersect")
        return [99999999999, 99999999999]

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]
    

def get_voxel_points(vox, num_vox=-1):
    """
    Take the position of the low left point of a voxel transformed into an int 
    and return this voxel's four real points.
    Parameters
    ----------
    vox : list of two int
        Position of the voxel's low left point transformed into an int
    num_vox : int
        Number of the voxel, used later to differentiate voxels
        
    Returns
    -------
    list 
        list of the four points (a point is a list of two int)
    """
    tab_vox = []
    vox_float = [vox[0]*10**(-n_voxel-1), vox[1]*10**(-n_voxel-1)]
    vox_float.append(num_vox)
    vox_float.append(1)
    tab_vox.append(vox_float)
    tab_vox.append([vox_float[0]+nb_subvox*10**(-n_voxel-1), vox_float[1], num_vox, 1])
    tab_vox.append([vox_float[0]+nb_subvox*10**(-n_voxel-1), vox_float[1]+nb_subvox*10**(-n_voxel-1), num_vox, 1])
    tab_vox.append([vox_float[0], vox_float[1]+nb_subvox*10**(-n_voxel-1), num_vox, 1])
    tab_vox.append(vox_float)
    
    return tab_vox


def get_adjacent_voxel(vox, lat_diff, lon_diff):
    return [vox[0]+lat_diff*nb_subvox, vox[1]+lon_diff*nb_subvox]


def voxel_convolution(vox, dict_vox, dict_vox_used, num_vox, lat_diff, lon_diff):
    """
    With a voxel, check if one of his neighbour exists and if it has already been used.
    ----------
    vox : list of two int
        Position of the voxel's low left point transformed into an int
    dict_vox : dict
        Dictionary of existing voxels 
    dict_vox_used : dict
        Dictionary of voxels that have already been used
    num_vox : int
        Number of the voxel, used later to differentiate voxels
    lat_diff : int
        Difference of latitude (the unit is voxel) between the voxel and the neighbour
    lon_diff : int
        Difference of longitude (the unit is voxel) between the voxel and the neighbour
        
    Returns
    -------
    list
        If the voxel exists and has not been used : 
            A list containing the voxel's low left point transformed into an int and the list containing all routes
            that are going through the voxel
        Else:
            An empty list
    """
    vox_adj = get_adjacent_voxel(vox, lat_diff, lon_diff)
    key_adj = str(int(vox_adj[0]))+";"+str(int(vox_adj[1]))
    if(key_adj in dict_vox and not(key_adj in dict_vox_used)):
        return [vox_adj, dict_vox[key_adj], key_adj]
    return []
        
    
def is_voxel_neighbour(v1, v2):
    tab_vox_adj = []
    tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, -1, 0))
    tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, 1, 0))
    tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, 0, 1))
    tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, 0, -1))
    tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, -1, -1))
    tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, -1, 1))
    tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, 1, -1))
    tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, 1, 1))
    for vox_adj in tab_vox_adj:
        if(len(vox_adj)>0):
            if(next_vox_int == vox_adj[0]):
                vox_int = next_vox_int
                break


def differentiate_voxels_sequences(tab_voxels, dict_vox):
    seq=0
    tab_voxel_sequence = []
    tab_voxel_sequence.append([])
    vox_int = find_voxel_int([tab_voxels[0][0], tab_voxels[0][1]])
    for vox in range(0, len(tab_voxels)-4, 4):
        tab_voxel_sequence[seq].append(vox_int)
        next_vox_int = find_voxel_int([tab_voxels[vox+4][0], tab_voxels[vox+4][1]])
       
        if(vox_int != next_vox_int):
            for i in range(len(next_vox_int)):
                for j in range(len(next_vox)):
                    a=0 #TODO
            
            
            tab_voxel_sequence.append([])
            seq += 1
            vox_int = next_vox_int
    tab_voxel_sequence[seq].append(vox_int)
    return tab_voxel_sequence



def generate_voxels(df, starting, ending, bikepath=False):
    """
    With a dataframe containing gps points separated in routes, creates a dict of voxels.  
    Parameters
    ----------
     df : pandas' DataFrame with columns=['lat', 'lon', 'route_num']
        Dataframe to use 
    ending : int
        Number of routes to use in the dataframe 
        
    Returns
    -------
    dict of voxels
        Keys of this dict are strings containing the position of voxels' low left points transformed to int
        and separated by a ';'.
        Values of this dict are lists containing the number of all routes that pass through the voxel.
    """
    
    dict_vox = {}
    tab_routes_voxels = []
    for route_num in range(starting, ending+1):
        print("\rCalculating voxels for route {}/{}.".format(route_num, ending), end="")
        tab_routes_voxels.append([])
        route = df[df["route_num"]==route_num]
        points = route.values.tolist()

        if(len(points) > 1):
            vox_starting_routes = find_voxel_int(points[0])
            vox_finishing_routes = find_voxel_int(points[-1])

        elif(len(points)==1):
            vox_int = find_voxel_int(points[0])
            key = str(int(vox_int[0]))+";"+str(int(vox_int[1]))
            if key in dict_vox:
                if(route_num not in dict_vox[key]["tab_routes_real"]):
                    dict_vox[key]["tab_routes_real"].append(route_num)
            else :
                dict_vox[key] = {"tab_routes_real": [route_num], "tab_routes_extended": [], "tab_routes_starting": [], "tab_routes_finishing": [],
                            "cyclability_coeff": 0}

            if(route_num not in dict_vox[key]["tab_routes_starting"]):
                dict_vox[key]["tab_routes_starting"].append(route_num)

            if(route_num not in dict_vox[key]["tab_routes_finishing"]):
                dict_vox[key]["tab_routes_finishing"].append(route_num)

            if(bikepath==True):
                dict_vox[key]["cluster"] = route_num

            if(not key in tab_routes_voxels[-1]):
                tab_routes_voxels[-1].append(key)
                    
        for j in range(len(points)-1):
            p1 = points[j] #we take two points in the dataframe that create a line
            p2 = points[j+1]

            if(p1[0]>p2[0]):
                lat_orientation = -nb_subvox #the line is going down
            else:
                lat_orientation = nb_subvox #the line is going up

            if(p1[1]>p2[1]):
                lon_orientation = -nb_subvox #the line is going left 
            else:
                lon_orientation = nb_subvox #the line is goin right

            vox_int = find_voxel_int(p1) #find the start voxel
            vox_final_int = find_voxel_int(p2) #find the final voxel

            #while the final voxel has not been reached
            while(vox_int[0] != vox_final_int[0] or vox_int[1] != vox_final_int[1]):

                vox_float = [vox_int[0]*10**(-n_voxel-1), vox_int[1]*10**(-n_voxel-1)] #transform the vox into real points
                
                key = str(int(vox_int[0]))+";"+str(int(vox_int[1])) #save the voxel
                if key in dict_vox:
                    if(route_num not in dict_vox[key]["tab_routes_real"]):
                        dict_vox[key]["tab_routes_real"].append(route_num)
                else :
                    dict_vox[key] = {"tab_routes_real": [route_num], "tab_routes_extended": [], "tab_routes_starting": [], "tab_routes_finishing": [],
                                "cyclability_coeff": 0}

                if(vox_int == vox_starting_routes and route_num not in dict_vox[key]["tab_routes_starting"]):
                    dict_vox[key]["tab_routes_starting"].append(route_num)

                if(bikepath==True):
                    dict_vox[key]["cluster"] = route_num

                if(not key in tab_routes_voxels[-1]):
                    tab_routes_voxels[-1].append(key)
                    
                '''find the good intersection point (if the line is going up, we search the intersection between 
                it and the up line of the voxel for example)'''
                if(lat_orientation>0):
                    intersection_lat = line_intersection([p1, p2], [[vox_float[0]+nb_subvox*10**(-n_voxel-1), vox_float[1]],
                                                        [vox_float[0]+nb_subvox*10**(-n_voxel-1), vox_float[1]+nb_subvox*10**(-n_voxel-1)]])
                else:
                    intersection_lat = line_intersection([p1, p2], [vox_float, [vox_float[0], vox_float[1]+nb_subvox*10**(-n_voxel-1)]])

                    
                '''same for left and right'''
                if(lon_orientation>0): 
                    intersection_lon = line_intersection([p1, p2], [[vox_float[0], vox_float[1]+nb_subvox*10**(-n_voxel-1)], 
                                                        [vox_float[0]+nb_subvox*10**(-n_voxel-1), vox_float[1]+nb_subvox*10**(-n_voxel-1)]])
                else:
                    intersection_lon = line_intersection([p1, p2], [vox_float, [vox_float[0]+nb_subvox*10**(-n_voxel-1), vox_float[1]]])

                #calculate the distance between the first point of the line and the two intersection points
                intersection_lon_distance = math.sqrt((p1[0]-intersection_lon[0])**2+(p1[1]-intersection_lon[1])**2)
                intersection_lat_distance = math.sqrt((p1[0]-intersection_lat[0])**2+(p1[1]-intersection_lat[1])**2)

                #find the shorter distance then go to the next voxel using the orientation of the line
                if(intersection_lat_distance<=intersection_lon_distance): 
                    vox_int[0] += lat_orientation
                if(intersection_lon_distance<=intersection_lat_distance): 
                    vox_int[1] += lon_orientation
                    
            key = str(int(vox_int[0]))+";"+str(int(vox_int[1])) #end of the while loop, save the last voxel
            if key in dict_vox:
                if(route_num not in dict_vox[key]["tab_routes_real"]):
                    dict_vox[key]["tab_routes_real"].append(route_num)
            else :
                dict_vox[key] = {"tab_routes_real": [route_num], "tab_routes_extended": [], "tab_routes_starting": [], "tab_routes_finishing": [],
                                "cyclability_coeff": 0}

            if(vox_int == vox_finishing_routes and route_num not in dict_vox[key]["tab_routes_finishing"]):
                dict_vox[key]["tab_routes_finishing"].append(route_num)

            if(bikepath==True):
                dict_vox[key]["cluster"] = route_num

            if(not key in tab_routes_voxels[-1]):
                tab_routes_voxels[-1].append(key)

    nb_max_routes = 0

    tab_routes_voxels_global = [[] for i in range(ending+1-starting)]
               
    for key in dict_vox:
        tab_routes = dict_vox[key]["tab_routes_real"]

        for route in tab_routes:
            tab_routes_voxels_global[route].append(key)

        vox_str = key.split(";")
        vox_int = [int(vox_str[0]), int(vox_str[1])]
        
        #creation of a list containing all neighbours of the voxel
        tab_vox_adj = []
        
        tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, -1, 0))
        tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, 1, 0))
        tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, 0, 1))
        tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, 0, -1))
                
        tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, -1, -1))
        tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, 1, -1))
        tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, -1, 1))
        tab_vox_adj.append(voxel_convolution(vox_int, dict_vox, {}, 0, 1, 1))
        for vox in tab_vox_adj:
            if(len(vox)>0):
                if(not(set(tab_routes).issubset((set(vox[1]["tab_routes_real"]))))):
                    diff_tab_routes = list((set(tab_routes)-set(vox[1]["tab_routes_real"]))|(set(vox[1]["tab_routes_real"])-set(tab_routes)))
                    for i in range(len(diff_tab_routes)):
                        if(diff_tab_routes[i] not in tab_routes
                          and diff_tab_routes[i] not in dict_vox[key]["tab_routes_extended"]):
                            dict_vox[key]["tab_routes_extended"].append(diff_tab_routes[i])
                            tab_routes_voxels_global[diff_tab_routes[i]].append(key)
                            

        if(len(dict_vox[key]["tab_routes_real"]) + len(dict_vox[key]["tab_routes_extended"]) > nb_max_routes):
            nb_max_routes = len(dict_vox[key]["tab_routes_real"]) + len(dict_vox[key]["tab_routes_extended"])

    if(bikepath == False):
        for key in dict_vox:
            dict_vox[key]["cyclability_coeff"] = (len(dict_vox[key]["tab_routes_real"]) + len(dict_vox[key]["tab_routes_extended"]))/nb_max_routes

    print()
    return tab_routes_voxels, tab_routes_voxels_global, dict_vox



def get_tab_routes_voxels_global(dict_voxels, nb_routes, starting):
    t = []
    for i in range(nb_routes+1):
        if(len(t)<=i):
            t.append([])
    for key in dict_voxels:
        tab_routes = dict_voxels[key]["tab_routes_real"]+dict_voxels[key]["tab_routes_extended"]
        if(i+starting in tab_routes and i+starting not in t[i]):
            t[i].append(key)
    return t


def get_voxels_from_route(route):
    df_temp = pd.DataFrame(route, columns=["lat", "lon"])
    df_temp["route_num"] = 1
    return create_dict_vox(df_temp, 1)

def get_voxels_with_min_routes(dict_vox, min_routes, glob=True):
    """
    Return all voxels or groups of voxels that have at least a number of routes passing through themselves.
    Parameters
    ----------
    dict_vox : dict
        Dictionary of existing voxels 
    min_routes : int
        Minimum number of routes passing through voxels / groups of voxels
        
    Returns
    -------
    list 
        List of voxels that have or are part of a group that have at least 'min_routes' routes 
        passing through itself. A voxel is a list of four points.
    """
    num_vox = 0 #used to differentiate voxels
    dict_vox_used = {}
    tab_voxel_with_min_routes = [] #final list containing all voxels that matches with the conditions
    
    for key in dict_vox: #for all voxels
        tab_routes = dict_vox[key]["tab_routes_real"]
        if(glob):
            tab_routes += dict_vox[key]["tab_routes_extended"]
        
        #print(dict_vox[key][0], dict_vox[key][1])
        
        vox_str = key.split(";")
        vox_int = [int(vox_str[0]), int(vox_str[1])]
        
        
        #if the voxels has at least 'min_routes' routes and has not been saved we save it
        if(key not in dict_vox_used and len(tab_routes) >= min_routes):
            tab_voxel_with_min_routes += get_voxel_points(vox_int, num_vox)
            dict_vox_used[key] = True
            num_vox -= 1
            
    return tab_voxel_with_min_routes
    

def get_dict_routes_voxels(dict_voxels, nb_routes, df):
    dict_routes_voxels = {}
    for key in dict_voxels:
        for i in range(nb_routes):
            tab_routes = dict_voxels[key]["tab_routes_real"]+dict_voxels[key]["tab_routes_extended"]
    return t
        
        