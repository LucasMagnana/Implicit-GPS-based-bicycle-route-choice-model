import pickle
import pandas as pd
import argparse
import torch
import random

import python.voxels as voxels
import python.data as data


def find_cluster(route, network, voxels_frequency, dict_voxels, clustering, df_temp):

    nb_new_cluster = 0

    tab_voxels_int = []
    nb_vox = 0
    for vox in route:
        if(nb_vox%voxels_frequency==0):
            vox_str = vox.split(";")
            vox_int = [int(vox_str[0]), int(vox_str[1])]
            tab_points = voxels.get_voxel_points(vox_int)
            if vox in dict_voxels:
                cl = dict_voxels[vox]["cluster"]+1
            else:
                cl = clustering.predict([[tab_points[0][0], tab_points[0][1], 0]])[0]+1
                nb_new_cluster += 1
            points = [cl]
            tab_voxels_int.append(points)
        nb_vox += 1
    tab_voxels_int = torch.Tensor(tab_voxels_int)
    #print(route, tab_voxels_int)
    route = tab_voxels_int #torch.nn.utils.rnn.pad_sequence(tab_voxels_int, batch_first=True)
    tens_route = route.unsqueeze(0)

    output = network(tens_route)
    pred = output.argmax(dim=1, keepdim=True)
    
    return pred.item(), nb_new_cluster
    
