import pickle
import data
import torch
import torch.nn
import train_NN
import voxels
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import random
import os

from RNN import *
import clustering as cl


def main(args):
    
    project_folder = args.project_folder

    with open(args.path+"files/"+project_folder+"/data_processed/osmnx_pathfinding_simplified.df",'rb') as infile:
        df_pathfinding = pickle.load(infile)
    with open(args.path+"files/"+project_folder+"/clustering/dbscan_observations.tab",'rb') as infile:
        tab_clusters = pickle.load(infile)
    with open(args.path+"files/"+project_folder+"/clustering/voxels_clustered_osmnx.dict",'rb') as infile:
        dict_voxels = pickle.load(infile)

    df = df_pathfinding

    tab_routes_voxels, _, _ = voxels.generate_voxels(df, df.iloc[0]["route_num"], df.iloc[-1]["route_num"])

    tab_routes_voxels_int = []
    
    df_voxels = pd.DataFrame()

    df_voxels_train = pd.DataFrame()
    df_voxels_test = pd.DataFrame()

    max_cluster = max(tab_clusters)+1 #total number of cluster
    
    tab_num_test = [] #will contain the number of the routes in the testing set
        
    dict_clusters = cl.tab_clusters_to_dict(tab_clusters)
    
    for key in dict_clusters:
        if(key != -1):
            tab_num_test += random.sample(dict_clusters[key], round(args.percentage_test/100*len(dict_clusters[key])))

    
    if(os.path.isfile(args.path+"./files/"+project_folder+"/neural_networks/saved/num_test.tab")):
        print("Using previous tab num test")
        with open(args.path+"./files/"+project_folder+"/neural_networks/saved/num_test.tab",'rb') as infile:
            tab_num_test = pickle.load(infile)
    
    tab_num_train = list(range(len(tab_routes_voxels)))
    tab_num_noise = [i for i, e in enumerate(tab_clusters) if e == -1]
    tab_num_train = [x for x in tab_num_train if x not in tab_num_test]
    tab_num_train = [x for x in tab_num_train if x not in tab_num_noise]
          
    print(len(tab_num_train), "routes in the training set,", len(tab_num_test), "routes in the testing set,", len(tab_num_noise), "routes not used (DBSCAN noise).")
    for i in range(len(tab_routes_voxels)):
        nb_vox = 0
        tab_routes_voxels_int.append([])
        route_voxels = tab_routes_voxels[i]
        for vox in route_voxels:
            if(nb_vox%args.voxels_frequency==0): 
                points = dict_voxels[vox]["cluster"]+1
                tab_routes_voxels_int[i].append([points])
            nb_vox += 1
            
        tab_routes_voxels_int[i] = torch.Tensor(tab_routes_voxels_int[i])
           

    padded_data = torch.nn.utils.rnn.pad_sequence(tab_routes_voxels_int, batch_first=True)   
    
    size_data = 1

    learning_rate = args.lr

    lstm = RNN_LSTM(size_data, max_cluster, args.hidden_size, args.num_layers, args.bidirectional, args.dropout)


    network = lstm

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss = nn.NLLLoss()
    
    '''network.load_state_dict(torch.load(args.path+"files/"+project_folder+"/neural_networks/saved/network_temp.pt"))
    network.eval()'''

    tab_loss, tab_predict = train_NN.train(padded_data, tab_num_train, tab_num_test, tab_clusters, loss, optimizer, network, size_data, args.num_samples)


    '''g_predict = learning.test(df_test, None, tab_clusters, size_data, loss)
    print("Random:", g_predict*100, "%")'''
    
    g_predict, _ = train_NN.test(padded_data, tab_num_train, network, tab_clusters, size_data)
    print("Good train predict:", g_predict*100, "%")
    
    if(args.percentage_test > 0):
        g_predict, _ = train_NN.test(padded_data, tab_num_test, network, tab_clusters, size_data)
        print("Good test predict:", g_predict*100, "%")
    
    if(g_predict > 0.1):
        print("Saving network...")
        data.check_file(args.path+"files/"+project_folder+"/neural_networks/network_temp.pt", [])
        torch.save(network.state_dict(), args.path+"files/"+project_folder+"/neural_networks/network_temp.pt")
        with open(args.path+"files/"+project_folder+"/neural_networks/num_test.tab",'wb') as outfile:
            pickle.dump(tab_num_test, outfile)
        with open(args.path+"files/"+project_folder+"/neural_networks/network.param",'wb') as outfile:
            pickle.dump(args, outfile)
        with open(args.path+"files/"+project_folder+"/neural_networks/loss.tab",'wb') as outfile:
            pickle.dump(tab_loss, outfile)
        with open(args.path+"files/"+project_folder+"/neural_networks/predict.tab",'wb') as outfile:
            pickle.dump(tab_predict, outfile)
        
    
    
    plt.plot(tab_loss[0], label='train')
    plt.plot(tab_loss[1], label='test')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.show()
    
    plt.plot(tab_predict[0], color='blue', label='train')
    plt.plot(tab_predict[1], color='red', label='test')
    plt.ylabel('Prediction')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__": 
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', type=str, default="./", help="path to the project's main folder")
    parse.add_argument('--voxels-frequency', type=int, default=4, help="frequency of voxels to send to the network")
    parse.add_argument('--num-layers', type=int, default=2, help="number of layers in the LSTM network")
    parse.add_argument('--hidden-size', type=int, default=256, help="size of the hidden layer(s) in the network")
    parse.add_argument('--num-samples', type=int, default=7500, help="number of data (chosen randomly) to send to the network")
    parse.add_argument('--lr', type=float, default=5e-4, help='learning rate of the algorithm')
    parse.add_argument('--percentage-test', type=int, default=20, help='percentage of data to use as testing')
    parse.add_argument('--bidirectional', type=bool, default=False, help='change the LSTM in a bidirectional one')
    parse.add_argument('--dropout', type=float, default=0, help='set the dropout layers parameter')
    parse.add_argument('--project-folder', type=str, default="monresovelo", help='folder of the project')
    
    main(parse.parse_args())
