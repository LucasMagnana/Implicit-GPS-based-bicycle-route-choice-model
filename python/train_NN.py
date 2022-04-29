
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import data as data
import numpy as np
import matplotlib.pyplot as plt
import random
from RNN import *
import datetime


def train(data, tab_num_train, tab_num_test, tab_clusters, loss, optimizer, network, size_data, nb_step):


    tens_clusters = torch.Tensor(tab_clusters)

    loss_tab = [[], []]
    sum_loss = 0
    predict_tab = [[],[]]
    quarter = 1
    print("=========================")
    print("start:", datetime.datetime.now().time())
    for s in range(nb_step+1):
        print("\rStep {}/{}.".format(s, nb_step), end="")
        if(s > 0):           
            if(s%max(1, (nb_step//30))==0):
                predict_train, loss_train = test(data, tab_num_train, network, tab_clusters, size_data, loss)
                predict_tab[0].append(predict_train)
                loss_tab[0].append(loss_train)

                if(len(tab_num_test)>0):
                    predict_test, loss_test = test(data, tab_num_test, network, tab_clusters, size_data, loss)
                    predict_tab[1].append(predict_test)
                    loss_tab[1].append(loss_test)

        tens_num_batch = torch.LongTensor(random.sample(tab_num_train, 30))
        tens_batch = torch.index_select(data, 0, tens_num_batch)

        target = torch.index_select(tens_clusters, 0, tens_num_batch).long()

        #network.zero_grad()
        optimizer.zero_grad()
        output = network(tens_batch)
        
        l = loss(output, target)
        sum_loss += l.item()
        l.backward()
        optimizer.step()
    print()
    print("end:", datetime.datetime.now().time())
    print("=========================")
    return loss_tab, predict_tab

def test(data, tab_num, network, tab_clusters, size_data, loss=None):
    good_predict = 0
    nb_predict = 0
    sum_loss = 0
    for i in tab_num:
        tens_route = data[i].unsqueeze(0)
        target = torch.LongTensor(tab_clusters[i])
        output = network(tens_route)
            
        if(loss != None):
            l = loss(output, torch.LongTensor([tab_clusters[i]]))
            sum_loss += l.item()
        pred = output.argmax(dim=1, keepdim=True)
        if(tab_clusters[i] == pred.item()):
            good_predict += 1
        nb_predict += 1
    #print(good_predict, nb_predict)
    return good_predict/nb_predict, sum_loss/nb_predict


def test_random(df, tab_clusters):
    good_predict = 0
    nb_predict = 0
    last_clust = max(tab_clusters)
    for i in range(df.iloc[-1]["route_num"]+1):
        if(tab_clusters[i] != -1):
            pred = random.randint(0, last_clust)
            if(tab_clusters[i] == pred):
                good_predict += 1
            nb_predict += 1
    return good_predict/nb_predict
