
import networkx as nx
import time
import python.data as data
import copy
import heapq

def l(p, G):
    l_p = 0
    for e in range(len(p)-1):
        l_p += G[p[e]][p[e+1]][0]["length"]
    return l_p



def overlap_ratio(p_1, p, G, l_p=-1, debug=False):
    if(l_p < 0):
        l_p = l(p, G)
    overlap = []
    for n in p_1:
        if(n in p):
            overlap.append(n)
    l_ov = 0
    
    for e in range(len(overlap)-1):
        if(overlap[e+1] in G[overlap[e]]):
            l_ov += G[overlap[e]][overlap[e+1]][0]["length"]

    if(debug):
        print(len(p_1), len(p), len(overlap), l_ov, l_p)
    return l_ov/l_p


def lemma1(vsim, threshold):
    for sim in vsim:
        if(sim>threshold):
            return True
    return False
    

def lemma2(vsim_p_i, vsim_p_j, l_p_i, l_p_j):
    if(l_p_i<l_p_j):
        for i in range(len(vsim_p_i)):
            if(vsim_p_i[i]>vsim_p_j[i]):
                return False
        return True
    return False


def get_weight(s, t, G):
    if("length" not in G[0]):
        print(G[0])
    return G[0]["length"]




def MultiPass(G, tree, nodes, d_point, f_point, k=5, threshold=0.5):
    
    d_idx = tree.query([d_point], k=1, return_distance=False)[0]
    f_idx = tree.query([f_point], k=1, return_distance=False)[0]
    s = nodes.iloc[d_idx].index.values[0]
    t = nodes.iloc[f_idx].index.values[0]
    
    
    
    '''for n in G.nodes:
            G.nodes[n]["lower_bound"] = data.distance_between_points([G.nodes[n]["x"], G.nodes[n]["y"]], [G.nodes[t]["x"], G.nodes[t]["y"]])'''
    
    
    
    
    sp = nx.shortest_path(G, s, t, weight="length")
    P_lo = [tuple(sp)] 
    l_P_lo = [l(sp, G)]
   
    cont=True
    
    print(datetime.datetime.now().time())
           
    while(len(P_lo)<k and cont):
        Q = [(s, tuple([s]))]
        Q_blacklist = []
        dict_labels = dict.fromkeys(G.nodes.keys())
        dict_vsim = {}
        while(len(Q) > 0):
            cont=False
            label = Q.pop()
            #print("Q pop:", label[1])
            '''if(len(label[1]) > 3):
                break'''
            if(label[0] == t):
                P_lo.append(label[1])
                l_P_lo.append(l(label[1], G))
                #print("found")
                cont=True
                #print(label[1])
                break
            else:
                sorted_edges=sorted(G.edges(label[0], data=True), key=lambda e: e[2]["length"])
                for i in sorted_edges:
                    if(i[1] not in label[1]):                   
                        pruned = False
                        next_path = list(copy.deepcopy(label[1]))
                        next_path.append(i[1])
                        next_label = (i[1], tuple(next_path))
                        nl_l = l(next_label[1], G)
                        if(label[1] in dict_vsim):
                            if(len(dict_vsim[label[1]]) != len(P_lo)):
                                if(len(dict_vsim[label[1]]) == len(P_lo)-1):
                                    dict_vsim[label[1]].append(overlap_ratio(label[1], P_lo[-1], G, l_P_lo[-1]))
                                else:
                                    print("bouletto")
                            dict_vsim[next_label[1]] = copy.deepcopy(dict_vsim[label[1]])
                            for i in range(len(dict_vsim[next_label[1]])):
                                if(next_label[1][-1] in P_lo[i] and next_label[1][-2] in P_lo[i] and P_lo[i].index(next_label[1][-2]) == P_lo[i].index(next_label[1][-1])-1):
                                    dict_vsim[next_label[1]][i] += G[next_label[1][-2]][next_label[1][-1]][0]["length"]/l_P_lo[i]
                        else:
                            dict_vsim[next_label[1]] = []
                            for i in range(len(P_lo)):
                                dict_vsim[next_label[1]].append(overlap_ratio(next_label[1], P_lo[i], G, l_P_lo[i]))
                        #print(next_label, dict_vsim[next_label[1]])
                        if(not lemma1(dict_vsim[next_label[1]], threshold)):
                            if(dict_labels[next_label[0]] != None):
                                for p_t in dict_labels[next_label[0]]:
                                    if(lemma2(dict_vsim[p_t], dict_vsim[next_label[1]], l(p_t, G), nl_l)):
                                        pruned=True
                                        break   
                            if(not pruned):
                                if(dict_labels[next_label[0]] == None):
                                    dict_labels[next_label[0]] = [next_label[1]]
                                else:
                                    for p_t in dict_labels[next_label[0]]:
                                        if(lemma2(dict_vsim[next_label[1]], dict_vsim[p_t], nl_l, l(p_t, G))):
                                            dict_labels[next_label[0]].remove(p_t)
                                            if((next_label[0], p_t) in Q):
                                                Q.remove((next_label[0], p_t))
                                                Q_blacklist.append(p_t) 
                                    dict_labels[next_label[0]].append(next_label[1])
                                
                                if(next_label[1] not in Q_blacklist):
                                    Q.insert(0,next_label)
                                #print("Q add:", next_label, dict_vsim[next_label[1]])
                                #print(len(Q))
    
    print(datetime.datetime.now().time())
    if(len(l_P_lo)<k):
       raise NameError('MultiPass Failed')
    return P_lo
    
    
    
def OnePass(G, tree, nodes, d_point, f_point, k=5, threshold=0.5):
    
    start = time.time()
    
    '''G = nx.Graph()
    G.add_nodes_from(range(7))

    G.add_edges_from([(0, 1, {0:{"length":6}}), 
                      (0, 2, {0:{"length":4}}), 
                      (0, 3, {0:{"length":3}}),
                      (1, 3, {0:{"length":2}}), 
                      (1, 6, {0:{"length":6}}), 
                      (2, 3, {0:{"length":3}}),
                      (2, 4, {0:{"length":5}}),
                      (3, 4, {0:{"length":5}}),
                      (3, 5, {0:{"length":3}}),
                      (4, 5, {0:{"length":1}}), 
                      (4, 6, {0:{"length":2}}),
                      (5, 6, {0:{"length":2}})])
    
    s = 0
    t = 6'''
    
    
    d_idx = tree.query([d_point], k=1, return_distance=False)[0]
    f_idx = tree.query([f_point], k=1, return_distance=False)[0]
    s = nodes.iloc[d_idx].index.values[0]
    t = nodes.iloc[f_idx].index.values[0]
    
    
    sp = nx.shortest_path(G, s, t, weight=get_weight)
    P_lo = [tuple(sp)] 
    l_P_lo = [l(sp, G)]
    
    Q = [(s, tuple([s]))]
    Q_blacklist = []
    dict_labels = dict.fromkeys(G.nodes.keys())
    dict_vsim = {}
    while(len(P_lo)<k and len(Q) > 0 and time.time() - start < 600):
        label = Q.pop()
        '''if(len(label[1])>1 and lemma1(dict_vsim[label[1]], threshold)):
            print("chatte")'''
        #print("Q pop:", label[1])
        '''if(len(label[1]) > 3):
            break'''
        if(label[0] == t):
            P_lo.append(label[1])
            l_P_lo.append(l(label[1], G))
            #print("found")
            for i in dict_vsim:
                if(len(dict_vsim[i])<len(P_lo)):
                    dict_vsim[i] = []
                    for j in range(len(P_lo)):
                        dict_vsim[i].append(overlap_ratio(i, P_lo[j], G, l_P_lo[j]))
            Q_prime = []
            for i in Q:
                if(not lemma1(dict_vsim[i[1]], threshold)):
                    Q_prime.append(i)
            Q = Q_prime

        else:
            sorted_edges=sorted(G.edges(label[0], data=True), key=lambda e: e[2]["length"])
            for i in sorted_edges:
                if(i[1] not in label[1]):                   
                    pruned = False
                    next_path = list(copy.deepcopy(label[1]))
                    next_path.append(i[1])
                    next_label = (i[1], tuple(next_path))
                    nl_l = l(next_label[1], G)
                    if(label[1] in dict_vsim):
                        if(len(dict_vsim[label[1]]) != len(P_lo)):
                            if(len(dict_vsim[label[1]]) == len(P_lo)-1):
                                dict_vsim[label[1]].append(overlap_ratio(label[1], P_lo[-1], G, l_P_lo[-1]))
                            else:
                                print("bouletto")
                        dict_vsim[next_label[1]] = copy.deepcopy(dict_vsim[label[1]])
                        for i in range(len(dict_vsim[next_label[1]])):
                            if(next_label[1][-1] in P_lo[i] and next_label[1][-2] in P_lo[i] and P_lo[i].index(next_label[1][-2]) == P_lo[i].index(next_label[1][-1])-1):
                                dict_vsim[next_label[1]][i] += G[next_label[1][-2]][next_label[1][-1]][0]["length"]/l_P_lo[i]
                    else:
                        dict_vsim[next_label[1]] = []
                        for i in range(len(P_lo)):
                            dict_vsim[next_label[1]].append(overlap_ratio(next_label[1], P_lo[i], G, l_P_lo[i]))
                    #print(next_label, dict_vsim[next_label[1]])
                    if(not lemma1(dict_vsim[next_label[1]], threshold)):
                        if(dict_labels[next_label[0]] != None):
                            for p_t in dict_labels[next_label[0]]:
                                if(lemma2(dict_vsim[p_t], dict_vsim[next_label[1]], l(p_t, G), nl_l)):
                                    pruned=True
                                    break   
                        if(not pruned):
                            if(dict_labels[next_label[0]] == None):
                                dict_labels[next_label[0]] = [next_label[1]]
                            else:
                                for p_t in dict_labels[next_label[0]]:
                                    if(lemma2(dict_vsim[next_label[1]], dict_vsim[p_t], nl_l, l(p_t, G))):
                                        dict_labels[next_label[0]].remove(p_t)
                                        if((next_label[0], p_t) in Q):
                                            Q.remove((next_label[0], p_t))
                                            Q_blacklist.append(p_t) 
                                dict_labels[next_label[0]].append(next_label[1])

                            if(next_label[1] not in Q_blacklist):
                                Q.insert(0,next_label)
                            #print("Q add:", next_label, dict_vsim[next_label[1]])
                            #print(len(Q))

    print(time.time()-start)
    multipass = False
    if(len(P_lo)<k):
        return [-1]
        
    for i in range(len(P_lo)):
        for j in range(i+1,len(P_lo)):
            if(overlap_ratio(P_lo[i], P_lo[j], G) > threshold and overlap_ratio(P_lo[j], P_lo[i], G) > threshold):
                print("P_lo incorrect:", overlap_ratio(P_lo[i], P_lo[j], G), overlap_ratio(P_lo[j], P_lo[i], G), "Launching MultiPass")
                multipass = True
                break
        if(multipass):
            break
            
    if(multipass):
        MultiPass(G, tree, nodes, d_point, f_point)
        
    return P_lo
            
                                    











def prio(e, G):
    prio = 0
    for i in G.predecessors(e[0]):
        for j in G.neighbors(e[1]):
            if(i!=j):
                sp=nx.shortest_path(G, i, j, weight=get_weight)
                if(e[0] in sp and e[1] in sp and sp.index(e[0])==sp.index(e[1])-1):
                    prio+=1

    return prio

def choose_p_max(p_c, P_lo, l_P_lo, G):
    p_max = P_lo[0]
    sim_p_max = overlap_ratio(p_c, p_max, G, l_P_lo[0])
    for i in range(len(P_lo)):
        p=P_lo[i]
        if(p != p_max):
            sim_p = overlap_ratio(p_c, p, G, l_P_lo[i])
            if(sim_p > sim_p_max):
                sim_p_max = sim_p
                p_max = p
    return p_max, sim_p_max

def ESX(G_origin, tree, nodes, d_point, f_point, k=5, threshold=0.5):
    G=copy.deepcopy(G_origin)
    start = time.time()
    

    d_idx = tree.query([d_point], k=1, return_distance=False)[0]
    f_idx = tree.query([f_point], k=1, return_distance=False)[0]
    s = nodes.iloc[d_idx].index.values[0]
    t = nodes.iloc[f_idx].index.values[0]
    
    sp = nx.shortest_path(G, s, t, weight=get_weight)
    P_lo = [tuple(sp)] 
    l_P_lo = [l(sp, G)]
    

    max_heaps = []

    h = {}

    for i in range(len(P_lo[-1])-1):
        h[(P_lo[-1][i], P_lo[-1][i+1])] = prio((P_lo[-1][i], P_lo[-1][i+1]), G)
    h = {k: v for k, v in sorted(h.items(), key=lambda item: item[1])}
    max_heaps.append(list(h.keys()))

    non_removable_edges = []
    while(len(P_lo)<k and time.time() - start < 600):
        p_c = P_lo[-1]
        p_max, sim_p_max = choose_p_max(p_c, P_lo, l_P_lo, G_origin)
        h_max = max_heaps[P_lo.index(p_max)]
        if(len(h_max) == 0):
            for i in max_heaps:
                if(len(i)>0):
                    h_max=i
                    break
                
        if(len(h_max) == 0):
            break
        
        while(sim_p_max > threshold and len(h_max) > 0):

            e = h_max.pop()
            if(e in non_removable_edges):
                continue

            if(e[1] in G[e[0]]):
                #print("delete", e, G[e[0]][e[1]])
                e_data = copy.deepcopy(G[e[0]][e[1]])
                G.remove_edge(e[0], e[1])
            else:
                continue
            try:
                sp=nx.shortest_path(G, s, t, weight=get_weight)
            except nx.exception.NetworkXNoPath:
                G.add_edges_from([(e[0], e[1], e_data[0])])
                #print("re-added", e, G[e[0]][e[1]])
                non_removable_edges.append(e)
                continue

            p_c = sp
            p_max, sim_p_max = choose_p_max(p_c, P_lo, l_P_lo, G_origin)

        if(sim_p_max <= threshold):
            #print("add", p_c)
            P_lo.append(p_c)
            l_P_lo.append(l(P_lo[-1], G))
            for i in range(len(P_lo[-1])-1):
                h[(P_lo[-1][i], P_lo[-1][i+1])] = prio((P_lo[-1][i], P_lo[-1][i+1]), G)
            h = {k: v for k, v in sorted(h.items(), key=lambda item: item[1], reverse=True)}
            max_heaps.append(list(h.keys()))


    #print(time.time()-start)
    
    if(len(P_lo)<k):
        print("fail")
        return [-1]
        
    for i in range(len(P_lo)):
        for j in range(i+1,len(P_lo)):
            if(overlap_ratio(P_lo[i], P_lo[j], G_origin) > threshold and overlap_ratio(P_lo[j], P_lo[i], G_origin) > threshold):
                print("P_lo incorrect:", overlap_ratio(P_lo[i], P_lo[j], G_origin), overlap_ratio(P_lo[j], P_lo[i], G_origin))
                break


    return P_lo
                        




