from sklearn.metrics import silhouette_score

def cluster(X, clustering_method):
    clustering = clustering_method.fit(X)
    return clustering.labels_


def tab_clusters_to_dict(clusters):
    dict_cluster = {}
    for i in range(len(clusters)):
        if(clusters[i] in dict_cluster):
            dict_cluster[clusters[i]].append(i)
        else:
            dict_cluster[clusters[i]] = [i]
    return dict_cluster



def cluster_properties(dict_cl, X, cl, metric=None):
    if(metric != None):
        silhouette = silhouette_score(X, cl, metric=metric)
    else:
        silhouette = silhouette_score(X, cl)
    print("silhouette score :", silhouette)
    print()

    mean = 0
    mini_clusters = []
    big_clusters = []
    for i in dict_cl:
        if(i != -1):
            if(len(dict_cl[i]) >= len(X)*2/100):
                big_clusters.append(i)
            elif(len(dict_cl[i]) <= len(X)*0.25/100):
                mini_clusters.append(i)
            mean+=len(dict_cl[i])

    print("mean size :", mean/(len(dict_cl)-1))
    print()
    print(len(big_clusters), "big clusters:", big_clusters)
    print(len(mini_clusters), "mini clusters :", mini_clusters)



