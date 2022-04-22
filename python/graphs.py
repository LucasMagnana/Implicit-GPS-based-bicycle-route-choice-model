import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'size'   : 13}

matplotlib.rc('font', **font)

def mean_distance_barplot(tab_results_base, tab_results_improvement, project_folder, dataset):
    plt.clf()
    ind = np.arange(len(tab_results_base)) # the x locations for the groups
    width=0.3
    plt.bar(ind, tab_results_base, width, color='g', label='Mean distance')
    #plt.bar(ind, tab_results_improvement, width,bottom=tab_results_base, color='g', label="Modified path")
    x = np.arange(len(tab_results_base))
    plt.xticks(x, ["Computed", "Mapbox", "Global", "Experience-based"])
    plt.legend(loc='upper left')
    #plt.ylabel('Similarity between the path generated and the observation (%)')
    plt.yticks(np.arange(0, 1.1, step=0.1))

    plt.savefig("files/"+project_folder+"/images/barplot_mean_distance_"+dataset+".png")


def distance_boxplot_NN(tab_results, project_folder, dataset):
    plt.clf()
    fig1, ax1 = plt.subplots()
    ax1.set_title('')
    plt.xticks(rotation=45)
    plt.tight_layout(pad=5)
    ax1.boxplot(tab_results, labels=["Global", "LSTM", "Oracle"])
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.savefig("files/"+project_folder+"/images/boxplot_distance_NN_"+dataset+".png")
        
        
def distance_boxplot(tab_results, project_folder, dataset):
    plt.clf()
    fig1, ax1 = plt.subplots()
    ax1.set_title('')
    plt.xticks(rotation=45)
    plt.tight_layout(pad=5)
    ax1.boxplot(tab_results, labels=["Computed", "Mapbox", "Global", "Experience-based"])
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.savefig("files/"+project_folder+"/images/boxplot_distance_"+dataset+".png")