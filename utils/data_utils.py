import numpy as np
import pandas as pd

import torch

def load_banking_data(path):
    data_train = pd.read_csv(path + "train.csv", sep=",")
    texts_train = data_train["text"].to_list()

    data_train["cluster"] = data_train["category"].astype("category")
    data_train["cluster"] = data_train["cluster"].cat.codes

    clusters_train = data_train["cluster"].to_list()

    data_test = pd.read_csv(path + "test.csv", sep=",")
    texts_test = data_test["text"].to_list()

    data_test["cluster"] = data_test["category"].astype("category")
    data_test["cluster"] = data_test["cluster"].cat.codes

    clusters_test = data_test["cluster"].to_list()

    data_all = pd.concat((data_train, data_test)).reset_index() # fix indices
    clusters_all = clusters_train + clusters_test

    # load embeddings
    embeds_t5_train = torch.load(path + "embeds/t5_train.pt")
    embeds_t5_test = torch.load(path + "embeds/t5_test.pt")

    embeds_t5_all = torch.vstack((embeds_t5_train, embeds_t5_test))
    
    return data_all, clusters_all, embeds_t5_all

def sample_banking_clusters(dataframe, raw_embeds, cluster_num_list, noise_cluster_num_list, noise_frac=0.):
    target_idxs = dataframe[dataframe["cluster"].isin(cluster_num_list)].index.to_list()
    target_size = len(target_idxs)
    target_data = dataframe.loc[target_idxs]
    target_embeds = raw_embeds[target_idxs]
    target_clusters = target_data["cluster"].to_list()
                
    if noise_frac != 0. and noise_cluster_num_list is not None:
        non_target_idxs = dataframe[dataframe["cluster"].isin(noise_cluster_num_list)].index.to_list()
        noise_data_all = dataframe.loc[non_target_idxs]

        noise_num = int(noise_frac * target_size)
        noise_idxs = list(np.random.choice(non_target_idxs, noise_num, replace=False))
        noise_data = dataframe.loc[noise_idxs]
        noise_embeds = raw_embeds[noise_idxs]
        noise_clusters = noise_data["cluster"].to_list()
        subset_data = pd.concat((target_data, noise_data))
        subset_clusters = target_clusters + noise_clusters
        subset_embeds = torch.vstack((target_embeds, noise_embeds))
        subset_idxs = target_idxs + noise_idxs
     
        return subset_embeds, subset_idxs, subset_data, subset_clusters, target_idxs
    
    return target_embeds, target_idxs, target_data, target_clusters, target_idxs

