import numpy as np
import pandas as pd

import torch

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from typing import List, Tuple

from russian_names import RussianNames

########################################################################
#                        Utils for Banking77                           #
########################################################################

def load_banking_data(path: str) -> Tuple[pd.DataFrame, List[int], torch.Tensor]:
    '''Load Banking data and embeddings.
    
        :param path - path to the folder with Banking77 data; the folder should contain "embeds/" subfolder
        :return a tuple of:
            * pd.DataFrame with texts and markup
            * list with true cluster labels
            * T5 embeddings for texts
    '''
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

def sample_banking_clusters(
    dataframe: pd.DataFrame, raw_embeds: torch.Tensor, cluster_num_list: List[int]
) -> Tuple[torch.Tensor, List[int], pd.DataFrame, List[int]]:
    '''Dowsample the dataset and extract several cluters.
    
        :param dataframe - loaded pd.DataFrame with texts and markup
        :param raw_embeds - loaded T5 embeddings for texts
        :param cluster_num_list - cluster numbers to include into the subsample
        :return a tuple of:
            * T5 embeddings for subsample
            * indexes of rows taken into subsample
            * pd.DataFrame with texts and markup taken into subsample
            * list with true cluster labels for texts taken into subsample
    '''
    target_idxs = dataframe[dataframe["cluster"].isin(cluster_num_list)].index.to_list()
    target_size = len(target_idxs)
    target_data = dataframe.loc[target_idxs]
    target_embeds = raw_embeds[target_idxs]
    target_clusters = target_data["cluster"].to_list()
    
    return target_embeds, target_idxs, target_data, target_clusters

########################################################################
#                        Utils for Sber data                           #
########################################################################

# for name removal
RN = RussianNames(count=200, patronymic=False, surname=False)
NAMES_LIST = []
for name in RN:
    NAMES_LIST.append(name)

def clean_russian_dialogue(
    text: str,
    remove_stopwords: bool = False,
    remove_names: bool = False
) -> str:
    '''Clean Russian text.
    
        :param text - string to be cleaned
        :param remove_stopwords - if True, remove stopwords
        :param remove_names - if True, remove Russian names
        :return clean string
    '''
    tokenizer = RegexpTokenizer(r'[a-zа-яёЁА-ЯA-Z]\w+\'?\w*')
    noise_list = ["канал", "тикет", "закрыт", "спасибо", "благодарю"]
    if remove_stopwords:
        noise_list += stopwords.words('russian')
    if remove_names:
        noise_list += NAMES_LIST
    tok_text = tokenizer.tokenize(text)
    tok_text = [tok.lower() for tok in tok_text if tok.lower() not in noise_list]
    clean_text = " ".join(tok_text)
    return clean_text

def read_and_clean_sber_data(path: str) -> Tuple[pd.DataFrame, List[int], torch.Tensor]:
    '''Load Demo data and embeddings.
    
        :param path - path to the folder with Demo data; the folder should contain "embeds/" subfolder
        :return a tuple of:
            * pd.DataFrame with texts and markup
            * list with true cluster labels
            * T5 embeddings for texts
    '''
    data = pd.read_csv(path + "/demo.csv")
    data["cluster"] = data["category"].astype("category")
    data["cluster"] = data["cluster"].cat.codes

    clusters = data["cluster"].to_list()
    
    data["original_text"] = data["text"]
    
    data['text_with_stop'] = data['original_text'].apply(lambda row: clean_russian_dialogue(row))
    data['text'] = data['original_text'].apply(
        lambda row: clean_russian_dialogue(row, remove_stopwords=True, remove_names=True)
    )
    embeds = torch.load(path + "/sber_embeds/demo_embedings_all.pt")
    
    return data, clusters, embeds

def sample_sber_clusters(
    cluster_num: int,
    data_frame: pd.DataFrame,
    embeds: torch.Tensor,
    ignore_other: bool = True,
    verbose: bool = True
) -> Tuple[torch.Tensor, List[int], pd.DataFrame, List[int]]:
    '''Dowsample the dataset and extract several most popular clusters cluters.
    
        :param cluster_num - number of the most popular clusters to take into subsample
        :param data_frame - loaded pd.DataFrame with texts and markup
        :param embeds - loaded embeddings for texts
        :param ignore_other - if True, ignore "Другое" cluster (the most popular in the dataset)
        :param verbose - if True, print the results
        :return a tuple of:
            * T5 embeddings for subsample
            * indexes of rows taken into subsample
            * pd.DataFrame with texts and markup taken into subsample
            * list with true cluster labels for texts taken into subsample
    '''
    clusters_all = data_frame["cluster"].to_list()
    _, counts = np.unique(clusters_all, return_counts=True)
    
    top_k_clusters = np.argpartition(counts, -(cluster_num + 1))[-(cluster_num + 1):]
    
    if ignore_other:
        top_k_clusters = top_k_clusters[top_k_clusters != 9] # ignore cluster 9 - "Другое"
        
    top_k_cats = list(data_frame[data_frame["cluster"].isin(top_k_clusters)]["category"].unique())
    
    if verbose:
        print(f"Top {cluster_num} popular cluster nums: {top_k_clusters}")
        print(f"Corresponding categories: {top_k_cats}")
    
    target_idxs = data_frame[data_frame["cluster"].isin(top_k_clusters)].index.to_list()
    target_data = data_frame.loc[target_idxs]
    target_embeds = embeds[target_idxs]
    target_clusters = target_data["cluster"].to_list()
    
    return target_embeds, target_idxs, target_data, target_clusters