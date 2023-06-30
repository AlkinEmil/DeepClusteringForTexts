import numpy as np

import torch
import torch.nn as nn

import pandas as pd

from algos.deep_clustering import DeepClustering
from algos.classic_clustering import ClassicClustering
from utils.training_and_visualisation import train
from utils import topic_extraction

class TextClustering(nn.Module):
    def __init__(self, n_clusters, inp_dim, feat_dim, train_dataset,
                 loss_weights=None,
                 cluster_centers_init=None,
                 encoder=None,
                 decoder=None,
                 kind="deep clustering",
                 dim_reduction_type=None, 
                 clustering_type=None,
                 random_state=None):
        '''
            n_clusters: positive int - number of clusters
            inp_dim: positive int - dimension of the original space
            feat_dim: positive int - dimension of the feature space in which we do clustering
            alpha: float - parameter of the clustering loss
            hid_dim: positive int - dimension of the hidden space
            cluster_centers_init: torch.Tensor of shape (n_clusters, hid_dim)
        '''
        super().__init__()
        
        
        if not isinstance(n_clusters, int):
            raise TypeError("'n_clusters' must be integer")
        if not isinstance(inp_dim, int):
            raise TypeError("'inp_dim' must be integer")
        if not isinstance(feat_dim, int):
            raise TypeError("'feat_dim' must be integer")
        if not isinstance(kind, str):
            raise TypeError("'kind' must be string")
        
        if n_clusters <= 0:
            raise ValueError("'n_clusters' must be positive")
        if inp_dim <= 0:
            raise ValueError("'inp_dim' must be positive")
        if feat_dim <= 0:
            raise ValueError("'feat_dim' must be positive")
        
        self.n_clusters = n_clusters
        self.inp_dim = inp_dim
        self.feat_dim = feat_dim
        #self.encoder = None
        #self.decoder = None
        self.kind = kind
            
        if self.kind == "deep clustering":
            if encoder is not None and decoder is None:
                    raise ValueError("decoder must be not None")
            if decoder is not None and encoder is None:
                    raise ValueError("encoder must be not None")
                #self.kind = "deep clustering"
                #self.encoder = encoder
                #self.decoder = decoder
            self.model = DeepClustering(n_clusters,
                                        inp_dim,
                                        feat_dim,
                                        train_dataset,
                                        alpha=4, 
                                        loss_weights=[0.5, 0.5],
                                        encoder=encoder,
                                        decoder=decoder
                                       )
        elif self.kind == "classic clustering":
            self.model = ClassicClustering(n_clusters, 
                                           inp_dim, 
                                           feat_dim,
                                           dim_reduction_type=dim_reduction_type, 
                                           clustering_type=clustering_type, 
                                           random_state=random_state)
        
    def fit(self, base_embeds, device='cuda'):
        if self.kind == "deep clustering":
            self.model.to(device)
            N_ITERS = 20
            LR = 3e-3
            optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
            print("Phase 1: train embeddings")
            losses1 = train(self.model, base_embeds, optimizer, N_ITERS, device)
        
            self.model.train_clusters(base_embeds.to(device), [0.33, 0.33, 0.34])
        
            N_ITERS = 8
            LR = 1e-4
            # Change mode of the model to `train_clusters` and change weights of losses:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
            print("Phase 2: train clusters")
            losses2 = train(self.model, base_embeds, optimizer, N_ITERS, device)
            return losses1, losses2
        elif self.kind == "classic clustering":
            self.model.fit(base_embeds)
            return None, None
        
    def get_centers(self):
        if self.kind == "deep clustering":
            return self.model.centers.cpu().detach().numpy()
        elif self.kind == "classic clustering":
            return self.model.clustering.cluster_centers_
    
    def transform_and_cluster(self, inputs, batch_size=None):
        if self.kind == "deep clustering":
            inputs = inputs.to(self.model.centers.device)
            return self.model.transform_and_cluster(inputs, batch_size=batch_size)
        elif self.kind == "classic clustering":
            return self.model.transform_and_cluster(inputs)
    
    def get_topics(self, texts, inputs, language="english"):
        if self.kind == "deep clustering":
            inputs = inputs.to(self.model.centers.device)
            if self.model.mode == "train_embeds":
                raise PermissionError("model must be in `train_clusters` mode")
        _, pred_clusters = self.transform_and_cluster(inputs)
        if language in ["english", "russian"]:
            data_frame = pd.DataFrame({"text": texts, "pred_cluster": pred_clusters})
            topics = topic_extraction.get_topics(data_frame, language=language)
        else:
            raise ValueError("Unknown language `{}`".format(lang))
        return topics
