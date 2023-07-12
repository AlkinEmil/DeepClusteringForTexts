import numpy as np

import torch
import torch.nn as nn

import time

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

from algos.deep_clustering import DeepClustering
from algos.classic_clustering import ClassicClustering
from utils.training_and_visualisation import train
from utils import topic_extraction

import umap.umap_ as umap
import matplotlib.pyplot as plt

class TextClustering(nn.Module):
    def __init__(self, n_clusters, inp_dim, feat_dim, train_dataset, data_frame,
                 loss_weights=None,
                 cluster_centers_init=None,
                 encoder=None,
                 decoder=None,
                 kind="deep clustering",
                 dim_reduction_type=None, 
                 clustering_type=None,
                 deep_model_type="DEC",
                 deep_params=None,
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
        self.kind = kind
        self.data_frame = data_frame
        self.train_dataset = train_dataset
        self.times = {}
        self.deep_model_type = deep_model_type
        self.deep_params = deep_params
            
        if self.kind == "deep clustering":
            if encoder is not None and decoder is None:
                    raise ValueError("decoder must be not None")
            if decoder is not None and encoder is None:
                    raise ValueError("encoder must be not None")
            self.model = DeepClustering(n_clusters,
                                        inp_dim,
                                        feat_dim,
                                        train_dataset,
                                        alpha=4, 
                                        loss_weights=[0.5, 0.5],
                                        deep_model_type=deep_model_type,
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
        else:
            raise ValueError("Unknown kind `{}`".format(kind))
        
    def fit(self, base_embeds, device='cuda'):
        if self.kind == "deep clustering":
            N_ITERS_1 = 20
            LR_1 = 3e-3
            N_ITERS_2 = 8
            LR_2 = 1e-4
            LOSS_WEIGHTS = {
                "recon": 1.,
                "geom": 1.
            }
            if self.deep_model_type == "DEC" or self.deep_model_type == "DEC+DCN":
                LOSS_WEIGHTS["DEC"] = 1.
            if self.deep_model_type == "DCN" or self.deep_model_type == "DEC+DCN":
                LOSS_WEIGHTS["inv_pw_dist"] = 1.
                LOSS_WEIGHTS["modified_DCN"] = 1.
            #if self.deep_model_type == "DEC":
            #    LOSS_WEIGHTS = [1., 1., 1.]
            #elif self.deep_model_type == "DCN":
            #    LOSS_WEIGHTS = [1., 1., 1., 1.]
            #elif self.deep_model_type == "DEC+DCN":
            #    LOSS_WEIGHTS = [1., 1., 1., 1., 1.]
            #else:
            #    raise ValueError("Unknown deep_model_type `{}`".format(self.deep_model_type))
            
            if self.deep_params is not None:
                N_ITERS_1 = self.deep_params.get("N_ITERS_1", N_ITERS_1)
                LR_1 = self.deep_params.get("LR_1", LR_1)
                N_ITERS_2 = self.deep_params.get("N_ITERS_2", N_ITERS_2)
                LR_2 = self.deep_params.get("LR_2", LR_2)
                LOSS_WEIGHTS = self.deep_params.get("loss_weights", LOSS_WEIGHTS)
            
            self.model.to(device)
            
            optimizer = torch.optim.Adam(self.model.parameters(), lr=LR_1)
            print("Phase 1: train embeddings")
            
            start_time = time.time()
            losses1 = train(self.model, base_embeds, optimizer, N_ITERS_1, device)
            end_time = time.time()
            self.times["dim_red"] = end_time - start_time
            
            
            
            self.model.train_clusters(base_embeds.to(device), LOSS_WEIGHTS)
        
            
            # Change mode of the model to `train_clusters` and change weights of losses:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=LR_2)
            print("Phase 2: train clusters")
            
            start_time = time.time()
            losses2 = train(self.model, base_embeds, optimizer, N_ITERS_2, device)
            end_time = time.time()
            self.times["clust"] = end_time - start_time
            
            self.times["total"] = self.times["dim_red"] + self.times["clust"]
            
            return losses1, losses2
        
        elif self.kind == "classic clustering":
            start_time = time.time()
            self.model.fit(base_embeds)
            end_time = time.time()
            self.times["total"] = end_time - start_time
            return None, None
        
    def get_centers(self):
        if self.kind == "deep clustering":
            return self.model.centers.cpu().detach().numpy()
        elif self.kind == "classic clustering":
            return self.model.clustering.cluster_centers_
        
    def visualize_2d(self, inputs, true_cluster_labels, random_state=None):
        if self.kind == "deep clustering":
            inputs = inputs.to(self.model.centers.device)
        dec_features, pred_clusters = self.transform_and_cluster(inputs)
        dim_reduction = umap.UMAP(random_state=random_state, n_components=2)
        umap_features = dim_reduction.fit_transform(dec_features)
        topics = self.get_topics(inputs)
        new_centers = dim_reduction.transform(self.get_centers())
        _, pred_clusters = self.transform_and_cluster(inputs)
        plt.figure(figsize=(9, 9))
        plt.scatter(*umap_features.T, c=true_cluster_labels, s=1.0)
        for i, (x, y) in enumerate(new_centers):
            if topics.get(i) is not None:
                plt.text(x, y, "\n".join(topics[i]), 
                         horizontalalignment='center', 
                         verticalalignment='center', 
                         fontsize=12)#, backgroundcolor='white')
        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.show()
    
    def transform_and_cluster(self, inputs, batch_size=None):
        if self.kind == "deep clustering":
            inputs = inputs.to(self.model.centers.device)
            embeds, pred_clusters = self.model.transform_and_cluster(inputs, batch_size=batch_size)
        elif self.kind == "classic clustering":
            embeds, pred_clusters = self.model.transform_and_cluster(inputs)
        
        self.data_frame["pred_cluster"] = pred_clusters
            
        return embeds, pred_clusters
    
    def get_topics(self, inputs, language="english"):
        if self.kind == "deep clustering":
            inputs = inputs.to(self.model.centers.device)
            if self.model.mode == "train_embeds":
                raise PermissionError("model must be in `train_clusters` mode")
        _, pred_clusters = self.transform_and_cluster(inputs)
        if language in ["english", "russian"]:
            topics_cluster_dict = topic_extraction.get_topics(self.data_frame, language=language)
        else:
            raise ValueError("Unknown language `{}`".format(lang))
        return topics_cluster_dict
    
    def evaluate(self, use_true_clusters=False, language="english", verbose=True):
        assert "pred_cluster" in self.data_frame.columns, "Predicted cluster labels are not in the dataframe"
        pred_clusters = self.data_frame["pred_cluster"].to_list()
        
        metrics = dict()
        
        if len(set(pred_clusters)) > 1:
            ss = silhouette_score(self.train_dataset, pred_clusters, metric='euclidean')
        else:
            ss = 0
        
        metrics["silhouette_score"] = ss
        
        if use_true_clusters:
            assert "cluster" in self.data_frame.columns, "True cluster labels are not in the dataframe"
            true_clusters = self.data_frame["cluster"].to_list()
            
            ars = adjusted_rand_score(true_clusters, pred_clusters)
            ams = adjusted_mutual_info_score(true_clusters, pred_clusters)
            
            metrics["adjusted_rand_score"] = ars
            metrics["adjusted_mutual_info_score"] = ams
            
        topics_cluster_dict = self.get_topics(self.train_dataset, language=language)
        topics_coh_dict = topic_extraction.compute_coherence_for_clusters(topics_cluster_dict, self.data_frame)
        
        avg_coh = np.mean(list(topics_coh_dict.values()))
        metrics["average_topic_coherence"] = avg_coh
        
        if verbose:
            for k, v in metrics.items():
                print(f"{k}: {np.round(v, 4)}")
        
        return metrics          
        
