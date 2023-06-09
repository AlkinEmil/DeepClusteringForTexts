import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import time

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

from algos.deep_clustering import DeepClustering
from algos.contrastive_clustering import CoHiClustModel
from algos.classic_clustering import ClassicClustering

from utils.training_and_visualisation import train
from utils.cohiclust_utils import train_cohiclust, test_cohiclust, RepeatPairDataset
from utils import topic_extraction

class TextClustering(nn.Module):
    def __init__(self, n_clusters, inp_dim, train_dataset, data_frame,
                 cohiclust_cfg=None,
                 feat_dim=None,
                 loss_weights=None,
                 cluster_centers_init=None,
                 encoder=None,
                 decoder=None,
                 kind="deep clustering",
                 dim_reduction_type=None, 
                 clustering_type=None,
                 random_state=None,
                 min_samples=None,
                 min_cluster_size=None,
                 bandwidth=None              
                ):
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
        #if not isinstance(feat_dim, int):
        #    raise TypeError("'feat_dim' must be integer")
        if not isinstance(kind, str):
            raise TypeError("'kind' must be string")
        
        if n_clusters <= 0:
            raise ValueError("'n_clusters' must be positive")
        if inp_dim <= 0:
            raise ValueError("'inp_dim' must be positive")
        #if feat_dim <= 0:
        #    raise ValueError("'feat_dim' must be positive")
        
        self.n_clusters = n_clusters
        self.inp_dim = inp_dim
        self.feat_dim = feat_dim
        self.kind = kind
        self.data_frame = data_frame
        self.train_dataset = train_dataset
        self.times = {}
            
        if self.kind == "deep clustering":
            if encoder is not None and decoder is None:
                    raise ValueError("decoder must be not None")
            if decoder is not None and encoder is None:
                    raise ValueError("encoder must be not None")
            self.model = DeepClustering(
                n_clusters,
                inp_dim,
                feat_dim,
                train_dataset,
                alpha=4, 
                loss_weights=[0.5, 0.5],
                encoder=encoder,
                decoder=decoder
            )
        elif self.kind == "classic clustering":
            self.model = ClassicClustering(
                n_clusters, 
                inp_dim, 
                feat_dim,
                dim_reduction_type=dim_reduction_type, 
                clustering_type=clustering_type, 
                random_state=random_state,
                min_samples=min_samples,
                min_cluster_size=min_cluster_size,
                bandwidth=bandwidth
            )
            
        elif self.kind == "cohiclust":
            assert cohiclust_cfg is not None, "You must pass config for CoHiClust model."
            self.cohiclust_cfg = cohiclust_cfg
            
            self.model = CoHiClustModel(
                cohiclust_cfg,
                data_frame,
                train_dataset,
                n_clusters
            )
        
        else:
            raise ValueError(f"Wrong clustering type {self.kind}.")
        
    def fit(self, base_embeds, device='cuda'):
        if self.kind == "deep clustering":
            self.model.to(device)
            N_ITERS = 20
            LR = 3e-3
            optimizer = Adam(self.model.parameters(), lr=LR)
            print("Phase 1: train embeddings")
            
            start_time = time.time()
            losses1 = train(self.model, base_embeds, optimizer, N_ITERS, device)
            end_time = time.time()
            self.times["dim_red"] = end_time - start_time
            
            self.model.train_clusters(base_embeds.to(device), [0.33, 0.33, 0.34])
        
            N_ITERS = 8
            LR = 1e-4
            # Change mode of the model to `train_clusters` and change weights of losses:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
            print("Phase 2: train clusters")
            
            start_time = time.time()
            losses2 = train(self.model, base_embeds, optimizer, N_ITERS, device)
            end_time = time.time()
            self.times["clust"] = end_time - start_time
            
            self.times["total"] = self.times["dim_red"] + self.times["clust"]
            
            return losses1, losses2
        
        elif self.kind == "classic clustering":
            start_time = time.time()
            self.model.fit(base_embeds)
            end_time = time.time()
            self.times["dim_red"] = end_time - start_time
            return None, None
        
        elif self.kind == "cohiclust":
            optimizer = Adam(self.model.parameters(), lr=1e-3)
            start_time = time.time()
            results =  train_cohiclust(
                self.model,
                optimizer,
                device=device
            )
            end_time = time.time()
            
            self.times["clust"] = end_time - start_time
            self.times["dim_red"] = 0
            self.times["total"] = self.times["dim_red"] + self.times["clust"]
            
            return results
        
    def get_centers(self):
        if self.kind == "deep clustering":
            return self.model.centers.cpu().detach().numpy()
        elif self.kind == "classic clustering":
            return self.model.clustering.cluster_centers_
        elif self.kind == "cohiclust":
            raise NotImplementedError("Clusters centers for cohiclust if are not implemented.")
    
    def transform_and_cluster(self, inputs, batch_size=None):
        if self.kind == "deep clustering":
            inputs = inputs.to(self.model.centers.device)
            embeds, pred_clusters = self.model.transform_and_cluster(inputs, batch_size=batch_size)
        elif self.kind == "classic clustering":
            
            start_time = time.time()
            embeds, pred_clusters = self.model.transform_and_cluster(inputs)
            end_time = time.time()
            self.times["clust"] = end_time - start_time
            self.times["total"] = self.times["dim_red"] + self.times["clust"]
        
        elif self.kind == "cohiclust":
            if batch_size is None:
                batch_size = self.model.cfg.training.batch_size
            predict_dataset = RepeatPairDataset(inputs, self.data_frame["cluster"].to_list())
            predict_loader = DataLoader(predict_dataset, batch_size=batch_size)
            _, pred_clusters, labels = test_cohiclust(
                self.model, self.model.cfg.training.epochs, predict_loader, verbose=False
            )
            embeds = None

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
        